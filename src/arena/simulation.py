"""The Brawler Arena — a small self-contained top-down 2D environment.

The arena owns the Pygame window, the world entities (player, enemy bot,
collectible gems / power cubes), and the HUD. It is driver-agnostic: any
:class:`BaseDriver` can be injected, and the agent interacts only through that
HAL. The enemy bot patrols along cubic-Bézier paths from ``hands`` to dogfood
the kinematics layer.

Run modes::

    python src/arena/simulation.py            # windowed, demo WanderAgent drives the player
    python src/arena/simulation.py --play     # windowed, you drive with WASD / arrows, click to fire
    python src/arena/simulation.py --headless # off-screen smoke test (no window)
"""

from __future__ import annotations

import os
import pathlib
import sys

# Allow running as a loose script: `python src/arena/simulation.py`.
_SRC = pathlib.Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pygame

from arena.driver_interface import BaseDriver
from hands.bezier_motion import joystick_stream, plan_motion

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
WIDTH, HEIGHT = 800, 600
FPS = 60

C_BG = (18, 20, 28)
C_GRID = (30, 34, 46)
C_PLAYER = (70, 150, 255)
C_ENEMY = (235, 80, 80)
C_GEM = (80, 220, 130)
C_CUBE = (180, 110, 240)
C_PROJ = (255, 220, 120)
C_HUD = (235, 235, 245)
C_HP = (120, 230, 140)
C_AMMO = (250, 210, 120)

# Sprite half-sizes (radii) shared by rendering and bounding-box extraction.
PROJ_RADIUS = 4
GEM_RADIUS = 8
CUBE_RADIUS = 9


# --------------------------------------------------------------------------- #
# Entities
# --------------------------------------------------------------------------- #
class Player:
    def __init__(self, pos) -> None:
        self.pos = np.asarray(pos, dtype=float)
        self.radius = 16
        self.speed = 260.0  # px / s at full stick
        self.hp = 100.0
        self.max_hp = 100.0
        self.ammo = 3
        self.max_ammo = 3
        self.gems = 0
        self.move_input = np.zeros(2)
        self.facing = np.array([1.0, 0.0])


class Enemy:
    """A bot that patrols between random waypoints along Bézier paths."""

    def __init__(self, pos, rng, fps, bounds) -> None:
        self.pos = np.asarray(pos, dtype=float)
        self.radius = 18
        self.hp = 100
        self._rng = rng
        self._fps = fps
        self._bounds = bounds
        self._path = None
        self._i = 0
        self._replan()

    def _replan(self) -> None:
        w, h = self._bounds
        target = np.array([self._rng.uniform(80, w - 80),
                           self._rng.uniform(80, h - 80)])
        plan = plan_motion(self.pos, target,
                           duration=float(self._rng.uniform(1.8, 3.0)),
                           fps=self._fps, easing="ease_in_out_sine",
                           curvature=0.30, jitter=0.04, rng=self._rng)
        self._path = plan.positions
        self._i = 0

    def update(self) -> None:
        if self._i >= len(self._path):
            self._replan()
        self.pos = self._path[self._i].copy()
        self._i += 1


# --------------------------------------------------------------------------- #
# Arena engine
# --------------------------------------------------------------------------- #
class Arena:
    """The environment + game loop. Accepts any :class:`BaseDriver` via DI."""

    def __init__(self, size=(WIDTH, HEIGHT), *, seed=None, headless=False,
                 fps=FPS) -> None:
        self.size = (int(size[0]), int(size[1]))
        self.headless = headless
        self.fps = fps
        self.rng = np.random.default_rng(seed)

        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ["SDL_AUDIODRIVER"] = "dummy"

        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("Brawler Arena Sandbox")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 22)
        self.big_font = pygame.font.SysFont("consolas", 48, bold=True)

        w, h = self.size
        self.player = Player((w * 0.25, h * 0.5))
        self.enemy = Enemy((w * 0.75, h * 0.5), self.rng, self.fps, self.size)
        self.items: list[dict] = []
        self.projectiles: list[dict] = []

        self.driver: BaseDriver | None = None
        self.game_over = False
        self._spawn_timer = 0.0
        self._ammo_timer = 0.0
        self._hud_strings: dict[str, str] = {}

    # -- dependency injection ------------------------------------------------ #
    def set_driver(self, driver: BaseDriver) -> None:
        """Inject the controller/sensor interface (any BaseDriver)."""
        self.driver = driver

    def reset(self) -> None:
        """Reset to a fresh episode (keeps the Pygame window / clock / fonts)."""
        w, h = self.size
        self.player = Player((w * 0.25, h * 0.5))
        self.enemy = Enemy((w * 0.75, h * 0.5), self.rng, self.fps, self.size)
        self.items = []
        self.projectiles = []
        self.game_over = False
        self._spawn_timer = 0.0
        self._ammo_timer = 0.0
        self._hud_strings = {}

    # -- surface exposed to the driver (HAL plumbing) ------------------------ #
    def set_move_input(self, vector) -> None:
        v = np.asarray(vector, dtype=float)
        if v.shape != (2,):
            raise ValueError("joystick vector must be (x, y)")
        self.player.move_input = np.clip(v, -1.0, 1.0)

    def register_tap(self, x: float, y: float) -> None:
        self._fire_at(x, y)

    def hud_text(self) -> dict[str, str]:
        return dict(self._hud_strings)

    def entity_boxes(self) -> list[tuple[int, float, float, float, float]]:
        """True pixel-space boxes of on-screen entities, from internal state.

        Returns ``(class_id, cx, cy, w, h)`` tuples. Class ids:
        0=Player, 1=Enemy, 2=Projectile, 3=Gem, 4=PowerCube. These come straight
        from the simulation's coordinates, so labels are pixel-perfect.
        """
        boxes: list[tuple[int, float, float, float, float]] = [
            (0, float(self.player.pos[0]), float(self.player.pos[1]),
             2.0 * self.player.radius, 2.0 * self.player.radius),
            (1, float(self.enemy.pos[0]), float(self.enemy.pos[1]),
             2.0 * self.enemy.radius, 2.0 * self.enemy.radius),
        ]
        for pr in self.projectiles:
            boxes.append((2, float(pr["pos"][0]), float(pr["pos"][1]),
                          2.0 * PROJ_RADIUS, 2.0 * PROJ_RADIUS))
        for it in self.items:
            cid = 3 if it["kind"] == "gem" else 4
            r = GEM_RADIUS if it["kind"] == "gem" else CUBE_RADIUS
            boxes.append((cid, float(it["pos"][0]), float(it["pos"][1]),
                          2.0 * r, 2.0 * r))
        return boxes

    # -- game logic ---------------------------------------------------------- #
    def _fire_at(self, x: float, y: float) -> None:
        if self.player.ammo <= 0:
            return
        direction = np.array([x, y]) - self.player.pos
        dist = float(np.linalg.norm(direction))
        if dist < 1e-6:
            return
        direction = direction / dist
        self.player.facing = direction
        self.player.ammo -= 1
        self.projectiles.append(
            {"pos": self.player.pos.copy(), "vel": direction * 620.0, "life": 1.2}
        )

    def _clamp(self, ent) -> None:
        w, h = self.size
        ent.pos[0] = float(np.clip(ent.pos[0], ent.radius, w - ent.radius))
        ent.pos[1] = float(np.clip(ent.pos[1], ent.radius, h - ent.radius))

    def _update_player(self, dt: float) -> None:
        p = self.player
        speed = float(np.linalg.norm(p.move_input))
        if speed > 1e-6:
            p.pos = p.pos + p.move_input * p.speed * dt
            p.facing = p.move_input / speed
        self._clamp(p)
        self._ammo_timer += dt
        if self._ammo_timer >= 1.2:
            self._ammo_timer = 0.0
            p.ammo = min(p.max_ammo, p.ammo + 1)

    def _update_enemy(self, dt: float) -> None:
        self.enemy.update()
        self._clamp(self.enemy)

    def _update_projectiles(self, dt: float) -> None:
        w, h = self.size
        alive = []
        for pr in self.projectiles:
            pr["pos"] = pr["pos"] + pr["vel"] * dt
            pr["life"] -= dt
            x, y = pr["pos"]
            if pr["life"] <= 0 or x < 0 or y < 0 or x > w or y > h:
                continue
            if np.linalg.norm(pr["pos"] - self.enemy.pos) < self.enemy.radius:
                self.enemy.hp -= 25
                if self.enemy.hp <= 0:
                    self._respawn_enemy()
                continue
            alive.append(pr)
        self.projectiles = alive

    def _respawn_enemy(self) -> None:
        w, h = self.size
        self.enemy.hp = 100
        self.enemy.pos = np.array([self.rng.uniform(80, w - 80),
                                   self.rng.uniform(80, h - 80)])
        self.enemy._replan()

    def _update_items(self, dt: float) -> None:
        self._spawn_timer += dt
        if self._spawn_timer >= 1.5 and len(self.items) < 8:
            self._spawn_timer = 0.0
            self._spawn_item()
        keep = []
        for it in self.items:
            if np.linalg.norm(it["pos"] - self.player.pos) < self.player.radius + 10:
                if it["kind"] == "gem":
                    self.player.gems += 1
                else:  # power cube: refill ammo + small heal
                    self.player.ammo = self.player.max_ammo
                    self.player.hp = min(self.player.max_hp, self.player.hp + 10)
                continue
            keep.append(it)
        self.items = keep

    def _spawn_item(self) -> None:
        w, h = self.size
        kind = "cube" if self.rng.random() < 0.25 else "gem"
        pos = np.array([self.rng.uniform(40, w - 40), self.rng.uniform(40, h - 40)])
        self.items.append({"pos": pos, "kind": kind})

    def _handle_collisions(self, dt: float) -> None:
        contact = self.player.radius + self.enemy.radius
        if np.linalg.norm(self.player.pos - self.enemy.pos) < contact:
            self.player.hp -= 40.0 * dt  # contact damage ~40 hp/s
        if self.player.hp <= 0:
            self.player.hp = 0.0
            self.game_over = True

    # -- rendering ----------------------------------------------------------- #
    @staticmethod
    def _ipos(v) -> tuple[int, int]:
        return (int(round(v[0])), int(round(v[1])))

    def _draw_grid(self) -> None:
        w, h = self.size
        for x in range(0, w, 40):
            pygame.draw.line(self.screen, C_GRID, (x, 0), (x, h))
        for y in range(0, h, 40):
            pygame.draw.line(self.screen, C_GRID, (0, y), (w, y))

    def _draw_item(self, it) -> None:
        x, y = self._ipos(it["pos"])
        if it["kind"] == "gem":
            r = GEM_RADIUS
            pygame.draw.polygon(self.screen, C_GEM,
                                [(x, y - r), (x + r, y), (x, y + r), (x - r, y)])
        else:
            r = CUBE_RADIUS
            pygame.draw.rect(self.screen, C_CUBE, pygame.Rect(x - r, y - r, 2 * r, 2 * r))

    def _draw_hp_bar(self, pos, radius, frac) -> None:
        x, y = self._ipos(pos)
        bw, bh = 40, 5
        bx, by = x - bw // 2, y - radius - 12
        pygame.draw.rect(self.screen, (60, 60, 60), pygame.Rect(bx, by, bw, bh))
        pygame.draw.rect(self.screen, C_ENEMY,
                         pygame.Rect(bx, by, int(bw * max(0.0, frac)), bh))

    def _blit_text(self, text, pos, color) -> None:
        self.screen.blit(self.font.render(text, True, color), pos)

    def _render_hud(self) -> None:
        hp = max(0, int(round(self.player.hp)))
        self._hud_strings = {
            "hp": f"HP: {hp}",
            "ammo": f"AMMO: {self.player.ammo}",
            "gems": f"GEMS: {self.player.gems}",
        }
        self._blit_text(self._hud_strings["hp"], (16, 12), C_HP)
        self._blit_text(self._hud_strings["ammo"], (16, 40), C_AMMO)
        self._blit_text(self._hud_strings["gems"], (16, 68), C_HUD)
        if self.game_over:
            surf = self.big_font.render("MATCH OVER", True, (255, 120, 120))
            rect = surf.get_rect(center=(self.size[0] // 2, self.size[1] // 2))
            self.screen.blit(surf, rect)

    def _render(self) -> None:
        self.screen.fill(C_BG)
        self._draw_grid()
        for it in self.items:
            self._draw_item(it)
        pygame.draw.circle(self.screen, C_ENEMY, self._ipos(self.enemy.pos),
                           self.enemy.radius)
        self._draw_hp_bar(self.enemy.pos, self.enemy.radius, self.enemy.hp / 100.0)
        pygame.draw.circle(self.screen, C_PLAYER, self._ipos(self.player.pos),
                           self.player.radius)
        tip = self.player.pos + self.player.facing * self.player.radius
        pygame.draw.line(self.screen, C_HUD, self._ipos(self.player.pos),
                         self._ipos(tip), 3)
        for pr in self.projectiles:
            pygame.draw.circle(self.screen, C_PROJ, self._ipos(pr["pos"]), PROJ_RADIUS)
        self._render_hud()
        pygame.display.flip()

    # -- loop ---------------------------------------------------------------- #
    def step(self, dt: float) -> None:
        """Advance the world by ``dt`` seconds and render one frame."""
        self._update_player(dt)
        self._update_enemy(dt)
        self._update_projectiles(dt)
        self._update_items(dt)
        self._handle_collisions(dt)
        self._render()

    def _keyboard_control(self) -> None:
        keys = pygame.key.get_pressed()
        vx = (keys[pygame.K_d] or keys[pygame.K_RIGHT]) - \
             (keys[pygame.K_a] or keys[pygame.K_LEFT])
        vy = (keys[pygame.K_s] or keys[pygame.K_DOWN]) - \
             (keys[pygame.K_w] or keys[pygame.K_UP])
        v = np.array([float(vx), float(vy)])
        n = float(np.linalg.norm(v))
        self.player.move_input = v / n if n > 0 else v

    def run(self, agent=None, max_frames=None, fps=None) -> int:
        """Run the game loop.

        If ``agent`` and a driver are set, the agent acts each frame through the
        HAL. Otherwise the keyboard controls the player. ``max_frames`` bounds
        the loop (used for headless smoke tests). Returns the frames rendered.
        """
        fps = fps or self.fps
        running = True
        frames = 0
        while running:
            if self.headless:
                dt = 1.0 / fps
                pygame.event.pump()
            else:
                dt = self.clock.tick(fps) / 1000.0
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self.register_tap(*event.pos)

            if agent is not None and self.driver is not None:
                agent(self.driver)
            elif not self.headless:
                self._keyboard_control()

            self.step(dt)
            frames += 1
            if self.game_over or (max_frames is not None and frames >= max_frames):
                running = False
        return frames

    def close(self) -> None:
        pygame.quit()


# --------------------------------------------------------------------------- #
# Demo agent (placeholder for the future behavior tree)
# --------------------------------------------------------------------------- #
class WanderAgent:
    """Roams the arena along Bézier paths, using only the HAL.

    Proves driver injection + kinematics end-to-end without the (not-yet-built)
    vision layer: it generates organic joystick streams via ``plan_motion`` and
    ``joystick_stream`` and pushes them through ``hold_joystick``.
    """

    def __init__(self, resolution=(WIDTH, HEIGHT), seed=0, fps=FPS) -> None:
        self.w, self.h = resolution
        self.fps = fps
        self.rng = np.random.default_rng(seed)
        self._end = np.array([self.rng.uniform(0, self.w),
                              self.rng.uniform(0, self.h)])
        self._stream = np.zeros((0, 2))
        self._i = 0

    def _replan(self) -> None:
        start = self._end
        end = np.array([self.rng.uniform(0.1 * self.w, 0.9 * self.w),
                        self.rng.uniform(0.1 * self.h, 0.9 * self.h)])
        plan = plan_motion(start, end, duration=float(self.rng.uniform(1.2, 2.2)),
                           fps=self.fps, easing="ease_in_out_sine",
                           curvature=0.25, jitter=0.05, rng=self.rng)
        self._stream = joystick_stream(plan)
        self._end = end
        self._i = 0

    def __call__(self, driver: BaseDriver) -> None:
        if self._i >= len(self._stream):
            self._replan()
        vec = self._stream[self._i] if len(self._stream) else np.zeros(2)
        self._i += 1
        driver.hold_joystick(vec)
        if self.rng.random() < 0.01:  # occasional pot-shot
            driver.tap_coordinate(self.rng.uniform(0, self.w),
                                  self.rng.uniform(0, self.h))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    import argparse

    from arena.pygame_driver import PygameDriver

    parser = argparse.ArgumentParser(description="Brawler Arena Sandbox")
    parser.add_argument("--play", action="store_true",
                        help="control the player yourself (WASD/arrows, click to fire)")
    parser.add_argument("--headless", action="store_true",
                        help="run off-screen for a quick smoke test")
    parser.add_argument("--frames", type=int, default=None,
                        help="frame cap (defaults to 180 in headless)")
    args = parser.parse_args()

    arena = Arena(seed=7, headless=args.headless)
    driver = PygameDriver(arena)
    arena.set_driver(driver)

    if args.play:
        agent = None  # keyboard
    else:
        agent = WanderAgent(resolution=arena.size, seed=7, fps=arena.fps)

    max_frames = args.frames
    if args.headless and max_frames is None:
        max_frames = 180

    frames = arena.run(agent=agent, max_frames=max_frames)

    if args.headless:
        frame = driver.get_screen_frame()
        print("headless smoke OK")
        print(f"  frames rendered : {frames}")
        print(f"  frame shape     : {frame.shape} dtype={frame.dtype}")
        print(f"  hud text        : {driver.read_hud_text()}")
        print(f"  player pos      : {arena.player.pos.round(1).tolist()}")
    arena.close()


if __name__ == "__main__":
    main()
