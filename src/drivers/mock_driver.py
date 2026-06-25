"""MockDriver: a dependency-free offline environment (no Pygame).

A minimal stand-in world rendered with numpy so the full eyes -> brain -> hands
loop can run in CI / tests with zero display and zero Pygame. It exposes the same
``BaseDriver`` contract plus ground-truth ``entity_boxes()`` so ``OraclePerception``
works against it. This is the canonical "drop in a new safe driver" example.
"""

from __future__ import annotations

import numpy as np

from drivers.base_driver import BaseDriver, Frame, Vector

# class ids/colors mirror eyes.vision_pipeline.CLASS_NAMES (0 Player, 1 Enemy, 3 Gem)
_COLORS = {0: (70, 150, 255), 1: (235, 80, 80), 3: (80, 220, 130)}
_BG = (18, 20, 28)


class MockDriver(BaseDriver):
    def __init__(self, *, size=(800, 600), seed=None) -> None:
        self._size = (int(size[0]), int(size[1]))
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._player_r, self._enemy_r, self._gem_r = 16, 18, 8
        self.reset()

    @property
    def resolution(self) -> tuple[int, int]:
        return self._size

    def reset(self) -> None:
        w, h = self._size
        self._rng = np.random.default_rng(self._seed)
        self._player = np.array([w * 0.25, h * 0.5])
        self._enemy = np.array([w * 0.75, h * 0.5])
        self._enemy_dir = np.array([0.0, 1.0])
        self._move = np.zeros(2)
        self._hp, self._ammo, self._gems = 100, 3, 0
        self._ammo_timer = 0.0
        self._gemlist = [self._rand_point() for _ in range(3)]

    def _rand_point(self) -> np.ndarray:
        w, h = self._size
        return np.array([self._rng.uniform(40, w - 40), self._rng.uniform(40, h - 40)])

    def hold_joystick(self, vector: Vector) -> None:
        v = np.asarray(vector, dtype=float)
        self._move = np.clip(v, -1.0, 1.0) if v.shape == (2,) else np.zeros(2)

    def tap_coordinate(self, x: float, y: float) -> None:
        if self._ammo > 0:
            self._ammo -= 1

    def read_hud_text(self) -> dict[str, str]:
        return {"hp": f"HP: {self._hp}", "ammo": f"AMMO: {self._ammo}",
                "gems": f"GEMS: {self._gems}"}

    def entity_boxes(self):
        boxes = [
            (0, float(self._player[0]), float(self._player[1]),
             2.0 * self._player_r, 2.0 * self._player_r),
            (1, float(self._enemy[0]), float(self._enemy[1]),
             2.0 * self._enemy_r, 2.0 * self._enemy_r),
        ]
        for g in self._gemlist:
            boxes.append((3, float(g[0]), float(g[1]),
                          2.0 * self._gem_r, 2.0 * self._gem_r))
        return boxes

    def step(self, dt: float) -> None:
        w, h = self._size
        self._player = self._player + self._move * 240.0 * dt
        self._player = np.clip(self._player, [self._player_r, self._player_r],
                               [w - self._player_r, h - self._player_r])
        # enemy bounces vertically
        self._enemy = self._enemy + self._enemy_dir * 140.0 * dt
        if self._enemy[1] < 60 or self._enemy[1] > h - 60:
            self._enemy_dir = -self._enemy_dir
        self._enemy = np.clip(self._enemy, [self._enemy_r, self._enemy_r],
                              [w - self._enemy_r, h - self._enemy_r])
        # collect gems
        keep = []
        for g in self._gemlist:
            if np.linalg.norm(g - self._player) < self._player_r + self._gem_r:
                self._gems += 1
            else:
                keep.append(g)
        self._gemlist = keep or [self._rand_point() for _ in range(3)]
        # ammo regen
        self._ammo_timer += dt
        if self._ammo_timer >= 1.0:
            self._ammo_timer = 0.0
            self._ammo = min(3, self._ammo + 1)

    def get_screen_frame(self) -> Frame:
        w, h = self._size
        frame = np.empty((h, w, 3), dtype=np.uint8)
        frame[:] = _BG
        self._blit(frame, self._player, self._player_r, _COLORS[0])
        self._blit(frame, self._enemy, self._enemy_r, _COLORS[1])
        for g in self._gemlist:
            self._blit(frame, g, self._gem_r, _COLORS[3])
        return frame

    @staticmethod
    def _blit(frame, pos, r, color) -> None:
        h, w = frame.shape[:2]
        x, y = int(pos[0]), int(pos[1])
        x0, x1 = max(0, x - r), min(w, x + r)
        y0, y1 = max(0, y - r), min(h, y + r)
        frame[y0:y1, x0:x1] = color
