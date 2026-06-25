"""Draw the trained model's LIVE predictions over a rendered arena frame.

Runs the arena to a lively state, captures a frame, runs it through the real
VisionEngine (YOLO + OCR), and overlays the predicted boxes + confidences and
the OCR HUD readout. This is the proof-of-concept shot: our custom model's
bounding boxes on its own simulated world.

Run from the 3.12 vision env once `models/best.pt` is trained:
    .venv312\\Scripts\\python tools/predict_overlay.py [out.png]
"""

import os
import pathlib
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import pygame

from arena.pygame_driver import PygameDriver
from arena.simulation import Arena, WanderAgent
from eyes.vision_pipeline import VisionEngine

COLORS = {0: (70, 150, 255), 1: (235, 80, 80), 2: (255, 220, 120),
          3: (80, 220, 130), 4: (180, 110, 240)}


def main() -> None:
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else _ROOT / "vision_detections.png"

    arena = Arena(seed=5, headless=True)
    driver = PygameDriver(arena)
    arena.set_driver(driver)
    agent = WanderAgent(resolution=arena.size, seed=5, fps=arena.fps)

    # Liven the scene: roam, then guarantee a spread of items + a projectile.
    for _ in range(150):
        agent(driver)
        arena.step(1 / 60)
    arena.items.clear()
    for _ in range(6):
        arena._spawn_item()
    arena.player.ammo = 3
    driver.tap_coordinate(*arena.enemy.pos)
    arena.step(1 / 60)

    frame = driver.get_screen_frame()
    engine = VisionEngine(model_path=str(_ROOT / "models" / "best.pt"))
    state = engine.analyze(frame)
    engine.close()

    # Overlay predictions on the live render.
    surf = arena.screen
    font = pygame.font.SysFont("consolas", 13)
    for d in state.detections:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        color = COLORS.get(d.class_id, (255, 255, 255))
        pygame.draw.rect(surf, color, pygame.Rect(x1, y1, x2 - x1, y2 - y1), 2)
        surf.blit(font.render(f"{d.class_name} {d.confidence:.2f}", True, color),
                  (x1, max(0, y1 - 13)))

    banner = (f"YOLO predictions: {len(state.detections)}   "
              f"OCR  HP:{state.hp}  AMMO:{state.ammo}")
    surf.blit(font.render(banner, True, (255, 255, 255)), (10, arena.size[1] - 22))

    pygame.image.save(surf, str(out))
    arena.close()
    print(f"saved {out}")
    print(f"detections={len(state.detections)} hp={state.hp} ammo={state.ammo}")
    by_class = {}
    for d in state.detections:
        by_class[d.class_name] = by_class.get(d.class_name, 0) + 1
    print("by class:", by_class)


if __name__ == "__main__":
    main()
