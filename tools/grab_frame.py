"""Render the arena off-screen and save a PNG preview.

Doubles as the seed of the dataset tooling: the same headless render path will
later dump labeled frames for training YOLO on the arena sprites.

Usage:  python tools/grab_frame.py [out.png]
"""

import os
import pathlib
import sys

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pygame

from arena.pygame_driver import PygameDriver
from arena.simulation import Arena, WanderAgent


def main() -> None:
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else \
        pathlib.Path(__file__).resolve().parents[1] / "arena_preview.png"

    arena = Arena(seed=3, headless=True)
    driver = PygameDriver(arena)
    arena.set_driver(driver)
    agent = WanderAgent(resolution=arena.size, seed=3, fps=arena.fps)

    # Warm up so the player has roamed a bit.
    for _ in range(140):
        agent(driver)
        arena.step(1 / 60)

    # Guarantee a lively frame: a spread of collectibles, some score, a shot.
    arena.items.clear()
    for _ in range(6):
        arena._spawn_item()
    arena.player.gems = 5
    arena.player.ammo = 3
    driver.tap_coordinate(*arena.enemy.pos)
    arena.step(1 / 60)

    pygame.image.save(arena.screen, str(out))
    print(f"saved {out}")
    arena.close()


if __name__ == "__main__":
    main()
