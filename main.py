"""Orchestrator: eyes -> brain -> hands over a config-selected driver.

The environment driver is chosen *by name* (``config.DRIVER_NAME`` / ``--driver``)
from ``drivers.registry``, so swapping environments is: add a driver, register it,
change the name. This file is pure glue — it touches only the ``BaseDriver``
contract, ``brain``, and a perception backend; it never imports Pygame.

Run:
    .venv\\Scripts\\python main.py                       # config defaults (pygame + oracle)
    .venv\\Scripts\\python main.py --driver mock --headless --max-ticks 60
    .venv312\\Scripts\\python main.py --perception vision  # trained YOLO + OCR (3.12 env)
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

import config
from brain.tactics import BrawlerBot
from drivers.registry import DRIVERS, get_driver

TICK_PERIOD = 0.1   # seconds (~10 Hz decision rate)
FRAME_DT = 1.0 / 60.0


def build_perception(kind: str, driver, model_path: str):
    if kind == "oracle":
        from eyes.oracle import OraclePerception
        return OraclePerception(driver)
    # Imported lazily so oracle/mock runs need no torch / ultralytics / easyocr.
    from eyes.vision_pipeline import VisionEngine
    return VisionEngine(model_path=model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Brawler Arena orchestrator (driver registry)")
    parser.add_argument("--driver", default=config.DRIVER_NAME, choices=sorted(DRIVERS))
    parser.add_argument("--perception", default=config.PERCEPTION, choices=["oracle", "vision"])
    parser.add_argument("--model", default=str(_ROOT / config.MODEL_PATH))
    parser.add_argument("--headless", action="store_true", default=config.HEADLESS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--max-ticks", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    args = parser.parse_args()

    driver = get_driver(args.driver, headless=args.headless, seed=args.seed)
    driver.reset()
    perception = build_perception(args.perception, driver, args.model)
    bot = BrawlerBot(driver)

    print(f"driver={args.driver}  perception={args.perception}  headless={args.headless}")
    print(bot.ascii_tree())

    start = time.monotonic()
    accum = 0.0
    ticks = 0
    last_action = "—"
    while not driver.should_close:
        t0 = time.monotonic()
        driver.step(FRAME_DT)
        accum += FRAME_DT
        if accum >= TICK_PERIOD:
            accum = 0.0
            state = perception.analyze(driver.get_screen_frame())
            bot.tick(state)
            ticks += 1
            tip = bot.root.tip()
            last_action = tip.name if tip is not None else "—"
            if args.headless and ticks % 10 == 0:
                print(f"  tick {ticks:4d} | action={last_action:<12} "
                      f"hp={state.hp} ammo={state.ammo} dets={len(state.detections)}")

        if args.max_ticks is not None and ticks >= args.max_ticks:
            break
        if args.max_seconds is not None and (time.monotonic() - start) >= args.max_seconds:
            break
        if not args.headless:
            elapsed = time.monotonic() - t0
            if elapsed < FRAME_DT:
                time.sleep(FRAME_DT - elapsed)

    print(f"\nStopped after {ticks} ticks. Last action: {last_action}")
    perception.close()
    driver.close()


if __name__ == "__main__":
    main()
