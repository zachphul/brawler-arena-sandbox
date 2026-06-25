"""Verify the HUD OCR reads HP and AMMO off a real rendered arena frame.

Run from the 3.12 vision env:
    .venv312\\Scripts\\python tools/check_hud_ocr.py
Exits non-zero if HP/AMMO don't match the known HUD (HP 100, AMMO 3).
"""

import os
import pathlib
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from arena.pygame_driver import PygameDriver
from arena.simulation import Arena
from eyes.vision_pipeline import VisionEngine


def main() -> None:
    arena = Arena(seed=1, headless=True)
    driver = PygameDriver(arena)
    arena.set_driver(driver)
    arena.step(1 / 60)  # render the HUD: "HP: 100", "AMMO: 3", "GEMS: 0"
    frame = driver.get_screen_frame()

    engine = VisionEngine(model_path=str(_ROOT / "models" / "best.pt"))
    hp, ammo, text = engine.process_hud(frame)
    engine.close()
    arena.close()

    print(f"raw OCR text : {text!r}")
    print(f"parsed       : HP={hp}  AMMO={ammo}")
    ok = hp == 100 and ammo == 3
    print("RESULT       :", "OK ✅" if ok else "MISMATCH (expected HP=100 AMMO=3)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
