"""Overlay YOLO labels back onto a dataset image for visual QA.

Reads a normalized YOLO ``.txt`` and draws each box (denormalized) on the image,
proving the harvest math is correct.

Usage:  python tools/draw_labels.py <image.png> <label.txt> [out.png]
"""

import os
import pathlib
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

COLORS = {0: (70, 150, 255), 1: (235, 80, 80), 2: (255, 220, 120),
          3: (80, 220, 130), 4: (180, 110, 240)}
NAMES = ["Player", "Enemy", "Projectile", "Gem", "PowerCube"]


def main() -> None:
    img_path = pathlib.Path(sys.argv[1])
    lbl_path = pathlib.Path(sys.argv[2])
    out = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else \
        img_path.with_name("labeled_" + img_path.name)

    pygame.display.init()
    pygame.font.init()
    pygame.display.set_mode((1, 1))
    surf = pygame.image.load(str(img_path))
    w_img, h_img = surf.get_size()
    font = pygame.font.SysFont("consolas", 12)

    for ln in lbl_path.read_text().splitlines():
        if not ln.strip():
            continue
        cid, ncx, ncy, nw, nh = ln.split()
        cid = int(cid)
        bw, bh = float(nw) * w_img, float(nh) * h_img
        x = float(ncx) * w_img - bw / 2.0
        y = float(ncy) * h_img - bh / 2.0
        color = COLORS.get(cid, (255, 255, 255))
        pygame.draw.rect(surf, color, pygame.Rect(int(x), int(y), int(bw), int(bh)), 2)
        surf.blit(font.render(NAMES[cid], True, color), (int(x), max(0, int(y) - 12)))

    pygame.image.save(surf, str(out))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
