"""Autonomous synthetic-dataset generator for the Brawler Arena.

Runs the arena headlessly, drives entities with the Bézier kinematics for
diverse frames, and dumps YOLO-format images + labels into a standard
train/val hierarchy with a ``dataset.yaml`` — all derived from the engine's true
internal coordinates, so labels are pixel-perfect and free.

Usage:
    python tools/dataset_generator.py                        # 1000 frames -> dataset/
    python tools/dataset_generator.py --frames 100 --clean   # quick test harvest
    python tools/dataset_generator.py --out dataset --val-split 0.2
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from collections import Counter

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import pygame

from arena.pygame_driver import PygameDriver
from arena.simulation import Arena, WanderAgent

# class_id -> name; index must match arena.entity_boxes() class ids.
CLASS_NAMES = ["Player", "Enemy", "Projectile", "Gem", "PowerCube"]


def to_yolo(cx, cy, w, h, img_w, img_h):
    """Map a pixel box (center/size) to clipped, normalized YOLO ``xywh`` in [0,1].

    Boxes are clipped to the frame first (entities near an edge), then converted
    to normalized center/size. Returns ``None`` if the box is essentially
    off-screen after clipping.
    """
    x1, y1 = cx - w / 2.0, cy - h / 2.0
    x2, y2 = cx + w / 2.0, cy + h / 2.0
    x1 = min(max(x1, 0.0), img_w); x2 = min(max(x2, 0.0), img_w)
    y1 = min(max(y1, 0.0), img_h); y2 = min(max(y2, 0.0), img_h)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1.0 or bh <= 1.0:
        return None
    ncx = ((x1 + x2) / 2.0) / img_w
    ncy = ((y1 + y2) / 2.0) / img_h
    return (
        min(max(ncx, 0.0), 1.0),
        min(max(ncy, 0.0), 1.0),
        min(max(bw / img_w, 0.0), 1.0),
        min(max(bh / img_h, 0.0), 1.0),
    )


def _inject_dynamics(arena, rng):
    """Randomized spawning + frequent firing for diverse, busy frames."""
    if rng.random() < 0.30 and len(arena.items) < 10:
        arena._spawn_item()
    if rng.random() < 1.0 / 150.0:            # occasionally vary item density
        arena.items.clear()
    if rng.random() < 0.25:                    # frequent shots -> projectile samples
        if arena.player.ammo <= 0:
            arena.player.ammo = arena.player.max_ammo
        arena.register_tap(rng.uniform(0, arena.size[0]),
                           rng.uniform(0, arena.size[1]))


def prepare_dirs(out, clean):
    subs = [out / "images" / "train", out / "images" / "val",
            out / "labels" / "train", out / "labels" / "val"]
    for sub in subs:
        sub.mkdir(parents=True, exist_ok=True)
        if clean:
            for f in sub.glob("*"):
                f.unlink()


def write_yaml(out):
    lines = [
        "# Brawler Arena synthetic dataset (auto-generated)",
        f"path: {out.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
        *[f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)],
    ]
    (out / "dataset.yaml").write_text("\n".join(lines) + "\n")


def harvest(num_frames, out, *, seed=0, val_split=0.2, clean=False):
    prepare_dirs(out, clean)
    arena = Arena(seed=seed, headless=True)
    driver = PygameDriver(arena)
    arena.set_driver(driver)
    arena.player.max_ammo = 30                 # keep projectiles flowing
    agent = WanderAgent(resolution=arena.size, seed=seed + 1, fps=arena.fps)
    rng = np.random.default_rng(seed + 2)
    img_w, img_h = arena.size
    dt = 1.0 / arena.fps

    # Deterministic ~val_split fraction routed to validation.
    val_every = max(2, round(1.0 / val_split)) if val_split > 0 else None

    class_counts = Counter()
    n_train = n_val = 0
    for i in range(num_frames):
        agent(driver)                          # player: Bézier joystick stream
        _inject_dynamics(arena, rng)
        arena.step(dt)                         # enemy Bézier patrol + render

        split = "val" if (val_every and i % val_every == 0) else "train"
        stem = f"frame_{i:05d}"
        pygame.image.save(arena.screen, str(out / "images" / split / f"{stem}.png"))

        lines = []
        for cid, cx, cy, w, h in arena.entity_boxes():
            box = to_yolo(cx, cy, w, h, img_w, img_h)
            if box is None:
                continue
            class_counts[cid] += 1
            lines.append(f"{cid} " + " ".join(f"{v:.6f}" for v in box))
        (out / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
        n_train += split == "train"
        n_val += split == "val"

    write_yaml(out)
    arena.close()
    return {"train": n_train, "val": n_val, "class_counts": class_counts}


def verify(out):
    """Cross-check image/label pairing and that all values are normalized [0,1]."""
    issues = []
    totals = {}
    for split in ("train", "val"):
        imgs = sorted((out / "images" / split).glob("*.png"))
        lbls = sorted((out / "labels" / split).glob("*.txt"))
        totals[split] = len(imgs)
        if len(imgs) != len(lbls):
            issues.append(f"{split}: {len(imgs)} images vs {len(lbls)} labels")
        for img in imgs:
            lbl = out / "labels" / split / f"{img.stem}.txt"
            if not lbl.exists():
                issues.append(f"{split}: missing label for {img.name}")
                continue
            for ln in lbl.read_text().splitlines():
                if not ln.strip():
                    continue
                parts = ln.split()
                cid = int(parts[0])
                vals = [float(x) for x in parts[1:]]
                if not (0 <= cid < len(CLASS_NAMES)):
                    issues.append(f"{lbl.name}: bad class id {cid}")
                if len(vals) != 4 or any(v < 0.0 or v > 1.0 for v in vals):
                    issues.append(f"{lbl.name}: bad/un-normalized values {ln!r}")
    return totals, issues


def main():
    parser = argparse.ArgumentParser(description="Brawler Arena YOLO dataset generator")
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--out", type=pathlib.Path, default=_ROOT / "dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--clean", action="store_true",
                        help="empty the split dirs before harvesting")
    args = parser.parse_args()

    out = args.out.resolve()
    print(f"Harvesting {args.frames} frames -> {out}")
    stats = harvest(args.frames, out, seed=args.seed,
                    val_split=args.val_split, clean=args.clean)
    totals, issues = verify(out)

    print("\n=== harvest complete ===")
    print(f"  train images : {stats['train']}")
    print(f"  val images   : {stats['val']}")
    print("  instances per class:")
    for cid, name in enumerate(CLASS_NAMES):
        print(f"    {cid} {name:<10}: {stats['class_counts'].get(cid, 0)}")
    print(f"  dataset.yaml : {out / 'dataset.yaml'}")
    if issues:
        print(f"\n  !! {len(issues)} verification issue(s):")
        for msg in issues[:10]:
            print(f"     - {msg}")
        sys.exit(1)
    print("\n  verification OK — every label paired and normalized in [0, 1].")


if __name__ == "__main__":
    main()
