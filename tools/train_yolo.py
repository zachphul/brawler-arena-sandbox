"""Train YOLOv8n on the Brawler Arena synthetic dataset.

Usage:
    python tools/train_yolo.py                     # 10 epochs, yolov8n
    python tools/train_yolo.py --epochs 50 --device 0   # longer run on GPU 0

On completion the best weights are copied to ``models/best.pt`` — the default
path the VisionEngine loads.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on the arena dataset")
    parser.add_argument("--data", default=str(_ROOT / "dataset" / "dataset.yaml"))
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None,
                        help="'cpu', or a GPU index like '0' (default: auto)")
    parser.add_argument("--name", default="arena_yolov8n")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="fraction of the train set to use (e.g. 0.03 for a smoke run)")
    args = parser.parse_args()

    # Anchor downloads (yolov8n.pt) and outputs inside the project, regardless
    # of the shell's working directory.
    os.chdir(_ROOT)

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed. In the 3.12 env: "
                 "pip install ultralytics easyocr torch")

    if not pathlib.Path(args.data).exists():
        sys.exit(f"dataset yaml not found: {args.data}\n"
                 "Run tools/dataset_generator.py first.")

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        fraction=args.fraction,
        project=str(_ROOT / "runs"),
        name=args.name,
    )

    # Publish the best weights to the stable path the VisionEngine expects.
    best = pathlib.Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        models_dir = _ROOT / "models"
        models_dir.mkdir(exist_ok=True)
        dest = models_dir / "best.pt"
        shutil.copy2(best, dest)
        print(f"\nBest weights -> {dest}")
    else:
        print(f"\n(could not find best.pt under {results.save_dir})")
    print("Training complete.")


if __name__ == "__main__":
    main()
