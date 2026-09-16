#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect excavators in an image, video, or image directory.")
    parser.add_argument("--source", required=True, help="Image, video, directory, URL, or webcam index")
    parser.add_argument("--weights", type=Path, default=Path("results/train/yolo11n_excavator/weights/best.pt"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", type=Path, default=Path("results/predictions"))
    parser.add_argument("--name", default="predict")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        raise SystemExit(f"Checkpoint not found: {args.weights}. Train a model or pass --weights.")
    model = YOLO(str(args.weights))
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        save_conf=True,
        show=args.show,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        verbose=True,
    )
    if results:
        print(f"Saved annotated output to: {results[0].save_dir}")


if __name__ == "__main__":
    main()
