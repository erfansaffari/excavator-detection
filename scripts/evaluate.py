#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the best excavator checkpoint on the held-out test split.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("configs/excavator.yaml"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--name", default="test")
    return parser.parse_args()


def _number(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        raise SystemExit(f"Checkpoint not found: {args.weights}")
    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data.resolve()),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project.resolve()),
        name=args.name,
        plots=True,
        save_json=True,
    )
    box = metrics.box
    result = {
        "model": str(args.weights.resolve()),
        "split": "test",
        "AP@50": _number(box.map50),
        "mAP@50-95": _number(box.map),
        "precision": _number(box.mp),
        "recall": _number(box.mr),
        "save_dir": str(metrics.save_dir),
    }
    output = Path(metrics.save_dir) / "metrics.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
