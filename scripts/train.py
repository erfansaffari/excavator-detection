#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune COCO-pretrained YOLO11n on excavators.")
    parser.add_argument("--data", type=Path, default=Path("configs/excavator.yaml"))
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="0", help="CUDA device such as 0, or cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=Path("results/train"))
    parser.add_argument("--name", default="yolo11n_excavator")
    parser.add_argument("--smoke-test", action="store_true", help="Run one epoch with a small image/batch configuration")
    parser.add_argument("--allow-cpu", action="store_true", help="Permit CPU execution (GPU is required by default)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise SystemExit(f"Dataset config not found: {args.data}. Run prepare_dataset.py first.")
    wants_cuda = str(args.device).lower() != "cpu"
    if wants_cuda and not torch.cuda.is_available():
        if not args.allow_cpu:
            raise SystemExit("CUDA is unavailable. Use a Kaggle GPU session, or pass --device cpu --allow-cpu for a local check.")
        args.device = "cpu"
    settings = {
        "data": str(args.data.resolve()),
        "epochs": 1 if args.smoke_test else args.epochs,
        "imgsz": min(args.imgsz, 320) if args.smoke_test else args.imgsz,
        "batch": min(args.batch, 4) if args.smoke_test else args.batch,
        "fraction": 0.02 if args.smoke_test else 1.0,
        "patience": args.patience,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "deterministic": True,
        "pretrained": True,
        "project": str(args.project.resolve()),
        "name": f"{args.name}_smoke" if args.smoke_test else args.name,
        "exist_ok": False,
        "plots": True,
    }
    print(json.dumps(settings, indent=2))
    model = YOLO(args.weights)
    model.train(**settings)


if __name__ == "__main__":
    main()
