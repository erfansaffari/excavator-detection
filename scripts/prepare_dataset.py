#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from excavator_detection.dataset import (
    assign_splits,
    discover_records,
    is_excavator,
    write_dataset_yaml,
    write_yolo_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MOCS excavator annotations to a single-class YOLO dataset.")
    parser.add_argument("--data", type=Path, required=True, help="Root of the extracted raw dataset")
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--config", type=Path, default=Path("configs/excavator.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop-empty", action="store_true", help="Exclude images that contain no excavator boxes")
    parser.add_argument(
        "--image-mode",
        choices=("copy", "symlink", "hardlink"),
        default="copy",
        help="How prepared images reference source files; symlink saves Kaggle working-disk space",
    )
    parser.add_argument("--excavator-class-id", type=int, default=None, help="YOLO class ID; only use after inspecting class names")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {args.data}")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Output is not empty: {args.output}. Remove it or choose another --output to avoid mixed splits.")
    records, summary = discover_records(args.data, args.excavator_class_id)
    if not records:
        raise SystemExit("No supported annotations could be paired with images. Run inspect_dataset.py and review its warnings.")
    if not any(is_excavator(name) for name in summary.classes):
        raise SystemExit("No verified Excavator class. Do not guess: inspect the dataset class map first.")
    if args.drop_empty:
        records = [
            record for record in records
            if any(is_excavator(box.class_name) for box in record.boxes)
        ]
        if not records:
            raise SystemExit("No images containing excavator annotations remain after filtering.")
    records = assign_splits(records, args.seed)
    result = write_yolo_dataset(records, args.output, args.drop_empty, args.image_mode)
    write_dataset_yaml(args.output, args.config)
    manifest = {
        "source": str(args.data.resolve()),
        "seed": args.seed,
        "drop_empty": args.drop_empty,
        "inspection": summary.as_dict(),
        "prepared": result,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
