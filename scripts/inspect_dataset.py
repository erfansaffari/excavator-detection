#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from excavator_detection.dataset import discover_records, is_excavator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect MOCS files, classes, annotation format, and splits.")
    parser.add_argument("--data", type=Path, required=True, help="Root of the extracted MOCS dataset")
    parser.add_argument("--output", type=Path, default=Path("figures/dataset_samples"))
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--excavator-class-id", type=int, default=None, help="YOLO class ID; use only after verifying the class map")
    return parser.parse_args()


def draw_samples(records, output: Path, limit: int) -> int:
    selected = [record for record in records if any(is_excavator(box.class_name) for box in record.boxes)][:limit]
    output.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(selected):
        with Image.open(record.image_path) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        line_width = max(2, round(min(image.size) / 300))
        for box in record.boxes:
            if not is_excavator(box.class_name):
                continue
            draw.rectangle(box.xyxy, outline=(255, 60, 40), width=line_width)
            label = "Excavator"
            left, top, _, _ = box.xyxy
            text_box = draw.textbbox((left, top), label, font=ImageFont.load_default())
            draw.rectangle(text_box, fill=(255, 60, 40))
            draw.text((left, top), label, fill="white")
        image.thumbnail((1600, 1600))
        image.save(output / f"sample_{index:02d}_{record.image_path.stem}.jpg", quality=92)
    return len(selected)


def main() -> None:
    args = parse_args()
    if not args.data.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {args.data}")
    records, summary = discover_records(args.data, args.excavator_class_id)
    count = draw_samples(records, args.output, args.samples)
    report = summary.as_dict()
    report["matched_records"] = len(records)
    report["visualizations"] = count
    print(json.dumps(report, indent=2))
    if not any(is_excavator(name) for name in summary.classes):
        raise SystemExit("Excavator class not verified; inspect the class map and, for YOLO data only, pass --excavator-class-id.")


if __name__ == "__main__":
    main()
