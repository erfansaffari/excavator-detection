from __future__ import annotations

import json
import os
import random
import hashlib
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}


@dataclass(frozen=True)
class Box:
    class_name: str
    xyxy: tuple[float, float, float, float]


@dataclass
class ImageRecord:
    image_path: Path
    width: int
    height: int
    boxes: list[Box] = field(default_factory=list)
    split: str | None = None


@dataclass
class DatasetSummary:
    formats: Counter = field(default_factory=Counter)
    images: int = 0
    annotated_images: int = 0
    boxes: int = 0
    classes: Counter = field(default_factory=Counter)
    splits: Counter = field(default_factory=Counter)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "formats": dict(self.formats),
            "images": self.images,
            "annotated_images": self.annotated_images,
            "boxes": self.boxes,
            "classes": dict(self.classes),
            "splits": dict(self.splits),
            "warnings": self.warnings,
        }


def normalize_class_name(value: str) -> str:
    return "".join(char.lower() for char in str(value) if char.isalnum())


def is_excavator(value: str) -> bool:
    return normalize_class_name(value) in {"excavator", "excavators"}


def infer_split(path: Path, root: Path) -> str | None:
    def classify(value: str) -> str | None:
        lowered = value.lower()
        direct = SPLIT_ALIASES.get(lowered)
        if direct:
            return direct
        return next(
            (canonical for alias, canonical in SPLIT_ALIASES.items() if lowered.startswith(alias) and lowered[len(alias):].isdigit()),
            None,
        )

    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    for part in reversed(parts[:-1]):
        split = classify(part)
        if split:
            return split
    stem_tokens = path.stem.lower().replace("-", "_").split("_")
    return next((split for token in stem_tokens if (split := classify(token))), None)


def _official_split_index(root: Path) -> dict[str, str]:
    """Read VOC-style train/val/test membership lists when present."""
    membership: dict[str, str] = {}
    for path in sorted(root.rglob("*.txt")):
        split = SPLIT_ALIASES.get(path.stem.lower())
        if not split:
            continue
        try:
            lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, UnicodeDecodeError):
            continue
        # Membership lists contain one image path/ID per line; reject YOLO label files.
        if not lines or any(len(line.split()) != 1 for line in lines):
            continue
        for line in lines:
            item = Path(line)
            membership[item.name.lower()] = split
            membership[item.stem.lower()] = split
    return membership


def image_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _image_index(root: Path) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    by_name: dict[str, list[Path]] = {}
    by_stem: dict[str, list[Path]] = {}
    for path in image_files(root):
        by_name.setdefault(path.name.lower(), []).append(path)
        by_stem.setdefault(path.stem.lower(), []).append(path)
    return by_name, by_stem


def _resolve_image(name: str, annotation: Path, root: Path, by_name: dict[str, list[Path]], by_stem: dict[str, list[Path]]) -> Path | None:
    supplied = Path(name)
    candidates = [annotation.parent / supplied, root / supplied]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    matches = by_name.get(supplied.name.lower()) or by_stem.get(supplied.stem.lower()) or []
    if len(matches) == 1:
        return matches[0].resolve()
    if matches:
        # Prefer the candidate sharing the most parent components with its annotation.
        ann_parts = set(annotation.parent.parts)
        annotation_split = infer_split(annotation, root)
        return max(
            matches,
            key=lambda p: (infer_split(p, root) == annotation_split, len(ann_parts.intersection(p.parent.parts))),
        ).resolve()
    return None


def _safe_size(path: Path, width: int | None = None, height: int | None = None) -> tuple[int, int]:
    if width and height:
        return int(width), int(height)
    with Image.open(path) as image:
        return image.size


def _clip_box(xyxy: Iterable[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = map(float, xyxy)
    x1, x2 = sorted((max(0.0, min(x1, width)), max(0.0, min(x2, width))))
    y1, y2 = sorted((max(0.0, min(y1, height)), max(0.0, min(y2, height))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def parse_coco(path: Path, root: Path, by_name: dict[str, list[Path]], by_stem: dict[str, list[Path]]) -> list[ImageRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not all(key in data for key in ("images", "annotations", "categories")):
        return []
    categories = {item["id"]: str(item["name"]) for item in data["categories"]}
    annotations: dict[int, list[dict]] = {}
    for annotation in data["annotations"]:
        annotations.setdefault(annotation["image_id"], []).append(annotation)
    records: list[ImageRecord] = []
    declared_split = infer_split(path, root)
    for item in data["images"]:
        resolved = _resolve_image(str(item["file_name"]), path, root, by_name, by_stem)
        if not resolved:
            continue
        width, height = _safe_size(resolved, item.get("width"), item.get("height"))
        boxes: list[Box] = []
        for annotation in annotations.get(item["id"], []):
            category = categories.get(annotation.get("category_id"), str(annotation.get("category_id", "unknown")))
            bbox = annotation.get("bbox")
            if bbox and len(bbox) == 4:
                x, y, w, h = map(float, bbox)
                clipped = _clip_box((x, y, x + w, y + h), width, height)
                if clipped:
                    boxes.append(Box(category, clipped))
        records.append(ImageRecord(resolved, width, height, boxes, declared_split or infer_split(resolved, root)))
    return records


def parse_voc(path: Path, root: Path, by_name: dict[str, list[Path]], by_stem: dict[str, list[Path]]) -> ImageRecord | None:
    tree = ET.parse(path)
    root_node = tree.getroot()
    if root_node.tag.lower() != "annotation":
        return None
    filename = root_node.findtext("filename") or f"{path.stem}.jpg"
    resolved = _resolve_image(filename, path, root, by_name, by_stem)
    if not resolved:
        return None
    size = root_node.find("size")
    width = int(float(size.findtext("width"))) if size is not None and size.findtext("width") else None
    height = int(float(size.findtext("height"))) if size is not None and size.findtext("height") else None
    width, height = _safe_size(resolved, width, height)
    boxes: list[Box] = []
    for obj in root_node.findall("object"):
        name = obj.findtext("name", "unknown")
        node = obj.find("bndbox")
        if node is None:
            continue
        values = [node.findtext(key) for key in ("xmin", "ymin", "xmax", "ymax")]
        if all(value is not None for value in values):
            clipped = _clip_box((float(v) for v in values), width, height)
            if clipped:
                boxes.append(Box(name, clipped))
    return ImageRecord(resolved, width, height, boxes, infer_split(path, root) or infer_split(resolved, root))


def parse_labelme(path: Path, root: Path, by_name: dict[str, list[Path]], by_stem: dict[str, list[Path]]) -> ImageRecord | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "shapes" not in data or "imagePath" not in data:
        return None
    resolved = _resolve_image(str(data["imagePath"]), path, root, by_name, by_stem)
    if not resolved:
        return None
    width, height = _safe_size(resolved, data.get("imageWidth"), data.get("imageHeight"))
    boxes: list[Box] = []
    for shape in data["shapes"]:
        points = shape.get("points", [])
        if len(points) < 2:
            continue
        xs, ys = zip(*((float(point[0]), float(point[1])) for point in points))
        clipped = _clip_box((min(xs), min(ys), max(xs), max(ys)), width, height)
        if clipped:
            boxes.append(Box(str(shape.get("label", "unknown")), clipped))
    return ImageRecord(resolved, width, height, boxes, infer_split(path, root) or infer_split(resolved, root))


def _load_yolo_names(root: Path) -> dict[int, str]:
    for yaml_path in sorted([*root.rglob("*.yaml"), *root.rglob("*.yml")]):
        try:
            names = yaml.safe_load(yaml_path.read_text(encoding="utf-8")).get("names")
            if isinstance(names, list):
                return {index: str(name) for index, name in enumerate(names)}
            if isinstance(names, dict):
                return {int(index): str(name) for index, name in names.items()}
        except (AttributeError, OSError, ValueError, yaml.YAMLError):
            pass
    return {}


def parse_yolo(root: Path, class_id: int | None = None) -> list[ImageRecord]:
    names = _load_yolo_names(root)
    if class_id is not None:
        names[class_id] = "Excavator"
    if not names:
        return []
    records: list[ImageRecord] = []
    for image_path in image_files(root):
        candidates = [
            image_path.with_suffix(".txt"),
            Path(str(image_path.with_suffix(".txt")).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")),
        ]
        label_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if label_path is None:
            continue
        width, height = _safe_size(image_path)
        boxes: list[Box] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5:
                continue
            category_id = int(float(fields[0]))
            cx, cy, w, h = map(float, fields[1:])
            clipped = _clip_box(((cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height), width, height)
            if clipped:
                boxes.append(Box(names.get(category_id, str(category_id)), clipped))
        records.append(ImageRecord(image_path.resolve(), width, height, boxes, infer_split(image_path, root)))
    return records


def discover_records(root: Path, excavator_class_id: int | None = None) -> tuple[list[ImageRecord], DatasetSummary]:
    root = root.resolve()
    by_name, by_stem = _image_index(root)
    records: dict[Path, ImageRecord] = {}
    summary = DatasetSummary(images=len([p for paths in by_name.values() for p in paths]))

    json_paths = sorted(root.rglob("*.json"))
    for path in json_paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            parsed = parse_coco(path, root, by_name, by_stem) if {"images", "annotations", "categories"}.issubset(data) else []
            if parsed:
                summary.formats["coco"] += 1
            elif "shapes" in data and "imagePath" in data:
                record = parse_labelme(path, root, by_name, by_stem)
                parsed = [record] if record else []
                summary.formats["labelme"] += int(bool(parsed))
            else:
                parsed = []
            for record in parsed:
                records[record.image_path] = record
        except (json.JSONDecodeError, OSError, ValueError, KeyError):
            continue

    for path in sorted(root.rglob("*.xml")):
        try:
            record = parse_voc(path, root, by_name, by_stem)
            if record:
                records[record.image_path] = record
                summary.formats["pascal_voc"] += 1
        except (ET.ParseError, OSError, ValueError):
            continue

    if not records:
        for record in parse_yolo(root, excavator_class_id):
            records[record.image_path] = record
        if records:
            summary.formats["yolo"] = len(records)

    split_index = _official_split_index(root)
    for record in records.values():
        if record.split is None:
            record.split = split_index.get(record.image_path.name.lower()) or split_index.get(record.image_path.stem.lower())

    for record in records.values():
        if record.boxes:
            summary.annotated_images += 1
        summary.boxes += len(record.boxes)
        summary.classes.update(box_.class_name for box_ in record.boxes)
        summary.splits[record.split or "unspecified"] += 1
    if summary.images and len(records) < summary.images:
        summary.warnings.append(f"Matched annotations to {len(records)} of {summary.images} discovered images.")
    if not any(is_excavator(name) for name in summary.classes):
        summary.warnings.append("No class named 'Excavator' was verified. Supply --excavator-class-id only after checking the dataset's class map.")
    return sorted(records.values(), key=lambda record: str(record.image_path)), summary


def assign_splits(records: list[ImageRecord], seed: int = 42) -> list[ImageRecord]:
    official = {record.split for record in records if record.split}
    if {"train", "val", "test"}.issubset(official) and all(record.split for record in records):
        return records

    rng = random.Random(seed)
    if "test" in official and "train" in official and "val" not in official:
        for record in records:
            if record.split is None:
                record.split = "train"
        train_pool = [record for record in records if record.split == "train"]
        rng.shuffle(train_pool)
        val_count = max(1, round(len(train_pool) * 0.15 / 0.85)) if len(train_pool) > 1 else 0
        for record in train_pool[:val_count]:
            record.split = "val"
        return records

    shuffled = list(records)
    rng.shuffle(shuffled)
    count = len(shuffled)
    train_end = round(count * 0.70)
    val_end = train_end + round(count * 0.15)
    if count >= 3:
        train_end = min(max(train_end, 1), count - 2)
        val_end = min(max(val_end, train_end + 1), count - 1)
    for index, record in enumerate(shuffled):
        record.split = "train" if index < train_end else "val" if index < val_end else "test"
    return records


def _unique_destination(record: ImageRecord, split: str, output: Path) -> Path:
    destination = output / "images" / split / record.image_path.name
    if not destination.exists():
        return destination
    suffix = hashlib.sha1(str(record.image_path).encode("utf-8"), usedforsecurity=False).hexdigest()[:10]
    return destination.with_stem(f"{destination.stem}_{suffix}")


def write_yolo_dataset(
    records: list[ImageRecord], output: Path, drop_empty: bool = False, image_mode: str = "copy"
) -> dict:
    if image_mode not in {"copy", "symlink", "hardlink"}:
        raise ValueError(f"Unsupported image mode: {image_mode}")
    output = output.resolve()
    counts = Counter()
    box_counts = Counter()
    for record in records:
        excavator_boxes = [box for box in record.boxes if is_excavator(box.class_name)]
        if drop_empty and not excavator_boxes:
            continue
        split = record.split or "train"
        destination = _unique_destination(record, split, output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        label_path = output / "labels" / split / destination.with_suffix(".txt").name
        label_path.parent.mkdir(parents=True, exist_ok=True)
        if image_mode == "symlink":
            destination.symlink_to(record.image_path)
        elif image_mode == "hardlink":
            os.link(record.image_path, destination)
        else:
            shutil.copy2(record.image_path, destination)
        lines = []
        for box in excavator_boxes:
            x1, y1, x2, y2 = box.xyxy
            cx = ((x1 + x2) / 2) / record.width
            cy = ((y1 + y2) / 2) / record.height
            width = (x2 - x1) / record.width
            height = (y2 - y1) / record.height
            lines.append(f"0 {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts[split] += 1
        box_counts[split] += len(lines)
    return {
        "images": dict(counts),
        "excavator_boxes": dict(box_counts),
        "output": str(output),
        "image_mode": image_mode,
    }


def write_dataset_yaml(output: Path, config_path: Path) -> None:
    content = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "Excavator"},
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
