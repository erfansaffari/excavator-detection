from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from PIL import Image

from excavator_detection.dataset import assign_splits, discover_records, is_excavator, write_yolo_dataset


def make_image(path: Path, size=(100, 50)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)


def test_coco_filters_and_converts_excavator(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "frame.jpg"
    make_image(image_path)
    annotation = {
        "images": [{"id": 1, "file_name": "frame.jpg", "width": 100, "height": 50}],
        "categories": [{"id": 7, "name": "Excavator"}, {"id": 8, "name": "Worker"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 7, "bbox": [10, 5, 40, 20]},
            {"id": 2, "image_id": 1, "category_id": 8, "bbox": [0, 0, 10, 10]},
        ],
    }
    (tmp_path / "annotations.json").write_text(json.dumps(annotation), encoding="utf-8")

    records, summary = discover_records(tmp_path)
    assert len(records) == 1
    assert summary.classes == Counter({"Excavator": 1, "Worker": 1})
    records[0].split = "train"
    output = tmp_path / "processed"
    write_yolo_dataset(records, output)
    label = (output / "labels/train/frame.txt").read_text().strip()
    assert label == "0 0.30000000 0.30000000 0.40000000 0.40000000"


def test_labelme_polygon_becomes_box(tmp_path: Path) -> None:
    make_image(tmp_path / "scene.png", (200, 100))
    annotation = {
        "imagePath": "scene.png",
        "imageWidth": 200,
        "imageHeight": 100,
        "shapes": [{"label": "excavator", "points": [[20, 10], [80, 90], [120, 30]]}],
    }
    (tmp_path / "scene.json").write_text(json.dumps(annotation), encoding="utf-8")
    records, _ = discover_records(tmp_path)
    assert records[0].boxes[0].xyxy == (20.0, 10.0, 120.0, 90.0)
    assert is_excavator(records[0].boxes[0].class_name)


def test_seeded_split_is_deterministic_and_70_15_15(tmp_path: Path) -> None:
    annotation = {"images": [], "categories": [{"id": 1, "name": "Excavator"}], "annotations": []}
    for index in range(20):
        filename = f"{index:02d}.jpg"
        make_image(tmp_path / filename)
        annotation["images"].append({"id": index, "file_name": filename, "width": 100, "height": 50})
        annotation["annotations"].append({"id": index, "image_id": index, "category_id": 1, "bbox": [1, 1, 10, 10]})
    (tmp_path / "instances.json").write_text(json.dumps(annotation), encoding="utf-8")

    first, _ = discover_records(tmp_path)
    second, _ = discover_records(tmp_path)
    assign_splits(first, seed=42)
    assign_splits(second, seed=42)
    assert [record.split for record in first] == [record.split for record in second]
    assert Counter(record.split for record in first) == Counter({"train": 14, "val": 3, "test": 3})


def test_preserves_complete_official_split(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        make_image(tmp_path / split / f"{split}.jpg")
        xml = f"""<annotation><filename>{split}.jpg</filename><size><width>100</width><height>50</height></size>
        <object><name>Excavator</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>20</xmax><ymax>30</ymax></bndbox></object></annotation>"""
        (tmp_path / split / f"{split}.xml").write_text(xml, encoding="utf-8")
    records, _ = discover_records(tmp_path)
    before = {record.image_path.name: record.split for record in records}
    assign_splits(records, seed=42)
    assert {record.image_path.name: record.split for record in records} == before


def test_reads_voc_split_membership_files(tmp_path: Path) -> None:
    annotation_dir = tmp_path / "Annotations"
    image_dir = tmp_path / "JPEGImages"
    annotation_dir.mkdir()
    for split in ("train", "val", "test"):
        make_image(image_dir / f"{split}.jpg")
        xml = f"""<annotation><filename>{split}.jpg</filename><size><width>100</width><height>50</height></size>
        <object><name>Excavator</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>20</xmax><ymax>30</ymax></bndbox></object></annotation>"""
        (annotation_dir / f"{split}.xml").write_text(xml, encoding="utf-8")
        split_dir = tmp_path / "ImageSets/Main"
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / f"{split}.txt").write_text(f"{split}\n", encoding="utf-8")
    records, _ = discover_records(tmp_path)
    assert {record.image_path.stem: record.split for record in records} == {"train": "train", "val": "val", "test": "test"}
