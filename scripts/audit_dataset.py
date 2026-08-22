#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


CLASS_NAMES = {
    0: "ignored-regions",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}

EXPECTED_IMAGES = {
    "train": 6471,
    "val": 548,
    "test-dev": 1610,
}

SPLITS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test-dev": "VisDrone2019-DET-test-dev",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.width, im.height


def audit_split(root: Path, split: str, folder: str) -> dict:
    split_root = root / folder
    images_dir = split_root / "images"
    anns_dir = split_root / "annotations"

    if not images_dir.is_dir():
        raise FileNotFoundError(images_dir)
    if not anns_dir.is_dir():
        raise FileNotFoundError(anns_dir)

    images = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    annotations = sorted(p for p in anns_dir.glob("*.txt") if p.is_file())

    image_stems = {p.stem for p in images}
    ann_stems = {p.stem for p in annotations}
    image_by_stem = {p.stem: p for p in images}

    missing_annotations = sorted(image_stems - ann_stems)
    orphan_annotations = sorted(ann_stems - image_stems)

    class_counts = Counter()
    truncation_counts = Counter()
    occlusion_counts = Counter()

    malformed_rows = 0
    invalid_category = 0
    nonpositive_boxes = 0
    boxes_outside_image = 0
    duplicate_rows = 0
    kept_objects = 0
    ignored_regions = 0
    others = 0
    empty_after_filter = 0

    tiny_area_lt_16sq = 0
    small_area_lt_32sq = 0
    min_side_lt_16 = 0

    for stem in sorted(image_stems & ann_stems):
        image_path = image_by_stem[stem]
        width, height = image_size(image_path)
        ann_path = anns_dir / f"{stem}.txt"

        seen_rows = set()
        kept_this_image = 0

        for raw in ann_path.read_text(encoding="utf-8-sig").splitlines():
            raw = raw.strip().strip(",")
            if not raw:
                continue

            parts = tuple(p.strip() for p in raw.split(","))
            if parts in seen_rows:
                duplicate_rows += 1
            else:
                seen_rows.add(parts)

            if len(parts) < 8:
                malformed_rows += 1
                continue

            try:
                left, top, box_w, box_h = map(float, parts[:4])
                category = int(float(parts[5]))
                truncation = int(float(parts[6]))
                occlusion = int(float(parts[7]))
            except ValueError:
                malformed_rows += 1
                continue

            if category not in CLASS_NAMES:
                invalid_category += 1
                continue

            class_counts[category] += 1
            truncation_counts[truncation] += 1
            occlusion_counts[occlusion] += 1

            if category == 0:
                ignored_regions += 1
                continue
            if category == 11:
                others += 1
                continue

            if box_w <= 0 or box_h <= 0:
                nonpositive_boxes += 1
                continue

            if left < 0 or top < 0 or left + box_w > width or top + box_h > height:
                boxes_outside_image += 1

            kept_objects += 1
            kept_this_image += 1

            area = box_w * box_h
            if area < 16 * 16:
                tiny_area_lt_16sq += 1
            if area < 32 * 32:
                small_area_lt_32sq += 1
            if min(box_w, box_h) < 16:
                min_side_lt_16 += 1

        if kept_this_image == 0:
            empty_after_filter += 1

    return {
        "split": split,
        "folder": folder,
        "images": len(images),
        "expected_images": EXPECTED_IMAGES[split],
        "image_count_ok": len(images) == EXPECTED_IMAGES[split],
        "annotation_files": len(annotations),
        "missing_annotations": len(missing_annotations),
        "missing_annotation_examples": missing_annotations[:10],
        "orphan_annotations": len(orphan_annotations),
        "orphan_annotation_examples": orphan_annotations[:10],
        "kept_objects_1_to_10": kept_objects,
        "ignored_regions_category_0": ignored_regions,
        "others_category_11": others,
        "empty_images_after_filter": empty_after_filter,
        "malformed_rows": malformed_rows,
        "invalid_category_rows": invalid_category,
        "nonpositive_boxes": nonpositive_boxes,
        "boxes_outside_image": boxes_outside_image,
        "duplicate_rows": duplicate_rows,
        "size_stats": {
            "area_lt_16x16": tiny_area_lt_16sq,
            "area_lt_32x32": small_area_lt_32sq,
            "min_side_lt_16": min_side_lt_16,
        },
        "class_counts": {
            CLASS_NAMES[k]: class_counts[k]
            for k in sorted(CLASS_NAMES)
        },
        "truncation_counts": dict(sorted(truncation_counts.items())),
        "occlusion_counts": dict(sorted(occlusion_counts.items())),
    }


def markdown_report(report: dict) -> str:
    lines = [
        "# VisDrone2019-DET Dataset Audit",
        "",
        "Source: official train / val / test-dev folders on the self-hosted runner.",
        "",
        "## Split integrity",
        "",
        "| Split | Images | Expected | Ann files | Missing ann | Orphan ann | Kept objects | Empty after filter |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for split in ("train", "val", "test-dev"):
        s = report["splits"][split]
        lines.append(
            f"| {split} | {s['images']} | {s['expected_images']} | "
            f"{s['annotation_files']} | {s['missing_annotations']} | "
            f"{s['orphan_annotations']} | {s['kept_objects_1_to_10']} | "
            f"{s['empty_images_after_filter']} |"
        )

    lines += [
        "",
        "## Annotation health",
        "",
        "| Split | Malformed | Invalid category | Non-positive box | Outside image | Duplicate rows | Ignored regions | Others |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for split in ("train", "val", "test-dev"):
        s = report["splits"][split]
        lines.append(
            f"| {split} | {s['malformed_rows']} | {s['invalid_category_rows']} | "
            f"{s['nonpositive_boxes']} | {s['boxes_outside_image']} | "
            f"{s['duplicate_rows']} | {s['ignored_regions_category_0']} | "
            f"{s['others_category_11']} |"
        )

    lines += [
        "",
        "## Small-object indicators",
        "",
        "Definitions are pixel-space on original images; these are descriptive audit counts, not COCO size bins.",
        "",
        "| Split | Area < 16x16 | Area < 32x32 | Min side < 16 px |",
        "|---|---:|---:|---:|",
    ]

    for split in ("train", "val", "test-dev"):
        s = report["splits"][split]["size_stats"]
        lines.append(
            f"| {split} | {s['area_lt_16x16']} | {s['area_lt_32x32']} | {s['min_side_lt_16']} |"
        )

    lines += [
        "",
        "## Train/val class distribution",
        "",
        "| Class | Train | Val |",
        "|---|---:|---:|",
    ]

    for category in range(1, 11):
        name = CLASS_NAMES[category]
        lines.append(
            f"| {name} | {report['splits']['train']['class_counts'][name]} | "
            f"{report['splits']['val']['class_counts'][name]} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=Path("reports/dataset_audit.json"))
    parser.add_argument("--md", type=Path, default=Path("reports/dataset_audit.md"))
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    report = {
        "dataset_root": str(root),
        "splits": {
            split: audit_split(root, split, folder)
            for split, folder in SPLITS.items()
        },
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.md.write_text(markdown_report(report), encoding="utf-8")

    print(args.md.read_text(encoding="utf-8"))

    train = report["splits"]["train"]
    val = report["splits"]["val"]
    assert train["image_count_ok"]
    assert val["image_count_ok"]
    assert train["missing_annotations"] == 0
    assert val["missing_annotations"] == 0
    assert train["malformed_rows"] == 0
    assert val["malformed_rows"] == 0
    assert train["invalid_category_rows"] == 0
    assert val["invalid_category_rows"] == 0


if __name__ == "__main__":
    main()
