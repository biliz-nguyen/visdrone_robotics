from __future__ import annotations

from pathlib import Path
import os

import yaml
from PIL import Image

from .config import CLASS_NAMES


# Official VisDrone category id -> project class id.
# VisDrone 0=ignored regions, 11=others; both are excluded from training.
VISDRONE_TO_PROJECT = {
    1: 5,   # pedestrian
    2: 6,   # people
    3: 1,   # bicycle
    4: 3,   # car
    5: 9,   # van
    6: 8,   # truck
    7: 7,   # tricycle
    8: 0,   # awning-tricycle
    9: 2,   # bus
    10: 4,  # motor
}


def _ensure_symlink(link: Path, target: Path) -> None:
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(
            f"Expected symlink path but found existing file/directory: {link}"
        )

    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target.resolve(), link, target_is_directory=True)


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.width, im.height


def _convert_split(
    images_dir: Path,
    annotations_dir: Path,
    out_split: Path,
    *,
    require_annotations: bool,
) -> dict[str, int]:
    if not images_dir.is_dir():
        raise FileNotFoundError(images_dir)
    if require_annotations and not annotations_dir.is_dir():
        raise FileNotFoundError(annotations_dir)

    _ensure_symlink(out_split / "images", images_dir)
    labels_dir = out_split / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    converted_objects = 0
    ignored_objects = 0
    missing_annotations = 0

    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        ann_path = annotations_dir / f"{image_path.stem}.txt"

        if not ann_path.exists():
            if require_annotations:
                missing_annotations += 1
            label_path.write_text("", encoding="utf-8")
            continue

        width, height = _image_size(image_path)
        yolo_lines: list[str] = []

        for raw in ann_path.read_text(encoding="utf-8-sig").splitlines():
            raw = raw.strip().strip(",")
            if not raw:
                continue

            parts = [p.strip() for p in raw.split(",")]
            if len(parts) < 6:
                raise ValueError(f"Malformed VisDrone row in {ann_path}: {raw}")

            left, top, box_w, box_h = map(float, parts[:4])
            category = int(float(parts[5]))
            class_id = VISDRONE_TO_PROJECT.get(category)

            if class_id is None:
                ignored_objects += 1
                continue
            if box_w <= 0 or box_h <= 0 or width <= 0 or height <= 0:
                ignored_objects += 1
                continue

            # Clip the box to image bounds before normalization.
            x1 = max(0.0, min(float(width), left))
            y1 = max(0.0, min(float(height), top))
            x2 = max(0.0, min(float(width), left + box_w))
            y2 = max(0.0, min(float(height), top + box_h))
            if x2 <= x1 or y2 <= y1:
                ignored_objects += 1
                continue

            cx = ((x1 + x2) / 2.0) / width
            cy = ((y1 + y2) / 2.0) / height
            nw = (x2 - x1) / width
            nh = (y2 - y1) / height

            yolo_lines.append(
                f"{class_id} {cx:.8f} {cy:.8f} {nw:.8f} {nh:.8f}"
            )
            converted_objects += 1

        label_path.write_text(
            "\n".join(yolo_lines) + ("\n" if yolo_lines else ""),
            encoding="utf-8",
        )

    if require_annotations and missing_annotations:
        raise RuntimeError(
            f"{missing_annotations} images in {images_dir} have no matching annotation file"
        )

    return {
        "images": len(image_paths),
        "objects": converted_objects,
        "ignored": ignored_objects,
    }


def _prepare_official_visdrone(cfg: dict) -> tuple[Path, dict[str, str]]:
    root = Path(cfg["dataset_root"])
    cache_root = root / ".visdrone_yolo"
    cache_root.mkdir(parents=True, exist_ok=True)

    split_specs = {
        "train": (
            root / cfg["train_images"],
            root / cfg["train_annotations"],
            True,
        ),
        "val": (
            root / cfg["val_images"],
            root / cfg["val_annotations"],
            True,
        ),
        "test": (
            root / cfg["test_images"],
            root / cfg["test_annotations"],
            False,
        ),
    }

    stats = {}
    for split, (images_dir, annotations_dir, required) in split_specs.items():
        stats[split] = _convert_split(
            images_dir,
            annotations_dir,
            cache_root / split,
            require_annotations=required,
        )

    print("Official VisDrone -> YOLO cache:", cache_root)
    for split in ("train", "val", "test"):
        s = stats[split]
        print(
            f"  {split}: images={s['images']} objects={s['objects']} ignored={s['ignored']}"
        )

    return cache_root, {
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
    }


def build_data_yaml(cfg: dict) -> Path:
    generated = Path(cfg["generated_dir"])
    generated.mkdir(parents=True, exist_ok=True)

    if cfg.get("dataset_format") == "visdrone_official":
        root, splits = _prepare_official_visdrone(cfg)
    else:
        root = Path(cfg["dataset_root"])
        splits = {
            "train": cfg["train_images"],
            "val": cfg["val_images"],
            "test": cfg["test_images"],
        }

        for split, rel in splits.items():
            p = root / rel
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing {split} images directory: {p}\n"
                    "Edit config/local.yaml if your folder names differ."
                )

    data_yaml = generated / "visdrone_local.yaml"

    payload = {
        "path": str(root),
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        "nc": len(CLASS_NAMES),
        "names": {
            i: name
            for i, name in enumerate(CLASS_NAMES)
        },
    }

    data_yaml.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    return data_yaml
