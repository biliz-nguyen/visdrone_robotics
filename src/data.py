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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.width, im.height


def _prepare_image_links(images_dir: Path, out_images: Path) -> list[Path]:
    """Create real cache image paths so Ultralytics finds sibling labels/.

    A directory-level symlink is intentionally avoided because Ultralytics may
    resolve it back to the official VisDrone folder and then search for labels
    beside the raw images. Hard links keep storage overhead near zero while
    preserving the cache path layout expected by img2label_paths().
    """
    out_images.mkdir(parents=True, exist_ok=True)

    source_images = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    expected_names = {p.name for p in source_images}

    for stale in list(out_images.iterdir()):
        if stale.name not in expected_names:
            if stale.is_file() or stale.is_symlink():
                stale.unlink()

    for src in source_images:
        dst = out_images / src.name
        if dst.exists() or dst.is_symlink():
            # Replace old per-file symlinks/copies if necessary.
            try:
                if dst.stat().st_ino == src.stat().st_ino:
                    continue
            except FileNotFoundError:
                pass
            dst.unlink()
        os.link(src, dst)

    return [out_images / p.name for p in source_images]


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

    out_images = out_split / "images"
    labels_dir = out_split / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _prepare_image_links(images_dir, out_images)

    converted_objects = 0
    ignored_objects = 0
    duplicate_rows = 0
    missing_annotations = 0

    for cached_image_path in image_paths:
        source_image_path = images_dir / cached_image_path.name
        label_path = labels_dir / f"{cached_image_path.stem}.txt"
        ann_path = annotations_dir / f"{cached_image_path.stem}.txt"

        if not ann_path.exists():
            if require_annotations:
                missing_annotations += 1
            label_path.write_text("", encoding="utf-8")
            continue

        width, height = _image_size(source_image_path)
        yolo_lines: list[str] = []
        seen_rows: set[tuple[str, ...]] = set()

        for raw in ann_path.read_text(encoding="utf-8-sig").splitlines():
            raw = raw.strip().strip(",")
            if not raw:
                continue

            parts = tuple(p.strip() for p in raw.split(","))
            if parts in seen_rows:
                duplicate_rows += 1
                continue
            seen_rows.add(parts)

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

    # Remove stale Ultralytics caches so label discovery is rebuilt after conversion.
    for cache in (out_split / "labels.cache", out_images.parent / "labels.cache"):
        if cache.exists():
            cache.unlink()

    if require_annotations and missing_annotations:
        raise RuntimeError(
            f"{missing_annotations} images in {images_dir} have no matching annotation file"
        )

    return {
        "images": len(image_paths),
        "objects": converted_objects,
        "ignored": ignored_objects,
        "duplicates_removed": duplicate_rows,
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
            f"  {split}: images={s['images']} objects={s['objects']} "
            f"ignored={s['ignored']} duplicates_removed={s['duplicates_removed']}"
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
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }

    data_yaml.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    return data_yaml
