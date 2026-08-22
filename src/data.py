from __future__ import annotations

from pathlib import Path

import yaml

from .config import CLASS_NAMES


def build_data_yaml(cfg: dict) -> Path:
    root = Path(cfg["dataset_root"])
    generated = Path(cfg["generated_dir"])
    generated.mkdir(parents=True, exist_ok=True)

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
        "train": cfg["train_images"],
        "val": cfg["val_images"],
        "test": cfg["test_images"],
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
