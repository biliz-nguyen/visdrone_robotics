#!/usr/bin/env python3

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config


def main():
    cfg = load_config()

    repo = Path(cfg["ultra_repo"])
    tag = cfg["ultralytics_tag"]

    repo.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not (repo / ".git").exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                tag,
                "https://github.com/ultralytics/ultralytics.git",
                str(repo),
            ],
            check=True,
        )

    subprocess.run(
        ["git", "-C", str(repo), "checkout", "--force", tag],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "reset", "--hard", tag],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            str(repo),
        ],
        check=True,
    )

    actual_tag = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "describe",
            "--tags",
            "--exact-match",
        ],
        text=True,
    ).strip()

    assert actual_tag == tag, (
        actual_tag,
        tag,
    )

    print("✅ Ultralytics:", actual_tag)
    print("✅ Repo:", repo)
    print("Next: python scripts/sanity.py")


if __name__ == "__main__":
    main()
