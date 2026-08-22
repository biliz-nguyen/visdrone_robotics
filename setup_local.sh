#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ULTRA_DIR="$ROOT_DIR/third_party/ultralytics"
ULTRA_TAG="v8.4.56"

echo "============================================================"
echo "VisDrone local setup"
echo "Python: $($PYTHON_BIN --version)"
echo "Project: $ROOT_DIR"
echo "============================================================"

mkdir -p third_party generated runs state outputs

$PYTHON_BIN -m pip install --upgrade pip
$PYTHON_BIN -m pip install -r requirements.txt

if [ ! -d "$ULTRA_DIR/.git" ]; then
    git clone \
        --depth 1 \
        --branch "$ULTRA_TAG" \
        https://github.com/ultralytics/ultralytics.git \
        "$ULTRA_DIR"
fi

git -C "$ULTRA_DIR" fetch --tags --force
git -C "$ULTRA_DIR" checkout --force "$ULTRA_TAG"
git -C "$ULTRA_DIR" reset --hard "$ULTRA_TAG"

$PYTHON_BIN -m pip install -e "$ULTRA_DIR"

echo
echo "Ultralytics tag:"
git -C "$ULTRA_DIR" describe --tags --exact-match

echo
echo "============================================================"
echo "SETUP DONE"
echo "Next:"
echo "  cp config/local.example.yaml config/local.yaml"
echo "  # edit dataset_root in config/local.yaml"
echo "  $PYTHON_BIN scripts/sanity.py"
echo "============================================================"
