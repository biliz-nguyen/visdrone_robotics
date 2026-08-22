#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CMD="${1:-help}"

case "$CMD" in
  setup)
    bash setup_local.sh
    ;;
  prepare)
    python3 scripts/prepare.py
    ;;
  sanity)
    python3 scripts/sanity.py
    ;;
  train)
    python3 scripts/train.py
    ;;
  validate)
    shift
    python3 scripts/validate.py "$@"
    ;;
  infer)
    shift
    python3 scripts/infer.py "$@"
    ;;
  export-ncnn)
    python3 scripts/export.py --format ncnn
    ;;
  export-onnx)
    python3 scripts/export.py --format onnx
    ;;
  export-all)
    python3 scripts/export.py --format ncnn onnx
    ;;
  *)
    echo "Usage:"
    echo "  ./run.sh setup"
    echo "  ./run.sh sanity"
    echo "  ./run.sh train"
    echo "  ./run.sh validate [--weights /path/best.pt]"
    echo "  ./run.sh infer [--source /path/image.jpg]"
    echo "  ./run.sh export-ncnn"
    echo "  ./run.sh export-onnx"
    echo "  ./run.sh export-all"
    ;;
esac
