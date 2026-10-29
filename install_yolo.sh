#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo 'Run ./run.sh once before installing YOLO.'
  exit 1
fi
echo 'Installing Ultralytics and PyTorch (large download)...'
.venv/bin/python install_pytorch.py "$@"
echo 'Done. Restart the web server to use YOLO.'
