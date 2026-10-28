#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo 'Creating Python environment...'
  python3 -m venv .venv
fi
if [ ! -f .venv/.basic_installed ]; then
  .venv/bin/python -m pip install -r requirements.txt
  touch .venv/.basic_installed
fi
printf '\nPoseFrame Studio → http://127.0.0.1:8765\n'
printf 'Press Ctrl+C to stop the server.\n\n'
exec .venv/bin/python app.py
