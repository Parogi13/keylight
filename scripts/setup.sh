#!/usr/bin/env bash
# One-time setup on macOS (Apple Silicon). Run from the repo root: ./scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

command -v ffmpeg >/dev/null || { echo "Installing ffmpeg (Homebrew)…"; brew install ffmpeg; }
command -v node >/dev/null || { echo "Installing node (Homebrew)…"; brew install node; }

PY=${PYTHON:-python3}
if [ ! -d .venv ]; then "$PY" -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Fetch the segmentation weights once (about 6 MB) so the demo works offline.
mkdir -p backend/models
python - <<'EOF'
from pathlib import Path
import os, shutil
os.chdir("backend/models")
from ultralytics import YOLO
YOLO("yolo11n-seg.pt")
print("model ready:", Path("yolo11n-seg.pt").resolve())
EOF
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

(cd frontend && npm install --silent && npm run build --silent)
[ -f .env ] || cp .env.example .env
echo
echo "Setup done. Add your ANTHROPIC_API_KEY to .env (optional), then run ./scripts/start.sh"
