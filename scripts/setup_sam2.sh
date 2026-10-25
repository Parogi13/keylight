#!/usr/bin/env bash
# Experimental: install SAM 2 (official facebookresearch/sam2) in an ISOLATED venv
# (.venv-sam2) so the main .venv and the working app are untouched.
#   ./scripts/setup_sam2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3.12}
[ -d .venv-sam2 ] || "$PY" -m venv .venv-sam2
source .venv-sam2/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
# CPU/MPS build: skip the optional CUDA extension (not usable on Apple Silicon).
SAM2_BUILD_CUDA=0 pip install -q "git+https://github.com/facebookresearch/sam2.git"
mkdir -p backend/models
CKPT=backend/models/sam2.1_hiera_tiny.pt
if [ ! -f "$CKPT" ]; then
  echo "Downloading SAM 2.1 Hiera Tiny (official Meta checkpoint)…"
  if ! curl -fL -o "$CKPT" https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt; then
    echo "Meta CDN unreachable; using the Ultralytics-hosted copy (identical weights, re-keyed)"
    curl -fL -o /tmp/sam2.1_t.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_t.pt
    python -c "import torch; torch.save({'model': torch.load('/tmp/sam2.1_t.pt', map_location='cpu', weights_only=False)['model']}, '$CKPT')"
  fi
fi
[ -f backend/models/yolo11n-seg.pt ] || (cd backend/models && python -c "from ultralytics import YOLO; YOLO('yolo11n-seg.pt')")
python - <<'PY'
import torch, sam2
print("torch", torch.__version__, "| MPS available:", torch.backends.mps.is_available())
from sam2.build_sam import build_sam2_video_predictor
dev = "mps" if torch.backends.mps.is_available() else "cpu"
build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_t.yaml", "backend/models/sam2.1_hiera_tiny.pt", device=dev)
print("SAM 2.1 tiny loads on", dev)
PY
echo "Done. Benchmark:  source .venv-sam2/bin/activate && python scripts/bench_segmentation.py samples/<clip>.mp4"
