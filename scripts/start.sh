#!/usr/bin/env bash
# Demo mode: one process on http://localhost:8000 serving the API and the built UI.
#   ./scripts/start.sh          build UI if missing, then run
#   ./scripts/start.sh --dev    run backend + Vite dev server (hot reload on :5173)
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

if [ "${1:-}" = "--dev" ]; then
  [ -x frontend/node_modules/.bin/vite ] || (cd frontend && npm install)
  (cd backend && python main.py) &
  BACK=$!
  trap 'kill $BACK' EXIT
  cd frontend && npm run dev
else
  if [ ! -d frontend/dist ]; then
    # Install UI dependencies if setup didn't (e.g. npm was missing at the time).
    [ -x frontend/node_modules/.bin/tsc ] || (cd frontend && npm install)
    (cd frontend && npm run build)
  fi
  echo "Starting Keylight (first start can take ~20 s while PyTorch loads)…"
  # Open the browser only once the server answers, not after a fixed delay.
  (for _ in $(seq 1 120); do
     if curl -sf http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
       echo "Keylight running at http://localhost:8000"
       open http://localhost:8000 2>/dev/null || true
       break
     fi
     sleep 1
   done) &
  cd backend && exec python main.py
fi
