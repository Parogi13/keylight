#!/usr/bin/env bash
# Demo mode: one process on http://localhost:8000 serving the API and the built UI.
#   ./scripts/start.sh          build UI if missing, then run
#   ./scripts/start.sh --dev    run backend + Vite dev server (hot reload on :5173)
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

if [ "${1:-}" = "--dev" ]; then
  (cd backend && python main.py) &
  BACK=$!
  trap 'kill $BACK' EXIT
  cd frontend && npm run dev
else
  [ -d frontend/dist ] || (cd frontend && npm run build --silent)
  echo "Keylight running at http://localhost:8000"
  (sleep 2 && open http://localhost:8000 2>/dev/null || true) &
  cd backend && exec python main.py
fi
