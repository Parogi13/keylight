"""Export a processed job for the native Metal player.

    python scripts/export_native.py backend/outputs/<job_id> [still_seconds]
    python scripts/export_native.py bench/<clip>/<backend>
    -> <folder>/native/{video.mp4, masks.json, masks.bin, still.png, still_mask.png}

Web-app jobs are exported automatically when they finish (pipeline/native_export.py);
this script is for older jobs and benchmark folders. The masks include Claude's subject
choice for the job's focus instruction when it is cached in semantic.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from pipeline import native_export  # noqa: E402


def main(job: Path, still_at: float | None = None) -> int:
    out = native_export.export(job, still_at=still_at)
    print(f"exported -> {out}")
    print(f"run:  cd native/KeylightPlayer && swift run -c release KeylightPlayer {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), float(sys.argv[2]) if len(sys.argv) > 2 else None))
