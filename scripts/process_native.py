"""Process one video for the native Keylight app: full pipeline, then the native export.

    python scripts/process_native.py clip.mp4 [--backend yolo|sam2] [--focus "the cyclist"] [--name "Shown name"]

Creates a job exactly like the web app does (backend/outputs/<id>/ with job.json), so the
result also shows up in the web app. The pipeline writes <job>/native/ for the Metal player
when it finishes (pipeline/native_export.py); this script only exports again if that was skipped.

Machine-readable progress on stdout, one line each (everything else goes to stderr):
    PROGRESS {"stage": "perceive", "progress": 0.42, "label": "Detecting subjects"}
    DONE {"job": "/abs/path/to/job"}
    ERROR {"message": "..."}
A failed or cancelled (SIGTERM) run removes its half-built job folder.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import time
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))


def _load_dotenv() -> None:
    """Same as backend/main.py: ANTHROPIC_API_KEY etc. from .env, never overriding the shell."""
    for p in (ROOT / "backend" / ".env", ROOT / ".env"):
        if p.exists():
            for line in p.read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# The web app's DEFAULT_SETTINGS (backend/main.py), minus the encoded HDR copy: the native
# player renders EDR live, so that encode would only cost time.
SETTINGS = {"strength": 0.6, "background": 0.15, "glow": 0.35, "smoothing": 0.6, "hdr": False,
            "focus": "", "use_claude": True, "segmentation_backend": "yolo", "yolo_interval_s": 1.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--backend", choices=["yolo", "sam2"], default="yolo")
    ap.add_argument("--focus", default="", help="optional instruction for Claude, e.g. 'the person on the left'")
    ap.add_argument("--name")
    a = ap.parse_args()

    _load_dotenv()
    # Pipeline chatter must not mix with the PROGRESS protocol on stdout.
    real_stdout, sys.stdout = sys.stdout, sys.stderr
    from pipeline import run as pipeline  # noqa: E402  (after dotenv, like the server)
    import export_native  # noqa: E402

    def out(kind: str, payload: dict) -> None:
        real_stdout.write(f"{kind} {json.dumps(payload)}\n")
        real_stdout.flush()

    src = Path(a.src).expanduser().resolve()
    if not src.is_file():
        out("ERROR", {"message": f"File not found: {src.name}"})
        return 1
    outputs = Path(os.environ.get("KEYLIGHT_OUTPUTS", ROOT / "backend" / "outputs"))
    outputs.mkdir(parents=True, exist_ok=True)
    job = outputs / uuid.uuid4().hex[:10]
    job.mkdir()
    settings = {**SETTINGS, "segmentation_backend": a.backend, "focus": a.focus.strip()}
    (job / "job.json").write_text(json.dumps({"name": a.name or src.name, "settings": settings, "created": time.time()}))

    # Cancel from the app = SIGTERM: unwind through `finally` so the folder is cleaned up.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(130))
    ok = False
    try:
        dst = job / f"input{src.suffix.lower() or '.mp4'}"
        shutil.copyfile(src, dst)
        pipeline.process(job, dst, settings,
                         lambda stage, frac, label: out("PROGRESS", {"stage": stage, "progress": round(frac * 0.97, 4), "label": label}))
        if not (job / "native" / "video.mp4").exists():  # the pipeline's own export was skipped
            out("PROGRESS", {"stage": "export", "progress": 0.98, "label": "Preparing for the player"})
            export_native.main(job)
        ok = True
        out("DONE", {"job": str(job)})
        return 0
    except SystemExit:
        out("ERROR", {"message": "Cancelled"})
        return 130
    except Exception as e:
        traceback.print_exc()
        # ValueErrors from the pipeline are written for people; others aren't (as in the web app).
        msg = str(e) if isinstance(e, ValueError) else f"Something went wrong while processing ({type(e).__name__}: {e})"
        out("ERROR", {"message": msg[:500]})
        return 1
    finally:
        if not ok and job.resolve().parent == outputs.resolve():  # never delete outside outputs/
            shutil.rmtree(job, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
