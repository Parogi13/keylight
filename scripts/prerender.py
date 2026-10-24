"""Pre-render demo clips so judging never waits on processing.

    python scripts/prerender.py                  # every video in samples/
    python scripts/prerender.py my_clip.mp4 ...  # specific files
    python scripts/prerender.py --strength 0.8 --background 0.45 clip.mp4

Results land in backend/outputs/<id>/ and appear under "Open a finished video"
on the home screen the next time the backend starts.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from main import DEFAULT_SETTINGS, OUTPUTS, SAMPLES, VIDEO_EXT  # noqa: E402  (also loads .env)
from pipeline import run as pipeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    for k in ("strength", "background", "glow", "smoothing"):
        ap.add_argument(f"--{k}", type=float, default=DEFAULT_SETTINGS[k])
    ap.add_argument("--focus", default="")
    ap.add_argument("--no-claude", action="store_true")
    a = ap.parse_args()

    files = [Path(f) for f in a.files] or sorted(p for p in SAMPLES.iterdir() if p.suffix.lower() in VIDEO_EXT)
    if not files:
        print("No videos given and samples/ is empty.")
        return 1
    settings = {**DEFAULT_SETTINGS, "strength": a.strength, "background": a.background, "glow": a.glow,
                "smoothing": a.smoothing, "focus": a.focus, "use_claude": not a.no_claude}

    for src in files:
        jid = uuid.uuid4().hex[:10]
        job = OUTPUTS / jid
        job.mkdir(parents=True)
        dst = job / f"input{src.suffix.lower()}"
        shutil.copyfile(src, dst)
        (job / "job.json").write_text(json.dumps({"name": src.name, "settings": settings, "created": time.time()}))
        last = [-1]

        def cb(stage, frac, label):
            pct = int(frac * 100)
            if pct // 5 != last[0] // 5:
                print(f"  {src.name}: {label} {pct}%", flush=True)
                last[0] = pct

        t = time.time()
        meta = pipeline.process(job, dst, settings, cb)
        print(f"done {src.name} in {time.time() - t:.0f}s -> {job}  "
              f"(device {meta['device']}, {meta['realtime_factor']}x realtime, Claude: {meta['semantic'].get('ok', 0)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
