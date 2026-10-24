"""Export a processed job for the native Metal player.

    python scripts/export_native.py backend/outputs/<job_id>
    -> backend/outputs/<job_id>/native/{video.mp4, masks.json, masks.bin, still.png, still_mask.png}

masks.bin holds the *temporally smoothed, feathered* attention mask for every frame
(exactly what the Python renderer uses), as uint8 at the mask-grid resolution. The
native player memory-maps it and uploads one frame per displayed video frame, so
there is no IPC and no mask decoding at playback time.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from pipeline import attention, perceive, video_io  # noqa: E402


def main(job: Path, still_at: float | None = None) -> int:
    out = job / "native"
    out.mkdir(exist_ok=True)
    meta = json.loads((job / "metadata.json").read_text())
    smoothing = float(meta.get("settings", {}).get("smoothing", 0.6))
    p = perceive.Perception.load(job)
    att = attention.build(p, smoothing=smoothing)
    n = len(p.frames)
    with open(out / "masks.bin", "wb") as f:
        for t in range(n):
            f.write((np.clip(att.mask(t), 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
    (out / "masks.json").write_text(json.dumps({"width": p.mask_w, "height": p.mask_h, "count": n,
                                                "fps": p.fps, "source": str(job.name)}))
    src = job / "original.mp4"   # SDR original, BT.709-tagged; the player does the enhancement live
    dst = out / "video.mp4"
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)
    # A still frame + its full-resolution mask for the static EDR test (Stage 5).
    covs = [float(att.mask(t).mean()) for t in range(n)]
    t_still = int(round(still_at * p.fps)) if still_at is not None else int(np.argmax(covs))
    for t, frame in enumerate(video_io.iter_frames(job / "working.mp4")):
        if t == t_still:
            cv2.imwrite(str(out / "still.png"), frame)
            m = cv2.resize(att.mask(t), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(out / "still_mask.png"), (m * 255 + 0.5).astype(np.uint8))
            break
    print(f"exported {n} masks ({p.mask_w}x{p.mask_h}) + video + still (frame {t_still}) -> {out}")
    print(f"run:  cd native/KeylightPlayer && swift run -c release KeylightPlayer {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), float(sys.argv[2]) if len(sys.argv) > 2 else None))
