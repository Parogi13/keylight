"""Export a processed job for the native Metal EDR player (native/KeylightPlayer).

    job/native/video.mp4         the SDR original (hard link); the enhancement happens live on the GPU
    job/native/masks.json        {width, height, count, fps}
    job/native/masks.bin         count * width * height uint8: the temporally smoothed,
                                 feathered attention mask per frame, exactly what the
                                 Python renderer uses (including Claude's subject choice)
    job/native/still.png         one frame + its full-resolution mask, for the static EDR test
    job/native/still_mask.png

The web pipeline calls export() after every finished job, so the native library and its
"Add video" button see new videos without a manual step. scripts/export_native.py is the
command-line wrapper (also used for benchmark folders).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from . import attention, perceive, video_io


def _semantic_for(job: Path, settings: dict) -> dict:
    """Claude's cached per-shot choices for the job's focus instruction, if any."""
    if not settings.get("use_claude", True):
        return {}
    cache = job / "semantic.json"
    if not cache.exists():
        return {}
    entry = json.loads(cache.read_text()).get((settings.get("focus") or "").strip().lower())
    return {int(k): v for k, v in entry["decisions"].items()} if entry else {}


def export(job: Path, att: "attention.Attention | None" = None, still_at: float | None = None,
           p: "perceive.Perception | None" = None) -> Path:
    out = job / "native"
    out.mkdir(exist_ok=True)
    meta_path = job / "metadata.json"
    settings = json.loads(meta_path.read_text()).get("settings", {}) if meta_path.exists() else {}
    # Web-app jobs keep working.mp4 in the job folder; benchmark folders (bench/<clip>/<backend>)
    # keep it one level up.
    working = job / "working.mp4" if (job / "working.mp4").exists() else job.parent / "working.mp4"
    if p is None:
        p = perceive.Perception.load(job)
    if att is None:
        att = attention.build(p, smoothing=float(settings.get("smoothing", 0.6)),
                              semantic=_semantic_for(job, settings) or None,
                              companions=not (settings.get("focus") or "").strip())
    n = len(p.frames)

    tmp = out / "masks.bin.part"
    covs = np.zeros(n)
    with open(tmp, "wb") as f:
        for t in range(n):
            m = np.clip(att.mask(t), 0, 1)
            covs[t] = float(m.mean())
            f.write((m * 255 + 0.5).astype(np.uint8).tobytes())
    tmp.replace(out / "masks.bin")
    (out / "masks.json").write_text(json.dumps({"width": p.mask_w, "height": p.mask_h, "count": n,
                                                "fps": p.fps, "source": job.name,
                                                "focus": settings.get("focus", "")}))

    src = job / "original.mp4" if (job / "original.mp4").exists() else working
    dst = out / "video.mp4"
    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)

    t_still = int(round(still_at * p.fps)) if still_at is not None else int(np.argmax(covs))
    for t, frame in enumerate(video_io.iter_frames(working)):
        if t == t_still:
            cv2.imwrite(str(out / "still.png"), frame)
            m = cv2.resize(att.mask(t), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(out / "still_mask.png"), (m * 255 + 0.5).astype(np.uint8))
            break
    return out
