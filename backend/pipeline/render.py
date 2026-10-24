"""Pass 3b — Render: decode once, write original / enhanced / attention videos.

All three outputs come from the same decoded frames and the same encoder settings,
so the wipe comparison is apples to apples. Audio is copied from the working copy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import video_io
from .attention import Attention
from .enhance import Params, enhance, attention_view


def render(working: Path, info: video_io.VideoInfo, att: Attention, params: Params, out_dir: Path,
           progress: Callable[[str, float], None] | None = None, write_original: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    W, H, fr = info.width, info.height, info.fps_fraction
    audio = working if info.has_audio else None
    writers = {
        "enhanced": video_io.FrameWriter(out_dir / "enhanced.mp4.part", W, H, fr, audio),
        "attention": video_io.FrameWriter(out_dir / "attention.mp4.part", W, H, fr, None, crf=23),
    }
    if write_original:
        writers["original"] = video_io.FrameWriter(out_dir / "original.mp4.part", W, H, fr, audio)

    p = att.p
    coverage = []
    try:
        for t, frame in enumerate(video_io.iter_frames(working)):
            if t >= len(p.frames):
                break
            m_low = att.mask(t)
            m = cv2.resize(m_low, (W, H), interpolation=cv2.INTER_LINEAR)
            coverage.append(float(m_low.mean()))
            writers["enhanced"].write(enhance(frame, m, params))
            if "original" in writers:
                writers["original"].write(frame)

            label, box = None, None
            tid = att.raw[t]
            if tid is not None:
                o = next((o for o in p.frames[t] if o.id == tid), None)
                if o is not None:
                    label, box = (f"{o.cls} #{o.id}" if o.id >= 0 else o.cls), o.box
            writers["attention"].write(attention_view(frame, m, label, box))
            if progress and t % 5 == 0:
                progress("render", (t + 1) / max(1, info.frame_count))
        for w in writers.values():
            w.close()
    except Exception:
        for w in writers.values():
            try:
                w.proc.kill()
            except Exception:
                pass
        raise

    for name in writers:
        (out_dir / f"{name}.mp4.part").replace(out_dir / f"{name}.mp4")
    return {"mean_mask_coverage": round(float(np.mean(coverage)) if coverage else 0.0, 4)}
