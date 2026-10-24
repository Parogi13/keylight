"""Pass 3b — Render: decode once, write enhanced + attention videos, publish original.

Latency design
- Frames are enhanced on a small thread pool (OpenCV/NumPy release the GIL), with
  a bounded in-flight window so memory stays flat regardless of video length.
  Output order is preserved; ffmpeg encoders run as separate processes.
- `original.mp4` is re-encoded from the same decoded frames with the same encoder
  settings as `enhanced.mp4`, so unchanged regions are identical in both.
- The attention view is a debug overlay, so it is rendered at ≤960 px; the player
  scales it into the same box.
"""
from __future__ import annotations

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import video_io
from .attention import Attention
from . import hdr
from .enhance import Params, attention_view, enhance, enhance_hdr, hdr_gain

ATTENTION_LONG_SIDE = 960


def _workers() -> int:
    env = os.environ.get("KEYLIGHT_RENDER_THREADS")
    if env:
        return max(1, int(env))
    # Leave cores for the decoder and the two x264 processes.
    return max(1, min(6, (os.cpu_count() or 2) - 2))


def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def render(working: Path, info: video_io.VideoInfo, att: Attention, params: Params, out_dir: Path,
           progress: Callable[[str, float], None] | None = None, write_original: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    W, H, fr = info.width, info.height, info.fps_fraction
    scale = min(1.0, ATTENTION_LONG_SIDE / max(W, H))
    aw, ah = (_even(W * scale), _even(H * scale)) if scale < 1 else (W, H)
    audio = working if info.has_audio else None

    p = att.p
    # Fair comparison: the original is re-encoded from the *same decoded frames* with
    # the *same encoder settings* as the enhanced video. Wherever Keylight changes
    # nothing, both files are identical; any visible difference is our effect, not
    # one side having been compressed one more time than the other. The HDR pair
    # follows the same rule: both go through the identical SDR->HDR conversion and
    # HEVC encode; the baseline simply gets zero subject lift.
    write_original = write_original or not (out_dir / "original.mp4").exists()
    want_hdr = params.hdr and hdr.available()
    hdr_error = None if want_hdr or not params.hdr else "no HEVC encoder in this ffmpeg build"

    writers: dict = {
        "enhanced": video_io.FrameWriter(out_dir / "enhanced.mp4.part", W, H, fr, audio),
        "attention": video_io.FrameWriter(out_dir / "attention.mp4.part", aw, ah, fr, None, crf=23),
    }
    if write_original:
        writers["original"] = video_io.FrameWriter(out_dir / "original.mp4.part", W, H, fr, audio)
    if want_hdr:
        writers["enhanced_hdr"] = hdr.HDRWriter(out_dir / "enhanced_hdr.mp4.part", W, H, fr, audio)
        if write_original or not (out_dir / "original_hdr.mp4").exists():
            writers["original_hdr"] = hdr.HDRWriter(out_dir / "original_hdr.mp4.part", W, H, fr, audio)
    n_total = max(1, min(info.frame_count, len(p.frames)))
    coverage = np.zeros(n_total, np.float32)

    def work(t: int, frame: np.ndarray) -> dict:
        m_low = att.mask(t)
        coverage[t] = float(m_low.mean())
        out = {"enhanced": enhance(frame, m_low, params), "original": frame}
        if want_hdr:
            base48 = hdr.nits_to_rgb48(hdr.sdr_to_nits(frame))
            out["original_hdr"] = base48
            out["enhanced_hdr"] = enhance_hdr(frame, m_low, params, base48)
        small = cv2.resize(frame, (aw, ah), interpolation=cv2.INTER_AREA) if (aw, ah) != (W, H) else frame
        m_small = cv2.resize(m_low, (aw, ah), interpolation=cv2.INTER_LINEAR)
        label, box = None, None
        tid = att.raw[t]
        if tid is not None:
            o = next((o for o in p.frames[t] if o.id == tid), None)
            if o is not None:
                label = f"{o.cls} #{o.id}" if o.id >= 0 else o.cls
                box = tuple(v * (aw / W) for v in o.box)
        out["attention"] = attention_view(small, m_small, label, box)
        return out

    workers = _workers()
    window: deque = deque()
    written = 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            def drain(limit: int):
                nonlocal written
                while len(window) > limit:
                    frames = window.popleft().result()
                    for name, wr in writers.items():
                        wr.write(frames[name])
                    written += 1
                    if progress and written % 5 == 0:
                        progress("render", written / n_total)

            for t, frame in enumerate(video_io.iter_frames(working)):
                if t >= n_total:
                    break
                window.append(pool.submit(work, t, frame))
                drain(2 * workers)  # bounded in-flight frames => flat memory
            drain(0)
        for wr in writers.values():
            wr.close()
    except Exception:
        for wr in writers.values():
            try:
                wr.proc.kill()
            except Exception:
                pass
        raise

    for name in writers:
        (out_dir / f"{name}.mp4.part").replace(out_dir / f"{name}.mp4")
    return {"mean_mask_coverage": round(float(coverage[:written].mean()) if written else 0.0, 4),
            "frames": written, "threads": workers,
            "hdr": {"enabled": want_hdr, "encoder": hdr.pick_encoder() if want_hdr else None,
                    "ref_white_nits": hdr.REF_WHITE, "subject_gain": round(hdr_gain(params.strength), 2),
                    "error": hdr_error}}
