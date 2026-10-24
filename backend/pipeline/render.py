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
import subprocess
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import video_io
from .attention import Attention
from .enhance import Params, enhance, attention_view

ATTENTION_LONG_SIDE = 960


def _workers() -> int:
    env = os.environ.get("KEYLIGHT_RENDER_THREADS")
    if env:
        return max(1, int(env))
    # Leave cores for the decoder and the two x264 processes.
    return max(1, min(6, (os.cpu_count() or 2) - 2))


def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def publish_original(working: Path, dst: Path) -> None:
    tmp = dst.with_suffix(".mp4.part")
    subprocess.run([video_io.FFMPEG, "-v", "error", "-y", "-i", str(working), "-map", "0", "-c", "copy",
                    "-movflags", "+faststart", "-f", "mp4", str(tmp)], check=True, capture_output=True)
    tmp.replace(dst)


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
    # one side having been compressed one more time than the other.
    copy_original = False

    enc = video_io.FrameWriter(out_dir / "enhanced.mp4.part", W, H, fr, audio)
    orig = (video_io.FrameWriter(out_dir / "original.mp4.part", W, H, fr, audio)
            if write_original and not copy_original else None)
    attw = video_io.FrameWriter(out_dir / "attention.mp4.part", aw, ah, fr, None, crf=23)
    n_total = max(1, min(info.frame_count, len(p.frames)))
    coverage = np.zeros(n_total, np.float32)

    def work(t: int, frame: np.ndarray):
        m_low = att.mask(t)
        coverage[t] = float(m_low.mean())
        out = enhance(frame, m_low, params)
        small = cv2.resize(frame, (aw, ah), interpolation=cv2.INTER_AREA) if (aw, ah) != (W, H) else frame
        m_small = cv2.resize(m_low, (aw, ah), interpolation=cv2.INTER_LINEAR)
        label, box = None, None
        tid = att.raw[t]
        if tid is not None:
            o = next((o for o in p.frames[t] if o.id == tid), None)
            if o is not None:
                label = f"{o.cls} #{o.id}" if o.id >= 0 else o.cls
                box = tuple(v * (aw / W) for v in o.box)
        return out, attention_view(small, m_small, label, box), frame

    workers = _workers()
    window: deque = deque()
    written = 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            def drain(limit: int):
                nonlocal written
                while len(window) > limit:
                    out, av, src = window.popleft().result()
                    enc.write(out)
                    attw.write(av)
                    if orig is not None:
                        orig.write(src)
                    written += 1
                    if progress and written % 5 == 0:
                        progress("render", written / n_total)

            for t, frame in enumerate(video_io.iter_frames(working)):
                if t >= n_total:
                    break
                window.append(pool.submit(work, t, frame))
                drain(2 * workers)  # bounded in-flight frames => flat memory
            drain(0)
        for w in (enc, attw, orig):
            if w is not None:
                w.close()
    except Exception:
        for w in (enc, attw, orig):
            if w is None:
                continue
            try:
                w.proc.kill()
            except Exception:
                pass
        raise

    (out_dir / "enhanced.mp4.part").replace(out_dir / "enhanced.mp4")
    (out_dir / "attention.mp4.part").replace(out_dir / "attention.mp4")
    if orig is not None:
        (out_dir / "original.mp4.part").replace(out_dir / "original.mp4")
    return {"mean_mask_coverage": round(float(coverage[:written].mean()) if written else 0.0, 4),
            "frames": written, "threads": workers, "original": "stream copy" if copy_original else "re-encoded"}
