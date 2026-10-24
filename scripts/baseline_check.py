"""Does the enhanced side match the original wherever we didn't mean to change it?

    python scripts/baseline_check.py clip.mp4            # effects off: whole frame
    python scripts/baseline_check.py clip.mp4 --effects  # default effects: background only

Decodes original.mp4 and enhanced.mp4 the way a browser does (honouring their
colour tags) and reports PSNR and the worst per-pixel error. With effects off,
anything above ~40 dB is visually identical (the remaining difference is normal
lossy-encoding noise). With effects on, only pixels with no subject nearby are
compared, which checks that the background really is left alone.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from pipeline import attention, perceive, run as pipeline, video_io  # noqa: E402


def browser_frames(path: Path, w: int, h: int):
    """Decode like a browser: use the file's colour tags (BT.709 when unlabeled)."""
    proc = subprocess.Popen([video_io.FFMPEG, "-v", "error", "-i", str(path), "-vf",
                             "scale=in_color_matrix=auto:in_range=auto,format=bgr24",
                             "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    n = w * h * 3
    while True:
        buf = proc.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    proc.wait()


def psnr(mse: float) -> float:
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--effects", action="store_true", help="default effects; compare background only")
    a = ap.parse_args()
    job = Path(tempfile.mkdtemp(prefix="keylight_baseline_"))
    settings = {"strength": 0.6, "background": 0.15, "glow": 0.35, "smoothing": 0.6} if a.effects else \
        {"strength": 0.0, "background": 0.0, "glow": 0.0, "smoothing": 0.6}
    pipeline.process(job, Path(a.clip), {**settings, "focus": "", "use_claude": False}, lambda *_: None)
    info = video_io.info(job / "enhanced.mp4")
    att = attention.build(perceive.Perception.load(job), smoothing=0.6) if a.effects else None

    import cv2
    sq, count, worst, frames = 0.0, 0, 0, 0
    for t, (o, e) in enumerate(zip(browser_frames(job / "original.mp4", info.width, info.height),
                                   browser_frames(job / "enhanced.mp4", info.width, info.height))):
        d = o.astype(np.int16) - e.astype(np.int16)
        if att is not None:
            m = cv2.resize(att.mask(t), (info.width, info.height))
            # Background = no subject within the feathered edge, plus a safety margin.
            far = cv2.erode((m < 1e-3).astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
            # The dim is intentional when a subject is present; only count frames with none.
            if float(m.max()) > 1e-3:
                continue
            d = d[far]
        if d.size == 0:
            continue
        sq += float((d.astype(np.float64) ** 2).sum())
        count += d.size
        worst = max(worst, int(np.abs(d).max()))
        frames += 1
    if count == 0:
        print("no comparable pixels (every frame had a subject)")
        return 0
    p = psnr(sq / count)
    print(f"frames compared {frames}  PSNR {p:.1f} dB  worst pixel error {worst}/255")
    ok = p >= 40
    print("PASS (visually identical)" if ok else "FAIL (visible difference)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
