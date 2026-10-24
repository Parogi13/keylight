"""End-to-end smoke test. Run this first on a new machine:

    python scripts/smoke_test.py [optional/path/to/clip.mp4]

Without a clip it synthesizes a 4 s test video with a moving shape and a beep
track. Checks: all three outputs exist, identical frame counts and fps, audio
present and aligned with video duration, metadata written. Prints timings so
you know how fast this machine is.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from pipeline import run as pipeline, video_io  # noqa: E402


def synth(dst: Path) -> None:
    subprocess.run([
        video_io.FFMPEG, "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=0x101418:s=1280x720:r=30,drawbox=x='200+t*150':y=300:w=180:h=260:color=0x9a7b5a:t=fill",
        "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000",
        "-t", "4", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(dst)], check=True)


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="keylight_smoke_"))
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else work / "synthetic.mp4"
    if len(sys.argv) <= 1:
        synth(src)
    job = work / "job"
    job.mkdir()
    t = time.time()
    meta = pipeline.process(job, src, {"strength": 0.6, "background": 0.35, "glow": 0.35, "smoothing": 0.6,
                                       "focus": "", "use_claude": False},
                            lambda s, f, label: None)
    took = time.time() - t

    ok = True
    infos = {}
    for name in ("original", "enhanced", "attention"):
        f = job / f"{name}.mp4"
        if not f.exists():
            print("FAIL missing", f)
            return 1
        infos[name] = video_io.info(f)
    counts = {k: v.frame_count for k, v in infos.items()}
    if len(set(counts.values())) != 1:
        print("FAIL frame counts differ", counts)
        ok = False
    if infos["enhanced"].has_audio != infos["original"].has_audio:
        print("FAIL audio missing on enhanced")
        ok = False
    if infos["enhanced"].has_audio:
        streams = {s["codec_type"]: float(s["duration"]) for s in video_io.probe(job / "enhanced.mp4")["streams"]}
        gap = abs(streams["audio"] - streams["video"])
        print(f"audio/video gap {gap * 1000:.0f} ms")
        if gap > 0.1:
            ok = False
    print("frames", counts, "device", meta["device"])
    print("timings", meta["timings_s"], f"total {took:.1f}s for {infos['original'].duration:.1f}s of video "
          f"({took / infos['original'].duration:.1f}x realtime)")
    print("PASS" if ok else "FAIL", job)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
