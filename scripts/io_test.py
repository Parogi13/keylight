"""Phase 1 check: normalize -> decode -> re-encode -> remux audio -> verify.

Usage: python scripts/io_test.py path/to/video.mp4
Prints source vs output frame count / fps / duration, and the audio/video
duration gap, which should be under one frame.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from pipeline import video_io  # noqa: E402


def stream_durations(path: Path) -> dict:
    meta = video_io.probe(path)
    return {s["codec_type"]: float(s.get("duration", "nan")) for s in meta["streams"]}


def main(src: str) -> int:
    out_dir = Path(tempfile.mkdtemp(prefix="keylight_io_"))
    work = out_dir / "working.mp4"
    info = video_io.normalize(src, work)
    print("working copy:", info)

    out = out_dir / "roundtrip.mp4"
    with video_io.FrameWriter(out, info.width, info.height, info.fps_fraction, audio_from=work) as w:
        for frame in video_io.iter_frames(work):
            w.write(frame)
    rt = video_io.info(out)
    print("round trip:  ", rt)

    ok = True
    if rt.frame_count != info.frame_count:
        print(f"FAIL frame count {rt.frame_count} != {info.frame_count}")
        ok = False
    if abs(rt.fps - info.fps) > 1e-6:
        print(f"FAIL fps {rt.fps} != {info.fps}")
        ok = False
    if rt.has_audio:
        d = stream_durations(out)
        gap = abs(d.get("audio", 0) - d.get("video", 0))
        print(f"audio/video duration gap: {gap * 1000:.1f} ms (one frame = {1000 / info.fps:.1f} ms)")
        if gap > 2 / info.fps + 0.05:
            print("FAIL audio/video duration mismatch")
            ok = False
    print("PASS" if ok else "FAIL", "->", out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
