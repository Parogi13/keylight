"""Video I/O: probe, normalize to a constant-frame-rate working copy, decode, encode.

Design notes
- Every input is first normalized with ffmpeg into `working.mp4`: constant frame
  rate, H.264 at near-lossless quality, height capped at 1080, AAC audio. Phone
  videos are often variable-frame-rate, which is the classic cause of audio drift;
  normalizing once means everything downstream can assume frame i is at i / fps.
- We never touch audio samples ourselves. Encoders mux the audio stream from the
  working copy back in, so sync is guaranteed as long as frame counts match.
- The browser-facing `original.mp4` and `enhanced.mp4` are produced by the same
  encoder settings from the same decoded frames, so the wipe comparison is fair.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, asdict
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import av
import numpy as np

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"

MAX_HEIGHT = 1080
MAX_FPS = 60


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    fps_fraction: str  # e.g. "30000/1001"
    frame_count: int
    duration: float
    has_audio: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{proc.stderr[-2000:]}")
    return proc


def probe(path: str | Path) -> dict:
    out = _run([FFPROBE, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)])
    return json.loads(out.stdout)


def _choose_fps(stream: dict) -> Fraction:
    """Pick a constant output frame rate. Prefer the nominal rate; fall back to the
    average rate (VFR phone footage), then to 30."""
    for key in ("r_frame_rate", "avg_frame_rate"):
        val = stream.get(key)
        if val and val != "0/0":
            f = Fraction(val)
            if 1 <= f <= MAX_FPS:
                return f.limit_denominator(1001)
    return Fraction(30, 1)


def normalize(src: str | Path, dst: str | Path, max_height: int = MAX_HEIGHT) -> VideoInfo:
    """Create a constant-frame-rate, browser-safe, near-lossless working copy.

    The long side is capped so that the *shorter* side is at most `max_height`
    (portrait 4K -> 1080x1920, landscape 4K -> 1920x1080).
    """
    meta = probe(src)
    vstreams = [s for s in meta["streams"] if s.get("codec_type") == "video"]
    if not vstreams:
        raise ValueError("input has no video stream")
    v = vstreams[0]
    has_audio = any(s.get("codec_type") == "audio" for s in meta["streams"])
    fps = _choose_fps(v)

    m = max_height
    # Scale so min(w, h) <= m, keep aspect, force even dimensions (yuv420p).
    vf = (
        f"fps={fps.numerator}/{fps.denominator},"
        f"scale='trunc(iw*min(1,{m}/min(iw,ih))/2)*2':'trunc(ih*min(1,{m}/min(iw,ih))/2)*2',"
        "setsar=1,format=yuv420p"
    )
    cmd = [FFMPEG, "-v", "error", "-y", "-i", str(src), "-map", "0:v:0"]
    if has_audio:
        cmd += ["-map", "0:a:0"]
    cmd += ["-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "12"]
    if has_audio:
        # aresample=async keeps audio aligned to timestamps if the source drifts.
        cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-af", "aresample=async=1:first_pts=0"]
    cmd += ["-movflags", "+faststart", str(dst)]
    _run(cmd)
    return info(dst)


def info(path: str | Path) -> VideoInfo:
    meta = probe(path)
    v = next(s for s in meta["streams"] if s.get("codec_type") == "video")
    has_audio = any(s.get("codec_type") == "audio" for s in meta["streams"])
    fps = Fraction(v["r_frame_rate"])
    with av.open(str(path)) as c:
        n = c.streams.video[0].frames
    if not n:
        n = sum(1 for _ in iter_frames(path))
    duration = float(meta["format"].get("duration", n / float(fps)))
    return VideoInfo(
        width=int(v["width"]),
        height=int(v["height"]),
        fps=float(fps),
        fps_fraction=f"{fps.numerator}/{fps.denominator}",
        frame_count=int(n),
        duration=duration,
        has_audio=has_audio,
    )


def iter_frames(path: str | Path) -> Iterator[np.ndarray]:
    """Yield frames as BGR uint8 arrays (OpenCV convention)."""
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in c.decode(stream):
            yield frame.to_ndarray(format="bgr24")


class FrameWriter:
    """Pipe raw BGR frames into ffmpeg -> H.264 mp4, optionally muxing audio from
    `audio_from` (copied, never re-timed)."""

    def __init__(self, dst: str | Path, width: int, height: int, fps_fraction: str,
                 audio_from: str | Path | None = None, crf: int = 18):
        cmd = [
            FFMPEG, "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
            "-framerate", fps_fraction, "-i", "-",
        ]
        if audio_from is not None:
            cmd += ["-i", str(audio_from), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy"]
        cmd += [
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", str(dst),
        ]
        self.dst = Path(dst)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.count = 0

    def write(self, frame: np.ndarray) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        self.count += 1

    def close(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode() if self.proc.stderr else ""
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg encode failed for {self.dst}: {err[-2000:]}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self.close()
        else:
            try:
                self.proc.kill()
            except Exception:
                pass
