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
import os
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
MAX_DURATION_S = int(os.environ.get("KEYLIGHT_MAX_DURATION_S", 900))


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
    try:
        out = _run([FFPROBE, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)])
    except RuntimeError:
        raise ValueError("This file couldn't be opened. It may be damaged or not a video.") from None
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


def _can_remux(meta: dict, v: dict, max_height: int) -> bool:
    """True if the source can be stream-copied as the working copy."""
    try:
        if v.get("codec_name") != "h264" or v.get("pix_fmt") != "yuv420p":
            return False
        if min(int(v["width"]), int(v["height"])) > max_height or int(v["width"]) % 2 or int(v["height"]) % 2:
            return False
        r, avg = Fraction(v.get("r_frame_rate", "0/1")), Fraction(v.get("avg_frame_rate", "0/1"))
        if r <= 0 or avg <= 0 or abs(float(r - avg)) > 0.01 or r > MAX_FPS:
            return False  # variable frame rate (phones) -> needs resampling
        if any(sd.get("side_data_type", "").lower().startswith("display matrix") for sd in v.get("side_data_list", [])):
            return False  # rotated phone video -> let ffmpeg bake in the rotation
        if (v.get("tags") or {}).get("rotate") not in (None, "0"):
            return False
        if v.get("color_transfer") in ("smpte2084", "arib-std-b67"):
            return False  # HDR -> needs tone mapping
        if abs(float(v.get("start_time", 0) or 0)) > 0.001:
            return False
        sar = v.get("sample_aspect_ratio", "1:1")
        return sar in ("1:1", "0:1", None)
    except Exception:
        return False


def normalize(src: str | Path, dst: str | Path, max_height: int = MAX_HEIGHT) -> VideoInfo:
    """Create a constant-frame-rate, browser-safe, near-lossless working copy.

    The long side is capped so that the *shorter* side is at most `max_height`
    (portrait 4K -> 1080x1920, landscape 4K -> 1920x1080).
    """
    meta = probe(src)
    vstreams = [s for s in meta["streams"] if s.get("codec_type") == "video"]
    if not vstreams:
        raise ValueError("This file has no video track.")
    dur = float(meta.get("format", {}).get("duration") or 0)
    if dur > MAX_DURATION_S:
        raise ValueError(f"This video is {dur / 60:.0f} minutes long. Keylight handles clips up to "
                         f"{MAX_DURATION_S // 60} minutes; trim it first.")
    v = vstreams[0]
    has_audio = any(s.get("codec_type") == "audio" for s in meta["streams"])
    fps = _choose_fps(v)

    if _can_remux(meta, v, max_height):
        # Already constant-frame-rate, browser-safe H.264 at a sane size: copy the
        # video stream (seconds instead of a full transcode). Audio is made AAC.
        cmd = [FFMPEG, "-v", "error", "-y", "-i", str(src), "-map", "0:v:0", "-c:v", "copy"]
        if has_audio:
            a = next(s for s in meta["streams"] if s.get("codec_type") == "audio")
            # Copy AAC as-is: re-encoding would add encoder priming delay (~21 ms).
            acodec = ["-c:a", "copy"] if a.get("codec_name") == "aac" else ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
            cmd += ["-map", "0:a:0", *acodec]
        cmd += ["-movflags", "+faststart", str(dst)]
        try:
            _run(cmd)
            out = info(dst)
            if out.frame_count > 0:
                return out
        except Exception:
            pass  # fall through to the full normalize

    m = max_height
    scale = f"scale='trunc(iw*min(1,{m}/min(iw,ih))/2)*2':'trunc(ih*min(1,{m}/min(iw,ih))/2)*2'"
    base = f"fps={fps.numerator}/{fps.denominator},{scale},setsar=1"
    hdr = v.get("color_transfer") in ("smpte2084", "arib-std-b67")
    chains = []
    if hdr:
        # iPhone / HDR footage (HLG or PQ, BT.2020). Without tone mapping it looks
        # grey and washed out in SDR. zscale needs libzimg; fall back if missing.
        tin = v.get("color_transfer")
        chains.append(
            f"zscale=tin={tin}:pin=bt2020:min=bt2020nc:t=linear:npl=100,format=gbrpf32le,"
            "zscale=p=bt709,tonemap=mobius:desat=0,zscale=t=bt709:m=bt709:r=tv,"
            f"format=yuv420p,{base}")
    chains.append(f"{base},format=yuv420p")

    last_err = None
    for vf in chains:
        cmd = [FFMPEG, "-v", "error", "-y", "-i", str(src), "-map", "0:v:0"]
        if has_audio:
            cmd += ["-map", "0:a:0"]
        cmd += ["-vf", vf, "-c:v", "libx264", "-preset", "superfast", "-crf", "12",
                "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
        if has_audio:
            # aresample=async keeps audio aligned to timestamps if the source drifts.
            cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-af", "aresample=async=1:first_pts=0"]
        cmd += ["-movflags", "+faststart", str(dst)]
        try:
            _run(cmd)
            out = info(dst)
            if out.frame_count <= 0:
                raise ValueError("no frames could be decoded")
            return out
        except Exception as e:  # try the next (simpler) chain
            last_err = e
    raise ValueError(f"This file couldn't be read as a video ({str(last_err).splitlines()[-1][:200]})")


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
