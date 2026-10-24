"""HDR output: SDR BT.709 frames -> BT.2020 / PQ (SMPTE ST 2084), 10-bit HEVC.

Why: on an XDR display, macOS shows HDR video with *extra* brightness above SDR
reference white (EDR headroom). Keeping the background at reference white and
pushing only the subject above it makes the subject genuinely brighter than any
pixel the original SDR video can show, which is the core Keylight effect.

Conversion (per pixel, display-referred, following ITU-R BT.2408):
  1. 8-bit BT.709 code value -> linear light with the BT.1886 EOTF (gamma 2.4)
  2. linear BT.709 primaries -> linear BT.2020 primaries (3x3 matrix)
  3. scale so SDR 100% white = REF_WHITE nits (203 nits, the BT.2408 reference)
  4. absolute nits -> PQ code values (ST 2084 inverse EOTF)
  5. R'G'B' -> Y'CbCr BT.2020 non-constant-luminance, 10-bit limited range,
     4:2:0 (done by ffmpeg's scaler), tagged bt2020 / smpte2084 / bt2020nc.

Steps 1-4 are the same for every pixel of a given 8-bit colour except the matrix,
so they use lookup tables and one cv2.transform: fast enough per frame.

Nothing here reinterprets SDR code values as HDR: SDR content keeps its intended
appearance at reference white, and only explicitly boosted pixels exceed it.
"""
from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from . import video_io

# Nits that SDR 100% white maps to. BT.2408 says 203; macOS may display PQ with
# 100 nits as SDR white, which would make 203 look ~2x too bright (washed out).
# Pick the value whose gray steps match SDR on the calibration page, set it in .env.
REF_WHITE = float(os.environ.get("KEYLIGHT_HDR_REF_WHITE", 203.0))
MAX_NITS = 1000.0  # never encode above this: XDR sustains ~1000 nits full-screen

# --- transfer functions ------------------------------------------------------
_v = np.arange(256, dtype=np.float64) / 255.0
BT1886 = np.power(_v, 2.4).astype(np.float32)  # 8-bit code -> relative linear light

# BT.709 -> BT.2020 linear-light primaries (ITU-R BT.2087). Rows produce R, G, B
# (RGB order, what ffmpeg's rgb48le expects); columns take OpenCV's B, G, R input.
_M709_TO_2020_RGB = np.array([[0.6274, 0.3293, 0.0433],
                              [0.0691, 0.9195, 0.0114],
                              [0.0164, 0.0880, 0.8956]], np.float64)
M_BGR709_TO_RGB2020 = _M709_TO_2020_RGB[:, ::-1].astype(np.float32)

_PQ_M1, _PQ_M2 = 2610 / 16384, 2523 / 4096 * 128
_PQ_C1, _PQ_C2, _PQ_C3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32


def pq_oetf(nits: np.ndarray) -> np.ndarray:
    """Absolute luminance (cd/m^2) -> PQ signal in [0, 1] (exact formula)."""
    y = np.clip(np.asarray(nits, np.float64) / 10000.0, 0, 1)
    yp = np.power(y, _PQ_M1)
    return np.power((_PQ_C1 + _PQ_C2 * yp) / (1 + _PQ_C3 * yp), _PQ_M2)


def pq_eotf(signal: np.ndarray) -> np.ndarray:
    """PQ signal in [0, 1] -> absolute luminance (cd/m^2)."""
    e = np.power(np.clip(np.asarray(signal, np.float64), 0, 1), 1 / _PQ_M2)
    return 10000.0 * np.power(np.maximum(e - _PQ_C1, 0) / (_PQ_C2 - _PQ_C3 * e), 1 / _PQ_M1)


# PQ via LUT indexed on sqrt(nits / MAX_NITS): sqrt spacing keeps precision in the
# shadows, where PQ is steepest. 65536 entries -> error far below one 10-bit step.
_LUT_N = 65536
_s = np.linspace(0, 1, _LUT_N, dtype=np.float64)
PQ_LUT16 = np.round(pq_oetf((_s ** 2) * MAX_NITS) * 65535).astype(np.uint16)


def nits_to_rgb48(rgb_nits: np.ndarray) -> np.ndarray:
    """HxWx3 float32 linear BT.2020 RGB in nits -> HxWx3 uint16 PQ R'G'B'."""
    x = np.clip(rgb_nits * (1.0 / MAX_NITS), 0, 1)
    cv2.sqrt(x, x)
    idx = (x * (_LUT_N - 1) + 0.5).astype(np.int32)
    return PQ_LUT16[idx]


def sdr_to_nits(frame_bgr: np.ndarray, ref_white: float | None = None) -> np.ndarray:
    """8-bit BT.709 BGR frame -> float32 linear BT.2020 RGB in nits (SDR white = REF_WHITE)."""
    lin = BT1886[frame_bgr]                       # relative linear BT.709, BGR
    rgb2020 = cv2.transform(lin, M_BGR709_TO_RGB2020)
    rgb2020 *= REF_WHITE if ref_white is None else ref_white
    return rgb2020


# --- encoding ----------------------------------------------------------------
HDR_TAGS = ["-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc",
            "-color_range", "tv"]


@lru_cache(maxsize=1)
def _encoders() -> str:
    try:
        return subprocess.run([video_io.FFMPEG, "-hide_banner", "-encoders"], capture_output=True,
                              text=True, timeout=20).stdout
    except Exception:
        return ""


def pick_encoder() -> str:
    """'hevc_videotoolbox' on Macs (hardware, fast), else 'libx265'."""
    forced = os.environ.get("KEYLIGHT_HDR_ENCODER")
    if forced:
        return forced
    enc = _encoders()
    if "hevc_videotoolbox" in enc:
        return "hevc_videotoolbox"
    if "libx265" in enc:
        return "libx265"
    raise RuntimeError("No HEVC encoder in this ffmpeg build (need hevc_videotoolbox or libx265).")


def available() -> bool:
    try:
        pick_encoder()
        return True
    except RuntimeError:
        return False


def _encoder_args(name: str, w: int, h: int, fps: float) -> list[str]:
    if name == "hevc_videotoolbox":
        # Constant bitrate target sized to resolution; ~0.35 bits/pixel/frame is
        # visually transparent for this content. main10 = 10-bit.
        mbps = max(8.0, w * h * fps * 0.35 / 1e6)
        return ["-c:v", "hevc_videotoolbox", "-profile:v", "main10", "-b:v", f"{mbps:.0f}M",
                "-allow_sw", "1", "-pix_fmt", "p010le"]
    # libx265 software fallback; HDR10 SEI with BT.2020 / D65 mastering display.
    params = ("log-level=error:hdr10=1:hdr10-opt=1:repeat-headers=1:colorprim=bt2020:"
              "transfer=smpte2084:colormatrix=bt2020nc:range=limited:"
              "master-display=G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(10000000,1):"
              f"max-cll={int(MAX_NITS)},{int(REF_WHITE)}")
    return ["-c:v", "libx265", "-preset", "fast", "-crf", "16", "-x265-params", params,
            "-pix_fmt", "yuv420p10le"]


class HDRWriter:
    """Pipe uint16 PQ R'G'B' frames into ffmpeg -> 10-bit HEVC HDR mp4 (Safari-ready)."""

    def __init__(self, dst: str | Path, width: int, height: int, fps_fraction: str,
                 audio_from: str | Path | None = None, encoder: str | None = None):
        from fractions import Fraction
        self.encoder = encoder or pick_encoder()
        fps = float(Fraction(fps_fraction))
        cmd = [video_io.FFMPEG, "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{width}x{height}",
               "-framerate", fps_fraction, "-i", "-"]
        if audio_from is not None:
            cmd += ["-i", str(audio_from), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy"]
        enc = _encoder_args(self.encoder, width, height, fps)
        pix = enc[enc.index("-pix_fmt") + 1]
        cmd += ["-vf", f"scale=in_range=full:out_color_matrix=bt2020:out_range=tv,format={pix}",
                *enc, *HDR_TAGS,
                "-tag:v", "hvc1",  # Safari/QuickTime require hvc1, not hev1
                "-movflags", "+faststart", "-f", "mp4", str(dst)]
        self.dst = Path(dst)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, rgb48: np.ndarray) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(rgb48, dtype="<u2").tobytes())

    def close(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode() if self.proc.stderr else ""
        if self.proc.wait() != 0:
            raise RuntimeError(f"HDR encode failed ({self.encoder}) for {self.dst}: {err[-2000:]}")

    def kill(self) -> None:
        try:
            self.proc.kill()
        except Exception:
            pass
