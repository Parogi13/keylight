"""Pass 3a — Adaptation: make the subject easier to separate from its surroundings.

    out = M * E(I) + (1 - M) * D(I) + glow

E (subject): small exposure lift, local-contrast boost, slight saturation boost,
             soft highlight shoulder so nothing clips harshly.
D (background): luminance reduction only (colour untouched), scaled by subject presence.
glow: bright parts of the subject, blurred, added back faintly. On SDR screens a
      soft halo makes a region read as brighter than the panel can physically
      show (the glare illusion). Kept faint so it never becomes a neon outline.

All luminance math is in linear light (sRGB decoded via LUT), so dimming does not
shift hues and a "x0.7" really means 70% of the light.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

# sRGB <-> linear lookup tables
_x = np.arange(256, dtype=np.float32) / 255.0
TO_LINEAR = np.where(_x <= 0.04045, _x / 12.92, ((_x + 0.055) / 1.055) ** 2.4).astype(np.float32)
_N = 16384  # fine enough that a linear round trip is exact for every 8-bit value
_y = np.linspace(0, 1, _N, dtype=np.float32)
TO_SRGB = np.clip(np.where(_y <= 0.0031308, _y * 12.92, 1.055 * np.power(_y, 1 / 2.4) - 0.055) * 255 + 0.5,
                  0, 255).astype(np.uint8)
LUMA = np.array([0.0722, 0.7152, 0.2126], np.float32)  # BGR order


@lru_cache(maxsize=32)
def _dim_lut(k: float) -> np.ndarray:
    """8-bit LUT that multiplies *linear* light by (1 - k). Per-channel, so cv2.LUT
    can apply it to a whole frame in a couple of milliseconds."""
    lin = TO_LINEAR * (1.0 - k)
    return TO_SRGB[np.clip(lin * (_N - 1) + 0.5, 0, _N - 1).astype(np.int32)]


@dataclass
class Params:
    strength: float = 0.6          # subject enhancement 0..1
    background: float = 0.15       # background luminance reduction 0..0.8 (dim only)
    glow: float = 0.35             # glow amount 0..1
    hdr: bool = True               # also write HDR (BT.2020/PQ) outputs

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        return cls(
            strength=float(np.clip(d.get("strength", cls.strength), 0, 1)),
            background=float(np.clip(d.get("background", cls.background), 0, 0.8)),
            glow=float(np.clip(d.get("glow", cls.glow), 0, 1)),
            hdr=bool(d.get("hdr", cls.hdr)),
        )


def _shoulder(x: np.ndarray, knee: float = 0.75) -> np.ndarray:
    """Identity below `knee`, smooth roll-off to 1.0 above it."""
    over = x > knee
    if over.any():
        r = 1.0 - knee
        x = x.copy()
        x[over] = knee + r * np.tanh((x[over] - knee) / r)
    return x


def to_linear(frame: np.ndarray) -> np.ndarray:
    return TO_LINEAR[frame]


def to_srgb(lin: np.ndarray) -> np.ndarray:
    idx = np.clip(lin * (_N - 1) + 0.5, 0, _N - 1).astype(np.int32)
    return TO_SRGB[idx]


_GN = 4096


@lru_cache(maxsize=64)
def _gain_lut(gamma: float, s: float) -> np.ndarray:
    """Per-pixel gain for the shadow lift as a function of luminance in [0, 1]."""
    yy = np.linspace(0, 1, _GN, dtype=np.float32)
    gain = (np.power(yy + 1e-5, gamma) / (yy + 1e-5)) * (1.0 + 0.15 * s)
    return np.minimum(gain, 1.0 + 7.0 * s).astype(np.float32)


def _lut_index(y: np.ndarray) -> np.ndarray:
    return np.clip(y * (_GN - 1) + 0.5, 0, _GN - 1).astype(np.int32)


def upsample_region(mask_low: np.ndarray, full_w: int, full_h: int,
                    x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Exactly the [y0:y1, x0:x1] window of cv2.resize(mask_low, (full_w, full_h)),
    without materializing the full-resolution mask."""
    lh, lw = mask_low.shape
    sx, sy = lw / full_w, lh / full_h
    M = np.array([[sx, 0, (x0 + 0.5) * sx - 0.5], [0, sy, (y0 + 0.5) * sy - 0.5]], np.float32)
    return cv2.warpAffine(mask_low, M, (x1 - x0, y1 - y0),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


def background(frame: np.ndarray, k: float) -> np.ndarray:
    """D(I): a linear-light dim by (1 - k), nothing else. Hue and saturation are left
    exactly as filmed. One uint8 LUT pass with cv2: ~1 ms for 1080p."""
    if k <= 0:
        return frame
    return cv2.LUT(frame, _dim_lut(round(k, 3)))


def _region(mask_low: np.ndarray, w: int, h: int, pad: int) -> tuple[int, int, int, int] | None:
    """Full-res bounding box (x0, y0, x1, y1) of the attention mask, padded."""
    lh, lw = mask_low.shape
    ys, xs = np.nonzero(mask_low > 2e-3)
    if len(xs) == 0:
        return None
    fx, fy = w / lw, h / lh
    return (max(0, int(xs.min() * fx) - pad), max(0, int(ys.min() * fy) - pad),
            min(w, int((xs.max() + 1) * fx) + pad), min(h, int((ys.max() + 1) * fy) + pad))


def _shape_subject(lc: np.ndarray, mc: np.ndarray, s: float, sig_base: float) -> np.ndarray:
    """Subject detail + shadow lift in relative linear light (shared by SDR and HDR).

    Local contrast boosts detail against a blurred base (blur at half res). The
    shadow-weighted lift is a luminance power curve (gamma < 1) scaled by how dark the
    subject is: a black car at night gets stops, a lit face barely changes.
    """
    ch, cw = lc.shape[:2]
    # Detail is boosted as a *ratio* of luminance to its blurred surround (contrast in
    # log space). An additive boost in linear light drives dark pixels next to bright
    # ones negative — a black car beside its own tail lights was clipped to pure black.
    y0 = lc @ LUMA
    small = cv2.resize(y0, (max(1, cw // 2), max(1, ch // 2)), interpolation=cv2.INTER_AREA)
    yb = cv2.resize(cv2.GaussianBlur(small, (0, 0), sig_base / 2), (cw, ch), interpolation=cv2.INTER_LINEAR)
    eps = 1e-4
    detail = np.power((y0 + eps) / (yb + eps), 0.5 * s)
    e = lc * np.minimum(detail, 4.0)[..., None]
    y = e @ LUMA
    mw = mc[..., 0]
    subject_luma = float((y * mw).sum() / max(mw.sum(), 1e-6))
    dark = float(np.clip(1.0 - subject_luma / 0.18, 0.2, 1.0))
    gamma = 1.0 / (1.0 + 0.7 * s * dark)
    e *= _gain_lut(round(gamma, 3), round(s, 3))[_lut_index(y)][..., None]
    return e


def enhance(frame: np.ndarray, mask_low: np.ndarray, p: Params) -> np.ndarray:
    """frame: HxWx3 BGR uint8. mask_low: low-res float32 attention mask in [0, 1].

    Only the subject's bounding region is processed in float; everything else is
    the uint8 background path. Output is HxWx3 uint8.
    """
    if p.strength <= 0 and p.background <= 0 and p.glow <= 0:
        return frame
    # No subject in view -> the frame is returned untouched. The background dim fades
    # in and out with the subject's (temporally smoothed) presence, so there is no pop
    # when a subject appears or leaves.
    presence = float(mask_low.max())
    if presence < 2e-3:
        return frame
    h, w = frame.shape[:2]
    s = p.strength
    diag = float(np.hypot(w, h))
    out = background(frame, round(p.background * presence, 2))
    if out is frame:
        out = frame.copy()

    sig_base = diag * 0.012  # full-res sigma of the local-contrast base
    sig_glow = diag * 0.02
    pad = int(3 * max(sig_base, sig_glow if p.glow > 0 else 0)) + 4
    x0, y0, x1, y1 = _region(mask_low, w, h, pad)

    mc = upsample_region(mask_low, w, h, x0, y0, x1, y1)[..., None]
    lc = to_linear(frame[y0:y1, x0:x1])
    oc = to_linear(out[y0:y1, x0:x1])
    ch, cw = lc.shape[:2]

    e = _shape_subject(lc, mc, s, sig_base)
    y3 = (e @ LUMA)[..., None]
    e -= y3                       # saturation: push chroma away from luma, in place
    e *= 1.0 + 0.15 * s
    e += y3
    np.maximum(e, 0, out=e)
    e = _shoulder(e)

    # ---- glow: bright subject pixels, blurred, added faintly -----------------
    # Glow only helps where the scene is dark; on a bright scene it reads as a halo.
    thumb = cv2.resize(frame, (64, max(1, int(64 * h / w))), interpolation=cv2.INTER_AREA)
    scene_luma = float((to_linear(thumb) @ LUMA).mean())
    glow = p.glow * float(np.clip(1.0 - scene_luma / 0.12, 0.0, 1.0))
    halo = None
    if glow > 0 and s > 0:
        q = 4
        small_e = cv2.resize(e * mc, (max(1, cw // q), max(1, ch // q)), interpolation=cv2.INTER_AREA)
        bright = np.clip(small_e - 0.18, 0, None)
        halo = cv2.resize(cv2.GaussianBlur(bright, (0, 0), sig_glow / q), (cw, ch), interpolation=cv2.INTER_LINEAR)

    # Two-zone blend: the lift uses a tighter core (m^2.5) than the dimming (m), so the
    # feathered edge shows the untouched original rather than lifted background — the
    # edge reads as a soft pool of light instead of a bright cut-out ring.
    #   blended = oc + mc * ((lc + core * (e - lc)) - oc)      (computed in place)
    core = mc * mc * np.sqrt(mc)
    e -= lc
    e *= core
    e += lc
    e -= oc
    e *= mc
    e += oc
    blended = e
    if halo is not None:
        blended += halo * (0.5 * glow * s)

    out[y0:y1, x0:x1] = to_srgb(np.clip(blended, 0, 1))
    return out


def attention_view(frame: np.ndarray, mask: np.ndarray, label: str | None,
                   box: tuple[float, float, float, float] | None) -> np.ndarray:
    """Debug visualization: dim cool-gray background, subject in true color,
    amber contour + label. This is what "Show AI attention" plays."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bg = cv2.merge([cv2.convertScaleAbs(gray, alpha=0.35), cv2.convertScaleAbs(gray, alpha=0.28),
                    cv2.convertScaleAbs(gray, alpha=0.28)])  # dim, slightly blue
    tint = cv2.addWeighted(frame, 0.85, np.full_like(frame, (20, 120, 255)), 0.15, 0)
    m = np.ascontiguousarray(mask, np.float32)
    out = cv2.blendLinear(tint, bg, m, 1.0 - m)  # per-pixel alpha blend in uint8
    contour = (mask > 0.5).astype(np.uint8)
    cnts, _ = cv2.findContours(contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    th = max(2, frame.shape[1] // 480)
    cv2.drawContours(out, cnts, -1, (40, 180, 255), th, cv2.LINE_AA)
    if label and box is not None:
        x1, y1 = int(box[0]), int(box[1])
        fs = max(0.5, frame.shape[1] / 1600)
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        y = max(tht + 10, y1 - 8)
        cv2.rectangle(out, (x1, y - tht - 8), (x1 + tw + 12, y + 6), (40, 180, 255), -1)
        cv2.putText(out, label, (x1 + 6, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), th, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# HDR: subject lifted above SDR reference white; background untouched
# ---------------------------------------------------------------------------
LUMA2020_RGB = np.array([0.2627, 0.6780, 0.0593], np.float32)  # BT.2020 luminance, RGB order
HDR_KNEE = 500.0  # nits: subject highlights roll off above this toward hdr.MAX_NITS


def hdr_gain(s: float) -> float:
    """Subject brightness multiple vs. the original: 1x at 0, ~2.2x at the 0.6
    default (SDR white -> ~450 nits), 3x at full strength (-> ~600 nits)."""
    return 1.0 + 2.0 * float(np.clip(s, 0, 1))


def _rolloff(y: np.ndarray, knee: float, peak: float) -> np.ndarray:
    out = y.copy()
    over = y > knee
    if over.any():
        r = peak - knee
        out[over] = knee + r * np.tanh((y[over] - knee) / r)
    return out


def enhance_hdr(frame: np.ndarray, mask_low: np.ndarray, p: Params, base_rgb48: np.ndarray) -> np.ndarray:
    """HDR version of the subject enhancement.

    base_rgb48 is the frame already converted to PQ (the HDR baseline). Pixels with
    no subject nearby are copied from it unchanged, so the background matches the
    HDR baseline exactly. On the subject: the same detail + shadow lift as SDR, then
    a brightness multiple (hdr_gain) applied to luminance with a soft roll-off above
    HDR_KNEE, so highlights land around 400-600 nits instead of clipping.
    """
    from . import hdr as H

    s = p.strength
    if s <= 0 or float(mask_low.max()) < 2e-3:
        return base_rgb48
    h, w = frame.shape[:2]
    diag = float(np.hypot(w, h))
    sig_base = diag * 0.012
    reg = _region(mask_low, w, h, int(3 * sig_base) + 4)
    if reg is None:
        return base_rgb48
    x0, y0, x1, y1 = reg
    mc = upsample_region(mask_low, w, h, x0, y0, x1, y1)[..., None]
    crop = frame[y0:y1, x0:x1]

    lc = H.BT1886[crop]                                  # relative linear BT.709 (BGR)
    e = _shape_subject(lc, mc, s, sig_base)
    y3 = (e @ LUMA)[..., None]                           # gentle saturation, as in SDR
    e -= y3
    e *= 1.0 + 0.1 * s
    e += y3
    np.maximum(e, 0, out=e)

    e_n = cv2.transform(e, H.M_BGR709_TO_RGB2020) * H.REF_WHITE   # nits, BT.2020 RGB
    e_n *= hdr_gain(s)
    yl = e_n @ LUMA2020_RGB
    scale = _rolloff(yl, HDR_KNEE, H.MAX_NITS) / np.maximum(yl, 1e-6)
    e_n *= scale[..., None]                              # luminance roll-off, hue kept

    base_n = H.sdr_to_nits(crop)
    core = mc * mc * np.sqrt(mc)                         # same two-zone edge as SDR
    e_n -= base_n
    e_n *= core
    e_n += base_n

    out = base_rgb48.copy()
    out[y0:y1, x0:x1] = H.nits_to_rgb48(e_n)
    return out
