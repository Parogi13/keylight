"""Pass 3a — Adaptation: make the subject easier to separate from its surroundings.

    out = M * E(I) + (1 - M) * D(I) + glow

E (subject): small exposure lift, local-contrast boost, slight saturation boost,
             soft highlight shoulder so nothing clips harshly.
D (background): luminance reduction plus slight desaturation.
glow: bright parts of the subject, blurred, added back faintly. On SDR screens a
      soft halo makes a region read as brighter than the panel can physically
      show (the glare illusion). Kept faint so it never becomes a neon outline.

All luminance math is in linear light (sRGB decoded via LUT), so dimming does not
shift hues and a "x0.7" really means 70% of the light.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# sRGB <-> linear lookup tables
_x = np.arange(256, dtype=np.float32) / 255.0
TO_LINEAR = np.where(_x <= 0.04045, _x / 12.92, ((_x + 0.055) / 1.055) ** 2.4).astype(np.float32)
_y = np.linspace(0, 1, 4096, dtype=np.float32)
TO_SRGB = np.clip(np.where(_y <= 0.0031308, _y * 12.92, 1.055 * np.power(_y, 1 / 2.4) - 0.055) * 255 + 0.5,
                  0, 255).astype(np.uint8)
LUMA = np.array([0.0722, 0.7152, 0.2126], np.float32)  # BGR order


@dataclass
class Params:
    strength: float = 0.6          # subject enhancement 0..1
    background: float = 0.35       # background luminance reduction 0..0.8
    glow: float = 0.35             # glow amount 0..1

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        return cls(
            strength=float(np.clip(d.get("strength", cls.strength), 0, 1)),
            background=float(np.clip(d.get("background", cls.background), 0, 0.8)),
            glow=float(np.clip(d.get("glow", cls.glow), 0, 1)),
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
    idx = np.clip(lin * 4095.0 + 0.5, 0, 4095).astype(np.int32)
    return TO_SRGB[idx]


def enhance(frame: np.ndarray, mask: np.ndarray, p: Params) -> np.ndarray:
    """frame: HxWx3 BGR uint8. mask: HxW float32 in [0,1] (full resolution)."""
    if p.strength <= 0 and p.background <= 0 and p.glow <= 0:
        return frame
    h, w = frame.shape[:2]
    lin = to_linear(frame)
    s = p.strength
    diag = float(np.hypot(w, h))

    # ---- background: dim + slight desaturation (whole frame, cheap) ----------
    k = p.background
    yb = lin @ LUMA
    out = lin + (yb[..., None] - lin) * (0.35 * k)
    out *= (1.0 - k)

    # ---- subject work happens only inside the mask's bounding region --------
    ys, xs = np.nonzero(mask > 2e-3)
    if len(xs) == 0:
        return to_srgb(np.clip(out, 0, 1))
    sig_base = diag * 0.012  # full-res sigma of the local-contrast base
    sig_glow = diag * 0.02
    pad = int(3 * max(sig_base, sig_glow if p.glow > 0 else 0)) + 4
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    lc, mc, oc = lin[y0:y1, x0:x1], mask[y0:y1, x0:x1, None], out[y0:y1, x0:x1]
    ch, cw = lc.shape[:2]

    # Local contrast: boost detail relative to a blurred base (blur at half res).
    small = cv2.resize(lc, (max(1, cw // 2), max(1, ch // 2)), interpolation=cv2.INTER_AREA)
    base = cv2.resize(cv2.GaussianBlur(small, (0, 0), sig_base / 2), (cw, ch), interpolation=cv2.INTER_LINEAR)
    e = np.clip(base + (1.0 + 0.5 * s) * (lc - base), 0, None)
    # Shadow-weighted lift: a luminance power curve (gamma < 1) raises dark subjects a
    # lot and bright ones a little — a black car at night needs stops, a lit face doesn't.
    # The lift is scaled by how dark the subject is, so well-lit subjects stay natural.
    y = e @ LUMA
    mw = mc[..., 0]
    subject_luma = float((y * mw).sum() / max(mw.sum(), 1e-6))
    dark = float(np.clip(1.0 - subject_luma / 0.18, 0.2, 1.0))
    gamma = 1.0 / (1.0 + 0.7 * s * dark)
    gain = (np.power(y + 1e-5, gamma) / (y + 1e-5)) * (1.0 + 0.15 * s)
    e *= np.minimum(gain, 1.0 + 7.0 * s)[..., None]
    y = e @ LUMA
    e = y[..., None] + (e - y[..., None]) * (1.0 + 0.15 * s)  # saturation
    e = _shoulder(np.clip(e, 0, None))

    # Two-zone blend: the lift uses a tighter core (m^2.5) than the dimming (m), so the
    # feathered edge shows the untouched original rather than lifted background — the
    # edge reads as a soft pool of light instead of a bright cut-out ring.
    core = mc ** 2.5
    blended = (1.0 - mc) * oc + mc * ((1.0 - core) * lc + core * e)

    # ---- glow: bright subject pixels, blurred, added faintly -----------------
    # Glow only helps where the scene is dark; on a bright scene it reads as a halo.
    scene_luma = float(yb[::8, ::8].mean())
    glow = p.glow * float(np.clip(1.0 - scene_luma / 0.12, 0.0, 1.0))
    if glow > 0 and s > 0:
        q = 4
        small_e = cv2.resize(e * mc, (max(1, cw // q), max(1, ch // q)), interpolation=cv2.INTER_AREA)
        bright = np.clip(small_e - 0.18, 0, None)
        halo = cv2.GaussianBlur(bright, (0, 0), sig_glow / q)
        blended += cv2.resize(halo, (cw, ch), interpolation=cv2.INTER_LINEAR) * (0.5 * glow * s)

    out[y0:y1, x0:x1] = blended
    return to_srgb(np.clip(out, 0, 1))


def attention_view(frame: np.ndarray, mask: np.ndarray, label: str | None,
                   box: tuple[float, float, float, float] | None) -> np.ndarray:
    """Debug visualization: dim cool-gray background, subject in true color,
    amber contour + label. This is what "Show AI attention" plays."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bg = cv2.merge([gray, gray, gray]).astype(np.float32) * 0.28
    bg[..., 0] *= 1.25  # slight blue tint
    m = mask[..., None]
    tint = frame.astype(np.float32) * 0.85 + np.array([20, 120, 255], np.float32) * 0.15
    out = np.clip(m * tint + (1 - m) * bg, 0, 255).astype(np.uint8)
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
