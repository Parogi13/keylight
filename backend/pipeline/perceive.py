"""Pass 1 — Perception: what exists, and where?

Runs YOLO segmentation + ByteTrack over the whole working video and records, per
frame, each tracked object's ID, class, confidence, box and a low-resolution mask.
Also detects hard scene cuts and records how dark each frame is.

Notes
- Dark frames are gamma-lifted *only for the detector* (the rendered video is never
  touched here). Low-light footage is our target use case and is exactly where
  detectors lose recall, so this is cheap insurance.
- If nothing is detected, the frame is left unchanged (no subject, no effect). An
  optional spectral-residual saliency fallback exists behind KEYLIGHT_SALIENCY_FALLBACK=1.
- Masks are stored as packed bits at ~320px wide: tens of MB for a minute of video.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import video_io

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "yolo11n-seg.pt"
MASK_LONG_SIDE = 480
SALIENT_ID = -1
# Off by default: with no detected subject the frame is left untouched. The saliency
# fallback picked a different "interesting" region every frame (random glowing).
SALIENCY_FALLBACK = os.environ.get("KEYLIGHT_SALIENCY_FALLBACK") == "1"

ProgressFn = Callable[[str, float], None]


def pick_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


@dataclass
class Obj:
    id: int
    cls: str
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in full-res pixels
    mask: int  # index into Perception.packed

    def to_dict(self) -> dict:
        return {"id": self.id, "cls": self.cls, "conf": round(self.conf, 3),
                "box": [round(v, 1) for v in self.box], "mask": self.mask}


@dataclass
class Perception:
    width: int
    height: int
    fps: float
    mask_w: int
    mask_h: int
    frames: list[list[Obj]] = field(default_factory=list)
    packed: list[np.ndarray] = field(default_factory=list)
    cuts: list[int] = field(default_factory=lambda: [0])  # first frame of each shot
    luma: list[float] = field(default_factory=list)
    device: str = "cpu"
    seconds: float = 0.0
    stride: int = 1

    # ---- mask helpers -------------------------------------------------------
    def add_mask(self, m: np.ndarray) -> int:
        self.packed.append(np.packbits(m.astype(bool), axis=None))
        return len(self.packed) - 1

    def mask(self, idx: int) -> np.ndarray:
        n = self.mask_w * self.mask_h
        return np.unpackbits(self.packed[idx], count=n).reshape(self.mask_h, self.mask_w).astype(np.float32)

    # ---- shots --------------------------------------------------------------
    def shot_bounds(self) -> list[tuple[int, int]]:
        ends = self.cuts[1:] + [len(self.frames)]
        return list(zip(self.cuts, ends))

    # ---- persistence --------------------------------------------------------
    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        if self.packed:
            np.savez_compressed(d / "masks.npz", masks=np.stack(self.packed))
        meta = {
            "width": self.width, "height": self.height, "fps": self.fps,
            "mask_w": self.mask_w, "mask_h": self.mask_h, "cuts": self.cuts,
            "luma": [round(x, 2) for x in self.luma], "device": self.device,
            "seconds": round(self.seconds, 2), "n_masks": len(self.packed), "stride": self.stride,
            "frames": [[o.to_dict() for o in objs] for objs in self.frames],
        }
        (d / "perception.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, d: Path) -> "Perception":
        meta = json.loads((d / "perception.json").read_text())
        p = cls(meta["width"], meta["height"], meta["fps"], meta["mask_w"], meta["mask_h"])
        p.cuts, p.luma, p.device, p.seconds = meta["cuts"], meta["luma"], meta["device"], meta["seconds"]
        p.stride = meta.get("stride", 1)
        p.frames = [[Obj(o["id"], o["cls"], o["conf"], tuple(o["box"]), o["mask"]) for o in objs]
                    for objs in meta["frames"]]
        if meta["n_masks"]:
            with np.load(d / "masks.npz") as z:
                p.packed = list(z["masks"])
        return p


# ---------------------------------------------------------------------------
# Scene cuts
# ---------------------------------------------------------------------------
class CutDetector:
    """Cheap hard-cut detector: HSV histogram distance + thumbnail difference."""

    def __init__(self, fps: float, min_shot_s: float = 0.5):
        self.prev_hist = None
        self.prev_thumb = None
        self.min_gap = max(1, int(fps * min_shot_s))
        self.last_cut = 0

    def __call__(self, frame: np.ndarray, t: int) -> bool:
        small = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        thumb = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        cut = False
        if self.prev_hist is not None and t - self.last_cut >= self.min_gap:
            bh = cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            mad = float(np.abs(thumb - self.prev_thumb).mean())
            cut = bh > 0.5 or (bh > 0.25 and mad > 30) or mad > 60
            if cut:
                self.last_cut = t
        self.prev_hist, self.prev_thumb = hist, thumb
        return cut


# ---------------------------------------------------------------------------
# Fallback saliency (spectral residual, Hou & Zhang 2007)
# ---------------------------------------------------------------------------
def spectral_saliency(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    g = cv2.cvtColor(cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(g.astype(np.float32))
    log_amp = np.log(np.abs(f) + 1e-6)
    residual = log_amp - cv2.blur(log_amp, (3, 3))
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(f)))) ** 2
    sal = cv2.GaussianBlur(sal.astype(np.float32), (9, 9), 2.5)
    sal = cv2.resize(sal, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-6)
    binary = (sal > 0.45).astype(np.uint8)
    # Keep only the strongest blob: scattered blobs make the effect look random.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if n <= 1:
        return binary.astype(bool)
    best = 1 + int(np.argmax([sal[labels == i].sum() for i in range(1, n)]))
    return labels == best


# ---------------------------------------------------------------------------
def _detector_input(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """Gamma-lift dark frames so the detector can see. Returns (image, mean luma)."""
    luma = float(cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY).mean())
    if luma >= 90:
        return frame, luma
    gamma = float(np.clip(np.log(0.42) / np.log(max(luma, 4) / 255.0), 0.35, 1.0))
    lut = (np.power(np.arange(256) / 255.0, gamma) * 255).astype(np.uint8)
    return cv2.LUT(frame, lut), luma


_MODEL = None


def _load_model():
    """Load YOLO once per process: saves ~1-3 s of startup on every job and re-render."""
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        # Bundled weights if present; otherwise Ultralytics downloads them once (needs internet).
        _MODEL = YOLO(str(MODEL_PATH) if MODEL_PATH.exists() else MODEL_PATH.name)
    return _MODEL


def default_stride(fps: float) -> int:
    """Run the detector at ~30 Hz: every frame for <=30 fps video, every 2nd at 60 fps."""
    env = os.environ.get("KEYLIGHT_DETECT_STRIDE")
    if env:
        return max(1, int(env))
    return max(1, int(round(fps / 30.0)))


def run(working: Path, info: video_io.VideoInfo, progress: ProgressFn | None = None,
        imgsz: int = 640, conf: float = 0.25, device: str | None = None,
        stride: int | None = None) -> Perception:
    device = device or pick_device()
    stride = stride or default_stride(info.fps)
    model = _load_model()
    _reset_tracker(model)  # the cached model may carry tracks from the previous video
    names = model.names

    scale = min(1.0, MASK_LONG_SIDE / max(info.width, info.height))
    mask_w, mask_h = max(1, round(info.width * scale)), max(1, round(info.height * scale))
    sx, sy = mask_w / info.width, mask_h / info.height
    p = Perception(info.width, info.height, info.fps, mask_w, mask_h, device=device)
    p.stride = stride
    cuts = CutDetector(info.fps)
    t0 = time.time()

    def track(img):
        return model.track(img, persist=True, imgsz=imgsz, conf=conf, iou=0.6,
                           tracker="bytetrack.yaml", verbose=False, device=device)[0]

    last_key: int | None = None      # last frame the detector actually ran on
    pending: list[int] = []          # skipped frames waiting for the next keyframe

    for t, frame in enumerate(video_io.iter_frames(working)):
        is_cut = cuts(frame, t)
        if is_cut:
            p.cuts.append(t)
            _reset_tracker(model)
            _fill(p, pending, last_key, None)  # never interpolate across a cut
            pending = []

        inp, luma = _detector_input(frame)
        p.luma.append(luma)

        if not is_cut and last_key is not None and (t - last_key) < stride:
            p.frames.append([])  # placeholder, filled at the next keyframe
            pending.append(t)
            continue

        try:
            res = track(inp)
        except Exception:
            if device == "cpu":
                raise
            device = p.device = "cpu"  # MPS op gaps: fall back rather than debug at 4am
            res = track(inp)

        objs: list[Obj] = []
        boxes = res.boxes
        if boxes is not None and boxes.id is not None and res.masks is not None:
            ids = boxes.id.int().tolist()
            clss = boxes.cls.int().tolist()
            confs = boxes.conf.tolist()
            xyxy = boxes.xyxy.tolist()
            for k, poly in enumerate(res.masks.xy):
                if len(poly) < 3:
                    continue
                m = np.zeros((mask_h, mask_w), np.uint8)
                pts = np.round(np.asarray(poly) * [sx, sy]).astype(np.int32)
                cv2.fillPoly(m, [pts], 1)
                if m.sum() < 4:
                    continue
                objs.append(Obj(ids[k], names[clss[k]], float(confs[k]), tuple(xyxy[k]), p.add_mask(m)))

        if not objs and SALIENCY_FALLBACK:
            sal = spectral_saliency(frame, mask_w, mask_h)
            ys, xs = np.nonzero(sal)
            if len(xs):
                box = (xs.min() / sx, ys.min() / sy, xs.max() / sx, ys.max() / sy)
                objs.append(Obj(SALIENT_ID, "salient region", 0.2, box, p.add_mask(sal)))

        p.frames.append(objs)
        _fill(p, pending, last_key, t)
        pending = []
        last_key = t
        if progress and t % 5 == 0:
            progress("perceive", min(1.0, (t + 1) / max(1, info.frame_count)))

    _fill(p, pending, last_key, None)
    p.seconds = time.time() - t0
    return p


def _fill(p: Perception, pending: list[int], k0: int | None, k1: int | None) -> None:
    """Fill skipped frames between keyframes k0 and k1 (k1=None: hold k0).

    Objects seen at both keyframes get a linearly interpolated box, and the nearer
    keyframe's mask translated by the box-centre motion. Objects seen at only one
    keyframe are held from it while it is the nearer one.
    """
    if not pending or k0 is None:
        return
    a = {o.id: o for o in p.frames[k0]}
    b = {o.id: o for o in p.frames[k1]} if k1 is not None else {}
    sx, sy = p.mask_w / p.width, p.mask_h / p.height
    for t in pending:
        u = (t - k0) / (k1 - k0) if k1 is not None else 0.0
        objs = []
        for oid in set(a) | set(b):
            oa, ob = a.get(oid), b.get(oid)
            if oa is not None and ob is not None:
                box = tuple(x0 + (x1 - x0) * u for x0, x1 in zip(oa.box, ob.box))
                src = oa if u < 0.5 else ob
            elif oa is not None and u < 0.5 or (oa is not None and k1 is None):
                box, src = oa.box, oa
            elif ob is not None and u >= 0.5:
                box, src = ob.box, ob
            else:
                continue
            dx = ((box[0] + box[2]) - (src.box[0] + src.box[2])) / 2 * sx
            dy = ((box[1] + box[3]) - (src.box[1] + src.box[3])) / 2 * sy
            if abs(dx) < 0.5 and abs(dy) < 0.5:
                idx = src.mask  # share the packed mask: no extra memory
            else:
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                idx = p.add_mask(cv2.warpAffine(p.mask(src.mask), M, (p.mask_w, p.mask_h)) > 0.5)
            objs.append(Obj(oid, src.cls, src.conf, box, idx))
        p.frames[t] = objs


def _reset_tracker(model) -> None:
    pred = getattr(model, "predictor", None)
    for tr in getattr(pred, "trackers", None) or []:
        try:
            tr.reset()
        except Exception:
            pass
