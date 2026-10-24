"""Experimental perception backend: YOLO proposes, SAM 2 segments and propagates.

    segmentation_backend = "yolo"  -> perceive.run (YOLO11n-seg + ByteTrack, every frame)
    segmentation_backend = "sam2"  -> this module

Division of labour
- YOLO runs only on keyframes (shot starts, then every `yolo_interval_s`, default 1 s).
  It answers "what objects are here, and what class are they?".
- SAM 2 (official facebookresearch/sam2, SAM 2.1 Hiera Tiny by default) is prompted
  with YOLO's boxes and *propagates* precise masks through every frame in between
  using its memory attention. No mask is ever copied unchanged across frames.
- The output is a regular `perceive.Perception`, so temporal attention, scene cuts,
  the SDR/HDR renderers and the player work unchanged.

Windows: each shot is processed in windows of `yolo_interval_s`. A window starts
at a YOLO keyframe; objects carried over from the previous window are re-seeded
with their last SAM mask (identity preserved), YOLO detections that match no
carried object become new objects (prompted with their box), and carried objects
that YOLO hasn't confirmed for `drop_after` keyframes and whose mask has collapsed
are dropped. Windows never cross a scene cut, so masks never leak between shots,
and memory stays bounded regardless of video length.

Device: Apple Silicon uses MPS (PYTORCH_ENABLE_MPS_FALLBACK=1 lets unsupported
ops fall back to CPU). If SAM 2 fails on MPS it retries the video on CPU.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from . import perceive, video_io  # noqa: E402
from .perceive import CutDetector, Obj, Perception, _detector_input, _load_model, pick_device  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAM2_CKPT = Path(os.environ.get("KEYLIGHT_SAM2_CKPT", ROOT / "models" / "sam2.1_hiera_tiny.pt"))
SAM2_CFG = os.environ.get("KEYLIGHT_SAM2_CFG", "configs/sam2.1/sam2.1_hiera_t.yaml")
SAM_LONG_SIDE = 1024          # frames handed to SAM 2 (it resizes to 1024 internally anyway)
MAX_OBJECTS = int(os.environ.get("KEYLIGHT_SAM2_MAX_OBJECTS", 3))
MIN_BOX_FRAC = 0.002          # ignore YOLO boxes smaller than 0.2% of the frame
MIN_CONF = 0.35

ProgressFn = Callable[[str, float], None]


def available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("sam2") is not None and SAM2_CKPT.exists()
    except Exception:
        return False


@dataclass
class _Track:
    id: int
    cls: str
    conf: float
    last_mask: np.ndarray | None = None  # bool, SAM resolution
    unconfirmed: int = 0                 # keyframes since YOLO last matched it


def _box_of(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def _iou_box(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class _SAM2:
    """Thin wrapper around the official video predictor, with an MPS->CPU fallback."""

    def __init__(self, device: str):
        from sam2.build_sam import build_sam2_video_predictor
        self.device = device
        self.pred = build_sam2_video_predictor(SAM2_CFG, str(SAM2_CKPT), device=device)

    def run_window(self, frame_dir: str, seeds: list[tuple[int, np.ndarray | None, tuple | None]]):
        """seeds: (obj_id, mask_or_None, box_or_None) on window frame 0.
        Yields (frame_idx, {obj_id: bool mask}) for every frame in the window."""
        import torch
        with torch.inference_mode():
            state = self.pred.init_state(video_path=frame_dir, offload_video_to_cpu=True,
                                         offload_state_to_cpu=self.device == "mps")
            for oid, mask, box in seeds:
                if mask is not None and mask.any():
                    self.pred.add_new_mask(inference_state=state, frame_idx=0, obj_id=oid, mask=mask)
                elif box is not None:
                    self.pred.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=oid,
                                                    box=np.asarray(box, np.float32))
            for fidx, obj_ids, logits in self.pred.propagate_in_video(state):
                masks = (logits[:, 0] > 0.0).cpu().numpy()
                yield fidx, {int(o): masks[i] for i, o in enumerate(obj_ids)}
            self.pred.reset_state(state)


def run(working: Path, info: video_io.VideoInfo, progress: ProgressFn | None = None,
        yolo_interval_s: float = 1.0, device: str | None = None, max_objects: int = MAX_OBJECTS,
        stats: dict | None = None) -> Perception:
    device = device or pick_device()
    yolo = _load_model()
    names = yolo.names
    stats = stats if stats is not None else {}
    t_start = time.time()

    scale_m = min(1.0, perceive.MASK_LONG_SIDE / max(info.width, info.height))
    mask_w, mask_h = max(1, round(info.width * scale_m)), max(1, round(info.height * scale_m))
    s_sam = min(1.0, SAM_LONG_SIDE / max(info.width, info.height))
    sam_w, sam_h = max(2, round(info.width * s_sam)), max(2, round(info.height * s_sam))
    to_sam = np.array([sam_w / info.width, sam_h / info.height] * 2)

    p = Perception(info.width, info.height, info.fps, mask_w, mask_h, device=device)
    p.stride = 1
    window = max(2, int(round(info.fps * yolo_interval_s)))

    # ---- pass 1: decode once, cut detection, SAM-sized JPEGs, YOLO on keyframes ----
    tmp = Path(tempfile.mkdtemp(prefix="keylight_sam2_"))
    cuts = CutDetector(info.fps)
    keyframes: dict[int, list[tuple[str, float, tuple]]] = {}
    t0 = time.time()
    t_yolo = 0.0
    shot_start = 0
    n = 0
    for t, frame in enumerate(video_io.iter_frames(working)):
        if cuts(frame, t):
            p.cuts.append(t)
            shot_start = t
        inp, luma = _detector_input(frame)
        p.luma.append(luma)
        cv2.imwrite(str(tmp / f"{t:06d}.jpg"), cv2.resize(frame, (sam_w, sam_h), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        if (t - shot_start) % window == 0:
            ty = time.time()
            res = yolo.predict(inp, imgsz=640, conf=0.25, iou=0.6, verbose=False, device=device)[0]
            t_yolo += time.time() - ty
            dets = []
            if res.boxes is not None:
                for c, cf, b in zip(res.boxes.cls.int().tolist(), res.boxes.conf.tolist(), res.boxes.xyxy.tolist()):
                    area = (b[2] - b[0]) * (b[3] - b[1]) / (info.width * info.height)
                    if cf >= MIN_CONF and area >= MIN_BOX_FRAC:
                        dets.append((names[c], float(cf), tuple(b)))
            keyframes[t] = dets
        n = t + 1
        if progress and t % 10 == 0:
            progress("perceive", 0.15 * (t + 1) / max(1, info.frame_count))
    stats["decode_cut_yolo_s"] = round(time.time() - t0, 2)
    stats["yolo_inference_s"] = round(t_yolo, 2)
    stats["yolo_keyframes"] = len(keyframes)
    p.frames = [[] for _ in range(n)]

    # ---- pass 2: SAM 2 propagation, window by window, never across a cut ----
    try:
        sam = _SAM2(device)
    except Exception:
        if device == "cpu":
            raise
        device = p.device = "cpu"
        sam = _SAM2(device)
    t0 = time.time()
    next_id = 1
    failures = 0
    bounds = p.cuts[1:] + [n]
    done = 0
    for shot_lo, shot_hi in zip(p.cuts, bounds):
        tracks: dict[int, _Track] = {}
        for w_lo in range(shot_lo, shot_hi, window):
            w_hi = min(shot_hi, w_lo + window)
            dets = keyframes.get(w_lo, [])
            # Match YOLO detections to carried tracks by box IoU against their last mask.
            used = set()
            for tr in tracks.values():
                tb = _box_of(tr.last_mask) if tr.last_mask is not None else None
                best, best_iou = None, 0.3
                for k, (cls, cf, b) in enumerate(dets):
                    if k in used or tb is None:
                        continue
                    iou = _iou_box(tb, np.asarray(b) * to_sam)
                    if iou > best_iou:
                        best, best_iou = k, iou
                if best is not None:
                    used.add(best)
                    tr.unconfirmed, tr.cls, tr.conf = 0, dets[best][0], dets[best][1]
                else:
                    tr.unconfirmed += 1
            # Drop tracks YOLO stopped confirming whose masks have collapsed / vanished.
            for oid in [o for o, tr in tracks.items()
                        if tr.last_mask is None or not tr.last_mask.any()
                        or (tr.unconfirmed >= 2 and tr.last_mask.sum() < 0.0005 * sam_w * sam_h)]:
                tracks.pop(oid)
            # New objects: unmatched detections, largest first, up to max_objects total.
            for k, (cls, cf, b) in sorted(enumerate(dets), key=lambda kv: -((kv[1][2][2] - kv[1][2][0]) * (kv[1][2][3] - kv[1][2][1]))):
                if k in used or len(tracks) >= max_objects:
                    continue
                tracks[next_id] = _Track(next_id, cls, cf)
                tracks[next_id].box_seed = tuple(np.asarray(b) * to_sam)  # type: ignore[attr-defined]
                next_id += 1
            if not tracks:
                done += w_hi - w_lo
                continue

            wdir = tmp / f"w{w_lo:06d}"
            wdir.mkdir()
            for i, t in enumerate(range(w_lo, w_hi)):
                os.link(tmp / f"{t:06d}.jpg", wdir / f"{i:05d}.jpg")
            seeds = [(oid, tr.last_mask, getattr(tr, "box_seed", None)) for oid, tr in tracks.items()]
            for tr in tracks.values():
                if hasattr(tr, "box_seed"):
                    del tr.box_seed  # type: ignore[attr-defined]
            try:
                for fidx, masks in sam.run_window(str(wdir), seeds):
                    t = w_lo + fidx
                    for oid, m in masks.items():
                        if oid not in tracks:
                            continue
                        tracks[oid].last_mask = m
                        if not m.any():
                            continue  # occluded / left the frame on this frame
                        small = cv2.resize(m.astype(np.uint8), (mask_w, mask_h), interpolation=cv2.INTER_AREA)
                        bb = _box_of(m)
                        box = tuple(v / s for v, s in zip(bb, to_sam))
                        tr = tracks[oid]
                        p.frames[t].append(Obj(oid, tr.cls, tr.conf, box, p.add_mask(small > 0)))
            except Exception:
                failures += 1
                if device != "cpu":
                    raise
            shutil.rmtree(wdir, ignore_errors=True)
            done += w_hi - w_lo
            if progress:
                progress("perceive", 0.15 + 0.85 * done / max(1, n))
    stats["sam2_propagation_s"] = round(time.time() - t0, 2)
    stats["sam2_window_failures"] = failures
    stats["device"] = device
    shutil.rmtree(tmp, ignore_errors=True)
    p.seconds = time.time() - t_start
    return p
