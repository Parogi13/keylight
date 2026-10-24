"""Orchestrates the three passes for one job directory.

    job_dir/
      input.<ext>        the uploaded file (or a referenced local path)
      working.mp4        constant-frame-rate working copy
      perception.json    pass 1 cache (objects, cuts, luma)
      masks.npz          pass 1 cache (packed object masks)
      semantic.json      pass 2 Claude decisions, keyed by focus instruction
      original.mp4       browser-ready original  \
      enhanced.mp4       enhanced video            > same frames, same encoder
      attention.mp4      "Show AI attention" view /
      metadata.json      timings, focus timeline, settings

Re-rendering with new settings reuses the perception cache, so slider changes
cost one render pass rather than a full re-analysis.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from . import attention, enhance, perceive, render, understand, video_io

# User-facing stage names (no jargon on screen).
STAGES = {
    "prepare": "Preparing video",
    "perceive": "Detecting subjects",
    "understand": "Understanding the scene",
    "attention": "Tracking attention",
    "render": "Enhancing frames",
    "finalize": "Finalizing",
}
# Rough share of total time per stage, for a single smooth progress bar.
WEIGHTS_FULL = {"prepare": 0.05, "perceive": 0.45, "understand": 0.08, "attention": 0.02, "render": 0.38, "finalize": 0.02}
WEIGHTS_RERENDER = {"understand": 0.15, "attention": 0.03, "render": 0.80, "finalize": 0.02}

Progress = Callable[[str, float, str], None]  # (stage, overall 0..1, label)


def _tracker(weights: dict, cb: Progress | None):
    order = list(weights)

    def report(stage: str, frac: float):
        if cb is None:
            return
        done = sum(weights[s] for s in order[: order.index(stage)])
        cb(stage, min(0.999, done + weights[stage] * max(0.0, min(1.0, frac))), STAGES[stage])
    return report


def _semantic(job: Path, p: perceive.Perception, focus: str | None, use_claude: bool, report) -> tuple[dict, dict]:
    if not use_claude:
        return {}, {"enabled": False, "error": "disabled"}
    cache_path = job / "semantic.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    key = (focus or "").strip().lower()
    if key in cache and cache[key]["stats"].get("ok"):
        return {int(k): v for k, v in cache[key]["decisions"].items()}, {**cache[key]["stats"], "cached": True}
    decisions, stats = understand.run(job / "working.mp4", p, focus or None,
                                      progress=lambda s, f: report("understand", f))
    if stats.get("ok"):
        cache[key] = {"decisions": {str(k): v for k, v in decisions.items()}, "stats": stats}
        cache_path.write_text(json.dumps(cache))
    return decisions, stats


def _finish(job: Path, info, p, att, params, settings, timings, sem_stats, render_stats) -> dict:
    meta = {
        "video": info.to_dict(),
        "settings": settings,
        "params": params.__dict__,
        "device": p.device,
        "shots": len(p.cuts),
        "cuts_s": [round(c / p.fps, 3) for c in p.cuts],
        "focus_timeline": [s.to_dict(p.fps) for s in att.segments],
        "semantic": sem_stats,
        "timings_s": {k: round(v, 2) for k, v in timings.items()},
        "render": render_stats,
        "realtime_factor": round(sum(timings.values()) / max(info.duration, 1e-6), 2),
    }
    (job / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def _perceive(job: Path, info, settings: dict, report) -> "perceive.Perception":
    """segmentation_backend: 'yolo' (default, YOLO11n-seg + ByteTrack every frame) or
    'sam2' (experimental: YOLO keyframes ~1/s + SAM 2 mask propagation). Falls back
    to YOLO if SAM 2 isn't installed."""
    backend = str(settings.get("segmentation_backend", "yolo")).lower()
    if backend == "sam2":
        from . import sam2_perceive
        if sam2_perceive.available():
            stats: dict = {}
            p = sam2_perceive.run(job / "working.mp4", info, progress=report,
                                  yolo_interval_s=float(settings.get("yolo_interval_s", 1.0)), stats=stats)
            (job / "sam2_stats.json").write_text(json.dumps(stats, indent=2))
            return p
    return perceive.run(job / "working.mp4", info, progress=report)


def process(job: Path, src: Path, settings: dict, cb: Progress | None = None) -> dict:
    report = _tracker(WEIGHTS_FULL, cb)
    timings = {}

    t = time.time(); report("prepare", 0)
    info = video_io.normalize(src, job / "working.mp4")
    timings["prepare"] = time.time() - t
    if src.parent == job:
        src.unlink(missing_ok=True)  # the working copy replaces the upload: halve disk use

    t = time.time()
    p = _perceive(job, info, settings, report)
    p.save(job)
    timings["perceive"] = time.time() - t

    return _decide_and_render(job, info, p, settings, report, timings, write_original=True)


def rerender(job: Path, settings: dict, cb: Progress | None = None) -> dict:
    report = _tracker(WEIGHTS_RERENDER, cb)
    info = video_io.info(job / "working.mp4")
    p = perceive.Perception.load(job)
    return _decide_and_render(job, info, p, settings, report, {}, write_original=False)


def _decide_and_render(job, info, p, settings, report, timings, write_original) -> dict:
    t = time.time(); report("understand", 0)
    decisions, sem_stats = _semantic(job, p, settings.get("focus"), bool(settings.get("use_claude", True)), report)
    timings["understand"] = time.time() - t

    t = time.time(); report("attention", 0)
    att = attention.build(p, smoothing=float(settings.get("smoothing", 0.6)), semantic=decisions)
    timings["attention"] = time.time() - t

    t = time.time()
    params = enhance.Params.from_dict(settings)
    render_stats = render.render(job / "working.mp4", info, att, params, job,
                                 progress=report, write_original=write_original or not (job / "original.mp4").exists())
    timings["render"] = time.time() - t

    report("finalize", 0.5)
    return _finish(job, info, p, att, params, settings, timings, sem_stats, render_stats)
