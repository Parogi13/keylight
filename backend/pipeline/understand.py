"""Comprehension (optional): which detected object matters to the viewer right now?

YOLO answers "where are the people?". Claude answers "which person matters?".

- Runs on keyframes only: the first frame of every shot, then about once per
  second, never per frame.
- Each keyframe is sent with every tracked object's box and ID drawn on the image
  ("set-of-mark" prompting), plus the user's focus instruction and the previous
  choice. Claude must pick one of the IDs it was shown; anything else is rejected.
- Only downscaled keyframes leave the machine. The video itself never does.
- Every failure path (no key, no network, timeout, bad JSON, unknown ID) returns
  nothing for that keyframe, and attention.py falls back to the local heuristic.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import video_io
from .perceive import Perception, SALIENT_ID

DEFAULT_MODEL = os.environ.get("KEYLIGHT_CLAUDE_MODEL", "claude-sonnet-5-5")
MAX_CALLS = int(os.environ.get("KEYLIGHT_MAX_CLAUDE_CALLS", "60"))
BUDGET_S = float(os.environ.get("KEYLIGHT_CLAUDE_BUDGET_S", "60"))  # wall-clock cap per video
IMAGE_LONG_SIDE = 768

SYSTEM = (
    "You help people with low vision follow video. You are shown one video frame. "
    "Every candidate subject is outlined with a numbered label like '#3 person'. "
    "Choose the ONE candidate the viewer most needs to see to follow what is happening "
    "(the main character, whoever is acting or speaking, the object the scene is about). "
    "If the viewer gave a focus instruction, follow it whenever a candidate plausibly matches. "
    "Only choose from the numbered candidates. Respond with JSON only: "
    '{"target": <id number or null>, "confidence": <0..1>, "reason": "<one short sentence>"}'
)


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def keyframes(p: Perception, interval_s: float = 1.0, max_calls: int = MAX_CALLS) -> list[int]:
    """Shot starts + roughly every `interval_s`, skipping frames with fewer than two
    real candidates (nothing to choose between)."""
    n = len(p.frames)
    duration = n / p.fps
    interval_s = max(interval_s, duration / max(1, max_calls))
    step = max(1, int(round(p.fps * interval_s)))
    picks = []
    for lo, hi in p.shot_bounds():
        t = min(hi - 1, lo + max(1, int(p.fps * 0.2)))  # just after the cut, tracker has settled
        while t < hi:
            picks.append(t)
            t += step
    return picks


MAX_CANDIDATES = 12


def _real(objs):
    """Detected objects worth asking about: no saliency blobs, and in a crowd only the
    largest dozen (more labels than that make the image unreadable for the model)."""
    real = [o for o in objs if o.id != SALIENT_ID]
    if len(real) > MAX_CANDIDATES:
        real = sorted(real, key=lambda o: -(o.box[2] - o.box[0]) * (o.box[3] - o.box[1]))[:MAX_CANDIDATES]
    return real


def _draw_marks(frame: np.ndarray, objs) -> np.ndarray:
    h, w = frame.shape[:2]
    s = IMAGE_LONG_SIDE / max(h, w)
    img = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else frame.copy()
    s = min(s, 1.0)
    # Brighten dark frames so the model can actually see (only the copy we send).
    if img.mean() < 70:
        lut = (np.power(np.arange(256) / 255.0, 0.55) * 255).astype(np.uint8)
        img = cv2.LUT(img, lut)
    palette = [(0, 200, 255), (255, 120, 0), (80, 220, 80), (220, 80, 255), (0, 80, 255), (255, 255, 0)]
    for i, o in enumerate(objs):
        c = palette[i % len(palette)]
        x1, y1, x2, y2 = [int(v * s) for v in o.box]
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        label = f"#{o.id} {o.cls}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = max(th + 4, y1 - 4)
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 6, ty + 3), c, -1)
        cv2.putText(img, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def _ask(client, model: str, img: np.ndarray, objs, focus: str | None, previous: str | None,
         t_sec: float) -> dict | None:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None
    cands = "\n".join(f"#{o.id}: {o.cls}, box={[int(v) for v in o.box]}" for o in objs)
    text = (
        f"Time in video: {t_sec:.1f}s\nCandidates:\n{cands}\n"
        f"Previous focus: {previous or 'none'}\n"
        f"Viewer focus instruction: {focus or 'none (choose what matters most)'}"
    )
    msg = client.messages.create(
        model=model,
        max_tokens=200,
        system=SYSTEM,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                          "data": base64.b64encode(buf.tobytes()).decode()}},
            {"type": "text", "text": text},
        ]}],
    )
    raw = "".join(getattr(b, "text", "") for b in msg.content)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    data = json.loads(m.group(0))
    target = data.get("target")
    ids = {o.id for o in objs}
    if target is not None:
        try:
            target = int(str(target).lstrip("#"))
        except ValueError:
            return None
        if target not in ids:
            return None  # the model may only choose what local perception found
    return {"target": target, "confidence": float(data.get("confidence", 0.5)),
            "reason": str(data.get("reason", ""))[:200]}


def run(working: Path, p: Perception, focus: str | None = None,
        progress: Callable[[str, float], None] | None = None,
        model: str = DEFAULT_MODEL, interval_s: float = 1.0) -> tuple[dict[int, dict], dict]:
    """Returns ({keyframe: {target, confidence, reason}}, stats). Never raises."""
    stats = {"enabled": False, "model": model, "calls": 0, "ok": 0, "failed": 0,
             "seconds": 0.0, "error": None, "focus": focus}
    if not available():
        stats["error"] = "ANTHROPIC_API_KEY not set — using local heuristic"
        return {}, stats
    try:
        import anthropic
        client = anthropic.Anthropic(timeout=15.0, max_retries=1)
    except Exception as e:  # pragma: no cover
        stats["error"] = f"anthropic SDK unavailable: {e}"
        return {}, stats

    stats["enabled"] = True
    t0 = time.time()
    kfs = keyframes(p, interval_s)
    # With no instruction, a keyframe with a single candidate has nothing to decide.
    kfs = [t for t in kfs if len(_real(p.frames[t])) >= (1 if focus else 2)]
    if not kfs:
        stats["seconds"] = round(time.time() - t0, 2)
        return {}, stats
    want = set(kfs)
    images: dict[int, np.ndarray] = {}
    for t, frame in enumerate(video_io.iter_frames(working)):
        if t in want:
            images[t] = _draw_marks(frame, _real(p.frames[t]))
        if t >= max(kfs):
            break

    results: dict[int, dict] = {}

    def job(t):
        return t, _ask(client, model, images[t], _real(p.frames[t]), focus, None, t / p.fps)

    def record(fut_result=None, err: Exception | None = None):
        stats["calls"] += 1
        if err is not None:
            stats["failed"] += 1
            stats["error"] = f"{type(err).__name__}: {str(err)[:200]}"
            return
        t, r = fut_result
        if r is not None:
            results[t] = r
            stats["ok"] += 1
        else:
            stats["failed"] += 1

    calls_start = time.time()
    # Probe with the first keyframe. If Claude is unreachable (venue Wi-Fi, bad key),
    # stop here instead of waiting on dozens of timeouts: the demo must not stall.
    try:
        record(job(kfs[0]))
    except Exception as e:
        record(err=e)
        stats["error"] = f"Claude unreachable, using local choice ({type(e).__name__})"
        stats["seconds"] = round(time.time() - t0, 2)
        return results, stats
    if progress:
        progress("understand", 1 / len(kfs))

    rest = kfs[1:]
    deadline = calls_start + BUDGET_S
    ex = ThreadPoolExecutor(max_workers=6)
    try:
        futures = {ex.submit(job, t): t for t in rest}
        pending = set(futures)
        while pending:
            done_set, pending = wait(pending, timeout=max(0.0, deadline - time.time()), return_when=FIRST_COMPLETED)
            if not done_set:  # budget exhausted: keep what we have, heuristic covers the rest
                stats["error"] = f"time budget reached after {stats['calls']} of {len(kfs)} frames"
                break
            for f in done_set:
                try:
                    record(f.result())
                except Exception as e:
                    record(err=e)
                if progress:
                    progress("understand", stats["calls"] / len(kfs))
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    stats["seconds"] = round(time.time() - t0, 2)
    return results, stats
