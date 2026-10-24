"""Pass 2 — Temporal attention: which object gets the light, and how does focus move?

1. Target selection (per shot). Each frame gets a raw target ID, chosen by:
   a. a semantic decision (Claude) for that moment, if one exists and that object
      is visible, otherwise
   b. a local heuristic score: size, confidence, class prior, centrality and how
      persistently the track appears in the shot.
   Switching uses hysteresis + a minimum dwell so focus does not ping-pong.

2. Weight smoothing. Each object ID gets a scalar weight w_i(t): 1 when it is the
   target, 0 otherwise. Those scalars are smoothed with a symmetric window that
   looks up to ~1 s into the future, so a new subject starts brightening *before*
   focus arrives and the old one fades out after. The window never crosses a scene
   cut. We smooth weights, not masks: averaging masks across frames would make the
   highlight trail behind moving subjects.

3. Mask composition. M(t) = clip(sum_i w_i(t) * mask_i(t)), using each object's
   mask *in that frame*, plus a tiny 3-tap temporal filter for edge shimmer and a
   spatial feather so there are no cut-out edges.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field

import cv2
import numpy as np

from .perceive import SALIENCY_FALLBACK, SALIENT_ID, Perception

SALIENT_WEIGHT = 0.5

CLASS_PRIOR = {
    "person": 1.0,
    "sports ball": 1.1,
    "car": 0.9, "truck": 0.85, "bus": 0.85, "motorcycle": 0.95, "bicycle": 0.9,
    "train": 0.8, "airplane": 0.9, "boat": 0.85,
    "dog": 0.95, "cat": 0.95, "horse": 0.95, "bird": 0.85, "cow": 0.85, "sheep": 0.85,
    "elephant": 0.9, "bear": 0.9, "zebra": 0.9, "giraffe": 0.9,
    "salient region": 0.25,
}
DEFAULT_PRIOR = 0.45


@dataclass
class Segment:
    start: int
    end: int  # exclusive
    target: int | None
    cls: str | None
    source: str  # "semantic" | "heuristic" | "none"
    reason: str = ""

    def to_dict(self, fps: float) -> dict:
        return {"start": self.start, "end": self.end,
                "t0": round(self.start / fps, 3), "t1": round(self.end / fps, 3),
                "target": self.target, "cls": self.cls, "source": self.source, "reason": self.reason}


@dataclass
class Attention:
    p: Perception
    weights: list[dict[int, float]]          # per frame: object id -> weight
    raw: list[int | None]                     # per frame raw target
    segments: list[Segment]
    feather: float
    _index: dict = field(default_factory=dict)  # id -> sorted frames where id present
    # Small per-instance LRU of composed raw masks (the 3-tap filter reuses each one
    # three times). Bounded so long videos don't grow memory; locked for render threads.
    _cache: OrderedDict = field(default_factory=OrderedDict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    CACHE_SIZE = 48

    # ---- masks ---------------------------------------------------------------
    def _nearest_mask(self, t: int, oid: int) -> np.ndarray | None:
        for o in self.p.frames[t]:
            if o.id == oid:
                return self.p.mask(o.mask)
        frames = self._index.get(oid)
        if not frames:
            return None
        lo, hi = _shot_range(self.p, t)
        i = int(np.searchsorted(frames, t))
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(frames) and lo <= frames[j] < hi:
                if best is None or abs(frames[j] - t) < abs(best - t):
                    best = frames[j]
        if best is None or abs(best - t) > self.p.fps * 1.5:
            return None
        o = next(o for o in self.p.frames[best] if o.id == oid)
        return self.p.mask(o.mask)

    def _raw_mask(self, t: int) -> np.ndarray:
        with self._lock:
            hit = self._cache.get(t)
            if hit is not None:
                self._cache.move_to_end(t)
                return hit
        m = np.zeros((self.p.mask_h, self.p.mask_w), np.float32)
        for oid, w in self.weights[t].items():
            if w < 1e-3:
                continue
            mk = self._nearest_mask(t, oid)
            if mk is not None:
                m += w * mk
        np.clip(m, 0, 1, out=m)
        with self._lock:
            self._cache[t] = m
            while len(self._cache) > self.CACHE_SIZE:
                self._cache.popitem(last=False)
        return m

    def mask(self, t: int) -> np.ndarray:
        """Feathered low-res attention mask for frame t, float32 in [0, 1]."""
        lo, hi = _shot_range(self.p, t)
        acc = 0.5 * self._raw_mask(t)
        for dt in (-1, 1):
            u = t + dt
            acc += 0.25 * self._raw_mask(u if lo <= u < hi else t)
        # Feather: grow slightly, then blur, so the edge falls off outside the subject.
        diag = float(np.hypot(self.p.mask_w, self.p.mask_h))
        # Feather scales with the subject: a small car gets a tight edge, a close-up a soft one.
        size = float(np.sqrt(max(acc.sum(), 1.0)))
        r = int(min(diag * 0.006, 0.04 * size))
        if r >= 1:
            acc = cv2.dilate(acc, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
        sigma = float(np.clip(0.12 * size, 0.8, diag * self.feather))
        acc = cv2.GaussianBlur(acc, (0, 0), sigma)
        return np.clip(acc, 0, 1)


def _shot_range(p: Perception, t: int) -> tuple[int, int]:
    i = int(np.searchsorted(p.cuts, t, side="right") - 1)
    lo = p.cuts[i]
    hi = p.cuts[i + 1] if i + 1 < len(p.cuts) else len(p.frames)
    return lo, hi


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------
def _scores(p: Perception, lo: int, hi: int) -> list[dict[int, float]]:
    n = hi - lo
    presence: dict[int, int] = {}
    for t in range(lo, hi):
        for o in p.frames[t]:
            presence[o.id] = presence.get(o.id, 0) + 1
    area_full = p.width * p.height
    cx0, cy0 = p.width / 2, p.height / 2
    half_diag = np.hypot(cx0, cy0)
    out = []
    for t in range(lo, hi):
        s = {}
        for o in p.frames[t]:
            x1, y1, x2, y2 = o.box
            area = max(1.0, (x2 - x1) * (y2 - y1)) / area_full
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            central = 1.0 - 0.5 * min(1.0, np.hypot(cx - cx0, cy - cy0) / half_diag)
            persist = 0.5 + 0.5 * presence[o.id] / n
            prior = CLASS_PRIOR.get(o.cls, DEFAULT_PRIOR)
            # Huge boxes (>60% of frame) are usually background-ish; damp them.
            size = np.sqrt(area) * (0.6 if area > 0.6 else 1.0)
            s[o.id] = float(size * (0.5 + 0.5 * o.conf) * prior * central * persist)
        out.append(s)
    return out


def select_targets(p: Perception, semantic: dict[int, dict] | None = None,
                   dwell_s: float = 0.6, hold_s: float = 0.5) -> tuple[list[int | None], list[str], list[str]]:
    """Return per-frame (target id, source, reason)."""
    semantic = semantic or {}
    sem_frames = sorted(semantic)
    raw: list[int | None] = [None] * len(p.frames)
    src = ["none"] * len(p.frames)
    why = [""] * len(p.frames)
    dwell = max(1, int(p.fps * dwell_s))
    hold = max(1, int(p.fps * hold_s))

    for lo, hi in p.shot_bounds():
        scores = _scores(p, lo, hi)
        shot_sem = [f for f in sem_frames if lo <= f < hi]
        cur: int | None = None
        missing = 0
        challenger, chal_count = None, 0
        for t in range(lo, hi):
            s = scores[t - lo]
            # Semantic preference: the most recent decision in this shot. Before the
            # shot's first decision, use that first decision (we have lookahead), so
            # the shot does not start on the heuristic subject and then swing over.
            sem = None
            if shot_sem:
                i = int(np.searchsorted(shot_sem, t, side="right") - 1)
                sem = semantic[shot_sem[max(i, 0)]]
            pref = sem.get("target") if sem else None

            best = max(s, key=s.get) if s else None
            if pref is not None and pref in s:
                if cur != pref:
                    challenger, chal_count = None, 0
                cur, missing = pref, 0
            elif cur is None:
                cur = best
                missing = 0
            elif cur not in s:
                missing += 1  # detector dropout: hold the current subject briefly
                if missing > hold:
                    cur, missing = best, 0
            elif cur != pref:
                missing = 0
                if best is not None and best != cur and s[best] > 1.6 * s[cur]:
                    if challenger == best:
                        chal_count += 1
                    else:
                        challenger, chal_count = best, 1
                    if chal_count >= dwell:
                        cur, challenger, chal_count = best, None, 0
                else:
                    challenger, chal_count = None, 0
            raw[t] = cur
            if cur is None:
                src[t] = "none"
            elif pref is not None and cur == pref:
                src[t], why[t] = "semantic", sem.get("reason", "")
            else:
                src[t] = "heuristic"
    return raw, src, why


# ---------------------------------------------------------------------------
def build(p: Perception, smoothing: float = 0.6, semantic: dict[int, dict] | None = None,
          feather: float = 0.025) -> Attention:
    if not SALIENCY_FALLBACK:
        # Perception caches from older runs may still contain saliency blobs; with
        # the fallback off they must never be highlighted (no subject = untouched).
        p.frames = [[o for o in objs if o.id != SALIENT_ID] for objs in p.frames]
    raw, src, why = select_targets(p, semantic)

    # Half-window of 0.15 s .. 1.0 s. Lookahead never exceeds one second.
    half = max(1, int(round(p.fps * (0.15 + 0.85 * float(np.clip(smoothing, 0, 1))))))
    kernel = np.hanning(2 * half + 3)[1:-1]  # strictly positive taps

    ids = sorted({i for i in raw if i is not None})
    weights: list[dict[int, float]] = [dict() for _ in p.frames]
    for lo, hi in p.shot_bounds():
        n = hi - lo
        for oid in ids:
            onehot = np.array([1.0 if raw[t] == oid else 0.0 for t in range(lo, hi)])
            if not onehot.any():
                continue
            # 'full' then centre-crop: np.convolve(mode="same") returns len(kernel)
            # samples when the shot is shorter than the kernel.
            num = np.convolve(onehot, kernel, mode="full")[half:half + n]
            den = np.convolve(np.ones(n), kernel, mode="full")[half:half + n]  # truncate at cuts, no fade-from-black
            w = num / den
            if oid == SALIENT_ID:
                w = w * SALIENT_WEIGHT  # no real subject: stay gentle
            for k in np.nonzero(w > 1e-3)[0]:
                weights[lo + k][oid] = float(w[k])

    # Segments for the UI / metadata. Track IDs restart after a cut, so class names
    # are looked up within the segment's own frames.
    segments: list[Segment] = []
    cut_set = set(p.cuts)
    for t in range(len(p.frames)):
        cls = next((o.cls for o in p.frames[t] if o.id == raw[t]), None)
        if not segments or t in cut_set or segments[-1].target != raw[t]:
            segments.append(Segment(t, t + 1, raw[t], cls, src[t], why[t]))
        else:
            segments[-1].end = t + 1
            segments[-1].cls = segments[-1].cls or cls
            if src[t] == "semantic":
                segments[-1].source = "semantic"
            if why[t] and not segments[-1].reason:
                segments[-1].reason = why[t]

    att = Attention(p, weights, raw, segments, feather)
    index: dict[int, list[int]] = {}
    for t, objs in enumerate(p.frames):
        for o in objs:
            index.setdefault(o.id, []).append(t)
    att._index = index
    return att
