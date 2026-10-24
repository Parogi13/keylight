"""Benchmark segmentation backends on identical clips: YOLO (A) vs YOLO + SAM 2 (B).

    python scripts/bench_segmentation.py clip.mp4 [more clips...] [--out bench/]

For each clip, each backend runs in its *own process* (so peak memory is that
backend's alone) on the same normalized working copy. Reported per backend:
  time / effective FPS / peak RSS                       (cost)
  temporal stability: IoU of the followed subject's mask between consecutive
      frames (mean, 5th percentile) and the share of "jumps" (IoU < 0.8)
  tracking gaps: frames inside the subject's lifetime where it has no mask
  edge alignment: image gradient strength along the mask boundary relative to the
      frame average (>1 = boundary sits on real edges; higher is tighter)
and, between backends, the per-frame IoU of the followed subject's masks.
It also writes a side-by-side mask overlay video (A | B) and the enhanced SDR
render for each backend, for eyeballing artifacts. No ground truth is used, so
these are proxies; look at the videos before concluding anything.
"""
from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from pipeline import attention, perceive, video_io  # noqa: E402


def run_backend(backend: str, working: Path, out: Path) -> dict:
    """Child-process entry: perceive with one backend, save, report cost."""
    info = video_io.info(working)
    stats: dict = {}
    t = time.time()
    if backend == "sam2":
        from pipeline import sam2_perceive
        p = sam2_perceive.run(working, info, stats=stats)
    else:
        p = perceive.run(working, info)
    dt = time.time() - t
    p.save(out)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)
    return {"backend": backend, "seconds": round(dt, 2), "frames": len(p.frames),
            "fps": round(len(p.frames) / dt, 2), "peak_rss_mb": round(peak_mb), "device": p.device, **stats}


def quality(p: perceive.Perception, frames_dir: Path | None, working: Path) -> dict:
    att = attention.build(p)
    target = att.raw
    ious, gaps, span = [], 0, 0
    edge = []
    prev = None
    grads = None
    it = video_io.iter_frames(working)
    for t in range(len(p.frames)):
        frame = next(it)
        tid = target[t]
        o = next((o for o in p.frames[t] if o.id == tid), None) if tid is not None else None
        if tid is not None:
            span += 1
            if o is None:
                gaps += 1
        m = p.mask(o.mask) > 0.5 if o is not None else None
        if m is not None and prev is not None and t not in p.cuts:
            u = (m | prev).sum()
            if u:
                ious.append((m & prev).sum() / u)
        prev = m
        if m is not None and t % 5 == 0:
            g = cv2.cvtColor(cv2.resize(frame, (p.mask_w, p.mask_h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            g = g.astype(np.float32)
            grads = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
            ring = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
            if ring.any() and grads.mean() > 0:
                edge.append(float(grads[ring].mean() / grads.mean()))
    ious_a = np.array(ious) if ious else np.array([np.nan])
    return {"stability_mean_iou": round(float(np.nanmean(ious_a)), 3),
            "stability_p5_iou": round(float(np.nanpercentile(ious_a, 5)), 3),
            "jumps_pct": round(100 * float(np.mean(ious_a < 0.8)), 1) if ious else None,
            "tracking_gap_frames": gaps, "subject_frames": span,
            "edge_alignment": round(float(np.mean(edge)), 2) if edge else None}


def followed_masks(p: perceive.Perception) -> list:
    att = attention.build(p)
    out = []
    for t in range(len(p.frames)):
        o = next((o for o in p.frames[t] if o.id == att.raw[t]), None)
        out.append(p.mask(o.mask) > 0.5 if o is not None else None)
    return out


def overlay_video(working: Path, a: perceive.Perception, b: perceive.Perception, dst: Path) -> None:
    info = video_io.info(working)
    ma, mb = followed_masks(a), followed_masks(b)
    W = 960
    H = int(round(info.height * W / info.width / 2)) * 2
    w = video_io.FrameWriter(dst, W * 2, H, info.fps_fraction, None, crf=20)

    def draw(frame, m, label, col):
        f = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        if m is not None:
            mm = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            f[mm] = (0.55 * f[mm] + 0.45 * np.array(col)).astype(np.uint8)
            cnt, _ = cv2.findContours(mm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(f, cnt, -1, col, 2)
        cv2.putText(f, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        return f

    for t, frame in enumerate(video_io.iter_frames(working)):
        if t >= len(ma):
            break
        w.write(np.hstack([draw(frame, ma[t], "A: YOLO11n-seg + ByteTrack", (255, 140, 40)),
                           draw(frame, mb[t], "B: YOLO keyframes + SAM 2.1 tiny", (40, 200, 255))]))
    w.close()


def main() -> int:
    if len(sys.argv) >= 4 and sys.argv[1] == "--child":
        print(json.dumps(run_backend(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]))))
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--out", default=str(ROOT / "bench"))
    ap.add_argument("--render", action="store_true", help="also render enhanced SDR video per backend")
    a = ap.parse_args()
    report = []
    for clip in map(Path, a.clips):
        d = Path(a.out) / clip.stem
        d.mkdir(parents=True, exist_ok=True)
        working = d / "working.mp4"
        if not working.exists():
            video_io.normalize(clip, working)
        row = {"clip": clip.name, "video": video_io.info(working).to_dict()}
        for backend in ("yolo", "sam2"):
            out = d / backend
            out.mkdir(exist_ok=True)
            res = subprocess.run([sys.executable, __file__, "--child", backend, str(working), str(out)],
                                 capture_output=True, text=True)
            if res.returncode != 0:
                row[backend] = {"error": res.stderr[-1500:]}
                print(f"[{clip.name}] {backend} FAILED\n{res.stderr[-1500:]}")
                continue
            cost = json.loads(res.stdout.strip().splitlines()[-1])
            p = perceive.Perception.load(out)
            row[backend] = {**cost, **quality(p, None, working)}
            print(f"[{clip.name}] {backend}: {row[backend]}")
            if a.render:
                from pipeline import enhance, render
                att = attention.build(p)
                render.render(working, video_io.info(working), att, enhance.Params(hdr=False), out)
        if "error" not in row.get("yolo", {}) and "error" not in row.get("sam2", {}):
            pa, pb = perceive.Perception.load(d / "yolo"), perceive.Perception.load(d / "sam2")
            ma, mb = followed_masks(pa), followed_masks(pb)
            agree = [float((x & y).sum() / max((x | y).sum(), 1)) for x, y in zip(ma, mb) if x is not None and y is not None]
            row["subject_iou_A_vs_B"] = round(float(np.mean(agree)), 3) if agree else None
            overlay_video(working, pa, pb, d / "masks_A_vs_B.mp4")
        report.append(row)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
