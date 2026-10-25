"""Render one clip twice with two different focus instructions, for the Claude demo.

    python scripts/focus_pair.py clip.mp4 "the cyclist" "the white van"
    python scripts/focus_pair.py clip.mp4 "the cyclist" "the white van" --sam2

Detection runs once; the second version reuses it, so only Claude's choice and the
render differ. Both appear in the web app ("Your videos", after a backend restart)
and in the native player's library, named "<clip> · follow <instruction>".

Claude runs here, during processing, not during playback. In the demo, switch
between the two finished versions. Needs ANTHROPIC_API_KEY in .env; the script
stops if Claude didn't run, rather than producing two identical videos.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from main import DEFAULT_SETTINGS, OUTPUTS  # noqa: E402  (also loads .env)
from pipeline import run as pipeline  # noqa: E402

SHARED = ("working.mp4", "perception.json", "masks.npz", "sam2_stats.json")


def _job(name: str, settings: dict) -> Path:
    d = OUTPUTS / uuid.uuid4().hex[:10]
    d.mkdir(parents=True)
    (d / "job.json").write_text(json.dumps({"name": name, "settings": settings, "created": time.time()}))
    return d


def _report(meta: dict, focus: str) -> None:
    sem = meta.get("semantic", {})
    picks = [s for s in meta.get("focus_timeline", []) if s.get("target") is not None]
    print(f'  "{focus}": Claude {"ok" if sem.get("ok") else "NOT used"} '
          f'({sem.get("ok", 0)}/{sem.get("calls", 0)} calls), {len(picks)} focus segments')
    for s in picks[:6]:
        print(f"    {s}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", type=Path)
    ap.add_argument("focus_a")
    ap.add_argument("focus_b")
    ap.add_argument("--sam2", action="store_true", help="segment with SAM 2 (slow, tighter masks)")
    a = ap.parse_args()
    if not a.clip.exists():
        print(f"not found: {a.clip}")
        return 1

    base = {**DEFAULT_SETTINGS, "use_claude": True,
            "segmentation_backend": "sam2" if a.sam2 else DEFAULT_SETTINGS["segmentation_backend"]}
    title = a.clip.stem.replace("-", " ").replace("_", " ").capitalize()
    progress = lambda stage, f, label: print(f"\r  {label:<26} {f * 100:5.1f}%", end="", flush=True)

    sa = {**base, "focus": a.focus_a}
    ja = _job(f"{title} · follow {a.focus_a}", sa)
    print(f"[1/2] {ja.name}: full pass")
    shutil.copyfile(a.clip, ja / f"input{a.clip.suffix.lower()}")
    meta_a = pipeline.process(ja, ja / f"input{a.clip.suffix.lower()}", sa, progress)
    print()
    _report(meta_a, a.focus_a)

    sb = {**base, "focus": a.focus_b}
    jb = _job(f"{title} · follow {a.focus_b}", sb)
    for f in SHARED:
        if (ja / f).exists():
            shutil.copyfile(ja / f, jb / f)
    print(f"[2/2] {jb.name}: reusing detection, new focus")
    meta_b = pipeline.rerender(jb, sb, progress)
    print()
    _report(meta_b, a.focus_b)

    if not (meta_a.get("semantic", {}).get("ok") and meta_b.get("semantic", {}).get("ok")):
        print("\nClaude did not run for both versions (check ANTHROPIC_API_KEY and network). "
              "Both videos used the local heuristic, so they may be identical.")
        return 2
    ta = [s.get("target") for s in meta_a["focus_timeline"]]
    tb = [s.get("target") for s in meta_b["focus_timeline"]]
    if ta == tb:
        print("\nBoth instructions led to the same subjects. Try instructions that name clearly "
              "different objects that are both visible in the clip.")
        return 3
    print(f"\nDone. Two versions with different subjects:\n  {ja}\n  {jb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
