"""Dev helper: python scripts/quick_render.py <working_dir> [strength background glow smoothing]
Re-runs perception if missing, then attention + render into <working_dir>/out."""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from pipeline import video_io, perceive, attention, render, enhance
d = Path(sys.argv[1]); a = [float(x) for x in sys.argv[2:]] + [None]*4
info = video_io.info(d / "working.mp4")
if not (d / "perception.json").exists() or json.loads((d/"perception.json").read_text()).get("n_masks") is None:
    p = perceive.run(d / "working.mp4", info); p.save(d)
try:
    p = perceive.Perception.load(d)
except Exception:
    p = perceive.run(d / "working.mp4", info); p.save(d)
params = enhance.Params(*(x for x in a[:3] if x is not None)) if a[0] is not None else enhance.Params()
t = time.time(); att = attention.build(p, smoothing=a[3] if a[3] is not None else 0.6)
print("attention s", round(time.time()-t, 2), [s.to_dict(p.fps) for s in att.segments][:8])
t = time.time(); stats = render.render(d / "working.mp4", info, att, params, d / "out")
print("render s", round(time.time()-t, 1), stats, "frames", info.frame_count)
