"""Dev helper: render a few frames to a stacked PNG (original over enhanced) for tuning.
python scripts/preview_frames.py <working_dir> <out.png> t1,t2 [strength background glow]"""
import sys, time
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from pipeline import video_io, perceive, attention, enhance
d = Path(sys.argv[1]); out = sys.argv[2]; ts = [int(x) for x in sys.argv[3].split(",")]
vals = [float(x) for x in sys.argv[4:]]
params = enhance.Params(*vals)
p = perceive.Perception.load(d); att = attention.build(p)
rows = []; times = []
for t, f in enumerate(video_io.iter_frames(d / "working.mp4")):
    if t in ts:
        t0 = time.time(); m = cv2.resize(att.mask(t), (f.shape[1], f.shape[0])); e = enhance.enhance(f, m, params); times.append(time.time() - t0)
        rows.append(np.hstack([f, e]))
    if t > max(ts): break
img = np.vstack(rows); s = 1600 / img.shape[1]
cv2.imwrite(out, cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
print("ms/frame", [round(x * 1000) for x in times])
