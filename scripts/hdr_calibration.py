"""HDR calibration test: does this Mac + Safari show our HDR video correctly?

    python scripts/hdr_calibration.py        # writes backend/calibration/
    then open http://localhost:8000/calibration in Safari (backend running)

Makes two short clips with identical layout, encoded through Keylight's real
SDR and HDR paths:
  - gray steps 0..100% SDR (the "background" test: must look the same in both)
  - patches at 1x, 2x, 3x SDR reference white and 1000 nits (the headroom test:
    in HDR they should step up in brightness; in SDR they are all plain white)

Pass:  gray steps match across the two clips AND the 2x/3x/1000 patches glow
       visibly brighter than the 1x patch and the page's white text.
Fail:  patches all look the same, a clip won't play, or the HDR grays look
       washed out / darker than the SDR grays.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from pipeline import hdr, video_io  # noqa: E402

OUT = ROOT / "backend" / "calibration"
W, H, FPS, SECONDS = 1920, 1080, "30/1", 4
STEPS = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]          # SDR percent
PATCHES = [("1x white", 1.0), ("2x white", 2.0), ("3x white", 3.0), ("4x white", 4.0)]
REFS = [100, 140, 203]  # candidate SDR reference whites (nits)


def layout(ref: float):
    """Return (sdr_bgr_uint8, hdr_rgb_nits) for one calibration frame."""
    sdr = np.full((H, W, 3), 30, np.uint8)  # dark-gray surround (SDR ~12%)
    sw = W // len(STEPS)
    for i, pct in enumerate(STEPS):
        v = int(round(pct / 100 * 255))
        sdr[120:420, i * sw:(i + 1) * sw] = v
    font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
    for i, pct in enumerate(STEPS):
        cv2.putText(sdr, f"{pct}%", (i * sw + 30, 470), font, scale, (200, 200, 200), th, cv2.LINE_AA)
    cv2.putText(sdr, "Gray steps (SDR %): should look the same in both clips", (40, 80), font, 1.1,
                (255, 255, 255), th, cv2.LINE_AA)
    cv2.putText(sdr, "Patches: HDR should get brighter left to right; SDR all the same white", (40, 580),
                font, 1.1, (255, 255, 255), th, cv2.LINE_AA)

    nits = hdr.sdr_to_nits(sdr, ref)  # everything above maps exactly like real video frames
    pw = W // len(PATCHES)
    for i, (label, mult) in enumerate(PATCHES):
        x0, x1 = i * pw + 40, (i + 1) * pw - 40
        sdr[640:940, x0:x1] = 255                      # SDR can only show white
        nits[640:940, x0:x1] = min(hdr.MAX_NITS, mult * ref)
        for img_text in (sdr,):
            cv2.putText(img_text, label, (x0, 1000), font, scale, (200, 200, 200), th, cv2.LINE_AA)
        # labels under the patches in the HDR frame, at SDR-gray level
        lab = np.zeros((H, W, 3), np.uint8)
        cv2.putText(lab, label, (x0, 1000), font, scale, (200, 200, 200), th, cv2.LINE_AA)
        m = lab[..., 0] > 0
        nits[m] = hdr.sdr_to_nits(lab, ref)[m]
    return sdr, nits


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    n = SECONDS * 30
    sdr, _ = layout(REFS[0])
    sw = video_io.FrameWriter(OUT / "calibration_sdr.mp4", W, H, FPS, None, crf=10)
    for _ in range(n):
        sw.write(sdr)
    sw.close()
    encoder = None
    for ref in REFS:
        _, nits = layout(ref)
        rgb48 = hdr.nits_to_rgb48(nits)
        hw = hdr.HDRWriter(OUT / f"calibration_hdr_{ref}.mp4", W, H, FPS, None)
        for _ in range(n):
            hw.write(rgb48)
        hw.close()
        encoder = hw.encoder
    import shutil as _sh
    _sh.copyfile(OUT / f"calibration_hdr_{REFS[-1]}.mp4", OUT / "calibration_hdr.mp4")

    # Verify what we wrote: tags, bit depth, and decoded patch luminance.
    meta = video_io.probe(OUT / "calibration_hdr.mp4")
    v = next(s for s in meta["streams"] if s["codec_type"] == "video")
    tags = {k: v.get(k) for k in ("codec_name", "codec_tag_string", "profile", "pix_fmt",
                                  "color_space", "color_transfer", "color_primaries", "color_range")}
    raw = subprocess.run([video_io.FFMPEG, "-v", "error", "-i", str(OUT / "calibration_hdr.mp4"),
                          "-frames:v", "1", "-vf", "scale=in_color_matrix=bt2020:in_range=tv:out_range=full,format=rgb48le",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    dec = np.frombuffer(raw, "<u2").reshape(H, W, 3).astype(np.float64) / 65535
    pw = W // len(PATCHES)
    measured = {}
    for i, (label, mult) in enumerate(PATCHES):
        sig = dec[740:840, i * pw + 100:(i + 1) * pw - 100, 1].mean()
        measured[label] = {"target_nits": round(min(hdr.MAX_NITS, mult * REFS[-1]), 1),
                           "decoded_nits": round(float(hdr.pq_eotf(sig)), 1)}
    sw_ = W // len(STEPS)
    grays = {f"{p}%": round(float(hdr.pq_eotf(dec[200:350, i * sw_ + 20:(i + 1) * sw_ - 20, 1].mean())), 2)
             for i, p in enumerate(STEPS)}
    report = {"encoder": encoder, "ref_whites_tested": REFS, "stream": tags,
              "patches": measured, "gray_steps_nits": grays}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    (OUT / "index.html").write_text(PAGE)
    print(json.dumps(report, indent=2))
    ok = (tags["color_transfer"] == "smpte2084" and tags["color_primaries"] == "bt2020"
          and "10" in (tags["pix_fmt"] or "") and tags["codec_tag_string"] == "hvc1"
          and all(abs(m["decoded_nits"] - m["target_nits"]) / m["target_nits"] < 0.05 for m in measured.values()))
    print("\nFILE CHECK:", "PASS" if ok else "FAIL", "-> now open http://localhost:8000/calibration in Safari")
    return 0 if ok else 1


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Keylight HDR calibration</title>
<style>
 body{margin:0;background:#101010;color:#ddd;font:16px/1.5 -apple-system,system-ui,sans-serif;padding:24px}
 h1{font-size:22px;margin:0 0 8px} p{margin:4px 0 16px;max-width:80ch}
 .row{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px} figure{margin:0}
 video{width:100%;display:block;background:#000} figcaption{margin-top:6px;color:#aaa}
 .white{display:inline-block;width:80px;height:24px;background:#fff;vertical-align:middle;margin-left:8px}
</style></head><body>
<h1>Keylight HDR calibration</h1>
<p>Safari, auto-brightness off, display brightness around 50&ndash;75%. Compare each HDR clip with the SDR clip:
<b>which HDR clip's gray steps (and the 1x patch) look the same as SDR</b>, not lighter or more washed out, not darker?
In that clip, the 2x&ndash;4x patches should glow brighter than this page's white swatch <span class="white"></span>.</p>
<div class="row">
 <figure><video src="calibration_sdr.mp4" autoplay muted loop playsinline></video><figcaption>SDR reference</figcaption></figure>
 <figure><video src="calibration_hdr_100.mp4" autoplay muted loop playsinline></video><figcaption>HDR, SDR white = 100 nits</figcaption></figure>
</div>
<div class="row">
 <figure><video src="calibration_hdr_140.mp4" autoplay muted loop playsinline></video><figcaption>HDR, SDR white = 140 nits</figcaption></figure>
 <figure><video src="calibration_hdr_203.mp4" autoplay muted loop playsinline></video><figcaption>HDR, SDR white = 203 nits (current default)</figcaption></figure>
</div>
<p>Tell Claude the number of the matching clip, or put <code>KEYLIGHT_HDR_REF_WHITE=&lt;number&gt;</code> in <code>.env</code>
and restart. Screen reports HDR: <span id="hdr"></span></p>
<script>document.getElementById('hdr').textContent = matchMedia('(dynamic-range: high)').matches ? 'yes' : 'no'</script>
</body></html>
"""

if __name__ == "__main__":
    sys.exit(main())
