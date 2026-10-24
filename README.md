# Keylight

**Make what matters easier to see.**

Keylight is a local-first accessibility layer for video. It finds the subject of each scene and lights it the way a cinematographer's *key light* would: the subject lifts, and the background settles back. It's built for people with low vision, especially those with reduced contrast sensitivity, who can see the screen but lose track of the action in dark, low-contrast, or busy scenes.

> Brightness and contrast controls change the whole frame, so the background gets brighter along with the subject and the scene washes out. Keylight increases the *perceptual separation* of what matters instead.

It runs on ordinary SDR screens because it works with how the eye judges brightness *relative to surroundings*. No HDR or OLED panel is required.

## How it works

```
 video ─► PERCEPTION ─────► COMPREHENSION ─────► TEMPORAL ATTENTION ─────► ADAPTATION ─► enhanced.mp4
          what exists?      what matters?        how should focus move?    make it easier to see
          YOLO11-seg        Claude (optional)    per-shot target choice    linear-light subject lift,
          + ByteTrack       on ~1 fps keyframes  weight smoothing with     background dim, dark-scene
          + scene cuts      with numbered boxes  ≤1 s lookahead            glow, feathered edges
```

| Stage | What happens | Where |
|---|---|---|
| Normalize | Constant-frame-rate working copy (phone VFR video is the classic cause of audio drift); long side capped so the short side is at most 1080 px | `pipeline/video_io.py` |
| Perception | YOLO11n-seg plus ByteTrack gives stable IDs, masks, and boxes per frame. Dark frames are gamma-lifted *for the detector only*. Spectral-residual saliency is the fallback when nothing is detected. Histogram/difference scene-cut detection | `pipeline/perceive.py` |
| Comprehension | Optional. Keyframes (shot starts, then about 1 per second) are sent to Claude with each tracked object's ID drawn on the image (set-of-mark prompting). Claude must choose one of those IDs or the answer is rejected. Every failure falls back to the local heuristic | `pipeline/understand.py` |
| Temporal attention | Per-shot target choice with hysteresis and minimum dwell. Each object's **weight** w_i(t) is smoothed with a ≤1 s look-ahead window that never crosses a cut. Smoothing weights rather than masks avoids ghost trails on moving subjects. M(t) = Σ w_i(t)·mask_i(t) | `pipeline/attention.py` |
| Adaptation | In linear light: a shadow-weighted lift scaled by subject darkness (a black car at night gets stops, a lit face barely changes), local contrast, background dim plus slight desaturation, and glow only in dark scenes. A two-zone blend keeps feathered edges natural | `pipeline/enhance.py` |
| Render | One decode feeds three encoders: `original.mp4`, `enhanced.mp4`, and `attention.mp4`. They use the same frames and the same settings, so the comparison is fair. Audio is copied, never re-timed | `pipeline/render.py` |

Re-rendering with new slider settings reuses the cached perception pass, so it costs one render instead of a full re-analysis.

## Privacy

All video processing happens on your computer. The browser sends the file to `localhost:8000`, which is **this machine**, not a server. If Claude is enabled, only downscaled still keyframes with boxes drawn on them are sent, never the video or its audio. Without an API key, nothing leaves the machine.

## Run it (macOS, Apple Silicon)

```bash
./scripts/setup.sh        # once: venv, Python deps, model weights, UI build
./scripts/start.sh        # http://localhost:8000
```

Optional: put `ANTHROPIC_API_KEY=...` in `.env` to enable Claude subject choice and the "What should Keylight follow?" box.

Other commands:

```bash
./scripts/start.sh --dev                         # backend + Vite hot reload on :5173
python scripts/prerender.py                      # pre-render everything in samples/ for the demo
python scripts/prerender.py clip.mp4 --strength 0.8
python scripts/io_test.py clip.mp4               # verify frame count / fps / audio sync round trip
```

## Repo layout

```
backend/
  main.py              FastAPI: /api/process, /status, /result, /rerender, /files (Range-capable)
  jobs/manager.py      in-memory job queue, one worker thread, rediscovers finished jobs on disk
  pipeline/            video_io, perceive, understand, attention, enhance, render, run
  models/              yolo11n-seg.pt (downloaded by setup)
  outputs/<job>/       working.mp4, perception cache, original/enhanced/attention.mp4, metadata.json
frontend/              React + Vite + TS; wipe player, attention view, focus timeline
samples/               drop demo clips here (git-ignored)
scripts/               setup, start, prerender, io_test, dev helpers
```
