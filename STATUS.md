# Overnight build status (Sun Oct 25, morning)

## Morning checklist (about 30 minutes)

1. `git pull` (or clone), then `./scripts/setup.sh`. This installs ffmpeg/node via Homebrew if missing, creates `.venv`, downloads the 6 MB model, and builds the UI.
2. `source .venv/bin/activate && python scripts/smoke_test.py`. It should end in **PASS** and print `device mps` plus a realtime factor. That number tells you whether live processing is fast enough to demo or whether you should pre-render.
3. Put your key in `.env` (`ANTHROPIC_API_KEY=...`). Run one clip with a focus instruction to confirm the Claude path works (see "Untested" below).
4. **Get better demo footage.** Put 2–3 clips in `samples/`, then run `python scripts/prerender.py`.
5. `./scripts/start.sh` opens http://localhost:8000. Pre-rendered clips appear under "Open a finished video".

## What's done and verified here (Linux, CPU only)

| | Status |
|---|---|
| Video I/O: constant-frame-rate normalize, decode, encode, audio remux | Verified: identical frame counts and fps; audio/video gap under one frame on CFR, VFR, AC3-audio, and no-audio inputs; a flash/beep sync test lines up |
| Perception: YOLO11n-seg + ByteTrack, dark-frame detector boost, scene cuts | Verified: your night-drive car is tracked as one ID in all 97 frames (conf 0.71–0.83); cuts detected on a multi-shot test clip |
| Temporal attention: per-shot targets, hysteresis, ≤1 s lookahead weight smoothing, no smoothing across cuts | Verified on a 4-shot test clip |
| Enhancement: linear-light adaptive lift, local contrast, background dim, dark-scene glow, two-zone feathering | Tuned visually on night, daylight-crowd, and dim-interior footage |
| Outputs: original / enhanced / attention from the same frames and encoder | Verified |
| Backend: upload, job queue, progress, re-render from cache, Range requests for seeking | Verified through HTTP and the browser |
| Frontend: drop zone, progress, wipe player, Original/Keylight flip, attention view, focus timeline, adjust and re-render | Verified in headless Chromium: all three videos load, followers stay within ~5 ms of the master |
| Claude semantic layer (set-of-mark keyframes, ID validation, fallback) | Logic verified with a **fake** client, including network errors and made-up IDs |
| Fresh clone → build → prerender | Verified |
| Edge cases (second pass) | Verified end to end: HDR/HLG 10-bit HEVC (tone-mapped), rotated phone video, variable frame rate, 60 fps, AC3 and MP3 audio, 1-frame odd-size video, truncated file, garbage file and audio-only file (friendly errors), shots shorter than the smoothing window |
| Claude failure modes | Verified with fakes: unreachable API stops after one probe (~2 s, no stall); slow API is cut off by a 60 s budget; bad IDs rejected; UI notes when Claude wasn't used |

## Untested (needs your Mac or key)

- **MPS acceleration.** If MPS errors on some op, `perceive.py` automatically retries that video on CPU. Speed on your Mac is unknown. Here, on 2 slow CPU cores, a 1080p clip now takes about 7× its length (13× before the latency pass). On a Mac, rendering also spreads across cores (up to 6 threads).
- **Real iPhone HDR footage.** The HDR path was tested on a synthetic HLG clip. If real iPhone clips look too dark or flat, re-export them as SDR or tell me.
- **Live Claude calls.** The model ID defaults to `claude-sonnet-5-5` (`KEYLIGHT_CLAUDE_MODEL` in `.env`). If calls fail, the "How this was made" panel shows "On this computer" and `metadata.json` → `semantic.error` has the reason. Nothing breaks, because it falls back to the heuristic.
- Real playback on an HDR/XDR screen and Safari. I tested Chromium only, so check Safari before judging.

## Demo footage

- `night-drive` (your clip) is the strongest example, but it's only **3 seconds** long. Find a 15–40 s version, since judges need time to drag the wipe.
- The mountain and beach clips have no subject, so they only exercise the fallback, which stays gentle on purpose. Don't use them in the demo.
- Ideal clips: one clear subject in a dark or busy scene, for example a night street with one walker, a concert with one performer, a dim interior with two people (good for "follow the woman on the left"; Claude only sees still frames, so it can't tell who is speaking), or sports with one ball carrier. Pexels or Pixabay footage is royalty-free; avoid film clips.

## Demo flow (about 30 seconds)

1. Start in **Original** mode. Say: "Brightness and contrast settings change the whole frame."
2. Switch to **Wipe** and drag the divider. Say: "We find what matters and separate it from everything else."
3. Press **A** to show the attention view. Say: "A local vision model segments and tracks subjects; this is what it's following."
4. Point at the timeline under the video. Say: "Focus changes ramp in ahead of time, because we keep about one second of lookahead, and never smear across a scene cut."
5. With a key: type "follow the person on the left" and click **Apply changes**. Say: "Segmentation tells us what exists; Claude decides which of those things matters. Only a few still frames leave the laptop."

Keys: Space plays, W wipe, F flips between versions, A attention, `,` / `.` step one frame.

## Latency and memory (second pass)

| Stage | Change | Effect here |
|---|---|---|
| Prepare | Constant-frame-rate H.264 inputs are stream-copied instead of transcoded; others use `superfast` | 0.25 s instead of several seconds for typical stock clips |
| Perceive | YOLO loaded once per process; detector runs at ~30 Hz (every 2nd frame on 60 fps video), interpolating in between | 60 fps video perceives in half the time; mask IoU vs every-frame detection 0.92 |
| Render | uint8 background path, mask upsampled only around the subject, LUT shadow lift, in-place math, threaded frame pool, original stream-copied, attention view at ≤960 px | night-drive render 38 s → 8 s |
| Claude | One probe call, then parallel calls under a 60 s budget | Wi-Fi outage costs ~2 s, not minutes |
| Player | Hidden overlay paused; transport redraws at 15 Hz | Browser decodes 2 videos instead of 3 |
| Memory | Bounded per-render mask cache; frames in flight capped at 2× threads; upload deleted after the working copy is made | Peak RSS ~1.1 GB on a 70 s clip, flat across length (mostly PyTorch itself) |

To make it faster: `KEYLIGHT_DETECT_STRIDE=2` runs YOLO on every 2nd frame for any video. To use a specific number of render threads: `KEYLIGHT_RENDER_THREADS=N`.

## Known limitations and next steps

- The subject heuristic favors large, central, persistent, person-like objects. In a crowd it picks one arbitrarily, and that's exactly where Claude or a focus instruction helps.
- In bright daylight scenes, the undimmed area around the subject can read as a faint light pool. That's intended for night scenes; turn down "Background dimming" for daylight clips.
- Only YOLO's 80 everyday classes are recognized; anything else falls back to a gentle saliency highlight.
- Not built: live streaming playback (the pipeline is 1-second-lookahead compatible, but runs per file), Tauri packaging, multiple simultaneous subjects.

## Where things are

- Pipeline: `backend/pipeline/` (the docstring at the top of each file explains its design decisions; read these before Q&A)
- Tuning knobs: `enhance.py` (`Params`, lift and contrast constants), `attention.py` (`CLASS_PRIOR`, dwell/hold, smoothing window), `perceive.py` (cut thresholds, dark-frame boost)
- Dev helpers: `scripts/preview_frames.py` renders a few frames side by side for fast tuning without encoding video
