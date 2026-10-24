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

## Untested (needs your Mac or key)

- **MPS acceleration.** If MPS errors on some op, `perceive.py` automatically retries that video on CPU. Speed on your Mac is unknown. Here, on 2 slow CPU cores, a 1080p clip takes about 13× its length.
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

## Known limitations and next steps if time allows

- The subject heuristic favors large, central, persistent, person-like objects. In a crowd it picks one arbitrarily, and that's exactly where Claude or a focus instruction helps.
- Masks are computed at about 320 px on the long side. Tiny subjects get coarse edges, which feathering hides.
- No Tauri packaging (deliberately skipped).
- Possible speed-up if needed: run the detector every 2nd frame. That's not implemented, so ask if the Mac is too slow.

## Where things are

- Pipeline: `backend/pipeline/` (the docstring at the top of each file explains its design decisions; read these before Q&A)
- Tuning knobs: `enhance.py` (`Params`, lift and contrast constants), `attention.py` (`CLASS_PRIOR`, dwell/hold, smoothing window), `perceive.py` (cut thresholds, dark-frame boost)
- Dev helpers: `scripts/preview_frames.py` renders a few frames side by side for fast tuning without encoding video
