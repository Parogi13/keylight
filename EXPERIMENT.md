# experiment/sam2-metal

Branch for two experiments that sit **next to** the working app (main is untouched):

1. **SAM 2 segmentation:** YOLO proposes objects about once a second; SAM 2.1 (official
   `facebookresearch/sam2`, Hiera Tiny) segments them precisely and propagates masks
   through every frame in between.
2. **Native Metal EDR player:** renders the enhancement live on the XDR display
   (subject above SDR white, background untouched) instead of encoding an HDR video.

Both plug into what already works:

| Existing piece | Reused by the experiment |
|---|---|
| `perceive.Perception` (per-frame objects + masks) | `sam2_perceive.run` returns the same type, so attention, scene cuts, SDR/HDR renderers and the web player work unchanged |
| `attention.Attention.mask(t)` (smoothed, feathered) | `export_native.py` writes exactly these masks for the Metal player |
| Scene-cut detector | SAM 2 windows never cross a cut |
| HDR conversion (BT.1886 to linear, SDR white = reference) | Metal shader uses the same convention: 1.0 = SDR white, above 1.0 = EDR |

Switches: `segmentation_backend: yolo | sam2` (job setting or `KEYLIGHT_SEGMENTATION`);
renderer = web player with encoded HDR (default) or `native/KeylightPlayer`.

## Run it on the Mac

Use a separate worktree so your normal folder stays on `main`:

```bash
cd ~/Documents/keylight
git fetch
git worktree add ../keylight-sam2-metal experiment/sam2-metal
cd ../keylight-sam2-metal
cp ../keylight/.env . 2>/dev/null || true
```

### 1. Metal EDR calibration (Stage 5, needs no Python)

```bash
xcode-select --install            # once, if `swift` isn't found
cd native/KeylightPlayer
swift run -c release KeylightPlayer --calibration
```

Pass: the 2x, 3x and 4x patches are visibly brighter than the 1x patch and than white
UI elsewhere on screen; the HUD shows `EDR headroom` above 1.0 (it rises when you lower
display brightness). Report the headroom number and any build error verbatim.

### 2. SAM 2 setup + benchmark (Stages 2 to 4)

```bash
cd ~/Documents/keylight-sam2-metal
./scripts/setup_sam2.sh                         # isolated .venv-sam2; prints "loads on mps"
source .venv-sam2/bin/activate
python scripts/bench_segmentation.py samples/night-drive.mp4 --render
```

Outputs `bench/<clip>/`: `masks_A_vs_B.mp4` (YOLO | SAM 2 overlay), enhanced renders per
backend, `bench/report.json` (time, FPS, peak memory, stability, edge alignment).

### 3. SAM 2 masks in the Metal player (Stages 6 and 7)

```bash
source .venv-sam2/bin/activate
cd backend && KEYLIGHT_SEGMENTATION=sam2 python main.py     # process a clip in the web UI
# then, in another terminal:
python scripts/export_native.py backend/outputs/<job_id>
cd native/KeylightPlayer && swift run -c release KeylightPlayer ../../backend/outputs/<job_id>/native
# static test with the exported still frame:
swift run -c release KeylightPlayer --image ../../backend/outputs/<job_id>/native/still.png \
                                    --mask  ../../backend/outputs/<job_id>/native/still_mask.png
```

Keys: Space play/pause, W wipe (drag to move), E enhanced, O original, A attention,
C calibration, Up/Down subject brightness, S whole-frame saturation (on by default
at 1.1x; `--saturation 1` starts with it off), [ / ] less / more saturation.

Whole-frame saturation pushes chroma in linear Display P3 and stops at the P3 gamut edge
(hue and luminance preserved), so colours can go past sRGB on the XDR panel. It is the one
mode that touches the background; it is on by default for a richer look, S turns it off
for the strict "background untouched" comparison, and the wipe's left side always stays
the untouched original.

## Results so far (cloud CPU, 2 cores: **not** representative of the M3 Pro)

Night drive, 97 frames, 1080p:

| | A: YOLO11n-seg + ByteTrack | B: YOLO keyframes + SAM 2.1 Tiny |
|---|---|---|
| Mask stability, consecutive-frame IoU (mean / p5) | 0.953 / 0.908 | **0.983 / 0.970** |
| Edge alignment (boundary gradient vs frame mean) | 5.15 | **7.13** |
| Tracking gaps | 0 | 0 |
| Time | **~14 s** | 1,196 s (12 s/frame; grows from 4 s to 15 s/frame as SAM 2's memory bank fills) |
| A vs B subject agreement (IoU) | 0.886 | |

One short clip: evidence, not proof. The decisive number is SAM 2 speed on MPS.

## Results on the M3 Pro (MPS), Oct 24

Night drive, 97 frames, 1080p, `bench/report.json`:

| | A: YOLO11n-seg + ByteTrack | B: YOLO keyframes + SAM 2.1 Tiny |
|---|---|---|
| Time / FPS | **9.9 s / 9.8 fps** | 117 s / 0.83 fps (SAM 2 propagation 111 s) |
| Peak memory | **681 MB** | 1,139 MB |
| Mask stability, consecutive-frame IoU (mean / p5) | 0.960 / 0.920 | **0.983 / 0.968** |
| Edge alignment | 5.72 | **7.11** |
| Tracking gaps | 0 | 0 |
| A vs B subject agreement (IoU) | 0.908 | |

SAM 2 ran fully on MPS (no CPU fallback, 0 window failures), about 10x faster than the
cloud CPU run but ~36x slower than real time: offline "precise masks" only. Visual check of
`masks_A_vs_B.mp4` agrees: SAM 2 fits the car more tightly and flickers less.

Metal player: builds and runs. Calibration 2x/3x/4x patches are brighter than 1x and the
title bar; headroom rises as display brightness drops. With SAM 2 masks the car glows above
SDR white, the background matches the original, audio stays in sync, and the still test
is aligned, upright and correctly coloured.

No-subject clips (`beach`, aerial coastline, 187 frames; `mountain`, portrait landscape,
473 frames, both 4K scaled to 1080p): neither backend finds a subject (0 frames), SAM 2 has
nothing to propagate (YOLO 6.2 s / 16.8 s, SAM 2 path 7.8 s / 21.1 s). In the Metal player
both play untouched (Keylight == Original); portrait plays upright and pillarboxed with audio
in sync. They only exercise the fallback; night-drive is still the only A-vs-B comparison.

Person walking (first 5 s of an iPhone-style clip: 4K portrait, 60 fps, HLG; 302 frames
at 1080x1920):

| | A: YOLO | B: YOLO + SAM 2.1 Tiny |
|---|---|---|
| Time / FPS | **10.8 s / 27.9 fps** | 575 s / 0.52 fps (3 objects tracked) |
| Peak memory | **725 MB** | 1,326 MB |
| Stability (mean / p5) | 0.979 / 0.945 | **0.982 / 0.972** |
| Edge alignment | 1.78 | **3.14** |
| A vs B subject IoU | 0.903 | |

Visual check agrees: SAM 2 hugs the walker and is steadier; YOLO's mask is a loose blob.
Both play correctly in the Metal player. Cost: SAM 2 runs on every frame (60 fps here) and
on every tracked object, so it is ~2 s/frame. Next speed-ups: run SAM 2 at a frame stride
(e.g. 15 fps effective) and track only the followed subject.

Speed-ups (now the defaults): SAM 2 runs at ~15 Hz (stride 4 at 60 fps; skipped frames get
the nearest SAM mask shifted by box motion) on up to 3 objects ranked like the attention
heuristic, with a drift guard. Tracking only the subject was tried and rejected: at stride
4 the walker's mask intermittently leaked 20x onto a parked car; tracked neighbours stop it.

| Full clip, 689 frames | YOLO | SAM 2, stride 4, 3 objects |
|---|---|---|
| Time / FPS | **19.3 s / 35.7 fps** | 336 s / 2.05 fps (was 0.52 fps) |
| Peak memory | **767 MB** | 1,177 MB |
| Stability (mean / p5) | 0.979 / **0.940** | **0.980** / 0.928 |
| Edge alignment | 2.15 | **3.27** |
| A vs B subject IoU | 0.913 | |

Known issue (also on main): Homebrew ffmpeg 9 has no `zscale`, so `normalize()` falls back
to plain scaling for HLG/PQ input. The working copy is not tone-mapped and keeps its HLG tag.

## Status

| Stage | State |
|---|---|
| 1 Inspect + worktree | done |
| 2 SAM 2 install + image test | done (CPU); MPS: `setup_sam2.sh` checks it |
| 3 YOLO to SAM 2 propagation | done, verified on CPU |
| 4 Benchmark harness | done; Mac numbers pending |
| 5 Metal static EDR | written, **not compiled** (no macOS here) |
| 6 Metal video (AVPlayer clock, masks mmap'd) | written, not compiled |
| 7 SAM 2 masks in Metal | wired via `export_native.py`, untested on Mac |
| 8 Interactive controls | slider, modes, wipe, attention view written |
| 9 Evaluation | after Mac numbers |

## Known risks

- **SAM 2 speed on MPS is unknown.** If it's far from real time, it stays an offline
  "precise masks" option and YOLO remains the default.
- Some SAM 2 ops may fall back to CPU on MPS (`PYTORCH_ENABLE_MPS_FALLBACK=1` is set).
- The Swift code has never been compiled; expect a round of small compiler fixes.
- `still_mask.png` is loaded as an 8-bit image; the shader reads its red channel.
