# Session handoff: Phase 4 in progress (2026-11-05)

Phase 4 (relighting) is in progress and **not complete**. M0–M4 are done,
verified and pushed (`origin/main`). M5, the live-camera evaluation with the
developer, is paused after Section 2 of `docs/phase4_quality.md`. Still to
do: Sections 3–8, the open issues below, M6 docs, and the final regression
run (Debug, Release, ASan, TSan, display, parity). Phase 5 is not started.

## Phase 4 state
- **Design:** `docs/architecture.md`, "Relighting".
- **Decisions, the three approved deviations and the ablation:**
  `docs/decisions.md` (Phase 4) and `docs/phase4_ablation.json`.
- **Default temporal configuration:** A+B+C (α 0.5, crossfade 0.5 × the
  interval). It is the simplest configuration meeting every approved
  threshold.
- **Last full checks (2026-11-02):** Debug and Release 208/208, display 3/3,
  Python 77/77, parity PASS.
  - ASan and TSan have no sanitizer reports. The tests that timed out (one
    under TSan, a 120 s limit; and while the Mac slept) pass on rerun under
    `caffeinate`.
  - `test_depth_stabilizer` has a 600 s per-test timeout.

## Open issues from the camera session (investigate before completion)
1. **Ghost outline and face seam during head motion.** Likely cause: the
   distance term steps sharply at the old silhouette (depth lags about
   180 ms).
2. **Shading reads as face-area brightness, not face shape.** The face's
   depth relief is small after whole-frame normalization.
3. **662 scene-change hard cuts (2.1%) live.** The stats log now counts them
   by rejection reason (`9e76efc`, `1fe5c69`).

## Phase 3 summary (2026-10-28)


Phase 3 (native depth inference) is complete, verified, and **approved by
the developer on 2026-10-28**. The next phase is **Phase 4**; the developer
will start it in a fresh session. Do not start it without approval. Read `CLAUDE.md`, this file, PRD §8, then `docs/decisions.md`
(Phase 3), `docs/architecture.md` and `docs/model.md`.

## Git state
- **Branch:** `main`. `origin/main` is at `f501ca9` (pushed with approval on
  2026-10-25). Everything after it is **local and not pushed**; the developer
  will push the verified commits themselves.
- **Phase 3 commits** (14 with the final handoff commit; oldest first; the first
  three are already on `origin/main`):
  - `b64291b` opt-in `--save-frames`; `efc6420` preprocessing comparison on local frames; `f501ca9` guided capture
  - `9d2bf52` stretch selected as the default preprocessing (gate 1)
  - `23ae079` C++ preprocessing (bit-exact with Python)
  - `f448266` ONNX Runtime CPU estimator, `relume_depth`, native parity
  - `b1539c4` depth worker, inference controller, app integration
  - `52df41c` aligned depth debug view, live provider switching test
  - `c799a8c` CoreML matrix and compile cache
  - `66fcb87` CoreML CPUAndGPU default with CPU fallback (gate 2)
  - `d9246fa` opt-in timed test guide (`--test-guide switching`)
  - `cda9a2d` fix: no depth match while no camera image is shown
  - plus the final docs commit that contains this file

## What Phase 3 delivers
- **Preprocessing contract:** stretch to 448×448 by default (letterbox
  supported), implemented once in Python and reproduced in C++ bit-exactly.
- **Inference:** `IDepthEstimator` / `OnnxDepthEstimator` (ONNX Runtime
  1.31.0), on a `DepthWorker` thread that never holds a camera frame during a
  run; `InferenceController` switches providers without blocking the main
  thread. Status: Off, Loading, Ready, CPU fallback, Error (passthrough).
- **Default provider (approved at gate 2, confirmed 2026-10-28 after the camera checks):** CoreML
  `CPUAndGPU`, falling back to CPU automatically. 481 of 517 nodes run through
  CoreML (36 `Gather` nodes on CPU); no Neural Engine use is claimed. CoreML
  passes the FP32 parity tolerances; it is not bit-identical to CPU.
- **Depth debug view** in the Controls window only: aligned pixel for pixel
  with the output, overlay toggle, age and lag label ("in sync" only at lag
  0). The Output window stays camera passthrough.
- **Tools:** `relume_depth` (run, run-tensor(s), bench), `check_native_parity.py`,
  `benchmark_coreml.py`, `summarize_ort_profile.py`.

## Measured on the real camera (FaceTime HD, 1280×720 NV12 30 fps)
- CPU: 2.8 depth updates/s, run p50 351 ms, depth age p50 384 ms.
- CoreML CPUAndGPU: 7.1 updates/s, run p50 131 ms, age p50 168 ms.
- Both: capture and output 30.0 fps, 0 drops, render-loop p95 17.1–17.4 ms.
- 5-minute CoreML run: memory 675 → 692 MiB, levelling off (+1.7 MiB over
  the last 2.5 min); no loop iteration over 50 ms.
- Full table: `docs/decisions.md`, "Real-camera checks (Step 8)".

## Tests (last run 2026-10-28)
- `ctest --preset debug|release|asan|tsan`: **156/156 pass on each**; no warnings in Relume targets.
  Includes the C++ golden and native-parity tests, the ONNX Runtime estimator, the worker and controller
  (fake estimator, also under TSan), the GPU depth-alignment tests, and headless inference runs (CPU, CoreML
  default, CPU fallback, missing model).
- `ctest --preset display`: **3/3 pass** (occlusion with CoreML running, guided self-test, live provider
  switching: max iteration 39.3 ms).
- `pytest tests/python`: 77 passed; `check_parity.py`: PASS; `check_native_parity.py`: PASS.
- The camera checks are manual (Step 8 table in `docs/decisions.md`).

## Phase 4 design note (from the developer)
Depth currently updates about **7 times per second** (CoreML) or about 3 (CPU),
and it **flickers** visibly. The Phase 4 relighting design must account for
temporal stability, so lighting effects do not visibly jump between depth
updates. (Depth carries its frame ID, age and lag; `docs/architecture.md`.)

## Unresolved limitations
- **CoreML startup:** about 87 s the first time (compile, then cached in
  `~/Library/Caches/dev.relume.app/coreml/`) and 15–20 s on later launches;
  passthrough with Loading meanwhile. Must be addressed or reassessed before
  production.
- **Depth flicker** with both providers, slightly more noticeable with CoreML
  (more updates per second); no smoothing added. For a later phase.
- **Camera switching stalls the render loop once, 66–92 ms** (synchronous
  AVFoundation stop on the main thread; pre-existing Phase 2 design).
- **Facial depth is fairly smooth** in several frames; a limit for judging
  relighting.
- **CPU depth is about 3 updates/s**, which looks choppy.
- **Dragging, live-resizing and minimizing windows pause compositing**
  (AppKit modal loops; Phase 5).
- **OBS compatibility is untested** (OBS is not installed; Phase 7).
- **Long-term memory stability is unverified:** memory levelled off within the
  5-minute camera run (+1.7 MiB over its last 2.5 min); runs of hours were not
  tested.

## Not fully verified (do not claim)
1. **OBS compatibility** (OBS is not installed). Whether ScreenCaptureKit captures layer-contents windows, covered or not, is untested; that's Phase 7.
2. **Live permission denial (M5).** Needs the user's explicit approval to change camera permission. Never run `tccutil reset` automatically.
3. **Automatic reconnect of the same camera** (unit-tested status logic only).
4. **Live ASan/TSan with the camera.** Camera runs are Release builds; ASan and TSan cover the file-source and headless inference tests.
8. **Neural Engine execution:** no op is planned on the ANE; nothing is claimed.
9. **Per-run CoreML hardware placement:** only CoreML's planned placement (compute plan) was recorded.
10. **Long-run memory:** flat within 5 minutes; hours were not tested.
11. **Windows preprocessing parity:** float32 without fast-math makes it likely; it counts as verified only after the golden tests run on Windows.
5. **Signing with a real local certificate** (none exists).
6. **OBS Virtual Camera exclusion with OBS present** (unit-tested only).
7. **The IOSurface reuse rule's assumption:** Core Animation applies commits in order and never shows superseded contents. It is measured, not proven, and the rule's hold-back path never triggered in a real run. Not claimed race-free.

## Decisions this phase
- **Gate 1 (2026-10-26):** stretch is the default preprocessing method.
- **Gate 2 (2026-10-27):** CoreML CPUAndGPU default with CPU fallback;
  FP32 tolerances unchanged; no FP16 tolerances.

## Rules learned
- Every rebuild changes the ad-hoc signature, so macOS asks for camera permission again. Tell the user before each camera launch.
- Launch the app with `open`. The file source and `--headless` never touch AVFoundation and can be run directly.
- macOS has no `AVCaptureSessionPresetInputPriority`; use `activeFormat` on the locked device.
- `std::jthread` needs `-fexperimental-library` on Apple libc++ 18, so it is not used.
- `imgui_impl_wgpu.cpp` must be compiled as Objective-C++ without ARC.
- Don't call `wgpu::Surface::GetCurrentTexture` for windows in the main loop on macOS; it can block for 1 s. Use the `WindowPresenter` layer path.
- An occlusion test needs an **opaque** covering window; an undrawn window covers nothing.
- Use `--trace` (slow-iteration phase timings) and `[stats]` (rates, drops, skipped frames, memory) for any new stall.
- The CoreML compile cache is keyed by the model file's size and mtime; a new export gets a fresh cache. Clear `~/Library/Caches/dev.relume.app/coreml/` to measure a cold start.
- Never read webcam images or screenshots with tools: the developer requires that no webcam pixels reach Claude. Report numbers and local paths only.
- `--test-guide switching` shows timed steps for the manual switching test; `build/camera_tests/` holds camera-test logs (text only).
