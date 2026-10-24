# Session handoff: end of Phase 2 + stabilization (2026-10-24)

The next session starts **Phase 3**. Read `CLAUDE.md`, this file, PRD §8
Phase 3, then `docs/decisions.md` (Phase 2 sections) and `docs/architecture.md`.

## Git state
- **Branch:** `main`. Everything after `5bb80ad` is **local and not pushed**
  (no push without approval). The working tree was clean at the last commit.
- **Phase 2:**
  - `20328e3` GLFW pin shared; GoogleTest 1.17.0
  - `453f434` VideoFrame and CPU conversions
  - `3d37a64` frame pool and latest-frame slot
  - `a552c5b` format negotiation, aspect fit, status
  - `fb1e06e` file source
  - `81ac2a6` Dawn upload, NV12 conversion, blit
  - `7045522` Controls/Output windows and headless smoke tests
  - `4c91aea` Dear ImGui 1.93.0 pin
  - `adce000` minimal Controls UI
  - `7c6fee5` Relume.app bundle and optional signing identity
  - `5c78aad` AVFoundation capture and device handling
  - `6f6c642` Phase 2 docs
- **Stabilization** (fix for the occlusion freeze):
  - `e95cf6b` render-loop trace and occlusion harness
  - `3101838` WindowPresenter refactor
  - `c51142a` IOSurface layer presentation (the fix)
  - `c65e36c` memory in the stats log
  - `c0cb28f` stabilization docs
  - `8bcb611` IOSurface reuse hardening (CompositorReuseTracker, 4 buffers)
  - plus the final docs commit that contains this file

## What Phase 2 delivers
- **`Relume.app`** (`dev.relume.app`, signed ad-hoc by default) has two windows:
  - **Relume Controls:** ImGui with a camera dropdown, status, an Output toggle and a preview.
  - **Relume Output:** clean, video only, aspect locked to 16:9.
  - Both present one **fixed 1280×720 `OutputTexture`**, which is the clean-output contract.
  - They open side by side.
- **AVFoundation capture:**
  - The device format and the output pixel format are negotiated separately.
  - Delivered on the FaceTime camera: 1280×720 `420v` @ 30.0 fps, unmirrored, verified on the first frame.
  - Colour matrix and chroma location are read per buffer.
  - Choosing a virtual camera needs explicit confirmation.
  - On disconnect the output goes black and the status reads "Camera unavailable".
- **The file source** (PPM or a generated orientation pattern) drives the headless tests.
- **WGSL conversion** (NV12, BGRA, RGB → RGBA) matches the CPU reference exactly.

## Stabilization: the occlusion freeze
- **Cause:** after a window is covered, `[CAMetalLayer nextDrawable]` (called by Dawn's `Surface::GetCurrentTexture` while it holds the device lock) blocks for up to 1 s. That froze compositing and both windows.
- **Fix (architectural change in the presentation layer only):**
  - `WindowPresenter` interface with two implementations:
    - **`LayerPresenter` (default)** in `src/platform/macos/layer_presenter.mm`: 4 IOSurface-backed Dawn textures per window, shown as `CALayer.contents`. It never acquires a drawable and never waits.
    - **`SurfacePresenter`** (`--presenter surface`): the old CAMetalLayer path, kept as the fallback and used automatically if the IOSurface device features are missing. Re-verified working.
  - **Ownership and synchronization:**
    - The GPU has finished before Core Animation gets a surface.
    - Reuse is decided by `CompositorReuseTracker` (`include/relume/compositor_reuse.hpp`, unit-tested): a committed surface is reused only when it is not in use **and** a newer committed surface has been seen in use. The displayed surface is never reused.
    - Weak-pointer GPU callbacks; Dawn and `CALayer` retain the IOSurfaces they use.
  - **Loop pacing:** the display refresh rate (60 Hz here) for the layer presenter; vsync with an 8 ms floor for the surface presenter.
- **Results:**
  - `ctest --preset display` (`app_window_occlusion`: covers, resizes, minimize) passes, also under ASan and TSan.
  - 0 skipped frames in 7 harness runs with 4 buffers.
  - The surface presenter still reproduces the 1 s freeze in the same harness.
  - 5-minute release camera run (before the reuse hardening): no crash, 0 WebGPU errors, capture 30.0 fps, memory flat at 243–245 MiB.
  - The user confirmed that covering and uncovering no longer freezes.

## Tests (last run 2026-10-24)
- `ctest --preset debug|release|asan|tsan`: **82/82 pass on each**; no warnings in Relume targets.
- `ctest --preset display`: 1/1 pass. It opens windows, needs a GUI session and takes about 1 minute.
- The camera paths are manual (they need the permission prompt; launch with `open`; see `docs/build.md`).

## Unresolved limitations: Phase 5
- **Dragging, live-resizing and minimizing windows pause compositing.** AppKit's modal loops block `glfwPollEvents` on the main thread. Measured: up to 6.3 s while a resize was held, about 0.55 s for the minimize animation. Capture continues; frames are dropped as unread. Fixing it needs the render loop off the main thread, which is an architectural change and was not attempted.

## Not fully verified (do not claim)
1. **OBS compatibility** (OBS is not installed). Whether ScreenCaptureKit captures layer-contents windows, covered or not, is untested; that's Phase 7.
2. **Live permission denial (M5).** Needs the user's explicit approval to change camera permission. Never run `tccutil reset` automatically.
3. **Automatic reconnect of the same camera** (unit-tested status logic only).
4. **Live ASan stability with the camera.** The 5-minute run was a release build; ASan and TSan covered the file-source occlusion test.
5. **Signing with a real local certificate** (none exists).
6. **OBS Virtual Camera exclusion with OBS present** (unit-tested only).
7. **The IOSurface reuse rule's assumption:** Core Animation applies commits in order and never shows superseded contents. It is measured, not proven, and the rule's hold-back path never triggered in a real run. Not claimed race-free.

## Open decision before Phase 3
**Letterbox vs stretch preprocessing** is still provisional. Compare the two on local webcam frames with faces (in the git-ignored `third_party/assets/local_frames/`, using `scripts/compare_preprocessing.py`). Change the contract only if that gives a clear reason.

## Rules learned
- Every rebuild changes the ad-hoc signature, so macOS asks for camera permission again. Tell the user before each camera launch.
- Launch the app with `open`. The file source and `--headless` never touch AVFoundation and can be run directly.
- macOS has no `AVCaptureSessionPresetInputPriority`; use `activeFormat` on the locked device.
- `std::jthread` needs `-fexperimental-library` on Apple libc++ 18, so it is not used.
- `imgui_impl_wgpu.cpp` must be compiled as Objective-C++ without ARC.
- Don't call `wgpu::Surface::GetCurrentTexture` for windows in the main loop on macOS; it can block for 1 s. Use the `WindowPresenter` layer path.
- An occlusion test needs an **opaque** covering window; an undrawn window covers nothing.
- Use `--trace` (slow-iteration phase timings) and `[stats]` (rates, drops, skipped frames, memory) for any new stall.
