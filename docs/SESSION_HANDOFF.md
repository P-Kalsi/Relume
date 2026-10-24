# Session handoff: end of Phase 2 + stabilization (2026-10-24)

## Git state
- **Branch:** `main`. Phase 2 commits are local and **not pushed**.
- Correction to the Phase 1 handoff: Phase 0 and 1 *were* already on `origin/main` (up to `5bb80ad`).
- **Phase 2 commits:**
  - `20328e3` build: GLFW pin shared, GoogleTest 1.17.0
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
  - Stabilization:
    - `e95cf6b` trace and occlusion harness
    - `3101838` WindowPresenter refactor
    - `c51142a` IOSurface layer presentation
    - `c65e36c` memory in stats
    - plus a docs commit

## What is done (Phase 2 of PRD §8)
- **App:** `Relume.app` (`dev.relume.app`) has a Relume Controls window (ImGui: camera dropdown, status, Output toggle, preview) and a clean Relume Output window. Both present one fixed 1280×720 `OutputTexture`.
- **AVFoundation source:** the device format and output pixel format are negotiated separately. Delivered: 1280×720 `420v` @ 30.0 fps on the FaceTime HD camera, unmirrored, verified on the first frame.
- **File source:** PPM or a generated orientation pattern, encoded as BGRA, RGB or NV12; used for headless tests.
- **WGSL conversion:** NV12, BGRA and RGB to RGBA. It matches the CPU reference exactly (0 differing values; tolerance ±1).
- **Tests:** 77 CTest tests pass on the debug, release, asan and tsan presets. `ctest --preset display` (the occlusion test) passes, including under asan and tsan.
- **5-minute camera stability run (release):** no crash, 0 WebGPU errors, memory flat at 243–245 MiB; see decisions.md.
- **User-confirmed:** orientation, aspect and resizing, clean output, NV12 vs BGRA colour parity, camera switching, and disconnect leading to "Camera unavailable" with black output.
- Full results: `docs/decisions.md`, Phase 2 section.

## Open items (not verified; do not claim)
1. **Denied-permission path, live (M5).** Needs the user's explicit approval before changing or resetting camera permission. Never run `tccutil reset` automatically.
2. **Automatic reconnect of the same camera** (unit-tested only).
3. **5-minute live run under ASan with the camera** (the 5-minute run was done in release; ASan covered the occlusion test with the file source).
4. **Signing with a real local certificate** (none exists; the user creates it; see `docs/signing.md`).
5. **OBS Virtual Camera exclusion with OBS installed** (OBS is not installed; unit-tested only).
6. **Fixed:** the ~1 s freeze when a window is covered. Windows now present through IOSurface layer contents (`--presenter layer`, the default), with no `CAMetalDrawable`. Still open: window drags and resizes, and the minimize animation, pause compositing (AppKit modal loops on the main thread). See decisions.md, "Phase 2 stabilization".
7. Still open from Phase 1: **letterbox vs stretch** must be decided on local webcam frames with faces **before Phase 3**.

## Rules learned this phase
- Every rebuild changes the ad-hoc signature, so macOS asks for camera permission again. Tell the user before each camera launch.
- Launch the app with `open` (see `docs/build.md`). The file source and `--headless` never touch AVFoundation, so they can run directly.
- macOS has no `AVCaptureSessionPresetInputPriority`. Use `activeFormat` on the locked device.
- `std::jthread` needs `-fexperimental-library` on Apple libc++ 18, so it is not used.
- `imgui_impl_wgpu.cpp` must be compiled as Objective-C++ without ARC.

## Next: Phase 3 (not started)
Native ORT C++ CPU inference, a side-by-side depth debug view, then CoreML EP verification. Decide letterbox vs stretch first (open item 7).
