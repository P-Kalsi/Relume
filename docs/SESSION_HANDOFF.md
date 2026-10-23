# Session handoff: end of Phase 1 (2026-10-23)

## Git state
- **Branch:** `main`, **7 commits ahead of `origin/main`, not pushed.** The working tree was clean at `126a9be` before this handoff.
- **Phase 0:**
  - `532cf54` dependency bootstrap
  - `6acad1d` probes and presets
  - `8306612` docs
- **Phase 1:**
  - `f1743b5` Python environment
  - `309e40e` preprocessing contract
  - `f1cb7e3` ONNX export
  - `720a0fe` parity
  - `0b4aafc` preprocessing comparison
  - `e271af7` docs
  - `126a9be` strip local paths from the ONNX file

## What is done

**Phase 0 (feasibility), all gates passed.** It consists of three probe programs in `probes/`, not application code.
- `dawn_hello` draws a textured quad in a GLFW window, which the user visually confirmed. Its `--offscreen` mode is a CTest pixel-readback test.
- `ort_probe` lists ORT 1.31.0 execution providers (CoreML, WebGPU, CPU). It confirms only that CoreML *registers*.
- `camera_probe` is an AVFoundation `.app`. After the user allowed access, it captured BGRA frames at about 30 FPS. The denied-permission path was also tested.
- The `debug`, `release`, `asan` and `tsan` presets all build and pass.

**Phase 1 (model reproducibility), all gates passed.**
- **Checkpoint:** Depth-Anything-V2-Small (Apache-2.0), HF revision `03876f86…`, SHA-256 `715fade1…`. Upstream code pinned at `a561b849…`.
- **ONNX:** `models/depth_anything_v2_small_448.onnx` (git-ignored), SHA-256 `6ca89260…`.
  - Signature: `image` f32[1,3,448,448] → `depth` f32[1,448,448]; opset 18, IR 10.
  - Byte-reproducible across runs, checkout locations and Python 3.14/3.12, all verified from fresh clones.
- **Output:** relative, disparity-like depth; **larger = nearer** (the user confirmed this visually); values ≥ 0 because of an in-graph ReLU. Far regions are often exactly 0.
- **Parity (PyTorch CPU vs ORT CPU, identical tensors, 22 comparisons):**
  - Worst relative max 3.3e-5, against a 1e-3 ceiling.
  - Worst mean error ÷ range 1.4e-6, against a 1e-4 ceiling.
- **Tests:** 75 pytest cases pass, and CTest passes 4/4 in debug and release.
- **CPU latency (p50 across runs):** PyTorch 316–319 ms and ORT 328–332 ms per inference, roughly 3 FPS.

## Decisions to keep (see `docs/decisions.md`, `docs/model.md`)
- **Minimum macOS 14.0**, because ORT 1.31.0's dylib requires it.
- **Dawn:** built from source at the pinned commit, Metal backend only.
- **ORT CMake:** consumed through our own `cmake/FindOnnxRuntime.cmake`, because the package's config is broken.
- **Preprocessing contract** (`scripts/relume_model/preprocess.py`), **provisional:**
  - RGB uint8 → /255 → letterbox (`s = 448/max(W,H)`, round half to even, centred)
  - bilinear resize, half-pixel centres, clamped edges, horizontal pass first
  - ImageNet-mean padding, which becomes exactly 0 after normalization
  - ImageNet normalization, then NCHW float32
  - The inverse is `depth_to_source`.
- **Parity tolerances:** committed in `scripts/relume_model/tolerances.py`. They are never raised automatically.

## Unresolved (do not silently decide)
1. **Letterbox vs stretch is undecided.** On upstream demo images, stretch agreed more closely with upstream preprocessing on 9 of 10 images. No faces have been tested.
   - The user's decision: before **Phase 3**, compare the two on local webcam frames with faces, hair, shoulders and varied lighting.
   - Frames go in `third_party/assets/local_frames/` (git-ignored); the script is `scripts/compare_preprocessing.py`.
   - Change the contract only if that comparison gives a clear reason.
2. **CoreML execution is unverified.** No model has been run on the CoreML EP, and no node placement or speed has been measured. That is Phase 3.
3. **Camera format:** asking for the 1280×720 preset returned **1920×1080** frames. Phase 2 must set the format explicitly (`activeFormat` or output settings) and test the result.
4. **Camera permission resets on every rebuild,** because the ad-hoc signature changes. A stable signing identity is open for Phase 2.
5. **C++ unit-test framework not chosen yet.** The Phase 0 probes are the only C++ tests. The PRD lists `tests/test_frame_queue.cpp` and similar files, so a framework needs choosing, with approval for any new dependency.
6. **Repository layout differs from PRD §7.**
   - There is no `include/relume/` or `src/` yet; the code so far is `probes/` plus `scripts/relume_model/`.
   - Fixtures are generated in code rather than stored in `tests/fixtures/`.
   - Phase 2 should introduce `src/` and `include/relume/` as the PRD lays out.

## Past failures to avoid repeating
- **Apple clang 15 can't build Dawn** (C++20 aggregate CTAD). Use CLT 16+.
- **Stale libc++ 13 headers** in `/Library/Developer/CommandLineTools/usr/include/c++` once shadowed the SDK. If libc++ errors look odd, check that `_LIBCPP_VERSION` is 180100 or newer.
- **CMake must enable `OBJC` as well as `OBJCXX`,** or GLFW's `.m` files get compiled as Objective-C++.
- **`DawnConfig.cmake` doesn't load Threads,** so call `find_package(Threads)` before `find_package(Dawn)`.
- **python.org Python has no CA certificates,** so `scripts/fetch_assets.py` downloads with `curl`.
- **torch.onnx embeds absolute paths** in node `pkg.torch.onnx.stack_trace`. `export.py` strips them, and a test enforces it.
- **`screencapture` can't see other apps' windows** without Screen Recording permission. Visual checks need the user.
- **Don't launch the camera app without telling the user first.** An unannounced prompt led to a "Don't Allow" earlier.

## Phase 2 per PRD §8 (not started)
**Deliverable:**
- an AVFoundation camera source plus a file source (for headless tests)
- upload and conversion into Dawn
- a **Relume Controls** window and a clean **Relume Output** window (16:9 video only), both showing **passthrough** video at first

**Gate:**
- the camera renders with correct aspect ratio and orientation
- no crashes when permission is denied or no camera is present
- the video-only output surface exists

**Related requirements:**
- PRD §4.1: camera enumeration and selection; 1280×720 at 30 FPS negotiated; BGRA/NV12/RGB conversions explicitly tested; never auto-select OBS Virtual Camera; status labels; disconnect and reconnect.
- PRD §4.5: two surfaces sharing one processed frame.
- PRD §5: non-blocking camera callback, bounded latest-frame queue.
- PRD §7: interfaces carry frame ID, timestamp, size, pixel format and alignment.

No inference in Phase 2; inference is Phase 3.

## Read first in a new session
`CLAUDE.md`, then this file, then PRD §4.1, §4.5, §5, §7 and §8, then `docs/decisions.md`, then `docs/build.md`.

Then these sources:
- `CMakeLists.txt`, `CMakePresets.json`, `cmake/`
- `probes/camera_probe/main.mm` (AVFoundation patterns)
- `probes/dawn_hello/main.cpp` and `metal_layer.mm` (Dawn/GLFW surface patterns)
- `scripts/relume_model/preprocess.py` (contract that C++ must match later)
