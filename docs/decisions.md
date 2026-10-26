# Engineering decisions and feasibility log

Each entry records what was decided, why, and the evidence. Results are from
the development machine unless stated otherwise.

## Phase 0 — Environment and feasibility (2026-10-23)

### Development machine

| Item | Value |
|---|---|
| Hardware | Apple M2 MacBook Air, 16 GB RAM |
| OS | macOS Sonoma 14.8.9 (23J631) |
| Compiler | Apple clang 16.0.0 (clang-1600.0.26.6), Command Line Tools 16.2 (no Xcode.app) |
| SDK | macOS SDK 15.2, libc++ 18.1 (`_LIBCPP_VERSION 180100`) |
| Build | CMake 4.4.4, Ninja 1.13.2 |
| Cameras | FaceTime HD Camera (built-in), one iPhone Continuity Camera |

### Feasibility results

| Gate (PRD §8 Phase 0) | Result | Evidence |
|---|---|---|
| Dawn textured quad draws | **Pass** | Windowed probe presented frames on a 2560×1440 Metal surface; the developer visually confirmed the orange/blue checkerboard. The automated offscreen test reads back 66 pixels and matches all of them. |
| ORT lists available EPs | **Pass** | `CoreMLExecutionProvider`, `WebGpuExecutionProvider`, `CPUExecutionProvider` |
| Native camera access tested | **Pass** | Authorized after the user approved the prompt: 60 frames, 1920×1080 BGRA, 29.3 FPS. The denied path was also exercised: a clear message and exit code 2. |
| Exact build commands work | **Pass** | `debug`, `release`, `asan` and `tsan` presets build, and CTest passes (2/2) on each. |

### Toolchain blockers found and resolved

1. **Apple clang 15.0.0 could not compile the pinned Dawn.**
   - Error: `src/tint/lang/core/ir/transform/multiplanar_external_texture.cc:159: no viable constructor or deduction guide for deduction of template arguments of 'overloaded'`.
   - Cause: the code relies on C++20 class template argument deduction for aggregates (P1816R0/P2082R1), which arrived in LLVM Clang 17. Apple clang 15 is based on LLVM 16.
   - Resolution: updated macOS 14.4.1 → 14.8.9 and installed CLT 16.2 (Apple clang 16). Dawn was not patched and no other compiler was used.
2. **Stale libc++ 13 headers shadowed the SDK.**
   - `/Library/Developer/CommandLineTools/usr/include/c++/v1` contained libc++ 13 headers from an older CLT install. No package owned them, and clang searched that folder before the SDK's libc++ 18 headers. As a result, `std::format` and `std::views::transform` failed to compile.
   - Resolution: a clean reinstall of the Command Line Tools. Verify with `clang++ -std=c++20 -E -dM -x c++ - <<< '#include <ranges>' | grep _LIBCPP_VERSION`, which must print `180100` or later.
   - An earlier note blamed "Apple clang 15 lacks `std::format`" on the compiler. That diagnosis was wrong; the stale headers were the likely cause.

### Decisions

- **Minimum macOS is 14.0.**
  - The ONNX Runtime 1.31.0 prebuilt dylib declares `minos 14.0` (`vtool -show-build`), so Relume and Dawn are both built with `CMAKE_OSX_DEPLOYMENT_TARGET=14.0`.
  - This is stricter than the PRD's "macOS 13+" note, which concerns OBS window capture.
- **Dawn is built from source, not taken from prebuilt releases.**
  - Dawn's GitHub prebuilts are produced on `macos-latest` runners, with an unverified deployment target and SDK.
  - Building from source follows the PRD's documented CMake route and uses our toolchain.
  - Only the Metal backend is enabled. Samples, tests, Tint tools, protobuf and the other backends are off, and the build uses 4 parallel jobs.
- **Dawn's CMake package is missing a dependency.** `DawnConfig.cmake` references `Threads::Threads` without `find_dependency(Threads)`, so `probes/dawn_hello/CMakeLists.txt` calls `find_package(Threads)` first. Dawn itself is not modified.
- **ONNX Runtime's bundled CMake config is wrong.** The 1.31.0 config points at `include/onnxruntime/`, but the archive ships its headers in `include/`. We use `cmake/FindOnnxRuntime.cmake` instead.
- **ORT's internal Dawn copy does not clash with ours.** The ORT package includes a WebGPU EP (and therefore its own Dawn), but `libonnxruntime.dylib` exports only 3 symbols, so it cannot collide with our statically linked Dawn.
- **CoreML wording.** "CoreML EP is listed and accepted on `SessionOptions`" is all Phase 0 proves. It does **not** show that any model node executes on CoreML. Model-level verification (ORT profiling, node placement, parity) is Phase 1/3 work and must not be claimed earlier.
- **Surface creation uses our own code.** We create the window surface with a small Objective-C++ helper that attaches a `CAMetalLayer` (`probes/dawn_hello/metal_layer.mm`), not Dawn's GLFW utilities, which its install does not export. The platform-specific code stays in one `.mm` file.
- **Objective-C must be enabled alongside Objective-C++ in CMake.** With only `OBJCXX` enabled, CMake compiles GLFW's `.m` files as Objective-C++ and they fail.
- **The camera probe is a signed `.app` bundle.** It is ad-hoc signed and launched with `open`, so macOS privacy (TCC) attributes the camera request to the bundle and reads `NSCameraUsageDescription`. Privacy controls are never bypassed.
- **Tests use no framework.** Phase 0 tests are the probes themselves, registered with CTest. A unit-test framework will be chosen when there is unit-testable code (Phase 1+).

### Sanitizers

| Preset | Flags | Status |
|---|---|---|
| `asan` | `-fsanitize=address,undefined -fno-sanitize-recover=undefined` | Builds; tests pass; `libclang_rt.asan_osx_dynamic.dylib` linked |
| `tsan` | `-fsanitize=thread` | Builds; tests pass; `libclang_rt.tsan_osx_dynamic.dylib` linked (optional preset) |

- Sanitizers instrument Relume's code and GLFW only. Dawn (a prebuilt static library) and ONNX Runtime (a prebuilt dylib) are not instrumented.
- LeakSanitizer and MemorySanitizer are not supported by Apple clang on macOS.
- MSVC will need different flags; `cmake/Sanitizers.cmake` errors out on MSVC instead of guessing.

### Known issues and limitations

- **Camera frame size.** Requesting `AVCaptureSessionPreset1280x720` still delivered **1920×1080** frames from the built-in camera. Phase 2 must choose the capture format explicitly (`activeFormat` / output size) and test the result.
- **Camera permission resets after rebuilds.** The ad-hoc signature changes on every rebuild, so macOS asks for camera permission again. A stable signing identity would fix this and should be considered for Phase 2.
- **No screenshot of the window.** `screencapture` without Screen Recording permission shows only the wallpaper, so the windowed Dawn check relies on the developer's visual confirmation plus the automated offscreen pixel test.
- **GLFW warnings under ASan.** GLFW 3.4 emits `sprintf` deprecation warnings in the `asan` preset. They are third-party and harmless.
- **Possible future `.gitignore` conflict.** `.gitignore` ignores all of `models/`, but the PRD expects a tracked `models/README.md`. This needs adjusting in Phase 1.
- **Disk use.** Dawn source with its fetched dependencies takes about 1.3 GB under `third_party/`.

## Phase 1 — Model reproducibility (2026-10-23)

Details and numbers: `docs/model.md`.

### Decisions

- **Python 3.14 in an isolated `.venv`.** Every pinned package publishes
  cp314 macOS arm64 wheels, and the install and import smoke test passed. Python
  3.12 remains a supported fallback (`RELUME_PYTHON=python3.12`). The system
  Python is not modified.
- **Downloads use `curl`, not Python `urllib`.** The python.org build of
  Python 3.14 has no CA certificates configured, so its HTTPS requests failed.
  `curl` uses the macOS trust store, and every file is still SHA-256 verified in
  Python. The system Python's certificate setup was left unchanged.
- **The upstream Depth-Anything-V2 code is fetched, not vendored.** It is pinned
  by commit in `third_party/src`. This keeps the official reference untouched
  and out of Relume's history.
- **Exporter: dynamo, opset 18, single file.** `external_data=False` keeps the
  roughly 100 MB weights inside one `.onnx` file. torch 2.14's default would
  write them to a separate file.
- **The export strips per-node stack traces.** torch embeds absolute source
  paths in `pkg.torch.onnx.stack_trace` node metadata. The fresh-clone check
  caught this: the clone's hash differed even though weights and graph were
  bit-identical, and the file leaked local paths. With that key removed, the
  export is byte-identical across runs and checkout locations
  (SHA-256 `6ca89260…`).
- **Preprocessing is pure NumPy** with an explicitly specified bilinear
  resize. That makes it exactly reproducible in C++; OpenCV's resize is not
  formally specified. Golden vectors pin the contract.
- **Letterbox is provisional.** On the pinned demo images, stretch agreed more
  closely with upstream preprocessing (9 of 10 images). The decision is
  deferred until a comparison on local webcam frames with faces, before
  Phase 3.
- **Tolerances.** Hard ceilings were fixed before any measurement (relative
  max 1e-3, mean ÷ range 1e-4). Working tolerances were calibrated once at 10×
  the observed errors, committed, and are never raised automatically.
- **Depth direction.** Larger values are nearer. This was confirmed visually by
  the developer on 5 demo images.

### Known issues and limitations

- **ONNX Runtime CPU latency is about 328 ms (p50) per 448×448 inference on
  the M2.** That is about 3 FPS, far below the 30 FPS goal. CoreML is
  unmeasured (Phase 3).
- **The model outputs exact zeros in far regions** because of the in-graph
  ReLU, so later depth normalization must handle large flat areas.
- **No upstream demo image is webcam-like.** None has a face, so the
  preprocessing comparison needs real local frames.
- **The `.gitignore` issue from Phase 0 is fixed.** `models/` now ignores
  everything except `README.md` and `*.manifest.json`.

## Phase 2 — Live camera renderer (2026-10-24)

Architecture details: `docs/architecture.md`. Signing: `docs/signing.md`.

### Decisions

- **New pinned dependencies (approved).** Both are fetched with FetchContent,
  SHA-256 pinned, and listed in `docs/dependencies.md`.
  - **GoogleTest 1.17.0** for native unit tests.
  - **Dear ImGui 1.93.0** for the Controls window. A standalone gate compiled
    its GLFW and WebGPU (Dawn) backends against the pinned Dawn and GLFW 3.4
    and rendered offscreen with no WebGPU errors. On macOS,
    `imgui_impl_wgpu.cpp` must be compiled as Objective-C++ without ARC, as
    upstream documents.
- **Frames are copied out of the CVPixelBuffer in the callback** into pooled
  buffers. That returns the buffer to AVFoundation at once and keeps the
  interface portable. Measured copy time is p50 0.6 ms and p95 0.9 ms for
  1280×720 NV12.
- **Device format and output pixel format are chosen separately**
  (`SelectFormat`, `ChooseOutputPixelFormat`). The delivered format is then
  verified on the first frame.
- **No session preset.** macOS lacks `AVCaptureSessionPresetInputPriority`.
  Setting `activeFormat` on the locked device, locked across `startRunning`,
  delivered exactly 1280×720 @ 30 fps. That fixes the Phase 0 issue, where
  the 720p preset gave 1080p.
- **Native NV12 is the default output**, converted in WGSL. With
  `--pixel-format bgra`, AVFoundation converts instead.
- **NV12 chroma is nearest-neighbour**, a documented approximation (see
  architecture.md). The GPU matched the CPU reference with 0 differing
  values; the tolerance is ±1, fixed in advance.
- **Chroma location and matrix are read per buffer.** The FaceTime HD
  Camera reports *top-left* chroma; the iPhone Continuity Camera reports
  *left*. Both report BT.709 and video range (`420v`).
- **A fixed 1280×720 OutputTexture is the output contract.** Both windows
  present it; window framebuffers never change it.
- **Status is derived from facts each frame** (`DeriveStatus`) rather than
  from a separate state machine.
- **The output is black whenever capture isn't live**, never a frozen frame.
- **Unmirrored orientation.** The user confirmed that text reads correctly
  and a raised right hand appears on the viewer's left.
- **The render loop has an 8 ms floor** (about 125 Hz at most), because
  macOS stops vsync-pacing occluded windows. It is below one 60 Hz frame, so
  it never costs a vsync.
- **`std::jthread` is not used**: Apple's libc++ 18 provides it only with
  `-fexperimental-library`. Threads are `std::thread`, always joined.
- **Ad-hoc signing stays the default.** `RELUME_CODESIGN_IDENTITY` is
  optional and checked at configure time.

### Results on the development Mac

| Check | Result |
|---|---|
| Automated tests (debug, release, asan, tsan) | 77/77 pass on each preset, no warnings in Relume targets |
| GPU vs CPU colour conversion | Exact for BGRA8 and RGB8; NV12 0 differing values (tolerance ±1) |
| Headless smoke (file source → output → window-sized target) | Pass for 1280×720 BGRA, 640×480 NV12 (pillarbox), 480×640 RGB |
| Live negotiated/delivered format (FaceTime HD) | `420v` 1280×720 @ 30 fps; delivered 1280×720 NV12 video range, stride 1280/1280, BT.709, chroma top-left |
| Live capture rate | 30.0 fps; timestamp interval p50/p95 33.4/33.4 ms; no driver or pool drops |
| Render loop | 60 fps with both windows vsync-paced (p50 interval 16.7 ms) |
| Output window framebuffer | 1280×720 px on the 2× display |
| Permission prompt (first launch) | The UI kept rendering while waiting; Allow led to Live |
| Orientation, aspect, resizing, clean output (user) | Correct, never stretched, video only |
| NV12 vs BGRA colours (user) | Identical to the eye |
| Camera switching (FaceTime ↔ iPhone) | Works within one run |
| Selected camera disconnected (iPhone) | Capture stopped, "Camera unavailable", black output (user confirmed) |
| Bogus `--camera` ID | "Camera unavailable", no crash, clean exit |

### Known issues and limitations

- **About 1 s render stalls on window occlusion changes:** fixed by the Phase 2 stabilization below.
- **Not verified live:**
  - automatic reconnect of the *same* camera (unit-tested only)
  - the denied-permission path (M5; it needs the user's approval to change camera permission)
  - the 5-minute ASan stability run (M10)
  - signing with a real local certificate
  - exclusion of OBS Virtual Camera with OBS installed (unit-tested only; OBS is not installed)
- **Capture timestamps** are recorded on the CoreMedia host-time clock. They are not compared with `steady_clock`, and no end-to-end latency is claimed.
- **Camera permission is requested again after every rebuild** with ad-hoc signing. `docs/signing.md` explains the optional fix.

## Phase 2 stabilization — window occlusion freeze (2026-10-24)

### Investigation

Instruments used: `--trace` (phase timings of any iteration over 50 ms, each
window's acquire/present time and `NSWindow.occlusionState`) and
`--occlusion-test` (an opaque covering window covers each window, then the
Output window is resized and minimized, on a fixed schedule).

- **Where it blocks:** every freeze was about 1003 ms inside the Output or
  Controls window's `Surface::GetCurrentTexture`. Polling, capture,
  compositing and submission were all under 1 ms.
- **The call chain** (from the Dawn source at the pinned commit):
  `Surface::GetCurrentTexture` takes the device guard, then
  `SwapChainMTL` calls `[CAMetalLayer nextDrawable]`. When a window has just
  been covered, the window server stops returning drawables and the call
  waits out its 1 s timeout.
  - Because the device guard is held, rendering from another thread on the
    same device would block too.
  - Dawn does not expose the drawable, so its availability cannot be checked
    first.
- **The trigger:** the moment of becoming covered. Sustained coverage and
  uncovering did not block. The occlusion notification often arrives *after*
  the stall.
- **In practice**, the main trigger was the Controls window: both windows
  opened centred and overlapped, so Controls covered Output when it came to
  the front.
- **The test harness** needs an opaque covering window. An undrawn window is
  transparent to the window server and covers nothing.

| Option tried (surface presenter) | Result |
|---|---|
| Fifo, 3 drawables (original) | ~1 s stall on every Output cover |
| Immediate present mode | No help; Controls covers stalled too |
| `maximumDrawableCount = 2` | Worse: more stalls, up to 2.3 s |
| Skip presenting while not on screen | 0 stalls in 6 Output covers, but 1 in 3 Controls covers (race); a covered Output window would also stop updating, a risk for OBS |

### Decision (approved): IOSurface layer presentation

- **A feasibility test came first, in isolation.** A scratch program, not in
  the repo, presented two windows through IOSurface `layer.contents`, with no
  `CAMetalLayer` or `wgpu::Surface`.
  - It passed on this M2 with macOS 14.8.9: covering, uncovering and resizing
    with no stall, max GPU and present work 1.7 ms, 0 WebGPU errors, and every
    buffer freed at teardown. The first run caught a lifetime bug (callbacks
    held `shared_ptr`s); the fix was `weak_ptr`.
  - The user confirmed it looked correct: smooth, no tearing or flicker,
    right orientation, no stretching.
- **The implementation is `LayerPresenter`**, the default (see
  architecture.md). `SurfacePresenter` is kept as `--presenter surface` and
  as the automatic fallback.
- **No dependency or version change.** It uses features of the pinned Dawn,
  plus the system IOSurface framework.

### Results

| Test | Result |
|---|---|
| `app_window_occlusion` (file source; covers, resize, minimize), layer presenter | PASS in 5 of 5 runs, plus under asan and tsan. 30.0 fps composed in every covered and resize phase; max presentation work 3.0 ms |
| Same harness, `--presenter surface` | FAIL: 1002–1003 ms stalls on Output and Controls covers (bug reproduced) |
| Automated suites | 77/77 pass on the debug, release, asan and tsan presets; no warnings in Relume targets |
| Hand test with camera (user) | No freeze when covering or uncovering with other apps' windows |
| 5-minute camera run (release, FaceTime 1280×720 NV12) | No crash, exit 0, 0 WebGPU errors. Capture 30.0 fps throughout (0 driver or pool drops). After the interactive first minute: 4 minutes of composed 30.0 fps, render loop 60.0 fps, max interval 19.3 ms, no new drops. 359 frames went unread, all during the user's window drags and resizes in the first minute. Memory: 227 MiB at 30 s, 243–245 MiB from 60 s to 300 s (flat; no growth) |
| Latency from GPU submit to layer contents | About one loop iteration (p50 17.0 ms, p95 17.7 ms, measured in the feasibility test). Display (scan-out) latency was not measured |

### Remaining limitations (not fixed; recorded)

- **Window drags and live resizes pause compositing.** AppKit's modal
  tracking loop blocks `glfwPollEvents`; this was measured up to 6.3 s while a
  resize was held, and the user confirmed it.
- **The minimize/restore animation pauses compositing** for about 0.55 s.
- Both pauses would need the render loop moved off the main thread. That is
  an architectural change (Dawn device synchronization, GLFW main-thread
  rules) and is not part of this fix; Phase 5 can evaluate it.
- **OBS compatibility is not verified.** OBS is not installed. Covered
  windows do keep receiving new layer contents, but whether ScreenCaptureKit
  captures them correctly is untested (Phase 7).
- **The IOSurface reuse rule rests on one assumption:** Core Animation
  applies commits in order and never shows superseded contents. It is
  measured, not proven, and the hold-back path was never triggered in a real
  run. The implementation is not claimed to be race-free.
- **The layer path is macOS-only.** Other platforms use `SurfacePresenter`.
- **The occlusion test needs a GUI session.** It carries the `display` label:
  `ctest --preset display`.

## Phase 3 — Native depth inference (in progress, started 2026-10-25)

### Decision (approved 2026-10-26): stretch is the default preprocessing method

- **What changed:** `scripts/relume_model/preprocess.py` now defaults to
  `stretch` (resize directly to 448×448; a 16:9 frame is squeezed 1.78×
  horizontally). `letterbox` stays a supported method: the golden vectors,
  parity checks and comparison script keep covering both.
- **Evidence:** `scripts/compare_preprocessing.py` on 12 local webcam stills
  (faces, head turns, hands, glasses, hair and shoulders, backlit, dim, close,
  far) and two 20-frame bursts, captured with `--guided-capture`. The rule was
  fixed before looking at results:

  | Condition | Required | Result |
  |---|---|---|
  | Stretch beats letterbox against upstream on both Pearson and Spearman | ≥ 75% of stills | 11 of 12 (92%) |
  | Mean normalized abs diff against upstream | ≥ 10% lower | 0.0425 → 0.0318 (25% lower) |
  | Developer's visual review (faces, hair, shoulders, boundaries) | stretch no worse | Approved: no meaningful overall disadvantage |

  Per-still and burst numbers are in `docs/preprocessing_comparison.json`
  (metrics and SHA-256 fingerprints only). The frames and panels stay local
  and git-ignored.
- **Where stretch did worse:** the backlit still (mean error 0.051 vs
  0.031), and whole-frame flicker in burst B (0.0086 vs 0.0066, although
  flicker on the subject was lower). Upstream is a reference, not ground
  truth.
- **PRD deviation, recorded:** PRD §4.2 describes "aspect-preserving resize and
  letterbox" as the MVP policy and asks for the difference from upstream to be
  tested and disclosed. The PRD is unchanged; this approved decision overrides
  that wording for Relume's default, and letterbox remains available.
- **Unchanged:** the ONNX model (SHA-256 `6ca89260…a42c`, re-export
  byte-identical), the weights, the tolerances and every dependency pin. The
  manifest's `preprocessing` block was regenerated by `scripts/export.py`.

### C++ preprocessing and ONNX Runtime CPU (Steps 3–4)

- **C++ preprocessing** (`relume_core`: `Preprocessor`, `ComputeTransform`,
  `ResizeBilinear`, `DepthToSource`) repeats the Python steps in the same
  order and precision, compiled with `-ffp-contract=off` (no fused
  multiply-add; numpy rounds each product and sum). Golden vectors: 12 of 12
  cases, max difference **0** (contract: 1e-6).
- **Golden vectors for C++** come from `preprocess_golden.txt`, a text twin of
  the JSON written by `golden.py --write-text`; a Python test checks they
  agree. No C++ JSON dependency was added.
- **`relume_inference`** (`IDepthEstimator`, `OnnxDepthEstimator`): portable
  C++, links only ONNX Runtime. Input and output `Ort::Value`s are created once
  over the estimator's own buffers, so Relume allocates nothing per run;
  allocations inside ONNX Runtime are not claimed to be zero (not measured).
  The session's tensor names, types and shapes are checked against
  `image [1,3,448,448] -> depth [1,448,448]`; a mismatch, a missing or corrupt
  model, an ORT exception and a non-finite output are errors, never crashes.
  `Terminate()` (`RunOptions::SetTerminate`) stopped a 1-thread run about
  5 ms after it was called.
- **Native parity** (`scripts/check_native_parity.py`, 11 fixtures × 2 methods,
  also a CTest): C++ vs Python preprocessing max difference **0** on every
  fixture, including 1280×720; C++ ORT vs Python ORT on identical tensors
  **0**; C++ end to end vs PyTorch worst rel_max 3.27e-5 and mean/range
  1.41e-6 (working tolerances 4e-4 and 2e-5, unchanged).
- **CPU threads: 4 by default.** `relume_depth bench`, release, 1280×720
  stretch, 40 runs after 5 warm-ups (M2, 4 P + 4 E cores):

  | intra-op threads | Run p50 | Run p95 | Run max |
  |---|---|---|---|
  | 1 | 978 ms | 982 ms | 987 ms |
  | 2 | 513 ms | 526 ms | 532 ms |
  | 4 | 329 ms | 335 ms | 337 ms |
  | 6 | 349 ms | 369 ms | 382 ms |
  | 8 | 324 ms | 341 ms | 375 ms |
  | ORT default | 348 ms | 357 ms | 358 ms |

  Four threads are as fast as eight at p50, vary less, and leave the
  efficiency cores for capture, rendering and the UI. Preprocessing takes
  1.7 ms and `DepthToSource` 0.6 ms (p50).

### Inference scheduling (Step 5)

- **`DepthWorker`** (one joined `std::thread` per worker) creates the
  estimator on its own thread, runs 2 warm-ups, then loops: wait for a frame
  newer than the last one it saw (`LatestFrameSlot::WaitTakeIfNew`), convert it
  to RGB, **release the camera frame**, preprocess, run, publish to a
  `LatestDepthSlot`. Stale frames are skipped, never queued. Results come from a
  4-buffer recycling pool.
- **Independent consumers:** taking a frame never removes it. The renderer and
  the worker each keep their own last-seen frame, so neither starves the other.
- **Camera generation:** `LatestFrameSlot` counts `Clear()` calls, and the
  capture controller clears it whenever a source stops. Each frame taken
  carries that generation, and so does each depth result. A new source numbers
  its frames from 1 again, so the worker compares (generation, frame ID); the
  first test run found that comparing the ID alone skipped a new camera's first
  frame.
- **Matching:** the main thread uses depth only if its camera generation,
  source size and transform match the image being shown (`MatchDepth`).
  Mismatches are counted and never displayed. Stats report depth age (now −
  the source frame's arrival, steady clock) and frame lag (camera frames shown
  since the depth's frame); only lag 0 is "in sync".
- **`InferenceController`** (main thread): a configuration change or "off"
  stops the active worker (stop flag, `Terminate()`, wake) and moves it to a
  retiring slot; it is joined only after `finished()` is true. At most one
  active and one retiring worker; the newest request is kept as pending and
  starts once nothing is retiring. Results carry the worker generation; a
  retired worker's depth is never handed out. Fake-estimator tests: 20 rapid
  changes, each `Request` + `Update` under 1 ms, at most 2 live sessions, all
  workers joined.
- **Status** is derived from the worker's facts: Off, Loading (session
  creation, warm-up or switching), Ready (provider), CPU fallback (CoreML was
  requested and failed to load), Error (the output stays passthrough).
- **Shutdown:** stop capture, hide both windows, stop and join the workers.
  A join may wait for a session creation that ONNX Runtime cannot interrupt;
  it happens after the UI is gone and the wait is logged (57–104 ms in the
  display tests on CPU).
- **Defaults:** windowed runs use CPU inference (4 threads). `--headless`,
  `--guided-capture` and `--save-frames` default to off unless `--inference`
  is given. Reason, measured: with inference on, the Debug build's guided
  self-test lost a burst frame (its writer could not keep up while ORT used 4
  cores); Release passed 6 of 6 runs (RGB and NV12, 0 frames skipped). Capture
  tools do not need depth.
- **Responsiveness with inference on (display tests, Debug):** the occlusion
  test passes with CPU inference running: 30.0 fps composed and max iteration
  17.7 ms in every judged phase (unchanged from Phase 2).

### Known limitations (recorded, out of Phase 3 scope)

- **Facial depth is fairly smooth.** In several frames, both letterbox and
  stretch give smooth, low-relief depth across the face (little separation of
  nose, eyes and cheeks). That limits how much face-shaped relighting the depth
  alone can support. It should be considered when relighting quality is
  evaluated (Phase 4 onwards); Phase 3 does not try to fix it.

