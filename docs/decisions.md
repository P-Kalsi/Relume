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

- **About 1 s render stalls on window occlusion changes.** `[surface] Relume Output waited ~1000 ms for a drawable` was logged.
  - Cause: `CAMetalLayer.nextDrawable` blocks for up to 1 s when no drawable is free, which happens when the Output window's occlusion changes (for example, the Controls window raised over it).
  - Seen once per affected run, after which the loop recovers. Capture continues; about 30 frames are dropped as unread.
  - Candidate fixes for Phase 5: present each window without blocking the other (separate present thread, or skip-and-retry on occlusion notifications), checked against OBS capture of occluded windows.
- **Not verified live:**
  - automatic reconnect of the *same* camera (unit-tested only)
  - the denied-permission path (M5; it needs the user's approval to change camera permission)
  - the 5-minute ASan stability run (M10)
  - signing with a real local certificate
  - exclusion of OBS Virtual Camera with OBS installed (unit-tested only; OBS is not installed)
- **Capture timestamps** are recorded on the CoreMedia host-time clock. They are not compared with `steady_clock`, and no end-to-end latency is claimed.
- **Camera permission is requested again after every rebuild** with ad-hoc signing. `docs/signing.md` explains the optional fix.
