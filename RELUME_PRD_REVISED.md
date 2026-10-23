# Relume — Implementation PRD for Claude Code

**Version:** 1.1 (reviewed MVP specification)  
**Priority:** Deliver a trustworthy **macOS Apple Silicon MVP** before adding Windows support.  
**Primary stack:** C++20, CMake, Dawn/WebGPU, WGSL, ONNX Runtime C++ with CoreML EP, GLFW, Dear ImGui, Python model tools.  
**Output:** Live, on-device, depth-guided webcam relighting; **OBS Virtual Camera bridge** for video calls in MVP.  
**Status:** Technical targets are hypotheses until benchmarked on the development machine.

> **Instructions to the coding agent:** Treat this document as the source of truth for scope, interfaces, and acceptance criteria. Read it first, then implement **one phase at a time**. Do not generate the entire application in one pass. Preserve a working build after every phase. State assumptions, show command outputs, and report deviations; never silently replace a specified library, execution provider, camera integration, or model.

## 1. Product definition

Relume is a native desktop app that takes a local webcam feed, estimates **relative monocular depth**, and applies interactive, depth-guided lighting effects to a live image. A user moves a virtual light, changes presets, and sends the processed video to calling apps through OBS Studio's virtual camera.

**Purpose:** Build a demonstrable, well-engineered C++/graphics/ML inference project showcasing native APIs, reproducible model export, GPU rendering, resource ownership, profiling, and honest benchmarks.

**MVP user journey:**
1. Start Relume and allow webcam access; select a camera if multiple are available.
2. See a live processed preview; if a model/accelerator is unavailable, get a visible explanation and a safe passthrough fallback.
3. Drag the virtual light and switch presets; the output updates without restarting the stream.
4. Open a dedicated **clean output window** (720p, video only), capture that window in OBS, start **OBS Virtual Camera**, and select **OBS Virtual Camera** in a meeting app.
5. Keep controls and performance diagnostics in the **separate Relume control window**; none appear in the clean output.

**What this is not:** full inverse rendering, reconstruction of real material reflectance/albedo, metric 3D, guaranteed physically accurate cast shadows, beauty filtering, virtual backgrounds, recording, or a new native system-camera device in the MVP. The source webcam's original illumination is already baked into its pixels. Describe the result as **depth-guided / physically inspired lighting**, not ground-truth relighting.

## 2. Firm scope and decisions

| Area | MVP commitment | Deferred |
|---|---|---|
| OS | macOS on Apple Silicon (test on actual target Mac) | Windows, Linux |
| Model | Official **Depth Anything V2 Small** (ViT-S) exported to fixed-shape ONNX | Other model families, training |
| Model input | Fixed **448 × 448**, with one documented, consistent aspect-ratio policy | Adaptive/dynamic shapes |
| Inference | ONNX Runtime **C++**; begin with CPU, enable/test CoreML EP | CUDA, TensorRT, WinML |
| Camera | Native **AVFoundation** capture adapter; use a test-file source for headless tests | Media Foundation |
| Rendering | **Dawn/WebGPU** + WGSL; 720p output | 1080p, alternative renderer |
| Lighting | One movable pseudo-3D key light, adjustable ambient/fill, specular and tint, optional cheap background effect | Multi-light, facial landmarks, full cast-shadow simulation |
| UI | GLFW + Dear ImGui control window; dedicated clean output window | Custom UI framework |
| Video-call output | **OBS window capture → OBS Virtual Camera** documented and tested | Native CMIO/Media Foundation device |
| Optimization | Measure, keep buffers reusable, prevent blocking; copies allowed when required | Unproven zero-copy promise, CUDA kernel |
| Privacy | Local camera processing only; no cloud services or telemetry | Accounts, analytics, upload |

**Non-negotiable architecture decision:** ONNX Runtime/CoreML output is **not assumed** to be directly usable as a Dawn texture. The reliable baseline is an explicitly measured CPU staging path. Any zero-copy/device-sharing path must be a separate experiment with verified support, synchronization, correctness, and a benchmark showing it wins. Do not stall MVP delivery trying to force this interop.

**Single-light rule:** The "Studio" preset uses one key light plus ambient/fill approximation; it must not secretly instantiate three lights. "Ring Light" is an even frontal look, **not** a promise of real eye catchlights.

## 3. Target platform and dependency contract

- **macOS:** Apple Silicon, macOS 13+ for the MVP/OBS ScreenCaptureKit window capture path. Test at least one real USB or built-in camera; allow TCC camera and OBS screen-recording permissions.
- **Compiler/build:** Apple Clang with C++20, CMake, Ninja (preferred); Debug and Release presets. Use AddressSanitizer/UndefinedBehaviorSanitizer where compatible in the native debug build; do not assume the same flags work on MSVC later.
- **Graphics:** Pin a known-working Dawn revision and follow Dawn's documented **CMake installation / `find_package(Dawn)`** route initially. Do not invent/guess an unverified `FetchContent` setup. Verify the GLFW surface path with a minimal sample first.
- **ML:** Pin an ONNX Runtime version whose macOS arm64 C++ distribution includes CoreML EP; verify available providers at runtime. Do not claim CoreML is active merely because the session was configured for it. Log which nodes (if any) fall back to CPU using ORT profiling where supported.
- **Camera:** AVFoundation integration in Objective-C++ (`.mm`) behind a C++ interface. Set the app's camera usage description (`NSCameraUsageDescription`) and handle denied permission.
- **UI:** Dear ImGui, version pinned and built against the same Dawn/WebGPU integration used by the app. Avoid mixing independent GPU devices/backends for control/output windows.
- **Dependencies:** Record exact versions and source URLs in `docs/dependencies.md`; preserve reproducible CMake presets and lockfiles/revisions where appropriate. Never check large model weights or local machine paths into Git.

**License gate:** Use only the **Small** checkpoint for this scope, record its origin/version and SHA-256, and include required Apache-2.0 notices and dependency attributions before publishing a distributable. Base/Large/Giant have different licensing and are not drop-in distribution substitutes.

## 4. Functional requirements

### 4.1 Capture and application lifecycle

- Enumerate connected cameras, select one, display permission/error status, handle disconnect and reconnect without crashing.
- Prefer 1280 × 720 at 30 FPS for camera capture, negotiating the nearest available supported format; preserve correct orientation, aspect ratio, and color ordering (BGRA/NV12/RGB conversions explicitly tested).
- Avoid silently selecting **OBS Virtual Camera** as the input; warn about feedback loops.
- Keep the UI responsive during model/session warm-up; show `Loading`, `Ready`, `CPU fallback`, `Camera unavailable`, or `Error` based on verified state.
- Inference failure must never leave an uninitialized frame: display an understandable message and use unmodified camera passthrough where possible.
- No microphone access, no frame uploads, no saving raw frames by default. The user may explicitly generate local benchmark/test artifacts.

### 4.2 Inference correctness contract

Use official Depth Anything V2 Small source code/checkpoint as the reference, not an undocumented third-party ONNX artifact.

- Python `scripts/export.py` loads the pinned checkpoint, exports a **fixed 1 × 3 × 448 × 448** model to `models/depth_anything_v2_small_448.onnx`, runs `onnx.checker`, and writes a model manifest (revision/hash, opset, input/output names, layout, shapes, data types, preprocessing).
- Camera→network preprocessing is **identical** in Python reference, Python ONNX comparison, and C++: BGR/BGRA/NV12 conversion to RGB; RGB float values in `[0,1]`; ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`; contiguous **NCHW** float32.
- Fixed square policy for MVP: **aspect-preserving resize and letterbox** to 448 × 448, padding with ImageNet-mean RGB (equivalent to zero after normalization). Record scale/padding per frame; crop padded regions from predicted depth and resample into the camera image's coordinates. This differs from the upstream variable-aspect resize; test its visual quality against the upstream reference and disclose that difference.
- The original official reference uses a 518 baseline and aspect-preserving dimensions divisible by 14; 448 is an intentional *performance tradeoff*, not a claimed default. Do not accidentally use a 518 transform in Python while using 448 in C++.
- Inspect the exported graph's **actual** tensor names, layouts and dimensions; never hardcode an assumed output name/shape without checking. Verify fixed output with a test image.
- Output is **relative/disparity-like depth, not metres**. Confirm depth direction and value range on two reference fixtures; normalize robustly (e.g., stable percentile/clipped range), guard constant/NaN values, and avoid scene-to-scene or frame-to-frame normalization flicker. Preserve the original unnormalized tensor for diagnostics.
- `scripts/check_parity.py` compares PyTorch and ONNX CPU inference **after identical preprocessing** on fixed test fixtures, emitting mean/max/p95 error and rendered depth images. Record an explicit tolerance grounded in observed FP32 parity; any model/provider variant that violates it is rejected or separately documented.
- Enable CoreML only after ONNX CPU correctness. Benchmark each provider separately, record what actually executed, and allow CPU fallback with an honest performance warning.
- FP16 graph conversion is an **optional experiment**, used only if the selected EP supports it and parity/latency improve; **INT8 is out of MVP**. Do not call casting a model to FP16 "INT8 quantization."

### 4.3 Depth-guided image effects

The default image pipeline must be deterministic for a given frame, depth map, and light settings:

1. Align the depth map with the camera image, rejecting letterbox regions.
2. Spatially smooth depth using an edge-aware filter or conservative local smoothing; optionally use an EMA with reset on camera/scene change. Do not blur all depth edges indiscriminately.
3. Construct approximate camera-space surface positions from normalized screen coordinates and **relative depth** (scale/clamp is a tunable heuristic). Derive normals with finite differences; use one documented coordinate convention and a planar test to check normal direction.
4. Lighting pass: compute one light vector from a user-controlled screen-space `x/y` and pseudo-depth `z`; apply clamped Lambert diffuse, optional low-strength Blinn–Phong-like specular, distance attenuation, user-adjustable warm/cool tint, and an ambient term.
5. Modify source image brightness/color in a controlled way (prefer linear-light compositing with explicit sRGB conversion). Preserve identity, skin texture and baseline visibility; clamp output and avoid NaNs, halos, neon highlights, and totally black shadows.
6. Optional background depth-conditioned fill is a subtle artistic effect, **not true bounced light**. Implement only after the primary relighting is stable.

**MVP does not promise:** true cast shadows from hands onto walls, physically accurate face occlusion, material-aware specular/catchlights, or reconstructed real-world light transport. If a simple screen-space occlusion approximation is later added, gate it separately and benchmark/artifact-test it; never describe it as physically correct.

### 4.4 Controls and presets

Controls: light **X/Y** (normalized viewport), pseudo-distance **Z**, intensity, Kelvin-style warm↔cool tint (artistic color transform), ambient fill, specular strength, background-effect amount, Reset, and Enable/Disable Effect.

Preset requirements (each is a documented configuration of the **same** shader and single light):

| Preset | Intended appearance |
|---|---|
| Soft Key | Front/upper-side light, mild fill, soft contrast |
| Dramatic Side | Lateral key, lower fill, stronger but bounded contrast |
| Ring Light | Near-camera frontal lighting, even fill; no eye-catchlight claim |
| Golden Hour | Warm lateral low-angle look |
| Studio | Key plus simulated ambient/fill; no multiple independent lights |

Preset selection updates sliders, manual edits override the selected values, and Reset restores a documented default. Light drag maps correctly to output image coordinates even when the window is resized or letterboxed.

### 4.5 Clean output and OBS bridge

- The app renders **two distinct surfaces**: a **Relume Controls** window with preview/ImGui/HUD and a **Relume Output** window containing **only the 16:9 processed video** (no cursor indicator, HUD, text, window chrome inside the frame). Both share the same processed frame rather than running inference twice.
- The clean output surface uses 1280 × 720 at up to 30 FPS; prevent unintended stretching. If UI scaling changes, exported pixels remain a known aspect ratio.
- OBS setup (macOS 13+): install OBS Studio; add a **macOS Screen Capture** source in **Window Capture** mode targeting `Relume Output`; hide the cursor if the capture option is available; set OBS canvas/output to 1280 × 720 and frame rate to 30; start OBS Virtual Camera; select **"OBS Virtual Camera"** in Zoom/Meet/Teams.
- The clean output window must be visible/capturable; do not assume minimized or closed windows still stream. Document macOS Screen Recording permission, feedback-loop avoidance, and app compatibility caveats. Verify actual meeting output and that controls are not visible.
- **No in-app "Relume" virtual camera is created in MVP.** OBS is an explicit external dependency, not a hidden API bridge. Since OBS captures a window, a Dawn GPU→CPU readback **is not required** for MVP virtual-camera delivery.

## 5. Pipeline and concurrency

Baseline MVP uses an honest **CPU staging boundary** between capture, ORT and Dawn. CoreML uses hardware acceleration internally when successfully configured; this does not imply WebGPU texture interoperability.

```text
AVFoundation webcam (timestamped camera frame)
  ├──► color convert/upload ──────────────────────────────┐
  └──► resize+letterbox+normalize (CPU)                   │
          ↓                                               │
      ONNX Runtime C++ (CPU initially; CoreML when viable)│
          ↓                                               │
      depth tensor in accessible CPU memory               │
          ↓                                               │
      align/temporal filter + upload depth texture        │
          ↓                                               │
      WGSL normal/lighting passes ◄───────────────────────┘
          ↓
      single processed GPU texture
          ├──► Relume Controls preview (plus separate UI)
          └──► clean Relume Output window ─► OBS window capture
                                         └► OBS Virtual Camera
```

**Concurrency rules:**
- Camera callback is non-blocking: timestamps and publishes latest valid frame to a bounded queue; no ORT execution or GPU waits in callbacks.
- A dedicated inference worker consumes the newest available frame (drop stale inputs instead of allowing backlog). The render loop displays the latest camera frame with the most recent available depth, and marks/report depth age; avoid pairing a stale depth map with a radically different camera pose without user-visible indication.
- Resource ownership is RAII; GPU resource creation is isolated on the rendering/device thread. Use explicit synchronization for CPU/GPU buffers, never re-use mapped storage before completed work. No detached lifetime-unsafe threads; join workers cleanly on shutdown.
- Cope with mismatched camera/render/inference rates: render may target 30 FPS while inference runs at a measured lower cadence. Track **capture FPS, render FPS, inference FPS, depth age, and dropped frames** separately.
- Cache model/session/shaders; reuse staging allocations and GPU textures; cap queues. Reload camera/model without leaking session or frame buffers.
- Optional advanced path (`EXPERIMENTAL_GPU_INTEROP`, **off by default**) may explore EP-specific device buffers. Before merging, prove buffer API compatibility, memory ownership, synchronization, numerical parity, and improvement against the measured staging baseline. Do not add speculative Metal-to-Dawn/CUDA-to-Dawn casts.

## 6. Performance and quality gates

**Reference target:** 1280 × 720 live output; aim for **30 FPS** on a tested Apple Silicon machine after warm-up. A fixed **33.3 ms** frame interval is a *goal*, not evidence that inference is 8–15 ms. CoreML may split work between CPU/GPU/ANE; report actual provider execution and overhead.

Record these separately (p50 and p95; after warm-up, over at least 60 s):

| Metric | How evaluated |
|---|---|
| Capture and render FPS | Timestamped FPS and dropped frame counts |
| Inference latency | Preprocess, ORT `Run`, output availability; provider identified |
| Shader/GPU time | GPU timestamp queries if available; otherwise state that CPU timing is only an estimate |
| End-to-end latency | Camera timestamp → displayed/output frame where measurable |
| Depth age | Age of the depth map used for each rendered frame |
| Memory usage | Resident memory trend across 5 min; check steady state |
| Model parity | PyTorch vs ONNX CPU, then CoreML vs ONNX CPU |
| Visual quality | Before/after fixtures: front/side light, hand near face, glasses/hair, cluttered background, motion |

**Acceptance tiers:**
- **Functional MVP (required):** 720p processed output; movable light and five presets; responsive controls; selected real webcam; validated ONNX CPU path and a tested CoreML path or clearly reported fallback; 5 min continuous use without crash, runaway allocations, uncontrolled queue buildup or leaking overlays; OBS window capture and one real meeting-app test.
- **Performance goal (measured, not fabricated):** near-30 FPS render and good interactive response on documented Apple Silicon test hardware. If not attained, present exact results and optimize measured bottlenecks; allow a lower inference cadence and visible `Performance limited` status rather than falsely declaring 30 FPS.
- **No-Go:** incorrect RGB/normal orientation; parity failures; actual CPU-only execution described as CoreML; no reproducible build; virtual-camera instructions that do not produce live video; UI overlays appearing in meeting output; unexplained crashes.

## 7. Code architecture and repository layout

Prefer small independently testable interfaces; separate OS-specific `.mm` files from portable C++:

```text
Relume/
  CMakeLists.txt
  CMakePresets.json
  README.md
  LICENSES/                    # license notices for redistributed assets
  cmake/
  include/relume/
    camera.hpp                 # ICameraSource, timestamped VideoFrame
    depth_estimator.hpp        # IDepthEstimator, DepthResult, provider info
    renderer.hpp               # IRenderer, RenderSettings, output texture
    app_state.hpp              # presets, settings, runtime states
  src/
    main.cpp
    app/                        # app lifecycle, worker coordination
    camera/avfoundation.mm      # native macOS capture; permission handling
    camera/file_source.cpp      # deterministic testing source
    inference/onnx_depth.cpp    # ORT C++ ownership and providers
    render/dawn_renderer.cpp    # Dawn resources and WGSL passes
    ui/control_window.cpp       # Dear ImGui UI + preview
    output/clean_window.cpp     # video-only capture target for OBS
  shaders/
    preprocess.wgsl             # only if needed/tested
    normals.wgsl
    relight.wgsl
  scripts/
    export.py
    check_parity.py
    benchmark.py
    convert_fp16.py             # optional; no INT8 requirement
  tests/
    fixtures/                   # small licensed or generated static images
    test_depth_alignment.cpp
    test_frame_queue.cpp
    test_lighting_math.cpp
  models/
    README.md                   # model retrieval instructions; no weights in git
  docs/
    architecture.md
    dependencies.md
    decisions.md
    benchmarks.md
    obs_setup.md
```

Interfaces must carry frame ID, capture timestamp, dimensions/pixel format, and image/depth alignment transform; avoid silently mixing camera frames and depth predictions. Prefer bounded `std::vector`/owned buffer data and explicit GPU handles/lifetimes; avoid raw owning pointers, unchecked casts, ad-hoc globals, and exceptions escaping callbacks/threads.

**Testing:** CTest for fast native unit/integration tests, a static-frame smoke test independent of a camera, manual AVFoundation permission/disconnect tests, and explicit OBS verification. CI on a macOS arm64 runner is desirable but not required if unavailable; scripts must work on the actual local development Mac.

## 8. Implementation phases — execute strictly in order

Each phase must end with: **files changed, build/test commands and real results, screenshots or metrics if relevant, risks, next phase, and a clean git commit suggestion**. Stop and report a blocker rather than quietly replacing technology. Do not add future-phase features prematurely.

| Phase | Deliverable | Gate / done when |
|---|---|---|
| **0. Environment and feasibility** | Check target Mac/OS/camera; CMake/Ninja/Clang toolchain; pinned Dawn hello-window; ONNX Runtime C++ arm64 provider probe; small objective-C++ camera permission test. `docs/decisions.md` records feasibility. | Dawn textured quad draws; ORT lists available EPs; native camera access tested; exact build commands work. |
| **1. Model reproducibility** | Official Small checkpoint acquisition with hash/manifest; fixed 448 export; reference input/output fixtures; parity script. | ONNX checker succeeds; PyTorch↔ONNX CPU parity results stored; transformation round-trip verified. |
| **2. Live camera renderer** | AVFoundation source + file source; upload/convert to Dawn; controls/clean windows first with passthrough. | Camera renders at correct aspect/orientation; no crashes on denied/missing camera; video-only output surface exists. |
| **3. Native depth inference** | ORT C++ CPU integration and side-by-side depth debug view; then enable/verify CoreML EP. | Live depth aligns with live camera; provider/profiling printed; CPU fallback works; failure states handled. |
| **4. Relighting** | WGSL normal reconstruction, light drag, ambient/diffuse/specular, bounded color mixing, tests for math/orientation. | Light moves in expected direction; faces remain identifiable; synthetic fixture tests pass; no claimed cast shadows. |
| **5. Stabilize + profile** | Temporal/edge-aware smoothing if needed; bounded latest-frame queues, reuse memory/textures, measure latency/FPS. | Five-minute stability test; benchmarks in `docs/benchmarks.md`; performance bottleneck identified, no invented numbers. |
| **6. Product UI** | Five presets, full manual controls, reset, enable/disable, status/error labels, FPS/inference/depth-age HUD. | Presets reproducible; settings do not freeze output; HUD absent from clean window. |
| **7. OBS end-to-end** | `docs/obs_setup.md`, clean window targeting tested with OBS virtual camera, Zoom or Meet smoke test. | Relit feed appears in actual calling app as **OBS Virtual Camera**, with no UI/cursor overlay and acceptable aspect/orientation. |
| **8. Public repository polish** | README, 20–60 sec demo, architecture explanation, before/after, limitations, benchmark table, dependencies/license credits. | Fresh-clone steps are accurate; all implemented features demoed; no unpublished feature claimed complete. |

**Stop/adjust conditions:** If CoreML model incompatibility, Dawn integration, or macOS video capture creates a reproducible blocker, document it with a minimal failing test; propose the smallest scoped resolution for approval. Never quietly replace C++/Dawn with Electron, Python runtime, OpenCV UI or a webapp. The Python model-prep scripts are allowed; application/runtime remains C++.

## 9. Deferred roadmap (not MVP acceptance criteria)

1. **Windows native app** after macOS MVP: Media Foundation webcam capture; choose/test a Windows-native inference EP. DirectML still exists but is in sustained engineering; evaluate Windows ML/WinML or CUDA with compatibility/performance measurements instead of committing blindly to a deprecated path.
2. **Windows system virtual camera**: investigate Windows 11 **Media Foundation `MFCreateVirtualCamera`** and Microsoft's sample; this requires a media source/registration/packaging, not just one call. Avoid defaulting to a legacy DirectShow virtual camera as the new architecture.
3. **macOS native system virtual camera**: Apple's **Core Media I/O Camera Extension**, signed/bundled/approved using system extension entitlement; not equivalent to OBS bridge.
4. **GPU interop R&D**: provider-specific device memory transfer/shared textures where documented and measurably better; no unsupported blanket "zero-copy" promise.
5. **Optional CUDA/C++ kernel** on NVIDIA: separate benchmarked prototype. A CUDA kernel cannot simply read/write a Dawn/WebGPU texture; budget device interop/copy overhead and compare end-to-end cost against WGSL before integrating.
6. Optional higher-resolution output, more lights, face segmentation, more advanced screen-space shadows, saved local presets, and FP16/INT8 **only after** parity/performance tests.

## 10. Security, privacy and release requirements

- No network upload/telemetry/account sign-in, no automatic video save, and no secrets committed. If checkpoint download occurs, document its one-time network requirement.
- Keep OBS setup explicit and avoid capturing other windows inadvertently. If the app crashes or loses permission, show/fall back gracefully; do not display private stack traces to users.
- Before public release: add LICENSE for **Relume's own code** (owner chooses), notices for model weights/third parties, a software bill of materials/dependency list, checksums/provenance, packaging instructions, and honest platform requirements.
- Do not claim a signed/notarized app unless the actual signing/notarization workflow has completed. A local unsigned developer build is fine for the MVP.

## 11. Required documentation and final acceptance checklist

A release candidate must have:

- [ ] Fresh-clone/build/run commands confirmed on the target Apple Silicon Mac.
- [ ] Native C++ app using pinned Dawn and ONNX Runtime, with camera and permission handling.
- [ ] Versioned Small model provenance, preprocessing contract and PyTorch↔ONNX tests.
- [ ] Verified provider usage/fallback and explicit inference/compositing buffer boundaries.
- [ ] One movable-light effect, 5 presets, controls, 720p processed-only output window.
- [ ] 5-minute stability run and real p50/p95 performance figures (with hardware/OS/EP/resolution).
- [ ] OBS Virtual Camera tested in one actual meeting application, no UI overlay leaks.
- [ ] README demo, source credits, license notes and known limitations.
- [ ] No claims of true physically correct relighting, guaranteed 30 FPS, zero-copy, or a native Relume camera unless separately proven.

## 12. Instructions for the first Claude Code turn

Use this exact starting request with the file in Claude Code's project context:

> Read `RELUME_PRD_REVISED.md` in full. We are building Relume **one phase at a time**, starting with **Phase 0 only**. First inspect the repo and local OS/hardware/toolchain. Summarize Phase 0 deliverables, dependencies, and your proposed pinned versions. Check official documentation for the APIs you need, and explain any feasibility risk before writing nontrivial code. Then implement the minimum CMake/Dawn textured-window smoke test and ONNX Runtime C++ provider probe, plus a small macOS camera-permission/capture test, according to Phase 0. Run what you can, report the exact results and commands, update `docs/decisions.md`, and STOP. Do not move to Phase 1 or fabricate successful tests.

## 13. Reference documentation and attribution

- Depth Anything V2 official source/checkpoints/preprocessing/licensing: https://github.com/DepthAnything/Depth-Anything-V2 ; especially `depth_anything_v2/dpt.py` and `depth_anything_v2/util/transform.py`.
- ONNX Runtime CoreML EP: https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
- ONNX Runtime I/O Binding (GPU buffers are EP-specific; interop not automatic): https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html
- Dawn official CMake quickstart: https://github.com/google/dawn/blob/main/docs/quickstart-cmake.md
- OBS macOS Screen Capture: https://obsproject.com/kb/macos-screen-capture-source
- OBS Virtual Camera: https://obsproject.com/kb/virtual-camera-guide
- Apple CMIO camera extension (future): https://developer.apple.com/documentation/CoreMediaIO/creating-a-camera-extension-with-core-media-i-o
- Windows Media Foundation virtual camera API (future): https://learn.microsoft.com/en-us/windows/win32/api/mfvirtualcamera/nf-mfvirtualcamera-mfcreatevirtualcamera
- Microsoft Windows Camera sample (future): https://github.com/microsoft/Windows-Camera/tree/master/Samples/VirtualCamera
- Windows DirectML status: https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
- **Prior art/inspiration:** Konrad **Reczko** (`@reczko_konrad`, GitHub `reczkok`) and Software Mansion/TypeGPU's depth-aware light-injection demo. Credit visibly in README. This implementation is independent; do not copy code or imply features match without testing.

---

**Final engineering rule:** Correctness and a working end-to-end demonstration come first. Optimize with evidence; report limitations plainly. Every implemented behavior must be testable, and every performance/compatibility claim must be measured or explicitly labeled a goal.
