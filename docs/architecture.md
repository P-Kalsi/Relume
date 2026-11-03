# Relume architecture (as of Phase 4)

Frames go from a camera (or file source) into a fixed 1280×720 output
texture shown in two windows. A depth worker runs Depth Anything V2 Small on
the newest frame with ONNX Runtime (CPU, or the CoreML EP) and stabilizes the
result over time. Since Phase 4, the main thread relights the camera image
with one movable light, guided by that depth ("Relighting" below). Without
matching depth the output is camera passthrough.

## Data flow

```text
AVFoundation (frame queue)          FileSource (own thread)
   │ lock CVPixelBuffer, copy planes      │ encode still image once, copy per frame
   ▼                                      ▼
FramePool (4 recycled VideoFrames) ──► LatestFrameSlot (newest frame only)
                                          │ main thread: TakeIfNew
                                          ▼
FrameUploader: WriteTexture per plane (camera stride) → WGSL convert → camera texture (RGBA8, camera size)
                                          ▼
OutputCompositor: aspect-fit, black bars → OutputTexture (RGBA8, fixed 1280×720)  ◄── the output contract
                                          ├──► Presenter → Relume Output window (video only)
                                          └──► ImGui::Image → Relume Controls preview
```

- **The camera callback never waits.** If all 4 pool buffers are in use, the
  frame is dropped and counted. Replacing a frame nobody took also counts as a
  drop (`dropped_unread`).
- **Frames own their memory.** Each one carries its frame ID (starting at 1
  per source), capture timestamp and clock domain, arrival time, size, pixel
  format, YCbCr range, matrix and chroma location, and per-plane stride.
- **Frames can outlive the pool.** A frame's deleter holds only a `weak_ptr`
  to the pool.
- **Windows never sample the camera texture.** `Presenter` accepts only an
  `OutputTexture`, which keeps it that way.

### Depth (Phase 3)

```text
LatestFrameSlot ──(WaitTakeIfNew: newest frame, never removes it)──► DepthWorker thread
                   ToRgb8 → release the camera frame → Preprocessor (stretch default)
                   → IDepthEstimator::Run (ONNX Runtime CPU or CoreML EP)
                   → DepthResult (recycled pool of 4: depth 448×448, transform, frame ID,
                     camera generation, arrival time, display range, timings)
                                          ▼
                                  LatestDepthSlot (newest result only)
                                          │ main thread: TakeIfNew(active worker generation)
                                          ▼
DepthTracker: MatchDepth (camera generation, source size, transform) — anything else is never shown
                                          ▼
DepthUploader: WriteTexture of the valid region only → R32Float (new_w × new_h)
DepthVisualizer (depth_vis.wgsl): same FitRect as the output, DepthToSource's bilinear mapping
                                          ▼
DepthDebugTexture (RGBA8, fixed 1280×720) ──► ImGui::Image → Controls window only
```

- **Consumers are independent.** Taking a frame never removes it; the
  renderer and the worker each keep their own last-seen frame, so neither can
  starve the other. The worker waits on a condition variable; publishing only
  notifies and never waits.
- **The worker never holds a camera frame during inference.** It converts the
  frame to RGB and releases it before preprocessing and `Run`, so a 300 ms
  run cannot exhaust the 4-frame pool. Frames that arrive meanwhile are
  skipped, never queued.
- **Camera generation.** `LatestFrameSlot` counts `Clear()` calls; the capture
  controller clears it whenever a source stops. Frames and depth results carry
  that generation, and a source numbers its frames from 1 again, so "new" means
  a different (generation, frame ID).
- **Depth is never shown out of place.** The renderer uses a result only if
  its camera generation, source size and transform match the image being
  shown; otherwise the depth view is black. The label gives the depth's age
  and frame lag; only lag 0 is "in sync".
- **`DepthDebugTexture` is not an `OutputTexture`.** No presenter accepts it,
  so the Output window cannot show depth.
- **Allocation:** the estimator's input and output tensors are created once
  over its own buffers and results come from a pool, so Relume allocates
  nothing per run. ONNX Runtime's internal allocations are not claimed to be
  zero.

## Modules

| Library | Contents | Platform code |
|---|---|---|
| `relume_core` | `VideoFrame`, CPU colour conversion, frame pool/slot, format negotiation, `FitRect`, status derivation, event queue, PPM I/O, latency histograms, opt-in frame saving and guided capture, the preprocessing contract (`Preprocessor`, `DepthToSource`), `IDepthEstimator`, `DepthWorker`, `InferenceController`, lighting math (CPU reference), `DepthStabilizer`, `CrossfadeState` | none |
| `relume_inference` | `OnnxDepthEstimator` (ONNX Runtime C++ API; CPU and CoreML EP) | none (links ONNX Runtime) |
| `relume_camera` | `FileSource`; `avfoundation.mm` on macOS, `camera_stub.cpp` elsewhere | `.mm` only |
| `relume_render` | `GpuContext`, `OutputTexture`, `FrameUploader`, `OutputCompositor`, `Presenter`, `DepthDebugTexture`, `DepthUploader`, `DepthVisualizer`, `DepthFieldTexture`, `NormalPass`, `Relighter`, `GpuTimer`, readback (tests only); WGSL embedded at build time | none (WebGPU) |
| `relume_imgui` | Dear ImGui 1.93.0 + GLFW/WebGPU backends | third party |
| `Relume` (app) | `CaptureController`, `DepthTracker`, `DepthView`, windows, Controls UI, Output window, `main` | `platform/macos/*.mm` |
| `relume_depth` (tool) | native preprocessing + ONNX Runtime from the command line: parity, benchmarks, CoreML evidence | none |
| `relume_relight` (tool) | the temporal-stabilization ablation (metrics only) | none |

The portable interfaces are `ICameraSystem` and `ICameraSource`
(`include/relume/camera.hpp`). A Windows port would add a Media Foundation
adapter and a `surface` implementation; nothing else is platform-specific.

## Threads

| Thread | Work | Shutdown |
|---|---|---|
| Main | GLFW events, ImGui, all WebGPU calls, `CaptureController` | — |
| AVFoundation session queue (serial GCD) | configure, `startRunning`, `stopRunning` | `Stop()` uses `dispatch_sync` |
| AVFoundation frame queue (serial GCD) | sample-buffer callback: copy + publish | drained with `dispatch_sync` after the delegate is detached |
| FileSource thread | paced copy + publish | always joined by `Stop()` or the destructor |
| DepthWorker thread (one active, at most one retiring) | creates, warms up, runs and destroys its ONNX Runtime session; RGB conversion; preprocessing; display range | stop flag + `Terminate()` + wake; joined by the main thread only once `finished()` is true; at exit after the windows are hidden (logged; a CoreML compile in progress cannot be interrupted) |
| ONNX Runtime intra-op pool (4 threads by default) | inside `Run` | owned by the session |
| FrameSaver writer (only with `--save-frames` or `--guided-capture`) | RGB conversion and PPM writes of its own copies | joined by the destructor |

- **The main thread never calls ONNX Runtime and never waits for a worker.**
  `InferenceController` retires a worker on a settings change (Off, CPU,
  CoreML compute units) and coalesces rapid changes; see `docs/decisions.md`.
- **Workers never call Dawn.** All WebGPU calls stay on the main thread.

- `Stop()` refuses to run on the source's own queues, which would deadlock.
- Notification blocks hold only shared state (`EventRelay`, a failure flag),
  never the source.
- `std::jthread` is not used: Apple's libc++ 18 ships it only as an
  experimental feature.

## Capture format negotiation

The device format and the delivered pixel format are chosen separately:

1. **Device format (`SelectFormat`).** Keys are compared in order:
   achievable fps closest to 30, exact 1280×720, aspect ratio, area. The
   choice is applied as `activeFormat` with min/max frame duration on the
   locked device, which stays locked across `startRunning`. macOS has no
   `InputPriority` session preset; no preset is set.
2. **Output pixel format (`ChooseOutputPixelFormat`).** Chosen from
   `AVCaptureVideoDataOutput.availableVideoCVPixelFormatTypes`: NV12 in the
   device's native range (`420v`/`420f`) if offered, otherwise BGRA, otherwise
   an Error status. `--pixel-format bgra` asks AVFoundation to convert to BGRA.
3. **Verification.** The first frame's real width, height, format, range,
   strides, matrix and chroma location are logged and shown in Controls. A
   mismatch with what was configured produces a warning.

## Colour conversion (NV12 → RGB)

The CPU reference is `src/core/pixel_convert.cpp`; the GPU version is
`shaders/convert.wgsl`. Both do the same arithmetic in float:

- **Video range:** `y = (Y−16)/219`, `cb = (Cb−128)/224`, `cr = (Cr−128)/224`
- **Full range:** `y = Y/255`, `cb = (Cb−128)/255`, `cr = (Cr−128)/255`
- `R = y + 2(1−Kr)·cr`, `B = y + 2(1−Kb)·cb`, `G = (y − Kr·R − Kb·B)/(1−Kr−Kb)`
- BT.601: Kr = 0.299, Kb = 0.114. BT.709: Kr = 0.2126, Kb = 0.0722.
- Each channel is clamped to [0, 1], then quantized as `floor(255·v + 0.5)`.

Metadata comes from each frame:
- **Range:** taken from the pixel format (`420v` video, `420f` full); it is
  never guessed.
- **Matrix:** read from `kCVImageBufferYCbCrMatrixKey`. If it is missing, or
  is a matrix other than 601/709, Relume uses BT.601 below 720 lines and
  BT.709 at 720 lines or more, and shows a warning once.
- **Chroma location:** read from `kCVImageBufferChromaLocationTopFieldKey`.
  If it is missing, it is recorded as *left* and a warning is shown once.
- **Layout:** buffers that are not exactly 4:2:0 are rejected, never
  converted with the wrong layout.

**Chroma is nearest-neighbour, an approximation.** All four luma samples of
a 2×2 block use that block's single Cb/Cr sample, so the result is the same
for every 4:2:0 chroma siting. Compared with interpolated chroma, it can show
2-pixel colour stair-steps or fringing at sharp colour edges, such as
saturated text or clothing edges. It was chosen so the GPU path can be
compared exactly with an independently tested CPU reference. Interpolated,
siting-aware chroma would be a reviewed change with new tests.

Measured agreement: the GPU output equals the CPU reference with 0 differing
values for all ranges and matrices, on gradients and on every Y code with a
sweep of Cb/Cr codes. The tolerance, fixed in advance, is ±1.

## Windows, Retina and the output contract

- **`OutputTexture` is always 1280×720 RGBA8.** No window size, Retina scale
  or display change alters it.
- **Startup placement:** the Output and Controls windows open side by side,
  so Controls does not cover Output.
- **The Output window** has its content area locked to 16:9. It starts at
  1280/scale × 720/scale points, so its framebuffer is 1280×720 px; this
  measured 1280×720 on the M2's 2× display. `Presenter` aspect-fits the
  output with black bars into whatever framebuffer the window has.
- **The orientation is unmirrored and unrotated**, which is what the camera
  sees and what remote meeting participants should receive.
- **When the camera is unavailable, denied or failed,** the output is cleared
  to black instead of freezing on the last frame.

## Window presentation

Each window presents through a `WindowPresenter` (`src/app/window.hpp`).
There are two implementations.

**`layer` (default): `src/platform/macos/layer_presenter.mm`.** It never
acquires a `CAMetalDrawable`.

- Each window has 4 IOSurfaces, imported into Dawn as
  `SharedTextureMemoryIOSurface` (device features `SharedTextureMemoryIOSurface`
  and `SharedFenceMTLSharedEvent`, requested only when the adapter has them).
  - Why four: the window server can hold two surfaces at once (on screen and
    being composited) while a third waits on the GPU. With three, some runs
    skipped up to about 30% of frames; with four, none were skipped in
    testing.
- Buffer lifecycle: `free → rendering → gpu-pending → ready → displayed`.
  - `Present()` calls `EndAccess`, then `queue.OnSubmittedWorkDone`. The
    completion callback runs inside `Instance::ProcessEvents` and holds only
    a `weak_ptr`.
  - `FinishFrames()` sets `layer.contents` to the newest completed IOSurface
    in a `CATransaction` with actions disabled. **The GPU has finished before
    Core Animation sees a frame.**
- **Reuse rule** (`CompositorReuseTracker`, `include/relume/compositor_reuse.hpp`,
  unit-tested). GPU completion alone never makes a surface reusable.
  - The displayed surface is never rendered into.
  - A surface that was ever committed to Core Animation is rendered into
    again only if `IOSurfaceIsInUse` is false **and** a *newer* committed
    surface has since been seen in use. That newer sighting shows the render
    server has applied a later transaction.
  - "Not in use" alone is not trusted. At window startup, and at the instant
    a window is covered, some committed surfaces were measured never to be
    marked in use, so "not in use" there doesn't prove the render server is
    done with them.
  - Surfaces never committed (new, or superseded before display) need only
    the GPU to have finished.
  - If no surface qualifies, the window skips that frame; nothing waits.
- **Remaining assumption (not proven here):** Core Animation applies
  committed transactions in order and never returns to superseded contents.
  Measured support:
  - In 99.9% of commits, the surface was in use within 0.1 ms.
  - With four buffers, no surface was ever reused without first being seen
    in use.
  - The rule's hold-back path was **not triggered** in the measured runs. It
    is a guard, covered by unit tests rather than by observed engagement.
- Ownership: Dawn's SharedTextureMemory retains its IOSurface while GPU work
  uses it, and `CALayer` retains the surface it displays. Dropping buffers on
  resize or teardown is therefore safe; all were freed at exit in testing.
- `contentsScale` matches the window's backing scale. `contentsGravity` is
  *center* on a black background, so a frame that lags a live resize is
  centred, never stretched.
- The loop is paced to the display refresh rate (`NSScreen.maximumFramesPerSecond`;
  60 Hz here), because there is no vsync.
- Surfaces are untagged BGRA8, the same as the old `CAMetalLayer` default, so
  passthrough colours are unchanged. Relighting does its colour work in the
  shader ("Colour" under Relighting); the surfaces stay untagged.

**`surface` (fallback): `SurfacePresenter`**, a `wgpu::Surface` on a
`CAMetalLayer` (Fifo, vsync).
- Selected with `--presenter surface`, or automatically when the IOSurface
  features are missing.
- It has the known freeze: after a window is covered, `[CAMetalLayer nextDrawable]`
  (called by Dawn's `Surface::GetCurrentTexture` while it holds the device
  lock) blocks for up to 1 s and stalls all rendering.

**Remaining main-thread stalls (both presenters).** GLFW runs on the main
thread. While AppKit runs a modal loop, `glfwPollEvents` does not return, so
compositing pauses:
- dragging or live-resizing a window by hand (measured up to 6.3 s while the
  resize was held);
- the minimize/restore animation (about 0.55 s).

Camera capture continues on its own queue during these pauses; frames are
dropped as unread, not queued.

## Relighting (Phase 4)

```text
DepthWorker thread: model run → DisplayRange → DepthStabilizer (stages A, B) → DepthResult.stable
                                           ▼ (LatestDepthSlot, MatchDepth as before)
Main thread (Relighting, src/app/app.cpp):
  CrossfadeState (stage C) decides: snapshot? which slot to build? which two to show?
  DepthUploader → NormalPass (normals.wgsl) → DepthFieldTexture slot (RGBA16Float, depth size)
  camera texture + from/to fields + LightSettings → Relighter (relight.wgsl) → OutputTexture
                                                                           └ debug views → DepthDebugTexture (Controls only)
```

**Coordinate convention** (`include/relume/lighting.hpp`):
- Image uv runs 0..1 from the top left.
- Lighting space has x right (`(u − 0.5)·2·aspect`), y **up** (`(0.5 − v)·2`;
  image y is flipped here, once) and z toward the viewer (`relief · h`).
- h is normalized depth, larger = nearer. The depth is treated as a height
  field, not unprojected, because its scale and shift are unknown. `relief` is
  a heuristic.
- n = normalize(−∂z/∂x, −∂z/∂y, 1). Tests check that a ramp nearer on the
  right faces left and a ramp nearer at the bottom faces up.

**Depth field.**
- One RGBA16Float texel per depth texel of the valid region, so padding is
  never sampled. It holds the slopes of h, the smoothed h and a confidence.
- The steps:
  - a 5×5 bilateral filter (it doesn't blur across depth edges);
  - gradients over 2 texels (large-scale shape, not fine noise);
  - one-sided differences at discontinuities;
  - lower confidence there.
- The field stores slopes, not normals, so relief applies at shading time.
  Changing it needs no rebuild and keeps the crossfade history valid.

**Shading.** A bounded gain in linear light; the camera pixel already holds
the room's light, so no albedo is estimated:
- a wrapped Lambert key with distance attenuation and a warm/cool tint, plus
  ambient;
- the gain is clamped to [0.35, 1.8], so shadows are never totally black and
  highlights never neon;
- a low-strength Blinn–Phong highlight scaled by the pixel's luma;
- a highlight-preserving boost above gain 1;
- low-confidence pixels (depth edges) get a camera-facing normal, so
  silhouettes don't halo.

Neutral settings, fade 0 and Bypass reproduce passthrough: at most 1 level
off, and exactly for an unscaled 16:9 source. Bypass is the compositor itself.
No cast shadows, occlusion or catchlights are modelled or claimed.

**Colour.**
- Every texture on the colour path is non-sRGB: camera RGBA8Unorm,
  OutputTexture RGBA8Unorm, the IOSurfaces BGRA8Unorm, the surface fallback
  BGRA8Unorm. Nothing converts automatically.
- `relight.wgsl` decodes sRGB once and encodes once.
- The camera signal is BT.709-encoded and is decoded with the sRGB curve. The
  two curves differ by up to 0.047 in linear light (at code 151; recorded by
  `test_lighting_math`). That affects only relit, non-neutral output. No exact
  colour accuracy is claimed.

**Temporal stabilization.** These are the minimal stages, chosen by the
ablation in `docs/phase4_ablation.json`.
- **A (worker): scale/shift alignment.** The per-result percentile
  normalization is aligned to the previous output with a robust affine fit,
  leaking 2% toward identity per result so errors decay instead of drifting.
  An untrustworthy fit resets instead of forcing alignment, and then the
  renderer hard-cuts. A fit is untrustworthy with:
  - fewer than 15% low-gradient pixels;
  - near-zero variance;
  - under 60% inliers;
  - a median residual above 0.08;
  - a scale outside [0.8, 1.25].
- **B (worker): motion-adaptive EMA.** α is 0.5 for changes under 0.03 and
  rises to 1 at 0.12, so moving heads, hands and cuts pass through at once.
- **C (main thread): crossfade.** Each new result fades in over T (0.5 × the
  median result interval, 60–200 ms). T is fixed when the crossfade starts.
- **Measured** (synthetic static scene): flicker F drops from 0.0090 to 0.0037,
  stepping S from 7.97 to 0.56, response lag is 2 frames, and there is no
  ghosting after a scene cut. Without A, a cut ghosts, because without its
  fit there is no scene-change detection.
- **Not compensated:** depth is about 168 ms behind the image, so its
  silhouette trails fast motion. That is Phase 5.

**Three slots and ownership** (`include/relume/crossfade.hpp`).
- Each field slot is `from`, `to` or free. A write (a new field, or a
  snapshot) only ever targets a slot that is neither the current `from` nor
  `to`.
- If a result arrives mid-crossfade, the field shown at that instant is
  snapshotted into the free slot and becomes the new `from`, so nothing snaps
  backward.
- WebGPU runs queue work in submission order, so a read submitted earlier
  always finishes before a later write to the same slot. A GPU test submits a
  whole overlap/burst/cut sequence with one wait and checks every step.

**Resets (hard cut, never a blend).**
- Passthrough at once: another camera, size, transform or worker (including a
  provider change or inference turned off), or no matching depth (including
  no camera image).
- The new field at once, with no blend: a stabilizer reset (a new
  configuration) or a scene change.
- The effect fades in over 0.25 s after a reset to passthrough. It fades out
  over 0.25 s when the depth is older than 1 s ("depth stale" in Controls).

**Threads and cost.**
- The worker adds the stabilizer: p50 1.4 ms, p95 1.6 ms per result (Release, file source).
- The main thread does a 448² upload and a field pass (GPU p50 about 1 ms)
  per result.
- The relight pass (GPU p50 0.7–0.8 ms, Release, while CoreML runs) is redrawn
  when the frame, the depth, the crossfade, the fade or a control changes.
- GPU times come from timestamp queries mapped asynchronously; nothing waits.
- Light controls are never filtered; they apply on the next frame.

**Controls (engineering, Phase 4).** These are in the Controls window only:
- light X/Y/Z, diffuse intensity, ambient, specular, tint and relief;
- dragging on the output preview, with a marker drawn on the preview only;
- Bypass (A/B) and the A, B and C toggles;
- debug views: height, normals, gain and confidence.

The Output window never shows any of them. Presets, Reset, Enable/Disable and
the HUD are Phase 6.

## Privacy

- No microphone is opened. Device notifications for non-video devices are
  ignored.
- No network calls, and signing uses `--timestamp=none`.
- No frames are written to disk by default. Only `--save-frames` and
  `--guided-capture`, which the user must pass, save camera frames, to the
  git-ignored `third_party/assets/local_frames/`. `ReadbackTexture` exists
  only for tests and the headless self-check.
- Inference is on the device (ONNX Runtime, CPU or CoreML). The CoreML compile
  cache (`~/Library/Caches/dev.relume.app/coreml/`) holds compiled models
  only, never frames.
- `imgui.ini` writing is disabled.
- Camera permission goes through the standard macOS prompt; `Relume.app` must
  be launched with `open`.
