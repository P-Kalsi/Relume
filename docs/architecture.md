# Relume architecture (as of Phase 2)

Phase 2 is camera passthrough: frames go from a camera (or file source) into
a fixed 1280×720 output texture shown in two windows. There is no inference
yet; ONNX Runtime arrives in Phase 3.

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

## Modules

| Library | Contents | Platform code |
|---|---|---|
| `relume_core` | `VideoFrame`, CPU colour conversion, frame pool/slot, format negotiation, `FitRect`, status derivation, event queue, PPM I/O, latency histogram | none |
| `relume_camera` | `FileSource`; `avfoundation.mm` on macOS, `camera_stub.cpp` elsewhere | `.mm` only |
| `relume_render` | `GpuContext`, `OutputTexture`, `FrameUploader`, `OutputCompositor`, `Presenter`, readback (tests only); WGSL embedded at build time | none (WebGPU) |
| `relume_imgui` | Dear ImGui 1.93.0 + GLFW/WebGPU backends | third party |
| `Relume` (app) | `CaptureController`, windows, Controls UI, Output window, `main` | `platform/macos/surface.mm` |

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
- **The Output window** has its content area locked to 16:9. It starts at
  1280/scale × 720/scale points, so its framebuffer is 1280×720 px; this
  measured 1280×720 on the M2's 2× display. After a resize or a display
  change, only the surface is reconfigured, and `Presenter` aspect-fits with
  black bars.
- **Surfaces** use `BGRA8Unorm` (not sRGB), so passthrough is an identity,
  with opaque alpha and Fifo present mode.
- **The orientation is unmirrored and unrotated**, which is what the camera
  sees and what remote meeting participants should receive.
- **When the camera is unavailable, denied or failed,** the output is cleared
  to black instead of freezing on the last frame.

## Privacy

- No microphone is opened. Device notifications for non-video devices are
  ignored.
- No network calls, and signing uses `--timestamp=none`.
- No frames are written to disk. `ReadbackTexture` exists only for tests and
  the headless self-check.
- `imgui.ini` writing is disabled.
- Camera permission goes through the standard macOS prompt; `Relume.app` must
  be launched with `open`.
