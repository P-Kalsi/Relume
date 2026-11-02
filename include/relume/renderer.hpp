#pragma once

// Dawn/WebGPU rendering of camera frames (portable; no window-system code).
//
//   VideoFrame --FrameUploader--> camera texture (RGBA8, camera size)
//              --OutputCompositor--> OutputTexture (RGBA8, fixed 1280x720)
//              --Presenter--> any window surface or offscreen target
//
// OutputTexture is the clean-output contract: both windows present it, and
// Presenter accepts nothing else, so a window can never show the
// camera-resolution texture directly. All objects must be created and used on
// the thread that owns the GpuContext.
//
// Depth debug view (Controls window only):
//   depth tensor --DepthUploader--> depth texture (R32Float, valid region only)
//                --DepthVisualizer--> DepthDebugTexture (RGBA8, fixed 1280x720,
//                                     aligned pixel for pixel with OutputTexture)
// DepthDebugTexture is a different type from OutputTexture, so it can never be
// handed to Presenter or a window presenter: the Output window stays video only.

#include "relume/lighting.hpp"
#include "relume/preprocess.hpp"
#include "relume/video_frame.hpp"

#include <webgpu/webgpu_cpp.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace relume {

// Errors reported by Dawn callbacks, which may run on other threads.
class GpuErrorLog {
 public:
  void Record(std::string message);
  std::uint32_t count() const { return count_.load(); }
  std::string last() const;
  bool device_lost() const { return device_lost_.load(); }
  void MarkDeviceLost(std::string message);

 private:
  std::atomic<std::uint32_t> count_{0};
  std::atomic<bool> device_lost_{false};
  mutable std::mutex mutex_;
  std::string last_;
};

class GpuContext {
 public:
  // Creates an instance, a Metal adapter (on macOS) and a device. Returns null
  // and sets `error` on failure.
  static std::unique_ptr<GpuContext> Create(std::string& error);
  GpuContext(const GpuContext&) = delete;
  GpuContext& operator=(const GpuContext&) = delete;

  const wgpu::Instance& instance() const { return instance_; }
  const wgpu::Adapter& adapter() const { return adapter_; }
  const wgpu::Device& device() const { return device_; }
  const wgpu::Queue& queue() const { return queue_; }
  const GpuErrorLog& errors() const { return *errors_; }
  const std::string& adapter_name() const { return adapter_name_; }
  // True if the device was created with the features that IOSurface window
  // presentation needs (SharedTextureMemoryIOSurface, SharedFenceMTLSharedEvent).
  bool supports_layer_presentation() const { return supports_layer_presentation_; }
  // True if the device was created with TimestampQuery (GPU pass timing).
  bool supports_timestamps() const { return supports_timestamps_; }

  // Waits for all submitted work to finish (tests and shutdown only).
  bool WaitIdle();

 private:
  GpuContext() = default;
  std::unique_ptr<GpuErrorLog> errors_ = std::make_unique<GpuErrorLog>();  // outlives the device
  wgpu::Instance instance_;
  wgpu::Adapter adapter_;
  wgpu::Device device_;
  wgpu::Queue queue_;
  std::string adapter_name_;
  bool supports_layer_presentation_ = false;
  bool supports_timestamps_ = false;
};

// The fixed-size clean video output. Its size never follows any window.
class OutputTexture {
 public:
  static constexpr std::uint32_t kWidth = 1280;
  static constexpr std::uint32_t kHeight = 720;
  static constexpr wgpu::TextureFormat kFormat = wgpu::TextureFormat::RGBA8Unorm;

  explicit OutputTexture(const GpuContext& gpu);
  const wgpu::Texture& texture() const { return texture_; }
  const wgpu::TextureView& view() const { return view_; }

 private:
  wgpu::Texture texture_;
  wgpu::TextureView view_;
};

// Uploads CPU frames and converts them to an RGBA8 texture of the frame's
// size. Textures are reused until the size or format changes.
class FrameUploader {
 public:
  explicit FrameUploader(const GpuContext& gpu);

  // Uploads and converts `frame`. Returns false (and leaves the previous
  // camera texture untouched) if the frame's layout is invalid.
  bool Upload(const VideoFrame& frame);

  // The converted frame, or null before the first successful upload.
  const wgpu::Texture& camera_texture() const { return camera_texture_; }
  std::uint32_t width() const { return width_; }
  std::uint32_t height() const { return height_; }

 private:
  void EnsureTextures(const VideoFrame& frame);

  const GpuContext& gpu_;
  wgpu::ShaderModule shader_;
  wgpu::BindGroupLayout rgba_layout_;
  wgpu::BindGroupLayout nv12_layout_;
  wgpu::RenderPipeline rgba_pipeline_;
  wgpu::RenderPipeline nv12_pipeline_;
  wgpu::Buffer nv12_params_;

  PixelFormat format_ = PixelFormat::kBgra8;
  std::uint32_t width_ = 0;
  std::uint32_t height_ = 0;
  wgpu::Texture input0_;  // BGRA8 / RGBA8 frame, or NV12 luma
  wgpu::Texture input1_;  // NV12 chroma
  wgpu::BindGroup bind_group_;
  wgpu::Texture camera_texture_;
  wgpu::TextureView camera_view_;
  std::vector<std::uint8_t> rgba_scratch_;  // RGB8 frames are expanded here
};

// Draws the camera texture aspect-fit into the OutputTexture (black bars),
// or clears the output to black when there is no frame to show.
class OutputCompositor {
 public:
  OutputCompositor(const GpuContext& gpu, const OutputTexture& output);
  void Compose(const wgpu::Texture& camera_texture);
  void ClearToBlack();

 private:
  const GpuContext& gpu_;
  const OutputTexture& output_;
  wgpu::BindGroupLayout layout_;
  wgpu::RenderPipeline pipeline_;
  wgpu::Sampler sampler_;
  wgpu::Texture bound_texture_;
  wgpu::BindGroup bind_group_;
};

// Presents the OutputTexture aspect-fit into a target of any size, e.g. a
// Retina window surface, letterboxing instead of stretching.
class Presenter {
 public:
  Presenter(const GpuContext& gpu, const OutputTexture& output);
  void Present(const wgpu::TextureView& target, wgpu::TextureFormat target_format, std::uint32_t target_width,
               std::uint32_t target_height);

 private:
  const wgpu::RenderPipeline& PipelineFor(wgpu::TextureFormat format);

  const GpuContext& gpu_;
  wgpu::BindGroupLayout layout_;
  wgpu::PipelineLayout pipeline_layout_;
  wgpu::ShaderModule shader_;
  wgpu::BindGroup bind_group_;
  std::vector<std::pair<wgpu::TextureFormat, wgpu::RenderPipeline>> pipelines_;
};

// The depth visualization, the same size as OutputTexture and drawn with the
// same aspect-fit rectangle, so its pixels line up with the output's.
class DepthDebugTexture {
 public:
  static constexpr std::uint32_t kWidth = OutputTexture::kWidth;
  static constexpr std::uint32_t kHeight = OutputTexture::kHeight;
  static constexpr wgpu::TextureFormat kFormat = wgpu::TextureFormat::RGBA8Unorm;

  explicit DepthDebugTexture(const GpuContext& gpu);
  const wgpu::Texture& texture() const { return texture_; }
  const wgpu::TextureView& view() const { return view_; }

 private:
  wgpu::Texture texture_;
  wgpu::TextureView view_;
};

// Uploads the valid region of a 448x448 depth tensor, [pad_y : pad_y+new_h,
// pad_x : pad_x+new_w], straight from the tensor: data offset
// (pad_y * 448 + pad_x) * 4 bytes, 1792 bytes per row, new_h rows. The R32Float
// texture is reused until the region's size changes. Dawn copies the data
// during WriteTexture, so the CPU buffer can be reused at once.
class DepthUploader {
 public:
  explicit DepthUploader(const GpuContext& gpu);
  // Returns false (and uploads nothing) if the transform's region does not
  // fit in the tensor.
  bool Upload(const float* depth, const ResizeTransform& transform);
  const wgpu::Texture& texture() const { return texture_; }

 private:
  const GpuContext& gpu_;
  wgpu::Texture texture_;
};

struct DepthVisParams {
  float lo = 0.0f;  // display range (2nd-98th percentile of the valid region, smoothed)
  float hi = 1.0f;
  bool valid = false;            // false: neutral grey instead of dividing by a zero range
  float camera_weight = 0.0f;    // 0 = depth only; 0.5 = depth blended over the camera image
};

// Draws uploaded depth into the DepthDebugTexture, aspect-fit exactly like
// OutputCompositor draws the camera image (black bars, same FitRect).
class DepthVisualizer {
 public:
  DepthVisualizer(const GpuContext& gpu, const DepthDebugTexture& target);
  // `transform` describes the uploaded depth; `camera_texture` (the uploader's
  // camera texture, used for the overlay) must have the transform's source
  // size.
  void Draw(const DepthUploader& depth, const ResizeTransform& transform, const DepthVisParams& params,
            const wgpu::Texture& camera_texture);
  // "No depth": a black image.
  void Clear();

 private:
  const GpuContext& gpu_;
  const DepthDebugTexture& target_;
  wgpu::BindGroupLayout layout_;
  wgpu::RenderPipeline pipeline_;
  wgpu::Sampler sampler_;
  wgpu::Buffer params_;
  wgpu::Texture bound_depth_;
  wgpu::Texture bound_camera_;
  wgpu::BindGroup bind_group_;
};

// Relighting (Phase 4):
//   uploaded depth --NormalPass--> DepthFieldTexture (RGBA16Float: slopes, height, confidence)
//   camera texture + two depth fields (crossfade) --Relighter--> OutputTexture
// The field covers exactly the camera image (the depth's valid region), so it
// is sampled with the image's uv; see shaders/normals.wgsl and relight.wgsl,
// and include/relume/lighting.hpp for the CPU reference.

class DepthFieldTexture {
 public:
  static constexpr wgpu::TextureFormat kFormat = wgpu::TextureFormat::RGBA16Float;

  explicit DepthFieldTexture(const GpuContext& gpu);
  // (Re)creates the texture if its size differs; the content is then undefined.
  void Ensure(std::uint32_t width, std::uint32_t height);
  const wgpu::Texture& texture() const { return texture_; }
  const wgpu::TextureView& view() const { return view_; }
  std::uint32_t width() const { return width_; }
  std::uint32_t height() const { return height_; }

 private:
  const GpuContext& gpu_;
  wgpu::Texture texture_;
  wgpu::TextureView view_;
  std::uint32_t width_ = 0;
  std::uint32_t height_ = 0;
};

// Builds depth fields, and crossfade snapshots of two fields.
class NormalPass {
 public:
  explicit NormalPass(const GpuContext& gpu);
  // `out` from the uploaded depth (its size), normalized as h = depth *
  // scale + offset. `aspect` is the camera image's width / height. Returns
  // false (and draws nothing) without uploaded depth.
  bool Build(const DepthUploader& depth, float scale, float offset, float aspect, DepthFieldTexture& out);
  // out = mix(from, to, t), the field shown at weight t. `out` must be a
  // different texture from both inputs; the sizes must match.
  bool Blend(const DepthFieldTexture& from, const DepthFieldTexture& to, float t, DepthFieldTexture& out);

 private:
  const GpuContext& gpu_;
  wgpu::BindGroupLayout build_layout_;
  wgpu::BindGroupLayout blend_layout_;
  wgpu::RenderPipeline build_pipeline_;
  wgpu::RenderPipeline blend_pipeline_;
  wgpu::Buffer build_params_;
  wgpu::Buffer blend_params_;
};

// What the relight shader draws. Views other than kRelit are debug views for
// the Controls window; Relighter::Render never uses them.
enum class RelightView : std::uint8_t { kRelit, kHeight, kNormals, kGain, kConfidence };

struct RelightFrame {
  const DepthFieldTexture* from = nullptr;
  const DepthFieldTexture* to = nullptr;
  float blend = 1.0f;  // crossfade weight from -> to
  float fade = 1.0f;   // effect weight: 0 = passthrough
  LightSettings light;  // clamped before use
};

class Relighter {
 public:
  Relighter(const GpuContext& gpu, const OutputTexture& output);
  // The relit camera image into the OutputTexture, aspect-fit with black bars
  // exactly like OutputCompositor. Returns false (and draws nothing) if the
  // camera texture or a field is missing, or the fields differ in size.
  bool Render(const wgpu::Texture& camera_texture, const RelightFrame& frame);
  // A view of the same frame into the DepthDebugTexture (Controls only).
  bool RenderDebug(const wgpu::Texture& camera_texture, const RelightFrame& frame, RelightView view,
                   const DepthDebugTexture& target);

 private:
  struct Target {
    wgpu::Buffer params;
    wgpu::BindGroup bind_group;
    wgpu::Texture camera, from, to;  // what bind_group refers to
  };
  bool Draw(Target& target, const wgpu::TextureView& view, const wgpu::Texture& camera_texture,
            const RelightFrame& frame, RelightView mode);

  const GpuContext& gpu_;
  const OutputTexture& output_;
  wgpu::BindGroupLayout layout_;
  wgpu::RenderPipeline pipeline_;
  wgpu::Sampler sampler_;
  Target output_target_;
  Target debug_target_;
};

// Copies an RGBA8/BGRA8 texture back to the CPU as tightly packed 4-byte
// pixels (in the texture's own channel order). Blocks until done. For tests
// and the headless smoke run only; the app never reads frames back.
bool ReadbackTexture(GpuContext& gpu, const wgpu::Texture& texture, std::vector<std::uint8_t>& pixels);
// The same for an R32Float texture (tests only).
bool ReadbackFloatTexture(GpuContext& gpu, const wgpu::Texture& texture, std::vector<float>& values);
// The same for a depth field, as 4 floats per texel (tests only).
bool ReadbackFieldTexture(GpuContext& gpu, const DepthFieldTexture& field, std::vector<float>& values);

}  // namespace relume
