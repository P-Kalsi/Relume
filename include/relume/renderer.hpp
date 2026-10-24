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

// Copies an RGBA8/BGRA8 texture back to the CPU as tightly packed 4-byte
// pixels (in the texture's own channel order). Blocks until done. For tests
// and the headless smoke run only; the app never reads frames back.
bool ReadbackTexture(GpuContext& gpu, const wgpu::Texture& texture, std::vector<std::uint8_t>& pixels);

}  // namespace relume
