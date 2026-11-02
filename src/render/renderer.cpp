#include "relume/renderer.hpp"

#include "relume/layout.hpp"
#include "shaders/blit_wgsl.hpp"
#include "shaders/convert_wgsl.hpp"
#include "shaders/depth_vis_wgsl.hpp"

#include <array>
#include <cstdio>
#include <cstring>
#include <string_view>

namespace relume {

namespace {

std::string ToStdString(wgpu::StringView view) {
  const std::string_view text = view;
  return std::string(text);
}

wgpu::ShaderModule CreateShader(const wgpu::Device& device, const char* source, const char* label) {
  wgpu::ShaderSourceWGSL wgsl{};
  wgsl.code = source;
  wgpu::ShaderModuleDescriptor desc{};
  desc.nextInChain = &wgsl;
  desc.label = label;
  return device.CreateShaderModule(&desc);
}

wgpu::Texture CreateTexture(const wgpu::Device& device, std::uint32_t width, std::uint32_t height,
                            wgpu::TextureFormat format, wgpu::TextureUsage usage, const char* label) {
  wgpu::TextureDescriptor desc{};
  desc.size = {width, height, 1};
  desc.format = format;
  desc.usage = usage;
  desc.label = label;
  return device.CreateTexture(&desc);
}

void WritePlane(const wgpu::Queue& queue, const wgpu::Texture& texture, const VideoFrame& frame,
                std::uint32_t plane) {
  const PlaneLayout& layout = frame.planes[plane];
  wgpu::TexelCopyTextureInfo destination{};
  destination.texture = texture;
  wgpu::TexelCopyBufferLayout data_layout{};
  data_layout.bytesPerRow = static_cast<std::uint32_t>(layout.stride_bytes);  // camera stride, no repacking
  data_layout.rowsPerImage = layout.height;
  const wgpu::Extent3D size = {layout.width, layout.height, 1};
  const std::size_t bytes = layout.stride_bytes * (layout.height - 1) + std::size_t{layout.width} * layout.bytes_per_sample;
  queue.WriteTexture(&destination, frame.data.data() + layout.offset, bytes, &data_layout, &size);
}

wgpu::BindGroupLayoutEntry TextureEntry(std::uint32_t binding, wgpu::TextureSampleType type) {
  wgpu::BindGroupLayoutEntry entry{};
  entry.binding = binding;
  entry.visibility = wgpu::ShaderStage::Fragment;
  entry.texture.sampleType = type;
  entry.texture.viewDimension = wgpu::TextureViewDimension::e2D;
  return entry;
}

template <std::size_t N>
wgpu::BindGroupLayout CreateLayout(const wgpu::Device& device, const std::array<wgpu::BindGroupLayoutEntry, N>& entries) {
  wgpu::BindGroupLayoutDescriptor desc{};
  desc.entryCount = entries.size();
  desc.entries = entries.data();
  return device.CreateBindGroupLayout(&desc);
}

wgpu::RenderPipeline CreatePipeline(const wgpu::Device& device, const wgpu::BindGroupLayout& layout,
                                    const wgpu::ShaderModule& shader, const char* vertex_entry,
                                    const char* fragment_entry, wgpu::TextureFormat format) {
  wgpu::PipelineLayoutDescriptor layout_desc{};
  layout_desc.bindGroupLayoutCount = 1;
  layout_desc.bindGroupLayouts = &layout;
  const wgpu::PipelineLayout pipeline_layout = device.CreatePipelineLayout(&layout_desc);

  wgpu::ColorTargetState target{};
  target.format = format;
  wgpu::FragmentState fragment{};
  fragment.module = shader;
  fragment.entryPoint = fragment_entry;
  fragment.targetCount = 1;
  fragment.targets = &target;
  wgpu::RenderPipelineDescriptor desc{};
  desc.layout = pipeline_layout;
  desc.vertex.module = shader;
  desc.vertex.entryPoint = vertex_entry;
  desc.fragment = &fragment;
  return device.CreateRenderPipeline(&desc);
}

wgpu::BindGroupLayout CreateBlitLayout(const wgpu::Device& device) {
  std::array<wgpu::BindGroupLayoutEntry, 2> entries{};
  entries[0].binding = 0;
  entries[0].visibility = wgpu::ShaderStage::Fragment;
  entries[0].sampler.type = wgpu::SamplerBindingType::Filtering;
  entries[1] = TextureEntry(1, wgpu::TextureSampleType::Float);
  return CreateLayout(device, entries);
}

wgpu::Sampler CreateLinearSampler(const wgpu::Device& device) {
  wgpu::SamplerDescriptor desc{};
  desc.magFilter = wgpu::FilterMode::Linear;
  desc.minFilter = wgpu::FilterMode::Linear;
  desc.addressModeU = wgpu::AddressMode::ClampToEdge;
  desc.addressModeV = wgpu::AddressMode::ClampToEdge;
  return device.CreateSampler(&desc);
}

wgpu::BindGroup CreateBlitBindGroup(const wgpu::Device& device, const wgpu::BindGroupLayout& layout,
                                    const wgpu::Sampler& sampler, const wgpu::TextureView& view) {
  std::array<wgpu::BindGroupEntry, 2> entries{};
  entries[0].binding = 0;
  entries[0].sampler = sampler;
  entries[1].binding = 1;
  entries[1].textureView = view;
  wgpu::BindGroupDescriptor desc{};
  desc.layout = layout;
  desc.entryCount = entries.size();
  desc.entries = entries.data();
  return device.CreateBindGroup(&desc);
}

// Clears `target` to black and, if `draw` is set, draws one full-viewport
// triangle into `rect`.
void EncodeFitPass(const GpuContext& gpu, const wgpu::TextureView& target, const wgpu::RenderPipeline* pipeline,
                   const wgpu::BindGroup* bind_group, PixelRect rect) {
  wgpu::RenderPassColorAttachment attachment{};
  attachment.view = target;
  attachment.loadOp = wgpu::LoadOp::Clear;
  attachment.storeOp = wgpu::StoreOp::Store;
  attachment.clearValue = {0.0, 0.0, 0.0, 1.0};
  wgpu::RenderPassDescriptor pass_desc{};
  pass_desc.colorAttachmentCount = 1;
  pass_desc.colorAttachments = &attachment;
  const wgpu::CommandEncoder encoder = gpu.device().CreateCommandEncoder();
  const wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&pass_desc);
  if (pipeline != nullptr && rect.width > 0 && rect.height > 0) {
    pass.SetPipeline(*pipeline);
    pass.SetBindGroup(0, *bind_group);
    pass.SetViewport(static_cast<float>(rect.x), static_cast<float>(rect.y), static_cast<float>(rect.width),
                     static_cast<float>(rect.height), 0.0f, 1.0f);
    pass.Draw(3);
  }
  pass.End();
  const wgpu::CommandBuffer commands = encoder.Finish();
  gpu.queue().Submit(1, &commands);
}

}  // namespace

// --- GpuErrorLog ---

void GpuErrorLog::Record(std::string message) {
  std::fprintf(stderr, "WebGPU error: %s\n", message.c_str());
  std::lock_guard lock(mutex_);
  last_ = std::move(message);
  count_.fetch_add(1);
}

std::string GpuErrorLog::last() const {
  std::lock_guard lock(mutex_);
  return last_;
}

void GpuErrorLog::MarkDeviceLost(std::string message) {
  device_lost_ = true;
  Record("device lost: " + message);
}

// --- GpuContext ---

std::unique_ptr<GpuContext> GpuContext::Create(std::string& error) {
  std::unique_ptr<GpuContext> gpu(new GpuContext());
  // TimedWaitAny lets us block on futures (adapter, device, map) with a timeout.
  const wgpu::InstanceFeatureName features[] = {wgpu::InstanceFeatureName::TimedWaitAny};
  wgpu::InstanceDescriptor instance_desc{};
  instance_desc.requiredFeatureCount = 1;
  instance_desc.requiredFeatures = features;
  gpu->instance_ = wgpu::CreateInstance(&instance_desc);
  if (!gpu->instance_) {
    error = "could not create a WebGPU instance";
    return nullptr;
  }

  wgpu::RequestAdapterOptions adapter_options{};
#if defined(__APPLE__)
  adapter_options.backendType = wgpu::BackendType::Metal;
#endif
  adapter_options.powerPreference = wgpu::PowerPreference::HighPerformance;
  std::string adapter_error;
  wgpu::WaitStatus wait = gpu->instance_.WaitAny(
      gpu->instance_.RequestAdapter(
          &adapter_options, wgpu::CallbackMode::WaitAnyOnly,
          [&](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter, wgpu::StringView message) {
            if (status == wgpu::RequestAdapterStatus::Success) {
              gpu->adapter_ = std::move(adapter);
            } else {
              adapter_error = ToStdString(message);
            }
          }),
      UINT64_MAX);
  if (wait != wgpu::WaitStatus::Success || !gpu->adapter_) {
    error = "no GPU adapter: " + adapter_error;
    return nullptr;
  }
  wgpu::AdapterInfo info{};
  gpu->adapter_.GetInfo(&info);
  gpu->adapter_name_ = ToStdString(info.device);

  GpuErrorLog* log = gpu->errors_.get();
  wgpu::DeviceDescriptor device_desc{};
  // Optional: the IOSurface window presenter needs the first two; GPU pass
  // timing needs TimestampQuery. Each is requested only if the adapter has it.
  std::vector<wgpu::FeatureName> device_features;
  gpu->supports_layer_presentation_ = gpu->adapter_.HasFeature(wgpu::FeatureName::SharedTextureMemoryIOSurface) &&
                                      gpu->adapter_.HasFeature(wgpu::FeatureName::SharedFenceMTLSharedEvent);
  if (gpu->supports_layer_presentation_) {
    device_features.push_back(wgpu::FeatureName::SharedTextureMemoryIOSurface);
    device_features.push_back(wgpu::FeatureName::SharedFenceMTLSharedEvent);
  }
  gpu->supports_timestamps_ = gpu->adapter_.HasFeature(wgpu::FeatureName::TimestampQuery);
  if (gpu->supports_timestamps_) {
    device_features.push_back(wgpu::FeatureName::TimestampQuery);
  }
  device_desc.requiredFeatureCount = device_features.size();
  device_desc.requiredFeatures = device_features.data();
  device_desc.SetDeviceLostCallback(
      wgpu::CallbackMode::AllowSpontaneous,
      [log](const wgpu::Device&, wgpu::DeviceLostReason reason, wgpu::StringView message) {
        if (reason != wgpu::DeviceLostReason::Destroyed && reason != wgpu::DeviceLostReason::CallbackCancelled) {
          log->MarkDeviceLost(ToStdString(message));
        }
      });
  device_desc.SetUncapturedErrorCallback(
      [](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView message, GpuErrorLog* target) {
        target->Record(ToStdString(message));
      },
      log);
  std::string device_error;
  wait = gpu->instance_.WaitAny(
      gpu->adapter_.RequestDevice(
          &device_desc, wgpu::CallbackMode::WaitAnyOnly,
          [&](wgpu::RequestDeviceStatus status, wgpu::Device device, wgpu::StringView message) {
            if (status == wgpu::RequestDeviceStatus::Success) {
              gpu->device_ = std::move(device);
            } else {
              device_error = ToStdString(message);
            }
          }),
      UINT64_MAX);
  if (wait != wgpu::WaitStatus::Success || !gpu->device_) {
    error = "could not create a GPU device: " + device_error;
    return nullptr;
  }
  gpu->queue_ = gpu->device_.GetQueue();
  return gpu;
}

bool GpuContext::WaitIdle() {
  bool done = false;
  const wgpu::WaitStatus wait = instance_.WaitAny(
      queue_.OnSubmittedWorkDone(wgpu::CallbackMode::WaitAnyOnly,
                                 [&done](wgpu::QueueWorkDoneStatus status, wgpu::StringView) {
                                   done = status == wgpu::QueueWorkDoneStatus::Success;
                                 }),
      UINT64_MAX);
  return wait == wgpu::WaitStatus::Success && done;
}

// --- OutputTexture ---

OutputTexture::OutputTexture(const GpuContext& gpu)
    : texture_(CreateTexture(gpu.device(), kWidth, kHeight, kFormat,
                             wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding |
                                 wgpu::TextureUsage::CopySrc,
                             "relume clean output")),
      view_(texture_.CreateView()) {}

// --- FrameUploader ---

FrameUploader::FrameUploader(const GpuContext& gpu) : gpu_(gpu) {
  const wgpu::Device& device = gpu.device();
  shader_ = CreateShader(device, kConvertWgsl, "relume convert");
  rgba_layout_ = CreateLayout(device, std::array{TextureEntry(0, wgpu::TextureSampleType::UnfilterableFloat)});
  std::array<wgpu::BindGroupLayoutEntry, 3> nv12_entries = {
      TextureEntry(0, wgpu::TextureSampleType::UnfilterableFloat),
      TextureEntry(1, wgpu::TextureSampleType::UnfilterableFloat), wgpu::BindGroupLayoutEntry{}};
  nv12_entries[2].binding = 2;
  nv12_entries[2].visibility = wgpu::ShaderStage::Fragment;
  nv12_entries[2].buffer.type = wgpu::BufferBindingType::Uniform;
  nv12_layout_ = CreateLayout(device, nv12_entries);
  rgba_pipeline_ = CreatePipeline(device, rgba_layout_, shader_, "vs_fullscreen", "fs_rgba",
                                  wgpu::TextureFormat::RGBA8Unorm);
  nv12_pipeline_ = CreatePipeline(device, nv12_layout_, shader_, "vs_fullscreen", "fs_nv12",
                                  wgpu::TextureFormat::RGBA8Unorm);
  wgpu::BufferDescriptor params_desc{};
  params_desc.size = 32;
  params_desc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
  nv12_params_ = device.CreateBuffer(&params_desc);
}

void FrameUploader::EnsureTextures(const VideoFrame& frame) {
  if (camera_texture_ && frame.format == format_ && frame.width == width_ && frame.height == height_) {
    return;
  }
  const wgpu::Device& device = gpu_.device();
  const auto input_usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
  format_ = frame.format;
  width_ = frame.width;
  height_ = frame.height;
  input1_ = nullptr;
  wgpu::BindGroupDescriptor desc{};
  std::array<wgpu::BindGroupEntry, 3> entries{};
  switch (frame.format) {
    case PixelFormat::kBgra8:
    case PixelFormat::kRgb8:
      input0_ = CreateTexture(device, width_, height_,
                              frame.format == PixelFormat::kBgra8 ? wgpu::TextureFormat::BGRA8Unorm
                                                                  : wgpu::TextureFormat::RGBA8Unorm,
                              input_usage, "relume camera input");
      entries[0].binding = 0;
      entries[0].textureView = input0_.CreateView();
      desc.layout = rgba_layout_;
      desc.entryCount = 1;
      break;
    case PixelFormat::kNv12:
      input0_ = CreateTexture(device, width_, height_, wgpu::TextureFormat::R8Unorm, input_usage, "relume luma");
      input1_ = CreateTexture(device, frame.planes[1].width, frame.planes[1].height, wgpu::TextureFormat::RG8Unorm,
                              input_usage, "relume chroma");
      entries[0].binding = 0;
      entries[0].textureView = input0_.CreateView();
      entries[1].binding = 1;
      entries[1].textureView = input1_.CreateView();
      entries[2].binding = 2;
      entries[2].buffer = nv12_params_;
      entries[2].size = 20;
      desc.layout = nv12_layout_;
      desc.entryCount = 3;
      break;
  }
  desc.entries = entries.data();
  bind_group_ = device.CreateBindGroup(&desc);
  camera_texture_ = CreateTexture(device, width_, height_, wgpu::TextureFormat::RGBA8Unorm,
                                  wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding |
                                      wgpu::TextureUsage::CopySrc,
                                  "relume camera texture");
  camera_view_ = camera_texture_.CreateView();
}

bool FrameUploader::Upload(const VideoFrame& frame) {
  if (!ValidateLayout(frame).empty()) {
    return false;
  }
  EnsureTextures(frame);
  const wgpu::Queue& queue = gpu_.queue();
  const wgpu::RenderPipeline* pipeline = &rgba_pipeline_;
  switch (frame.format) {
    case PixelFormat::kBgra8:
      WritePlane(queue, input0_, frame, 0);
      break;
    case PixelFormat::kRgb8: {
      // There is no 3-byte texture format; expand to RGBA on the CPU.
      const PlaneLayout& plane = frame.planes[0];
      rgba_scratch_.resize(std::size_t{frame.width} * frame.height * 4);
      const auto* base = reinterpret_cast<const std::uint8_t*>(frame.data.data()) + plane.offset;
      for (std::uint32_t y = 0; y < frame.height; ++y) {
        const std::uint8_t* src = base + plane.stride_bytes * y;
        std::uint8_t* dst = rgba_scratch_.data() + std::size_t{y} * frame.width * 4;
        for (std::uint32_t x = 0; x < frame.width; ++x) {
          std::memcpy(dst + x * 4, src + x * 3, 3);
          dst[x * 4 + 3] = 255;
        }
      }
      wgpu::TexelCopyTextureInfo destination{};
      destination.texture = input0_;
      wgpu::TexelCopyBufferLayout layout{};
      layout.bytesPerRow = frame.width * 4;
      layout.rowsPerImage = frame.height;
      const wgpu::Extent3D size = {frame.width, frame.height, 1};
      queue.WriteTexture(&destination, rgba_scratch_.data(), rgba_scratch_.size(), &layout, &size);
      break;
    }
    case PixelFormat::kNv12: {
      WritePlane(queue, input0_, frame, 0);
      WritePlane(queue, input1_, frame, 1);
      const bool bt601 = frame.matrix == YuvMatrix::kBt601;
      const bool video = frame.range == YuvRange::kVideo;
      // Matches Nv12Params in shaders/convert.wgsl.
      const std::array<float, 8> params = {bt601 ? 0.299f : 0.2126f, bt601 ? 0.114f : 0.0722f,
                                           video ? 16.0f : 0.0f, video ? 219.0f : 255.0f,
                                           video ? 224.0f : 255.0f, 0.0f, 0.0f, 0.0f};
      queue.WriteBuffer(nv12_params_, 0, params.data(), sizeof(params));
      pipeline = &nv12_pipeline_;
      break;
    }
  }
  EncodeFitPass(gpu_, camera_view_, pipeline, &bind_group_, PixelRect{0, 0, width_, height_});
  return true;
}

// --- OutputCompositor ---

OutputCompositor::OutputCompositor(const GpuContext& gpu, const OutputTexture& output)
    : gpu_(gpu), output_(output) {
  layout_ = CreateBlitLayout(gpu.device());
  pipeline_ = CreatePipeline(gpu.device(), layout_, CreateShader(gpu.device(), kBlitWgsl, "relume blit"), "vs_main",
                             "fs_main", OutputTexture::kFormat);
  sampler_ = CreateLinearSampler(gpu.device());
}

void OutputCompositor::Compose(const wgpu::Texture& camera_texture) {
  if (!camera_texture) {
    ClearToBlack();
    return;
  }
  if (bound_texture_.Get() != camera_texture.Get()) {
    bound_texture_ = camera_texture;
    bind_group_ = CreateBlitBindGroup(gpu_.device(), layout_, sampler_, camera_texture.CreateView());
  }
  const PixelRect rect = FitRect(camera_texture.GetWidth(), camera_texture.GetHeight(), OutputTexture::kWidth,
                                 OutputTexture::kHeight);
  EncodeFitPass(gpu_, output_.view(), &pipeline_, &bind_group_, rect);
}

void OutputCompositor::ClearToBlack() { EncodeFitPass(gpu_, output_.view(), nullptr, nullptr, {}); }

// --- Presenter ---

Presenter::Presenter(const GpuContext& gpu, const OutputTexture& output) : gpu_(gpu) {
  layout_ = CreateBlitLayout(gpu.device());
  shader_ = CreateShader(gpu.device(), kBlitWgsl, "relume present");
  bind_group_ = CreateBlitBindGroup(gpu.device(), layout_, CreateLinearSampler(gpu.device()), output.view());
}

const wgpu::RenderPipeline& Presenter::PipelineFor(wgpu::TextureFormat format) {
  for (const auto& [cached_format, pipeline] : pipelines_) {
    if (cached_format == format) {
      return pipeline;
    }
  }
  pipelines_.emplace_back(format, CreatePipeline(gpu_.device(), layout_, shader_, "vs_main", "fs_main", format));
  return pipelines_.back().second;
}

void Presenter::Present(const wgpu::TextureView& target, wgpu::TextureFormat target_format,
                        std::uint32_t target_width, std::uint32_t target_height) {
  const PixelRect rect = FitRect(OutputTexture::kWidth, OutputTexture::kHeight, target_width, target_height);
  EncodeFitPass(gpu_, target, &PipelineFor(target_format), &bind_group_, rect);
}

// --- Depth debug view ---

DepthDebugTexture::DepthDebugTexture(const GpuContext& gpu)
    : texture_(CreateTexture(gpu.device(), kWidth, kHeight, kFormat,
                             wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding |
                                 wgpu::TextureUsage::CopySrc,
                             "relume depth debug view")),
      view_(texture_.CreateView()) {}

DepthUploader::DepthUploader(const GpuContext& gpu) : gpu_(gpu) {}

bool DepthUploader::Upload(const float* depth, const ResizeTransform& t) {
  if (depth == nullptr || t.new_w == 0 || t.new_h == 0 || t.pad_x + t.new_w > kNetSize ||
      t.pad_y + t.new_h > kNetSize) {
    return false;
  }
  if (!texture_ || texture_.GetWidth() != t.new_w || texture_.GetHeight() != t.new_h) {
    texture_ = CreateTexture(gpu_.device(), t.new_w, t.new_h, wgpu::TextureFormat::R32Float,
                             wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst |
                                 wgpu::TextureUsage::CopySrc,
                             "relume depth");
  }
  wgpu::TexelCopyTextureInfo destination{};
  destination.texture = texture_;
  wgpu::TexelCopyBufferLayout layout{};
  layout.offset = (std::uint64_t{t.pad_y} * kNetSize + t.pad_x) * sizeof(float);
  layout.bytesPerRow = kNetSize * sizeof(float);  // 1792, the full tensor row
  layout.rowsPerImage = t.new_h;
  const wgpu::Extent3D size = {t.new_w, t.new_h, 1};
  gpu_.queue().WriteTexture(&destination, depth, kNetDepthFloats * sizeof(float), &layout, &size);
  return true;
}

DepthVisualizer::DepthVisualizer(const GpuContext& gpu, const DepthDebugTexture& target) : gpu_(gpu), target_(target) {
  const wgpu::Device& device = gpu.device();
  std::array<wgpu::BindGroupLayoutEntry, 4> entries = {
      TextureEntry(0, wgpu::TextureSampleType::UnfilterableFloat), wgpu::BindGroupLayoutEntry{},
      wgpu::BindGroupLayoutEntry{}, TextureEntry(3, wgpu::TextureSampleType::Float)};
  entries[1].binding = 1;
  entries[1].visibility = wgpu::ShaderStage::Fragment;
  entries[1].buffer.type = wgpu::BufferBindingType::Uniform;
  entries[2].binding = 2;
  entries[2].visibility = wgpu::ShaderStage::Fragment;
  entries[2].sampler.type = wgpu::SamplerBindingType::Filtering;
  layout_ = CreateLayout(device, entries);
  pipeline_ = CreatePipeline(device, layout_, CreateShader(device, kDepthVisWgsl, "relume depth view"), "vs_main",
                             "fs_main", DepthDebugTexture::kFormat);
  sampler_ = CreateLinearSampler(device);
  wgpu::BufferDescriptor params_desc{};
  params_desc.size = 64;
  params_desc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
  params_ = device.CreateBuffer(&params_desc);
}

void DepthVisualizer::Draw(const DepthUploader& depth, const ResizeTransform& t, const DepthVisParams& params,
                           const wgpu::Texture& camera_texture) {
  if (!depth.texture() || !camera_texture || depth.texture().GetWidth() != t.new_w ||
      depth.texture().GetHeight() != t.new_h) {
    Clear();
    return;
  }
  if (bound_depth_.Get() != depth.texture().Get() || bound_camera_.Get() != camera_texture.Get()) {
    bound_depth_ = depth.texture();
    bound_camera_ = camera_texture;
    std::array<wgpu::BindGroupEntry, 4> entries{};
    entries[0].binding = 0;
    entries[0].textureView = bound_depth_.CreateView();
    entries[1].binding = 1;
    entries[1].buffer = params_;
    entries[1].size = 64;
    entries[2].binding = 2;
    entries[2].sampler = sampler_;
    entries[3].binding = 3;
    entries[3].textureView = bound_camera_.CreateView();
    wgpu::BindGroupDescriptor desc{};
    desc.layout = layout_;
    desc.entryCount = entries.size();
    desc.entries = entries.data();
    bind_group_ = gpu_.device().CreateBindGroup(&desc);
  }
  const PixelRect rect = FitRect(t.src_w, t.src_h, DepthDebugTexture::kWidth, DepthDebugTexture::kHeight);
  // Matches Params in shaders/depth_vis.wgsl.
  const std::array<float, 16> values = {
      static_cast<float>(rect.x), static_cast<float>(rect.y), static_cast<float>(rect.width),
      static_cast<float>(rect.height), static_cast<float>(t.src_w), static_cast<float>(t.src_h),
      static_cast<float>(t.sx()), static_cast<float>(t.sy()), static_cast<float>(t.new_w),
      static_cast<float>(t.new_h), params.lo, params.hi, params.valid ? 1.0f : 0.0f, params.camera_weight,
      0.0f, 0.0f};
  gpu_.queue().WriteBuffer(params_, 0, values.data(), sizeof(values));
  EncodeFitPass(gpu_, target_.view(), &pipeline_, &bind_group_, rect);
}

void DepthVisualizer::Clear() { EncodeFitPass(gpu_, target_.view(), nullptr, nullptr, {}); }

// --- Readback ---

namespace {

// Copies `texture` (4 bytes per texel) into `out` as tightly packed rows.
bool ReadbackBytes(GpuContext& gpu, const wgpu::Texture& texture, std::vector<std::uint8_t>& out) {
  const std::uint32_t width = texture.GetWidth();
  const std::uint32_t height = texture.GetHeight();
  const std::uint32_t row_bytes = width * 4;
  const std::uint32_t padded_row = (row_bytes + 255) / 256 * 256;  // copy alignment rule
  wgpu::BufferDescriptor buffer_desc{};
  buffer_desc.size = std::uint64_t{padded_row} * height;
  buffer_desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
  const wgpu::Buffer buffer = gpu.device().CreateBuffer(&buffer_desc);

  wgpu::TexelCopyTextureInfo source{};
  source.texture = texture;
  wgpu::TexelCopyBufferInfo destination{};
  destination.buffer = buffer;
  destination.layout.bytesPerRow = padded_row;
  destination.layout.rowsPerImage = height;
  const wgpu::Extent3D size = {width, height, 1};
  const wgpu::CommandEncoder encoder = gpu.device().CreateCommandEncoder();
  encoder.CopyTextureToBuffer(&source, &destination, &size);
  const wgpu::CommandBuffer commands = encoder.Finish();
  gpu.queue().Submit(1, &commands);

  bool mapped = false;
  const wgpu::WaitStatus wait = gpu.instance().WaitAny(
      buffer.MapAsync(wgpu::MapMode::Read, 0, buffer_desc.size, wgpu::CallbackMode::WaitAnyOnly,
                      [&mapped](wgpu::MapAsyncStatus status, wgpu::StringView) {
                        mapped = status == wgpu::MapAsyncStatus::Success;
                      }),
      UINT64_MAX);
  if (wait != wgpu::WaitStatus::Success || !mapped) {
    return false;
  }
  const auto* data = static_cast<const std::uint8_t*>(buffer.GetConstMappedRange(0, buffer_desc.size));
  out.resize(std::size_t{row_bytes} * height);
  for (std::uint32_t y = 0; y < height; ++y) {
    std::memcpy(out.data() + std::size_t{y} * row_bytes, data + std::size_t{y} * padded_row, row_bytes);
  }
  buffer.Unmap();
  return true;
}

}  // namespace

bool ReadbackTexture(GpuContext& gpu, const wgpu::Texture& texture, std::vector<std::uint8_t>& pixels) {
  return ReadbackBytes(gpu, texture, pixels);
}

bool ReadbackFloatTexture(GpuContext& gpu, const wgpu::Texture& texture, std::vector<float>& values) {
  std::vector<std::uint8_t> bytes;
  if (texture.GetFormat() != wgpu::TextureFormat::R32Float || !ReadbackBytes(gpu, texture, bytes)) {
    return false;
  }
  values.resize(bytes.size() / sizeof(float));
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return true;
}

}  // namespace relume
