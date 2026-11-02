#include "relume/renderer.hpp"

#include "relume/layout.hpp"
#include "shaders/blit_wgsl.hpp"
#include "shaders/convert_wgsl.hpp"
#include "shaders/depth_vis_wgsl.hpp"
#include "shaders/normals_wgsl.hpp"
#include "shaders/relight_wgsl.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
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

// --- Relighting ---

namespace {

wgpu::BindGroupLayoutEntry UniformEntry(std::uint32_t binding) {
  wgpu::BindGroupLayoutEntry entry{};
  entry.binding = binding;
  entry.visibility = wgpu::ShaderStage::Fragment;
  entry.buffer.type = wgpu::BufferBindingType::Uniform;
  return entry;
}

wgpu::Buffer CreateUniformBuffer(const wgpu::Device& device, std::uint64_t size) {
  wgpu::BufferDescriptor desc{};
  desc.size = size;
  desc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
  return device.CreateBuffer(&desc);
}

}  // namespace

DepthFieldTexture::DepthFieldTexture(const GpuContext& gpu) : gpu_(gpu) {}

void DepthFieldTexture::Ensure(std::uint32_t width, std::uint32_t height) {
  if (texture_ && width == width_ && height == height_) {
    return;
  }
  texture_ = CreateTexture(gpu_.device(), width, height, kFormat,
                           wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding |
                               wgpu::TextureUsage::CopySrc,
                           "relume depth field");
  view_ = texture_.CreateView();
  width_ = width;
  height_ = height;
}

NormalPass::NormalPass(const GpuContext& gpu) : gpu_(gpu) {
  const wgpu::Device& device = gpu.device();
  const wgpu::ShaderModule shader = CreateShader(device, kNormalsWgsl, "relume depth field");
  build_layout_ = CreateLayout(device, std::array{TextureEntry(0, wgpu::TextureSampleType::UnfilterableFloat),
                                                  UniformEntry(1)});
  blend_layout_ = CreateLayout(device, std::array{TextureEntry(0, wgpu::TextureSampleType::UnfilterableFloat),
                                                  TextureEntry(1, wgpu::TextureSampleType::UnfilterableFloat),
                                                  UniformEntry(2)});
  build_pipeline_ = CreatePipeline(device, build_layout_, shader, "vs_main", "fs_main", DepthFieldTexture::kFormat);
  blend_pipeline_ = CreatePipeline(device, blend_layout_, shader, "vs_main", "fs_blend", DepthFieldTexture::kFormat);
  build_params_ = CreateUniformBuffer(device, 32);
  blend_params_ = CreateUniformBuffer(device, 16);
}

bool NormalPass::Build(const DepthUploader& depth, float scale, float offset, float aspect, DepthFieldTexture& out) {
  const wgpu::Texture& input = depth.texture();
  if (!input) {
    return false;
  }
  const std::uint32_t width = input.GetWidth();
  const std::uint32_t height = input.GetHeight();
  out.Ensure(width, height);
  // Matches Params in shaders/normals.wgsl.
  const std::array<float, 8> values = {static_cast<float>(width), static_cast<float>(height), scale, offset,
                                       2.0f * aspect / static_cast<float>(width), -2.0f / static_cast<float>(height),
                                       0.0f, 0.0f};
  gpu_.queue().WriteBuffer(build_params_, 0, values.data(), sizeof(values));
  std::array<wgpu::BindGroupEntry, 2> entries{};
  entries[0].binding = 0;
  entries[0].textureView = input.CreateView();
  entries[1].binding = 1;
  entries[1].buffer = build_params_;
  entries[1].size = 32;
  wgpu::BindGroupDescriptor desc{};
  desc.layout = build_layout_;
  desc.entryCount = entries.size();
  desc.entries = entries.data();
  const wgpu::BindGroup bind_group = gpu_.device().CreateBindGroup(&desc);
  EncodeFitPass(gpu_, out.view(), &build_pipeline_, &bind_group, PixelRect{0, 0, width, height});
  return true;
}

bool NormalPass::Blend(const DepthFieldTexture& from, const DepthFieldTexture& to, float t, DepthFieldTexture& out) {
  if (!from.texture() || !to.texture() || from.width() != to.width() || from.height() != to.height() ||
      &out == &from || &out == &to) {
    return false;
  }
  out.Ensure(from.width(), from.height());
  const std::array<float, 4> values = {std::clamp(t, 0.0f, 1.0f), 0.0f, 0.0f, 0.0f};
  gpu_.queue().WriteBuffer(blend_params_, 0, values.data(), sizeof(values));
  std::array<wgpu::BindGroupEntry, 3> entries{};
  entries[0].binding = 0;
  entries[0].textureView = from.view();
  entries[1].binding = 1;
  entries[1].textureView = to.view();
  entries[2].binding = 2;
  entries[2].buffer = blend_params_;
  entries[2].size = 16;
  wgpu::BindGroupDescriptor desc{};
  desc.layout = blend_layout_;
  desc.entryCount = entries.size();
  desc.entries = entries.data();
  const wgpu::BindGroup bind_group = gpu_.device().CreateBindGroup(&desc);
  EncodeFitPass(gpu_, out.view(), &blend_pipeline_, &bind_group, PixelRect{0, 0, from.width(), from.height()});
  return true;
}

Relighter::Relighter(const GpuContext& gpu, const OutputTexture& output) : gpu_(gpu), output_(output) {
  const wgpu::Device& device = gpu.device();
  std::array<wgpu::BindGroupLayoutEntry, 5> entries = {
      wgpu::BindGroupLayoutEntry{}, TextureEntry(1, wgpu::TextureSampleType::Float),
      TextureEntry(2, wgpu::TextureSampleType::Float), TextureEntry(3, wgpu::TextureSampleType::Float),
      UniformEntry(4)};
  entries[0].binding = 0;
  entries[0].visibility = wgpu::ShaderStage::Fragment;
  entries[0].sampler.type = wgpu::SamplerBindingType::Filtering;
  layout_ = CreateLayout(device, entries);
  // OutputTexture and DepthDebugTexture share one non-sRGB format.
  static_assert(OutputTexture::kFormat == DepthDebugTexture::kFormat);
  pipeline_ = CreatePipeline(device, layout_, CreateShader(device, kRelightWgsl, "relume relight"), "vs_main",
                             "fs_main", OutputTexture::kFormat);
  sampler_ = CreateLinearSampler(device);
  output_target_.params = CreateUniformBuffer(device, 64);
  debug_target_.params = CreateUniformBuffer(device, 64);
}

bool Relighter::Render(const wgpu::Texture& camera_texture, const RelightFrame& frame) {
  return Draw(output_target_, output_.view(), camera_texture, frame, RelightView::kRelit);
}

bool Relighter::RenderDebug(const wgpu::Texture& camera_texture, const RelightFrame& frame, RelightView view,
                            const DepthDebugTexture& target) {
  return Draw(debug_target_, target.view(), camera_texture, frame, view);
}

bool Relighter::Draw(Target& target, const wgpu::TextureView& view, const wgpu::Texture& camera_texture,
                     const RelightFrame& frame, RelightView mode) {
  if (!camera_texture || frame.from == nullptr || frame.to == nullptr || !frame.from->texture() ||
      !frame.to->texture() || frame.from->width() != frame.to->width() ||
      frame.from->height() != frame.to->height()) {
    return false;
  }
  const wgpu::Texture& from = frame.from->texture();
  const wgpu::Texture& to = frame.to->texture();
  if (target.camera.Get() != camera_texture.Get() || target.from.Get() != from.Get() || target.to.Get() != to.Get()) {
    target.camera = camera_texture;
    target.from = from;
    target.to = to;
    std::array<wgpu::BindGroupEntry, 5> entries{};
    entries[0].binding = 0;
    entries[0].sampler = sampler_;
    entries[1].binding = 1;
    entries[1].textureView = camera_texture.CreateView();
    entries[2].binding = 2;
    entries[2].textureView = frame.from->view();
    entries[3].binding = 3;
    entries[3].textureView = frame.to->view();
    entries[4].binding = 4;
    entries[4].buffer = target.params;
    entries[4].size = 64;
    wgpu::BindGroupDescriptor desc{};
    desc.layout = layout_;
    desc.entryCount = entries.size();
    desc.entries = entries.data();
    target.bind_group = gpu_.device().CreateBindGroup(&desc);
  }
  const float aspect = static_cast<float>(camera_texture.GetWidth()) / static_cast<float>(camera_texture.GetHeight());
  const LightSettings s = frame.light.Clamped();
  const Vec3 light = LightPosition(s, aspect);
  const Vec3 tint = TintColor(s.tint);
  // Matches Light in shaders/relight.wgsl.
  const std::array<float, 16> values = {light.x,
                                        light.y,
                                        light.z,
                                        aspect,
                                        s.intensity,
                                        s.ambient,
                                        s.specular,
                                        s.relief,
                                        tint.x,
                                        tint.y,
                                        tint.z,
                                        s.z,
                                        std::clamp(frame.blend, 0.0f, 1.0f),
                                        std::clamp(frame.fade, 0.0f, 1.0f),
                                        static_cast<float>(mode),
                                        0.0f};
  gpu_.queue().WriteBuffer(target.params, 0, values.data(), sizeof(values));
  const PixelRect rect = FitRect(camera_texture.GetWidth(), camera_texture.GetHeight(), OutputTexture::kWidth,
                                 OutputTexture::kHeight);
  EncodeFitPass(gpu_, view, &pipeline_, &target.bind_group, rect);
  return true;
}

// --- Readback ---

namespace {

// Copies `texture` into `out` as tightly packed rows.
bool ReadbackBytes(GpuContext& gpu, const wgpu::Texture& texture, std::vector<std::uint8_t>& out,
                   std::uint32_t bytes_per_texel = 4) {
  const std::uint32_t width = texture.GetWidth();
  const std::uint32_t height = texture.GetHeight();
  const std::uint32_t row_bytes = width * bytes_per_texel;
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

namespace {

float HalfToFloat(std::uint16_t h) {
  const int exponent = (h >> 10) & 0x1f;
  const int mantissa = h & 0x3ff;
  float value = 0.0f;
  if (exponent == 0) {
    value = std::ldexp(static_cast<float>(mantissa), -24);
  } else if (exponent == 31) {
    value = mantissa == 0 ? std::numeric_limits<float>::infinity() : std::numeric_limits<float>::quiet_NaN();
  } else {
    value = std::ldexp(static_cast<float>(mantissa + 1024), exponent - 25);
  }
  return (h & 0x8000) != 0 ? -value : value;
}

}  // namespace

bool ReadbackFieldTexture(GpuContext& gpu, const DepthFieldTexture& field, std::vector<float>& values) {
  std::vector<std::uint8_t> bytes;
  if (!field.texture() || !ReadbackBytes(gpu, field.texture(), bytes, 8)) {
    return false;
  }
  values.resize(bytes.size() / 2);
  for (std::size_t i = 0; i < values.size(); ++i) {
    std::uint16_t h = 0;
    std::memcpy(&h, bytes.data() + i * 2, 2);
    values[i] = HalfToFloat(h);
  }
  return true;
}

}  // namespace relume
