// Phase 0 probe: Dawn/WebGPU draws a textured quad.
//
//   relume_dawn_hello               open a window and draw until it is closed
//   relume_dawn_hello --frames N    open a window, draw N frames, then exit
//   relume_dawn_hello --offscreen   render into a texture, read it back and
//                                   check the pixels (used by CTest)
//
// The texture is an 8x8 checkerboard generated on the CPU and uploaded with
// Queue::WriteTexture; the quad covers the middle 80% of the target.

#include "metal_layer.hpp"

#include <GLFW/glfw3.h>
#include <webgpu/webgpu_cpp.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string_view>
#include <vector>

namespace {

constexpr uint32_t kCheckerCells = 8;
constexpr std::array<uint8_t, 4> kColorA = {255, 128, 0, 255};  // orange
constexpr std::array<uint8_t, 4> kColorB = {0, 64, 255, 255};   // blue
constexpr wgpu::Color kClearColor = {0.0, 0.0, 0.0, 1.0};
constexpr float kQuadExtent = 0.8f;  // quad spans [-0.8, 0.8] in clip space

constexpr char kShader[] = R"(
struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) uv : vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) index : u32) -> VertexOut {
  // Two triangles forming a quad; uv (0,0) is the top-left of the texture.
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let corner = corners[index];
  var out : VertexOut;
  out.position = vec4f(corner * 0.8, 0.0, 1.0);
  out.uv = vec2f(corner.x * 0.5 + 0.5, 0.5 - corner.y * 0.5);
  return out;
}

@group(0) @binding(0) var checker_sampler : sampler;
@group(0) @binding(1) var checker_texture : texture_2d<f32>;

@fragment
fn fs_main(in : VertexOut) -> @location(0) vec4f {
  return textureSample(checker_texture, checker_sampler, in.uv);
}
)";

std::array<uint8_t, 4> ExpectedCellColor(uint32_t x, uint32_t y) {
  return ((x + y) % 2 == 0) ? kColorA : kColorB;
}

void PrintMessage(const char* prefix, wgpu::StringView message) {
  const std::string_view text = message;
  std::fprintf(stderr, "%s: %.*s\n", prefix, static_cast<int>(text.size()), text.data());
}

// Holds the GPU objects shared by both modes. All members are reference-counted
// wgpu handles that release their object when destroyed (RAII).
struct Gpu {
  wgpu::Instance instance;
  wgpu::Adapter adapter;
  wgpu::Device device;
  wgpu::Queue queue;
  wgpu::BindGroup bind_group;
  wgpu::ShaderModule shader;
  wgpu::BindGroupLayout bind_group_layout;
  wgpu::PipelineLayout pipeline_layout;
};

std::optional<Gpu> CreateGpu() {
  Gpu gpu;
  // TimedWaitAny lets us block on futures (adapter/device/map) with a timeout.
  const wgpu::InstanceFeatureName features[] = {wgpu::InstanceFeatureName::TimedWaitAny};
  wgpu::InstanceDescriptor instance_desc{};
  instance_desc.requiredFeatureCount = 1;
  instance_desc.requiredFeatures = features;
  gpu.instance = wgpu::CreateInstance(&instance_desc);
  if (!gpu.instance) {
    std::fprintf(stderr, "FAIL: could not create a WebGPU instance\n");
    return std::nullopt;
  }

  wgpu::RequestAdapterOptions adapter_options{};
  adapter_options.backendType = wgpu::BackendType::Metal;
  adapter_options.powerPreference = wgpu::PowerPreference::HighPerformance;
  wgpu::WaitStatus wait = gpu.instance.WaitAny(
      gpu.instance.RequestAdapter(
          &adapter_options, wgpu::CallbackMode::WaitAnyOnly,
          [&gpu](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter, wgpu::StringView message) {
            if (status != wgpu::RequestAdapterStatus::Success) {
              PrintMessage("RequestAdapter failed", message);
              return;
            }
            gpu.adapter = std::move(adapter);
          }),
      UINT64_MAX);
  if (wait != wgpu::WaitStatus::Success || !gpu.adapter) {
    std::fprintf(stderr, "FAIL: no Metal adapter\n");
    return std::nullopt;
  }

  wgpu::AdapterInfo info{};
  gpu.adapter.GetInfo(&info);
  const std::string_view device_name = info.device;
  const std::string_view description = info.description;
  std::printf("Adapter: %.*s (%.*s), backend Metal\n", static_cast<int>(device_name.size()),
              device_name.data(), static_cast<int>(description.size()), description.data());

  wgpu::DeviceDescriptor device_desc{};
  device_desc.SetDeviceLostCallback(
      wgpu::CallbackMode::AllowSpontaneous,
      [](const wgpu::Device&, wgpu::DeviceLostReason reason, wgpu::StringView message) {
        if (reason != wgpu::DeviceLostReason::Destroyed &&
            reason != wgpu::DeviceLostReason::CallbackCancelled) {
          PrintMessage("Device lost", message);
        }
      });
  device_desc.SetUncapturedErrorCallback(
      [](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView message) {
        PrintMessage("WebGPU validation error", message);
        std::exit(1);  // A probe must never report PASS after a GPU error.
      });
  wait = gpu.instance.WaitAny(
      gpu.adapter.RequestDevice(
          &device_desc, wgpu::CallbackMode::WaitAnyOnly,
          [&gpu](wgpu::RequestDeviceStatus status, wgpu::Device device, wgpu::StringView message) {
            if (status != wgpu::RequestDeviceStatus::Success) {
              PrintMessage("RequestDevice failed", message);
              return;
            }
            gpu.device = std::move(device);
          }),
      UINT64_MAX);
  if (wait != wgpu::WaitStatus::Success || !gpu.device) {
    std::fprintf(stderr, "FAIL: could not create a device\n");
    return std::nullopt;
  }
  gpu.queue = gpu.device.GetQueue();

  // Upload the checkerboard texture.
  std::vector<uint8_t> pixels(kCheckerCells * kCheckerCells * 4);
  for (uint32_t y = 0; y < kCheckerCells; ++y) {
    for (uint32_t x = 0; x < kCheckerCells; ++x) {
      std::memcpy(&pixels[(y * kCheckerCells + x) * 4], ExpectedCellColor(x, y).data(), 4);
    }
  }
  wgpu::TextureDescriptor texture_desc{};
  texture_desc.size = {kCheckerCells, kCheckerCells, 1};
  texture_desc.format = wgpu::TextureFormat::RGBA8Unorm;
  texture_desc.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
  wgpu::Texture texture = gpu.device.CreateTexture(&texture_desc);
  wgpu::TexelCopyTextureInfo destination{};
  destination.texture = texture;
  wgpu::TexelCopyBufferLayout layout{};
  layout.bytesPerRow = kCheckerCells * 4;
  layout.rowsPerImage = kCheckerCells;
  gpu.queue.WriteTexture(&destination, pixels.data(), pixels.size(), &layout, &texture_desc.size);

  // Nearest filtering keeps the checker edges sharp so readback is exact.
  wgpu::SamplerDescriptor sampler_desc{};
  sampler_desc.magFilter = wgpu::FilterMode::Nearest;
  sampler_desc.minFilter = wgpu::FilterMode::Nearest;
  wgpu::Sampler sampler = gpu.device.CreateSampler(&sampler_desc);

  wgpu::ShaderSourceWGSL wgsl{};
  wgsl.code = kShader;
  wgpu::ShaderModuleDescriptor shader_desc{};
  shader_desc.nextInChain = &wgsl;
  gpu.shader = gpu.device.CreateShaderModule(&shader_desc);

  std::array<wgpu::BindGroupLayoutEntry, 2> layout_entries{};
  layout_entries[0].binding = 0;
  layout_entries[0].visibility = wgpu::ShaderStage::Fragment;
  layout_entries[0].sampler.type = wgpu::SamplerBindingType::Filtering;
  layout_entries[1].binding = 1;
  layout_entries[1].visibility = wgpu::ShaderStage::Fragment;
  layout_entries[1].texture.sampleType = wgpu::TextureSampleType::Float;
  layout_entries[1].texture.viewDimension = wgpu::TextureViewDimension::e2D;
  wgpu::BindGroupLayoutDescriptor bgl_desc{};
  bgl_desc.entryCount = layout_entries.size();
  bgl_desc.entries = layout_entries.data();
  gpu.bind_group_layout = gpu.device.CreateBindGroupLayout(&bgl_desc);

  wgpu::PipelineLayoutDescriptor pl_desc{};
  pl_desc.bindGroupLayoutCount = 1;
  pl_desc.bindGroupLayouts = &gpu.bind_group_layout;
  gpu.pipeline_layout = gpu.device.CreatePipelineLayout(&pl_desc);

  std::array<wgpu::BindGroupEntry, 2> entries{};
  entries[0].binding = 0;
  entries[0].sampler = sampler;
  entries[1].binding = 1;
  entries[1].textureView = texture.CreateView();
  wgpu::BindGroupDescriptor bg_desc{};
  bg_desc.layout = gpu.bind_group_layout;
  bg_desc.entryCount = entries.size();
  bg_desc.entries = entries.data();
  gpu.bind_group = gpu.device.CreateBindGroup(&bg_desc);
  return gpu;
}

wgpu::RenderPipeline CreatePipeline(const Gpu& gpu, wgpu::TextureFormat format) {
  wgpu::ColorTargetState target{};
  target.format = format;
  wgpu::FragmentState fragment{};
  fragment.module = gpu.shader;
  fragment.entryPoint = "fs_main";
  fragment.targetCount = 1;
  fragment.targets = &target;
  wgpu::RenderPipelineDescriptor desc{};
  desc.layout = gpu.pipeline_layout;
  desc.vertex.module = gpu.shader;
  desc.vertex.entryPoint = "vs_main";
  desc.fragment = &fragment;
  return gpu.device.CreateRenderPipeline(&desc);
}

void EncodeQuad(const Gpu& gpu, const wgpu::RenderPipeline& pipeline, const wgpu::TextureView& view) {
  wgpu::RenderPassColorAttachment attachment{};
  attachment.view = view;
  attachment.loadOp = wgpu::LoadOp::Clear;
  attachment.storeOp = wgpu::StoreOp::Store;
  attachment.clearValue = kClearColor;
  wgpu::RenderPassDescriptor pass_desc{};
  pass_desc.colorAttachmentCount = 1;
  pass_desc.colorAttachments = &attachment;

  wgpu::CommandEncoder encoder = gpu.device.CreateCommandEncoder();
  wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&pass_desc);
  pass.SetPipeline(pipeline);
  pass.SetBindGroup(0, gpu.bind_group);
  pass.Draw(6);
  pass.End();
  wgpu::CommandBuffer commands = encoder.Finish();
  gpu.queue.Submit(1, &commands);
}

// Renders into a 64x64 texture, copies it to a buffer and checks the centre of
// every checker cell plus the cleared border.
int RunOffscreen(const Gpu& gpu) {
  constexpr uint32_t kSize = 64;
  constexpr uint32_t kBytesPerRow = kSize * 4;  // 256: meets the copy alignment rule
  wgpu::TextureDescriptor target_desc{};
  target_desc.size = {kSize, kSize, 1};
  target_desc.format = wgpu::TextureFormat::RGBA8Unorm;
  target_desc.usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc;
  wgpu::Texture target = gpu.device.CreateTexture(&target_desc);

  EncodeQuad(gpu, CreatePipeline(gpu, target_desc.format), target.CreateView());

  wgpu::BufferDescriptor buffer_desc{};
  buffer_desc.size = kBytesPerRow * kSize;
  buffer_desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
  wgpu::Buffer readback = gpu.device.CreateBuffer(&buffer_desc);

  wgpu::TexelCopyTextureInfo source{};
  source.texture = target;
  wgpu::TexelCopyBufferInfo destination{};
  destination.buffer = readback;
  destination.layout.bytesPerRow = kBytesPerRow;
  destination.layout.rowsPerImage = kSize;
  wgpu::CommandEncoder encoder = gpu.device.CreateCommandEncoder();
  encoder.CopyTextureToBuffer(&source, &destination, &target_desc.size);
  wgpu::CommandBuffer commands = encoder.Finish();
  gpu.queue.Submit(1, &commands);

  bool mapped = false;
  const wgpu::WaitStatus wait = gpu.instance.WaitAny(
      readback.MapAsync(wgpu::MapMode::Read, 0, buffer_desc.size, wgpu::CallbackMode::WaitAnyOnly,
                        [&mapped](wgpu::MapAsyncStatus status, wgpu::StringView message) {
                          mapped = status == wgpu::MapAsyncStatus::Success;
                          if (!mapped) {
                            PrintMessage("MapAsync failed", message);
                          }
                        }),
      UINT64_MAX);
  if (wait != wgpu::WaitStatus::Success || !mapped) {
    std::fprintf(stderr, "FAIL: could not read back the rendered texture\n");
    return 1;
  }

  const auto* data = static_cast<const uint8_t*>(readback.GetConstMappedRange(0, buffer_desc.size));
  int mismatches = 0;
  auto check = [&](uint32_t px, uint32_t py, std::array<uint8_t, 4> expected) {
    const uint8_t* actual = data + py * kBytesPerRow + px * 4;
    for (int c = 0; c < 4; ++c) {
      if (std::abs(int{actual[c]} - int{expected[static_cast<size_t>(c)]}) > 1) {
        std::fprintf(stderr, "mismatch at (%u,%u): got %u,%u,%u,%u expected %u,%u,%u,%u\n", px, py,
                     actual[0], actual[1], actual[2], actual[3], expected[0], expected[1],
                     expected[2], expected[3]);
        ++mismatches;
        return;
      }
    }
  };

  // The quad occupies pixels [6.4, 57.6) on each axis; each cell is 6.4 px.
  const float quad_origin = (1.0f - kQuadExtent) * 0.5f * kSize;
  const float cell = kQuadExtent * kSize / kCheckerCells;
  for (uint32_t y = 0; y < kCheckerCells; ++y) {
    for (uint32_t x = 0; x < kCheckerCells; ++x) {
      const auto px = static_cast<uint32_t>(quad_origin + (static_cast<float>(x) + 0.5f) * cell);
      const auto py = static_cast<uint32_t>(quad_origin + (static_cast<float>(y) + 0.5f) * cell);
      check(px, py, ExpectedCellColor(x, y));
    }
  }
  const std::array<uint8_t, 4> clear = {0, 0, 0, 255};
  check(0, 0, clear);
  check(kSize - 1, kSize - 1, clear);
  readback.Unmap();

  if (mismatches != 0) {
    std::fprintf(stderr, "FAIL: %d pixel mismatches\n", mismatches);
    return 1;
  }
  std::printf("Offscreen readback: 66 pixels match the expected checkerboard and border\nPASS\n");
  return 0;
}

// Owns GLFW initialisation and the window; terminates GLFW on scope exit.
class GlfwWindow {
 public:
  GlfwWindow(int width, int height, const char* title) {
    if (glfwInit() != GLFW_TRUE) {
      return;
    }
    initialized_ = true;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // WebGPU, not OpenGL
    window_ = glfwCreateWindow(width, height, title, nullptr, nullptr);
  }
  ~GlfwWindow() {
    if (window_ != nullptr) {
      glfwDestroyWindow(window_);
    }
    if (initialized_) {
      glfwTerminate();
    }
  }
  GlfwWindow(const GlfwWindow&) = delete;
  GlfwWindow& operator=(const GlfwWindow&) = delete;

  GLFWwindow* get() const { return window_; }

 private:
  bool initialized_ = false;
  GLFWwindow* window_ = nullptr;
};

int RunWindowed(const Gpu& gpu, std::optional<long> max_frames) {
  GlfwWindow window(1280, 720, "Relume Dawn Probe");
  if (window.get() == nullptr) {
    std::fprintf(stderr, "FAIL: could not create a GLFW window\n");
    return 1;
  }

  wgpu::SurfaceSourceMetalLayer metal_source{};
  metal_source.layer = AttachMetalLayer(window.get());
  wgpu::SurfaceDescriptor surface_desc{};
  surface_desc.nextInChain = &metal_source;
  wgpu::Surface surface = gpu.instance.CreateSurface(&surface_desc);

  wgpu::SurfaceCapabilities capabilities{};
  surface.GetCapabilities(gpu.adapter, &capabilities);
  if (capabilities.formatCount == 0) {
    std::fprintf(stderr, "FAIL: surface reports no formats\n");
    return 1;
  }
  const wgpu::TextureFormat format = capabilities.formats[0];
  const wgpu::RenderPipeline pipeline = CreatePipeline(gpu, format);

  auto configure = [&] {
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window.get(), &width, &height);  // pixels, not points
    wgpu::SurfaceConfiguration config{};
    config.device = gpu.device;
    config.format = format;
    config.width = static_cast<uint32_t>(width);
    config.height = static_cast<uint32_t>(height);
    config.presentMode = wgpu::PresentMode::Fifo;  // vsync
    surface.Configure(&config);
    std::printf("Surface configured: %dx%d\n", width, height);
  };
  configure();

  long frames = 0;
  while (glfwWindowShouldClose(window.get()) == GLFW_FALSE &&
         (!max_frames || frames < *max_frames)) {
    glfwPollEvents();
    wgpu::SurfaceTexture frame{};
    surface.GetCurrentTexture(&frame);
    if (frame.status == wgpu::SurfaceGetCurrentTextureStatus::Outdated ||
        frame.status == wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal) {
      configure();  // window resized or moved to another display
      continue;
    }
    if (frame.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal) {
      std::fprintf(stderr, "FAIL: could not acquire a surface texture (status %u)\n",
                   static_cast<unsigned>(frame.status));
      return 1;
    }
    EncodeQuad(gpu, pipeline, frame.texture.CreateView());
    surface.Present();
    gpu.instance.ProcessEvents();
    ++frames;
  }
  std::printf("Presented %ld frames\nPASS\n", frames);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  bool offscreen = false;
  std::optional<long> max_frames;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--offscreen") {
      offscreen = true;
    } else if (arg == "--frames" && i + 1 < argc) {
      max_frames = std::strtol(argv[++i], nullptr, 10);
    } else {
      std::fprintf(stderr, "usage: %s [--offscreen] [--frames N]\n", argv[0]);
      return 2;
    }
  }

  std::optional<Gpu> gpu = CreateGpu();
  if (!gpu) {
    return 1;
  }
  return offscreen ? RunOffscreen(*gpu) : RunWindowed(*gpu, max_frames);
}
