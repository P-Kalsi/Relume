#include "app/window.hpp"

#include "platform/surface.hpp"

#include <GLFW/glfw3.h>

#include <chrono>
#include <cstdio>

namespace relume {

GlfwLibrary::GlfwLibrary() {
  glfwSetErrorCallback([](int code, const char* description) {
    std::fprintf(stderr, "GLFW error %d: %s\n", code, description);
  });
  // Keep the working directory when running inside an .app bundle.
  glfwInitHint(GLFW_COCOA_CHDIR_RESOURCES, GLFW_FALSE);
  ok_ = glfwInit() == GLFW_TRUE;
}

GlfwLibrary::~GlfwLibrary() {
  if (ok_) {
    glfwTerminate();
  }
}

Window::Window(int width, int height, const char* title, bool visible, bool decorated) {
  glfwDefaultWindowHints();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // WebGPU, not OpenGL
  glfwWindowHint(GLFW_VISIBLE, visible ? GLFW_TRUE : GLFW_FALSE);
  glfwWindowHint(GLFW_DECORATED, decorated ? GLFW_TRUE : GLFW_FALSE);
  window_ = glfwCreateWindow(width, height, title, nullptr, nullptr);
}

Window::~Window() {
  if (window_ != nullptr) {
    glfwDestroyWindow(window_);
  }
}

SurfacePresenter::SurfacePresenter(const GpuContext& gpu, GLFWwindow* window) : gpu_(gpu), window_(window) {
  surface_ = CreateSurfaceForWindow(gpu.instance(), window);
  if (!surface_) {
    error_ = "could not create a window surface";
    return;
  }
  wgpu::SurfaceCapabilities capabilities{};
  surface_.GetCapabilities(gpu.adapter(), &capabilities);
  // A non-sRGB format, so passthrough is an identity (no implicit gamma).
  for (std::size_t i = 0; i < capabilities.formatCount; ++i) {
    if (capabilities.formats[i] == wgpu::TextureFormat::BGRA8Unorm ||
        (format_ == wgpu::TextureFormat::Undefined && capabilities.formats[i] == wgpu::TextureFormat::RGBA8Unorm)) {
      format_ = capabilities.formats[i];
    }
  }
  if (format_ == wgpu::TextureFormat::Undefined) {
    error_ = "the window surface offers no BGRA8Unorm or RGBA8Unorm format";
    return;
  }
  for (std::size_t i = 0; i < capabilities.alphaModeCount; ++i) {
    if (capabilities.alphaModes[i] == wgpu::CompositeAlphaMode::Opaque) {
      alpha_mode_ = wgpu::CompositeAlphaMode::Opaque;
    }
  }
}

void SurfacePresenter::Configure(std::uint32_t width, std::uint32_t height) {
  wgpu::SurfaceConfiguration config{};
  config.device = gpu_.device();
  config.format = format_;
  config.usage = wgpu::TextureUsage::RenderAttachment;
  config.width = width;
  config.height = height;
  config.alphaMode = alpha_mode_;
  config.presentMode = wgpu::PresentMode::Fifo;  // vsync
  surface_.Configure(&config);
  width_ = width;
  height_ = height;
  needs_configure_ = false;
}

std::optional<WindowPresenter::Frame> SurfacePresenter::Acquire() {
  if (!ok()) {
    return std::nullopt;
  }
  int fb_width = 0;
  int fb_height = 0;
  glfwGetFramebufferSize(window_, &fb_width, &fb_height);  // pixels, not points
  if (fb_width <= 0 || fb_height <= 0) {
    return std::nullopt;
  }
  const auto w = static_cast<std::uint32_t>(fb_width);
  const auto h = static_cast<std::uint32_t>(fb_height);
  if (needs_configure_ || w != width_ || h != height_) {
    Configure(w, h);
  }
  for (int attempt = 0; attempt < 2; ++attempt) {
    wgpu::SurfaceTexture surface_texture{};
    const auto wait_start = std::chrono::steady_clock::now();
    surface_.GetCurrentTexture(&surface_texture);
    const double waited_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wait_start).count();
    last_acquire_ms_ = waited_ms;
    if (waited_ms > 100.0) {
      // Diagnostic: the window system had no drawable ready (CAMetalLayer
      // blocks for up to 1 s), which stalls the whole render loop.
      std::printf("[surface] %s waited %.0f ms for a drawable (status %u)\n", glfwGetWindowTitle(window_),
                  waited_ms, static_cast<unsigned>(surface_texture.status));
    }
    switch (surface_texture.status) {
      case wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal:
      case wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal:
        if (surface_texture.status == wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal) {
          needs_configure_ = true;  // use this one, reconfigure next time
        }
        return Frame{surface_texture.texture, surface_texture.texture.CreateView(), width_, height_};
      case wgpu::SurfaceGetCurrentTextureStatus::Outdated:
      case wgpu::SurfaceGetCurrentTextureStatus::Lost:
        Configure(w, h);
        continue;
      default:
        return std::nullopt;  // timeout or error: skip this frame
    }
  }
  return std::nullopt;
}

void SurfacePresenter::Present() {
  const auto start = std::chrono::steady_clock::now();
  surface_.Present();
  last_present_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace relume
