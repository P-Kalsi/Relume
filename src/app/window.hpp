#pragma once

// RAII wrappers for GLFW and for a window's WebGPU surface. Main thread only.

#include "relume/renderer.hpp"

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <optional>
#include <string>

struct GLFWwindow;

namespace relume {

// Initializes GLFW for the lifetime of the object.
class GlfwLibrary {
 public:
  GlfwLibrary();
  ~GlfwLibrary();
  GlfwLibrary(const GlfwLibrary&) = delete;
  GlfwLibrary& operator=(const GlfwLibrary&) = delete;
  bool ok() const { return ok_; }

 private:
  bool ok_ = false;
};

// A GLFW window without a GL context (WebGPU renders into it).
class Window {
 public:
  Window(int width, int height, const char* title, bool visible, bool decorated = true);
  ~Window();
  Window(const Window&) = delete;
  Window& operator=(const Window&) = delete;
  GLFWwindow* get() const { return window_; }

 private:
  GLFWwindow* window_ = nullptr;  // owned; destroyed in ~Window
};

// The window's swap chain. It is reconfigured whenever the framebuffer size
// changes (resize, Retina scale change, moving to another display); that never
// affects the OutputTexture, which has a fixed size.
class WindowSurface {
 public:
  WindowSurface(const GpuContext& gpu, GLFWwindow* window);
  bool ok() const { return error_.empty(); }
  const std::string& error() const { return error_; }
  wgpu::TextureFormat format() const { return format_; }

  struct Frame {
    wgpu::Texture texture;
    wgpu::TextureView view;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
  };
  // The next texture to draw into, or nullopt if the window has no area
  // (minimized) or the surface is temporarily unavailable.
  std::optional<Frame> Acquire();
  void Present();

  std::uint32_t width() const { return width_; }
  std::uint32_t height() const { return height_; }
  // Time the last Acquire() spent in GetCurrentTexture and the last Present().
  double last_acquire_ms() const { return last_acquire_ms_; }
  double last_present_ms() const { return last_present_ms_; }

 private:
  void Configure(std::uint32_t width, std::uint32_t height);

  const GpuContext& gpu_;
  GLFWwindow* window_;
  wgpu::Surface surface_;
  wgpu::TextureFormat format_ = wgpu::TextureFormat::Undefined;
  wgpu::CompositeAlphaMode alpha_mode_ = wgpu::CompositeAlphaMode::Auto;
  std::uint32_t width_ = 0;
  std::uint32_t height_ = 0;
  bool needs_configure_ = true;
  double last_acquire_ms_ = 0.0;
  double last_present_ms_ = 0.0;
  std::string error_;
};

}  // namespace relume
