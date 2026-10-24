#pragma once

// RAII wrappers for GLFW and for a window's WebGPU surface. Main thread only.

#include "relume/renderer.hpp"

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <memory>
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

// Presents frames to one window. The caller draws into the texture from
// Acquire(), submits its commands, then calls Present(); FinishFrames() runs
// once per loop iteration after Instance::ProcessEvents(). A framebuffer
// change (resize, Retina scale change, moving to another display) only
// changes what Acquire() returns; it never affects the OutputTexture.
class WindowPresenter {
 public:
  struct Frame {
    wgpu::Texture texture;
    wgpu::TextureView view;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
  };

  virtual ~WindowPresenter() = default;
  virtual const char* kind() const = 0;
  virtual std::string diagnostics() const { return {}; }
  bool ok() const { return error_.empty(); }
  const std::string& error() const { return error_; }
  wgpu::TextureFormat format() const { return format_; }

  // The next texture to draw into, or nullopt if this window cannot take a
  // frame now (no area, minimized, or no buffer free); the frame is skipped.
  virtual std::optional<Frame> Acquire() = 0;
  virtual void Present() = 0;
  virtual void FinishFrames() {}

  std::uint32_t width() const { return width_; }
  std::uint32_t height() const { return height_; }
  // Time the last Acquire() and Present() took.
  double last_acquire_ms() const { return last_acquire_ms_; }
  double last_present_ms() const { return last_present_ms_; }
  // Frames handed to the window system, and frames skipped because no buffer
  // could take one (only the layer presenter skips; the surface waits).
  std::uint64_t presented_frames() const { return presented_frames_; }
  std::uint64_t skipped_frames() const { return skipped_frames_; }

 protected:
  wgpu::TextureFormat format_ = wgpu::TextureFormat::Undefined;
  std::uint32_t width_ = 0;
  std::uint32_t height_ = 0;
  double last_acquire_ms_ = 0.0;
  double last_present_ms_ = 0.0;
  std::uint64_t presented_frames_ = 0;
  std::uint64_t skipped_frames_ = 0;
  std::string error_;
};

// Presents through a wgpu::Surface on a CAMetalLayer (Fifo, vsync paced).
// Kept as the fallback path: on macOS, acquiring the next drawable blocks for
// up to 1 s when the window has just been covered, and Dawn holds its device
// lock meanwhile, which stalls all rendering.
class SurfacePresenter final : public WindowPresenter {
 public:
  SurfacePresenter(const GpuContext& gpu, GLFWwindow* window);
  const char* kind() const override { return "surface"; }
  std::optional<Frame> Acquire() override;
  void Present() override;

 private:
  void Configure(std::uint32_t width, std::uint32_t height);

  const GpuContext& gpu_;
  GLFWwindow* window_;
  wgpu::Surface surface_;
  wgpu::CompositeAlphaMode alpha_mode_ = wgpu::CompositeAlphaMode::Auto;
  bool needs_configure_ = true;
};

enum class PresenterKind : std::uint8_t {
  kLayer,    // IOSurface + Core Animation; never acquires a CAMetalDrawable
  kSurface,  // wgpu::Surface on a CAMetalLayer (fallback)
};

// Creates the requested presenter for `window`. If kLayer is unavailable it
// falls back to kSurface and says why in `note`.
std::unique_ptr<WindowPresenter> CreateWindowPresenter(const GpuContext& gpu, GLFWwindow* window,
                                                       PresenterKind kind, std::string& note);

}  // namespace relume
