// IOSurface + Core Animation window presentation (macOS).
//
// Why: CAMetalLayer's nextDrawable blocks for up to 1 s right after a window
// is covered, and Dawn holds its device lock while it waits, so every window
// and the clean-output compositing stall. Nothing here waits.
//
// Each window owns four buffers, each an IOSurface imported into Dawn as
// SharedTextureMemory. Four, not three: the window server can hold two
// surfaces at once (on screen and being composited) while one waits on the
// GPU; with three, measured runs skipped up to ~30% of frames depending on the
// loop's phase relative to vsync, with four none were skipped.
//
//   free --Acquire (if reusable, see below)--> rendering
//        --Present: EndAccess, OnSubmittedWorkDone--> gpu-pending
//        --GPU done callback (in ProcessEvents)--> ready
//        --FinishFrames: layer.contents = surface--> displayed --replaced--> free
//
// - The GPU has finished a frame before Core Animation is given it.
// - The displayed surface is never rendered into.
// - A surface that was ever given to Core Animation is rendered into again
//   only if IOSurfaceIsInUse is false AND a newer committed surface has since
//   been seen in use (IOSurfaceIsInUse true). "Not in use" alone is not
//   trusted: right after a window starts being covered, or before it is first
//   shown, committed surfaces were observed never to be marked in use, so the
//   render server's progress past them could not be confirmed. Seeing a newer
//   surface in use shows the render server has applied a later transaction.
//   Remaining assumption: Core Animation applies committed transactions in
//   order and never goes back to superseded contents. That is how layer
//   contents are documented to behave, but it is not proven here.
// - If no buffer is reusable, the window skips that frame; nothing waits.
// - GPU-completion callbacks hold weak references, so dropping a buffer is
//   always safe; Dawn's SharedTextureMemory retains the IOSurface for as long
//   as GPU work uses it, and CALayer retains the surface it displays.

#include "platform/layer_presenter.hpp"

#include "relume/compositor_reuse.hpp"

#import <Cocoa/Cocoa.h>
#import <IOSurface/IOSurface.h>
#import <QuartzCore/QuartzCore.h>

#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <string>

namespace relume {
namespace {

using Clock = std::chrono::steady_clock;

double MsSince(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct Buffer {
  enum class State : std::uint8_t { kFree, kRendering, kGpuPending, kReady, kDisplayed };

  Buffer() = default;
  ~Buffer() {
    texture = nullptr;
    memory = nullptr;
    if (surface != nullptr) {
      CFRelease(surface);
    }
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  IOSurfaceRef surface = nullptr;  // owned (+1 from IOSurfaceCreate)
  wgpu::SharedTextureMemory memory;
  wgpu::Texture texture;
  wgpu::TextureView view;
  State state = State::kFree;
  std::uint64_t generation = 0;
  bool committed = false;  // ever given to Core Animation as layer contents
};

class LayerPresenter final : public WindowPresenter {
 public:
  LayerPresenter(const GpuContext& gpu, GLFWwindow* window) : gpu_(gpu), window_(window) {
    format_ = wgpu::TextureFormat::BGRA8Unorm;
    NSWindow* ns_window = glfwGetCocoaWindow(window);
    if (ns_window == nil) {
      error_ = "no native window";
      return;
    }
    layer_ = [CALayer layer];
    layer_.opaque = YES;
    layer_.backgroundColor = NSColor.blackColor.CGColor;
    // Frames match the framebuffer exactly. While a live resize is ahead of
    // the next frame, centre it rather than stretch it.
    layer_.contentsGravity = kCAGravityCenter;
    ns_window.contentView.wantsLayer = YES;
    ns_window.contentView.layer = layer_;
  }

  ~LayerPresenter() override {
    // Buffers may still be referenced by queued GPU work (Dawn keeps its own
    // references) or shown by the layer (CALayer retains the surface).
    current_.reset();
    displayed_.reset();
    for (auto& buffer : buffers_) {
      buffer.reset();
    }
  }

  const char* kind() const override { return "layer"; }
  std::string diagnostics() const override {
    return "buffers held back until the render server was seen past them: " + std::to_string(held_back_);
  }

  std::optional<Frame> Acquire() override {
    const Clock::time_point start = Clock::now();
    std::optional<Frame> frame = AcquireImpl();
    last_acquire_ms_ = MsSince(start);
    return frame;
  }

  void Present() override {
    const Clock::time_point start = Clock::now();
    std::shared_ptr<Buffer> buffer = std::move(current_);
    if (buffer) {
      wgpu::SharedTextureMemoryEndAccessState end{};
      if (buffer->memory.EndAccess(buffer->texture, &end) != wgpu::Status::Success) {
        // The texture can no longer be trusted; let it be recreated.
        buffer->state = Buffer::State::kFree;
        DropAll();
      } else {
        buffer->state = Buffer::State::kGpuPending;
        buffer->generation = ++generation_;
        const std::uint64_t generation = buffer->generation;
        std::weak_ptr<Buffer> weak = buffer;
        gpu_.queue().OnSubmittedWorkDone(
            wgpu::CallbackMode::AllowProcessEvents,
            [weak, generation](wgpu::QueueWorkDoneStatus status, wgpu::StringView) {
              const std::shared_ptr<Buffer> done = weak.lock();
              if (done && done->generation == generation && done->state == Buffer::State::kGpuPending) {
                done->state = status == wgpu::QueueWorkDoneStatus::Success ? Buffer::State::kReady
                                                                            : Buffer::State::kFree;
              }
            });
      }
    }
    last_present_ms_ = MsSince(start);
  }

  void FinishFrames() override {
    ObserveRenderServer();
    std::shared_ptr<Buffer> newest;
    for (const auto& buffer : buffers_) {
      if (buffer && buffer->state == Buffer::State::kReady &&
          (!newest || buffer->generation > newest->generation)) {
        newest = buffer;
      }
    }
    if (!newest) {
      return;
    }
    for (const auto& buffer : buffers_) {
      if (buffer && buffer != newest && buffer->state == Buffer::State::kReady) {
        buffer->state = Buffer::State::kFree;  // superseded before it was shown
      }
    }
    [CATransaction begin];
    [CATransaction setDisableActions:YES];  // no implicit cross-fade
    layer_.contentsScale = static_cast<NSWindow*>(glfwGetCocoaWindow(window_)).backingScaleFactor;
    layer_.contents = (__bridge id)newest->surface;
    [CATransaction commit];
    newest->committed = true;
    ++presented_frames_;
    if (displayed_ && displayed_ != newest && displayed_->state == Buffer::State::kDisplayed) {
      displayed_->state = Buffer::State::kFree;
    }
    newest->state = Buffer::State::kDisplayed;
    displayed_ = std::move(newest);
  }

 private:
  std::optional<Frame> AcquireImpl() {
    if (!ok()) {
      return std::nullopt;
    }
    int fb_width = 0;
    int fb_height = 0;
    glfwGetFramebufferSize(window_, &fb_width, &fb_height);  // pixels, not points
    if (fb_width <= 0 || fb_height <= 0) {
      return std::nullopt;  // minimized or no area
    }
    const auto width = static_cast<std::uint32_t>(fb_width);
    const auto height = static_cast<std::uint32_t>(fb_height);
    if (width != width_ || height != height_) {
      // A fresh set at the new size; old buffers live on only while queued
      // GPU work or the layer still references them.
      DropAll();
      width_ = width;
      height_ = height;
    }
    for (auto& buffer : buffers_) {
      if (!buffer) {
        buffer = CreateBuffer(width, height);
        if (!buffer) {
          return std::nullopt;
        }
      }
    }
    ObserveRenderServer();
    for (const auto& buffer : buffers_) {
      if (!Reusable(*buffer)) {
        continue;
      }
      wgpu::SharedTextureMemoryBeginAccessDescriptor begin{};
      begin.initialized = false;  // every frame clears the whole texture
      if (buffer->memory.BeginAccess(buffer->texture, &begin) != wgpu::Status::Success) {
        continue;
      }
      buffer->state = Buffer::State::kRendering;
      current_ = buffer;
      return Frame{buffer->texture, buffer->view, width, height};
    }
    ++skipped_frames_;
    return std::nullopt;  // no buffer confirmed reusable: skip this frame, never wait
  }

  // Records the newest committed generation the render server is holding.
  void ObserveRenderServer() {
    for (const auto& buffer : buffers_) {
      if (buffer && buffer->committed && buffer->generation > reuse_.seen_generation() &&
          IOSurfaceIsInUse(buffer->surface)) {
        reuse_.ObserveInUse(buffer->generation);
      }
    }
  }

  // See CompositorReuseTracker: a committed surface is reused only after the
  // render server has been seen holding a newer one.
  bool Reusable(const Buffer& buffer) {
    const bool free = buffer.state == Buffer::State::kFree;
    const bool in_use = IOSurfaceIsInUse(buffer.surface);
    const bool reusable = reuse_.CanReuse(free, in_use, buffer.committed, buffer.generation);
    if (free && !in_use && !reusable) {
      ++held_back_;
    }
    return reusable;
  }

  void DropAll() {
    // New buffers start uncommitted; generations keep increasing, so the
    // seen-generation evidence stays valid across the swap.
    displayed_.reset();
    for (auto& buffer : buffers_) {
      buffer.reset();
    }
  }

  std::shared_ptr<Buffer> CreateBuffer(std::uint32_t width, std::uint32_t height) {
    NSDictionary* properties = @{
      (id)kIOSurfaceWidth : @(width),
      (id)kIOSurfaceHeight : @(height),
      (id)kIOSurfaceBytesPerElement : @4,
      (id)kIOSurfacePixelFormat : @(static_cast<std::uint32_t>('BGRA')),
    };
    auto buffer = std::make_shared<Buffer>();
    buffer->surface = IOSurfaceCreate((__bridge CFDictionaryRef)properties);
    if (buffer->surface == nullptr) {
      error_ = "IOSurfaceCreate failed";
      return nullptr;
    }
    wgpu::SharedTextureMemoryIOSurfaceDescriptor io_desc{};
    io_desc.ioSurface = buffer->surface;
    wgpu::SharedTextureMemoryDescriptor memory_desc{};
    memory_desc.nextInChain = &io_desc;
    memory_desc.label = "relume window buffer";
    buffer->memory = gpu_.device().ImportSharedTextureMemory(&memory_desc);
    wgpu::TextureDescriptor texture_desc{};
    texture_desc.size = {width, height, 1};
    texture_desc.format = format_;
    texture_desc.usage = wgpu::TextureUsage::RenderAttachment;
    buffer->texture = buffer->memory.CreateTexture(&texture_desc);
    buffer->view = buffer->texture.CreateView();
    return buffer;
  }

  const GpuContext& gpu_;
  GLFWwindow* window_;
  CALayer* layer_ = nil;
  std::array<std::shared_ptr<Buffer>, 4> buffers_{};
  std::shared_ptr<Buffer> current_;    // being rendered this iteration
  std::shared_ptr<Buffer> displayed_;  // the layer's contents
  std::uint64_t generation_ = 0;
  CompositorReuseTracker reuse_;
  std::uint64_t held_back_ = 0;        // free, not in use, but not yet confirmed safe
};

}  // namespace

std::unique_ptr<WindowPresenter> CreateLayerPresenter(const GpuContext& gpu, GLFWwindow* window, std::string& error) {
  if (!gpu.supports_layer_presentation()) {
    error = "the GPU device lacks SharedTextureMemoryIOSurface or SharedFenceMTLSharedEvent";
    return nullptr;
  }
  auto presenter = std::make_unique<LayerPresenter>(gpu, window);
  if (!presenter->ok()) {
    error = presenter->error();
    return nullptr;
  }
  return presenter;
}

}  // namespace relume
