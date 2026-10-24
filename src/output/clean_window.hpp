#pragma once

// "Relume Output": the clean, video-only window that OBS captures. It shows
// nothing but the OutputTexture (no UI, text or cursor overlay), aspect-fit
// into its framebuffer. Its content area is locked to 16:9 and initially sized
// so the framebuffer is 1280x720 pixels at the display's content scale.

#include "app/window.hpp"
#include "relume/renderer.hpp"

namespace relume {

class CleanWindow {
 public:
  static constexpr const char* kTitle = "Relume Output";

  explicit CleanWindow(const GpuContext& gpu);
  bool ok() const;
  const std::string& error() const { return error_; }

  void Show();
  void Hide();
  bool visible() const { return visible_; }
  // Closing the window hides it; the Controls window can show it again.
  void HandleCloseRequest();

  void Render(Presenter& presenter);

  std::uint32_t framebuffer_width() const { return surface_ ? surface_->width() : 0; }
  std::uint32_t framebuffer_height() const { return surface_ ? surface_->height() : 0; }

 private:
  Window window_;
  std::optional<WindowSurface> surface_;
  bool visible_ = false;
  std::string error_;
};

}  // namespace relume
