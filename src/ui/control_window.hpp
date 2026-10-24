#pragma once

// "Relume Controls": the window for the person running Relume. It shows a
// preview of the OutputTexture (the same texture the Output window shows).

#include "app/window.hpp"
#include "relume/renderer.hpp"

namespace relume {

class ControlWindow {
 public:
  static constexpr const char* kTitle = "Relume Controls";

  explicit ControlWindow(const GpuContext& gpu);
  bool ok() const { return error_.empty(); }
  const std::string& error() const { return error_; }
  bool should_close() const;
  GLFWwindow* window() const { return window_.get(); }

  void Render(Presenter& presenter);

 private:
  Window window_;
  std::optional<WindowSurface> surface_;
  std::string error_;
};

}  // namespace relume
