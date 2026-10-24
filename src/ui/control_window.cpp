#include "ui/control_window.hpp"

#include <GLFW/glfw3.h>

namespace relume {

ControlWindow::ControlWindow(const GpuContext& gpu) : window_(960, 640, kTitle, true) {
  if (window_.get() == nullptr) {
    error_ = "could not create the Controls window";
    return;
  }
  surface_.emplace(gpu, window_.get());
  if (!surface_->ok()) {
    error_ = surface_->error();
  }
}

bool ControlWindow::should_close() const { return glfwWindowShouldClose(window_.get()) == GLFW_TRUE; }

void ControlWindow::Render(Presenter& presenter) {
  if (!ok()) {
    return;
  }
  std::optional<WindowSurface::Frame> frame = surface_->Acquire();
  if (!frame) {
    return;
  }
  presenter.Present(frame->view, surface_->format(), frame->width, frame->height);
  surface_->Present();
}

}  // namespace relume
