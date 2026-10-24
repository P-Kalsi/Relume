#include "output/clean_window.hpp"

#include <GLFW/glfw3.h>

#include <cmath>
#include <cstdio>

namespace relume {

CleanWindow::CleanWindow(const GpuContext& gpu)
    : window_(static_cast<int>(OutputTexture::kWidth), static_cast<int>(OutputTexture::kHeight), kTitle, false) {
  if (window_.get() == nullptr) {
    error_ = "could not create the Output window";
    return;
  }
  // Size the content area in points so the framebuffer is 1280x720 pixels on
  // this display (640x360 points at 2x). A convenience only: the video output
  // contract is the fixed OutputTexture, whatever this window's size.
  float scale_x = 1.0f;
  float scale_y = 1.0f;
  glfwGetWindowContentScale(window_.get(), &scale_x, &scale_y);
  const int width = static_cast<int>(std::lround(static_cast<float>(OutputTexture::kWidth) / scale_x));
  const int height = static_cast<int>(std::lround(static_cast<float>(OutputTexture::kHeight) / scale_y));
  glfwSetWindowSize(window_.get(), width, height);
  glfwSetWindowAspectRatio(window_.get(), 16, 9);
  presenter_ = std::make_unique<SurfacePresenter>(gpu, window_.get());
  if (!presenter_->ok()) {
    error_ = presenter_->error();
  }
}

bool CleanWindow::ok() const { return error_.empty(); }

void CleanWindow::Show() {
  glfwShowWindow(window_.get());
  visible_ = true;
}

void CleanWindow::Hide() {
  glfwHideWindow(window_.get());
  visible_ = false;
}

void CleanWindow::HandleCloseRequest() {
  if (glfwWindowShouldClose(window_.get()) == GLFW_TRUE) {
    glfwSetWindowShouldClose(window_.get(), GLFW_FALSE);
    Hide();
  }
}

void CleanWindow::Render(Presenter& presenter) {
  if (!visible_ || !presenter_ || !presenter_->ok()) {
    return;
  }
  const std::uint32_t old_width = presenter_->width();
  const std::uint32_t old_height = presenter_->height();
  std::optional<WindowPresenter::Frame> frame = presenter_->Acquire();
  if (!frame) {
    return;
  }
  if (frame->width != old_width || frame->height != old_height) {
    std::printf("%s framebuffer: %ux%u px\n", kTitle, frame->width, frame->height);
  }
  presenter.Present(frame->view, presenter_->format(), frame->width, frame->height);
  presenter_->Present();
}

}  // namespace relume
