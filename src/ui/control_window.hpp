#pragma once

// "Relume Controls": the window for the person running Relume. Phase 2 UI is
// deliberately minimal: camera selection, capture/permission status, an
// Output window toggle and a preview of the OutputTexture (the same texture
// the Output window shows). ImGui draws only into this window.

#include "app/window.hpp"
#include "relume/app_state.hpp"
#include "relume/camera.hpp"
#include "relume/guided_capture.hpp"
#include "relume/renderer.hpp"

#include <optional>
#include <string>
#include <vector>

namespace relume {

// What the Controls window displays this frame.
struct ControlsModel {
  CaptureStatus status;
  std::vector<CameraDeviceInfo> devices;
  std::string selected_id;
  std::string source_description;  // negotiated/delivered format
  std::string warning;             // e.g. missing colour metadata
  bool camera_selectable = true;   // false for the file source
  bool output_visible = true;
  const GuidedCapture* guided = nullptr;  // --guided-capture only
  GuidedCapture::Clock::time_point now{};  // for the guided-capture countdown
};

// What the user asked for this frame.
struct ControlsActions {
  std::optional<std::string> select_device;  // already confirmed for virtual cameras
  bool toggle_output = false;
  std::optional<std::size_t> guided_select;  // move to another shot
  bool guided_capture = false;               // Capture / Retake / Start burst
};

class ControlWindow {
 public:
  static constexpr const char* kTitle = "Relume Controls";

  ControlWindow(const GpuContext& gpu, const OutputTexture& output, PresenterKind kind);
  ~ControlWindow();
  ControlWindow(const ControlWindow&) = delete;
  ControlWindow& operator=(const ControlWindow&) = delete;

  GLFWwindow* window() const { return window_.get(); }
  const WindowPresenter* presenter() const { return presenter_.get(); }
  // Hands finished frames to the window; call after Instance::ProcessEvents().
  void FinishFrames() {
    if (presenter_) {
      presenter_->FinishFrames();
    }
  }
  bool ok() const { return error_.empty(); }
  const std::string& error() const { return error_; }
  bool should_close() const;

  ControlsActions Render(const ControlsModel& model);

 private:
  ControlsActions BuildUi(const ControlsModel& model);
  void BuildGuidedCapture(const GuidedCapture& guided, GuidedCapture::Clock::time_point now,
                          ControlsActions& actions);

  const GpuContext& gpu_;
  const OutputTexture& output_;
  Window window_;
  std::unique_ptr<WindowPresenter> presenter_;
  bool imgui_glfw_ = false;
  bool imgui_wgpu_ = false;
  std::string pending_virtual_id_;  // a virtual camera awaiting confirmation
  std::string error_;
};

}  // namespace relume
