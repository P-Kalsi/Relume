#pragma once

// "Relume Controls": the window for the person running Relume: camera
// selection, capture/permission status, an Output window toggle, a preview of
// the OutputTexture (the same texture the Output window shows) and, beside
// it, the depth debug view with the inference status and settings. ImGui
// draws only into this window.

#include "app/window.hpp"
#include "relume/app_state.hpp"
#include "relume/camera.hpp"
#include "relume/guided_capture.hpp"
#include "relume/inference_controller.hpp"
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

  // Depth inference.
  InferenceStatus inference;
  int inference_choice = 0;                 // index into kInferenceChoices
  std::string depth_label;                  // age and lag of the depth shown, or why there is none
  std::string inference_timing;             // e.g. "run p50 330 ms, p95 341 ms"
  const DepthDebugTexture* depth_view = nullptr;
  bool depth_overlay = false;
};

// The inference settings offered in the Controls window.
struct InferenceChoice {
  const char* label;
  std::optional<ExecutionProvider> provider;  // nullopt = off
  CoreMLUnits units;
};
inline constexpr InferenceChoice kInferenceChoices[] = {
    {"Off", std::nullopt, CoreMLUnits::kAll},
    {"CoreML: CPU and GPU (default)", ExecutionProvider::kCoreML, CoreMLUnits::kCPUAndGPU},
    {"CPU", ExecutionProvider::kCpu, CoreMLUnits::kAll},
    {"CoreML: ALL", ExecutionProvider::kCoreML, CoreMLUnits::kAll},
    {"CoreML: CPU and Neural Engine", ExecutionProvider::kCoreML, CoreMLUnits::kCPUAndNeuralEngine},
    {"CoreML: CPU only", ExecutionProvider::kCoreML, CoreMLUnits::kCPUOnly},
};

// What the user asked for this frame.
struct ControlsActions {
  std::optional<std::string> select_device;  // already confirmed for virtual cameras
  bool toggle_output = false;
  std::optional<std::size_t> guided_select;  // move to another shot
  bool guided_capture = false;               // Capture / Retake / Start burst
  std::optional<int> inference_choice;       // a new entry of kInferenceChoices
  bool toggle_depth_overlay = false;
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
  void BuildInference(const ControlsModel& model, ControlsActions& actions);

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
