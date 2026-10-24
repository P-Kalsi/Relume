#include "relume/app_state.hpp"

namespace relume {

CaptureStatus DeriveStatus(const CaptureFacts& facts) {
  switch (facts.permission) {
    case PermissionState::kNotDetermined:
      return {CaptureState::kPermissionPending, "Waiting for camera permission. Answer the macOS prompt."};
    case PermissionState::kDenied:
    case PermissionState::kRestricted:
      return {CaptureState::kPermissionDenied,
              "Camera access is off. Enable Relume in System Settings > Privacy & Security > Camera, then "
              "restart Relume."};
    case PermissionState::kAuthorized:
      break;
  }
  if (facts.source == SourceState::kFailed) {
    return {CaptureState::kError, facts.error.empty() ? "The camera stopped with an error." : facts.error};
  }
  if (facts.has_selection && !facts.selected_present) {
    return {CaptureState::kCameraUnavailable, "The selected camera is not connected. Waiting for it to appear."};
  }
  if (!facts.any_device) {
    return {CaptureState::kNoCamera, "No camera found."};
  }
  if (!facts.has_selection) {
    return {CaptureState::kNoCamera, "No camera selected. Virtual cameras are never selected automatically."};
  }
  if (facts.source == SourceState::kRunning && facts.frame_received) {
    return {CaptureState::kLive, "Live."};
  }
  return {CaptureState::kStarting, "Starting the camera."};
}

std::string_view StatusLabel(CaptureState state) {
  switch (state) {
    case CaptureState::kPermissionPending: return "Waiting for permission";
    case CaptureState::kPermissionDenied: return "Camera access denied";
    case CaptureState::kNoCamera: return "Camera unavailable";
    case CaptureState::kStarting: return "Starting";
    case CaptureState::kLive: return "Live";
    case CaptureState::kCameraUnavailable: return "Camera unavailable";
    case CaptureState::kError: return "Error";
  }
  return "Error";
}

bool ShouldStartCapture(const CaptureFacts& facts) {
  return facts.permission == PermissionState::kAuthorized && facts.has_selection && facts.selected_present &&
         facts.source == SourceState::kIdle;
}

}  // namespace relume
