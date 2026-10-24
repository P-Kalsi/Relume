#pragma once

// Capture status shown to the user, derived from verified facts each frame
// rather than tracked separately, so it cannot drift from reality.

#include "relume/camera.hpp"

#include <string>
#include <string_view>

namespace relume {

enum class CaptureState : std::uint8_t {
  kPermissionPending,  // waiting for the user to answer the system prompt
  kPermissionDenied,   // denied or restricted
  kNoCamera,           // nothing usable connected, or nothing selected
  kStarting,           // source configured, no frame yet
  kLive,               // frames arriving
  kCameraUnavailable,  // the selected camera was disconnected
  kError,              // the source failed
};

struct CaptureFacts {
  PermissionState permission = PermissionState::kNotDetermined;
  bool any_device = false;        // at least one camera is listed (virtual or not)
  bool has_selection = false;     // a device is selected
  bool selected_present = false;  // the selected device is currently connected
  SourceState source = SourceState::kIdle;
  bool frame_received = false;    // the current source has delivered a frame
  std::string error;              // the source's failure message, if any
};

struct CaptureStatus {
  CaptureState state = CaptureState::kNoCamera;
  std::string detail;  // a sentence for the Controls window
};

CaptureStatus DeriveStatus(const CaptureFacts& facts);

// Short label in the PRD's wording ("Camera unavailable", "Error", ...).
std::string_view StatusLabel(CaptureState state);

// True when capture should be (re)started without user action: permission is
// granted, the selected camera is connected and nothing is running or failed.
bool ShouldStartCapture(const CaptureFacts& facts);

}  // namespace relume
