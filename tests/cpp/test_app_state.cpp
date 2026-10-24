#include "relume/app_state.hpp"

#include <gtest/gtest.h>

namespace relume {
namespace {

CaptureFacts Live() {
  CaptureFacts facts;
  facts.permission = PermissionState::kAuthorized;
  facts.any_device = true;
  facts.has_selection = true;
  facts.selected_present = true;
  facts.source = SourceState::kRunning;
  facts.frame_received = true;
  return facts;
}

TEST(CaptureStatus, PermissionStates) {
  CaptureFacts facts = Live();
  facts.permission = PermissionState::kNotDetermined;
  EXPECT_EQ(DeriveStatus(facts).state, CaptureState::kPermissionPending);
  facts.permission = PermissionState::kDenied;
  EXPECT_EQ(DeriveStatus(facts).state, CaptureState::kPermissionDenied);
  EXPECT_NE(DeriveStatus(facts).detail.find("System Settings"), std::string::npos);
  facts.permission = PermissionState::kRestricted;
  EXPECT_EQ(DeriveStatus(facts).state, CaptureState::kPermissionDenied);
  EXPECT_FALSE(ShouldStartCapture(facts));
}

TEST(CaptureStatus, NoDevices) {
  CaptureFacts facts = Live();
  facts.any_device = false;
  facts.has_selection = false;
  facts.selected_present = false;
  facts.source = SourceState::kIdle;
  facts.frame_received = false;
  EXPECT_EQ(DeriveStatus(facts).state, CaptureState::kNoCamera);
  EXPECT_EQ(StatusLabel(CaptureState::kNoCamera), "Camera unavailable");
  EXPECT_FALSE(ShouldStartCapture(facts));
}

TEST(CaptureStatus, OnlyVirtualCamerasMeansNoSelection) {
  CaptureFacts facts = Live();
  facts.has_selection = false;
  facts.selected_present = false;
  facts.source = SourceState::kIdle;
  const CaptureStatus status = DeriveStatus(facts);
  EXPECT_EQ(status.state, CaptureState::kNoCamera);
  EXPECT_NE(status.detail.find("Virtual"), std::string::npos);
}

TEST(CaptureStatus, StartingUntilFirstFrame) {
  CaptureFacts facts = Live();
  facts.source = SourceState::kStarting;
  facts.frame_received = false;
  EXPECT_EQ(DeriveStatus(facts).state, CaptureState::kStarting);
  facts.source = SourceState::kRunning;
  EXPECT_EQ(DeriveStatus(facts).state, CaptureState::kStarting);
  facts.frame_received = true;
  EXPECT_EQ(DeriveStatus(facts).state, CaptureState::kLive);
}

TEST(CaptureStatus, DisconnectThenReconnect) {
  CaptureFacts facts = Live();
  // Disconnect: the app stops the source and the device leaves the list.
  facts.selected_present = false;
  facts.source = SourceState::kIdle;
  facts.frame_received = false;
  EXPECT_EQ(DeriveStatus(facts).state, CaptureState::kCameraUnavailable);
  EXPECT_FALSE(ShouldStartCapture(facts));
  // Even with no devices left at all, the selection is remembered.
  facts.any_device = false;
  EXPECT_EQ(DeriveStatus(facts).state, CaptureState::kCameraUnavailable);
  // Reconnect.
  facts.any_device = true;
  facts.selected_present = true;
  EXPECT_TRUE(ShouldStartCapture(facts));
  facts.source = SourceState::kStarting;
  EXPECT_EQ(DeriveStatus(facts).state, CaptureState::kStarting);
  EXPECT_FALSE(ShouldStartCapture(facts));
  facts.source = SourceState::kRunning;
  facts.frame_received = true;
  EXPECT_EQ(DeriveStatus(facts).state, CaptureState::kLive);
}

TEST(CaptureStatus, FailureIsReportedAndNotRetriedAutomatically) {
  CaptureFacts facts = Live();
  facts.source = SourceState::kFailed;
  facts.error = "Unsupported pixel format";
  const CaptureStatus status = DeriveStatus(facts);
  EXPECT_EQ(status.state, CaptureState::kError);
  EXPECT_EQ(status.detail, "Unsupported pixel format");
  EXPECT_EQ(StatusLabel(status.state), "Error");
  EXPECT_FALSE(ShouldStartCapture(facts));
}

}  // namespace
}  // namespace relume
