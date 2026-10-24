// Camera system for platforms without a native adapter yet (Media Foundation
// on Windows is future work). It reports no devices, so the app shows
// "Camera unavailable" and the file source remains usable.

#include "relume/camera.hpp"

namespace relume {

namespace {

class UnsupportedCameraSystem final : public ICameraSystem {
 public:
  PermissionState permission() const override { return PermissionState::kAuthorized; }
  void RequestPermission() override {}
  std::vector<CameraDeviceInfo> EnumerateDevices() override { return {}; }
  std::unique_ptr<ICameraSource> CreateSource(const std::string&, const CaptureRequest&, LatestFrameSlot&) override {
    return nullptr;
  }
};

}  // namespace

std::unique_ptr<ICameraSystem> CreatePlatformCameraSystem(CameraEventQueue&) {
  return std::make_unique<UnsupportedCameraSystem>();
}

}  // namespace relume
