#pragma once

// Portable camera interfaces and the pure decision logic behind them (format
// negotiation, default device choice). Platform adapters (AVFoundation now,
// Media Foundation later) implement ICameraSystem and ICameraSource.

#include "relume/video_frame.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace relume {

class LatestFrameSlot;
class CameraEventQueue;

struct FrameRateRange {
  double min_fps = 0.0;
  double max_fps = 0.0;
};

// One capture format offered by a device (AVCaptureDeviceFormat).
struct CameraFormat {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<FrameRateRange> frame_rates;
  std::string native_pixel_format;  // e.g. "420v"; informational
};

struct CameraDeviceInfo {
  std::string id;    // stable unique ID
  std::string name;  // localized display name
  bool is_virtual = false;
  std::vector<CameraFormat> formats;
};

struct CaptureRequest {
  std::uint32_t width = 1280;
  std::uint32_t height = 720;
  double fps = 30.0;
  PixelFormat pixel_format = PixelFormat::kNv12;  // kNv12 or kBgra8
};

struct FormatChoice {
  std::size_t index = 0;  // into the device's formats
  double fps = 0.0;       // frame rate to request, within one of its ranges
};

// Chooses the device format closest to the request: first by achievable
// frame rate, then exact size, then aspect ratio, then area. Returns nullopt
// if there are no usable formats.
std::optional<FormatChoice> SelectFormat(std::span<const CameraFormat> formats, const CaptureRequest& request);

// A pixel format the capture output can deliver (from the output's list of
// supported formats, which is separate from the device's native format).
struct PixelFormatOption {
  PixelFormat format = PixelFormat::kBgra8;
  YuvRange range = YuvRange::kVideo;  // for kNv12
  std::string code;                   // platform code, e.g. "420v"
};

// Chooses the output pixel format: the requested kind if offered (for NV12,
// the device's native code first, then video range, then full range), else
// the other supported kind. Returns nullopt if none is supported.
std::optional<std::size_t> ChooseOutputPixelFormat(std::span<const PixelFormatOption> options,
                                                   PixelFormat preferred, std::string_view native_code);

// Matrix to assume when a YCbCr buffer carries no matrix metadata: BT.601 for
// SD and BT.709 for HD (720 lines or more), the CoreVideo convention.
YuvMatrix FallbackYuvMatrix(std::uint32_t width, std::uint32_t height);

// True for cameras that must never be chosen automatically because they may
// carry Relume's own output (feedback loop), e.g. "OBS Virtual Camera".
bool IsLikelyVirtualCamera(std::string_view name);

// The device to open without asking: `preferred_id` if it is connected,
// otherwise the first non-virtual device. Virtual cameras are never chosen.
std::optional<std::size_t> ChooseDefaultDevice(std::span<const CameraDeviceInfo> devices,
                                               std::string_view preferred_id);

enum class PermissionState : std::uint8_t { kNotDetermined, kAuthorized, kDenied, kRestricted };

enum class SourceState : std::uint8_t {
  kIdle,      // not started, or stopped
  kStarting,  // configuring / waiting for the first frame
  kRunning,   // frames are arriving
  kFailed,    // could not start, or a runtime error stopped capture
};

struct CaptureStats {
  std::uint64_t frames_received = 0;   // callbacks that carried an image
  std::uint64_t frames_published = 0;  // copied and handed to the slot
  std::uint64_t dropped_pool = 0;      // no free buffer (consumers too slow)
  std::uint64_t dropped_driver = 0;    // discarded by the capture system as late
  std::uint64_t dropped_invalid = 0;   // unsupported format or layout
  double copy_ms_p50 = 0.0;            // time to copy one frame out of the camera buffer
  double copy_ms_p95 = 0.0;
};

// A running capture from one device. Frames are published to the
// LatestFrameSlot given at creation; events go to the CameraEventQueue. Both
// must outlive the source.
class ICameraSource {
 public:
  virtual ~ICameraSource() = default;
  // Begins capture asynchronously. Progress and failures are reported
  // through state() and events.
  virtual void Start() = 0;
  // Stops capture and returns once no further frame callback can run. Must
  // not be called from a source callback.
  virtual void Stop() = 0;
  virtual SourceState state() const = 0;
  virtual CaptureStats stats() const = 0;
  virtual std::string description() const = 0;  // negotiated format, for logs and UI
};

class ICameraSystem {
 public:
  virtual ~ICameraSystem() = default;
  virtual PermissionState permission() const = 0;
  // Shows the system prompt if permission is undetermined. Asynchronous: the
  // answer arrives as a kPermissionChanged event. Never blocks.
  virtual void RequestPermission() = 0;
  virtual std::vector<CameraDeviceInfo> EnumerateDevices() = 0;
  virtual std::unique_ptr<ICameraSource> CreateSource(const std::string& device_id, const CaptureRequest& request,
                                                      LatestFrameSlot& slot) = 0;
};

// The platform's camera system (AVFoundation on macOS). Elsewhere it reports
// no devices. `events` must outlive the system and its sources.
std::unique_ptr<ICameraSystem> CreatePlatformCameraSystem(CameraEventQueue& events);

}  // namespace relume
