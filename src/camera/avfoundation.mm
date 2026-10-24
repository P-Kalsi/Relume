// AVFoundation implementation of ICameraSystem / ICameraSource (macOS).
//
// Threads:
//   - main thread: the app calls every public method from here.
//   - session queue (serial GCD): configures, starts and stops the session,
//     because -startRunning blocks.
//   - frame queue (serial GCD): sample-buffer callbacks. A callback only locks
//     the CVPixelBuffer, copies its planes into a pooled VideoFrame, unlocks
//     and publishes. It never waits for the main thread or the GPU.
// Stop() detaches the delegate on the session queue, then drains the frame
// queue, so no callback can run afterwards. It refuses to run on either queue
// (that would deadlock).

#include "relume/camera.hpp"

#include "relume/event_queue.hpp"
#include "relume/frame_queue.hpp"
#include "relume/stats.hpp"

#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace relume {
namespace {

using Clock = std::chrono::steady_clock;

std::string FourCc(FourCharCode code) {
  const char chars[4] = {static_cast<char>((code >> 24) & 0xFF), static_cast<char>((code >> 16) & 0xFF),
                         static_cast<char>((code >> 8) & 0xFF), static_cast<char>(code & 0xFF)};
  return std::string(chars, 4);
}

std::string ToStd(NSString* text) { return text != nil ? std::string(text.UTF8String) : std::string(); }

// Posts to the app's event queue while it exists. System callbacks (the
// permission completion handler, notifications) can outlive the camera
// system, so they hold this by shared_ptr and the system detaches it.
class EventRelay {
 public:
  explicit EventRelay(CameraEventQueue& queue) : queue_(&queue) {}
  void Post(CameraEvent event) {
    std::lock_guard lock(mutex_);
    if (queue_ != nullptr) {
      queue_->Post(std::move(event));
    }
  }
  void Detach() {
    std::lock_guard lock(mutex_);
    queue_ = nullptr;
  }

 private:
  std::mutex mutex_;
  CameraEventQueue* queue_;
};

// Pixel formats Relume can consume, by CoreVideo code.
std::optional<PixelFormatOption> OptionFor(OSType code) {
  switch (code) {
    case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange: return PixelFormatOption{PixelFormat::kNv12, YuvRange::kVideo, "420v"};
    case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange: return PixelFormatOption{PixelFormat::kNv12, YuvRange::kFull, "420f"};
    case kCVPixelFormatType_32BGRA: return PixelFormatOption{PixelFormat::kBgra8, YuvRange::kFull, "BGRA"};
    default: return std::nullopt;
  }
}

// BT.709 or BT.601 from the buffer's attachment; nullopt if missing or
// another matrix (e.g. SMPTE 240M, BT.2020), which takes the logged fallback.
std::optional<YuvMatrix> MatrixFrom(CVImageBufferRef image) {
  CFTypeRef value = CVBufferCopyAttachment(image, kCVImageBufferYCbCrMatrixKey, nullptr);
  if (value == nullptr) {
    return std::nullopt;
  }
  std::optional<YuvMatrix> matrix;
  if (CFEqual(value, kCVImageBufferYCbCrMatrix_ITU_R_709_2)) {
    matrix = YuvMatrix::kBt709;
  } else if (CFEqual(value, kCVImageBufferYCbCrMatrix_ITU_R_601_4)) {
    matrix = YuvMatrix::kBt601;
  }
  CFRelease(value);
  return matrix;
}

std::optional<ChromaLocation> ChromaLocationFrom(CVImageBufferRef image) {
  CFTypeRef value = CVBufferCopyAttachment(image, kCVImageBufferChromaLocationTopFieldKey, nullptr);
  if (value == nullptr) {
    return std::nullopt;
  }
  std::optional<ChromaLocation> location;
  if (CFEqual(value, kCVImageBufferChromaLocation_Left)) {
    location = ChromaLocation::kLeft;
  } else if (CFEqual(value, kCVImageBufferChromaLocation_Center)) {
    location = ChromaLocation::kCenter;
  } else if (CFEqual(value, kCVImageBufferChromaLocation_TopLeft)) {
    location = ChromaLocation::kTopLeft;
  } else if (CFEqual(value, kCVImageBufferChromaLocation_Top)) {
    location = ChromaLocation::kTop;
  } else if (CFEqual(value, kCVImageBufferChromaLocation_BottomLeft)) {
    location = ChromaLocation::kBottomLeft;
  } else if (CFEqual(value, kCVImageBufferChromaLocation_Bottom)) {
    location = ChromaLocation::kBottom;
  }
  CFRelease(value);
  return location;
}

std::vector<CameraFormat> FormatsOf(AVCaptureDevice* device) {
  std::vector<CameraFormat> formats;
  for (AVCaptureDeviceFormat* format in device.formats) {
    const CMVideoDimensions dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription);
    CameraFormat entry;
    entry.width = static_cast<std::uint32_t>(dims.width);
    entry.height = static_cast<std::uint32_t>(dims.height);
    entry.native_pixel_format = FourCc(CMFormatDescriptionGetMediaSubType(format.formatDescription));
    for (AVFrameRateRange* range in format.videoSupportedFrameRateRanges) {
      entry.frame_rates.push_back({range.minFrameRate, range.maxFrameRate});
    }
    formats.push_back(std::move(entry));
  }
  return formats;
}

NSArray<AVCaptureDevice*>* DiscoverDevices() {
  AVCaptureDeviceDiscoverySession* discovery = [AVCaptureDeviceDiscoverySession
      discoverySessionWithDeviceTypes:@[ AVCaptureDeviceTypeBuiltInWideAngleCamera, AVCaptureDeviceTypeExternal,
                                         AVCaptureDeviceTypeContinuityCamera ]
                            mediaType:AVMediaTypeVideo
                             position:AVCaptureDevicePositionUnspecified];
  return discovery.devices;
}

class AvfCameraSource;

}  // namespace
}  // namespace relume

// Forwards sample-buffer callbacks to the C++ source. The source outlives the
// delegate's use: Stop() detaches it and drains the queue before destruction.
@interface RelumeSampleDelegate : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>
- (instancetype)initWithSource:(relume::AvfCameraSource*)source;
@end

namespace relume {
namespace {

int kQueueKey = 0;  // address used as a dispatch queue-specific key

class AvfCameraSource final : public ICameraSource {
 public:
  AvfCameraSource(std::string device_id, CaptureRequest request, LatestFrameSlot& slot,
                  std::shared_ptr<EventRelay> relay)
      : device_id_(std::move(device_id)), request_(request), slot_(slot), relay_(std::move(relay)) {
    session_queue_ = dispatch_queue_create("dev.relume.camera.session", DISPATCH_QUEUE_SERIAL);
    frame_queue_ = dispatch_queue_create("dev.relume.camera.frames", DISPATCH_QUEUE_SERIAL);
    dispatch_queue_set_specific(session_queue_, &kQueueKey, this, nullptr);
    dispatch_queue_set_specific(frame_queue_, &kQueueKey, this, nullptr);
  }

  ~AvfCameraSource() override { Stop(); }

  void Start() override {
    if (started_) {
      return;
    }
    started_ = true;
    state_ = SourceState::kStarting;
    dispatch_async(session_queue_, ^{
      Configure();
    });
  }

  void Stop() override {
    if (dispatch_get_specific(&kQueueKey) == this) {
      std::fprintf(stderr, "relume: AvfCameraSource::Stop called from its own queue; ignored\n");
      return;  // would deadlock
    }
    if (!started_) {
      return;
    }
    started_ = false;
    accepting_ = false;
    dispatch_sync(session_queue_, ^{
      if (session_ != nil) {
        [session_ stopRunning];
      }
      [output_ setSampleBufferDelegate:nil queue:nil];
      if (runtime_error_observer_ != nil) {
        [[NSNotificationCenter defaultCenter] removeObserver:runtime_error_observer_];
        runtime_error_observer_ = nil;
      }
      session_ = nil;
      output_ = nil;
      delegate_ = nil;
    });
    dispatch_sync(frame_queue_, ^{
                  });  // wait for any callback already running
    if (state() != SourceState::kFailed) {
      state_ = SourceState::kIdle;
    }
  }

  SourceState state() const override {
    return runtime_failed_->load() ? SourceState::kFailed : state_.load();
  }

  CaptureStats stats() const override {
    CaptureStats stats;
    stats.frames_received = received_.load();
    stats.frames_published = published_.load();
    stats.dropped_pool = pool_.exhausted_count();
    stats.dropped_driver = dropped_driver_.load();
    stats.dropped_invalid = dropped_invalid_.load();
    stats.copy_ms_p50 = copy_ms_.Percentile(0.5);
    stats.copy_ms_p95 = copy_ms_.Percentile(0.95);
    stats.interval_ms_p50 = interval_ms_.Percentile(0.5);
    stats.interval_ms_p95 = interval_ms_.Percentile(0.95);
    return stats;
  }

  std::string description() const override {
    std::lock_guard lock(description_mutex_);
    return description_;
  }

  // Called on the frame queue.
  void OnSampleBuffer(CMSampleBufferRef sample) noexcept {
    try {
      HandleSample(sample);
    } catch (...) {
      dropped_invalid_.fetch_add(1);  // never let an exception escape into AVFoundation
    }
  }
  void OnDroppedSample() noexcept { dropped_driver_.fetch_add(1); }

 private:
  void Fail(const std::string& message) {
    state_ = SourceState::kFailed;
    relay_->Post({CameraEvent::Kind::kSourceFailed, device_id_, message});
  }

  void SetDescription(std::string text) {
    std::lock_guard lock(description_mutex_);
    description_ = std::move(text);
  }

  // Runs on the session queue.
  void Configure() {
    AVCaptureDevice* device = [AVCaptureDevice deviceWithUniqueID:@(device_id_.c_str())];
    if (device == nil) {
      Fail("The camera is not available.");
      return;
    }
    const std::vector<CameraFormat> formats = FormatsOf(device);
    const std::optional<FormatChoice> choice = SelectFormat(formats, request_);
    if (!choice) {
      Fail("The camera offers no usable video format.");
      return;
    }
    AVCaptureDeviceFormat* format = device.formats[choice->index];
    const CameraFormat& chosen = formats[choice->index];

    NSError* error = nil;
    AVCaptureDeviceInput* input = [AVCaptureDeviceInput deviceInputWithDevice:device error:&error];
    if (input == nil) {
      Fail("Cannot open the camera: " + ToStd(error.localizedDescription));
      return;
    }
    AVCaptureSession* session = [[AVCaptureSession alloc] init];
    [session beginConfiguration];
    // No session preset: on macOS, setting the locked device's activeFormat
    // (below) selects the capture format. The Phase 0 probe's 1280x720
    // preset delivered 1920x1080; the delivered size is verified on the first
    // frame and reported if it differs.
    if (![session canAddInput:input]) {
      [session commitConfiguration];
      Fail("The capture session rejected the camera.");
      return;
    }
    [session addInput:input];

    AVCaptureVideoDataOutput* output = [[AVCaptureVideoDataOutput alloc] init];
    // The output's supported pixel formats are separate from the device's
    // native format; choose among what the output can actually deliver.
    std::vector<PixelFormatOption> options;
    std::vector<OSType> codes;
    for (NSNumber* number in output.availableVideoCVPixelFormatTypes) {
      const OSType code = number.unsignedIntValue;
      if (std::optional<PixelFormatOption> option = OptionFor(code)) {
        options.push_back(*option);
        codes.push_back(code);
      }
    }
    const std::optional<std::size_t> pixel =
        ChooseOutputPixelFormat(options, request_.pixel_format, chosen.native_pixel_format);
    if (!pixel) {
      [session commitConfiguration];
      Fail("The camera output supports none of NV12 ('420v'/'420f') or BGRA.");
      return;
    }
    output.videoSettings = @{(id)kCVPixelBufferPixelFormatTypeKey : @(codes[*pixel])};
    output.alwaysDiscardsLateVideoFrames = YES;  // drop stale frames instead of queueing
    delegate_ = [[RelumeSampleDelegate alloc] initWithSource:this];
    [output setSampleBufferDelegate:delegate_ queue:frame_queue_];
    if (![session canAddOutput:output]) {
      [session commitConfiguration];
      Fail("The capture session rejected the video output.");
      return;
    }
    [session addOutput:output];

    // Lock across startRunning so the session cannot replace the format.
    if (![device lockForConfiguration:&error]) {
      [session commitConfiguration];
      Fail("Cannot configure the camera: " + ToStd(error.localizedDescription));
      return;
    }
    device.activeFormat = format;
    CMTime duration = CMTimeMakeWithSeconds(1.0 / choice->fps, 600);
    for (AVFrameRateRange* range in format.videoSupportedFrameRateRanges) {
      if (std::abs(range.maxFrameRate - choice->fps) < 1e-3) {
        duration = range.minFrameDuration;  // exact value for e.g. 29.97
      }
    }
    device.activeVideoMinFrameDuration = duration;
    device.activeVideoMaxFrameDuration = duration;

    AVCaptureConnection* connection = [output connectionWithMediaType:AVMediaTypeVideo];
    // Unmirrored and unrotated: what the camera sees, as remote viewers expect.
    if (connection.isVideoMirroringSupported) {
      connection.automaticallyAdjustsVideoMirroring = NO;
      connection.videoMirrored = NO;
    }
    if ([connection isVideoRotationAngleSupported:0]) {
      connection.videoRotationAngle = 0;
    }
    [session commitConfiguration];

    // The block may still be running when Stop() removes the observer, so it
    // holds only shared state, never `this`.
    std::shared_ptr<EventRelay> relay = relay_;
    std::shared_ptr<std::atomic<bool>> failed = runtime_failed_;
    std::string device_id = device_id_;
    runtime_error_observer_ = [[NSNotificationCenter defaultCenter]
        addObserverForName:AVCaptureSessionRuntimeErrorNotification
                    object:session
                     queue:nil
                usingBlock:^(NSNotification* note) {
                  NSError* runtime = note.userInfo[AVCaptureSessionErrorKey];
                  failed->store(true);
                  relay->Post({CameraEvent::Kind::kSourceFailed, device_id,
                               "The camera stopped: " + ToStd(runtime.localizedDescription)});
                }];

    session_ = session;
    output_ = output;
    char text[256];
    std::snprintf(text, sizeof(text), "%s: device format %ux%u '%s' @ %.2f fps; output '%s'%s",
                  ToStd(device.localizedName).c_str(), chosen.width, chosen.height,
                  chosen.native_pixel_format.c_str(), choice->fps, options[*pixel].code.c_str(),
                  connection.isVideoMirrored ? " (mirrored!)" : ", unmirrored");
    SetDescription(text);
    requested_format_ = options[*pixel];
    requested_width_ = chosen.width;
    requested_height_ = chosen.height;
    accepting_ = true;
    [session startRunning];
    [device unlockForConfiguration];
    if (!session.isRunning) {
      Fail("The camera did not start.");
    }
  }

  // Runs on the frame queue.
  void HandleSample(CMSampleBufferRef sample) {
    if (!accepting_.load()) {
      return;
    }
    received_.fetch_add(1);
    const CMTime pts = CMSampleBufferGetPresentationTimeStamp(sample);
    const std::int64_t pts_ns = CMTIME_IS_VALID(pts) ? CMTimeConvertScale(pts, 1000000000, kCMTimeRoundingMethod_Default).value : 0;
    if (last_pts_ns_ != 0 && pts_ns > last_pts_ns_) {
      interval_ms_.Record(static_cast<double>(pts_ns - last_pts_ns_) / 1e6);
    }
    last_pts_ns_ = pts_ns;

    CVImageBufferRef image = CMSampleBufferGetImageBuffer(sample);
    if (image == nullptr) {
      dropped_invalid_.fetch_add(1);
      return;
    }
    const std::optional<PixelFormatOption> option = OptionFor(CVPixelBufferGetPixelFormatType(image));
    if (!option) {
      dropped_invalid_.fetch_add(1);
      WarnOnce(warned_format_, "The camera delivered unsupported pixel format '" +
                                   FourCc(CVPixelBufferGetPixelFormatType(image)) + "'; frames are dropped.");
      return;
    }
    std::shared_ptr<VideoFrame> frame = pool_.Acquire();
    if (!frame) {
      return;  // consumers are behind; counted by the pool
    }
    const Clock::time_point copy_start = Clock::now();
    const auto width = static_cast<std::uint32_t>(CVPixelBufferGetWidth(image));
    const auto height = static_cast<std::uint32_t>(CVPixelBufferGetHeight(image));
    AllocateFrame(*frame, option->format, width, height);  // reuses the pooled buffer
    frame->range = option->range;

    if (CVPixelBufferLockBaseAddress(image, kCVPixelBufferLock_ReadOnly) != kCVReturnSuccess) {
      dropped_invalid_.fetch_add(1);
      return;
    }
    const bool planar = CVPixelBufferIsPlanar(image);
    const std::size_t plane_count = planar ? CVPixelBufferGetPlaneCount(image) : 1;
    bool layout_ok = plane_count == frame->plane_count;
    std::array<std::size_t, 2> source_strides{};
    for (std::uint32_t p = 0; layout_ok && p < frame->plane_count; ++p) {
      const PlaneLayout& dst = frame->planes[p];
      const auto* src = static_cast<const std::uint8_t*>(planar ? CVPixelBufferGetBaseAddressOfPlane(image, p)
                                                                : CVPixelBufferGetBaseAddress(image));
      const std::size_t stride = planar ? CVPixelBufferGetBytesPerRowOfPlane(image, p) : CVPixelBufferGetBytesPerRow(image);
      const std::size_t plane_w = planar ? CVPixelBufferGetWidthOfPlane(image, p) : width;
      const std::size_t plane_h = planar ? CVPixelBufferGetHeightOfPlane(image, p) : height;
      const std::size_t row_bytes = std::size_t{dst.width} * dst.bytes_per_sample;
      // The chroma plane must be exactly 4:2:0; anything else is rejected
      // rather than converted with the wrong layout.
      layout_ok = src != nullptr && plane_w == dst.width && plane_h == dst.height && stride >= row_bytes;
      if (!layout_ok) {
        break;
      }
      source_strides[p] = stride;
      std::byte* out = frame->data.data() + dst.offset;
      for (std::uint32_t row = 0; row < dst.height; ++row) {
        std::memcpy(out + dst.stride_bytes * row, src + stride * row, row_bytes);
      }
    }
    CVPixelBufferUnlockBaseAddress(image, kCVPixelBufferLock_ReadOnly);
    if (!layout_ok) {
      dropped_invalid_.fetch_add(1);
      WarnOnce(warned_layout_, "The camera delivered a buffer layout Relume cannot use (expected " +
                                   std::string(ToString(option->format)) + " 4:2:0); frames are dropped.");
      return;
    }

    if (option->format == PixelFormat::kNv12) {
      if (const std::optional<YuvMatrix> matrix = MatrixFrom(image)) {
        frame->matrix = *matrix;
      } else {
        frame->matrix = FallbackYuvMatrix(width, height);
        WarnOnce(warned_matrix_, "No supported YCbCr matrix in the camera metadata; assuming " +
                                     std::string(ToString(frame->matrix)) + " for this size.");
      }
      if (const std::optional<ChromaLocation> location = ChromaLocationFrom(image)) {
        frame->chroma_location = *location;
      } else {
        frame->chroma_location = ChromaLocation::kLeft;
        WarnOnce(warned_chroma_, "No chroma location in the camera metadata; assuming left (MPEG-2/H.264 "
                                 "default). Relume's per-2x2-block chroma does not depend on it.");
      }
    }
    frame->frame_id = next_frame_id_++;
    frame->capture_time_ns = pts_ns;
    frame->clock = TimestampClock::kHostTime;
    frame->arrival_time = Clock::now();
    copy_ms_.Record(std::chrono::duration<double, std::milli>(frame->arrival_time - copy_start).count());

    if (frame->frame_id == 1) {
      ReportFirstFrame(*frame, source_strides);
    }
    slot_.Publish(std::move(frame));
    published_.fetch_add(1);
  }

  void ReportFirstFrame(const VideoFrame& frame, const std::array<std::size_t, 2>& strides) {
    char text[320];
    std::snprintf(text, sizeof(text), "delivered %ux%u %s%s%s, stride %zu%s%s, %s, chroma %s", frame.width,
                  frame.height, ToString(frame.format).data(), frame.format == PixelFormat::kNv12 ? " " : "",
                  frame.format == PixelFormat::kNv12 ? ToString(frame.range).data() : "", strides[0],
                  frame.plane_count == 2 ? "/" : "",
                  frame.plane_count == 2 ? std::to_string(strides[1]).c_str() : "",
                  frame.format == PixelFormat::kNv12 ? ToString(frame.matrix).data() : "RGB",
                  frame.format == PixelFormat::kNv12 ? ToString(frame.chroma_location).data() : "n/a");
    std::string full = description() + "; " + text;
    SetDescription(full);
    if (frame.width != requested_width_ || frame.height != requested_height_ ||
        frame.format != requested_format_.format || frame.range != requested_format_.range) {
      relay_->Post({CameraEvent::Kind::kWarning, device_id_,
                    "The camera delivered a different format than configured: " + std::string(text)});
    }
    state_ = SourceState::kRunning;
    relay_->Post({CameraEvent::Kind::kSourceStarted, device_id_, full});
  }

  void WarnOnce(std::atomic<bool>& flag, const std::string& message) {
    if (!flag.exchange(true)) {
      relay_->Post({CameraEvent::Kind::kWarning, device_id_, message});
    }
  }

  const std::string device_id_;
  const CaptureRequest request_;
  LatestFrameSlot& slot_;
  std::shared_ptr<EventRelay> relay_;
  FramePool pool_{4};
  bool started_ = false;  // main thread only
  std::atomic<bool> accepting_{false};
  std::atomic<SourceState> state_{SourceState::kIdle};
  std::shared_ptr<std::atomic<bool>> runtime_failed_ = std::make_shared<std::atomic<bool>>(false);

  // Session-queue state.
  dispatch_queue_t session_queue_;
  dispatch_queue_t frame_queue_;
  AVCaptureSession* session_ = nil;
  AVCaptureVideoDataOutput* output_ = nil;
  RelumeSampleDelegate* delegate_ = nil;
  id runtime_error_observer_ = nil;
  PixelFormatOption requested_format_;
  std::uint32_t requested_width_ = 0;
  std::uint32_t requested_height_ = 0;

  // Frame-queue state.
  std::uint64_t next_frame_id_ = 1;
  std::int64_t last_pts_ns_ = 0;
  std::atomic<bool> warned_format_{false};
  std::atomic<bool> warned_layout_{false};
  std::atomic<bool> warned_matrix_{false};
  std::atomic<bool> warned_chroma_{false};

  // Shared (atomics / histograms are lock-free).
  std::atomic<std::uint64_t> received_{0};
  std::atomic<std::uint64_t> published_{0};
  std::atomic<std::uint64_t> dropped_driver_{0};
  std::atomic<std::uint64_t> dropped_invalid_{0};
  LatencyHistogram copy_ms_;
  LatencyHistogram interval_ms_;
  mutable std::mutex description_mutex_;
  std::string description_;
};

class AvfCameraSystem final : public ICameraSystem {
 public:
  explicit AvfCameraSystem(CameraEventQueue& events) : relay_(std::make_shared<EventRelay>(events)) {
    std::shared_ptr<EventRelay> relay = relay_;
    NSNotificationCenter* center = [NSNotificationCenter defaultCenter];
    auto on_change = ^(NSNotification* note) {
      AVCaptureDevice* device = note.object;
      if (![device hasMediaType:AVMediaTypeVideo]) {
        return;  // e.g. microphones, which Relume never opens
      }
      relay->Post({CameraEvent::Kind::kDevicesChanged, ToStd(device.uniqueID),
                   ToStd(device.localizedName) + ([note.name isEqualToString:AVCaptureDeviceWasConnectedNotification]
                                                      ? " connected"
                                                      : " disconnected")});
    };
    connected_observer_ = [center addObserverForName:AVCaptureDeviceWasConnectedNotification
                                              object:nil
                                               queue:nil
                                          usingBlock:on_change];
    disconnected_observer_ = [center addObserverForName:AVCaptureDeviceWasDisconnectedNotification
                                                 object:nil
                                                  queue:nil
                                             usingBlock:on_change];
  }

  ~AvfCameraSystem() override {
    [[NSNotificationCenter defaultCenter] removeObserver:connected_observer_];
    [[NSNotificationCenter defaultCenter] removeObserver:disconnected_observer_];
    relay_->Detach();  // a late permission answer must not touch a dead queue
  }

  PermissionState permission() const override {
    switch ([AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo]) {
      case AVAuthorizationStatusNotDetermined: return PermissionState::kNotDetermined;
      case AVAuthorizationStatusRestricted: return PermissionState::kRestricted;
      case AVAuthorizationStatusDenied: return PermissionState::kDenied;
      case AVAuthorizationStatusAuthorized: return PermissionState::kAuthorized;
    }
    return PermissionState::kDenied;
  }

  void RequestPermission() override {
    if (permission() != PermissionState::kNotDetermined || requested_) {
      return;
    }
    requested_ = true;
    std::shared_ptr<EventRelay> relay = relay_;
    // Shows the system prompt; the answer arrives on an arbitrary queue.
    [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo
                             completionHandler:^(BOOL granted) {
                               relay->Post({CameraEvent::Kind::kPermissionChanged, "",
                                            granted ? "Camera access allowed" : "Camera access denied"});
                             }];
  }

  std::vector<CameraDeviceInfo> EnumerateDevices() override {
    std::vector<CameraDeviceInfo> devices;
    for (AVCaptureDevice* device in DiscoverDevices()) {
      CameraDeviceInfo info;
      info.id = ToStd(device.uniqueID);
      info.name = ToStd(device.localizedName);
      info.is_virtual = IsLikelyVirtualCamera(info.name);
      info.formats = FormatsOf(device);
      devices.push_back(std::move(info));
    }
    return devices;
  }

  std::unique_ptr<ICameraSource> CreateSource(const std::string& device_id, const CaptureRequest& request,
                                              LatestFrameSlot& slot) override {
    return std::make_unique<AvfCameraSource>(device_id, request, slot, relay_);
  }

 private:
  std::shared_ptr<EventRelay> relay_;
  id connected_observer_ = nil;
  id disconnected_observer_ = nil;
  bool requested_ = false;
};

}  // namespace

std::unique_ptr<ICameraSystem> CreatePlatformCameraSystem(CameraEventQueue& events) {
  return std::make_unique<AvfCameraSystem>(events);
}

}  // namespace relume

@implementation RelumeSampleDelegate {
  relume::AvfCameraSource* _source;
}

- (instancetype)initWithSource:(relume::AvfCameraSource*)source {
  self = [super init];
  if (self != nil) {
    _source = source;
  }
  return self;
}

- (void)captureOutput:(AVCaptureOutput*)output
    didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
           fromConnection:(AVCaptureConnection*)connection {
  (void)output;
  (void)connection;
  _source->OnSampleBuffer(sampleBuffer);
}

- (void)captureOutput:(AVCaptureOutput*)output
    didDropSampleBuffer:(CMSampleBufferRef)sampleBuffer
         fromConnection:(AVCaptureConnection*)connection {
  (void)output;
  (void)sampleBuffer;
  (void)connection;
  _source->OnDroppedSample();
}

@end
