// Phase 0 probe: AVFoundation camera permission + short capture.
//
// Run it as an app bundle (`open`), not by launching the inner executable
// from a terminal: macOS attributes camera permission (TCC) to the bundle only
// when it is launched as an app. Output goes to stdout; pass `--stdout <file>`
// to `open` to capture it.
//
// Exit codes: 0 frames captured, 2 permission denied/restricted,
// 3 no usable camera, 4 capture failed or timed out.

#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>

namespace {

constexpr int kFramesToCapture = 60;
constexpr double kTimeoutSeconds = 15.0;

const char* AuthStatusName(AVAuthorizationStatus status) {
  switch (status) {
    case AVAuthorizationStatusNotDetermined: return "not determined";
    case AVAuthorizationStatusRestricted: return "restricted";
    case AVAuthorizationStatusDenied: return "denied";
    case AVAuthorizationStatusAuthorized: return "authorized";
  }
  return "unknown";
}

// Spins the run loop until `done` is set or the timeout expires. Returns
// whether `done` was set.
bool WaitFor(const std::atomic<bool>& done, double timeout_seconds) {
  NSDate* deadline = [NSDate dateWithTimeIntervalSinceNow:timeout_seconds];
  while (!done.load() && [deadline timeIntervalSinceNow] > 0) {
    [[NSRunLoop currentRunLoop] runMode:NSDefaultRunLoopMode
                             beforeDate:[NSDate dateWithTimeIntervalSinceNow:0.05]];
  }
  return done.load();
}

bool IsVirtualCamera(AVCaptureDevice* device) {
  return [device.localizedName rangeOfString:@"OBS Virtual Camera"
                                     options:NSCaseInsensitiveSearch]
             .location != NSNotFound;
}

}  // namespace

// The sample-buffer callback only reads metadata and updates atomics. It never
// blocks, prints or waits; main() reports the results afterwards.
@interface FrameCounter : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>
@end

@implementation FrameCounter {
 @public
  std::atomic<int> frames;
  std::atomic<int> width;
  std::atomic<int> height;
  std::atomic<std::uint32_t> pixelFormat;
  std::atomic<double> firstTimestamp;
  std::atomic<double> lastTimestamp;
  std::atomic<bool> done;
}

- (void)captureOutput:(AVCaptureOutput*)output
    didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
           fromConnection:(AVCaptureConnection*)connection {
  (void)output;
  (void)connection;
  CVImageBufferRef image = CMSampleBufferGetImageBuffer(sampleBuffer);
  if (image == nullptr) {
    return;
  }
  const double seconds = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer));
  const int count = frames.fetch_add(1) + 1;
  if (count == 1) {
    width = static_cast<int>(CVPixelBufferGetWidth(image));
    height = static_cast<int>(CVPixelBufferGetHeight(image));
    pixelFormat = CVPixelBufferGetPixelFormatType(image);
    firstTimestamp = seconds;
  }
  lastTimestamp = seconds;
  if (count >= kFramesToCapture) {
    done = true;
  }
}
@end

int main() {
  @autoreleasepool {
    AVAuthorizationStatus status = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo];
    std::printf("Camera authorization at launch: %s\n", AuthStatusName(status));

    if (status == AVAuthorizationStatusNotDetermined) {
      std::printf("Requesting camera access (approve the macOS prompt)...\n");
      std::fflush(stdout);
      // The completion handler may run after we stop waiting, so the flag it
      // writes is shared-owned rather than living on this stack frame.
      auto answered = std::make_shared<std::atomic<bool>>(false);
      [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo
                               completionHandler:^(BOOL) {
                                 *answered = true;
                               }];
      // Give the person time to read and answer the system prompt.
      if (!WaitFor(*answered, 120.0)) {
        std::printf("FAIL: no answer to the permission prompt within 120 s\n");
        return 2;
      }
      status = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo];
      std::printf("Camera authorization after prompt: %s\n", AuthStatusName(status));
    }

    if (status != AVAuthorizationStatusAuthorized) {
      std::printf("FAIL: camera access is %s. Enable it in System Settings > Privacy & Security > Camera.\n",
                  AuthStatusName(status));
      return 2;
    }

    AVCaptureDeviceDiscoverySession* discovery = [AVCaptureDeviceDiscoverySession
        discoverySessionWithDeviceTypes:@[ AVCaptureDeviceTypeBuiltInWideAngleCamera,
                                           AVCaptureDeviceTypeExternal,
                                           AVCaptureDeviceTypeContinuityCamera ]
                              mediaType:AVMediaTypeVideo
                               position:AVCaptureDevicePositionUnspecified];

    AVCaptureDevice* chosen = nil;
    std::printf("Cameras found: %lu\n", static_cast<unsigned long>(discovery.devices.count));
    for (AVCaptureDevice* device in discovery.devices) {
      const bool is_virtual = IsVirtualCamera(device);
      std::printf("  - %s [%s]%s\n", device.localizedName.UTF8String, device.uniqueID.UTF8String,
                  is_virtual ? "  (virtual camera: skipped to avoid feedback loops)" : "");
      if (chosen == nil && !is_virtual) {
        chosen = device;
      }
    }
    if (chosen == nil) {
      std::printf("FAIL: no usable (non-virtual) camera\n");
      return 3;
    }
    std::printf("Using: %s\n", chosen.localizedName.UTF8String);

    NSError* error = nil;
    AVCaptureDeviceInput* input = [AVCaptureDeviceInput deviceInputWithDevice:chosen error:&error];
    if (input == nil) {
      std::printf("FAIL: cannot open camera: %s\n", error.localizedDescription.UTF8String);
      return 4;
    }

    AVCaptureSession* session = [[AVCaptureSession alloc] init];
    [session beginConfiguration];
    if ([session canSetSessionPreset:AVCaptureSessionPreset1280x720]) {
      session.sessionPreset = AVCaptureSessionPreset1280x720;
      std::printf("Requested preset: 1280x720\n");
    } else {
      std::printf("Preset 1280x720 unsupported; using the device default\n");
    }
    if (![session canAddInput:input]) {
      std::printf("FAIL: session rejected the camera input\n");
      return 4;
    }
    [session addInput:input];

    AVCaptureVideoDataOutput* output = [[AVCaptureVideoDataOutput alloc] init];
    output.videoSettings = @{(id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA)};
    output.alwaysDiscardsLateVideoFrames = YES;  // Drop stale frames instead of queueing.
    FrameCounter* counter = [[FrameCounter alloc] init];
    dispatch_queue_t queue = dispatch_queue_create("relume.camera_probe", DISPATCH_QUEUE_SERIAL);
    [output setSampleBufferDelegate:counter queue:queue];
    if (![session canAddOutput:output]) {
      std::printf("FAIL: session rejected the video output\n");
      return 4;
    }
    [session addOutput:output];
    [session commitConfiguration];

    [session startRunning];
    const bool finished = WaitFor(counter->done, kTimeoutSeconds);
    [session stopRunning];
    // Detach the delegate and drain the queue so no callback outlives `counter`.
    [output setSampleBufferDelegate:nil queue:nullptr];
    dispatch_sync(queue, ^{});

    const int frames = counter->frames.load();
    const std::uint32_t format = counter->pixelFormat.load();
    std::printf("Frames received: %d\n", frames);
    if (frames > 0) {
      std::printf("Frame size: %dx%d\n", counter->width.load(), counter->height.load());
      std::printf("Pixel format: '%c%c%c%c'\n", static_cast<char>((format >> 24) & 0xFF),
                  static_cast<char>((format >> 16) & 0xFF), static_cast<char>((format >> 8) & 0xFF),
                  static_cast<char>(format & 0xFF));
      const double span = counter->lastTimestamp.load() - counter->firstTimestamp.load();
      if (frames > 1 && span > 0) {
        std::printf("Measured capture rate: %.1f FPS over %.2f s\n", (frames - 1) / span, span);
      }
    }
    if (!finished) {
      std::printf("FAIL: expected %d frames within %.0f s\n", kFramesToCapture, kTimeoutSeconds);
      return 4;
    }
    std::printf("PASS\n");
    return 0;
  }
}
