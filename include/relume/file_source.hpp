#pragma once

// Deterministic camera substitute for headless tests and development: emits
// one still image (a P6 PPM, or the built-in orientation pattern) as a stream
// of frames with consecutive IDs and synthetic timestamps. It can encode the
// image as BGRA8, RGB8 or NV12 so every upload path is testable without a
// camera. It never touches the platform camera API.
//
// Its thread is always joined by Stop() or the destructor. (std::jthread is
// not used: Apple's libc++ 18 ships it only as an experimental feature.)

#include "relume/camera.hpp"
#include "relume/frame_queue.hpp"
#include "relume/stats.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

namespace relume {

class CameraEventQueue;

struct FileSourceOptions {
  std::string path;  // P6 PPM; empty means the orientation pattern
  std::uint32_t pattern_width = 1280;
  std::uint32_t pattern_height = 720;
  PixelFormat format = PixelFormat::kBgra8;
  YuvRange range = YuvRange::kVideo;     // for kNv12
  YuvMatrix matrix = YuvMatrix::kBt709;  // for kNv12
  double fps = 30.0;          // also defines the synthetic timestamps
  std::uint64_t max_frames = 0;  // 0 = unlimited
  bool realtime = true;       // false: emit without pacing (tests)
};

class FileSource final : public ICameraSource {
 public:
  static constexpr const char* kDeviceId = "relume.file-source";

  FileSource(FileSourceOptions options, LatestFrameSlot& slot, CameraEventQueue& events);
  ~FileSource() override;
  FileSource(const FileSource&) = delete;
  FileSource& operator=(const FileSource&) = delete;

  void Start() override;
  void Stop() override;
  SourceState state() const override { return state_.load(); }
  CaptureStats stats() const override;
  std::string description() const override;

 private:
  void Run();

  const FileSourceOptions options_;
  LatestFrameSlot& slot_;
  CameraEventQueue& events_;
  FramePool pool_{4};
  VideoFrame source_frame_;  // the encoded image; written only before the thread starts
  std::atomic<SourceState> state_{SourceState::kIdle};
  std::atomic<std::uint64_t> received_{0};
  std::atomic<std::uint64_t> published_{0};
  LatencyHistogram copy_ms_;
  std::mutex wake_mutex_;
  std::condition_variable wake_;
  bool stop_requested_ = false;  // guarded by wake_mutex_
  std::thread thread_;
};

}  // namespace relume
