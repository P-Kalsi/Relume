#include "relume/file_source.hpp"

#include "relume/event_queue.hpp"
#include "relume/image_io.hpp"
#include "relume/pixel_convert.hpp"

#include <chrono>
#include <cstdio>
#include <utility>

namespace relume {

FileSource::FileSource(FileSourceOptions options, LatestFrameSlot& slot, CameraEventQueue& events)
    : options_(std::move(options)), slot_(slot), events_(events) {}

FileSource::~FileSource() { Stop(); }

void FileSource::Start() {
  if (thread_.joinable()) {
    if (state_.load() != SourceState::kIdle) {
      return;  // already running
    }
    thread_.join();  // a finite run ended on its own
  }
  state_ = SourceState::kStarting;
  RgbImage image;
  if (options_.path.empty()) {
    image = MakeOrientationPattern(options_.pattern_width, options_.pattern_height);
  } else {
    std::string error;
    std::optional<RgbImage> loaded = ReadPpm(options_.path, error);
    if (!loaded) {
      state_ = SourceState::kFailed;
      events_.Post({CameraEvent::Kind::kSourceFailed, kDeviceId, error});
      return;
    }
    image = std::move(*loaded);
  }
  if (image.width == 0 || image.height == 0 || options_.fps <= 0.0) {
    state_ = SourceState::kFailed;
    events_.Post({CameraEvent::Kind::kSourceFailed, kDeviceId, "invalid file source options"});
    return;
  }
  EncodeRgb8(image.pixels.data(), image.width, image.height, options_.format, options_.range, options_.matrix,
             source_frame_);
  source_frame_.chroma_location = ChromaLocation::kLeft;
  source_frame_.clock = TimestampClock::kSynthetic;
  {
    std::lock_guard lock(wake_mutex_);
    stop_requested_ = false;
  }
  thread_ = std::thread([this] { Run(); });
}

void FileSource::Stop() {
  if (thread_.joinable()) {
    {
      std::lock_guard lock(wake_mutex_);
      stop_requested_ = true;
    }
    wake_.notify_all();
    thread_.join();
  }
  if (state_.load() != SourceState::kFailed) {
    state_ = SourceState::kIdle;
  }
}

void FileSource::Run() {
  using Clock = std::chrono::steady_clock;
  const auto period = std::chrono::duration<double>(1.0 / options_.fps);
  const Clock::time_point start = Clock::now();
  auto stopping = [this] {
    std::lock_guard lock(wake_mutex_);
    return stop_requested_;
  };
  for (std::uint64_t index = 0; !stopping(); ++index) {
    if (options_.max_frames != 0 && index >= options_.max_frames) {
      state_ = SourceState::kIdle;
      events_.Post({CameraEvent::Kind::kSourceEnded, kDeviceId, "file source finished"});
      return;
    }
    if (options_.realtime) {
      const auto due = start + std::chrono::duration_cast<Clock::duration>(period * static_cast<double>(index));
      std::unique_lock lock(wake_mutex_);
      if (wake_.wait_until(lock, due, [this] { return stop_requested_; })) {
        return;  // woken early by Stop()
      }
    }
    received_.fetch_add(1);
    std::shared_ptr<VideoFrame> frame = pool_.Acquire();
    if (!frame) {
      continue;  // counted by the pool
    }
    const auto copy_start = Clock::now();
    const std::vector<std::byte>& data = source_frame_.data;
    // Copy metadata, then data into the recycled buffer (reusing its capacity).
    std::vector<std::byte> storage = std::move(frame->data);
    *frame = VideoFrame{};
    frame->width = source_frame_.width;
    frame->height = source_frame_.height;
    frame->format = source_frame_.format;
    frame->range = source_frame_.range;
    frame->matrix = source_frame_.matrix;
    frame->chroma_location = source_frame_.chroma_location;
    frame->planes = source_frame_.planes;
    frame->plane_count = source_frame_.plane_count;
    frame->clock = TimestampClock::kSynthetic;
    storage.assign(data.begin(), data.end());
    frame->data = std::move(storage);
    frame->frame_id = index + 1;
    frame->capture_time_ns = static_cast<std::int64_t>(static_cast<double>(index) * 1e9 / options_.fps);
    frame->arrival_time = Clock::now();
    copy_ms_.Record(std::chrono::duration<double, std::milli>(frame->arrival_time - copy_start).count());
    slot_.Publish(std::move(frame));
    published_.fetch_add(1);
    if (index == 0) {
      state_ = SourceState::kRunning;
      events_.Post({CameraEvent::Kind::kSourceStarted, kDeviceId, description()});
    }
  }
}

CaptureStats FileSource::stats() const {
  CaptureStats stats;
  stats.frames_received = received_.load();
  stats.frames_published = published_.load();
  stats.dropped_pool = pool_.exhausted_count();
  stats.copy_ms_p50 = copy_ms_.Percentile(0.5);
  stats.copy_ms_p95 = copy_ms_.Percentile(0.95);
  return stats;
}

std::string FileSource::description() const {
  char text[160];
  std::snprintf(text, sizeof(text), "File source: %ux%u %.*s @ %.1f fps (%s)", source_frame_.width,
                source_frame_.height, static_cast<int>(ToString(options_.format).size()),
                ToString(options_.format).data(), options_.fps,
                options_.path.empty() ? "orientation pattern" : "PPM");
  return text;
}

}  // namespace relume
