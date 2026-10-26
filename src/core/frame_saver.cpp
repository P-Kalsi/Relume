#include "relume/frame_saver.hpp"

#include "relume/image_io.hpp"
#include "relume/pixel_convert.hpp"

#include <cstdio>
#include <ctime>
#include <filesystem>
#include <system_error>

namespace relume {

std::string FrameSaver::Validate(const FrameSaverOptions& options) {
  if (options.directory.empty()) {
    return "no directory for saved frames";
  }
  if (options.count < 1 || options.count > kMaxFrames) {
    return "the number of frames to save must be 1.." + std::to_string(kMaxFrames);
  }
  if (!(options.interval_s >= 0.0) || !(options.delay_s >= 0.0)) {
    return "the save interval and delay must be 0 or more seconds";
  }
  return {};
}

FrameSaver::FrameSaver(FrameSaverOptions options) : options_(std::move(options)) {
  error_ = Validate(options_);
  if (error_.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(options_.directory, ec);
    if (ec || !std::filesystem::is_directory(options_.directory)) {
      error_ = "cannot create " + options_.directory + ": " + ec.message();
    }
  }
  if (!error_.empty()) {
    failed_ = options_.count;  // nothing will be accepted; done() is true
    accepted_ = options_.count;
    return;
  }
  char stamp[32] = {};
  const std::time_t now = std::time(nullptr);
  if (const std::tm* local = std::localtime(&now)) {  // main thread only, once
    std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", local);
  }
  prefix_ = stamp;
  writer_ = std::thread([this] { WriterLoop(); });
}

FrameSaver::~FrameSaver() {
  {
    std::lock_guard lock(mutex_);
    stop_ = true;
  }
  wake_.notify_all();
  if (writer_.joinable()) {
    writer_.join();
  }
}

void FrameSaver::Offer(const VideoFrame& frame, Clock::time_point now) {
  if (accepted_ >= options_.count || frame.frame_id == last_frame_id_) {
    return;
  }
  if (first_offer_ == Clock::time_point{}) {
    first_offer_ = now;
  }
  const auto since = [&](Clock::time_point t) { return std::chrono::duration<double>(now - t).count(); };
  if (since(first_offer_) < options_.delay_s) {
    return;
  }
  if (accepted_ > 0 && since(last_accept_) < options_.interval_s) {
    return;
  }
  last_frame_id_ = frame.frame_id;
  {
    std::lock_guard lock(mutex_);
    if (buffers_out_ >= kBuffers) {
      ++skipped_busy_;
      return;
    }
    VideoFrame buffer;
    if (!free_.empty()) {
      buffer = std::move(free_.back());
      free_.pop_back();
    }
    buffer = frame;  // copy assignment reuses the buffer's capacity
    pending_.push_back(std::move(buffer));
    ++buffers_out_;
  }
  wake_.notify_one();
  last_accept_ = now;
  ++accepted_;
}

void FrameSaver::WriterLoop() {
  RgbImage image;
  std::uint32_t sequence = 0;
  for (;;) {
    VideoFrame frame;
    {
      std::unique_lock lock(mutex_);
      wake_.wait(lock, [&] { return stop_ || !pending_.empty(); });
      if (pending_.empty()) {
        return;  // stopping, and everything queued is written
      }
      frame = std::move(pending_.front());
      pending_.pop_front();
    }
    ++sequence;
    char name[96];
    std::snprintf(name, sizeof(name), "%s_%02u_f%llu.ppm", prefix_.c_str(), sequence,
                  static_cast<unsigned long long>(frame.frame_id));
    const std::string path = (std::filesystem::path(options_.directory) / name).string();
    std::string failure;
    if (!ToRgb8(frame, image.pixels)) {
      failure = "frame " + std::to_string(frame.frame_id) + " has an invalid layout";
    } else {
      image.width = frame.width;
      image.height = frame.height;
      if (!WritePpm(path, image)) {
        failure = "could not write " + path;
      }
    }
    std::lock_guard lock(mutex_);
    if (failure.empty()) {
      ++written_;
      paths_.push_back(path);
    } else {
      ++failed_;
      error_ = failure;
    }
    free_.push_back(std::move(frame));
    --buffers_out_;
  }
}

bool FrameSaver::done() const {
  std::lock_guard lock(mutex_);
  return written_ + failed_ >= options_.count;
}

std::uint32_t FrameSaver::written() const {
  std::lock_guard lock(mutex_);
  return written_;
}

std::uint32_t FrameSaver::failed() const {
  std::lock_guard lock(mutex_);
  return failed_;
}

std::uint32_t FrameSaver::skipped_busy() const {
  std::lock_guard lock(mutex_);
  return skipped_busy_;
}

std::string FrameSaver::error() const {
  std::lock_guard lock(mutex_);
  return error_;
}

std::vector<std::string> FrameSaver::paths() const {
  std::lock_guard lock(mutex_);
  return paths_;
}

}  // namespace relume
