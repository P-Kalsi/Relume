#pragma once

// Opt-in saving of camera frames as local test fixtures (`--save-frames`).
// Off unless the user asks for it on the command line; nothing is uploaded.
//
// The caller (the render thread) offers every new frame. A frame that is due
// is copied into one of the saver's own buffers, so no pool buffer is held;
// a writer thread converts it to RGB with ToRgb8 (the same conversion as the
// rest of Relume) and writes a lossless binary PPM. The caller never waits on
// the disk: if both buffers are busy, the frame is skipped and counted.

#include "relume/video_frame.hpp"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace relume {

struct FrameSaverOptions {
  std::string directory;     // created if missing
  std::uint32_t count = 0;   // frames to save, 1..FrameSaver::kMaxFrames
  double interval_s = 1.0;   // between saved frames; 0 = consecutive frames
  double delay_s = 5.0;      // after the first offered frame, before the first save
};

class FrameSaver {
 public:
  using Clock = std::chrono::steady_clock;
  static constexpr std::uint32_t kMaxFrames = 60;
  static constexpr std::size_t kBuffers = 2;

  // Returns an empty string if the options are usable, otherwise a reason.
  static std::string Validate(const FrameSaverOptions& options);

  // Creates the directory and starts the writer thread. Check error() after
  // construction: on failure the saver accepts nothing.
  explicit FrameSaver(FrameSaverOptions options);
  ~FrameSaver();  // writes what is queued, then joins the writer
  FrameSaver(const FrameSaver&) = delete;
  FrameSaver& operator=(const FrameSaver&) = delete;

  // Offers a frame seen at `now`. Copies it if a save is due and a buffer is
  // free; never blocks on the writer.
  void Offer(const VideoFrame& frame, Clock::time_point now);

  // Every requested frame has been written or has failed.
  bool done() const;
  std::uint32_t written() const;
  std::uint32_t failed() const;
  std::uint32_t skipped_busy() const;  // due frames dropped because both buffers were busy
  std::string error() const;           // last failure, if any
  std::vector<std::string> paths() const;  // files written so far

 private:
  void WriterLoop();

  const FrameSaverOptions options_;
  std::string prefix_;  // run timestamp, keeps runs apart in one directory

  // Render-thread state.
  Clock::time_point first_offer_{};
  Clock::time_point last_accept_{};
  std::uint64_t last_frame_id_ = 0;
  std::uint32_t accepted_ = 0;

  mutable std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<VideoFrame> pending_;  // copies waiting for the writer
  std::vector<VideoFrame> free_;    // recycled buffers
  std::size_t buffers_out_ = 0;     // buffers pending or being written
  bool stop_ = false;
  std::uint32_t written_ = 0;
  std::uint32_t failed_ = 0;
  std::uint32_t skipped_busy_ = 0;
  std::string error_;
  std::vector<std::string> paths_;

  std::thread writer_;  // declared last: started after, joined before the rest goes
};

}  // namespace relume
