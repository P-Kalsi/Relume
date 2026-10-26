#pragma once

// Opt-in saving of camera frames as local test fixtures (`--save-frames`,
// `--guided-capture`). Off unless the user asks for it on the command line;
// nothing is uploaded.
//
// The caller (the render thread) hands over frames. A frame to save is copied
// into one of the saver's own buffers, so no pool buffer is held; a writer
// thread converts it to RGB with ToRgb8 (the same conversion as the rest of
// Relume) and writes a lossless binary PPM. The caller never waits on the
// disk: if both buffers are busy, the frame is refused and counted.
//
// Two modes:
//   automatic (options.count > 0): Offer() saves `count` frames on a timer.
//   manual    (options.count == 0): Enqueue() saves exactly the frames given,
//             under the names given (used by GuidedCapture).
// Every write goes to a temporary file that is then renamed, so a file is
// either complete or absent; a failed write also removes the destination, so
// a failed retake never leaves the previous shot looking current.

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
  std::uint32_t count = 0;   // automatic mode: frames to save, 1..FrameSaver::kMaxFrames; 0 = manual mode
  double interval_s = 1.0;   // automatic mode: between saved frames; 0 = consecutive frames
  double delay_s = 5.0;      // automatic mode: after the first offered frame, before the first save
};

// The outcome of one queued job, in queue order.
struct SaveResult {
  std::uint64_t job = 0;
  bool ok = false;
  std::string path;   // the file or directory written
  std::string error;  // why it failed
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
  ~FrameSaver();  // finishes what is queued, then joins the writer
  FrameSaver(const FrameSaver&) = delete;
  FrameSaver& operator=(const FrameSaver&) = delete;

  // Automatic mode: offers a frame seen at `now`. Copies it if a save is due
  // and a buffer is free; never blocks on the writer.
  void Offer(const VideoFrame& frame, Clock::time_point now);

  // Manual mode: copies `frame` and queues it to be written to
  // `relative_path` (inside the directory; parent folders are created).
  // Returns the job number, or 0 if both buffers are busy or the saver failed.
  std::uint64_t Enqueue(const VideoFrame& frame, const std::string& relative_path);
  // Manual mode: queues the removal of folder `relative_dir` (if it exists).
  std::uint64_t EnqueueRemoveDir(const std::string& relative_dir);
  // Manual mode: after the jobs queued before it, replaces folder `final_dir`
  // with `staging_dir` if the staging folder holds exactly `expected_files`
  // files; otherwise removes the staging folder and fails. `final_dir` is
  // never left half written.
  std::uint64_t EnqueueCommitDir(const std::string& staging_dir, const std::string& final_dir,
                                 std::uint32_t expected_files);
  // Results finished since the last call.
  std::vector<SaveResult> TakeResults();

  bool manual() const { return options_.count == 0; }
  bool available() const { return !broken_; }  // false if construction failed
  const std::string& directory() const { return options_.directory; }
  // Automatic mode: every requested frame has been written or has failed.
  bool done() const;
  std::uint32_t written() const;       // frame files written successfully
  std::uint32_t failed() const;        // frame writes that failed
  std::uint32_t skipped_busy() const;  // frames refused because both buffers were busy
  std::string error() const;           // last failure, if any
  std::vector<std::string> paths() const;  // automatic mode: files written so far

 private:
  struct Job {
    enum class Kind : std::uint8_t { kWrite, kRemoveDir, kCommitDir };
    Kind kind = Kind::kWrite;
    std::uint64_t id = 0;
    std::string path;     // kWrite: destination; kRemoveDir: folder; kCommitDir: staging folder
    std::string target;   // kCommitDir: final folder
    std::uint32_t expected = 0;
    VideoFrame frame;     // kWrite only
  };

  std::uint64_t Queue(Job job);  // lock held by the caller
  void WriterLoop();
  SaveResult Execute(Job& job, std::vector<std::uint8_t>& rgb);

  const FrameSaverOptions options_;
  std::string prefix_;  // automatic mode: run timestamp, keeps runs apart in one directory

  // Render-thread state (automatic mode).
  Clock::time_point first_offer_{};
  Clock::time_point last_accept_{};
  std::uint64_t last_frame_id_ = 0;
  std::uint32_t accepted_ = 0;
  std::uint32_t sequence_ = 0;

  mutable std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<Job> pending_;
  std::vector<VideoFrame> free_;  // recycled frame buffers
  std::size_t buffers_out_ = 0;   // frame buffers pending or being written
  std::uint64_t next_job_ = 1;
  bool stop_ = false;
  bool broken_ = false;           // construction failed (set before the writer starts, then constant)
  std::uint32_t written_ = 0;
  std::uint32_t failed_ = 0;
  std::uint32_t skipped_busy_ = 0;
  std::string error_;
  std::vector<std::string> paths_;
  std::vector<SaveResult> results_;

  std::thread writer_;  // declared last: started after, joined before the rest goes
};

}  // namespace relume
