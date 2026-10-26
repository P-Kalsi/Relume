#pragma once

// Guided capture (`--guided-capture`): a fixed list of local test shots
// (12 stills and 2 bursts of consecutive frames) that the user takes one at
// a time from the Controls window. The user positions themselves, clicks
// Capture, a short countdown runs, and the next camera frame (or the next 20
// consecutive frames) is saved through a manual-mode FrameSaver.
//
// Main-thread logic only; all disk work happens on the FrameSaver's writer.
// A shot counts as saved only after the writer reports success:
//   still  -> written to a temporary file, then renamed to still_NN.ppm. A
//             retake replaces the file; a failed retake removes it.
//   burst  -> written into burst_x.partial/, then the whole folder replaces
//             burst_x/ only if every frame arrived. A frame refused because
//             the writer was busy breaks the burst; nothing is kept.

#include "relume/frame_saver.hpp"
#include "relume/video_frame.hpp"

#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace relume {

struct CaptureStep {
  enum class Kind : std::uint8_t { kStill, kBurst };
  Kind kind = Kind::kStill;
  std::string name;         // still_01 or burst_a: the file or folder name
  std::string title;        // short label
  std::string instruction;  // what to do for this shot
  std::uint32_t frames = 1;
};

// The 12 stills and 2 bursts used for the letterbox/stretch comparison.
std::vector<CaptureStep> DefaultCapturePlan();

enum class ShotStatus : std::uint8_t {
  kNotTaken,
  kCountdown,  // Capture was clicked; waiting for the countdown to end
  kCapturing,  // taking the next frame(s) from the camera
  kWriting,    // frames handed to the writer; waiting for its result
  kSaved,      // the writer confirmed the file(s)
  kFailed,     // nothing was kept for this shot
};

class GuidedCapture {
 public:
  using Clock = FrameSaver::Clock;
  static constexpr double kCountdownS = 3.0;

  // `saver` must be in manual mode and outlive this object.
  GuidedCapture(FrameSaver& saver, std::vector<CaptureStep> plan);

  std::size_t size() const { return steps_.size(); }
  std::size_t current() const { return current_; }
  const CaptureStep& step(std::size_t i) const { return steps_[i].step; }
  ShotStatus status(std::size_t i) const { return steps_[i].status; }
  const std::string& message(std::size_t i) const { return steps_[i].message; }
  // A shot is in progress (countdown, capturing or writing).
  bool busy() const;

  // Moves to another shot. Ignored while busy.
  void Select(std::size_t index);
  // Capture (or retake) the current shot: starts the countdown. Ignored while busy.
  void StartCapture(Clock::time_point now);
  // Seconds left in the countdown (0 when not counting down).
  double countdown_remaining(Clock::time_point now) const;
  // Frames handed to the writer for the current burst so far.
  std::uint32_t burst_progress() const { return burst_frames_; }

  // Call once per render-loop iteration: collects writer results and ends
  // the countdown.
  void Update(Clock::time_point now);
  // Call with every new camera frame. Copies at most one frame; never waits.
  void OnFrame(const VideoFrame& frame, Clock::time_point now);

  std::uint32_t saved_stills() const;
  std::uint32_t saved_bursts() const;
  std::uint32_t stills_total() const;
  std::uint32_t bursts_total() const;
  std::uint32_t files_written() const { return files_written_; }  // this session, confirmed by the writer
  const std::string& directory() const { return saver_.directory(); }

 private:
  struct Shot {
    CaptureStep step;
    ShotStatus status = ShotStatus::kNotTaken;
    std::string message;
    std::uint64_t final_job = 0;  // the job whose result decides the shot
  };

  void Finish(Shot& shot, bool ok, const std::string& message);
  std::string StagingDir(const Shot& shot) const { return shot.step.name + ".partial"; }

  FrameSaver& saver_;
  std::vector<Shot> steps_;
  std::size_t current_ = 0;
  Clock::time_point countdown_end_{};
  std::uint64_t last_frame_id_ = 0;
  std::uint32_t burst_frames_ = 0;
  std::uint32_t files_written_ = 0;
};

std::string_view ToString(ShotStatus status);

}  // namespace relume
