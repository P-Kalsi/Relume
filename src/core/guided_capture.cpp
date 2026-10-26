#include "relume/guided_capture.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <system_error>

namespace relume {

std::vector<CaptureStep> DefaultCapturePlan() {
  using Kind = CaptureStep::Kind;
  std::vector<CaptureStep> plan = {
      {Kind::kStill, "still_01", "Frontal",
       "Face the camera at arm's length (about 60 cm), shoulders square, normal room light.", 1},
      {Kind::kStill, "still_02", "Head turned left",
       "Same distance. Turn your head about 45 degrees to your left; keep your shoulders facing the camera.", 1},
      {Kind::kStill, "still_03", "Head turned right",
       "Same distance. Turn your head about 45 degrees to your right; keep your shoulders facing the camera.", 1},
      {Kind::kStill, "still_04", "Head tilted",
       "Tilt your head down slightly (about 20 degrees) while still looking at the camera.", 1},
      {Kind::kStill, "still_05", "Hand near face",
       "Raise one open hand beside your cheek, about 10 cm from your face, palm towards the camera.", 1},
      {Kind::kStill, "still_06", "Hand in front of face",
       "Hold a hand in front of your chin or mouth so it partly covers your face, about 15 cm in front of it.", 1},
      {Kind::kStill, "still_07", "Glasses or accessory",
       "Wear glasses if you have them (or headphones or a hat). Frontal, arm's length.", 1},
      {Kind::kStill, "still_08", "Hair and shoulders",
       "Lean back a little so your hair, both shoulders and some of your upper body are in frame.", 1},
      {Kind::kStill, "still_09", "Backlit",
       "Sit with a bright window or lamp behind you, so your face is darker than the background.", 1},
      {Kind::kStill, "still_10", "Dim light",
       "Turn off or dim the room light so you are lit mainly by the screen.", 1},
      {Kind::kStill, "still_11", "Close",
       "Move close to the camera (about 30 cm) so your face fills most of the frame.", 1},
      {Kind::kStill, "still_12", "Far, busy background",
       "Move back 1.5 to 2 m so your upper body and a cluttered background (shelves, objects) are visible.", 1},
      {Kind::kBurst, "burst_a", "Burst A: still, normal light",
       "Normal room light, frontal, arm's length. Sit as still as you comfortably can; blinking is fine. "
       "Saves 20 consecutive frames (under a second).",
       20},
      {Kind::kBurst, "burst_b", "Burst B: still, harder light",
       "Side light or a window behind you. Sit as still as you comfortably can. Saves 20 consecutive frames.", 20},
  };
  return plan;
}

std::string_view ToString(ShotStatus status) {
  switch (status) {
    case ShotStatus::kNotTaken: return "not taken";
    case ShotStatus::kCountdown: return "countdown";
    case ShotStatus::kCapturing: return "capturing";
    case ShotStatus::kWriting: return "writing";
    case ShotStatus::kSaved: return "saved";
    case ShotStatus::kFailed: return "failed";
  }
  return "?";
}

GuidedCapture::GuidedCapture(FrameSaver& saver, std::vector<CaptureStep> plan) : saver_(saver) {
  // Shots already on disk from an earlier session (a handful of stat calls,
  // once at startup). A retake replaces them.
  for (CaptureStep& step : plan) {
    Shot shot;
    const std::filesystem::path path = std::filesystem::path(saver_.directory()) /
                                       (step.kind == CaptureStep::Kind::kStill ? step.name + ".ppm" : step.name);
    std::error_code ec;
    if (std::filesystem::exists(path, ec)) {
      shot.status = ShotStatus::kSaved;
      shot.message = "On disk from an earlier session; a retake replaces it.";
    }
    shot.step = std::move(step);
    steps_.push_back(std::move(shot));
  }
}

bool GuidedCapture::busy() const {
  const ShotStatus s = steps_[current_].status;
  return s == ShotStatus::kCountdown || s == ShotStatus::kCapturing || s == ShotStatus::kWriting;
}

void GuidedCapture::Select(std::size_t index) {
  if (!busy() && index < steps_.size()) {
    current_ = index;
  }
}

void GuidedCapture::StartCapture(Clock::time_point now) {
  if (busy()) {
    return;
  }
  Shot& shot = steps_[current_];
  shot.status = ShotStatus::kCountdown;
  shot.message.clear();
  shot.final_job = 0;
  countdown_end_ = now + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(kCountdownS));
}

double GuidedCapture::countdown_remaining(Clock::time_point now) const {
  if (steps_[current_].status != ShotStatus::kCountdown) {
    return 0.0;
  }
  return std::max(0.0, std::chrono::duration<double>(countdown_end_ - now).count());
}

void GuidedCapture::Finish(Shot& shot, bool ok, const std::string& message) {
  shot.status = ok ? ShotStatus::kSaved : ShotStatus::kFailed;
  shot.message = message;
  shot.final_job = 0;
}

void GuidedCapture::Update(Clock::time_point now) {
  for (const SaveResult& result : saver_.TakeResults()) {
    // Results of individual burst frames decide nothing: the burst's commit
    // job counts the files that were actually written.
    for (Shot& shot : steps_) {
      if (shot.final_job != 0 && shot.final_job == result.job) {
        if (shot.step.kind == CaptureStep::Kind::kStill) {
          files_written_ += result.ok ? 1 : 0;
          Finish(shot, result.ok, result.ok ? "Saved " + result.path : "Not saved: " + result.error);
        } else {
          files_written_ += result.ok ? shot.step.frames : 0;
          char text[64];
          std::snprintf(text, sizeof(text), "Saved %u frames in ", shot.step.frames);
          Finish(shot, result.ok,
                 result.ok ? text + result.path
                           : "Not saved: " + (shot.message.empty() ? result.error : shot.message));
        }
      }
    }
  }
  Shot& shot = steps_[current_];
  if (shot.status == ShotStatus::kCountdown && now >= countdown_end_) {
    shot.status = ShotStatus::kCapturing;
    burst_frames_ = 0;
    if (shot.step.kind == CaptureStep::Kind::kBurst && saver_.EnqueueRemoveDir(StagingDir(shot)) == 0) {
      Finish(shot, false, "Not saved: the frame saver is not available");
    }
  }
}

void GuidedCapture::OnFrame(const VideoFrame& frame, Clock::time_point /*now*/) {
  Shot& shot = steps_[current_];
  if (shot.status != ShotStatus::kCapturing || frame.frame_id == last_frame_id_) {
    return;
  }
  last_frame_id_ = frame.frame_id;
  if (shot.step.kind == CaptureStep::Kind::kStill) {
    const std::uint64_t job = saver_.Enqueue(frame, shot.step.name + ".ppm");
    if (job != 0) {
      shot.final_job = job;
      shot.status = ShotStatus::kWriting;
    } else if (!saver_.available()) {
      Finish(shot, false, "Not saved: the frame saver is not available");
    }
    // Otherwise both buffers were busy: take the next frame instead.
    return;
  }
  char name[32];
  std::snprintf(name, sizeof(name), "frame_%02u.ppm", burst_frames_ + 1);
  const std::uint64_t job = saver_.Enqueue(frame, StagingDir(shot) + "/" + name);
  if (job == 0) {
    // A refused frame breaks the run of consecutive frames: discard the burst.
    shot.message = "a frame was skipped because the writer was busy; retake the burst";
    shot.final_job = saver_.EnqueueCommitDir(StagingDir(shot), shot.step.name, shot.step.frames);
    shot.status = ShotStatus::kWriting;
    if (shot.final_job == 0) {
      Finish(shot, false, "Not saved: the frame saver is not available");
    }
    return;
  }
  ++burst_frames_;
  if (burst_frames_ == shot.step.frames) {
    shot.final_job = saver_.EnqueueCommitDir(StagingDir(shot), shot.step.name, shot.step.frames);
    shot.status = ShotStatus::kWriting;
    if (shot.final_job == 0) {
      Finish(shot, false, "Not saved: the frame saver is not available");
    }
  }
}

std::uint32_t GuidedCapture::saved_stills() const {
  return static_cast<std::uint32_t>(std::count_if(steps_.begin(), steps_.end(), [](const Shot& s) {
    return s.step.kind == CaptureStep::Kind::kStill && s.status == ShotStatus::kSaved;
  }));
}

std::uint32_t GuidedCapture::saved_bursts() const {
  return static_cast<std::uint32_t>(std::count_if(steps_.begin(), steps_.end(), [](const Shot& s) {
    return s.step.kind == CaptureStep::Kind::kBurst && s.status == ShotStatus::kSaved;
  }));
}

std::uint32_t GuidedCapture::stills_total() const {
  return static_cast<std::uint32_t>(std::count_if(
      steps_.begin(), steps_.end(), [](const Shot& s) { return s.step.kind == CaptureStep::Kind::kStill; }));
}

std::uint32_t GuidedCapture::bursts_total() const {
  return static_cast<std::uint32_t>(steps_.size()) - stills_total();
}

}  // namespace relume
