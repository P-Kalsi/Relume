#include "relume/test_guide.hpp"

#include <algorithm>

namespace relume {

std::vector<GuideStep> SwitchingTestGuide() {
  return {
      {"Wait for CoreML",
       "Do nothing yet. Wait until the Depth status reads \"Ready (CoreML (CPUAndGPU) ...)\". The video stays "
       "normal (passthrough) while it loads, about 10-20 s.",
       30.0},
      {"Switch to CPU",
       "Inference menu -> \"CPU\". Wait until the status reads \"Ready (CPU, 4 threads)\" (about 1 s). "
       "Watch for any freeze in either window.",
       20.0},
      {"Back to CoreML",
       "Inference menu -> \"CoreML: CPU and GPU (default)\". It shows \"Loading\" for about 16 s, then Ready. "
       "The video must keep running meanwhile.",
       40.0},
      {"Turn depth off",
       "Inference menu -> \"Off\". The status reads \"Off\", the depth view turns black, the video continues.", 20.0},
      {"CoreML on again",
       "Inference menu -> \"CoreML: CPU and GPU (default)\". Wait for Ready (about 16 s).", 40.0},
      {"Switch to the iPhone camera",
       "Camera menu -> your iPhone camera. Wait for live video; depth comes back after a moment. "
       "(No iPhone? Press Skip for this and the next three steps.)",
       30.0},
      {"Back to FaceTime",
       "Camera menu -> \"FaceTime HD Camera\". Live video and depth again. Old depth from the iPhone must never "
       "appear.",
       20.0},
      {"iPhone, then disconnect it",
       "Camera menu -> your iPhone camera. Once it is live, lock the iPhone or move it away until it drops. "
       "Expected: black output (no frozen frame), status \"Camera unavailable\", black depth view.",
       40.0},
      {"FaceTime again",
       "Camera menu -> \"FaceTime HD Camera\". Live video and depth again.", 30.0},
      {"Done", "Test 3 is finished. Close the Relume Controls window to quit, then tell Claude it is done.", 0.0},
  };
}

TestGuide::TestGuide(std::vector<GuideStep> steps, Clock::time_point start)
    : steps_(std::move(steps)), starts_(steps_.size(), 0.0), start_(start) {
  Reschedule(0, 0.0);
}

void TestGuide::Reschedule(std::size_t from, double start_s) {
  double t = start_s;
  for (std::size_t i = from; i < steps_.size(); ++i) {
    starts_[i] = t;
    t += steps_[i].seconds;
  }
}

double TestGuide::elapsed(Clock::time_point now) const {
  return std::chrono::duration<double>(now - start_).count();
}

bool TestGuide::Update(Clock::time_point now) {
  const double t = elapsed(now);
  const std::size_t before = current_;
  while (current_ + 1 < steps_.size() && t >= starts_[current_ + 1]) {
    ++current_;
  }
  return current_ != before;
}

void TestGuide::Skip(Clock::time_point now) {
  if (current_ + 1 < steps_.size()) {
    Reschedule(current_ + 1, std::max(elapsed(now), starts_[current_]));
    ++current_;
  }
}

double TestGuide::remaining(Clock::time_point now) const {
  if (current_ + 1 >= steps_.size()) {
    return 0.0;
  }
  return std::max(0.0, starts_[current_ + 1] - elapsed(now));
}

}  // namespace relume
