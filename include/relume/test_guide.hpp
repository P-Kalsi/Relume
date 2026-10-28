#pragma once

// On-screen guide for a manual test (`--test-guide switching`): a fixed list
// of timed steps shown in the Controls window. Off by default. It only shows
// instructions and logs step changes; the person does every action
// themselves, so it never changes what the app does.
//
// Steps follow the clock: each has a duration, and the guide moves to the
// next one when the time is up. Skip() moves on early; the later steps keep
// their durations, so the schedule shifts earlier.

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

namespace relume {

struct GuideStep {
  std::string title;
  std::string instruction;
  double seconds = 0.0;  // time slot; 0 for the last step (open-ended)
};

// Test 3 of the Phase 3 camera checks: provider and camera switching.
std::vector<GuideStep> SwitchingTestGuide();

class TestGuide {
 public:
  using Clock = std::chrono::steady_clock;

  TestGuide(std::vector<GuideStep> steps, Clock::time_point start);

  // Moves to the step the clock has reached. Returns true if the step changed.
  bool Update(Clock::time_point now);
  // Ends the current step now.
  void Skip(Clock::time_point now);

  std::size_t size() const { return steps_.size(); }
  std::size_t current() const { return current_; }
  const GuideStep& step(std::size_t i) const { return steps_[i]; }
  bool finished() const { return current_ + 1 == steps_.size(); }  // on the last, open-ended step

  // Seconds since the guide started, and the scheduled start of step `i`.
  double elapsed(Clock::time_point now) const;
  double step_start(std::size_t i) const { return starts_[i]; }
  // Seconds left in the current step (0 on the last step).
  double remaining(Clock::time_point now) const;

 private:
  void Reschedule(std::size_t from, double start_s);

  std::vector<GuideStep> steps_;
  std::vector<double> starts_;  // seconds since start_
  Clock::time_point start_;
  std::size_t current_ = 0;
};

}  // namespace relume
