#include "relume/crossfade.hpp"

#include <algorithm>

namespace relume {

void CrossfadeState::SetConfig(const CrossfadeConfig& config) {
  const bool enabled_changed = config.enabled != config_.enabled;
  config_ = config;
  if (enabled_changed) {
    Reset();
  }
}

int CrossfadeState::FreeSlot() const {
  for (int slot = 0; slot < kSlots; ++slot) {
    if (slot != from_ && slot != to_) {
      return slot;
    }
  }
  return kNone;  // unreachable: two roles never cover three slots
}

float CrossfadeState::Weight(double now) const {
  if (duration_ <= 0.0) {
    return 1.0f;
  }
  return static_cast<float>(std::clamp((now - start_) / duration_, 0.0, 1.0));
}

double CrossfadeState::MedianInterval() const {
  const std::size_t n = std::min(interval_count_, intervals_.size());
  if (n == 0) {
    return config_.default_interval_s;
  }
  std::array<double, 8> sorted = intervals_;
  std::sort(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(n));
  return n % 2 == 1 ? sorted[n / 2] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
}

CrossfadeState::Update CrossfadeState::OnResult(double now, bool hard_cut) {
  if (has_last_result_) {
    intervals_[interval_count_ % intervals_.size()] = std::max(0.0, now - last_result_);
    ++interval_count_;
  }
  last_result_ = now;
  has_last_result_ = true;

  Update update;
  if (hard_cut || !config_.enabled || !has_field_) {
    if (has_field_) {
      ++hard_cuts_;
    }
    update.hard_cut = true;
    update.build_slot = FreeSlot();
    from_ = to_ = update.build_slot;
    start_ = now;
    duration_ = 0.0;
    has_field_ = true;
    return update;
  }

  const float t = Weight(now);
  if (t < 1.0f) {
    // Freeze what is shown now; it becomes the start of the new crossfade.
    update.snapshot_slot = FreeSlot();
    update.snapshot_from = from_;
    update.snapshot_to = to_;
    update.snapshot_t = t;
    from_ = update.snapshot_slot;
    ++snapshots_;
  } else {
    // `to` is fully shown: it becomes the start. The free slot is the third.
    const int shown = to_;
    update.build_slot = FreeSlot();
    from_ = shown;
    to_ = update.build_slot;
  }
  if (update.build_slot == kNone) {
    // After a snapshot the roles are (snapshot, old to); build into the third.
    update.build_slot = FreeSlot();
    to_ = update.build_slot;
  }
  start_ = now;
  duration_ = std::clamp(config_.duration_factor * MedianInterval(), config_.min_duration_s, config_.max_duration_s);
  return update;
}

void CrossfadeState::Reset() {
  if (has_field_) {
    ++resets_;
  }
  has_field_ = false;
  from_ = to_ = kNone;
  fade_ = 0.0f;
  stale_ = false;
  duration_ = 0.0;
  interval_count_ = 0;
  has_last_result_ = false;
}

CrossfadeState::View CrossfadeState::Advance(double now, double depth_age_s) {
  const double dt = advanced_ ? std::max(0.0, now - last_advance_) : 0.0;
  last_advance_ = now;
  advanced_ = true;
  View view;
  if (!has_field_) {
    fade_ = 0.0f;
    return view;
  }
  stale_ = depth_age_s > config_.stale_s;
  const float target = stale_ ? 0.0f : 1.0f;
  if (fade_ < target) {
    fade_ = std::min(target, fade_ + static_cast<float>(dt / std::max(config_.fade_in_s, 1e-6)));
  } else if (fade_ > target) {
    fade_ = std::max(target, fade_ - static_cast<float>(dt / std::max(config_.fade_out_s, 1e-6)));
  }
  view.active = true;
  view.from = from_;
  view.to = to_;
  view.blend = Weight(now);
  view.fade = fade_;
  view.animating = view.blend < 1.0f || fade_ != target;
  return view;
}

}  // namespace relume
