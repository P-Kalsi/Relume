#pragma once

// Temporal transitions between depth results, as pure logic: which of three
// depth-field slots to write, which two to show, the crossfade weight and the
// effect's fade. The GPU only carries out these decisions (see
// docs/architecture.md, "Relighting"). Time is passed in (seconds on any
// monotonic clock), so tests are deterministic. Portable C++.
//
// Slots and ownership. Each slot is in one role: `from`, `to` or free. Every
// write (a new field, or a snapshot) goes into a slot that is neither the
// current `from` nor `to`; the roles change only after the write has been
// issued. WebGPU runs queue work in submission order, so a pass that reads a
// slot always finishes before a later-submitted write to it.
//
// Rules:
//   - Crossfade: a new result fades in over T, fixed when it starts (factor x
//     the median of the last 8 intervals between results, clamped), so the
//     weight only increases.
//   - Overlap: if a new result arrives before the weight reaches 1, the field
//     shown at that instant is snapshotted (mix(from, to, t) into the free
//     slot) and becomes the new `from`: nothing snaps backward.
//   - Hard cut (a scene change, or crossfading disabled): from = to = new, no
//     blend with incompatible history.
//   - Reset (another camera, size, transform or worker; no matching depth):
//     no field at all, the effect is off at once (passthrough); the next
//     result fades the effect in over fade_in_s.
//   - Stale depth (older than stale_s): the effect fades out over
//     fade_out_s; the next fresh result fades it in again.

#include <array>
#include <cstddef>
#include <cstdint>

namespace relume {

struct CrossfadeConfig {
  bool enabled = true;          // false: every result is a hard cut
  double duration_factor = 0.5;  // T = factor x median interval
  double min_duration_s = 0.06;
  double max_duration_s = 0.2;
  double default_interval_s = 1.0 / 7.0;  // before two results have arrived
  double fade_in_s = 0.25;
  double fade_out_s = 0.25;
  double stale_s = 1.0;
};

class CrossfadeState {
 public:
  static constexpr int kSlots = 3;
  static constexpr int kNone = -1;

  // GPU work for a new result, in this order: an optional snapshot, then the
  // new field into `build_slot`.
  struct Update {
    bool hard_cut = false;
    int snapshot_slot = kNone;  // write mix(snapshot_from, snapshot_to, snapshot_t) here first
    int snapshot_from = kNone;
    int snapshot_to = kNone;
    float snapshot_t = 0.0f;
    int build_slot = kNone;     // then build the new field here
  };

  // What to draw at a given time.
  struct View {
    bool active = false;  // false: passthrough (no field)
    int from = kNone;
    int to = kNone;
    float blend = 1.0f;   // from -> to
    float fade = 0.0f;    // effect weight
    bool animating = false;  // the image changes over time even without a new frame
  };

  explicit CrossfadeState(CrossfadeConfig config = {}) : config_(config) {}

  const CrossfadeConfig& config() const { return config_; }
  // Takes effect from the next result; changing `enabled` resets.
  void SetConfig(const CrossfadeConfig& config);

  // A new depth result at `now`. `hard_cut`: don't blend it with history.
  Update OnResult(double now, bool hard_cut);
  // No usable depth (incompatible or none): passthrough at once.
  void Reset();
  // Advances the fade to `now` (call once per frame, in time order).
  // `depth_age_s` is the age of the newest field's source frame.
  View Advance(double now, double depth_age_s);

  // Diagnostics.
  double duration_s() const { return duration_; }
  std::uint64_t snapshots() const { return snapshots_; }
  std::uint64_t hard_cuts() const { return hard_cuts_; }
  std::uint64_t resets() const { return resets_; }
  bool stale() const { return stale_; }

 private:
  int FreeSlot() const;
  float Weight(double now) const;
  double MedianInterval() const;

  CrossfadeConfig config_;
  bool has_field_ = false;
  int from_ = kNone;
  int to_ = kNone;
  double start_ = 0.0;     // of the current crossfade
  double duration_ = 0.0;  // T of the current crossfade
  float fade_ = 0.0f;
  double last_advance_ = 0.0;
  bool advanced_ = false;
  bool stale_ = false;
  std::array<double, 8> intervals_{};
  std::size_t interval_count_ = 0;
  double last_result_ = 0.0;
  bool has_last_result_ = false;
  std::uint64_t snapshots_ = 0;
  std::uint64_t hard_cuts_ = 0;
  std::uint64_t resets_ = 0;
};

}  // namespace relume
