// Crossfade logic (CrossfadeState) with injected time. Criteria fixed in
// advance (Phase 4 plan, clarification 1):
//   - a new result never makes the shown field jump: at the arrival instant
//     the shown value is unchanged within 1e-6 (overlap at t = 0.3 and 0.95,
//     bursts of 3 results within one crossfade, regular updates);
//   - within a crossfade the weight only increases and reaches 1 after T;
//   - T stays within its clamp for irregular intervals (60/300/90 ms);
//   - a 500 ms gap holds the newest field; a 1.5 s gap fades the effect out
//     (stale) and the next result fades it in again;
//   - hard cuts and resets never blend with history; resets go to
//     passthrough at once;
//   - over 1,000 random event sequences, a write never targets the current
//     `from` or `to` slot, snapshots read exactly the current roles, and the
//     weights stay within [0, 1].
//
// Each slot holds a scalar here (the GPU holds a field); "shown" is
// mix(value[from], value[to], blend), like the relight shader.

#include "relume/crossfade.hpp"

#include <gtest/gtest.h>

#include <random>

namespace relume {
namespace {

class Model {
 public:
  explicit Model(CrossfadeConfig config = {}) : state(config) {}

  // Applies a result's GPU work to the scalar slots, checking the ownership
  // rule at the moment of each write.
  CrossfadeState::Update Result(double now, float value, bool hard_cut = false) {
    const CrossfadeState::View before = state.Advance(now, 0.0);
    const CrossfadeState::Update u = state.OnResult(now, hard_cut);
    int from = before.active ? before.from : CrossfadeState::kNone;
    int to = before.active ? before.to : CrossfadeState::kNone;
    if (u.snapshot_slot != CrossfadeState::kNone) {
      EXPECT_FALSE(u.hard_cut);
      EXPECT_NE(u.snapshot_slot, from);
      EXPECT_NE(u.snapshot_slot, to);
      EXPECT_EQ(u.snapshot_from, from);
      EXPECT_EQ(u.snapshot_to, to);
      slots[Index(u.snapshot_slot)] = slots[Index(from)] + (slots[Index(to)] - slots[Index(from)]) * u.snapshot_t;
      from = u.snapshot_slot;  // the roles after the snapshot
    }
    EXPECT_GE(u.build_slot, 0);
    EXPECT_LT(u.build_slot, CrossfadeState::kSlots);
    EXPECT_NE(u.build_slot, from) << "write into the slot being shown as `from`";
    EXPECT_NE(u.build_slot, to) << "write into the slot being shown as `to`";
    slots[Index(u.build_slot)] = value;
    return u;
  }

  float Shown(double now, double age = 0.0) {
    const CrossfadeState::View v = state.Advance(now, age);
    last = v;
    if (!v.active) {
      return 0.0f;
    }
    EXPECT_GE(v.blend, 0.0f);
    EXPECT_LE(v.blend, 1.0f);
    EXPECT_GE(v.fade, 0.0f);
    EXPECT_LE(v.fade, 1.0f);
    return slots[Index(v.from)] + (slots[Index(v.to)] - slots[Index(v.from)]) * v.blend;
  }

  static std::size_t Index(int slot) { return static_cast<std::size_t>(slot); }

  CrossfadeState state;
  std::array<float, CrossfadeState::kSlots> slots{};
  CrossfadeState::View last;
};

TEST(Crossfade, FirstResultFadesInFromPassthrough) {
  Model m;
  EXPECT_FALSE(m.state.Advance(0.0, 0.0).active);
  const CrossfadeState::Update u = m.Result(1.0, 5.0f);
  EXPECT_TRUE(u.hard_cut);
  EXPECT_FLOAT_EQ(m.Shown(1.0), 5.0f);
  EXPECT_FLOAT_EQ(m.last.fade, 0.0f);
  m.Shown(1.125);
  EXPECT_NEAR(m.last.fade, 0.5f, 1e-6f);
  m.Shown(1.25);
  EXPECT_FLOAT_EQ(m.last.fade, 1.0f);
  EXPECT_FALSE(m.last.animating);
}

TEST(Crossfade, RegularUpdatesAreContinuousAndMonotonic) {
  Model m;
  const double interval = 1.0 / 7.0;
  double now = 0.0;
  m.Result(now, 0.0f);
  for (int i = 1; i <= 20; ++i) {
    now += interval;
    const float before = m.Shown(now);
    m.Result(now, static_cast<float>(i));
    EXPECT_NEAR(m.Shown(now), before, 1e-6f) << "result " << i;
    // Expected T after the first interval: 0.5 x 1/7 s = 71 ms.
    if (i >= 2) {
      EXPECT_NEAR(m.state.duration_s(), 0.5 * interval, 1e-9);
    }
    float previous = -1.0f;
    for (double dt = 0.0; dt < interval - 1e-9; dt += 0.01) {
      m.Shown(now + dt);
      EXPECT_GE(m.last.blend, previous);
      previous = m.last.blend;
    }
    m.Shown(now + m.state.duration_s());
    EXPECT_FLOAT_EQ(m.last.blend, 1.0f);
  }
}

void ExpectContinuousOverlap(float t_fraction) {
  Model m;
  m.Result(0.0, 0.0f);
  m.Result(1.0 / 7.0, 1.0f);  // a crossfade 0 -> 1 starts
  const double start = 1.0 / 7.0;
  const double arrive = start + t_fraction * m.state.duration_s();
  const float before = m.Shown(arrive);
  EXPECT_NEAR(before, t_fraction, 1e-6f);
  const CrossfadeState::Update u = m.Result(arrive, 10.0f);
  EXPECT_NE(u.snapshot_slot, CrossfadeState::kNone) << "an unfinished crossfade needs a snapshot";
  EXPECT_NEAR(m.Shown(arrive), before, 1e-6f) << "no jump at the new result";
  m.Shown(arrive + m.state.duration_s());
  EXPECT_FLOAT_EQ(m.Shown(arrive + m.state.duration_s()), 10.0f);
}

TEST(Crossfade, OverlapAt30PercentIsContinuous) { ExpectContinuousOverlap(0.3f); }
TEST(Crossfade, OverlapAt95PercentIsContinuous) { ExpectContinuousOverlap(0.95f); }

TEST(Crossfade, BurstOfThreeWithinOneCrossfade) {
  Model m;
  m.Result(0.0, 0.0f);
  m.Result(0.2, 1.0f);
  const double t0 = 0.2;
  for (int i = 0; i < 3; ++i) {
    const double now = t0 + 0.01 * (i + 1);
    const float before = m.Shown(now);
    m.Result(now, 2.0f + static_cast<float>(i));
    EXPECT_NEAR(m.Shown(now), before, 1e-6f) << "burst result " << i;
  }
  EXPECT_EQ(m.state.snapshots(), 3u);
  EXPECT_FLOAT_EQ(m.Shown(1.0), 4.0f);
}

TEST(Crossfade, IrregularIntervalsKeepTheDurationClamped) {
  Model m;
  double now = 0.0;
  m.Result(now, 0.0f);
  const double gaps[] = {0.06, 0.3, 0.09, 0.06, 0.3, 0.09, 0.5, 0.02};
  for (const double gap : gaps) {
    now += gap;
    const float before = m.Shown(now);
    m.Result(now, static_cast<float>(now));
    EXPECT_NEAR(m.Shown(now), before, 1e-6f);
    EXPECT_GE(m.state.duration_s(), 0.06);
    EXPECT_LE(m.state.duration_s(), 0.2);
  }
}

TEST(Crossfade, DelayedResultsHoldThenGoStale) {
  Model m;
  m.Result(0.0, 1.0f);
  m.Shown(0.3);
  ASSERT_FLOAT_EQ(m.last.fade, 1.0f);
  // 500 ms without a result: the newest field stays, fully on.
  EXPECT_FLOAT_EQ(m.Shown(0.5, 0.5), 1.0f);
  EXPECT_FLOAT_EQ(m.last.fade, 1.0f);
  // Past 1 s of depth age: fades out over 0.25 s (advanced every frame, as
  // the app does).
  m.Shown(1.0, 1.0);
  EXPECT_FALSE(m.state.stale());
  EXPECT_FLOAT_EQ(m.last.fade, 1.0f);
  m.Shown(1.125, 1.125);
  EXPECT_TRUE(m.state.stale());
  EXPECT_NEAR(m.last.fade, 0.5f, 1e-5f);
  m.Shown(1.5, 1.5);
  EXPECT_FLOAT_EQ(m.last.fade, 0.0f);
  // A fresh result after the 1.5 s gap: crossfades in and fades the effect in.
  m.Result(1.5, 2.0f);
  m.Shown(1.5, 0.1);
  EXPECT_FALSE(m.state.stale());
  m.Shown(1.75, 0.1);
  EXPECT_FLOAT_EQ(m.last.fade, 1.0f);
}

TEST(Crossfade, HardCutNeverBlends) {
  Model m;
  m.Result(0.0, 0.0f);
  m.Result(0.15, 1.0f);
  const double now = 0.15 + 0.3 * m.state.duration_s();  // mid-crossfade
  const CrossfadeState::Update u = m.Result(now, 7.0f, /*hard_cut=*/true);
  EXPECT_TRUE(u.hard_cut);
  EXPECT_EQ(u.snapshot_slot, CrossfadeState::kNone);
  EXPECT_FLOAT_EQ(m.Shown(now), 7.0f) << "the new field at once";
  EXPECT_EQ(m.last.from, m.last.to);
  EXPECT_EQ(m.state.hard_cuts(), 1u);
  // History restarts from the cut: the next result fades from 7.
  m.Result(now + 0.15, 9.0f);
  EXPECT_NEAR(m.Shown(now + 0.15), 7.0f, 1e-6f);
}

TEST(Crossfade, ResetIsPassthroughAtOnceAndFadesBackIn) {
  Model m;
  m.Result(0.0, 1.0f);
  m.Shown(0.5);
  ASSERT_FLOAT_EQ(m.last.fade, 1.0f);
  m.state.Reset();
  EXPECT_FALSE(m.state.Advance(0.5, 0.0).active);
  EXPECT_EQ(m.state.resets(), 1u);
  const CrossfadeState::Update u = m.Result(0.6, 3.0f);
  EXPECT_TRUE(u.hard_cut) << "never blends with the old camera's or worker's depth";
  EXPECT_FLOAT_EQ(m.Shown(0.6), 3.0f);
  EXPECT_FLOAT_EQ(m.last.fade, 0.0f);
}

TEST(Crossfade, DisabledMakesEveryResultAHardCut) {
  CrossfadeConfig config;
  config.enabled = false;
  Model m(config);
  m.Result(0.0, 1.0f);
  for (int i = 1; i < 5; ++i) {
    EXPECT_TRUE(m.Result(i * 0.14, static_cast<float>(i)).hard_cut);
    EXPECT_FLOAT_EQ(m.Shown(i * 0.14), static_cast<float>(i));
  }
}

TEST(Crossfade, RandomSequencesKeepOwnershipAndContinuity) {
  std::mt19937 rng(1234);
  std::uniform_real_distribution<double> gap(0.0, 0.4);
  std::uniform_real_distribution<float> value(-10.0f, 10.0f);
  std::uniform_int_distribution<int> event(0, 99);
  for (int sequence = 0; sequence < 1000; ++sequence) {
    Model m;
    double now = 0.0;
    for (int step = 0; step < 40; ++step) {
      now += gap(rng);
      const int e = event(rng);
      if (e < 5) {
        m.state.Reset();
        continue;
      }
      const bool cut = e < 12;
      const bool had_field = m.state.Advance(now, 0.0).active;
      const float before = m.Shown(now);
      m.Result(now, value(rng), cut);  // checks ownership
      if (had_field && !cut) {
        ASSERT_NEAR(m.Shown(now), before, 1e-4f) << "sequence " << sequence << " step " << step;
      }
      m.Shown(now + gap(rng) * 0.5);
    }
    if (testing::Test::HasFailure()) {
      break;
    }
  }
}

}  // namespace
}  // namespace relume
