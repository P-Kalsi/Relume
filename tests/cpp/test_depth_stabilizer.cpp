// DepthStabilizer (temporal stages A and B). Criteria fixed in advance
// (Phase 4 plan, M4 and clarification 2). Flicker is the mean absolute
// frame-to-frame change over a region, the quantity that is visible:
//   - global scale/shift jitter (including percentile shifts from an object
//     that comes and goes): stage A leaves at most 10% of the injected
//     flicker on the static region;
//   - a local step larger than delta1 passes within 1 result (no ghosting);
//   - small noise: stage B (alpha 0.5) reduces flicker by at least 40%;
//   - the history resets for a new camera generation, transform or
//     configuration; letterbox padding never affects the output;
//   - an invalid range bypasses (zeros, no NaN); no allocation in steady
//     state;
//   - drift: a 500-result random walk of the true scale keeps the output's
//     2nd/98th percentiles within [-0.05, 1.05]; a constant scene converges to
//     the percentile normalization within 1e-3; a cut to an unrelated scene
//     resets within 1 result; too few low-gradient pixels resets (no fit).

#include "relume/depth_stabilizer.hpp"
#include "relume/depth_worker.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <new>
#include <random>

namespace {
std::atomic<std::uint64_t> g_allocations{0};
}  // namespace

// Counts heap allocations in this test binary.
void* operator new(std::size_t size) {
  g_allocations.fetch_add(1, std::memory_order_relaxed);
  if (void* p = std::malloc(size == 0 ? 1 : size)) {
    return p;
  }
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

namespace relume {
namespace {

// A smooth scene in the valid region: a sphere-like bump on a gradient,
// values about 1.5..3.6 (larger = nearer). Padding is `padding`.
std::vector<float> Scene(const ResizeTransform& t, float scale, float offset, float padding = 0.0f) {
  std::vector<float> d(kNetDepthFloats, padding);
  for (std::uint32_t y = 0; y < t.new_h; ++y) {
    for (std::uint32_t x = 0; x < t.new_w; ++x) {
      const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(t.new_w);
      const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(t.new_h);
      const float r2 = ((u - 0.5f) * (u - 0.5f) + (v - 0.55f) * (v - 0.55f)) / 0.09f;
      const float bump = r2 < 1.0f ? 1.5f * std::sqrt(1.0f - r2) : 0.0f;
      d[std::size_t{t.pad_y + y} * kNetSize + t.pad_x + x] = scale * (1.5f + 0.6f * v + bump) + offset;
    }
  }
  return d;
}

struct Step {
  StabilizerReport report;
  DisplayRange range;
  std::vector<float> out = std::vector<float>(kNetDepthFloats);
};

Step Feed(DepthStabilizer& s, const std::vector<float>& depth, const ResizeTransform& t, std::uint64_t generation = 1) {
  static std::vector<float> scratch;
  Step step;
  step.range = ComputeDisplayRange(depth.data(), t, scratch);
  step.report = s.Process(depth.data(), t, step.range.lo, step.range.hi, step.range.valid, generation, step.out.data());
  return step;
}

// Percentile normalization of `depth` alone (what the output is without history).
std::vector<float> Normalized(const std::vector<float>& depth, const DisplayRange& r) {
  std::vector<float> p(kNetDepthFloats, 0.0f);
  for (std::size_t i = 0; i < p.size(); ++i) {
    p[i] = (depth[i] - r.lo) / (r.hi - r.lo);
  }
  return p;
}

// Mean |a - b| over valid rows [y0, y1) of the tensor.
double MeanAbs(const std::vector<float>& a, const std::vector<float>& b, const ResizeTransform& t, std::uint32_t y0,
               std::uint32_t y1) {
  double sum = 0.0;
  std::size_t n = 0;
  for (std::uint32_t y = t.pad_y + y0; y < t.pad_y + y1; ++y) {
    for (std::uint32_t x = t.pad_x; x < t.pad_x + t.new_w; ++x) {
      sum += std::abs(a[std::size_t{y} * kNetSize + x] - b[std::size_t{y} * kNetSize + x]);
      ++n;
    }
  }
  return sum / static_cast<double>(n);
}

const ResizeTransform kStretch = *ComputeTransform(1280, 720, ResizeMethod::kStretch);

StabilizerConfig Only(bool align, bool ema) {
  StabilizerConfig c;
  c.align = align;
  c.ema = ema;
  return c;
}

TEST(DepthStabilizer, AlignmentRemovesGlobalJitter) {
  // Each result: a random scale/shift, and in half of them a near object (4%
  // of the region, so it sets the 98th percentile) whose height widens the
  // range by exactly 1.15x: the normalization flickers although the static
  // scene does not change. 1.15x stays inside the approved scale bound
  // [0.8, 1.25]; a larger jump is a scene change (next test).
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> scale(0.8f, 1.25f), offset(-0.5f, 0.5f), coin(0.0f, 1.0f);
  std::vector<float> scratch;
  const DisplayRange base = ComputeDisplayRange(Scene(kStretch, 1.0f, 0.0f).data(), kStretch, scratch);
  const float object = base.lo + 1.15f * (base.hi - base.lo);  // in base units
  DepthStabilizer s(Only(true, false));
  std::vector<float> prev_out, prev_p;
  double injected = 0.0, residual = 0.0;
  float min_a = 1.0f, max_a = 1.0f;
  const std::uint32_t static_y0 = kStretch.new_h / 2;  // the object is only in the top rows
  for (int i = 0; i < 60; ++i) {
    const float k = scale(rng);
    const float o = offset(rng);
    std::vector<float> d = Scene(kStretch, k, o);
    if (coin(rng) < 0.5f) {
      for (std::uint32_t y = 0; y < 90; ++y) {
        for (std::uint32_t x = 0; x < 90; ++x) {
          d[std::size_t{y} * kNetSize + x] = k * object + o;
        }
      }
    }
    const Step step = Feed(s, d, kStretch);
    const std::vector<float> p = Normalized(d, step.range);
    if (i > 0) {
      min_a = std::min(min_a, step.report.fitted ? step.report.a : 1.0f);
      max_a = std::max(max_a, step.report.fitted ? step.report.a : 1.0f);
      EXPECT_FALSE(step.report.scene_change)
          << "result " << i << ": valid " << step.report.valid_fraction << ", inliers " << step.report.inlier_fraction
          << ", median residual " << step.report.median_residual << ", a " << step.report.a;
      injected += MeanAbs(p, prev_p, kStretch, static_y0, kStretch.new_h);
      residual += MeanAbs(step.out, prev_out, kStretch, static_y0, kStretch.new_h);
    }
    prev_out = step.out;
    prev_p = p;
  }
  std::printf("global jitter: injected flicker %.5f, after alignment %.5f (%.1f%%); applied a in [%.3f, %.3f]\n",
              injected / 59, residual / 59, 100.0 * residual / injected, static_cast<double>(min_a),
              static_cast<double>(max_a));
  EXPECT_GT(injected / 59, 0.01) << "the fixture must actually flicker";
  EXPECT_LE(residual, 0.10 * injected);
}

TEST(DepthStabilizer, NormalizationJumpBeyondTheScaleBoundIsASceneChange) {
  DepthStabilizer s(Only(true, false));
  const std::vector<float> base = Scene(kStretch, 1.0f, 0.0f);
  Feed(s, base, kStretch);
  std::vector<float> near = base;
  for (std::uint32_t y = 0; y < kStretch.new_h / 4; ++y) {
    for (std::uint32_t x = 0; x < kStretch.new_w / 3; ++x) {
      near[std::size_t{y} * kNetSize + x] += 6.0f;  // the range roughly triples
    }
  }
  const Step step = Feed(s, near, kStretch);
  EXPECT_TRUE(step.report.scene_change) << "fit a outside [0.8, 1.25] must not be forced";
  EXPECT_LE(MeanAbs(step.out, Normalized(near, step.range), kStretch, 0, kStretch.new_h), 1e-7);
}

TEST(DepthStabilizer, LargeLocalChangePassesWithinOneResult) {
  DepthStabilizer s;  // both stages
  const std::vector<float> base = Scene(kStretch, 1.0f, 0.0f);
  for (int i = 0; i < 5; ++i) {
    Feed(s, base, kStretch);
  }
  // A "hand": a block that moves 0.25 of the range nearer at once.
  std::vector<float> moved = base;
  const DisplayRange r = Feed(s, base, kStretch).range;
  for (std::uint32_t y = 300; y < 380; ++y) {
    for (std::uint32_t x = 40; x < 140; ++x) {
      moved[std::size_t{y} * kNetSize + x] += 0.25f * (r.hi - r.lo);
    }
  }
  const Step step = Feed(s, moved, kStretch);
  const std::vector<float> p = Normalized(moved, step.range);
  double worst = 0.0;
  for (std::uint32_t y = 304; y < 376; ++y) {
    for (std::uint32_t x = 44; x < 136; ++x) {
      const std::size_t i = std::size_t{y} * kNetSize + x;
      worst = std::max(worst, static_cast<double>(std::abs(step.out[i] - (step.report.a * p[i] + step.report.b))));
    }
  }
  EXPECT_LE(worst, 1e-5) << "the moved block is not held back by the EMA";
}

TEST(DepthStabilizer, EmaReducesSmallNoiseFlicker) {
  std::mt19937 rng(11);
  std::normal_distribution<float> noise(0.0f, 1.0f);
  DepthStabilizer s(Only(false, true));
  const std::vector<float> base = Scene(kStretch, 1.0f, 0.0f);
  std::vector<float> prev_out, prev_in;
  double before = 0.0, after = 0.0;
  // Fixed normalization: identical range each time, so only the noise moves.
  const DisplayRange r = Feed(s, base, kStretch).range;
  for (int i = 0; i < 80; ++i) {
    std::vector<float> d = base;
    for (std::uint32_t y = 0; y < kStretch.new_h; ++y) {
      for (std::uint32_t x = 0; x < kStretch.new_w; ++x) {
        d[std::size_t{y} * kNetSize + x] += 0.01f * (r.hi - r.lo) * noise(rng);  // sigma 0.01 of the range
      }
    }
    std::vector<float> out(kNetDepthFloats);
    s.Process(d.data(), kStretch, r.lo, r.hi, true, 1, out.data());
    const std::vector<float> p = Normalized(d, r);
    if (i > 0) {
      before += MeanAbs(p, prev_in, kStretch, 0, kStretch.new_h);
      after += MeanAbs(out, prev_out, kStretch, 0, kStretch.new_h);
    }
    prev_out = out;
    prev_in = p;
  }
  std::printf("small noise: flicker %.5f -> %.5f (%.1f%% lower)\n", before / 79, after / 79,
              100.0 * (1.0 - after / before));
  EXPECT_LE(after, 0.6 * before);
}

TEST(DepthStabilizer, ResetsForGenerationTransformAndConfiguration) {
  DepthStabilizer s;
  const std::vector<float> a = Scene(kStretch, 1.0f, 0.0f);
  EXPECT_TRUE(Feed(s, a, kStretch, 1).report.reset) << "first result";
  EXPECT_FALSE(Feed(s, a, kStretch, 1).report.reset);
  Step step = Feed(s, a, kStretch, 2);
  EXPECT_TRUE(step.report.reset) << "new camera generation";
  EXPECT_FALSE(step.report.scene_change);
  const ResizeTransform letterbox = *ComputeTransform(1280, 720, ResizeMethod::kLetterbox);
  const std::vector<float> b = Scene(letterbox, 1.0f, 0.0f);
  step = Feed(s, b, letterbox, 2);
  EXPECT_TRUE(step.report.reset) << "new transform";
  const std::vector<float> p = Normalized(b, step.range);
  EXPECT_LE(MeanAbs(step.out, p, letterbox, 0, letterbox.new_h), 1e-7) << "a reset outputs the normalization";
  EXPECT_FALSE(Feed(s, b, letterbox, 2).report.reset);
  s.SetConfig(Only(true, false));
  EXPECT_TRUE(Feed(s, b, letterbox, 2).report.reset) << "new configuration";
}

TEST(DepthStabilizer, PaddingNeverAffectsTheOutput) {
  const ResizeTransform t = *ComputeTransform(1280, 720, ResizeMethod::kLetterbox);
  ASSERT_GT(t.pad_y, 0u);
  DepthStabilizer s1, s2;
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> scale(0.9f, 1.1f);
  for (int i = 0; i < 10; ++i) {
    const float k = scale(rng);
    const Step a = Feed(s1, Scene(t, k, 0.0f, 0.0f), t);
    const Step b = Feed(s2, Scene(t, k, 0.0f, 1e6f), t);
    ASSERT_EQ(a.out, b.out) << "result " << i;
    for (std::uint32_t y = 0; y < t.pad_y; ++y) {
      ASSERT_EQ(a.out[std::size_t{y} * kNetSize + 10], 0.0f);
    }
  }
}

TEST(DepthStabilizer, InvalidRangeBypassesWithoutNaN) {
  DepthStabilizer s;
  const std::vector<float> a = Scene(kStretch, 1.0f, 0.0f);
  Feed(s, a, kStretch);
  std::vector<float> out(kNetDepthFloats, 5.0f);
  const StabilizerReport r = s.Process(a.data(), kStretch, 1.0f, 1.0f, false, 1, out.data());
  EXPECT_TRUE(r.bypassed);
  EXPECT_TRUE(std::all_of(out.begin(), out.end(), [](float v) { return v == 0.0f; }));
  std::vector<float> with_nan = a;
  with_nan[1000] = std::nanf("");
  EXPECT_TRUE(s.Process(with_nan.data(), kStretch, 1.5f, 3.6f, true, 1, out.data()).bypassed);
  EXPECT_TRUE(Feed(s, a, kStretch).report.reset) << "the history restarts after a bypass";
}

TEST(DepthStabilizer, SteadyStateDoesNotAllocate) {
  DepthStabilizer s;
  const std::vector<float> a = Scene(kStretch, 1.0f, 0.0f);
  const std::vector<float> b = Scene(kStretch, 1.1f, 0.2f);
  std::vector<float> out(kNetDepthFloats);
  s.Process(a.data(), kStretch, 1.5f, 3.6f, true, 1, out.data());
  s.Process(b.data(), kStretch, 1.85f, 4.2f, true, 1, out.data());
  const std::uint64_t before = g_allocations.load();
  for (int i = 0; i < 20; ++i) {
    s.Process((i % 2 ? a : b).data(), kStretch, i % 2 ? 1.5f : 1.85f, i % 2 ? 3.6f : 4.2f, true, 1, out.data());
  }
  EXPECT_EQ(g_allocations.load() - before, 0u);
}

TEST(DepthStabilizer, RandomWalkOfScaleDoesNotDrift) {
  std::mt19937 rng(5);
  std::uniform_real_distribution<float> step(-0.03f, 0.03f);
  std::normal_distribution<float> noise(0.0f, 1.0f);
  DepthStabilizer s;
  float scale = 1.0f, offset = 0.0f;
  static std::vector<float> scratch;
  float worst_lo = 0.0f, worst_hi = 1.0f;
  for (int i = 0; i < 500; ++i) {
    scale = std::clamp(scale * (1.0f + step(rng)), 0.3f, 3.0f);
    offset += step(rng);
    std::vector<float> d = Scene(kStretch, scale, offset);
    for (float& v : d) {
      v += 0.005f * scale * noise(rng);
    }
    const Step st = Feed(s, d, kStretch);
    ASSERT_FALSE(st.report.scene_change) << "result " << i;
    const DisplayRange out_range = ComputeDisplayRange(st.out.data(), kStretch, scratch);
    worst_lo = std::min(worst_lo, out_range.lo);
    worst_hi = std::max(worst_hi, out_range.hi);
  }
  std::printf("random walk: output p2 >= %.4f, p98 <= %.4f\n", static_cast<double>(worst_lo),
              static_cast<double>(worst_hi));
  EXPECT_GE(worst_lo, -0.05f);
  EXPECT_LE(worst_hi, 1.05f);
}

TEST(DepthStabilizer, ConstantSceneConvergesToTheNormalization) {
  DepthStabilizer s;
  // The first result has an extra near object (another normalization); then
  // the scene stays constant. The leak must bring the output back to the
  // plain normalization.
  std::vector<float> first = Scene(kStretch, 1.0f, 0.0f);
  for (std::uint32_t y = 0; y < 100; ++y) {
    for (std::uint32_t x = 0; x < 150; ++x) {
      first[std::size_t{y} * kNetSize + x] += 6.0f;
    }
  }
  Feed(s, first, kStretch);
  const std::vector<float> constant = Scene(kStretch, 1.0f, 0.0f);
  Step step;
  for (int i = 0; i < 500; ++i) {
    step = Feed(s, constant, kStretch);
  }
  const std::vector<float> p = Normalized(constant, step.range);
  double worst = 0.0;
  for (std::uint32_t y = 0; y < kStretch.new_h; ++y) {
    for (std::uint32_t x = 0; x < kStretch.new_w; ++x) {
      const std::size_t i = std::size_t{y} * kNetSize + x;
      worst = std::max(worst, static_cast<double>(std::abs(step.out[i] - p[i])));
    }
  }
  EXPECT_LE(worst, 1e-3);
}

TEST(DepthStabilizer, CutToAnUnrelatedSceneResetsWithinOneResult) {
  DepthStabilizer s;
  const std::vector<float> a = Scene(kStretch, 1.0f, 0.0f);
  for (int i = 0; i < 5; ++i) {
    Feed(s, a, kStretch);
  }
  // Unrelated: depth that increases to the left, with a far bump.
  std::vector<float> b(kNetDepthFloats);
  for (std::uint32_t y = 0; y < kNetSize; ++y) {
    for (std::uint32_t x = 0; x < kNetSize; ++x) {
      const float u = static_cast<float>(x) / kNetSize;
      const float v = static_cast<float>(y) / kNetSize;
      const float r2 = ((u - 0.5f) * (u - 0.5f) + (v - 0.55f) * (v - 0.55f)) / 0.09f;
      b[std::size_t{y} * kNetSize + x] = 3.0f - 1.5f * u - (r2 < 1.0f ? 1.2f * std::sqrt(1.0f - r2) : 0.0f);
    }
  }
  const Step step = Feed(s, b, kStretch);
  EXPECT_TRUE(step.report.scene_change) << "inliers " << step.report.inlier_fraction << ", median residual "
                                        << step.report.median_residual << ", a " << step.report.a;
  EXPECT_TRUE(step.report.reset);
  EXPECT_LE(MeanAbs(step.out, Normalized(b, step.range), kStretch, 0, kStretch.new_h), 1e-7);
}

TEST(DepthStabilizer, TooFewLowGradientPixelsResetsWithoutAFit) {
  DepthStabilizer s;
  // Fine stripes: the gradient is high almost everywhere.
  std::vector<float> d(kNetDepthFloats);
  for (std::size_t i = 0; i < d.size(); ++i) {
    d[i] = ((i / 2) % 2 == 0) ? 1.0f : 2.0f;  // period 4: a central difference sees it
  }
  Feed(s, d, kStretch);
  const Step step = Feed(s, d, kStretch);
  EXPECT_TRUE(step.report.scene_change);
  EXPECT_FALSE(step.report.fitted);
  EXPECT_LT(step.report.valid_fraction, 0.15f);
  EXPECT_TRUE(std::all_of(step.out.begin(), step.out.end(), [](float v) { return std::isfinite(v); }));
}

}  // namespace
}  // namespace relume
