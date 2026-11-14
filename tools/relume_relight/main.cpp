// relume_relight: the Phase 4 temporal-stabilization ablation.
//
// Replays depth sequences at 30 fps, with a depth result every 4th frame that
// arrives 5 frames after its source frame (the measured CoreML cadence:
// about 7.5 results/s, lag about 5 frames), through the app's own temporal
// code: DepthStabilizer (stages A and B, as on the worker) and CrossfadeState
// (stage C, as on the main thread). The lighting gain is computed with the
// CPU reference (BuildDepthField, ComputeShading), which matches the GPU
// passes (tests/cpp/test_relight_gpu.cpp: field within 1e-3, output within
// 1 level), at a 160x90 grid of image points, in float.
//
// Scenarios:
//   static   a still scene with model-like jitter (global scale/shift,
//            low-frequency and per-texel noise): flicker F and stepping S
//   step     a near disc that jumps to a new place: response lag R
//   cut      a different scene: ghost area G
//   burst    (with --burst DIR) local camera frames run through ONNX Runtime
//            (CPU): F and S on real model output. Frames are read locally and
//            nothing is written unless --write-dir is given.
//
// Metrics (Phase 4 plan, M4; eps = half an 8-bit step in gain units, 0.5/255):
//   F = mean over frames of the 95th percentile over points of |gain(t) - gain(t-1)|
//   S = (mean |dgain| on frames where a result arrived + eps) / (mean on other frames + eps)
//   R = frames from the step's arrival until 90% of the gain change is reached
//       ("not measurable" if the total change is below 4 eps)
//   G = points still more than 10% of the old-to-new gain difference away from
//       the new gain 300 ms after the cut arrives (points whose difference is
//       below 4 eps are ignored)
// Thresholds (approved): F <= max(0.5 F_none, eps), S <= 1.5, R <= 6, G = 0.
//
// Usage: relume_relight [--burst DIR]... [--model PATH] [--json PATH]

#include "relume/crossfade.hpp"
#include "relume/depth_estimator.hpp"
#include "relume/depth_stabilizer.hpp"
#include "relume/depth_worker.hpp"
#include "relume/image_io.hpp"
#include "relume/lighting.hpp"
#include "relume/preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace relume {
namespace {

constexpr double kFps = 30.0;
constexpr int kInterval = 4;  // frames between depth results
constexpr int kLag = 5;       // frames from a result's source frame to its arrival
constexpr std::uint32_t kGridW = 160;
constexpr std::uint32_t kGridH = 90;
constexpr double kEps = 0.5 / 255.0;
const ResizeTransform kTransform = *ComputeTransform(1280, 720, ResizeMethod::kStretch);
constexpr float kAspect = 16.0f / 9.0f;

using Tensor = std::vector<float>;  // 448x448
using Grid = std::vector<FieldTexel>;  // the field sampled at the grid points
using Gains = std::vector<float>;

struct Config {
  std::string name;
  bool align = false;
  bool ema = false;
  float alpha = 0.5f;
  bool crossfade = false;
  double factor = 0.5;
  int stages() const { return (align ? 1 : 0) + (ema ? 1 : 0) + (crossfade ? 1 : 0); }
};

// Bilinear sample of a field at the grid points (half-pixel rule, clamped),
// as the GPU's filtering sampler does.
Grid SampleGrid(const std::vector<FieldTexel>& field, std::uint32_t w, std::uint32_t h) {
  Grid grid(std::size_t{kGridW} * kGridH);
  for (std::uint32_t gy = 0; gy < kGridH; ++gy) {
    for (std::uint32_t gx = 0; gx < kGridW; ++gx) {
      const float u = (static_cast<float>(gx) + 0.5f) / kGridW;
      const float v = (static_cast<float>(gy) + 0.5f) / kGridH;
      const float px = std::clamp(u * static_cast<float>(w) - 0.5f, 0.0f, static_cast<float>(w - 1));
      const float py = std::clamp(v * static_cast<float>(h) - 0.5f, 0.0f, static_cast<float>(h - 1));
      const auto x0 = static_cast<std::uint32_t>(px);
      const auto y0 = static_cast<std::uint32_t>(py);
      const std::uint32_t x1 = std::min(x0 + 1, w - 1);
      const std::uint32_t y1 = std::min(y0 + 1, h - 1);
      const float fx = px - static_cast<float>(x0);
      const float fy = py - static_cast<float>(y0);
      const auto at = [&](std::uint32_t x, std::uint32_t y) { return field[std::size_t{y} * w + x]; };
      const auto mix = [](const FieldTexel& a, const FieldTexel& b, float t) {
        return FieldTexel{a.sx + (b.sx - a.sx) * t, a.sy + (b.sy - a.sy) * t, a.h + (b.h - a.h) * t,
                          a.conf + (b.conf - a.conf) * t};
      };
      grid[std::size_t{gy} * kGridW + gx] = mix(mix(at(x0, y0), at(x1, y0), fx), mix(at(x0, y1), at(x1, y1), fx), fy);
    }
  }
  return grid;
}

Grid Lerp(const Grid& a, const Grid& b, float t) {
  Grid out(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    out[i] = {a[i].sx + (b[i].sx - a[i].sx) * t, a[i].sy + (b[i].sy - a[i].sy) * t, a[i].h + (b[i].h - a[i].h) * t,
              a[i].conf + (b[i].conf - a[i].conf) * t};
  }
  return out;
}

float g_depth_falloff = 1.0f;  // --depth-falloff
float g_relief = LightSettings{}.relief;  // relief 0 isolates the shape term (diagnostics)

Gains Shade(const Grid& grid, float fade) {
  LightSettings light;  // the default light
  light.depth_falloff = g_depth_falloff;
  light.relief = g_relief;
  Gains gains(grid.size());
  for (std::uint32_t gy = 0; gy < kGridH; ++gy) {
    for (std::uint32_t gx = 0; gx < kGridW; ++gx) {
      const std::size_t i = std::size_t{gy} * kGridW + gx;
      const float u = (static_cast<float>(gx) + 0.5f) / kGridW;
      const float v = (static_cast<float>(gy) + 0.5f) / kGridH;
      const Shading s = ComputeShading(grid[i], u, v, light, kAspect);
      const float g = 0.2126f * s.gain.x + 0.7152f * s.gain.y + 0.0722f * s.gain.z;
      gains[i] = 1.0f + (g - 1.0f) * fade;
    }
  }
  return gains;
}

// The gain seen at each 30 fps frame for a sequence of raw depth results.
struct Replay {
  std::vector<Gains> frames;
  std::vector<bool> arrival;      // a result arrived at this frame
  std::vector<int> arrival_of;    // frame at which result k arrived
  int scene_changes = 0;
  int hard_cuts = 0;
};

Replay Run(const std::vector<Tensor>& results, const Config& c, int extra_frames) {
  StabilizerConfig sc;
  sc.align = c.align;
  sc.ema = c.ema;
  sc.alpha = c.alpha;
  DepthStabilizer stabilizer(sc);
  CrossfadeConfig cc;
  cc.enabled = c.crossfade;
  cc.duration_factor = c.factor;
  cc.default_interval_s = kInterval / kFps;
  CrossfadeState crossfade(cc);
  std::array<Grid, CrossfadeState::kSlots> slots;
  std::vector<float> scratch;
  Tensor stable(kNetDepthFloats);

  Replay replay;
  const int frames = static_cast<int>(results.size()) * kInterval + kLag + extra_frames;
  std::size_t next = 0;
  for (int f = 0; f < frames; ++f) {
    const double now = f / kFps;
    bool arrived = false;
    if (next < results.size() && f == static_cast<int>(next) * kInterval + kLag) {
      const Tensor& raw = results[next];
      const DisplayRange range = ComputeDisplayRange(raw.data(), kTransform, scratch);
      const StabilizerReport report =
          stabilizer.Process(raw.data(), kTransform, range.lo, range.hi, range.valid, 1, stable.data());
      replay.scene_changes += report.scene_change ? 1 : 0;
      const CrossfadeState::Update u = crossfade.OnResult(now, report.reset);
      replay.hard_cuts += u.hard_cut && next > 0 ? 1 : 0;
      if (u.snapshot_slot != CrossfadeState::kNone) {
        slots[static_cast<std::size_t>(u.snapshot_slot)] = Lerp(slots[static_cast<std::size_t>(u.snapshot_from)],
                                                                slots[static_cast<std::size_t>(u.snapshot_to)],
                                                                u.snapshot_t);
      }
      const std::vector<FieldTexel> field = BuildDepthField(stable.data(), kTransform.new_w, kTransform.new_h,
                                                            kNetSize, 1.0f, 0.0f, kAspect);
      slots[static_cast<std::size_t>(u.build_slot)] = SampleGrid(field, kTransform.new_w, kTransform.new_h);
      replay.arrival_of.push_back(f);
      arrived = true;
      ++next;
    }
    const double age = (kLag + (f - (next == 0 ? 0 : replay.arrival_of.back()))) / kFps;
    const CrossfadeState::View view = crossfade.Advance(now, age);
    replay.arrival.push_back(arrived);
    if (!view.active) {
      replay.frames.emplace_back(std::size_t{kGridW} * kGridH, 1.0f);
      continue;
    }
    const Grid shown = Lerp(slots[static_cast<std::size_t>(view.from)], slots[static_cast<std::size_t>(view.to)],
                            view.blend);
    replay.frames.push_back(Shade(shown, view.fade));
  }
  return replay;
}

// --- Metrics ---

struct Temporal {
  double flicker = 0.0;  // F
  double stepping = 0.0;  // S
  double update_mean = 0.0, other_mean = 0.0;
};

// Over grid rows [row0, row1).
Temporal Measure(const Replay& r, int from_frame, std::uint32_t row0 = 0, std::uint32_t row1 = kGridH) {
  Temporal t;
  double f_sum = 0.0, upd = 0.0, oth = 0.0;
  int n = 0, n_upd = 0, n_oth = 0;
  std::vector<float> diff(std::size_t{kGridW} * (row1 - row0));
  const std::size_t first = std::size_t{row0} * kGridW;
  for (std::size_t f = static_cast<std::size_t>(from_frame) + 1; f < r.frames.size(); ++f) {
    double mean = 0.0;
    for (std::size_t i = 0; i < diff.size(); ++i) {
      diff[i] = std::abs(r.frames[f][first + i] - r.frames[f - 1][first + i]);
      mean += diff[i];
    }
    mean /= static_cast<double>(diff.size());
    const auto p95 = diff.begin() + static_cast<std::ptrdiff_t>(0.95 * static_cast<double>(diff.size() - 1));
    std::nth_element(diff.begin(), p95, diff.end());
    f_sum += *p95;
    ++n;
    if (r.arrival[f]) {
      upd += mean;
      ++n_upd;
    } else {
      oth += mean;
      ++n_oth;
    }
  }
  t.flicker = n > 0 ? f_sum / n : 0.0;
  t.update_mean = n_upd > 0 ? upd / n_upd : 0.0;
  t.other_mean = n_oth > 0 ? oth / n_oth : 0.0;
  t.stepping = (t.update_mean + kEps) / (t.other_mean + kEps);
  return t;
}

// R: frames from the arrival of result `k` until 90% of the change from the
// frame before it to the last frame. Negative: not measurable.
int ResponseLag(const Replay& r, std::size_t k, double& total_change) {
  const int arrive = r.arrival_of[k];
  const Gains& before = r.frames[static_cast<std::size_t>(arrive - 1)];
  const Gains& after = r.frames.back();
  double denom = 0.0, max_change = 0.0;
  for (std::size_t i = 0; i < before.size(); ++i) {
    const double d = after[i] - before[i];
    denom += d * d;
    max_change = std::max(max_change, std::abs(d));
  }
  total_change = max_change;
  if (max_change < 4.0 * kEps) {
    return -1;
  }
  for (std::size_t f = static_cast<std::size_t>(arrive); f < r.frames.size(); ++f) {
    double num = 0.0;
    for (std::size_t i = 0; i < before.size(); ++i) {
      num += (r.frames[f][i] - before[i]) * (after[i] - before[i]);
    }
    if (num / denom >= 0.9) {
      return static_cast<int>(f) - arrive;
    }
  }
  return 1000;
}

// G: points still more than 10% of the old-to-new difference away from the
// new gain, 300 ms (9 frames) after result `k` arrived.
int GhostArea(const Replay& r, std::size_t k) {
  const int arrive = r.arrival_of[k];
  const Gains& before = r.frames[static_cast<std::size_t>(arrive - 1)];
  const Gains& after = r.frames.back();
  const Gains& at = r.frames[std::min(r.frames.size() - 1, static_cast<std::size_t>(arrive + 9))];
  int count = 0;
  for (std::size_t i = 0; i < before.size(); ++i) {
    const double d = std::abs(after[i] - before[i]);
    if (d >= 4.0 * kEps && std::abs(at[i] - after[i]) > 0.1 * d) {
      ++count;
    }
  }
  return count;
}

// The sharpest gain step between horizontal neighbours in a frame: a seam
// shows up as a large value (open issue 1).
double SharpestStep(const Gains& g) {
  double worst = 0.0;
  for (std::uint32_t y = 0; y < kGridH; ++y) {
    for (std::uint32_t x = 1; x < kGridW; ++x) {
      worst = std::max(worst, static_cast<double>(std::abs(g[std::size_t{y} * kGridW + x] -
                                                           g[std::size_t{y} * kGridW + x - 1])));
    }
  }
  return worst;
}

// Open issue 2: how much the subject's surface tilts. Over confident texels
// of the near region (normalized h above 0.6), the 50th and 90th percentile
// of the tilt angle atan(relief * |slope|) at the default relief. A face
// tilted by only a few degrees can barely be shaped by the light.
struct Tilt {
  double p50_deg = 0.0;
  double p90_deg = 0.0;
  std::size_t texels = 0;
};

Tilt SubjectTilt(const std::vector<Tensor>& results) {
  std::vector<double> angles;
  std::vector<float> scratch;
  Tensor stable(kNetDepthFloats);
  DepthStabilizer stabilizer;
  const float relief = LightSettings{}.relief;
  for (const Tensor& raw : results) {
    const DisplayRange range = ComputeDisplayRange(raw.data(), kTransform, scratch);
    stabilizer.Process(raw.data(), kTransform, range.lo, range.hi, range.valid, 1, stable.data());
    const std::vector<FieldTexel> field =
        BuildDepthField(stable.data(), kTransform.new_w, kTransform.new_h, kNetSize, 1.0f, 0.0f, kAspect);
    for (const FieldTexel& t : field) {
      if (t.h > 0.6f && t.conf >= 0.99f) {
        const double g = std::min(static_cast<double>(kMaxGradient),
                                  relief * std::sqrt(static_cast<double>(t.sx * t.sx + t.sy * t.sy)));
        angles.push_back(std::atan(g) * 180.0 / 3.14159265358979);
      }
    }
  }
  Tilt tilt;
  tilt.texels = angles.size();
  if (!angles.empty()) {
    std::sort(angles.begin(), angles.end());
    tilt.p50_deg = angles[angles.size() / 2];
    tilt.p90_deg = angles[angles.size() * 9 / 10];
  }
  return tilt;
}

// Open issue 2: where the light's effect goes. For the last result of a
// sequence, at the default light: the mean gain on the subject (h > 0.6)
// minus the background (h < 0.4) is the subject-vs-background separation,
// and the spread (standard deviation) of the gain within the subject is the
// face-shaped part.
struct GainSplit {
  double separation = 0.0;
  double shape = 0.0;
};

GainSplit SplitGain(const std::vector<Tensor>& results, float falloff, float relief = LightSettings{}.relief) {
  std::vector<float> scratch;
  Tensor stable(kNetDepthFloats);
  DepthStabilizer stabilizer;
  for (const Tensor& raw : results) {
    const DisplayRange range = ComputeDisplayRange(raw.data(), kTransform, scratch);
    stabilizer.Process(raw.data(), kTransform, range.lo, range.hi, range.valid, 1, stable.data());
  }
  const std::vector<FieldTexel> field =
      BuildDepthField(stable.data(), kTransform.new_w, kTransform.new_h, kNetSize, 1.0f, 0.0f, kAspect);
  const Grid grid = SampleGrid(field, kTransform.new_w, kTransform.new_h);
  const float saved = g_depth_falloff;
  const float saved_relief = g_relief;
  g_depth_falloff = falloff;
  g_relief = relief;
  const Gains gains = Shade(grid, 1.0f);
  g_depth_falloff = saved;
  g_relief = saved_relief;
  double sub = 0.0, sub2 = 0.0, bg = 0.0;
  std::size_t n_sub = 0, n_bg = 0;
  for (std::size_t i = 0; i < gains.size(); ++i) {
    if (grid[i].h > 0.6f) {
      sub += gains[i];
      sub2 += static_cast<double>(gains[i]) * gains[i];
      ++n_sub;
    } else if (grid[i].h < 0.4f) {
      bg += gains[i];
      ++n_bg;
    }
  }
  GainSplit split;
  if (n_sub > 1 && n_bg > 0) {
    const double mean = sub / static_cast<double>(n_sub);
    split.separation = mean - bg / static_cast<double>(n_bg);
    split.shape = std::sqrt(std::max(0.0, sub2 / static_cast<double>(n_sub) - mean * mean));
  }
  return split;
}

// --- Scenes ---

// A head-and-shoulders-like scene: an ellipsoid head on a shoulder band in
// front of a far wall; values like the model's (about 0.5 far .. 5 near).
Tensor Person(float head_x, float head_y, float scale, float offset) {
  Tensor d(kNetDepthFloats);
  for (std::uint32_t y = 0; y < kNetSize; ++y) {
    for (std::uint32_t x = 0; x < kNetSize; ++x) {
      const float u = (static_cast<float>(x) + 0.5f) / kNetSize;
      const float v = (static_cast<float>(y) + 0.5f) / kNetSize;
      float value = 0.6f + 0.4f * v;  // wall and floor
      const float hx = (u - head_x) / 0.13f;
      const float hy = (v - head_y) / 0.22f;
      const float r2 = hx * hx + hy * hy;
      if (r2 < 1.0f) {
        value = std::max(value, 3.2f + 1.3f * std::sqrt(1.0f - r2));
      }
      const float sx = (u - head_x) / 0.32f;
      if (v > head_y + 0.2f && std::abs(sx) < 1.0f) {
        value = std::max(value, 2.8f + 0.8f * std::sqrt(1.0f - sx * sx));
      }
      d[std::size_t{y} * kNetSize + x] = scale * value + offset;
    }
  }
  return d;
}

// Model-like run-to-run variation: global scale/shift, smooth low-frequency
// noise and fine noise.
void AddJitter(Tensor& d, std::mt19937& rng) {
  std::normal_distribution<float> n(0.0f, 1.0f);
  const float scale = 1.0f + 0.04f * n(rng);
  const float shift = 0.05f * n(rng);
  float a[3], fx[3], fy[3], ph[3];
  for (int k = 0; k < 3; ++k) {
    a[k] = 0.03f * n(rng);
    fx[k] = 2.0f + 2.0f * static_cast<float>(k);
    fy[k] = 1.0f + 3.0f * static_cast<float>(k);
    ph[k] = 6.283f * std::abs(n(rng));
  }
  for (std::uint32_t y = 0; y < kNetSize; ++y) {
    for (std::uint32_t x = 0; x < kNetSize; ++x) {
      const float u = static_cast<float>(x) / kNetSize;
      const float v = static_cast<float>(y) / kNetSize;
      float low = 0.0f;
      for (int k = 0; k < 3; ++k) {
        low += a[k] * std::sin(6.283f * (fx[k] * u + fy[k] * v) + ph[k]);
      }
      float& value = d[std::size_t{y} * kNetSize + x];
      value = value * scale + shift + low + 0.01f * n(rng);
    }
  }
}

std::vector<Tensor> StaticScene() {
  std::mt19937 rng(42);
  std::vector<Tensor> out;
  for (int k = 0; k < 40; ++k) {
    Tensor d = Person(0.5f, 0.38f, 1.0f, 0.0f);
    AddJitter(d, rng);
    out.push_back(std::move(d));
  }
  return out;
}

// Informational (not used for selection): the static scene with a near hand
// in the bottom rows in half of the results, which moves the 98th percentile
// and so the whole normalization. Measured on the top half only.
std::vector<Tensor> HandScene() {
  std::mt19937 rng(9);
  std::uniform_real_distribution<float> coin(0.0f, 1.0f);
  std::vector<Tensor> out;
  for (int k = 0; k < 40; ++k) {
    Tensor d = Person(0.5f, 0.38f, 1.0f, 0.0f);
    AddJitter(d, rng);
    if (coin(rng) < 0.5f) {
      for (std::uint32_t y = 360; y < 448; ++y) {
        for (std::uint32_t x = 20; x < 120; ++x) {
          d[std::size_t{y} * kNetSize + x] = 5.4f;  // nearer than the face
        }
      }
    }
    out.push_back(std::move(d));
  }
  return out;
}

std::vector<Tensor> StepScene() {
  std::vector<Tensor> out;
  for (int k = 0; k < 24; ++k) {
    out.push_back(Person(k < 12 ? 0.42f : 0.58f, 0.38f, 1.0f, 0.0f));  // the head moves at result 12
  }
  return out;
}

std::vector<Tensor> CutScene() {
  std::vector<Tensor> out;
  for (int k = 0; k < 12; ++k) {
    out.push_back(Person(0.5f, 0.38f, 1.0f, 0.0f));
  }
  // An empty room: a slanted wall nearer on the left.
  Tensor room(kNetDepthFloats);
  for (std::uint32_t y = 0; y < kNetSize; ++y) {
    for (std::uint32_t x = 0; x < kNetSize; ++x) {
      room[std::size_t{y} * kNetSize + x] = 2.5f - 1.8f * static_cast<float>(x) / kNetSize;
    }
  }
  for (int k = 0; k < 12; ++k) {
    out.push_back(room);
  }
  return out;
}

std::optional<std::vector<Tensor>> BurstScene(const std::string& dir, IDepthEstimator& estimator) {
  std::vector<std::filesystem::path> files;
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (entry.path().extension() == ".ppm") {
      files.push_back(entry.path());
    }
  }
  std::sort(files.begin(), files.end());
  Preprocessor pre;
  std::vector<Tensor> out;
  for (const auto& path : files) {
    std::string error;
    const std::optional<RgbImage> image = ReadPpm(path.string(), error);
    if (!image || image->width != 1280 || image->height != 720) {
      std::fprintf(stderr, "skipping %s: %s\n", path.filename().string().c_str(),
                   image ? "not 1280x720" : error.c_str());
      continue;
    }
    if (!pre.Run(image->pixels.data(), image->width, image->height, ResizeMethod::kStretch, estimator.input()) ||
        !estimator.Run(error)) {
      std::fprintf(stderr, "inference failed: %s\n", error.c_str());
      return std::nullopt;
    }
    out.emplace_back(estimator.output(), estimator.output() + kNetDepthFloats);
  }
  return out;
}

std::vector<Config> Configs() {
  return {
      {"none", false, false, 0.5f, false, 0.5},
      {"A", true, false, 0.5f, false, 0.5},
      {"B (a .5)", false, true, 0.5f, false, 0.5},
      {"B+C (a .5, T .5)", false, true, 0.5f, true, 0.5},
      {"B+C (a .3, T 1.0)", false, true, 0.3f, true, 1.0},
      {"A+B (a .5)", true, true, 0.5f, false, 0.5},
      {"A+B (a .3)", true, true, 0.3f, false, 0.5},
      {"A+C (T .5)", true, false, 0.5f, true, 0.5},
      {"A+C (T 1.0)", true, false, 0.5f, true, 1.0},
      {"A+B+C (a .5, T .5)", true, true, 0.5f, true, 0.5},
      {"A+B+C (a .3, T 1.0)", true, true, 0.3f, true, 1.0},
      {"C only (T 1.0)", false, false, 0.5f, true, 1.0},  // for reference: no stabilizer
      {"A+B+C (a .5, T 1.0)", true, true, 0.5f, true, 1.0},
  };
}

}  // namespace
}  // namespace relume

int main(int argc, char** argv) {
  using namespace relume;
  std::vector<std::string> bursts;
  std::string model = RELUME_DEFAULT_MODEL;
  std::string json_path;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--burst" && i + 1 < argc) {
      bursts.push_back(argv[++i]);
    } else if (arg == "--model" && i + 1 < argc) {
      model = argv[++i];
    } else if (arg == "--json" && i + 1 < argc) {
      json_path = argv[++i];
    } else if (arg == "--depth-falloff" && i + 1 < argc) {
      g_depth_falloff = std::clamp(std::stof(argv[++i]), 0.0f, 1.0f);
    } else {
      std::fprintf(stderr,
                   "usage: relume_relight [--burst DIR]... [--model PATH] [--json PATH] [--depth-falloff 0..1]\n");
      return 2;
    }
  }

  const std::vector<Config> configs = Configs();
  const std::vector<Tensor> still = StaticScene();
  const std::vector<Tensor> step = StepScene();
  const std::vector<Tensor> cut = CutScene();
  const std::vector<Tensor> hand = HandScene();
  std::vector<std::pair<std::string, std::vector<Tensor>>> burst_scenes;
  if (!bursts.empty()) {
    EstimatorConfig ec;
    ec.model_path = model;
    ec.provider = ExecutionProvider::kCpu;
    ec.intra_op_threads = 4;
    std::unique_ptr<IDepthEstimator> estimator = MakeOnnxDepthEstimator(ec);
    std::string error;
    if (!estimator->Load(error)) {
      std::fprintf(stderr, "could not load the model: %s\n", error.c_str());
      return 1;
    }
    for (const std::string& dir : bursts) {
      std::optional<std::vector<Tensor>> results = BurstScene(dir, *estimator);
      if (!results || results->size() < 4) {
        std::fprintf(stderr, "burst %s: not enough frames\n", dir.c_str());
        return 1;
      }
      burst_scenes.emplace_back(std::filesystem::path(dir).filename().string(), std::move(*results));
    }
  }

  std::printf("Cadence: a result every %d frames at %.0f fps (%.1f/s), arriving %d frames late. eps = %.5f. "
              "Depth falloff %.2f\n",
              kInterval, kFps, kFps / kInterval, kLag, kEps, static_cast<double>(g_depth_falloff));
  std::printf("%-22s | %-27s | %-14s | %-7s | %-5s | %-13s", "config", "static: F (S; upd/other)", "step: R (frames)",
              "cut: G", "pass", "info hand: F");
  for (const auto& [name, unused] : burst_scenes) {
    std::printf(" | %-24s", (name + ": F (S)").c_str());
  }
  std::printf("\n");

  FILE* json = json_path.empty() ? nullptr : std::fopen(json_path.c_str(), "w");
  if (json != nullptr) {
    std::fprintf(json, "{\n  \"cadence\": {\"fps\": %.0f, \"interval_frames\": %d, \"lag_frames\": %d},\n  \"eps\": %.6f,\n"
                       "  \"configs\": [\n", kFps, kInterval, kLag, kEps);
  }
  const int warm = 30;  // skip the first second (fade-in)
  double f_none = 0.0;
  std::vector<double> burst_f_none(burst_scenes.size(), 0.0);
  std::string best;
  int best_stages = 99, best_r = 1 << 30;
  for (std::size_t ci = 0; ci < configs.size(); ++ci) {
    const Config& c = configs[ci];
    // 20 extra frames: the depth age stays under the 1 s stale limit, so the
    // effect never fades out inside a measurement.
    const Temporal st = Measure(Run(still, c, 20), warm);
    const Replay step_run = Run(step, c, 20);
    double change = 0.0;
    const int r = ResponseLag(step_run, 12, change);
    const double seam = SharpestStep(step_run.frames.back());
    const Replay cut_run = Run(cut, c, 20);
    const int g = GhostArea(cut_run, 12);
    const Temporal hand_t = Measure(Run(hand, c, 20), warm, 0, kGridH / 2);
    if (ci == 0) {
      f_none = st.flicker;
    }
    std::vector<Temporal> burst_t;
    for (std::size_t b = 0; b < burst_scenes.size(); ++b) {
      // From after the 0.25 s fade-in (frame kLag + 8).
      burst_t.push_back(Measure(Run(burst_scenes[b].second, c, 8), kLag + 10));
      if (ci == 0) {
        burst_f_none[b] = burst_t.back().flicker;
      }
    }
    bool pass = st.flicker <= std::max(0.5 * f_none, kEps) && st.stepping <= 1.5 && r >= 0 && r <= 6 && g == 0;
    for (std::size_t b = 0; b < burst_t.size(); ++b) {
      pass = pass && burst_t[b].flicker <= std::max(0.5 * burst_f_none[b], kEps) && burst_t[b].stepping <= 1.5;
    }
    char r_text[32];
    std::snprintf(r_text, sizeof(r_text), r < 0 ? "n/m" : "%d (chg %.3f)", r, change);
    std::printf("%-22s | %.5f (%5.2f; %.5f/%.5f) | %-16s | %-7d | %-5s | %.5f | seam %.3f ", c.name.c_str(),
                st.flicker, st.stepping, st.update_mean, st.other_mean, r_text, g, pass ? "yes" : "no", hand_t.flicker,
                seam);
    for (const Temporal& bt : burst_t) {
      std::printf(" | %.5f (%5.2f)           ", bt.flicker, bt.stepping);
    }
    std::printf("\n");
    if (pass && (c.stages() < best_stages || (c.stages() == best_stages && r < best_r))) {
      best = c.name;
      best_stages = c.stages();
      best_r = r;
    }
    if (json != nullptr) {
      std::fprintf(json,
                   "    {\"name\": \"%s\", \"align\": %s, \"ema\": %s, \"alpha\": %.2f, \"crossfade\": %s, "
                   "\"duration_factor\": %.2f, \"static\": {\"F\": %.6f, \"S\": %.3f, \"update_mean\": %.6f, "
                   "\"other_mean\": %.6f}, \"step_R_frames\": %d, \"step_change\": %.4f, \"cut_G\": %d, "
                   "\"scene_changes_static\": %d, \"info_hand_F_top_half\": %.6f, \"bursts\": [",
                   c.name.c_str(), c.align ? "true" : "false", c.ema ? "true" : "false", static_cast<double>(c.alpha),
                   c.crossfade ? "true" : "false", c.factor, st.flicker, st.stepping, st.update_mean, st.other_mean,
                   r, change, g, Run(still, c, 0).scene_changes, hand_t.flicker);
      for (std::size_t b = 0; b < burst_t.size(); ++b) {
        std::fprintf(json, "%s{\"name\": \"%s\", \"F\": %.6f, \"S\": %.3f, \"update_mean\": %.6f, \"other_mean\": %.6f}",
                     b == 0 ? "" : ", ", burst_scenes[b].first.c_str(), burst_t[b].flicker, burst_t[b].stepping,
                     burst_t[b].update_mean, burst_t[b].other_mean);
      }
      std::fprintf(json, "], \"pass\": %s}%s\n", pass ? "true" : "false", ci + 1 < configs.size() ? "," : "");
    }
  }
  std::printf("Thresholds: F <= max(0.5 x F(none), eps), S <= 1.5, R <= 6 frames, G = 0 (bursts: F and S)\n");
  const auto print_tilt = [](const char* name, const Tilt& t) {
    std::printf("Subject tilt (%s, relief %.2f): p50 %.1f deg, p90 %.1f deg over %zu texels\n", name,
                static_cast<double>(LightSettings{}.relief), t.p50_deg, t.p90_deg, t.texels);
  };
  print_tilt("synthetic person", SubjectTilt(still));
  for (const auto& [name, results] : burst_scenes) {
    print_tilt(name.c_str(), SubjectTilt(results));
  }
  const auto print_split = [](const char* name, const std::vector<Tensor>& results) {
    const GainSplit on = SplitGain(results, 1.0f);
    const GainSplit off = SplitGain(results, 0.0f);
    const GainSplit flat = SplitGain(results, 1.0f, 0.0f);
    std::printf("Gain split (%s, default light): subject-vs-background %.3f, within-subject spread %.3f "
                "(depth falloff 1) | %.3f, %.3f (falloff 0) | flat (relief 0): spread %.3f\n",
                name, on.separation, on.shape, off.separation, off.shape, flat.shape);
  };
  print_split("synthetic person", still);
  for (const auto& [name, results] : burst_scenes) {
    print_split(name.c_str(), results);
  }
  std::printf("Simplest passing configuration: %s\n", best.empty() ? "NONE (trade-off: see the table)" : best.c_str());
  if (json != nullptr) {
    std::fprintf(json, "  ],\n  \"simplest_passing\": \"%s\"\n}\n", best.c_str());
    std::fclose(json);
  }
  return 0;
}
