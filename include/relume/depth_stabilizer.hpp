#pragma once

// Minimal temporal stabilization of relative depth (Phase 4), run on the
// depth worker after each model run. Portable C++.
//
// Input: the raw 448x448 depth (relative, larger = nearer; scale and shift
// vary from run to run) and its 2nd-98th percentile range. Output: a
// normalized map (about 0..1 over the valid region; padding 0) in the same
// tensor layout. The raw depth is never modified.
//
//   p  = (depth - lo) / (hi - lo)                    per-result percentile normalization
//   Stage A (align): fit a, b so that a * p + b matches the previous output
//        on low-gradient pixels (least squares, two Huber and two Tukey
//        biweight reweightings), then pull the fit
//        toward identity by `anchor` (a leak, so fit errors decay instead of
//        accumulating into drift): (a, b) = mix(fit, (1, 0), anchor).
//   Stage B (EMA): s = prev + w * (aligned - prev), with
//        w = mix(alpha, 1, smoothstep(delta0, delta1, |aligned - prev|)):
//        small changes (the model's jitter) are damped, large ones (motion,
//        a hand, a new scene) pass through at once.
//
// The history is reset (output = p, `reset` set) for the first result, a new
// camera generation, a new transform, or a configuration change. A fit that
// cannot be trusted also resets and sets `scene_change`, so the renderer
// hard-cuts instead of blending unrelated depth:
//   - fewer than min_valid_fraction of the valid region pass the low-gradient
//     mask, or the weighted variance of p or of the previous output is below
//     1e-6 (a near-zero denominator);
//   - after the fit: inlier fraction below min_inlier_fraction, median
//     absolute residual above max_median_residual, or a outside
//     [min_scale, max_scale].
// An unusable range (not valid) bypasses everything: the output is zeros,
// `bypassed` is set and the history is reset.
//
// Not thread-safe: one instance per worker thread. Steady state does not
// allocate (buffers are sized on the first result of each transform).

#include "relume/preprocess.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace relume {

struct StabilizerConfig {
  bool align = true;   // stage A
  bool ema = true;     // stage B
  float alpha = 0.5f;  // EMA weight of a new value for small changes
  float delta0 = 0.03f;  // |change| (normalized) where the EMA starts opening
  float delta1 = 0.12f;  // |change| where it passes the new value unchanged
  float anchor = 0.02f;  // leak of the fit toward identity, per result
  float huber_k = 0.05f;
  float inlier_threshold = 0.1f;  // |residual| for the inlier fraction
  float max_gradient = 0.02f;     // low-gradient mask: |dp| per texel
  float min_valid_fraction = 0.15f;
  float min_inlier_fraction = 0.6f;
  float max_median_residual = 0.08f;
  float min_scale = 0.8f;
  float max_scale = 1.25f;
  bool operator==(const StabilizerConfig&) const = default;
};

// Why stage A rejected a fit (the first check that failed).
enum class FitRejection : std::uint8_t {
  kNone,
  kTooFewPixels,   // below min_valid_fraction of low-gradient pixels
  kFlat,           // near-zero variance (the slope is not determined)
  kNoInliers,      // nothing is close to the fit
  kFewInliers,     // below min_inlier_fraction
  kHighResidual,   // median residual above max_median_residual
  kScale,          // a outside [min_scale, max_scale]
};
inline constexpr std::size_t kFitRejectionCount = 7;
const char* ToString(FitRejection rejection);

struct StabilizerReport {
  bool bypassed = false;      // unusable input: no output
  bool reset = false;         // the history restarted (output = p)
  bool scene_change = false;  // reset because the fit could not be trusted
  bool fitted = false;        // stage A produced a fit (even if rejected)
  float a = 1.0f;             // applied affine on p (after the anchor)
  float b = 0.0f;
  float valid_fraction = 0.0f;   // of the region, in the low-gradient mask
  float inlier_fraction = 0.0f;
  float median_residual = 0.0f;
  FitRejection rejection = FitRejection::kNone;  // set when scene_change is
};

class DepthStabilizer {
 public:
  explicit DepthStabilizer(StabilizerConfig config = {}) : config_(config) {}

  const StabilizerConfig& config() const { return config_; }
  // A different configuration resets the history at the next Process.
  void SetConfig(const StabilizerConfig& config);
  void Reset() { has_history_ = false; }

  // `out` has kNetDepthFloats floats (tensor layout); `camera_generation`
  // identifies the camera session.
  StabilizerReport Process(const float* depth, const ResizeTransform& transform, float lo, float hi, bool range_valid,
                           std::uint64_t camera_generation, float* out);

 private:
  bool Fit(StabilizerReport& report, float& a, float& b);

  StabilizerConfig config_;
  bool config_changed_ = false;
  bool has_history_ = false;
  std::uint64_t generation_ = 0;
  ResizeTransform transform_;
  std::vector<float> p_;     // this result, normalized: new_h x new_w
  std::vector<float> prev_;  // the previous output: new_h x new_w
  std::vector<float> fit_x_, fit_y_, fit_w_, residuals_;  // fit samples (reused)
};

}  // namespace relume
