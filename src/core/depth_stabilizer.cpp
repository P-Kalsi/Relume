#include "relume/depth_stabilizer.hpp"

#include <algorithm>
#include <cmath>

namespace relume {

namespace {

float Smoothstep(float edge0, float edge1, float x) {
  const float t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

}  // namespace

void DepthStabilizer::SetConfig(const StabilizerConfig& config) {
  if (!(config == config_)) {
    config_ = config;
    config_changed_ = true;
  }
}

StabilizerReport DepthStabilizer::Process(const float* depth, const ResizeTransform& t, float lo, float hi,
                                          bool range_valid, std::uint64_t camera_generation, float* out) {
  StabilizerReport report;
  std::fill(out, out + kNetDepthFloats, 0.0f);
  const std::size_t w = t.new_w;
  const std::size_t h = t.new_h;
  const std::size_t n = w * h;
  const bool region_ok = n > 0 && t.pad_x + t.new_w <= kNetSize && t.pad_y + t.new_h <= kNetSize;
  if (!region_ok || !range_valid || !(hi > lo) || !std::isfinite(lo) || !std::isfinite(hi)) {
    report.bypassed = true;
    report.reset = true;
    has_history_ = false;
    return report;
  }
  if (p_.size() != n) {  // allocates only when the region size changes
    p_.resize(n);
    prev_.resize(n);
    const std::size_t max_samples = ((w + 1) / 2) * ((h + 1) / 2);
    fit_x_.reserve(max_samples);
    fit_y_.reserve(max_samples);
    fit_w_.reserve(max_samples);
    residuals_.reserve(max_samples);
  }
  const float inv = 1.0f / (hi - lo);
  for (std::size_t y = 0; y < h; ++y) {
    const float* row = depth + (t.pad_y + y) * kNetSize + t.pad_x;
    for (std::size_t x = 0; x < w; ++x) {
      const float v = (row[x] - lo) * inv;
      if (!std::isfinite(v)) {
        report.bypassed = true;
        report.reset = true;
        has_history_ = false;
        return report;
      }
      p_[y * w + x] = v;
    }
  }

  const bool restart =
      !has_history_ || config_changed_ || camera_generation != generation_ || !(t == transform_);
  float a = 1.0f;
  float b = 0.0f;
  bool scene_change = false;
  if (!restart && config_.align) {
    if (Fit(report, a, b)) {
      a += (1.0f - a) * config_.anchor;
      b *= 1.0f - config_.anchor;
    } else {
      scene_change = true;
    }
  }
  const auto write = [&](std::size_t i, float v) {
    prev_[i] = v;
    out[(t.pad_y + i / w) * kNetSize + t.pad_x + i % w] = v;
  };
  if (restart || scene_change) {
    for (std::size_t i = 0; i < n; ++i) {
      write(i, p_[i]);
    }
    report.reset = true;
    report.scene_change = scene_change;
    has_history_ = true;
    config_changed_ = false;
    generation_ = camera_generation;
    transform_ = t;
    return report;
  }
  report.a = a;
  report.b = b;
  for (std::size_t i = 0; i < n; ++i) {
    const float aligned = a * p_[i] + b;
    float s = aligned;
    if (config_.ema) {
      const float diff = aligned - prev_[i];
      const float weight =
          config_.alpha + (1.0f - config_.alpha) * Smoothstep(config_.delta0, config_.delta1, std::abs(diff));
      s = prev_[i] + weight * diff;
    }
    write(i, s);
  }
  return report;
}

// Robust affine fit of the previous output (y) on this result (x) over
// low-gradient pixels, sampled every second texel. Returns false if the fit
// cannot be trusted (see the header).
bool DepthStabilizer::Fit(StabilizerReport& report, float& a, float& b) {
  const std::size_t w = transform_.new_w;
  const std::size_t h = transform_.new_h;
  fit_x_.clear();
  fit_y_.clear();
  std::size_t sampled = 0;
  const auto grad = [w](const std::vector<float>& m, std::size_t x, std::size_t y) {
    const float gx = std::abs(m[y * w + x + 1] - m[y * w + x - 1]) * 0.5f;
    const float gy = std::abs(m[(y + 1) * w + x] - m[(y - 1) * w + x]) * 0.5f;
    return std::max(gx, gy);
  };
  for (std::size_t y = 1; y + 1 < h; y += 2) {
    for (std::size_t x = 1; x + 1 < w; x += 2) {
      ++sampled;
      if (grad(p_, x, y) <= config_.max_gradient && grad(prev_, x, y) <= config_.max_gradient) {
        fit_x_.push_back(p_[y * w + x]);
        fit_y_.push_back(prev_[y * w + x]);
      }
    }
  }
  const std::size_t count = fit_x_.size();
  report.valid_fraction = sampled == 0 ? 0.0f : static_cast<float>(count) / static_cast<float>(sampled);
  if (count < 16 || report.valid_fraction < config_.min_valid_fraction) {
    return false;
  }

  fit_w_.assign(count, 1.0f);
  residuals_.resize(count);
  // Least squares, two Huber reweightings, then two Tukey biweight passes
  // (zero weight beyond the inlier threshold), so that a near object or a
  // moved hand does not bias the fit.
  for (int pass = 0; pass < 5; ++pass) {
    double sw = 0, sx = 0, sy = 0;
    for (std::size_t i = 0; i < count; ++i) {
      sw += fit_w_[i];
      sx += fit_w_[i] * fit_x_[i];
      sy += fit_w_[i] * fit_y_[i];
    }
    const double mx = sx / sw;
    const double my = sy / sw;
    double vxx = 0, vyy = 0, vxy = 0;
    for (std::size_t i = 0; i < count; ++i) {
      const double dx = fit_x_[i] - mx;
      const double dy = fit_y_[i] - my;
      vxx += fit_w_[i] * dx * dx;
      vyy += fit_w_[i] * dy * dy;
      vxy += fit_w_[i] * dx * dy;
    }
    vxx /= sw;
    vyy /= sw;
    vxy /= sw;
    if (vxx < 1e-6 || vyy < 1e-6) {
      return false;  // nearly constant: the slope is not determined
    }
    a = static_cast<float>(vxy / vxx);
    b = static_cast<float>(my - vxy / vxx * mx);
    const bool tukey = pass >= 2;
    double total = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
      const float r = std::abs(a * fit_x_[i] + b - fit_y_[i]);
      residuals_[i] = r;
      if (tukey) {
        const float q = r / config_.inlier_threshold;
        fit_w_[i] = q < 1.0f ? (1.0f - q * q) * (1.0f - q * q) : 0.0f;
      } else {
        fit_w_[i] = r <= config_.huber_k ? 1.0f : config_.huber_k / r;
      }
      total += fit_w_[i];
    }
    if (!(total > 0.0)) {
      return false;  // nothing is close to the fit
    }
  }
  report.fitted = true;
  report.a = a;  // the raw fit; Process replaces it with the applied one
  report.b = b;
  std::size_t inliers = 0;
  for (const float r : residuals_) {
    inliers += r <= config_.inlier_threshold ? 1 : 0;
  }
  report.inlier_fraction = static_cast<float>(inliers) / static_cast<float>(count);
  const auto mid = residuals_.begin() + static_cast<std::ptrdiff_t>(count / 2);
  std::nth_element(residuals_.begin(), mid, residuals_.end());
  report.median_residual = *mid;
  return report.inlier_fraction >= config_.min_inlier_fraction &&
         report.median_residual <= config_.max_median_residual && a >= config_.min_scale && a <= config_.max_scale;
}

}  // namespace relume
