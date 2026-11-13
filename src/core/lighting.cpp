#include "relume/lighting.hpp"

#include <algorithm>
#include <cmath>

namespace relume {

namespace {

float Smoothstep(float edge0, float edge1, float x) {
  const float t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

float Dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

Vec3 Normalize(Vec3 v) {
  const float length = std::sqrt(Dot(v, v));
  if (!(length > 1e-12f)) {
    return {0.0f, 0.0f, 1.0f};
  }
  return {v.x / length, v.y / length, v.z / length};
}

float Attenuation(float distance) { return 1.0f / (1.0f + kAttenuationK * distance * distance); }

// x * g, but a brightened value approaches 1 instead of clipping. Identity at
// g = 1; continuous in g.
float Boost(float x, float g) {
  if (g <= 1.0f) {
    return x * g;
  }
  return x * g / (1.0f + (g - 1.0f) * x);
}

std::uint8_t Quantize(float v) {
  return static_cast<std::uint8_t>(std::floor(255.0f * std::clamp(v, 0.0f, 1.0f) + 0.5f));
}

}  // namespace

LightSettings LightSettings::Clamped() const {
  LightSettings s = *this;
  s.x = std::clamp(x, 0.0f, 1.0f);
  s.y = std::clamp(y, 0.0f, 1.0f);
  s.z = std::clamp(z, 0.2f, 4.0f);
  s.intensity = std::clamp(intensity, 0.0f, 1.5f);
  s.ambient = std::clamp(ambient, 0.2f, 1.2f);
  s.specular = std::clamp(specular, 0.0f, 0.4f);
  s.tint = std::clamp(tint, -1.0f, 1.0f);
  s.relief = std::clamp(relief, 0.0f, 1.5f);
  s.depth_falloff = std::clamp(depth_falloff, 0.0f, 1.0f);
  return s;
}

LightSettings NeutralLightSettings() {
  LightSettings s;
  s.intensity = 0.0f;
  s.ambient = 1.0f;
  s.specular = 0.0f;
  return s;
}

float SrgbToLinear(float c) { return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f); }

float LinearToSrgb(float l) { return l <= 0.0031308f ? l * 12.92f : 1.055f * std::pow(l, 1.0f / 2.4f) - 0.055f; }

float Bt709ToLinear(float v) { return v < 0.081f ? v / 4.5f : std::pow((v + 0.099f) / 1.099f, 1.0f / 0.45f); }

Vec3 TintColor(float tint) {
  const float t = std::clamp(tint, -1.0f, 1.0f);
  const Vec3 c{1.0f + 0.3f * t, 1.0f, 1.0f - 0.3f * t};
  const float luma = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
  return {c.x / luma, c.y / luma, c.z / luma};
}

std::vector<FieldTexel> BuildDepthField(const float* depth, std::uint32_t width, std::uint32_t height,
                                        std::size_t stride, float scale, float offset, float aspect) {
  const int w = static_cast<int>(width);
  const int h = static_cast<int>(height);
  const auto at = [](const std::vector<float>& v, int x, int y, int row) {
    return v[static_cast<std::size_t>(y) * static_cast<std::size_t>(row) + static_cast<std::size_t>(x)];
  };
  std::vector<float> height_map(std::size_t{width} * height);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const float d = depth[static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(x)];
      height_map[static_cast<std::size_t>(y * w + x)] = std::clamp(d * scale + offset, kHeightMin, kHeightMax);
    }
  }

  // Bilateral filter: neighbours across a depth discontinuity get little weight.
  std::vector<float> smooth(height_map.size());
  const float inv_s = 1.0f / (2.0f * kBilateralSigmaSpatial * kBilateralSigmaSpatial);
  const float inv_r = 1.0f / (2.0f * kBilateralSigmaRange * kBilateralSigmaRange);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const float centre = at(height_map, x, y, w);
      float sum = 0.0f;
      float weight = 0.0f;
      for (int dy = -kBilateralRadius; dy <= kBilateralRadius; ++dy) {
        for (int dx = -kBilateralRadius; dx <= kBilateralRadius; ++dx) {
          const float sample = at(height_map, std::clamp(x + dx, 0, w - 1), std::clamp(y + dy, 0, h - 1), w);
          const float diff = sample - centre;
          const auto d2 = static_cast<float>(dx * dx + dy * dy);
          const float wgt = std::exp(-d2 * inv_s) * std::exp(-diff * diff * inv_r);
          sum += wgt * sample;
          weight += wgt;
        }
      }
      // A weighted mean stays in range; the clamp removes float rounding.
      smooth[static_cast<std::size_t>(y * w + x)] = std::clamp(sum / weight, kHeightMin, kHeightMax);
    }
  }

  // Slope of h per texel along one axis at index i of n, from the two sides
  // kGradientStep away (clamped). A side whose change, scaled to
  // kGradientStep texels, exceeds the edge threshold is a discontinuity: the
  // smaller side is used alone. `edge` returns the larger scaled change.
  const auto slope = [](float centre, float before, int d_before, float after, int d_after, float& edge) {
    const bool has_b = d_before > 0;
    const bool has_a = d_after > 0;
    const float step = static_cast<float>(kGradientStep);
    const float e_b = has_b ? std::abs(centre - before) * step / static_cast<float>(d_before) : 0.0f;
    const float e_a = has_a ? std::abs(after - centre) * step / static_cast<float>(d_after) : 0.0f;
    edge = std::max(e_b, e_a);
    const float s_b = has_b ? (centre - before) / static_cast<float>(d_before) : 0.0f;
    const float s_a = has_a ? (after - centre) / static_cast<float>(d_after) : 0.0f;
    if (has_b && has_a) {
      if (edge <= kEdgeThreshold) {
        return 0.5f * (s_b + s_a);
      }
      return e_b <= e_a ? s_b : s_a;
    }
    return has_b ? s_b : s_a;
  };

  const float x_unit = 2.0f * aspect / static_cast<float>(width);   // lighting units per texel, +x right
  const float y_unit = -2.0f / static_cast<float>(height);          // texel rows go down, +y is up
  std::vector<FieldTexel> field(height_map.size());
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const float c = at(smooth, x, y, w);
      const int x0 = std::max(x - kGradientStep, 0);
      const int x1 = std::min(x + kGradientStep, w - 1);
      const int y0 = std::max(y - kGradientStep, 0);
      const int y1 = std::min(y + kGradientStep, h - 1);
      float edge_x = 0.0f;
      float edge_y = 0.0f;
      const float sx = slope(c, at(smooth, x0, y, w), x - x0, at(smooth, x1, y, w), x1 - x, edge_x);
      const float sy = slope(c, at(smooth, x, y0, w), y - y0, at(smooth, x, y1, w), y1 - y, edge_y);
      FieldTexel& t = field[static_cast<std::size_t>(y * w + x)];
      t.sx = std::clamp(sx / x_unit, -kMaxSlope, kMaxSlope);
      t.sy = std::clamp(sy / y_unit, -kMaxSlope, kMaxSlope);
      t.h = c;
      const float local = 1.0f - Smoothstep(kEdgeThreshold, 2.0f * kEdgeThreshold, std::max(edge_x, edge_y));
      float lo = at(height_map, x, y, w);
      float hi = lo;
      for (int d = 2; d <= kSilhouetteRadius; d += 2) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) {
              continue;
            }
            const float v = at(height_map, std::clamp(x + d * dx, 0, w - 1), std::clamp(y + d * dy, 0, h - 1), w);
            lo = std::min(lo, v);
            hi = std::max(hi, v);
          }
        }
      }
      const float wide = 1.0f - Smoothstep(kSilhouetteThreshold, 2.0f * kSilhouetteThreshold, hi - lo);
      t.conf = std::min(local, wide);
    }
  }
  return field;
}

Vec3 FieldNormal(const FieldTexel& field, float relief) {
  const float conf = std::clamp(field.conf, 0.0f, 1.0f);
  const float gx = conf * std::clamp(relief * field.sx, -kMaxGradient, kMaxGradient);
  const float gy = conf * std::clamp(relief * field.sy, -kMaxGradient, kMaxGradient);
  return Normalize({-gx, -gy, 1.0f});
}

Vec3 LightingPosition(float u, float v, float h, float aspect, float relief) {
  return {(u - 0.5f) * 2.0f * aspect, (0.5f - v) * 2.0f, relief * h};
}

Vec3 LightPosition(const LightSettings& s, float aspect) {
  return {(s.x - 0.5f) * 2.0f * aspect, (0.5f - s.y) * 2.0f, s.relief + s.z};
}

Shading ComputeShading(const FieldTexel& field, float u, float v, const LightSettings& s, float aspect) {
  const float conf = std::clamp(field.conf, 0.0f, 1.0f);
  const Vec3 n = FieldNormal(field, s.relief);
  const Vec3 p = LightingPosition(u, v, field.h * s.depth_falloff, aspect, s.relief);
  const Vec3 light = LightPosition(s, aspect);
  const Vec3 to_light{light.x - p.x, light.y - p.y, light.z - p.z};
  const float distance = std::sqrt(Dot(to_light, to_light));
  const Vec3 l = Normalize(to_light);
  const float diff = std::max((Dot(n, l) + kWrap) / (1.0f + kWrap), 0.0f);
  const float att = std::min(Attenuation(distance) / Attenuation(s.z), kMaxAttenuation);
  const Vec3 tint = TintColor(s.tint);
  const float key = s.intensity * diff * att;
  Shading out;
  out.gain = {std::clamp(s.ambient + key * tint.x, kMinGain, kMaxGain),
              std::clamp(s.ambient + key * tint.y, kMinGain, kMaxGain),
              std::clamp(s.ambient + key * tint.z, kMinGain, kMaxGain)};
  const Vec3 half = Normalize({l.x, l.y, l.z + 1.0f});
  out.spec = s.specular * std::pow(std::max(Dot(n, half), 0.0f), kShininess) * att * conf;
  return out;
}

std::array<std::uint8_t, 3> ShadePixel(const std::array<std::uint8_t, 3>& rgb, const FieldTexel& field, float u,
                                       float v, const LightSettings& settings, float aspect, float fade) {
  const Shading shading = ComputeShading(field, u, v, settings, aspect);
  const float lin[3] = {SrgbToLinear(static_cast<float>(rgb[0]) / 255.0f),
                        SrgbToLinear(static_cast<float>(rgb[1]) / 255.0f),
                        SrgbToLinear(static_cast<float>(rgb[2]) / 255.0f)};
  const float luma = 0.2126f * lin[0] + 0.7152f * lin[1] + 0.0722f * lin[2];
  const float gain[3] = {shading.gain.x, shading.gain.y, shading.gain.z};
  std::array<std::uint8_t, 3> out{};
  for (int c = 0; c < 3; ++c) {
    const float g = 1.0f + (gain[c] - 1.0f) * fade;
    const float value = std::clamp(Boost(lin[c], g) + shading.spec * luma * fade, 0.0f, 1.0f);
    out[static_cast<std::size_t>(c)] = Quantize(LinearToSrgb(value));
  }
  return out;
}

}  // namespace relume
