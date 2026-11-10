// Lighting math (CPU reference). Criteria fixed in advance (Phase 4 plan, M1):
//   - a plane faces the camera; a ramp nearer on the right faces left
//     (n.x < 0); a ramp nearer at the bottom faces up (n.y > 0);
//   - moving the light toward the side a surface faces brightens it, and the
//     brightest point of a sphere lies on the light's side in x and in y;
//   - neutral settings, or fade 0, reproduce every 8-bit input exactly;
//   - every combination of control bounds gives finite values, a gain within
//     [kMinGain, kMaxGain] and a non-negative highlight;
//   - constant and zero depth produce no NaN; a depth step lowers the
//     confidence at the edge and keeps the flat side's normal;
//   - the sRGB round trip of all 256 codes is exact.

#include "relume/lighting.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace relume {
namespace {

constexpr float kAspect = 16.0f / 9.0f;

struct Grid {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<float> depth;
  float& at(std::uint32_t x, std::uint32_t y) { return depth[std::size_t{y} * width + x]; }
};

template <typename F>
Grid MakeGrid(std::uint32_t width, std::uint32_t height, F f) {
  Grid g{width, height, std::vector<float>(std::size_t{width} * height)};
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      g.at(x, y) = f((static_cast<float>(x) + 0.5f) / static_cast<float>(width),
                     (static_cast<float>(y) + 0.5f) / static_cast<float>(height));
    }
  }
  return g;
}

std::vector<FieldTexel> Field(const Grid& g) {
  return BuildDepthField(g.depth.data(), g.width, g.height, g.width, 1.0f, 0.0f, kAspect);
}

Vec3 Normal(const FieldTexel& t) { return FieldNormal(t, 0.6f); }

const FieldTexel& At(const std::vector<FieldTexel>& f, const Grid& g, std::uint32_t x, std::uint32_t y) {
  return f[std::size_t{y} * g.width + x];
}

// A bump: a hemisphere of radius 0.35 (in v units) on a flat background.
Grid Sphere(std::uint32_t width, std::uint32_t height) {
  return MakeGrid(width, height, [](float u, float v) {
    const float dx = (u - 0.5f) * kAspect;
    const float dy = v - 0.5f;
    const float r2 = (dx * dx + dy * dy) / (0.35f * 0.35f);
    return r2 < 1.0f ? 0.3f + 0.7f * std::sqrt(1.0f - r2) : 0.3f;
  });
}

float MeanGain(const Shading& s) { return (s.gain.x + s.gain.y + s.gain.z) / 3.0f; }

TEST(LightingMath, PlaneFacesTheCamera) {
  const Grid g = MakeGrid(64, 36, [](float, float) { return 0.5f; });
  const std::vector<FieldTexel> f = Field(g);
  for (const FieldTexel& t : f) {
    EXPECT_FLOAT_EQ(t.sx, 0.0f);
    EXPECT_FLOAT_EQ(t.sy, 0.0f);
    EXPECT_FLOAT_EQ(Normal(t).z, 1.0f);
    EXPECT_FLOAT_EQ(t.conf, 1.0f);
    EXPECT_NEAR(t.h, 0.5f, 1e-6f);
  }
}

TEST(LightingMath, RampNearerOnTheRightFacesLeft) {
  const Grid g = MakeGrid(64, 36, [](float u, float) { return 0.2f + 0.5f * u; });
  const std::vector<FieldTexel> field = Field(g);
  const FieldTexel& t = At(field, g, 32, 18);
  EXPECT_LT(Normal(t).x, -0.05f);
  EXPECT_NEAR(Normal(t).y, 0.0f, 1e-6f);
  EXPECT_FLOAT_EQ(t.conf, 1.0f);
}

TEST(LightingMath, RampNearerAtTheBottomFacesUp) {
  const Grid g = MakeGrid(64, 36, [](float, float v) { return 0.2f + 0.5f * v; });
  const std::vector<FieldTexel> field = Field(g);
  const FieldTexel& t = At(field, g, 32, 18);
  EXPECT_GT(Normal(t).y, 0.05f);
  EXPECT_NEAR(Normal(t).x, 0.0f, 1e-6f);
}

TEST(LightingMath, LightTowardTheFacingSideBrightens) {
  const Grid g = MakeGrid(64, 36, [](float u, float) { return 0.2f + 0.5f * u; });  // faces left
  const std::vector<FieldTexel> field = Field(g);
  const FieldTexel& t = At(field, g, 32, 18);
  LightSettings left;
  left.x = 0.1f;
  left.y = 0.5f;
  LightSettings right = left;
  right.x = 0.9f;
  // The same distance from the pixel either way, so only the normal differs.
  EXPECT_GT(MeanGain(ComputeShading(t, 0.5f, 0.5f, left, kAspect)),
            MeanGain(ComputeShading(t, 0.5f, 0.5f, right, kAspect)) + 0.02f);
}

void ExpectBrightestOnLightSide(float light_x, float light_y) {
  const Grid g = Sphere(160, 90);
  const std::vector<FieldTexel> f = Field(g);
  LightSettings s;
  s.x = light_x;
  s.y = light_y;
  s.specular = 0.0f;
  s.intensity = 1.0f;
  s.ambient = 0.4f;
  float best = -1.0f;
  float best_u = 0.0f;
  float best_v = 0.0f;
  // Only inside the sphere: the background is a plane lit by distance alone.
  for (std::uint32_t y = 0; y < g.height; ++y) {
    for (std::uint32_t x = 0; x < g.width; ++x) {
      const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(g.width);
      const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(g.height);
      const float dx = (u - 0.5f) * kAspect;
      const float dy = v - 0.5f;
      if (dx * dx + dy * dy > 0.3f * 0.3f) {
        continue;
      }
      const float gain = MeanGain(ComputeShading(At(f, g, x, y), u, v, s, kAspect));
      if (gain > best) {
        best = gain;
        best_u = u;
        best_v = v;
      }
    }
  }
  if (light_x < 0.5f) {
    EXPECT_LT(best_u, 0.5f) << "light at x " << light_x;
  } else {
    EXPECT_GT(best_u, 0.5f) << "light at x " << light_x;
  }
  if (light_y < 0.5f) {
    EXPECT_LT(best_v, 0.5f) << "light at y " << light_y;
  } else {
    EXPECT_GT(best_v, 0.5f) << "light at y " << light_y;
  }
}

TEST(LightingMath, SphereBrightestPointFollowsTheLight) {
  ExpectBrightestOnLightSide(0.15f, 0.15f);  // upper left
  ExpectBrightestOnLightSide(0.85f, 0.15f);  // upper right
  ExpectBrightestOnLightSide(0.15f, 0.85f);  // lower left
  ExpectBrightestOnLightSide(0.85f, 0.85f);  // lower right
}

TEST(LightingMath, NeutralSettingsAndZeroFadeAreTheIdentity) {
  const Grid g = Sphere(64, 36);
  const std::vector<FieldTexel> f = Field(g);
  LightSettings strong;
  strong.intensity = 1.5f;
  strong.specular = 0.4f;
  strong.ambient = 0.2f;
  strong.tint = 1.0f;
  const LightSettings neutral = NeutralLightSettings();
  for (int code = 0; code < 256; ++code) {
    const auto c = static_cast<std::uint8_t>(code);
    // Each channel at this code, against a different code on the others.
    const std::array<std::array<std::uint8_t, 3>, 4> inputs = {{{c, c, c},
                                                                {c, 0, 255},
                                                                {255, c, 0},
                                                                {0, 255, c}}};
    for (const auto& rgb : inputs) {
      for (std::uint32_t i = 0; i < 6; ++i) {
        const std::uint32_t x = (i * 11 + 3) % g.width;
        const std::uint32_t y = (i * 7 + 2) % g.height;
        const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(g.width);
        const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(g.height);
        EXPECT_EQ(ShadePixel(rgb, At(f, g, x, y), u, v, neutral, kAspect, 1.0f), rgb) << "code " << code;
        EXPECT_EQ(ShadePixel(rgb, At(f, g, x, y), u, v, strong, kAspect, 0.0f), rgb) << "code " << code;
      }
    }
  }
}

TEST(LightingMath, EveryBoundCombinationStaysFiniteAndBounded) {
  Grid g = Sphere(48, 27);
  for (std::uint32_t y = 0; y < g.height; ++y) {
    for (std::uint32_t x = g.width * 3 / 4; x < g.width; ++x) {
      g.at(x, y) = 1.4f;  // a near object with a hard edge
    }
  }
  const LightSettings lo{0.0f, 0.0f, 0.2f, 0.0f, 0.2f, 0.0f, -1.0f, 0.0f};
  const LightSettings hi{1.0f, 1.0f, 4.0f, 1.5f, 1.2f, 0.4f, 1.0f, 1.5f};
  for (std::uint32_t mask = 0; mask < 256; ++mask) {
    const auto pick = [mask](int bit, float a, float b) { return (mask >> bit) & 1u ? b : a; };
    const LightSettings s{pick(0, lo.x, hi.x),         pick(1, lo.y, hi.y),           pick(2, lo.z, hi.z),
                          pick(3, lo.intensity, hi.intensity), pick(4, lo.ambient, hi.ambient),
                          pick(5, lo.specular, hi.specular),   pick(6, lo.tint, hi.tint),
                          pick(7, lo.relief, hi.relief)};
    ASSERT_EQ(s.Clamped(), s);
    const std::vector<FieldTexel> f = Field(g);
    for (std::uint32_t y = 0; y < g.height; y += 3) {
      for (std::uint32_t x = 0; x < g.width; x += 3) {
        const FieldTexel& t = At(f, g, x, y);
        ASSERT_TRUE(std::isfinite(t.sx) && std::isfinite(t.sy) && std::isfinite(t.h) && std::isfinite(t.conf));
        const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(g.width);
        const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(g.height);
        const Shading sh = ComputeShading(t, u, v, s, kAspect);
        for (const float gain : {sh.gain.x, sh.gain.y, sh.gain.z}) {
          ASSERT_TRUE(std::isfinite(gain));
          ASSERT_GE(gain, kMinGain);
          ASSERT_LE(gain, kMaxGain);
        }
        ASSERT_TRUE(std::isfinite(sh.spec));
        ASSERT_GE(sh.spec, 0.0f);
      }
    }
  }
}

TEST(LightingMath, ClampedLimitsEveryControl) {
  const LightSettings wild{-5.0f, 9.0f, 0.0f, 99.0f, -1.0f, 3.0f, 7.0f, -2.0f};
  const LightSettings c = wild.Clamped();
  EXPECT_EQ(c, (LightSettings{0.0f, 1.0f, 0.2f, 1.5f, 0.2f, 0.4f, 1.0f, 0.0f}));
}

TEST(LightingMath, ConstantAndZeroDepthHaveNoNaN) {
  for (const float value : {0.0f, 1e-30f, 3.0f}) {
    const Grid g = MakeGrid(16, 9, [value](float, float) { return value; });
    // A zero scale (unusable range) and a huge one.
    for (const float scale : {0.0f, 1e6f}) {
      const std::vector<FieldTexel> f =
          BuildDepthField(g.depth.data(), g.width, g.height, g.width, scale, 0.0f, kAspect);
      for (const FieldTexel& t : f) {
        ASSERT_TRUE(std::isfinite(t.sx) && std::isfinite(t.sy) && std::isfinite(t.h) && std::isfinite(t.conf));
        ASSERT_GE(t.h, kHeightMin);
        ASSERT_LE(t.h, kHeightMax);
        const Shading s = ComputeShading(t, 0.5f, 0.5f, LightSettings{}, kAspect);
        ASSERT_TRUE(std::isfinite(s.gain.x) && std::isfinite(s.spec));
      }
    }
  }
  // A 1x1 grid has no neighbours at all.
  const float one = 0.5f;
  const std::vector<FieldTexel> f = BuildDepthField(&one, 1, 1, 1, 1.0f, 0.0f, kAspect);
  EXPECT_FLOAT_EQ(f[0].sx, 0.0f);
  EXPECT_FLOAT_EQ(f[0].conf, 1.0f);
}

TEST(LightingMath, DepthStepLowersConfidenceAndKeepsTheFlatSide) {
  // Background 0.2 on the left, a near object 0.9 from column 32.
  const Grid g = MakeGrid(64, 36, [](float u, float) { return u < 0.5f ? 0.2f : 0.9f; });
  const std::vector<FieldTexel> f = Field(g);
  for (std::uint32_t x = 30; x <= 33; ++x) {
    const FieldTexel& t = At(f, g, x, 18);
    EXPECT_LT(t.conf, 0.05f) << "column " << x;
    // The flat side's slope is used: no steep normal at the step.
    EXPECT_LT(std::abs(Normal(t).x), 0.05f) << "column " << x;
  }
  EXPECT_FLOAT_EQ(At(f, g, 20, 18).conf, 1.0f);
  EXPECT_FLOAT_EQ(At(f, g, 44, 18).conf, 1.0f);
}

TEST(LightingMath, SrgbRoundTripIsExactForEveryCode) {
  for (int code = 0; code < 256; ++code) {
    const float back = LinearToSrgb(SrgbToLinear(static_cast<float>(code) / 255.0f));
    EXPECT_EQ(static_cast<int>(std::floor(255.0f * back + 0.5f)), code);
  }
}

TEST(LightingMath, TintKeepsLuminance) {
  for (const float t : {-1.0f, -0.4f, 0.0f, 0.5f, 1.0f}) {
    const Vec3 c = TintColor(t);
    EXPECT_NEAR(0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z, 1.0f, 1e-6f);
  }
  EXPECT_GT(TintColor(1.0f).x, TintColor(1.0f).z);   // warm: more red than blue
  EXPECT_LT(TintColor(-1.0f).x, TintColor(-1.0f).z);  // cool
}

TEST(LightingMath, NeutralTintGivesEqualChannelGains) {
  const Grid g = Sphere(64, 36);
  const std::vector<FieldTexel> f = Field(g);
  LightSettings s;
  s.tint = 0.0f;
  for (std::uint32_t x = 0; x < g.width; x += 7) {
    const Shading sh = ComputeShading(At(f, g, x, 18), 0.5f, 0.5f, s, kAspect);
    EXPECT_NEAR(sh.gain.x, sh.gain.y, 1e-6f);
    EXPECT_NEAR(sh.gain.y, sh.gain.z, 1e-6f);
  }
}

// Attenuation is normalized to the point straight under the light on the
// nearest plane: there a camera-facing surface gets ambient + intensity.
TEST(LightingMath, KeyStrengthIsNormalizedUnderTheLight) {
  LightSettings s;
  s.x = 0.5f;
  s.y = 0.5f;
  s.specular = 0.0f;
  s.relief = 0.0f;  // flat: the surface is at z = 0, the light at z = s.z
  const FieldTexel flat{0.0f, 0.0f, 0.0f, 1.0f};
  const Shading sh = ComputeShading(flat, 0.5f, 0.5f, s, kAspect);
  EXPECT_NEAR(sh.gain.y, s.ambient + s.intensity, 1e-5f);
  // Farther from the light: less key light, never below ambient.
  const Shading edge = ComputeShading(flat, 0.0f, 0.5f, s, kAspect);
  EXPECT_LT(edge.gain.y, sh.gain.y);
  EXPECT_GE(edge.gain.y, s.ambient);
}

// Open issue 1: with depth falloff 0, a depth step makes no step in the
// lighting on its flat sides (only the normals near the edge still matter,
// and their confidence is low there).
TEST(LightingMath, ZeroDepthFalloffRemovesTheStepAtADepthEdge) {
  const Grid g = MakeGrid(64, 36, [](float u, float) { return u < 0.5f ? 0.2f : 0.9f; });
  const std::vector<FieldTexel> f = Field(g);
  LightSettings s;
  s.specular = 0.0f;
  s.relief = 1.0f;
  const auto step = [&](float falloff) {
    s.depth_falloff = falloff;
    // Two flat points either side of the edge, 6 texels away, equally far
    // from a light centred on the edge.
    s.x = 0.5f;
    s.y = 0.5f;
    const float left = ComputeShading(At(f, g, 26, 18), 26.5f / 64.0f, 0.5f, s, kAspect).gain.y;
    const float right = ComputeShading(At(f, g, 37, 18), 37.5f / 64.0f, 0.5f, s, kAspect).gain.y;
    return std::abs(right - left);
  };
  const float with_depth = step(1.0f);
  const float without = step(0.0f);
  std::printf("gain step across a depth edge: falloff 1 %.4f, falloff 0 %.4f\n", static_cast<double>(with_depth),
              static_cast<double>(without));
  EXPECT_GT(with_depth, 0.05f) << "the step that shows as a seam when depth lags";
  EXPECT_LT(without, 1e-4f);
}

// Documents (does not fix) the transfer-curve approximation: the camera
// signal is BT.709-encoded and Relume decodes it with the sRGB curve.
TEST(LightingMath, RecordsBt709VersusSrgbDecodeDifference) {
  float worst = 0.0f;
  int worst_code = 0;
  for (int code = 0; code < 256; ++code) {
    const float v = static_cast<float>(code) / 255.0f;
    const float d = std::abs(SrgbToLinear(v) - Bt709ToLinear(v));
    if (d > worst) {
      worst = d;
      worst_code = code;
    }
  }
  std::printf("BT.709 vs sRGB decode: max |linear difference| %.4f at code %d\n", static_cast<double>(worst),
              worst_code);
  RecordProperty("bt709_vs_srgb_max_linear_diff", std::to_string(worst));
  EXPECT_LT(worst, 0.1f);  // sanity only: the two curves are close, not equal
}

}  // namespace
}  // namespace relume
