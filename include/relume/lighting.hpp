#pragma once

// Depth-guided relighting math: the CPU reference that shaders/normals.wgsl
// and shaders/relight.wgsl repeat step for step (tests compare them).
// Portable C++, no GPU dependency.
//
// Coordinate convention (one convention, used everywhere):
//   image space   u, v in [0, 1] across the camera image (its aspect-fit
//                 rectangle in the output); (0, 0) is the top left.
//   lighting space, right-handed, camera facing:
//                 x = (u - 0.5) * 2 * aspect   (right is +x; aspect = width / height)
//                 y = (0.5 - v) * 2            (up is +y: image y is flipped here, once)
//                 z = relief * h               (toward the viewer)
//   where h is normalized depth, larger = nearer (Depth Anything V2 is
//   disparity-like). The depth is treated as a height field, not unprojected:
//   it is relative and its scale and shift are unknown, so `relief` is a
//   tunable heuristic, not a measurement.
//   Normal: n = normalize(-dz/dx, -dz/dy, 1), so n.z > 0 always. A ramp that
//   is nearer on the right faces left (n.x < 0); one nearer at the bottom
//   faces up (n.y > 0).
//
// Shading. The camera pixel already contains the room's light, so Relume
// does not estimate albedo: it applies a bounded gain in linear light.
//   lin  = SrgbToLinear(camera)
//   n'   = normalize(mix((0,0,1), n, conf))       low confidence (depth edges) -> camera facing
//   diff = max((dot(n', l) + kWrap) / (1 + kWrap), 0)
//   att  = min(Attenuation(|L - P|) / Attenuation(light z above the plane), kMaxAttenuation)
//   gain = clamp(ambient + intensity * diff * att * tint, kMinGain, kMaxGain)
//   spec = specular * pow(max(dot(n', hv), 0), kShininess) * att * conf * luma(lin)
//   g    = mix(1, gain, fade)                      per channel
//   out  = LinearToSrgb(clamp(Boost(lin, g) + spec * fade, 0, 1))
// Boost(x, g) is x * g for g <= 1 and x * g / (1 + (g - 1) * x) above, so a
// brightened highlight approaches white instead of clipping. With fade = 0,
// or with neutral settings (ambient 1, intensity 0, specular 0), the result
// is exactly the input; tests check every 8-bit code. No cast shadows, no
// occlusion and no catchlights are modelled.

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace relume {

struct Vec3 {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

// Engineering controls (Phase 4). Values outside the ranges below are
// clamped by Clamped(); the shaders receive clamped values only.
struct LightSettings {
  float x = 0.25f;          // light position across the image, u in [0, 1] (0 = left)
  float y = 0.25f;          // v in [0, 1] (0 = top)
  float z = 1.2f;           // [0.2, 4]: height of the light above the nearest depth, lighting units
  float intensity = 0.7f;   // [0, 1.5]: diffuse key strength
  float ambient = 0.6f;     // [0.2, 1.2]: gain where the key does not reach
  float specular = 0.08f;   // [0, 0.4]: low-strength highlight
  float tint = 0.0f;        // [-1, 1]: cool (-1) to warm (+1), an artistic colour, not a temperature
  float relief = 0.6f;      // [0, 1.5]: height of the depth range in lighting units (heuristic)

  LightSettings Clamped() const;
  bool operator==(const LightSettings&) const = default;
};

// Settings for which relighting is the identity (gain 1, no highlight).
LightSettings NeutralLightSettings();

inline constexpr float kWrap = 0.3f;            // wrapped Lambert: shadows never reach zero
inline constexpr float kShininess = 24.0f;
inline constexpr float kAttenuationK = 0.25f;   // Attenuation(d) = 1 / (1 + k d^2)
inline constexpr float kMaxAttenuation = 2.0f;
inline constexpr float kMinGain = 0.35f;        // never totally black shadows
inline constexpr float kMaxGain = 1.8f;         // never neon highlights

// Depth-field construction (normals pass), in normalized depth units.
inline constexpr int kBilateralRadius = 2;             // 5x5
inline constexpr float kBilateralSigmaSpatial = 1.5f;  // texels
inline constexpr float kEdgeThreshold = 0.06f;         // |dh| over kGradientStep texels: a discontinuity
inline constexpr float kBilateralSigmaRange = kEdgeThreshold;
inline constexpr int kGradientStep = 2;                // texels: coarse-scale shape, not fine noise
inline constexpr float kMaxGradient = 3.0f;            // |dz/dx|, |dz/dy| clamp (lighting units)
inline constexpr float kHeightMin = -0.5f;             // normalized depth is clamped to this range
inline constexpr float kHeightMax = 1.5f;

// sRGB transfer functions (IEC 61966-2-1), in float.
float SrgbToLinear(float encoded);
float LinearToSrgb(float linear);
// BT.709 inverse OETF, for documenting the approximation only: the camera
// signal is BT.709-encoded, and Relume decodes it with the sRGB curve.
float Bt709ToLinear(float encoded);

// The warm/cool tint as an RGB multiplier with Rec. 709 luminance 1.
Vec3 TintColor(float tint);

// One texel of the depth field: normal x and y (z = sqrt(1 - x^2 - y^2)),
// smoothed height h, and confidence in [0, 1] (low at depth discontinuities).
struct FieldTexel {
  float nx = 0.0f;
  float ny = 0.0f;
  float h = 0.0f;
  float conf = 0.0f;
};

// Builds the depth field from normalized depth h = clamp(depth * scale +
// offset, kHeightMin, kHeightMax) on a width x height grid that covers the
// camera image (aspect = image width / height): a 5x5 bilateral filter, then
// gradients over kGradientStep texels with one-sided differences at
// discontinuities, then the normal and confidence. Edges are clamped.
std::vector<FieldTexel> BuildDepthField(const float* depth, std::uint32_t width, std::uint32_t height,
                                        std::size_t stride, float scale, float offset, float aspect, float relief);

// Lighting-space position of image point (u, v) at height h.
Vec3 LightingPosition(float u, float v, float h, float aspect, float relief);
// The light's lighting-space position.
Vec3 LightPosition(const LightSettings& settings, float aspect);

// Linear-light per-channel gain and additive highlight for one pixel, before
// `fade` (see the header comment).
struct Shading {
  Vec3 gain{1.0f, 1.0f, 1.0f};
  float spec = 0.0f;  // multiplied by the pixel's luma in ShadePixel
};
Shading ComputeShading(const FieldTexel& field, float u, float v, const LightSettings& settings, float aspect);

// The full per-pixel result on 8-bit sRGB values.
std::array<std::uint8_t, 3> ShadePixel(const std::array<std::uint8_t, 3>& rgb, const FieldTexel& field, float u,
                                       float v, const LightSettings& settings, float aspect, float fade);

}  // namespace relume
