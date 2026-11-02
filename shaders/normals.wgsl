// Depth field: height slopes, smoothed height and confidence, one texel per depth
// texel (the valid region of the tensor only, so padding is never read).
// Repeats BuildDepthField in src/core/lighting.cpp step for step; the
// constants below must match include/relume/lighting.hpp (tests compare the
// two). Output (RGBA16Float): slope dh/dx, slope dh/dy (per lighting unit,
// y up), height h, confidence. Relief is applied when shading, so the field
// does not depend on it.
//
//   h      = clamp(depth * scale + offset, kHeightMin, kHeightMax)
//   smooth = 5x5 bilateral filter of h
//   slope  = per axis, over kGradientStep texels each side (clamped at the
//            border); one-sided (the smaller side) where a side exceeds the
//            edge threshold
//   out    = clamp(slope / units, -kMaxSlope, kMaxSlope)
//   conf   = 1 - smoothstep(t, 2t, largest side change)

const kRadius : i32 = 2;
const kSigmaSpatial : f32 = 1.5;
const kEdge : f32 = 0.06;
const kSigmaRange : f32 = 0.06;
const kStep : i32 = 2;
const kMaxSlope : f32 = 8.0;
const kHeightMin : f32 = -0.5;
const kHeightMax : f32 = 1.5;

struct Params {
  size : vec2f,          // field size in texels (the depth texture's size)
  scale_offset : vec2f,  // normalization: h = depth * scale + offset
  units : vec2f,         // lighting units per texel: x = 2 * aspect / width, y = -2 / height
};

@group(0) @binding(0) var depth_tex : texture_2d<f32>;
@group(0) @binding(1) var<uniform> params : Params;

@vertex
fn vs_main(@builtin(vertex_index) index : u32) -> @builtin(position) vec4f {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(corners[index], 0.0, 1.0);
}

fn height_at(p : vec2i) -> f32 {
  let last = vec2i(params.size) - vec2i(1);
  let d = textureLoad(depth_tex, clamp(p, vec2i(0), last), 0).r;
  return clamp(d * params.scale_offset.x + params.scale_offset.y, kHeightMin, kHeightMax);
}

fn smooth_at(p : vec2i) -> f32 {
  let last = vec2i(params.size) - vec2i(1);
  let q = clamp(p, vec2i(0), last);
  let centre = height_at(q);
  let inv_s = 1.0 / (2.0 * kSigmaSpatial * kSigmaSpatial);
  let inv_r = 1.0 / (2.0 * kSigmaRange * kSigmaRange);
  var sum = 0.0;
  var weight = 0.0;
  for (var dy = -kRadius; dy <= kRadius; dy++) {
    for (var dx = -kRadius; dx <= kRadius; dx++) {
      let sample = height_at(q + vec2i(dx, dy));
      let diff = sample - centre;
      let d2 = f32(dx * dx + dy * dy);
      let w = exp(-d2 * inv_s) * exp(-diff * diff * inv_r);
      sum += w * sample;
      weight += w;
    }
  }
  return clamp(sum / weight, kHeightMin, kHeightMax);
}

struct Slope {
  value : f32,
  edge : f32,
};

fn slope(centre : f32, before : f32, d_before : i32, after : f32, d_after : i32) -> Slope {
  let has_b = d_before > 0;
  let has_a = d_after > 0;
  let step = f32(kStep);
  var e_b = 0.0;
  var e_a = 0.0;
  var s_b = 0.0;
  var s_a = 0.0;
  if (has_b) {
    e_b = abs(centre - before) * step / f32(d_before);
    s_b = (centre - before) / f32(d_before);
  }
  if (has_a) {
    e_a = abs(after - centre) * step / f32(d_after);
    s_a = (after - centre) / f32(d_after);
  }
  var out : Slope;
  out.edge = max(e_b, e_a);
  if (has_b && has_a) {
    if (out.edge <= kEdge) {
      out.value = 0.5 * (s_b + s_a);
    } else if (e_b <= e_a) {
      out.value = s_b;
    } else {
      out.value = s_a;
    }
  } else if (has_b) {
    out.value = s_b;
  } else {
    out.value = s_a;
  }
  return out;
}

@fragment
fn fs_main(@builtin(position) position : vec4f) -> @location(0) vec4f {
  let p = vec2i(floor(position.xy));
  let last = vec2i(params.size) - vec2i(1);
  let c = smooth_at(p);
  let x0 = max(p.x - kStep, 0);
  let x1 = min(p.x + kStep, last.x);
  let y0 = max(p.y - kStep, 0);
  let y1 = min(p.y + kStep, last.y);
  let sx = slope(c, smooth_at(vec2i(x0, p.y)), p.x - x0, smooth_at(vec2i(x1, p.y)), x1 - p.x);
  let sy = slope(c, smooth_at(vec2i(p.x, y0)), p.y - y0, smooth_at(vec2i(p.x, y1)), y1 - p.y);
  let gx = clamp(sx.value / params.units.x, -kMaxSlope, kMaxSlope);
  let gy = clamp(sy.value / params.units.y, -kMaxSlope, kMaxSlope);
  let conf = 1.0 - smoothstep(kEdge, 2.0 * kEdge, max(sx.edge, sy.edge));
  return vec4f(gx, gy, c, conf);
}

// Crossfade snapshot: writes mix(from, to, t) into a free field texture, so a
// new result can start its crossfade from exactly what was shown.
struct BlendParams {
  t : f32,
  pad0 : f32,
  pad1 : f32,
  pad2 : f32,
};

@group(0) @binding(0) var blend_from : texture_2d<f32>;
@group(0) @binding(1) var blend_to : texture_2d<f32>;
@group(0) @binding(2) var<uniform> blend : BlendParams;

@fragment
fn fs_blend(@builtin(position) position : vec4f) -> @location(0) vec4f {
  let p = vec2i(floor(position.xy));
  return mix(textureLoad(blend_from, p, 0), textureLoad(blend_to, p, 0), blend.t);
}
