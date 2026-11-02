// Relighting pass: the camera image, relit with one movable light, drawn over
// the same aspect-fit viewport that OutputCompositor uses (blit.wgsl's uv), so
// it lines up with passthrough pixel for pixel. Repeats ComputeShading and
// ShadePixel in src/core/lighting.cpp; the constants below must match
// include/relume/lighting.hpp (tests compare the two).
//
// Colour: every texture involved is a non-sRGB (Unorm) format, so nothing is
// converted automatically. This shader decodes sRGB once after sampling the
// camera and encodes once before writing; no other pass converts.
//
// Depth field sampling: the field covers exactly the camera image (the valid
// region of the tensor), so the image uv is also the field uv. A filtering
// sampler at uv reads texel position uv * size - 0.5, which is DepthToSource's
// half-pixel rule.

const kWrap : f32 = 0.3;
const kMaxGradient : f32 = 3.0;
const kShininess : f32 = 24.0;
const kAttenuationK : f32 = 0.25;
const kMaxAttenuation : f32 = 2.0;
const kMinGain : f32 = 0.35;
const kMaxGain : f32 = 1.8;

struct Light {
  position : vec4f,  // the light in lighting space (xyz); w: image aspect (width / height)
  shade : vec4f,     // intensity, ambient, specular, relief
  tint : vec4f,      // tint rgb (luminance 1); w: light height above the nearest depth (settings z)
  control : vec4f,   // x: crossfade weight from -> to; y: fade; z: view (0 relit, 1 height, 2 normals,
                     // 3 gain, 4 confidence; views other than 0 are for the Controls window only)
};

@group(0) @binding(0) var linear_sampler : sampler;
@group(0) @binding(1) var camera_tex : texture_2d<f32>;
@group(0) @binding(2) var field_from : texture_2d<f32>;
@group(0) @binding(3) var field_to : texture_2d<f32>;
@group(0) @binding(4) var<uniform> light : Light;

struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) uv : vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) index : u32) -> VertexOut {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  let corner = corners[index];
  var out : VertexOut;
  out.position = vec4f(corner, 0.0, 1.0);
  out.uv = vec2f(corner.x * 0.5 + 0.5, 0.5 - corner.y * 0.5);
  return out;
}

fn srgb_to_linear(c : vec3f) -> vec3f {
  return select(pow((c + vec3f(0.055)) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}

fn linear_to_srgb(l : vec3f) -> vec3f {
  return select(1.055 * pow(l, vec3f(1.0 / 2.4)) - vec3f(0.055), l * 12.92, l <= vec3f(0.0031308));
}

fn safe_normalize(v : vec3f) -> vec3f {
  let length2 = dot(v, v);
  if (!(length2 > 1e-24)) {
    return vec3f(0.0, 0.0, 1.0);
  }
  return v / sqrt(length2);
}

fn attenuation(distance : f32) -> f32 {
  return 1.0 / (1.0 + kAttenuationK * distance * distance);
}

fn boost(x : vec3f, g : vec3f) -> vec3f {
  return select(x * g / (vec3f(1.0) + (g - vec3f(1.0)) * x), x * g, g <= vec3f(1.0));
}

// FieldNormal: slopes scaled by relief, clamped, then by confidence.
fn field_normal(field : vec4f, relief : f32) -> vec3f {
  let conf = clamp(field.w, 0.0, 1.0);
  let g = conf * clamp(relief * field.xy, vec2f(-kMaxGradient), vec2f(kMaxGradient));
  return safe_normalize(vec3f(-g.x, -g.y, 1.0));
}

struct Shading {
  gain : vec3f,
  spec : f32,
};

fn compute_shading(field : vec4f, uv : vec2f) -> Shading {
  let aspect = light.position.w;
  let intensity = light.shade.x;
  let ambient = light.shade.y;
  let specular = light.shade.z;
  let relief = light.shade.w;
  let conf = clamp(field.w, 0.0, 1.0);
  let n = field_normal(field, relief);
  let p = vec3f((uv.x - 0.5) * 2.0 * aspect, (0.5 - uv.y) * 2.0, relief * field.z);
  let to_light = light.position.xyz - p;
  let distance = sqrt(dot(to_light, to_light));
  let l = safe_normalize(to_light);
  let diff = max((dot(n, l) + kWrap) / (1.0 + kWrap), 0.0);
  let att = min(attenuation(distance) / attenuation(light.tint.w), kMaxAttenuation);
  let key = intensity * diff * att;
  var out : Shading;
  out.gain = clamp(vec3f(ambient) + key * light.tint.rgb, vec3f(kMinGain), vec3f(kMaxGain));
  let half_vector = safe_normalize(vec3f(l.x, l.y, l.z + 1.0));
  out.spec = specular * pow(max(dot(n, half_vector), 0.0), kShininess) * att * conf;
  return out;
}

@fragment
fn fs_main(in : VertexOut) -> @location(0) vec4f {
  // Every sample in uniform control flow.
  let camera = textureSample(camera_tex, linear_sampler, in.uv).rgb;
  let f0 = textureSample(field_from, linear_sampler, in.uv);
  let f1 = textureSample(field_to, linear_sampler, in.uv);
  let field = mix(f0, f1, light.control.x);
  let view = u32(light.control.z + 0.5);
  if (view == 1u) {
    return vec4f(vec3f(clamp(field.z, 0.0, 1.0)), 1.0);
  }
  if (view == 2u) {
    return vec4f(field_normal(field, light.shade.w) * 0.5 + vec3f(0.5), 1.0);
  }
  if (view == 4u) {
    return vec4f(vec3f(clamp(field.w, 0.0, 1.0)), 1.0);
  }
  let shading = compute_shading(field, in.uv);
  if (view == 3u) {
    return vec4f(shading.gain / kMaxGain, 1.0);  // gain 1 is about 56% grey
  }
  let fade = light.control.y;
  let lin = srgb_to_linear(camera);
  let luma = dot(lin, vec3f(0.2126, 0.7152, 0.0722));
  let g = vec3f(1.0) + (shading.gain - vec3f(1.0)) * fade;
  let value = clamp(boost(lin, g) + vec3f(shading.spec * luma * fade), vec3f(0.0), vec3f(1.0));
  return vec4f(linear_to_srgb(value), 1.0);
}
