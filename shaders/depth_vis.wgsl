// Depth visualization for the Controls window (never the Output window).
//
// Drawn over the same aspect-fit viewport that OutputCompositor uses for the
// camera image, so a depth pixel lands on the output pixel showing the same
// source position. For an output pixel centre at uv (0..1 across the fitted
// rectangle), the source position is X = uv * src_size (continuous source
// coordinates). The depth texture holds only the valid (unpadded) region of
// the 448x448 tensor, new_w x new_h texels, and is resampled exactly like
// DepthToSource: texel position u = X * (new / src) - 0.5, clamped, bilinear,
// horizontal first, then vertical. textureLoad keeps it independent of the
// float32-filterable feature.

struct Params {
  rect : vec4f,       // fitted rectangle in target pixels: x, y, width, height
  src_size : vec2f,   // camera image size
  scale : vec2f,      // new / src per axis
  crop_size : vec2f,  // depth texture size (new_w, new_h)
  range : vec2f,      // display range lo, hi
  flags : vec2f,      // x: range valid (1) or neutral grey (0); y: camera overlay weight (0 = depth only)
  pad : vec2f,
};

@group(0) @binding(0) var depth_tex : texture_2d<f32>;
@group(0) @binding(1) var<uniform> params : Params;
@group(0) @binding(2) var camera_sampler : sampler;
@group(0) @binding(3) var camera_tex : texture_2d<f32>;

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

fn load_depth(p : vec2f) -> f32 {
  return textureLoad(depth_tex, vec2i(p), 0).r;
}

@fragment
fn fs_main(in : VertexOut) -> @location(0) vec4f {
  // The camera sample must be taken in uniform control flow.
  let camera = textureSample(camera_tex, camera_sampler, in.uv).rgb;

  // The pixel centre in target coordinates, relative to the fitted rectangle.
  let uv = (in.position.xy - params.rect.xy) / params.rect.zw;
  let source = uv * params.src_size;
  let last = params.crop_size - vec2f(1.0);
  let u = clamp(source * params.scale - vec2f(0.5), vec2f(0.0), last);
  let i0 = floor(u);
  let f = u - i0;
  let i1 = min(i0 + vec2f(1.0), last);
  let top = load_depth(vec2f(i0.x, i0.y)) * (1.0 - f.x) + load_depth(vec2f(i1.x, i0.y)) * f.x;
  let bottom = load_depth(vec2f(i0.x, i1.y)) * (1.0 - f.x) + load_depth(vec2f(i1.x, i1.y)) * f.x;
  let depth = top * (1.0 - f.y) + bottom * f.y;

  var v = 0.5;  // neutral grey when the range is unusable
  if (params.flags.x > 0.5) {
    v = clamp((depth - params.range.x) / (params.range.y - params.range.x), 0.0, 1.0);
  }
  let depth_color = vec3f(v);  // brighter = nearer
  let color = mix(depth_color, camera, params.flags.y);
  return vec4f(color, 1.0);
}
