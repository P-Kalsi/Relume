// Camera frame -> RGBA8 camera texture, one fragment per camera pixel.
// Texels are read with textureLoad (no filtering), so the result is exact
// and matches the CPU reference in src/core/pixel_convert.cpp.

@vertex
fn vs_fullscreen(@builtin(vertex_index) index : u32) -> @builtin(position) vec4f {
  // One triangle covering the whole target.
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(corners[index], 0.0, 1.0);
}

// BGRA8 or RGBA8 input: sampling returns channels in logical RGBA order for
// both formats, so this is a copy.
@group(0) @binding(0) var rgba_input : texture_2d<f32>;

@fragment
fn fs_rgba(@builtin(position) position : vec4f) -> @location(0) vec4f {
  return vec4f(textureLoad(rgba_input, vec2i(position.xy), 0).rgb, 1.0);
}

// NV12: Y in an R8 texture, interleaved CbCr in an RG8 texture at half size.
struct Nv12Params {
  kr : f32,        // luma weight of red (BT.601 0.299, BT.709 0.2126)
  kb : f32,        // luma weight of blue (BT.601 0.114, BT.709 0.0722)
  y_offset : f32,  // 16 (video range) or 0 (full range)
  y_scale : f32,   // 219 or 255
  c_scale : f32,   // 224 or 255
};

@group(0) @binding(0) var luma : texture_2d<f32>;
@group(0) @binding(1) var chroma : texture_2d<f32>;
@group(0) @binding(2) var<uniform> params : Nv12Params;

// Same quantization as the CPU reference: clamp, then floor(255 v + 0.5).
fn quantize(v : vec3f) -> vec3f {
  return floor(clamp(v, vec3f(0.0), vec3f(1.0)) * 255.0 + 0.5) / 255.0;
}

@fragment
fn fs_nv12(@builtin(position) position : vec4f) -> @location(0) vec4f {
  let pixel = vec2i(position.xy);
  // Recover the 8-bit code values. Nearest-neighbour chroma: one CbCr sample
  // per 2x2 luma block, whatever the stream's chroma siting.
  let y_code = round(textureLoad(luma, pixel, 0).r * 255.0);
  let c_code = round(textureLoad(chroma, pixel / 2, 0).rg * 255.0);
  let y = (y_code - params.y_offset) / params.y_scale;
  let cb = (c_code.x - 128.0) / params.c_scale;
  let cr = (c_code.y - 128.0) / params.c_scale;
  let r = y + 2.0 * (1.0 - params.kr) * cr;
  let b = y + 2.0 * (1.0 - params.kb) * cb;
  let g = (y - params.kr * r - params.kb * b) / (1.0 - params.kr - params.kb);
  return vec4f(quantize(vec3f(r, g, b)), 1.0);
}
