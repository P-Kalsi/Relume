// Draws a texture over the current viewport. The caller sets the viewport to
// an aspect-fit rectangle and clears the rest to black, so nothing is
// stretched. Bilinear sampling at texel centres reproduces the source exactly
// when the viewport and texture are the same size.

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
  // uv (0,0) is the top-left of the texture.
  out.uv = vec2f(corner.x * 0.5 + 0.5, 0.5 - corner.y * 0.5);
  return out;
}

@group(0) @binding(0) var source_sampler : sampler;
@group(0) @binding(1) var source : texture_2d<f32>;

@fragment
fn fs_main(in : VertexOut) -> @location(0) vec4f {
  return vec4f(textureSample(source, source_sampler, in.uv).rgb, 1.0);
}
