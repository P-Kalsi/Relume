#pragma once

#include <cstdint>

namespace relume {

struct PixelRect {
  std::uint32_t x = 0;
  std::uint32_t y = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
};

// The largest rectangle with the source's aspect ratio that fits inside the
// destination, centred (letterbox or pillarbox, never stretched). Sizes are
// whole pixels. Returns an empty rect if any dimension is zero.
PixelRect FitRect(std::uint32_t src_width, std::uint32_t src_height, std::uint32_t dst_width,
                  std::uint32_t dst_height);

// A position across the camera image: u, v in [0, 1], (0, 0) top left.
struct ImageUv {
  float u = 0.0f;
  float v = 0.0f;
};

// Where a preview of the OutputTexture is drawn on screen (any size; it
// shows the whole texture), and where the camera image sits inside the
// texture (its FitRect).
struct PreviewGeometry {
  float x = 0.0f;
  float y = 0.0f;
  float width = 0.0f;
  float height = 0.0f;
  std::uint32_t output_width = 0;
  std::uint32_t output_height = 0;
  PixelRect image;
};

// A screen point over the preview -> image uv, clamped to [0, 1] (so a
// drag past the image edge or into the letterbox bars stays on the edge).
// Returns {0.5, 0.5} for an empty geometry.
ImageUv PreviewToImageUv(float screen_x, float screen_y, const PreviewGeometry& preview);
// The inverse (not clamped): image uv -> screen point over the preview.
void ImageUvToPreview(ImageUv uv, const PreviewGeometry& preview, float& screen_x, float& screen_y);

}  // namespace relume
