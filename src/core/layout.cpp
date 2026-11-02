#include "relume/layout.hpp"

#include <algorithm>

namespace relume {

PixelRect FitRect(std::uint32_t src_width, std::uint32_t src_height, std::uint32_t dst_width,
                  std::uint32_t dst_height) {
  if (src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0) {
    return {};
  }
  // Compare src_w/src_h with dst_w/dst_h exactly in 64-bit integers.
  const std::uint64_t src_cross = std::uint64_t{src_width} * dst_height;
  const std::uint64_t dst_cross = std::uint64_t{dst_width} * src_height;
  PixelRect rect;
  if (src_cross >= dst_cross) {
    // Source is relatively wider: full width, bars above and below.
    rect.width = dst_width;
    rect.height = static_cast<std::uint32_t>((std::uint64_t{dst_width} * src_height + src_width / 2) / src_width);
  } else {
    // Source is relatively taller: full height, bars left and right.
    rect.height = dst_height;
    rect.width = static_cast<std::uint32_t>((std::uint64_t{dst_height} * src_width + src_height / 2) / src_height);
  }
  rect.width = rect.width == 0 ? 1 : rect.width;
  rect.height = rect.height == 0 ? 1 : rect.height;
  rect.x = (dst_width - rect.width) / 2;
  rect.y = (dst_height - rect.height) / 2;
  return rect;
}

ImageUv PreviewToImageUv(float screen_x, float screen_y, const PreviewGeometry& p) {
  if (p.width <= 0.0f || p.height <= 0.0f || p.output_width == 0 || p.output_height == 0 || p.image.width == 0 ||
      p.image.height == 0) {
    return {0.5f, 0.5f};
  }
  // Screen -> output texture pixels -> image.
  const float ox = (screen_x - p.x) / p.width * static_cast<float>(p.output_width);
  const float oy = (screen_y - p.y) / p.height * static_cast<float>(p.output_height);
  return {std::clamp((ox - static_cast<float>(p.image.x)) / static_cast<float>(p.image.width), 0.0f, 1.0f),
          std::clamp((oy - static_cast<float>(p.image.y)) / static_cast<float>(p.image.height), 0.0f, 1.0f)};
}

void ImageUvToPreview(ImageUv uv, const PreviewGeometry& p, float& screen_x, float& screen_y) {
  const float ox = static_cast<float>(p.image.x) + uv.u * static_cast<float>(p.image.width);
  const float oy = static_cast<float>(p.image.y) + uv.v * static_cast<float>(p.image.height);
  screen_x = p.output_width == 0 ? p.x : p.x + ox / static_cast<float>(p.output_width) * p.width;
  screen_y = p.output_height == 0 ? p.y : p.y + oy / static_cast<float>(p.output_height) * p.height;
}

}  // namespace relume
