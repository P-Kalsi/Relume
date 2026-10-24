#include "relume/layout.hpp"

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

}  // namespace relume
