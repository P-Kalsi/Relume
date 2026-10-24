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

}  // namespace relume
