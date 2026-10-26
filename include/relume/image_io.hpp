#pragma once

// Minimal image input for the file source and tests. Only binary PPM (P6,
// maxval 255) is supported, so no image-decoding dependency is needed.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace relume {

struct RgbImage {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> pixels;  // tightly packed RGB, top row first
};

// Reads a P6 PPM. On failure returns nullopt and sets `error`.
std::optional<RgbImage> ReadPpm(const std::string& path, std::string& error);

// Writes a P6 PPM. Used by tests to create fixtures, and by FrameSaver only
// when the user explicitly asks for frames with --save-frames.
bool WritePpm(const std::string& path, const RgbImage& image);

// Deterministic, orientation-revealing test pattern:
//   top-left red, top-right green, bottom-left blue, bottom-right light grey,
//   and a yellow marker square inside the top-left quadrant.
// A mirrored or flipped image therefore moves the marker to another corner.
RgbImage MakeOrientationPattern(std::uint32_t width, std::uint32_t height);

namespace pattern {
inline constexpr std::uint8_t kTopLeft[3] = {220, 40, 40};
inline constexpr std::uint8_t kTopRight[3] = {40, 200, 60};
inline constexpr std::uint8_t kBottomLeft[3] = {40, 60, 220};
inline constexpr std::uint8_t kBottomRight[3] = {230, 230, 230};
inline constexpr std::uint8_t kMarker[3] = {240, 220, 40};
}  // namespace pattern

}  // namespace relume
