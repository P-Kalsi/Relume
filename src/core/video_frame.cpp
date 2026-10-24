#include "relume/video_frame.hpp"

namespace relume {

std::string_view ToString(PixelFormat format) {
  switch (format) {
    case PixelFormat::kBgra8: return "BGRA8";
    case PixelFormat::kRgb8: return "RGB8";
    case PixelFormat::kNv12: return "NV12";
  }
  return "unknown";
}

std::string_view ToString(YuvRange range) {
  switch (range) {
    case YuvRange::kVideo: return "video";
    case YuvRange::kFull: return "full";
  }
  return "unknown";
}

std::string_view ToString(YuvMatrix matrix) {
  switch (matrix) {
    case YuvMatrix::kBt601: return "BT.601";
    case YuvMatrix::kBt709: return "BT.709";
  }
  return "unknown";
}

std::string_view ToString(ChromaLocation location) {
  switch (location) {
    case ChromaLocation::kLeft: return "left";
    case ChromaLocation::kCenter: return "center";
    case ChromaLocation::kTopLeft: return "top-left";
    case ChromaLocation::kTop: return "top";
    case ChromaLocation::kBottomLeft: return "bottom-left";
    case ChromaLocation::kBottom: return "bottom";
    case ChromaLocation::kUnknown: return "unknown";
  }
  return "unknown";
}

namespace {

std::size_t AlignUp(std::size_t value, std::size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

}  // namespace

void AllocateFrame(VideoFrame& frame, PixelFormat format, std::uint32_t width, std::uint32_t height,
                   std::size_t row_alignment) {
  if (row_alignment == 0) {
    row_alignment = 1;
  }
  frame.format = format;
  frame.width = width;
  frame.height = height;
  frame.planes = {};
  std::size_t total = 0;
  auto add_plane = [&](std::uint32_t index, std::uint32_t w, std::uint32_t h, std::uint32_t bytes) {
    PlaneLayout& plane = frame.planes[index];
    plane.offset = total;
    plane.width = w;
    plane.height = h;
    plane.bytes_per_sample = bytes;
    plane.stride_bytes = AlignUp(std::size_t{w} * bytes, row_alignment);
    total += plane.stride_bytes * h;
  };
  switch (format) {
    case PixelFormat::kBgra8:
      add_plane(0, width, height, 4);
      frame.plane_count = 1;
      break;
    case PixelFormat::kRgb8:
      add_plane(0, width, height, 3);
      frame.plane_count = 1;
      break;
    case PixelFormat::kNv12:
      add_plane(0, width, height, 1);
      add_plane(1, (width + 1) / 2, (height + 1) / 2, 2);
      frame.plane_count = 2;
      break;
  }
  frame.data.resize(total);
}

std::string ValidateLayout(const VideoFrame& frame) {
  if (frame.width == 0 || frame.height == 0) {
    return "frame has zero size";
  }
  std::uint32_t expected_planes = 1;
  std::array<std::uint32_t, 2> expected_bytes = {4, 0};
  std::array<std::uint32_t, 2> expected_width = {frame.width, 0};
  std::array<std::uint32_t, 2> expected_height = {frame.height, 0};
  switch (frame.format) {
    case PixelFormat::kBgra8:
      break;
    case PixelFormat::kRgb8:
      expected_bytes[0] = 3;
      break;
    case PixelFormat::kNv12:
      expected_planes = 2;
      expected_bytes = {1, 2};
      expected_width[1] = (frame.width + 1) / 2;
      expected_height[1] = (frame.height + 1) / 2;
      break;
  }
  if (frame.plane_count != expected_planes) {
    return "wrong plane count for " + std::string(ToString(frame.format));
  }
  for (std::uint32_t i = 0; i < expected_planes; ++i) {
    const PlaneLayout& plane = frame.planes[i];
    const std::string name = "plane " + std::to_string(i);
    if (plane.bytes_per_sample != expected_bytes[i]) {
      return name + ": wrong bytes per sample";
    }
    if (plane.width != expected_width[i] || plane.height != expected_height[i]) {
      return name + (i == 1 ? ": chroma is not 4:2:0 subsampled" : ": size does not match the frame");
    }
    if (plane.stride_bytes < std::size_t{plane.width} * plane.bytes_per_sample) {
      return name + ": stride is smaller than a row";
    }
    const std::size_t end = plane.offset + plane.stride_bytes * (plane.height - 1) +
                            std::size_t{plane.width} * plane.bytes_per_sample;
    if (plane.offset > frame.data.size() || end > frame.data.size()) {
      return name + ": extends past the end of the data";
    }
  }
  return {};
}

}  // namespace relume
