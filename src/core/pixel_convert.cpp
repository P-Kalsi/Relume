#include "relume/pixel_convert.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace relume {

namespace {

struct LumaWeights {
  float kr;
  float kb;
};

LumaWeights WeightsFor(YuvMatrix matrix) {
  return matrix == YuvMatrix::kBt601 ? LumaWeights{0.299f, 0.114f} : LumaWeights{0.2126f, 0.0722f};
}

// Quantization of normalized samples: value = offset + scale * normalized.
struct RangeScale {
  float y_offset;
  float y_scale;
  float c_scale;
};

RangeScale ScaleFor(YuvRange range) {
  return range == YuvRange::kVideo ? RangeScale{16.0f, 219.0f, 224.0f} : RangeScale{0.0f, 255.0f, 255.0f};
}

std::uint8_t Quantize(float value) {
  const float clamped = std::clamp(value, 0.0f, 1.0f);
  return static_cast<std::uint8_t>(std::floor(clamped * 255.0f + 0.5f));
}

std::uint8_t QuantizeCode(float value) {
  return static_cast<std::uint8_t>(std::clamp(std::floor(value + 0.5f), 0.0f, 255.0f));
}

const std::uint8_t* PlaneRow(const VideoFrame& frame, std::uint32_t plane, std::uint32_t row) {
  const PlaneLayout& layout = frame.planes[plane];
  return reinterpret_cast<const std::uint8_t*>(frame.data.data()) + layout.offset + layout.stride_bytes * row;
}

std::uint8_t* PlaneRow(VideoFrame& frame, std::uint32_t plane, std::uint32_t row) {
  const PlaneLayout& layout = frame.planes[plane];
  return reinterpret_cast<std::uint8_t*>(frame.data.data()) + layout.offset + layout.stride_bytes * row;
}

// Writes `channels` (3 or 4) bytes per pixel.
void Convert(const VideoFrame& frame, std::uint32_t channels, std::vector<std::uint8_t>& out) {
  const std::size_t width = frame.width;
  out.resize(width * frame.height * channels);
  for (std::uint32_t y = 0; y < frame.height; ++y) {
    std::uint8_t* dst = out.data() + y * width * channels;
    switch (frame.format) {
      case PixelFormat::kBgra8: {
        const std::uint8_t* src = PlaneRow(frame, 0, y);
        for (std::size_t x = 0; x < width; ++x) {
          dst[x * channels + 0] = src[x * 4 + 2];
          dst[x * channels + 1] = src[x * 4 + 1];
          dst[x * channels + 2] = src[x * 4 + 0];
        }
        break;
      }
      case PixelFormat::kRgb8: {
        const std::uint8_t* src = PlaneRow(frame, 0, y);
        for (std::size_t x = 0; x < width; ++x) {
          std::memcpy(dst + x * channels, src + x * 3, 3);
        }
        break;
      }
      case PixelFormat::kNv12: {
        const LumaWeights w = WeightsFor(frame.matrix);
        const RangeScale s = ScaleFor(frame.range);
        const float kg = 1.0f - w.kr - w.kb;
        const std::uint8_t* luma = PlaneRow(frame, 0, y);
        const std::uint8_t* chroma = PlaneRow(frame, 1, y / 2);
        for (std::size_t x = 0; x < width; ++x) {
          const float yn = (static_cast<float>(luma[x]) - s.y_offset) / s.y_scale;
          const float cb = (static_cast<float>(chroma[(x / 2) * 2 + 0]) - 128.0f) / s.c_scale;
          const float cr = (static_cast<float>(chroma[(x / 2) * 2 + 1]) - 128.0f) / s.c_scale;
          const float r = yn + 2.0f * (1.0f - w.kr) * cr;
          const float b = yn + 2.0f * (1.0f - w.kb) * cb;
          const float g = (yn - w.kr * r - w.kb * b) / kg;
          dst[x * channels + 0] = Quantize(r);
          dst[x * channels + 1] = Quantize(g);
          dst[x * channels + 2] = Quantize(b);
        }
        break;
      }
    }
    if (channels == 4) {
      for (std::size_t x = 0; x < width; ++x) {
        dst[x * 4 + 3] = 255;
      }
    }
  }
}

}  // namespace

bool ToRgb8(const VideoFrame& frame, std::vector<std::uint8_t>& out) {
  out.clear();
  if (!ValidateLayout(frame).empty()) {
    return false;
  }
  Convert(frame, 3, out);
  return true;
}

bool ToRgba8(const VideoFrame& frame, std::vector<std::uint8_t>& out) {
  out.clear();
  if (!ValidateLayout(frame).empty()) {
    return false;
  }
  Convert(frame, 4, out);
  return true;
}

void EncodeRgb8(const std::uint8_t* rgb, std::uint32_t width, std::uint32_t height, PixelFormat format,
                YuvRange range, YuvMatrix matrix, VideoFrame& out) {
  AllocateFrame(out, format, width, height);
  out.range = range;
  out.matrix = matrix;
  const std::size_t row_bytes = std::size_t{width} * 3;
  switch (format) {
    case PixelFormat::kRgb8:
      for (std::uint32_t y = 0; y < height; ++y) {
        std::memcpy(PlaneRow(out, 0, y), rgb + y * row_bytes, row_bytes);
      }
      return;
    case PixelFormat::kBgra8:
      for (std::uint32_t y = 0; y < height; ++y) {
        const std::uint8_t* src = rgb + y * row_bytes;
        std::uint8_t* dst = PlaneRow(out, 0, y);
        for (std::size_t x = 0; x < width; ++x) {
          dst[x * 4 + 0] = src[x * 3 + 2];
          dst[x * 4 + 1] = src[x * 3 + 1];
          dst[x * 4 + 2] = src[x * 3 + 0];
          dst[x * 4 + 3] = 255;
        }
      }
      return;
    case PixelFormat::kNv12:
      break;
  }

  const LumaWeights w = WeightsFor(matrix);
  const RangeScale s = ScaleFor(range);
  const float kg = 1.0f - w.kr - w.kb;
  auto luma_of = [&](float r, float g, float b) { return w.kr * r + kg * g + w.kb * b; };
  for (std::uint32_t y = 0; y < height; ++y) {
    const std::uint8_t* src = rgb + y * row_bytes;
    std::uint8_t* dst = PlaneRow(out, 0, y);
    for (std::size_t x = 0; x < width; ++x) {
      const float yn = luma_of(src[x * 3] / 255.0f, src[x * 3 + 1] / 255.0f, src[x * 3 + 2] / 255.0f);
      dst[x] = QuantizeCode(s.y_offset + s.y_scale * yn);
    }
  }
  const PlaneLayout& chroma = out.planes[1];
  for (std::uint32_t cy = 0; cy < chroma.height; ++cy) {
    std::uint8_t* dst = PlaneRow(out, 1, cy);
    for (std::uint32_t cx = 0; cx < chroma.width; ++cx) {
      // Mean RGB of the (up to) 2x2 block this chroma sample covers.
      float sum[3] = {0.0f, 0.0f, 0.0f};
      int count = 0;
      for (std::uint32_t dy = 0; dy < 2 && cy * 2 + dy < height; ++dy) {
        for (std::uint32_t dx = 0; dx < 2 && cx * 2 + dx < width; ++dx) {
          const std::uint8_t* p = rgb + (cy * 2 + dy) * row_bytes + (cx * 2 + dx) * 3;
          for (int c = 0; c < 3; ++c) {
            sum[c] += p[c] / 255.0f;
          }
          ++count;
        }
      }
      const float r = sum[0] / static_cast<float>(count);
      const float g = sum[1] / static_cast<float>(count);
      const float b = sum[2] / static_cast<float>(count);
      const float yn = luma_of(r, g, b);
      const float cb = (b - yn) / (2.0f * (1.0f - w.kb));
      const float cr = (r - yn) / (2.0f * (1.0f - w.kr));
      dst[cx * 2 + 0] = QuantizeCode(128.0f + s.c_scale * cb);
      dst[cx * 2 + 1] = QuantizeCode(128.0f + s.c_scale * cr);
    }
  }
}

}  // namespace relume
