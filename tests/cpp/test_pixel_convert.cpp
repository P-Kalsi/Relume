// CPU reference conversion tests. Expected values are independent of the
// implementation: exact hand-computed cases, and published 8-bit 75% colour
// bar code values (BT.601 and SMPTE RP 219 for BT.709).

#include "relume/pixel_convert.hpp"
#include "relume/video_frame.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace relume {
namespace {

// Analytic bound on decoding 8-bit YCbCr bar codes back to nominal RGB:
// 0.5 code of Y error (255/219/2 = 0.58) plus 0.5 code of chroma error
// (1.402*255/224/2 = 0.80) gives at most 1.38, so 2 after rounding. Fixed
// before any measurement.
constexpr int kBarTolerance = 2;
// Round trip RGB -> NV12 -> RGB on content whose 2x2 blocks are uniform:
// the same quantization bound applies.
constexpr int kRoundTripTolerance = 2;

std::byte* Bytes(VideoFrame& frame, std::uint32_t plane, std::uint32_t row) {
  return frame.data.data() + frame.planes[plane].offset + frame.planes[plane].stride_bytes * row;
}

// A 2x2 NV12 frame filled with one YCbCr value.
VideoFrame SolidNv12(std::uint8_t y, std::uint8_t cb, std::uint8_t cr, YuvRange range, YuvMatrix matrix) {
  VideoFrame frame;
  AllocateFrame(frame, PixelFormat::kNv12, 2, 2);
  frame.range = range;
  frame.matrix = matrix;
  for (std::uint32_t row = 0; row < 2; ++row) {
    Bytes(frame, 0, row)[0] = std::byte{y};
    Bytes(frame, 0, row)[1] = std::byte{y};
  }
  Bytes(frame, 1, 0)[0] = std::byte{cb};
  Bytes(frame, 1, 0)[1] = std::byte{cr};
  return frame;
}

std::array<int, 3> FirstRgb(const VideoFrame& frame) {
  std::vector<std::uint8_t> rgb;
  EXPECT_TRUE(ToRgb8(frame, rgb));
  if (rgb.size() < 3) {
    return {-1, -1, -1};
  }
  return {rgb[0], rgb[1], rgb[2]};
}

TEST(PixelConvert, Nv12ExactReferencePoints) {
  for (YuvMatrix m : {YuvMatrix::kBt601, YuvMatrix::kBt709}) {
    // Video range: black, white, mid grey ((126-16)/219*255 = 128.08 -> 128).
    EXPECT_EQ(FirstRgb(SolidNv12(16, 128, 128, YuvRange::kVideo, m)), (std::array<int, 3>{0, 0, 0}));
    EXPECT_EQ(FirstRgb(SolidNv12(235, 128, 128, YuvRange::kVideo, m)), (std::array<int, 3>{255, 255, 255}));
    EXPECT_EQ(FirstRgb(SolidNv12(126, 128, 128, YuvRange::kVideo, m)), (std::array<int, 3>{128, 128, 128}));
    // Footroom and headroom clamp instead of wrapping.
    EXPECT_EQ(FirstRgb(SolidNv12(0, 128, 128, YuvRange::kVideo, m)), (std::array<int, 3>{0, 0, 0}));
    EXPECT_EQ(FirstRgb(SolidNv12(255, 128, 128, YuvRange::kVideo, m)), (std::array<int, 3>{255, 255, 255}));
    // Full range: the code value is the grey level.
    EXPECT_EQ(FirstRgb(SolidNv12(0, 128, 128, YuvRange::kFull, m)), (std::array<int, 3>{0, 0, 0}));
    EXPECT_EQ(FirstRgb(SolidNv12(255, 128, 128, YuvRange::kFull, m)), (std::array<int, 3>{255, 255, 255}));
    EXPECT_EQ(FirstRgb(SolidNv12(77, 128, 128, YuvRange::kFull, m)), (std::array<int, 3>{77, 77, 77}));
  }
}

struct Bar {
  const char* name;
  std::uint8_t y, cb, cr;
  std::array<int, 3> rgb;  // nominal 75% RGB: 0.75 * 255 = 191.25 -> 191
};

void ExpectBars(const std::vector<Bar>& bars, YuvMatrix matrix) {
  for (const Bar& bar : bars) {
    const std::array<int, 3> got = FirstRgb(SolidNv12(bar.y, bar.cb, bar.cr, YuvRange::kVideo, matrix));
    for (int c = 0; c < 3; ++c) {
      EXPECT_LE(std::abs(got[static_cast<std::size_t>(c)] - bar.rgb[static_cast<std::size_t>(c)]), kBarTolerance)
          << ToString(matrix) << " " << bar.name << " channel " << c << ": got " << got[0] << "," << got[1]
          << "," << got[2];
    }
  }
}

TEST(PixelConvert, Bt601ColourBars75) {
  ExpectBars({{"white", 180, 128, 128, {191, 191, 191}},
              {"yellow", 162, 44, 142, {191, 191, 0}},
              {"cyan", 131, 156, 44, {0, 191, 191}},
              {"green", 112, 72, 58, {0, 191, 0}},
              {"magenta", 84, 184, 198, {191, 0, 191}},
              {"red", 65, 100, 212, {191, 0, 0}},
              {"blue", 35, 212, 114, {0, 0, 191}},
              {"black", 16, 128, 128, {0, 0, 0}}},
             YuvMatrix::kBt601);
}

TEST(PixelConvert, Bt709ColourBars75) {
  ExpectBars({{"white", 180, 128, 128, {191, 191, 191}},
              {"yellow", 168, 44, 136, {191, 191, 0}},
              {"cyan", 145, 147, 44, {0, 191, 191}},
              {"green", 133, 63, 52, {0, 191, 0}},
              {"magenta", 63, 193, 204, {191, 0, 191}},
              {"red", 51, 109, 212, {191, 0, 0}},
              {"blue", 28, 212, 120, {0, 0, 191}},
              {"black", 16, 128, 128, {0, 0, 0}}},
             YuvMatrix::kBt709);
}

TEST(PixelConvert, MatrixChoiceMatters) {
  // The same red code values decode differently under the two matrices, so a
  // wrong matrix would be visible rather than silently tolerated.
  const auto as601 = FirstRgb(SolidNv12(65, 100, 212, YuvRange::kVideo, YuvMatrix::kBt601));
  const auto as709 = FirstRgb(SolidNv12(65, 100, 212, YuvRange::kVideo, YuvMatrix::kBt709));
  EXPECT_NE(as601, as709);
}

TEST(PixelConvert, ChromaIsNearestNeighbourPer2x2Block) {
  // 4x2 frame: left block red chroma, right block blue chroma.
  VideoFrame frame;
  AllocateFrame(frame, PixelFormat::kNv12, 4, 2);
  frame.range = YuvRange::kVideo;
  frame.matrix = YuvMatrix::kBt601;
  for (std::uint32_t row = 0; row < 2; ++row) {
    std::byte* luma = Bytes(frame, 0, row);
    luma[0] = luma[1] = std::byte{65};
    luma[2] = luma[3] = std::byte{35};
  }
  std::byte* chroma = Bytes(frame, 1, 0);
  chroma[0] = std::byte{100};
  chroma[1] = std::byte{212};
  chroma[2] = std::byte{212};
  chroma[3] = std::byte{114};
  std::vector<std::uint8_t> rgb;
  ASSERT_TRUE(ToRgb8(frame, rgb));
  for (std::uint32_t row = 0; row < 2; ++row) {
    for (std::uint32_t x = 0; x < 4; ++x) {
      const std::uint8_t* p = &rgb[(row * 4 + x) * 3];
      if (x < 2) {
        EXPECT_GT(p[0], 180) << "red block pixel " << x;
        EXPECT_LT(p[2], 10) << "red block pixel " << x;
      } else {
        EXPECT_LT(p[0], 10) << "blue block pixel " << x;
        EXPECT_GT(p[2], 180) << "blue block pixel " << x;
      }
    }
  }
}

TEST(PixelConvert, ChromaLocationDoesNotChangeNearestNeighbourOutput) {
  VideoFrame frame = SolidNv12(84, 184, 198, YuvRange::kFull, YuvMatrix::kBt709);
  std::vector<std::uint8_t> reference;
  ASSERT_TRUE(ToRgb8(frame, reference));
  for (ChromaLocation loc : {ChromaLocation::kLeft, ChromaLocation::kCenter, ChromaLocation::kTopLeft,
                             ChromaLocation::kTop, ChromaLocation::kBottomLeft, ChromaLocation::kBottom,
                             ChromaLocation::kUnknown}) {
    frame.chroma_location = loc;
    std::vector<std::uint8_t> rgb;
    ASSERT_TRUE(ToRgb8(frame, rgb));
    EXPECT_EQ(rgb, reference) << ToString(loc);
  }
}

std::vector<std::uint8_t> TestPattern(std::uint32_t width, std::uint32_t height) {
  std::vector<std::uint8_t> rgb(std::size_t{width} * height * 3);
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      std::uint8_t* p = &rgb[(std::size_t{y} * width + x) * 3];
      p[0] = static_cast<std::uint8_t>(x * 255 / (width - 1));
      p[1] = static_cast<std::uint8_t>(y * 255 / (height - 1));
      p[2] = static_cast<std::uint8_t>((x * 7 + y * 13) % 256);
    }
  }
  return rgb;
}

TEST(PixelConvert, Bgra8AndRgb8AreExact) {
  const std::uint32_t w = 7;
  const std::uint32_t h = 5;
  const std::vector<std::uint8_t> rgb = TestPattern(w, h);
  for (PixelFormat format : {PixelFormat::kBgra8, PixelFormat::kRgb8}) {
    VideoFrame frame;
    EncodeRgb8(rgb.data(), w, h, format, YuvRange::kFull, YuvMatrix::kBt709, frame);
    std::vector<std::uint8_t> out;
    ASSERT_TRUE(ToRgb8(frame, out));
    EXPECT_EQ(out, rgb) << ToString(format);
    ASSERT_TRUE(ToRgba8(frame, out));
    for (std::size_t i = 0; i < std::size_t{w} * h; ++i) {
      EXPECT_EQ(out[i * 4 + 0], rgb[i * 3 + 0]);
      EXPECT_EQ(out[i * 4 + 1], rgb[i * 3 + 1]);
      EXPECT_EQ(out[i * 4 + 2], rgb[i * 3 + 2]);
      EXPECT_EQ(out[i * 4 + 3], 255);
    }
  }
}

TEST(PixelConvert, BgraByteOrder) {
  VideoFrame frame;
  AllocateFrame(frame, PixelFormat::kBgra8, 1, 1);
  frame.data = {std::byte{10}, std::byte{20}, std::byte{30}, std::byte{255}};  // B G R A
  EXPECT_EQ(FirstRgb(frame), (std::array<int, 3>{30, 20, 10}));
}

TEST(PixelConvert, Nv12RoundTripOnBlockUniformContent) {
  // Every 2x2 block has one colour, so chroma subsampling loses nothing and
  // only quantization error remains.
  const std::uint32_t w = 16;
  const std::uint32_t h = 8;
  std::vector<std::uint8_t> rgb(std::size_t{w} * h * 3);
  for (std::uint32_t y = 0; y < h; ++y) {
    for (std::uint32_t x = 0; x < w; ++x) {
      const std::uint32_t bx = x / 2;
      const std::uint32_t by = y / 2;
      std::uint8_t* p = &rgb[(std::size_t{y} * w + x) * 3];
      p[0] = static_cast<std::uint8_t>(bx * 36);
      p[1] = static_cast<std::uint8_t>(by * 85);
      p[2] = static_cast<std::uint8_t>(255 - bx * 30);
    }
  }
  for (YuvRange range : {YuvRange::kVideo, YuvRange::kFull}) {
    for (YuvMatrix matrix : {YuvMatrix::kBt601, YuvMatrix::kBt709}) {
      VideoFrame frame;
      EncodeRgb8(rgb.data(), w, h, PixelFormat::kNv12, range, matrix, frame);
      std::vector<std::uint8_t> out;
      ASSERT_TRUE(ToRgb8(frame, out));
      int worst = 0;
      for (std::size_t i = 0; i < out.size(); ++i) {
        worst = std::max(worst, std::abs(int{out[i]} - int{rgb[i]}));
      }
      EXPECT_LE(worst, kRoundTripTolerance) << ToString(range) << " " << ToString(matrix);
    }
  }
}

TEST(PixelConvert, PaddedStridesAreHonoured) {
  const std::uint32_t w = 5;
  const std::uint32_t h = 3;
  const std::vector<std::uint8_t> rgb = TestPattern(w, h);
  for (PixelFormat format : {PixelFormat::kBgra8, PixelFormat::kRgb8, PixelFormat::kNv12}) {
    VideoFrame packed;
    EncodeRgb8(rgb.data(), w, h, format, YuvRange::kVideo, YuvMatrix::kBt601, packed);
    // Copy into a frame with 64-byte aligned rows and poison the padding.
    VideoFrame padded = packed;
    AllocateFrame(padded, format, w, h, 64);
    std::fill(padded.data.begin(), padded.data.end(), std::byte{0xEE});
    for (std::uint32_t p = 0; p < packed.plane_count; ++p) {
      ASSERT_GT(padded.planes[p].stride_bytes, packed.planes[p].stride_bytes);
      for (std::uint32_t row = 0; row < packed.planes[p].height; ++row) {
        std::copy_n(Bytes(packed, p, row), packed.planes[p].stride_bytes, Bytes(padded, p, row));
      }
    }
    std::vector<std::uint8_t> a;
    std::vector<std::uint8_t> b;
    ASSERT_TRUE(ToRgb8(packed, a));
    ASSERT_TRUE(ToRgb8(padded, b));
    EXPECT_EQ(a, b) << ToString(format);
  }
}

TEST(PixelConvert, OddSizedNv12UsesCeilChromaPlane) {
  VideoFrame frame;
  AllocateFrame(frame, PixelFormat::kNv12, 5, 3);
  EXPECT_EQ(frame.planes[1].width, 3u);
  EXPECT_EQ(frame.planes[1].height, 2u);
  EXPECT_EQ(ValidateLayout(frame), "");
  std::vector<std::uint8_t> rgb;
  EXPECT_TRUE(ToRgb8(frame, rgb));
  EXPECT_EQ(rgb.size(), 5u * 3u * 3u);
}

TEST(PixelConvert, InvalidLayoutsAreRejected) {
  std::vector<std::uint8_t> rgb;

  VideoFrame empty;
  EXPECT_FALSE(ToRgb8(empty, rgb));
  EXPECT_TRUE(rgb.empty());

  VideoFrame truncated;
  AllocateFrame(truncated, PixelFormat::kBgra8, 4, 4);
  truncated.data.resize(truncated.data.size() - 1);
  EXPECT_NE(ValidateLayout(truncated), "");
  EXPECT_FALSE(ToRgb8(truncated, rgb));

  // A 4:2:2-shaped chroma plane must not be decoded as 4:2:0.
  VideoFrame not420;
  AllocateFrame(not420, PixelFormat::kNv12, 4, 4);
  not420.planes[1].height = 4;
  not420.data.resize(not420.data.size() * 2);
  EXPECT_NE(ValidateLayout(not420).find("4:2:0"), std::string::npos);
  EXPECT_FALSE(ToRgb8(not420, rgb));

  VideoFrame short_stride;
  AllocateFrame(short_stride, PixelFormat::kRgb8, 4, 2);
  short_stride.planes[0].stride_bytes = 11;
  EXPECT_FALSE(ToRgb8(short_stride, rgb));
}

}  // namespace
}  // namespace relume
