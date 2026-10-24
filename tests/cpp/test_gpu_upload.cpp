// GPU upload/convert/composite/present against the CPU reference, rendered
// offscreen with Dawn (no window or camera). Tolerances are fixed in advance:
// BGRA8 and RGB8 must match exactly; NV12 within 1 per channel (float
// evaluation order differs between CPU and GPU, which can move a value that
// lands on a rounding boundary by one step).

#include "relume/image_io.hpp"
#include "relume/layout.hpp"
#include "relume/pixel_convert.hpp"
#include "relume/renderer.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>

namespace relume {
namespace {

constexpr int kNv12Tolerance = 1;

class GpuTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    std::string error;
    gpu_ = GpuContext::Create(error);
    ASSERT_NE(gpu_, nullptr) << error;
  }
  static void TearDownTestSuite() { gpu_.reset(); }
  void TearDown() override {
    ASSERT_TRUE(gpu_->WaitIdle());
    EXPECT_EQ(gpu_->errors().count(), errors_before_) << "WebGPU error: " << gpu_->errors().last();
  }
  void SetUp() override { errors_before_ = gpu_->errors().count(); }

  // Shared by the suite: creating a device per test is slow.
  static std::unique_ptr<GpuContext> gpu_;
  std::uint32_t errors_before_ = 0;
};

std::unique_ptr<GpuContext> GpuTest::gpu_;

// Smooth gradients plus the orientation pattern, so every code value range is
// exercised and chroma varies inside 2x2 blocks.
RgbImage TestImage(std::uint32_t width, std::uint32_t height) {
  RgbImage image = MakeOrientationPattern(width, height);
  for (std::uint32_t y = height / 2; y < height; ++y) {
    for (std::uint32_t x = width / 2; x < width; ++x) {
      std::uint8_t* p = &image.pixels[(std::size_t{y} * width + x) * 3];
      p[0] = static_cast<std::uint8_t>(x * 255 / width);
      p[1] = static_cast<std::uint8_t>(y * 255 / height);
      p[2] = static_cast<std::uint8_t>((x * 5 + y * 3) % 256);
    }
  }
  return image;
}

int MaxDifference(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
  if (a.size() != b.size()) {
    return 256;
  }
  int worst = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    worst = std::max(worst, std::abs(int{a[i]} - int{b[i]}));
  }
  return worst;
}

std::vector<std::uint8_t> Gpu(GpuContext& gpu, FrameUploader& uploader, const VideoFrame& frame) {
  EXPECT_TRUE(uploader.Upload(frame));
  std::vector<std::uint8_t> pixels;
  EXPECT_TRUE(ReadbackTexture(gpu, uploader.camera_texture(), pixels));
  return pixels;
}

TEST_F(GpuTest, BgraAndRgbUploadsAreExact) {
  const RgbImage image = TestImage(70, 40);
  FrameUploader uploader(*gpu_);
  for (PixelFormat format : {PixelFormat::kBgra8, PixelFormat::kRgb8}) {
    VideoFrame frame;
    EncodeRgb8(image.pixels.data(), image.width, image.height, format, YuvRange::kFull, YuvMatrix::kBt709, frame);
    std::vector<std::uint8_t> expected;
    ASSERT_TRUE(ToRgba8(frame, expected));
    EXPECT_EQ(MaxDifference(Gpu(*gpu_, uploader, frame), expected), 0) << ToString(format);
  }
}

TEST_F(GpuTest, Nv12MatchesCpuReferenceForAllRangesAndMatrices) {
  const RgbImage image = TestImage(96, 54);
  FrameUploader uploader(*gpu_);
  for (YuvRange range : {YuvRange::kVideo, YuvRange::kFull}) {
    for (YuvMatrix matrix : {YuvMatrix::kBt601, YuvMatrix::kBt709}) {
      VideoFrame frame;
      EncodeRgb8(image.pixels.data(), image.width, image.height, PixelFormat::kNv12, range, matrix, frame);
      std::vector<std::uint8_t> expected;
      ASSERT_TRUE(ToRgba8(frame, expected));
      const std::vector<std::uint8_t> actual = Gpu(*gpu_, uploader, frame);
      const int worst = MaxDifference(actual, expected);
      EXPECT_LE(worst, kNv12Tolerance) << ToString(range) << " " << ToString(matrix);
      std::size_t off_by_one = 0;
      for (std::size_t i = 0; i < actual.size(); ++i) {
        off_by_one += actual[i] != expected[i] ? 1 : 0;
      }
      // Recorded for docs/decisions.md; not an assertion.
      std::printf("NV12 %s %s: max diff %d, %zu of %zu channel values differ\n", ToString(range).data(),
                  ToString(matrix).data(), worst, off_by_one, actual.size());
    }
  }
}

TEST_F(GpuTest, Nv12AllCodeValues) {
  // Every Y code with a sweep of Cb/Cr codes: 256 x 64 pixels, chroma 128 x 32.
  FrameUploader uploader(*gpu_);
  for (YuvRange range : {YuvRange::kVideo, YuvRange::kFull}) {
    for (YuvMatrix matrix : {YuvMatrix::kBt601, YuvMatrix::kBt709}) {
      VideoFrame frame;
      AllocateFrame(frame, PixelFormat::kNv12, 256, 64);
      frame.range = range;
      frame.matrix = matrix;
      for (std::uint32_t y = 0; y < 64; ++y) {
        auto* luma = reinterpret_cast<std::uint8_t*>(frame.data.data()) + frame.planes[0].stride_bytes * y;
        for (std::uint32_t x = 0; x < 256; ++x) {
          luma[x] = static_cast<std::uint8_t>(x);
        }
      }
      for (std::uint32_t y = 0; y < 32; ++y) {
        auto* chroma = reinterpret_cast<std::uint8_t*>(frame.data.data()) + frame.planes[1].offset +
                       frame.planes[1].stride_bytes * y;
        for (std::uint32_t x = 0; x < 128; ++x) {
          chroma[x * 2] = static_cast<std::uint8_t>(x * 2);
          chroma[x * 2 + 1] = static_cast<std::uint8_t>(y * 8);
        }
      }
      std::vector<std::uint8_t> expected;
      ASSERT_TRUE(ToRgba8(frame, expected));
      EXPECT_LE(MaxDifference(Gpu(*gpu_, uploader, frame), expected), kNv12Tolerance)
          << ToString(range) << " " << ToString(matrix);
    }
  }
}

TEST_F(GpuTest, PaddedCameraStridesUploadCorrectly) {
  const RgbImage image = TestImage(50, 20);
  FrameUploader uploader(*gpu_);
  for (PixelFormat format : {PixelFormat::kBgra8, PixelFormat::kNv12}) {
    VideoFrame packed;
    EncodeRgb8(image.pixels.data(), image.width, image.height, format, YuvRange::kVideo, YuvMatrix::kBt709, packed);
    VideoFrame padded = packed;
    AllocateFrame(padded, format, image.width, image.height, 64);  // like CoreVideo's 64-byte row alignment
    for (std::uint32_t p = 0; p < packed.plane_count; ++p) {
      for (std::uint32_t row = 0; row < packed.planes[p].height; ++row) {
        std::memcpy(padded.data.data() + padded.planes[p].offset + padded.planes[p].stride_bytes * row,
                    packed.data.data() + packed.planes[p].offset + packed.planes[p].stride_bytes * row,
                    packed.planes[p].stride_bytes);
      }
    }
    std::vector<std::uint8_t> expected;
    ASSERT_TRUE(ToRgba8(packed, expected));
    const int tolerance = format == PixelFormat::kNv12 ? kNv12Tolerance : 0;
    EXPECT_LE(MaxDifference(Gpu(*gpu_, uploader, padded), expected), tolerance) << ToString(format);
  }
}

TEST_F(GpuTest, InvalidFrameIsRejectedAndPreviousTextureKept) {
  FrameUploader uploader(*gpu_);
  VideoFrame good;
  const RgbImage image = TestImage(8, 8);
  EncodeRgb8(image.pixels.data(), 8, 8, PixelFormat::kBgra8, YuvRange::kFull, YuvMatrix::kBt709, good);
  ASSERT_TRUE(uploader.Upload(good));
  VideoFrame bad = good;
  bad.data.resize(3);
  EXPECT_FALSE(uploader.Upload(bad));
  EXPECT_EQ(uploader.width(), 8u);
  EXPECT_TRUE(uploader.camera_texture());
}

std::array<int, 3> PixelAt(const std::vector<std::uint8_t>& rgba, std::uint32_t width, std::uint32_t x,
                           std::uint32_t y) {
  const std::uint8_t* p = &rgba[(std::size_t{y} * width + x) * 4];
  return {p[0], p[1], p[2]};
}

void ExpectColor(std::array<int, 3> actual, const std::uint8_t (&expected)[3], int tolerance, const char* what) {
  for (std::size_t c = 0; c < 3; ++c) {
    EXPECT_LE(std::abs(actual[c] - expected[c]), tolerance) << what << " channel " << c;
  }
}

TEST_F(GpuTest, OrientationIsPreservedEndToEnd) {
  // 1280x720 pattern through upload and compositing: the output texture must
  // show the marker top-left and each colour in its own corner (not mirrored
  // or flipped), with exact values since sizes match 1:1.
  const RgbImage image = MakeOrientationPattern(1280, 720);
  FrameUploader uploader(*gpu_);
  OutputTexture output(*gpu_);
  OutputCompositor compositor(*gpu_, output);
  for (PixelFormat format : {PixelFormat::kBgra8, PixelFormat::kNv12}) {
    VideoFrame frame;
    EncodeRgb8(image.pixels.data(), 1280, 720, format, YuvRange::kVideo, YuvMatrix::kBt709, frame);
    ASSERT_TRUE(uploader.Upload(frame));
    compositor.Compose(uploader.camera_texture());
    std::vector<std::uint8_t> out;
    ASSERT_TRUE(ReadbackTexture(*gpu_, output.texture(), out));
    const int tol = format == PixelFormat::kNv12 ? 2 : 0;  // NV12 round-trip bound (test_pixel_convert)
    ExpectColor(PixelAt(out, 1280, 120, 60), pattern::kMarker, tol, "marker (top-left)");
    ExpectColor(PixelAt(out, 1280, 20, 300), pattern::kTopLeft, tol, "top-left");
    ExpectColor(PixelAt(out, 1280, 1260, 20), pattern::kTopRight, tol, "top-right");
    ExpectColor(PixelAt(out, 1280, 20, 700), pattern::kBottomLeft, tol, "bottom-left");
    ExpectColor(PixelAt(out, 1280, 1260, 700), pattern::kBottomRight, tol, "bottom-right");
    if (format == PixelFormat::kBgra8) {
      std::vector<std::uint8_t> expected;
      ASSERT_TRUE(ToRgba8(frame, expected));
      EXPECT_EQ(MaxDifference(out, expected), 0) << "1:1 compositing must be exact";
    }
  }
}

TEST_F(GpuTest, FourThreeCameraIsPillarboxedInOutput) {
  const RgbImage image = MakeOrientationPattern(640, 480);
  FrameUploader uploader(*gpu_);
  OutputTexture output(*gpu_);
  OutputCompositor compositor(*gpu_, output);
  VideoFrame frame;
  EncodeRgb8(image.pixels.data(), 640, 480, PixelFormat::kBgra8, YuvRange::kFull, YuvMatrix::kBt709, frame);
  ASSERT_TRUE(uploader.Upload(frame));
  compositor.Compose(uploader.camera_texture());
  std::vector<std::uint8_t> out;
  ASSERT_TRUE(ReadbackTexture(*gpu_, output.texture(), out));
  const PixelRect fit = FitRect(640, 480, 1280, 720);  // x 160..1119
  const std::uint8_t black[3] = {0, 0, 0};
  ExpectColor(PixelAt(out, 1280, fit.x - 1, 360), black, 0, "left bar");
  ExpectColor(PixelAt(out, 1280, fit.x + fit.width, 360), black, 0, "right bar");
  ExpectColor(PixelAt(out, 1280, fit.x + 4, 300), pattern::kTopLeft, 0, "image left edge");
  ExpectColor(PixelAt(out, 1280, fit.x + fit.width - 4, 20), pattern::kTopRight, 0, "image right edge");
}

TEST_F(GpuTest, ClearToBlackLeavesNoStaleFrame) {
  const RgbImage image = MakeOrientationPattern(64, 36);
  FrameUploader uploader(*gpu_);
  OutputTexture output(*gpu_);
  OutputCompositor compositor(*gpu_, output);
  VideoFrame frame;
  EncodeRgb8(image.pixels.data(), 64, 36, PixelFormat::kBgra8, YuvRange::kFull, YuvMatrix::kBt709, frame);
  ASSERT_TRUE(uploader.Upload(frame));
  compositor.Compose(uploader.camera_texture());
  compositor.ClearToBlack();
  std::vector<std::uint8_t> out;
  ASSERT_TRUE(ReadbackTexture(*gpu_, output.texture(), out));
  for (std::size_t i = 0; i < out.size(); i += 4) {
    ASSERT_EQ(out[i] + out[i + 1] + out[i + 2], 0) << "pixel " << i / 4;
  }
}

// The Presenter shows the OutputTexture in targets of any size without
// stretching: exact at 1:1, letterboxed when the target is taller.
TEST_F(GpuTest, PresenterIsExactAtOneToOneAndLetterboxesOtherwise) {
  const RgbImage image = MakeOrientationPattern(1280, 720);
  FrameUploader uploader(*gpu_);
  OutputTexture output(*gpu_);
  OutputCompositor compositor(*gpu_, output);
  Presenter presenter(*gpu_, output);
  VideoFrame frame;
  EncodeRgb8(image.pixels.data(), 1280, 720, PixelFormat::kBgra8, YuvRange::kFull, YuvMatrix::kBt709, frame);
  ASSERT_TRUE(uploader.Upload(frame));
  compositor.Compose(uploader.camera_texture());
  std::vector<std::uint8_t> output_pixels;
  ASSERT_TRUE(ReadbackTexture(*gpu_, output.texture(), output_pixels));

  auto target_of = [&](std::uint32_t w, std::uint32_t h, wgpu::TextureFormat format) {
    wgpu::TextureDescriptor desc{};
    desc.size = {w, h, 1};
    desc.format = format;
    desc.usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc;
    return gpu_->device().CreateTexture(&desc);
  };

  // 1:1 into an RGBA target: identical to the output texture.
  const wgpu::Texture same = target_of(1280, 720, wgpu::TextureFormat::RGBA8Unorm);
  presenter.Present(same.CreateView(), wgpu::TextureFormat::RGBA8Unorm, 1280, 720);
  std::vector<std::uint8_t> presented;
  ASSERT_TRUE(ReadbackTexture(*gpu_, same, presented));
  EXPECT_EQ(MaxDifference(presented, output_pixels), 0);

  // A BGRA surface-like target: same pixels, stored as B G R A.
  const wgpu::Texture bgra = target_of(1280, 720, wgpu::TextureFormat::BGRA8Unorm);
  presenter.Present(bgra.CreateView(), wgpu::TextureFormat::BGRA8Unorm, 1280, 720);
  ASSERT_TRUE(ReadbackTexture(*gpu_, bgra, presented));
  for (std::size_t i = 0; i < presented.size(); i += 4) {
    std::swap(presented[i], presented[i + 2]);
  }
  EXPECT_EQ(MaxDifference(presented, output_pixels), 0);

  // A tall target: letterboxed, never stretched.
  const wgpu::Texture tall = target_of(800, 1000, wgpu::TextureFormat::RGBA8Unorm);
  presenter.Present(tall.CreateView(), wgpu::TextureFormat::RGBA8Unorm, 800, 1000);
  ASSERT_TRUE(ReadbackTexture(*gpu_, tall, presented));
  const PixelRect fit = FitRect(1280, 720, 800, 1000);  // y 275..724
  const std::uint8_t black[3] = {0, 0, 0};
  ExpectColor(PixelAt(presented, 800, 400, fit.y - 1), black, 0, "top bar");
  ExpectColor(PixelAt(presented, 800, 400, fit.y + fit.height), black, 0, "bottom bar");
  ExpectColor(PixelAt(presented, 800, 75, fit.y + 37), pattern::kMarker, 0, "marker");
  ExpectColor(PixelAt(presented, 800, 790, fit.y + 5), pattern::kTopRight, 0, "top-right");
}

}  // namespace
}  // namespace relume
