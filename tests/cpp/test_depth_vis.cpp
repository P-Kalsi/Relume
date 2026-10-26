// Depth upload and visualization on the GPU (offscreen Dawn), against CPU
// references. Criteria fixed in advance:
//   - the cropped upload reproduces exactly the tensor's valid region;
//   - the visualization matches a CPU reference (the same source mapping as
//     DepthToSource and the same FitRect as OutputCompositor) within 1 per
//     8-bit channel, for 16:9, 4:3 (pillarboxed) and portrait sources, under
//     both letterbox and stretch;
//   - a depth step at a known source position lands on the expected output
//     pixel (within 1 pixel);
//   - the overlay at full camera weight equals the composed output exactly.

#include "relume/image_io.hpp"
#include "relume/layout.hpp"
#include "relume/pixel_convert.hpp"
#include "relume/preprocess.hpp"
#include "relume/renderer.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>

namespace relume {
namespace {

class DepthVisTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    std::string error;
    gpu_ = GpuContext::Create(error);
    ASSERT_NE(gpu_, nullptr) << error;
  }
  static void TearDownTestSuite() { gpu_.reset(); }
  void SetUp() override { errors_before_ = gpu_->errors().count(); }
  void TearDown() override {
    ASSERT_TRUE(gpu_->WaitIdle());
    EXPECT_EQ(gpu_->errors().count(), errors_before_) << "WebGPU error: " << gpu_->errors().last();
  }

  // An RGBA8 camera texture of the source size, filled with a test image.
  wgpu::Texture CameraTexture(std::uint32_t width, std::uint32_t height) {
    FrameUploader uploader(*gpu_);
    VideoFrame frame;
    const RgbImage image = MakeOrientationPattern(width, height);
    EncodeRgb8(image.pixels.data(), width, height, PixelFormat::kBgra8, YuvRange::kFull, YuvMatrix::kBt709, frame);
    frame.frame_id = 1;
    EXPECT_TRUE(uploader.Upload(frame));
    return uploader.camera_texture();  // the texture outlives the uploader (reference counted)
  }

  static std::unique_ptr<GpuContext> gpu_;
  std::uint32_t errors_before_ = 0;
};

std::unique_ptr<GpuContext> DepthVisTest::gpu_;

struct Case {
  const char* name;
  std::uint32_t width, height;
  ResizeMethod method;
};

const Case kCases[] = {
    {"16:9 letterbox", 1280, 720, ResizeMethod::kLetterbox},
    {"16:9 stretch", 1280, 720, ResizeMethod::kStretch},
    {"portrait letterbox", 720, 1280, ResizeMethod::kLetterbox},
    {"portrait stretch", 480, 640, ResizeMethod::kStretch},
    {"4:3 letterbox (pillarboxed)", 640, 480, ResizeMethod::kLetterbox},
    {"4:3 stretch (pillarboxed)", 640, 480, ResizeMethod::kStretch},
};

// A smooth depth map over the whole tensor, with extreme values in the
// letterbox padding (which must never show up).
std::vector<float> SmoothDepth(const ResizeTransform& t) {
  std::vector<float> depth(kNetDepthFloats, 1e6f);
  for (std::uint32_t y = 0; y < kNetSize; ++y) {
    for (std::uint32_t x = 0; x < kNetSize; ++x) {
      const bool inside = x >= t.pad_x && x < t.pad_x + t.new_w && y >= t.pad_y && y < t.pad_y + t.new_h;
      if (inside) {
        depth[std::size_t{y} * kNetSize + x] =
            2.0f + std::sin(static_cast<float>(x) * 0.05f) + 0.6f * std::cos(static_cast<float>(y) * 0.031f);
      }
    }
  }
  return depth;
}

// The CPU reference for one output pixel: FitRect, then the continuous source
// position of the pixel centre, then DepthToSource's bilinear rule on the
// cropped tensor, then the display range.
std::array<int, 3> ReferencePixel(const std::vector<float>& depth, const ResizeTransform& t, const DepthVisParams& p,
                                  std::uint32_t x, std::uint32_t y) {
  const PixelRect rect = FitRect(t.src_w, t.src_h, DepthDebugTexture::kWidth, DepthDebugTexture::kHeight);
  if (x < rect.x || x >= rect.x + rect.width || y < rect.y || y >= rect.y + rect.height) {
    return {0, 0, 0};
  }
  const double src_x = (x + 0.5 - rect.x) / rect.width * t.src_w;
  const double src_y = (y + 0.5 - rect.y) / rect.height * t.src_h;
  const double ux = std::clamp(src_x * t.sx() - 0.5, 0.0, t.new_w - 1.0);
  const double uy = std::clamp(src_y * t.sy() - 0.5, 0.0, t.new_h - 1.0);
  const auto x0 = static_cast<std::uint32_t>(std::floor(ux));
  const auto y0 = static_cast<std::uint32_t>(std::floor(uy));
  const std::uint32_t x1 = std::min(x0 + 1, t.new_w - 1);
  const std::uint32_t y1 = std::min(y0 + 1, t.new_h - 1);
  const double fx = ux - x0;
  const double fy = uy - y0;
  const auto at = [&](std::uint32_t cx, std::uint32_t cy) {
    return static_cast<double>(depth[std::size_t{t.pad_y + cy} * kNetSize + t.pad_x + cx]);
  };
  const double top = at(x0, y0) * (1 - fx) + at(x1, y0) * fx;
  const double bottom = at(x0, y1) * (1 - fx) + at(x1, y1) * fx;
  const double d = top * (1 - fy) + bottom * fy;
  const double v = p.valid ? std::clamp((d - p.lo) / (p.hi - p.lo), 0.0, 1.0) : 0.5;
  const int byte = static_cast<int>(std::lround(v * 255.0));
  return {byte, byte, byte};
}

TEST_F(DepthVisTest, CroppedUploadReproducesExactlyTheValidRegion) {
  for (const Case& c : kCases) {
    SCOPED_TRACE(c.name);
    const ResizeTransform t = *ComputeTransform(c.width, c.height, c.method);
    std::vector<float> depth(kNetDepthFloats);
    for (std::size_t i = 0; i < depth.size(); ++i) {
      depth[i] = static_cast<float>(i);  // a distinct value in every tensor cell
    }
    DepthUploader uploader(*gpu_);
    ASSERT_TRUE(uploader.Upload(depth.data(), t));
    std::vector<float> texels;
    ASSERT_TRUE(ReadbackFloatTexture(*gpu_, uploader.texture(), texels));
    ASSERT_EQ(texels.size(), std::size_t{t.new_w} * t.new_h);
    for (std::uint32_t y = 0; y < t.new_h; ++y) {
      for (std::uint32_t x = 0; x < t.new_w; ++x) {
        ASSERT_EQ(texels[std::size_t{y} * t.new_w + x], depth[std::size_t{t.pad_y + y} * kNetSize + t.pad_x + x])
            << "texel " << x << "," << y;
      }
    }
  }
}

TEST_F(DepthVisTest, UploadRejectsARegionOutsideTheTensor) {
  DepthUploader uploader(*gpu_);
  std::vector<float> depth(kNetDepthFloats);
  ResizeTransform bad = *ComputeTransform(1280, 720, ResizeMethod::kLetterbox);
  bad.pad_y = 300;  // 300 + 252 > 448
  EXPECT_FALSE(uploader.Upload(depth.data(), bad));
  EXPECT_FALSE(uploader.texture());
}

TEST_F(DepthVisTest, MatchesTheCpuReferenceWithinOneStep) {
  for (const Case& c : kCases) {
    SCOPED_TRACE(c.name);
    const ResizeTransform t = *ComputeTransform(c.width, c.height, c.method);
    const std::vector<float> depth = SmoothDepth(t);
    DepthUploader uploader(*gpu_);
    ASSERT_TRUE(uploader.Upload(depth.data(), t));
    DepthDebugTexture target(*gpu_);
    DepthVisualizer vis(*gpu_, target);
    DepthVisParams params;
    params.lo = 1.2f;
    params.hi = 3.4f;
    params.valid = true;
    vis.Draw(uploader, t, params, CameraTexture(c.width, c.height));
    std::vector<std::uint8_t> rgba;
    ASSERT_TRUE(ReadbackTexture(*gpu_, target.texture(), rgba));
    int worst = 0;
    for (std::uint32_t y = 0; y < DepthDebugTexture::kHeight; ++y) {
      for (std::uint32_t x = 0; x < DepthDebugTexture::kWidth; ++x) {
        const std::array<int, 3> expected = ReferencePixel(depth, t, params, x, y);
        const std::uint8_t* got = &rgba[(std::size_t{y} * DepthDebugTexture::kWidth + x) * 4];
        for (int ch = 0; ch < 3; ++ch) {
          worst = std::max(worst, std::abs(got[ch] - expected[static_cast<std::size_t>(ch)]));
        }
      }
    }
    EXPECT_LE(worst, 1) << "max |GPU - CPU| per channel";
  }
}

TEST_F(DepthVisTest, DepthStepLandsOnTheExpectedOutputPixel) {
  for (const Case& c : kCases) {
    SCOPED_TRACE(c.name);
    const ResizeTransform t = *ComputeTransform(c.width, c.height, c.method);
    // 0 left of cropped column c_step, 1 from it on. Bilinear reaches 0.5
    // half-way between texel centres c_step-1 and c_step, i.e. at texel
    // position c_step - 0.5, which is source x = c_step / sx.
    const std::uint32_t c_step = t.new_w / 2;
    std::vector<float> depth(kNetDepthFloats, 0.0f);
    for (std::uint32_t y = t.pad_y; y < t.pad_y + t.new_h; ++y) {
      for (std::uint32_t x = t.pad_x + c_step; x < t.pad_x + t.new_w; ++x) {
        depth[std::size_t{y} * kNetSize + x] = 1.0f;
      }
    }
    DepthUploader uploader(*gpu_);
    ASSERT_TRUE(uploader.Upload(depth.data(), t));
    DepthDebugTexture target(*gpu_);
    DepthVisualizer vis(*gpu_, target);
    DepthVisParams params;
    params.lo = 0.0f;
    params.hi = 1.0f;
    params.valid = true;
    vis.Draw(uploader, t, params, CameraTexture(c.width, c.height));
    std::vector<std::uint8_t> rgba;
    ASSERT_TRUE(ReadbackTexture(*gpu_, target.texture(), rgba));

    const PixelRect rect = FitRect(t.src_w, t.src_h, DepthDebugTexture::kWidth, DepthDebugTexture::kHeight);
    const double source_x = c_step / t.sx();
    const double expected = rect.x + source_x * rect.width / t.src_w;  // output x where the value crosses 0.5
    const std::uint32_t row = rect.y + rect.height / 2;
    std::uint32_t first_bright = rect.x + rect.width;
    for (std::uint32_t x = rect.x; x < rect.x + rect.width; ++x) {
      if (rgba[(std::size_t{row} * DepthDebugTexture::kWidth + x) * 4] >= 128) {
        first_bright = x;
        break;
      }
    }
    // The first pixel whose centre is past the crossing point.
    EXPECT_NEAR(static_cast<double>(first_bright), std::ceil(expected - 0.5), 1.0)
        << "crossing at output x " << expected;
  }
}

TEST_F(DepthVisTest, FullCameraOverlayEqualsTheComposedOutput) {
  for (const Case& c : kCases) {
    SCOPED_TRACE(c.name);
    const ResizeTransform t = *ComputeTransform(c.width, c.height, c.method);
    const wgpu::Texture camera = CameraTexture(c.width, c.height);
    OutputTexture output(*gpu_);
    OutputCompositor compositor(*gpu_, output);
    compositor.Compose(camera);
    const std::vector<float> depth = SmoothDepth(t);
    DepthUploader uploader(*gpu_);
    ASSERT_TRUE(uploader.Upload(depth.data(), t));
    DepthDebugTexture target(*gpu_);
    DepthVisualizer vis(*gpu_, target);
    DepthVisParams params;
    params.valid = true;
    params.camera_weight = 1.0f;
    vis.Draw(uploader, t, params, camera);
    std::vector<std::uint8_t> composed, overlay;
    ASSERT_TRUE(ReadbackTexture(*gpu_, output.texture(), composed));
    ASSERT_TRUE(ReadbackTexture(*gpu_, target.texture(), overlay));
    EXPECT_EQ(composed, overlay) << "the overlay's camera layer is aligned with the output pixel for pixel";
  }
}

TEST_F(DepthVisTest, InvalidRangeIsNeutralAndClearIsBlack) {
  const ResizeTransform t = *ComputeTransform(1280, 720, ResizeMethod::kStretch);
  std::vector<float> depth(kNetDepthFloats, 3.0f);  // constant: no usable range
  DepthUploader uploader(*gpu_);
  ASSERT_TRUE(uploader.Upload(depth.data(), t));
  DepthDebugTexture target(*gpu_);
  DepthVisualizer vis(*gpu_, target);
  DepthVisParams params;  // valid = false
  vis.Draw(uploader, t, params, CameraTexture(1280, 720));
  std::vector<std::uint8_t> rgba;
  ASSERT_TRUE(ReadbackTexture(*gpu_, target.texture(), rgba));
  for (std::size_t i = 0; i < rgba.size(); i += 4) {
    ASSERT_NEAR(rgba[i], 128, 1);
  }
  vis.Clear();
  ASSERT_TRUE(ReadbackTexture(*gpu_, target.texture(), rgba));
  EXPECT_TRUE(std::all_of(rgba.begin(), rgba.end(), [n = 0](std::uint8_t v) mutable {
    return (n++ % 4 == 3) ? v == 255 : v == 0;
  }));
}

}  // namespace
}  // namespace relume
