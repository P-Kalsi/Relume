// Depth field and relighting on the GPU (offscreen Dawn), against the CPU
// reference in relume/lighting.hpp. Criteria fixed in advance (Phase 4 plan,
// M2, and clarification 3):
//   - the depth field matches BuildDepthField: height, confidence and the
//     normal at the largest relief within 2e-3 (16-bit float storage);
//   - the relit output matches ShadePixel on the composed camera pixel within
//     2 per 8-bit channel, also mid-crossfade, and outside the image is black;
//   - neutral settings and fade 0 equal passthrough (OutputCompositor) within
//     1 for every 8-bit code on every channel (target 0; the maximum and the
//     number of differing values are printed);
//   - a depth step lands on the expected output pixel within 1 pixel for
//     16:9, 4:3 and portrait sources, stretch and letterbox;
//   - moving the light left/right/up/down brightens that side of a sphere,
//     and the sphere's normals add to the asymmetry (relief > 0 vs flat);
//   - values in the letterbox padding never affect the output;
//   - the crossfade snapshot equals the CPU lerp within 2e-3 and refuses to
//     write into one of its inputs;
//   - every texture on the colour path is a non-sRGB format (no automatic
//     conversion).

#include "relume/image_io.hpp"
#include "relume/layout.hpp"
#include "relume/lighting.hpp"
#include "relume/pixel_convert.hpp"
#include "relume/preprocess.hpp"
#include "relume/renderer.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>

namespace relume {
namespace {

class RelightGpuTest : public ::testing::Test {
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

  wgpu::Texture Camera(const RgbImage& image) {
    FrameUploader uploader(*gpu_);
    VideoFrame frame;
    EncodeRgb8(image.pixels.data(), image.width, image.height, PixelFormat::kBgra8, YuvRange::kFull,
               YuvMatrix::kBt709, frame);
    frame.frame_id = 1;
    EXPECT_TRUE(uploader.Upload(frame));
    return uploader.camera_texture();
  }

  std::vector<std::uint8_t> Composed(const wgpu::Texture& camera) {
    OutputTexture output(*gpu_);
    OutputCompositor compositor(*gpu_, output);
    compositor.Compose(camera);
    std::vector<std::uint8_t> rgba;
    EXPECT_TRUE(ReadbackTexture(*gpu_, output.texture(), rgba));
    return rgba;
  }

  std::vector<std::uint8_t> Relit(const wgpu::Texture& camera, const RelightFrame& frame) {
    OutputTexture output(*gpu_);
    Relighter relighter(*gpu_, output);
    EXPECT_TRUE(relighter.Render(camera, frame));
    std::vector<std::uint8_t> rgba;
    EXPECT_TRUE(ReadbackTexture(*gpu_, output.texture(), rgba));
    return rgba;
  }

  static std::unique_ptr<GpuContext> gpu_;
  std::uint32_t errors_before_ = 0;
};

std::unique_ptr<GpuContext> RelightGpuTest::gpu_;

struct Case {
  const char* name;
  std::uint32_t width, height;
  ResizeMethod method;
};

const Case kCases[] = {
    {"16:9 stretch", 1280, 720, ResizeMethod::kStretch},
    {"16:9 letterbox", 1280, 720, ResizeMethod::kLetterbox},
    {"4:3 stretch (pillarboxed)", 640, 480, ResizeMethod::kStretch},
    {"portrait letterbox", 720, 1280, ResizeMethod::kLetterbox},
};

constexpr float kScale = 1.0f / 3.0f;  // depth 1..4 -> h about 0..1
constexpr float kOffset = -1.0f / 3.0f;

// Smooth depth with a near object (a step) on the right; `padding` fills
// the letterbox padding.
std::vector<float> TestDepth(const ResizeTransform& t, float padding = 1e6f) {
  std::vector<float> depth(kNetDepthFloats, padding);
  for (std::uint32_t y = 0; y < t.new_h; ++y) {
    for (std::uint32_t x = 0; x < t.new_w; ++x) {
      float d = 2.0f + std::sin(static_cast<float>(x) * 0.05f) + 0.6f * std::cos(static_cast<float>(y) * 0.031f);
      if (x * 10 >= t.new_w * 7) {
        d += 1.5f;
      }
      depth[std::size_t{t.pad_y + y} * kNetSize + t.pad_x + x] = d;
    }
  }
  return depth;
}

float Aspect(const ResizeTransform& t) { return static_cast<float>(t.src_w) / static_cast<float>(t.src_h); }

// Bilinear sample of a read-back field (4 floats per texel) at image uv, with
// the half-pixel rule and edge clamping a filtering sampler uses.
FieldTexel SampleField(const std::vector<float>& f, std::uint32_t w, std::uint32_t h, float u, float v) {
  const float px = std::clamp(u * static_cast<float>(w) - 0.5f, 0.0f, static_cast<float>(w - 1));
  const float py = std::clamp(v * static_cast<float>(h) - 0.5f, 0.0f, static_cast<float>(h - 1));
  const auto x0 = static_cast<std::uint32_t>(std::floor(px));
  const auto y0 = static_cast<std::uint32_t>(std::floor(py));
  const std::uint32_t x1 = std::min(x0 + 1, w - 1);
  const std::uint32_t y1 = std::min(y0 + 1, h - 1);
  const float fx = px - static_cast<float>(x0);
  const float fy = py - static_cast<float>(y0);
  float out[4];
  for (std::size_t c = 0; c < 4; ++c) {
    const auto at = [&](std::uint32_t x, std::uint32_t y) { return f[(std::size_t{y} * w + x) * 4 + c]; };
    const float top = at(x0, y0) * (1.0f - fx) + at(x1, y0) * fx;
    const float bottom = at(x0, y1) * (1.0f - fx) + at(x1, y1) * fx;
    out[c] = top * (1.0f - fy) + bottom * fy;
  }
  return {out[0], out[1], out[2], out[3]};
}

FieldTexel Lerp(const FieldTexel& a, const FieldTexel& b, float t) {
  return {a.sx + (b.sx - a.sx) * t, a.sy + (b.sy - a.sy) * t, a.h + (b.h - a.h) * t,
          a.conf + (b.conf - a.conf) * t};
}

LightSettings StrongLight() {
  LightSettings s;
  s.x = 0.2f;
  s.y = 0.15f;
  s.z = 0.8f;
  s.intensity = 1.2f;
  s.ambient = 0.45f;
  s.specular = 0.3f;
  s.tint = 0.6f;
  s.relief = 1.0f;
  return s;
}

bool InsideRect(const PixelRect& r, std::uint32_t x, std::uint32_t y) {
  return x >= r.x && x < r.x + r.width && y >= r.y && y < r.y + r.height;
}

TEST_F(RelightGpuTest, ColourPathUsesOnlyNonSrgbFormats) {
  const wgpu::Texture camera = Camera(MakeOrientationPattern(64, 36));
  EXPECT_EQ(camera.GetFormat(), wgpu::TextureFormat::RGBA8Unorm);
  OutputTexture output(*gpu_);
  EXPECT_EQ(output.texture().GetFormat(), wgpu::TextureFormat::RGBA8Unorm);
  DepthDebugTexture debug(*gpu_);
  EXPECT_EQ(debug.texture().GetFormat(), wgpu::TextureFormat::RGBA8Unorm);
  DepthFieldTexture field(*gpu_);
  field.Ensure(8, 8);
  EXPECT_EQ(field.texture().GetFormat(), wgpu::TextureFormat::RGBA16Float);
}

TEST_F(RelightGpuTest, DepthFieldMatchesTheCpuReference) {
  for (const Case& c : kCases) {
    SCOPED_TRACE(c.name);
    const ResizeTransform t = *ComputeTransform(c.width, c.height, c.method);
    const std::vector<float> depth = TestDepth(t);
    DepthUploader uploader(*gpu_);
    ASSERT_TRUE(uploader.Upload(depth.data(), t));
    NormalPass pass(*gpu_);
    DepthFieldTexture field(*gpu_);
    ASSERT_TRUE(pass.Build(uploader, kScale, kOffset, Aspect(t), field));
    std::vector<float> gpu_field;
    ASSERT_TRUE(ReadbackFieldTexture(*gpu_, field, gpu_field));
    ASSERT_EQ(gpu_field.size(), std::size_t{t.new_w} * t.new_h * 4);
    const std::vector<FieldTexel> cpu = BuildDepthField(&depth[std::size_t{t.pad_y} * kNetSize + t.pad_x], t.new_w,
                                                        t.new_h, kNetSize, kScale, kOffset, Aspect(t));
    float worst_h = 0.0f, worst_conf = 0.0f, worst_n = 0.0f;
    for (std::size_t i = 0; i < cpu.size(); ++i) {
      const FieldTexel g{gpu_field[i * 4], gpu_field[i * 4 + 1], gpu_field[i * 4 + 2], gpu_field[i * 4 + 3]};
      worst_h = std::max(worst_h, std::abs(g.h - cpu[i].h));
      worst_conf = std::max(worst_conf, std::abs(g.conf - cpu[i].conf));
      const Vec3 a = FieldNormal(g, 1.5f);
      const Vec3 b = FieldNormal(cpu[i], 1.5f);
      worst_n = std::max({worst_n, std::abs(a.x - b.x), std::abs(a.y - b.y), std::abs(a.z - b.z)});
    }
    std::printf("%s: field max |GPU - CPU|: h %.2e, conf %.2e, normal %.2e\n", c.name,
                static_cast<double>(worst_h), static_cast<double>(worst_conf), static_cast<double>(worst_n));
    EXPECT_LE(worst_h, 2e-3f);
    EXPECT_LE(worst_conf, 2e-3f);
    EXPECT_LE(worst_n, 2e-3f);
  }
}

TEST_F(RelightGpuTest, RelitOutputMatchesTheCpuReference) {
  for (const Case& c : kCases) {
    SCOPED_TRACE(c.name);
    const ResizeTransform t = *ComputeTransform(c.width, c.height, c.method);
    const wgpu::Texture camera = Camera(MakeOrientationPattern(c.width, c.height));
    const std::vector<std::uint8_t> composed = Composed(camera);

    // Two different fields, to check the crossfade too.
    DepthUploader uploader(*gpu_);
    NormalPass pass(*gpu_);
    DepthFieldTexture from(*gpu_);
    DepthFieldTexture to(*gpu_);
    std::vector<float> depth = TestDepth(t);
    ASSERT_TRUE(uploader.Upload(depth.data(), t));
    ASSERT_TRUE(pass.Build(uploader, kScale, kOffset, Aspect(t), from));
    for (float& d : depth) {
      d = d < 1e5f ? 5.0f - d : d;  // inverted relief
    }
    ASSERT_TRUE(uploader.Upload(depth.data(), t));
    ASSERT_TRUE(pass.Build(uploader, kScale, kOffset, Aspect(t), to));
    std::vector<float> f_from, f_to;
    ASSERT_TRUE(ReadbackFieldTexture(*gpu_, from, f_from));
    ASSERT_TRUE(ReadbackFieldTexture(*gpu_, to, f_to));

    struct Mix {
      float blend, fade;
    };
    for (const Mix m : {Mix{1.0f, 1.0f}, Mix{0.3f, 0.6f}}) {
      SCOPED_TRACE(testing::Message() << "blend " << m.blend << ", fade " << m.fade);
      RelightFrame frame;
      frame.from = &from;
      frame.to = &to;
      frame.blend = m.blend;
      frame.fade = m.fade;
      frame.light = StrongLight();
      const std::vector<std::uint8_t> relit = Relit(camera, frame);
      const PixelRect rect = FitRect(c.width, c.height, OutputTexture::kWidth, OutputTexture::kHeight);
      int worst = 0;
      bool bars_black = true;
      for (std::uint32_t y = 0; y < OutputTexture::kHeight; ++y) {
        for (std::uint32_t x = 0; x < OutputTexture::kWidth; ++x) {
          const std::size_t i = (std::size_t{y} * OutputTexture::kWidth + x) * 4;
          if (!InsideRect(rect, x, y)) {
            bars_black = bars_black && relit[i] == 0 && relit[i + 1] == 0 && relit[i + 2] == 0;
            continue;
          }
          const float u = (static_cast<float>(x - rect.x) + 0.5f) / static_cast<float>(rect.width);
          const float v = (static_cast<float>(y - rect.y) + 0.5f) / static_cast<float>(rect.height);
          const FieldTexel ft = Lerp(SampleField(f_from, t.new_w, t.new_h, u, v),
                                     SampleField(f_to, t.new_w, t.new_h, u, v), m.blend);
          const std::array<std::uint8_t, 3> expected =
              ShadePixel({composed[i], composed[i + 1], composed[i + 2]}, ft, u, v, frame.light, Aspect(t), m.fade);
          for (std::size_t ch = 0; ch < 3; ++ch) {
            worst = std::max(worst, std::abs(int{relit[i + ch]} - int{expected[ch]}));
          }
        }
      }
      std::printf("%s (blend %.1f, fade %.1f): relit max |GPU - CPU| %d\n", c.name, static_cast<double>(m.blend),
                  static_cast<double>(m.fade), worst);
      EXPECT_LE(worst, 2);
      EXPECT_TRUE(bars_black);
    }
  }
}

// Every code 0..255 on every channel, in rows: grey, then each channel
// alone, then each channel against the others at their extremes.
RgbImage AllCodes(std::uint32_t width, std::uint32_t height) {
  RgbImage image{width, height, std::vector<std::uint8_t>(std::size_t{width} * height * 3)};
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      const auto code = static_cast<std::uint8_t>((x + y * 37) % 256);
      const auto other = static_cast<std::uint8_t>(((y / 8) % 2) * 255);
      std::uint8_t* p = &image.pixels[(std::size_t{y} * width + x) * 3];
      switch ((y / 4) % 4) {
        case 0: p[0] = p[1] = p[2] = code; break;
        case 1: p[0] = code; p[1] = other; p[2] = static_cast<std::uint8_t>(255 - other); break;
        case 2: p[1] = code; p[0] = other; p[2] = static_cast<std::uint8_t>(255 - other); break;
        default: p[2] = code; p[0] = other; p[1] = static_cast<std::uint8_t>(255 - other); break;
      }
    }
  }
  return image;
}

TEST_F(RelightGpuTest, NeutralSettingsAndZeroFadeEqualPassthrough) {
  for (const Case& c : {kCases[0], kCases[2]}) {
    SCOPED_TRACE(c.name);
    const ResizeTransform t = *ComputeTransform(c.width, c.height, c.method);
    const wgpu::Texture camera = Camera(AllCodes(c.width, c.height));
    const std::vector<std::uint8_t> composed = Composed(camera);
    const std::vector<float> depth = TestDepth(t);
    DepthUploader uploader(*gpu_);
    ASSERT_TRUE(uploader.Upload(depth.data(), t));
    NormalPass pass(*gpu_);
    DepthFieldTexture field(*gpu_);
    ASSERT_TRUE(pass.Build(uploader, kScale, kOffset, Aspect(t), field));
    RelightFrame neutral;
    neutral.from = neutral.to = &field;
    neutral.light = NeutralLightSettings();
    RelightFrame no_fade = neutral;
    no_fade.light = StrongLight();
    no_fade.fade = 0.0f;
    for (const RelightFrame& frame : {neutral, no_fade}) {
      const std::vector<std::uint8_t> relit = Relit(camera, frame);
      int worst = 0;
      std::size_t differing = 0;
      for (std::size_t i = 0; i < relit.size(); ++i) {
        const int d = std::abs(int{relit[i]} - int{composed[i]});
        worst = std::max(worst, d);
        differing += d != 0 ? 1 : 0;
      }
      std::printf("%s, %s: max |relit - passthrough| %d, %zu differing values\n", c.name,
                  frame.fade == 0.0f ? "fade 0" : "neutral settings", worst, differing);
      EXPECT_LE(worst, 1);
    }
  }
}

TEST_F(RelightGpuTest, DepthStepLandsOnTheExpectedOutputPixel) {
  const Case cases[] = {kCases[0], kCases[1], kCases[2], kCases[3], {"4:3 letterbox", 640, 480, ResizeMethod::kLetterbox}};
  for (const Case& c : cases) {
    SCOPED_TRACE(c.name);
    const ResizeTransform t = *ComputeTransform(c.width, c.height, c.method);
    // h = 0.2 left of cropped column c_step, 0.9 from it on; the bilinear
    // value crosses 0.55 at texel position c_step - 0.5, source x c_step / sx.
    const std::uint32_t c_step = t.new_w / 2;
    std::vector<float> depth(kNetDepthFloats, 1e6f);
    for (std::uint32_t y = t.pad_y; y < t.pad_y + t.new_h; ++y) {
      for (std::uint32_t x = t.pad_x; x < t.pad_x + t.new_w; ++x) {
        depth[std::size_t{y} * kNetSize + x] = x >= t.pad_x + c_step ? 0.9f : 0.2f;
      }
    }
    DepthUploader uploader(*gpu_);
    ASSERT_TRUE(uploader.Upload(depth.data(), t));
    NormalPass pass(*gpu_);
    DepthFieldTexture field(*gpu_);
    ASSERT_TRUE(pass.Build(uploader, 1.0f, 0.0f, Aspect(t), field));
    OutputTexture output(*gpu_);
    Relighter relighter(*gpu_, output);
    DepthDebugTexture target(*gpu_);
    RelightFrame frame;
    frame.from = frame.to = &field;
    const wgpu::Texture camera = Camera(MakeOrientationPattern(c.width, c.height));
    ASSERT_TRUE(relighter.RenderDebug(camera, frame, RelightView::kHeight, target));
    std::vector<std::uint8_t> rgba;
    ASSERT_TRUE(ReadbackTexture(*gpu_, target.texture(), rgba));

    const PixelRect rect = FitRect(t.src_w, t.src_h, OutputTexture::kWidth, OutputTexture::kHeight);
    const double expected = rect.x + (c_step / t.sx()) * rect.width / t.src_w;
    const std::uint32_t row = rect.y + rect.height / 2;
    std::uint32_t first_near = rect.x + rect.width;
    for (std::uint32_t x = rect.x; x < rect.x + rect.width; ++x) {
      if (rgba[(std::size_t{row} * OutputTexture::kWidth + x) * 4] >= 141) {  // 0.55 * 255 = 140.25
        first_near = x;
        break;
      }
    }
    EXPECT_NEAR(static_cast<double>(first_near), std::ceil(expected - 0.5), 1.0) << "crossing at output x " << expected;
  }
}

TEST_F(RelightGpuTest, PaddingNeverAffectsTheOutput) {
  const ResizeTransform t = *ComputeTransform(1280, 720, ResizeMethod::kLetterbox);
  ASSERT_GT(t.pad_y, 0u);
  const wgpu::Texture camera = Camera(MakeOrientationPattern(1280, 720));
  std::vector<std::vector<std::uint8_t>> outputs;
  for (const float padding : {1e6f, 0.0f, -1e6f}) {
    const std::vector<float> depth = TestDepth(t, padding);
    DepthUploader uploader(*gpu_);
    ASSERT_TRUE(uploader.Upload(depth.data(), t));
    NormalPass pass(*gpu_);
    DepthFieldTexture field(*gpu_);
    ASSERT_TRUE(pass.Build(uploader, kScale, kOffset, Aspect(t), field));
    RelightFrame frame;
    frame.from = frame.to = &field;
    frame.light = StrongLight();
    outputs.push_back(Relit(camera, frame));
  }
  EXPECT_EQ(outputs[0], outputs[1]);
  EXPECT_EQ(outputs[0], outputs[2]);
}

// Mean luminance of the sphere's halves, relit from a light at (x, y).
struct Halves {
  double left = 0, right = 0, top = 0, bottom = 0;
};

TEST_F(RelightGpuTest, LightMovesInTheExpectedDirection) {
  const std::uint32_t w = 1280, h = 720;
  const ResizeTransform t = *ComputeTransform(w, h, ResizeMethod::kStretch);
  const float aspect = Aspect(t);
  // A hemisphere (radius 0.35 of the height) on a flat background.
  std::vector<float> depth(kNetDepthFloats);
  for (std::uint32_t y = 0; y < kNetSize; ++y) {
    for (std::uint32_t x = 0; x < kNetSize; ++x) {
      const float u = (static_cast<float>(x) + 0.5f) / kNetSize;
      const float v = (static_cast<float>(y) + 0.5f) / kNetSize;
      const float dx = (u - 0.5f) * aspect;
      const float dy = v - 0.5f;
      const float r2 = (dx * dx + dy * dy) / (0.35f * 0.35f);
      depth[std::size_t{y} * kNetSize + x] = r2 < 1.0f ? 0.3f + 0.7f * std::sqrt(1.0f - r2) : 0.3f;
    }
  }
  DepthUploader uploader(*gpu_);
  ASSERT_TRUE(uploader.Upload(depth.data(), t));
  NormalPass pass(*gpu_);
  DepthFieldTexture field(*gpu_);
  ASSERT_TRUE(pass.Build(uploader, 1.0f, 0.0f, aspect, field));
  RgbImage grey{w, h, std::vector<std::uint8_t>(std::size_t{w} * h * 3, 128)};
  const wgpu::Texture camera = Camera(grey);

  const auto measure = [&](float lx, float ly, float relief) {
    RelightFrame frame;
    frame.from = frame.to = &field;
    frame.light.x = lx;
    frame.light.y = ly;
    frame.light.specular = 0.0f;
    frame.light.relief = relief;
    const std::vector<std::uint8_t> rgba = Relit(camera, frame);
    Halves halves;
    std::size_t n_l = 0, n_r = 0, n_t = 0, n_b = 0;
    for (std::uint32_t y = 0; y < h; ++y) {
      for (std::uint32_t x = 0; x < w; ++x) {
        const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(w);
        const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(h);
        const float dx = (u - 0.5f) * aspect;
        const float dy = v - 0.5f;
        if (dx * dx + dy * dy > 0.28f * 0.28f) {
          continue;  // inside the sphere, away from its rim
        }
        const double value = rgba[(std::size_t{y} * w + x) * 4 + 1];
        (dx < 0 ? halves.left : halves.right) += value;
        (dx < 0 ? n_l : n_r) += 1;
        (dy < 0 ? halves.top : halves.bottom) += value;
        (dy < 0 ? n_t : n_b) += 1;
      }
    }
    halves.left /= static_cast<double>(n_l);
    halves.right /= static_cast<double>(n_r);
    halves.top /= static_cast<double>(n_t);
    halves.bottom /= static_cast<double>(n_b);
    return halves;
  };
  constexpr double kMargin = 3.0;  // 8-bit levels
  const Halves left = measure(0.1f, 0.5f, 1.0f);
  const Halves right = measure(0.9f, 0.5f, 1.0f);
  const Halves top = measure(0.5f, 0.1f, 1.0f);
  const Halves bottom = measure(0.5f, 0.9f, 1.0f);
  std::printf("light left: L %.1f R %.1f | right: L %.1f R %.1f | top: T %.1f B %.1f | bottom: T %.1f B %.1f\n",
              left.left, left.right, right.left, right.right, top.top, top.bottom, bottom.top, bottom.bottom);
  EXPECT_GT(left.left, left.right + kMargin);
  EXPECT_GT(right.right, right.left + kMargin);
  EXPECT_GT(top.top, top.bottom + kMargin);
  EXPECT_GT(bottom.bottom, bottom.top + kMargin);
  // The normals must add to the asymmetry that distance alone gives (a
  // flipped normal would reduce it).
  const Halves flat_left = measure(0.1f, 0.5f, 0.0f);
  const Halves flat_top = measure(0.5f, 0.1f, 0.0f);
  std::printf("flat (relief 0): light left L-R %.1f, light top T-B %.1f\n", flat_left.left - flat_left.right,
              flat_top.top - flat_top.bottom);
  EXPECT_GT(left.left - left.right, flat_left.left - flat_left.right + kMargin);
  EXPECT_GT(top.top - top.bottom, flat_top.top - flat_top.bottom + kMargin);
}

TEST_F(RelightGpuTest, BlendSnapshotMatchesTheCpuLerpAndRefusesAliasing) {
  const ResizeTransform t = *ComputeTransform(1280, 720, ResizeMethod::kStretch);
  DepthUploader uploader(*gpu_);
  NormalPass pass(*gpu_);
  DepthFieldTexture a(*gpu_), b(*gpu_), out(*gpu_);
  std::vector<float> depth = TestDepth(t);
  ASSERT_TRUE(uploader.Upload(depth.data(), t));
  ASSERT_TRUE(pass.Build(uploader, kScale, kOffset, Aspect(t), a));
  for (float& d : depth) {
    d = 5.0f - d;
  }
  ASSERT_TRUE(uploader.Upload(depth.data(), t));
  ASSERT_TRUE(pass.Build(uploader, kScale, kOffset, Aspect(t), b));
  ASSERT_TRUE(pass.Blend(a, b, 0.3f, out));
  std::vector<float> fa, fb, fo;
  ASSERT_TRUE(ReadbackFieldTexture(*gpu_, a, fa));
  ASSERT_TRUE(ReadbackFieldTexture(*gpu_, b, fb));
  ASSERT_TRUE(ReadbackFieldTexture(*gpu_, out, fo));
  float worst = 0.0f;
  for (std::size_t i = 0; i < fo.size(); ++i) {
    worst = std::max(worst, std::abs(fo[i] - (fa[i] + (fb[i] - fa[i]) * 0.3f)));
  }
  EXPECT_LE(worst, 2e-3f);
  EXPECT_FALSE(pass.Blend(a, b, 0.5f, a));
  EXPECT_FALSE(pass.Blend(a, b, 0.5f, b));
}

TEST_F(RelightGpuTest, RenderRefusesMissingOrMismatchedFields) {
  const wgpu::Texture camera = Camera(MakeOrientationPattern(64, 36));
  OutputTexture output(*gpu_);
  Relighter relighter(*gpu_, output);
  DepthFieldTexture a(*gpu_), b(*gpu_);
  RelightFrame frame;
  EXPECT_FALSE(relighter.Render(camera, frame));
  frame.from = &a;
  frame.to = &b;
  EXPECT_FALSE(relighter.Render(camera, frame));  // never built
  a.Ensure(16, 16);
  b.Ensure(16, 8);
  EXPECT_FALSE(relighter.Render(camera, frame));  // sizes differ
  EXPECT_FALSE(relighter.Render(wgpu::Texture{}, frame));
}

}  // namespace
}  // namespace relume
