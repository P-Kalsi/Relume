#include "relume/preprocess.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace relume {
namespace {

// tests/python/golden/preprocess_golden.txt, written by
// scripts/relume_model/golden.py from the committed golden JSON.
struct GoldenSample {
  std::uint32_t y = 0;
  std::uint32_t x = 0;
  double rgb[3] = {};
};

struct GoldenCase {
  std::string method;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  ResizeTransform transform;
  std::vector<std::uint8_t> rgb;
  std::vector<GoldenSample> samples;
};

std::vector<std::uint8_t> FromHex(const std::string& hex) {
  std::vector<std::uint8_t> bytes(hex.size() / 2);
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<std::uint8_t>(std::stoul(hex.substr(2 * i, 2), nullptr, 16));
  }
  return bytes;
}

std::vector<GoldenCase> LoadGolden() {
  std::ifstream in(RELUME_GOLDEN_TEXT);
  EXPECT_TRUE(in) << "cannot open " << RELUME_GOLDEN_TEXT;
  std::vector<GoldenCase> cases;
  std::size_t expected_cases = 0;
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream fields(line);
    std::string tag;
    fields >> tag;
    if (tag.empty() || tag[0] == '#') {
      continue;
    }
    if (tag == "cases") {
      fields >> expected_cases;
    } else if (tag == "case") {
      GoldenCase c;
      std::size_t samples = 0;
      fields >> c.method >> c.width >> c.height >> c.transform.new_w >> c.transform.new_h >> c.transform.pad_x >>
          c.transform.pad_y >> samples;
      c.transform.method = ParseResizeMethod(c.method).value_or(ResizeMethod::kStretch);
      c.transform.src_w = c.width;
      c.transform.src_h = c.height;
      c.samples.reserve(samples);
      cases.push_back(std::move(c));
    } else if (tag == "rgb") {
      std::string hex;
      fields >> hex;
      cases.back().rgb = FromHex(hex);
    } else if (tag == "s") {
      GoldenSample s;
      fields >> s.y >> s.x >> s.rgb[0] >> s.rgb[1] >> s.rgb[2];
      cases.back().samples.push_back(s);
    }
    EXPECT_FALSE(fields.fail()) << "bad golden line: " << line;
  }
  EXPECT_EQ(cases.size(), expected_cases);
  return cases;
}

float TensorAt(const std::vector<float>& tensor, std::uint32_t c, std::uint32_t y, std::uint32_t x) {
  return tensor[(std::size_t{c} * kNetSize + y) * kNetSize + x];
}

// Every golden case: exact transform, every sample within 1e-6 (the stated
// contract). The largest difference seen is printed.
TEST(Preprocess, MatchesPythonGoldenVectors) {
  const std::vector<GoldenCase> cases = LoadGolden();
  ASSERT_EQ(cases.size(), 12u);
  Preprocessor pre;
  std::vector<float> tensor(kNetTensorFloats);
  double worst = 0.0;
  std::size_t methods_seen[2] = {};
  for (const GoldenCase& c : cases) {
    SCOPED_TRACE(c.method + " " + std::to_string(c.width) + "x" + std::to_string(c.height));
    ASSERT_TRUE(ParseResizeMethod(c.method).has_value());
    const ResizeMethod method = *ParseResizeMethod(c.method);
    ++methods_seen[static_cast<int>(method)];
    ASSERT_EQ(c.rgb.size(), std::size_t{c.width} * c.height * 3);
    ASSERT_EQ(c.samples.size(), 24u);
    const std::optional<ResizeTransform> t = pre.Run(c.rgb.data(), c.width, c.height, method, tensor.data());
    ASSERT_TRUE(t.has_value());
    EXPECT_EQ(*t, c.transform);
    for (const GoldenSample& s : c.samples) {
      for (std::uint32_t ch = 0; ch < 3; ++ch) {
        const double diff = std::abs(static_cast<double>(TensorAt(tensor, ch, s.y, s.x)) - s.rgb[ch]);
        worst = std::max(worst, diff);
        EXPECT_LE(diff, 1e-6) << "y " << s.y << " x " << s.x << " channel " << ch;
      }
    }
  }
  EXPECT_EQ(methods_seen[0], 6u);
  EXPECT_EQ(methods_seen[1], 6u);
  std::printf("[golden] 12 cases x 24 samples x 3 channels; max |C++ - Python| = %.3g\n", worst);
}

TEST(Preprocess, DefaultMethodIsStretch) {
  // Approved 2026-10-26 (docs/decisions.md); letterbox stays supported.
  EXPECT_EQ(kDefaultResizeMethod, ResizeMethod::kStretch);
  EXPECT_EQ(ComputeTransform(1280, 720)->method, ResizeMethod::kStretch);
  EXPECT_EQ(ToString(kDefaultResizeMethod), "stretch");
  EXPECT_EQ(ParseResizeMethod("letterbox"), ResizeMethod::kLetterbox);
  EXPECT_EQ(ParseResizeMethod("stretch"), ResizeMethod::kStretch);
  EXPECT_FALSE(ParseResizeMethod("crop").has_value());
}

TEST(Preprocess, LetterboxPlacementMatchesPython) {
  struct Case {
    std::uint32_t w, h, new_w, new_h, pad_x, pad_y;
  };
  // Same table as tests/python/test_preprocess.py.
  const Case cases[] = {
      {1280, 720, 448, 252, 0, 98}, {720, 1280, 252, 448, 98, 0}, {1920, 1080, 448, 252, 0, 98},
      {333, 517, 289, 448, 79, 0},  {517, 333, 448, 289, 0, 79},  {1, 1, 448, 448, 0, 0},
      {448, 448, 448, 448, 0, 0},
  };
  for (const Case& c : cases) {
    const ResizeTransform t = *ComputeTransform(c.w, c.h, ResizeMethod::kLetterbox);
    EXPECT_EQ(t.new_w, c.new_w) << c.w << "x" << c.h;
    EXPECT_EQ(t.new_h, c.new_h) << c.w << "x" << c.h;
    EXPECT_EQ(t.pad_x, c.pad_x) << c.w << "x" << c.h;
    EXPECT_EQ(t.pad_y, c.pad_y) << c.w << "x" << c.h;
  }
}

TEST(Preprocess, RoundsHalfToEvenLikePython) {
  // 3x2: s = 448/3; 2 * 448/3 = 298.67 -> 299. A case with an exact .5:
  // 896x1: s = 0.5, 1 * 0.5 = 0.5 -> 0 -> max(1, 0) = 1; 896x3: 1.5 -> 2;
  // 896x5: 2.5 -> 2 (half to even).
  EXPECT_EQ(ComputeTransform(896, 3, ResizeMethod::kLetterbox)->new_h, 2u);
  EXPECT_EQ(ComputeTransform(896, 5, ResizeMethod::kLetterbox)->new_h, 2u);
  EXPECT_EQ(ComputeTransform(896, 7, ResizeMethod::kLetterbox)->new_h, 4u);  // 3.5 -> 4
  EXPECT_EQ(ComputeTransform(896, 1, ResizeMethod::kLetterbox)->new_h, 1u);
}

TEST(Preprocess, StretchPlacement) {
  const ResizeTransform t = *ComputeTransform(1280, 720, ResizeMethod::kStretch);
  EXPECT_EQ(t.new_w, 448u);
  EXPECT_EQ(t.new_h, 448u);
  EXPECT_EQ(t.pad_x, 0u);
  EXPECT_EQ(t.pad_y, 0u);
  EXPECT_DOUBLE_EQ(t.sx(), 448.0 / 1280.0);
  EXPECT_DOUBLE_EQ(t.sy(), 448.0 / 720.0);
}

TEST(Preprocess, EmptyImageIsRejected) {
  EXPECT_FALSE(ComputeTransform(0, 720).has_value());
  EXPECT_FALSE(ComputeTransform(1280, 0, ResizeMethod::kLetterbox).has_value());
  Preprocessor pre;
  std::vector<float> tensor(kNetTensorFloats, 7.0f);
  EXPECT_FALSE(pre.Run(nullptr, 0, 0, ResizeMethod::kStretch, tensor.data()).has_value());
}

const std::uint32_t kSizes[][2] = {{1280, 720}, {720, 1280}, {333, 517}, {517, 333},
                                   {448, 448},  {1, 1},      {1920, 1080}, {17, 9}};

TEST(Preprocess, PointMappingRoundTrips) {
  for (const ResizeMethod method : {ResizeMethod::kLetterbox, ResizeMethod::kStretch}) {
    for (const auto& size : kSizes) {
      const ResizeTransform t = *ComputeTransform(size[0], size[1], method);
      std::mt19937 rng(size[0] * 31 + size[1]);
      std::uniform_real_distribution<double> ux(0.0, size[0]);
      std::uniform_real_distribution<double> uy(0.0, size[1]);
      for (int i = 0; i < 1000; ++i) {
        const double x = ux(rng);
        const double y = uy(rng);
        EXPECT_NEAR(t.NetToSrcX(t.SrcToNetX(x)), x, 1e-9);
        EXPECT_NEAR(t.NetToSrcY(t.SrcToNetY(y)), y, 1e-9);
      }
      // Source corners land on the edges of the placed region.
      EXPECT_NEAR(t.SrcToNetX(0), t.pad_x, 1e-9);
      EXPECT_NEAR(t.SrcToNetY(0), t.pad_y, 1e-9);
      EXPECT_NEAR(t.SrcToNetX(size[0]), t.pad_x + t.new_w, 1e-9);
      EXPECT_NEAR(t.SrcToNetY(size[1]), t.pad_y + t.new_h, 1e-9);
    }
  }
}

TEST(Preprocess, BilinearClosedForm) {
  // 2 pixels [0, 1] upsampled to 4 samples at u = 0 (clamped), 0.25, 0.75, 1 (clamped).
  const float row[2] = {0.0f, 1.0f};
  float out[4] = {};
  ResizeScratch scratch;
  ResizeBilinear(row, 2, 1, 2, 1, out, 4, 1, scratch);
  EXPECT_FLOAT_EQ(out[0], 0.0f);
  EXPECT_FLOAT_EQ(out[1], 0.25f);
  EXPECT_FLOAT_EQ(out[2], 0.75f);
  EXPECT_FLOAT_EQ(out[3], 1.0f);
}

TEST(Preprocess, BilinearIdentityIsExact) {
  std::vector<float> image(64 * 48 * 3);
  for (std::size_t i = 0; i < image.size(); ++i) {
    image[i] = static_cast<float>(i % 251) / 251.0f;
  }
  std::vector<float> out(image.size());
  ResizeScratch scratch;
  ResizeBilinear(image.data(), 64, 48, 64 * 3, 3, out.data(), 64, 48, scratch);
  EXPECT_EQ(out, image);
}

// The resize samples exactly where NetToSrc says each tensor pixel comes
// from: a ramp whose value is its own x index is reproduced by bilinear
// interpolation, so the resized ramp must equal the mapped coordinate.
TEST(Preprocess, ResamplerMatchesPointMapping) {
  for (const ResizeMethod method : {ResizeMethod::kLetterbox, ResizeMethod::kStretch}) {
    for (const auto& size : kSizes) {
      const std::uint32_t w = size[0];
      const std::uint32_t h = size[1];
      if (std::min(w, h) < 2) {
        continue;
      }
      const ResizeTransform t = *ComputeTransform(w, h, method);
      std::vector<float> ramp(std::size_t{w} * h);
      for (std::uint32_t y = 0; y < h; ++y) {
        for (std::uint32_t x = 0; x < w; ++x) {
          ramp[std::size_t{y} * w + x] = static_cast<float>(x);
        }
      }
      std::vector<float> resized(std::size_t{t.new_w} * t.new_h);
      ResizeScratch scratch;
      ResizeBilinear(ramp.data(), w, h, w, 1, resized.data(), t.new_w, t.new_h, scratch);
      for (std::uint32_t j = 0; j < t.new_w; ++j) {
        const double mapped = std::clamp(t.NetToSrcX(t.pad_x + j + 0.5) - 0.5, 0.0, w - 1.0);
        // 1e-4, widened to a few float32 steps for large coordinates (one
        // step is 1.2e-4 at x = 1913).
        const double bound = std::max(1e-4, 4.0 * std::numeric_limits<float>::epsilon() * mapped);
        EXPECT_NEAR(resized[j], mapped, bound) << ToString(method) << " " << w << "x" << h << " column " << j;
      }
    }
  }
}

std::vector<std::uint8_t> NoiseImage(std::uint32_t w, std::uint32_t h, unsigned seed) {
  std::mt19937 rng(seed);
  std::vector<std::uint8_t> rgb(std::size_t{w} * h * 3);
  for (std::uint8_t& v : rgb) {
    v = static_cast<std::uint8_t>(rng() & 0xFF);
  }
  return rgb;
}

TEST(Preprocess, LetterboxPaddingIsExactlyZero) {
  const std::vector<std::uint8_t> rgb = NoiseImage(1280, 720, 1);
  Preprocessor pre;
  std::vector<float> tensor(kNetTensorFloats, 9.0f);
  const ResizeTransform t = *pre.Run(rgb.data(), 1280, 720, ResizeMethod::kLetterbox, tensor.data());
  ASSERT_EQ(t.pad_y, 98u);
  for (std::uint32_t c = 0; c < 3; ++c) {
    for (std::uint32_t y = 0; y < kNetSize; ++y) {
      if (y >= t.pad_y && y < t.pad_y + t.new_h) {
        continue;
      }
      for (std::uint32_t x = 0; x < kNetSize; ++x) {
        ASSERT_EQ(TensorAt(tensor, c, y, x), 0.0f) << c << " " << y << " " << x;
      }
    }
  }
}

TEST(Preprocess, StretchFillsTheWholeTensor) {
  const std::vector<std::uint8_t> rgb = NoiseImage(1280, 720, 2);
  Preprocessor pre;
  std::vector<float> tensor(kNetTensorFloats, std::nanf(""));
  pre.Run(rgb.data(), 1280, 720, ResizeMethod::kStretch, tensor.data());
  EXPECT_TRUE(std::all_of(tensor.begin(), tensor.end(), [](float v) { return std::isfinite(v); }));
  // Every channel stays within the normalized range of [0, 1] input.
  EXPECT_LE(*std::max_element(tensor.begin(), tensor.end()), (1.0f - 0.406f) / 0.225f + 1e-5f);
  EXPECT_GE(*std::min_element(tensor.begin(), tensor.end()), (0.0f - 0.485f) / 0.229f - 1e-5f);
}

TEST(Preprocess, ChannelOrderIsRgb) {
  std::vector<std::uint8_t> red(20 * 10 * 3, 0);
  for (std::size_t i = 0; i < red.size(); i += 3) {
    red[i] = 255;
  }
  Preprocessor pre;
  std::vector<float> tensor(kNetTensorFloats);
  pre.Run(red.data(), 20, 10, kDefaultResizeMethod, tensor.data());
  const float mean[3] = {0.485f, 0.456f, 0.406f};
  const float stdev[3] = {0.229f, 0.224f, 0.225f};
  for (std::uint32_t c = 0; c < 3; ++c) {
    const float expected = ((c == 0 ? 1.0f : 0.0f) - mean[c]) / stdev[c];
    EXPECT_NEAR(TensorAt(tensor, c, kNetSize / 2, kNetSize / 2), expected, 1e-6f);
  }
}

TEST(Preprocess, SteadyStateReusesBuffers) {
  const std::vector<std::uint8_t> rgb = NoiseImage(1280, 720, 3);
  ResizeScratch scratch;
  std::vector<float> image(rgb.begin(), rgb.end());
  std::vector<float> out(448 * 448 * 3);
  ResizeBilinear(image.data(), 1280, 720, 1280 * 3, 3, out.data(), 448, 448, scratch);
  const float* rows = scratch.rows.data();
  const std::uint32_t* x0 = scratch.x0.data();
  ResizeBilinear(image.data(), 1280, 720, 1280 * 3, 3, out.data(), 448, 448, scratch);
  EXPECT_EQ(scratch.rows.data(), rows);
  EXPECT_EQ(scratch.x0.data(), x0);

  // Repeated runs give identical tensors.
  Preprocessor pre;
  std::vector<float> first(kNetTensorFloats);
  std::vector<float> second(kNetTensorFloats);
  pre.Run(rgb.data(), 1280, 720, kDefaultResizeMethod, first.data());
  pre.Run(rgb.data(), 1280, 720, kDefaultResizeMethod, second.data());
  EXPECT_EQ(first, second);
}

// The inverse: constant depth stays constant; letterbox padding never leaks
// into the source-resolution depth; stretch of a 448x448 source is exact.
TEST(DepthToSource, CropsPaddingAndResamples) {
  ResizeScratch scratch;
  std::vector<float> out;
  {
    const ResizeTransform t = *ComputeTransform(1280, 720, ResizeMethod::kStretch);
    std::vector<float> depth(kNetDepthFloats, 1.0f);
    DepthToSource(depth.data(), t, out, scratch);
    ASSERT_EQ(out.size(), 1280u * 720u);
    EXPECT_TRUE(std::all_of(out.begin(), out.end(), [](float v) { return v == 1.0f; }));
  }
  for (const auto& size : kSizes) {
    const ResizeTransform t = *ComputeTransform(size[0], size[1], ResizeMethod::kLetterbox);
    std::vector<float> depth(kNetDepthFloats, 1e6f);  // padding
    for (std::uint32_t y = t.pad_y; y < t.pad_y + t.new_h; ++y) {
      for (std::uint32_t x = t.pad_x; x < t.pad_x + t.new_w; ++x) {
        depth[std::size_t{y} * kNetSize + x] = 2.0f;
      }
    }
    DepthToSource(depth.data(), t, out, scratch);
    ASSERT_EQ(out.size(), std::size_t{size[0]} * size[1]);
    EXPECT_TRUE(std::all_of(out.begin(), out.end(), [](float v) { return v == 2.0f; }))
        << size[0] << "x" << size[1];
  }
  {
    const ResizeTransform t = *ComputeTransform(448, 448, ResizeMethod::kStretch);
    std::vector<float> depth(kNetDepthFloats);
    for (std::size_t i = 0; i < depth.size(); ++i) {
      depth[i] = static_cast<float>(i);
    }
    DepthToSource(depth.data(), t, out, scratch);
    EXPECT_EQ(out, depth);
  }
}

}  // namespace
}  // namespace relume
