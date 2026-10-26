#include "relume/preprocess.hpp"

#include <algorithm>
#include <array>
#include <cmath>

// Compiled with -ffp-contract=off (src/CMakeLists.txt): every product and sum
// below is rounded separately, as numpy does.

namespace relume {

namespace {

constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};  // ImageNet, RGB order
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

// _axis_samples() in preprocess.py: positions in double, weights in float.
void AxisSamples(std::uint32_t src_len, std::uint32_t dst_len, std::vector<std::uint32_t>& i0s,
                 std::vector<std::uint32_t>& i1s, std::vector<float>& w1s) {
  i0s.resize(dst_len);
  i1s.resize(dst_len);
  w1s.resize(dst_len);
  const double scale = static_cast<double>(src_len) / static_cast<double>(dst_len);
  const double last = static_cast<double>(src_len - 1);
  for (std::uint32_t j = 0; j < dst_len; ++j) {
    double u = (static_cast<double>(j) + 0.5) * scale - 0.5;
    u = std::clamp(u, 0.0, last);
    const double lower = std::floor(u);
    const auto i0 = static_cast<std::uint32_t>(lower);
    i0s[j] = i0;
    i1s[j] = std::min(i0 + 1, src_len - 1);
    w1s[j] = static_cast<float>(u - lower);
  }
}

// x / 255 in float32 for every byte value.
const std::array<float, 256>& UnitLut() {
  static const std::array<float, 256> lut = [] {
    std::array<float, 256> values{};
    for (std::size_t i = 0; i < values.size(); ++i) {
      values[i] = static_cast<float>(i) / 255.0f;
    }
    return values;
  }();
  return lut;
}

}  // namespace

std::string_view ToString(ResizeMethod method) {
  switch (method) {
    case ResizeMethod::kStretch: return "stretch";
    case ResizeMethod::kLetterbox: return "letterbox";
  }
  return "?";
}

std::optional<ResizeMethod> ParseResizeMethod(std::string_view name) {
  if (name == "stretch") {
    return ResizeMethod::kStretch;
  }
  if (name == "letterbox") {
    return ResizeMethod::kLetterbox;
  }
  return std::nullopt;
}

std::optional<ResizeTransform> ComputeTransform(std::uint32_t src_w, std::uint32_t src_h, ResizeMethod method) {
  if (src_w == 0 || src_h == 0) {
    return std::nullopt;
  }
  ResizeTransform t;
  t.method = method;
  t.src_w = src_w;
  t.src_h = src_h;
  if (method == ResizeMethod::kStretch) {
    t.new_w = kNetSize;
    t.new_h = kNetSize;
    return t;
  }
  const double scale = static_cast<double>(kNetSize) / static_cast<double>(std::max(src_w, src_h));
  // Python's round() rounds half to even, as nearbyint does in the default
  // rounding mode.
  const auto fit = [&](std::uint32_t dim) {
    const double rounded = std::nearbyint(static_cast<double>(dim) * scale);
    return static_cast<std::uint32_t>(std::clamp(rounded, 1.0, static_cast<double>(kNetSize)));
  };
  t.new_w = fit(src_w);
  t.new_h = fit(src_h);
  t.pad_x = (kNetSize - t.new_w) / 2;
  t.pad_y = (kNetSize - t.new_h) / 2;
  return t;
}

void ResizeBilinear(const float* src, std::uint32_t src_w, std::uint32_t src_h, std::size_t src_stride,
                    std::uint32_t channels, float* out, std::uint32_t out_w, std::uint32_t out_h,
                    ResizeScratch& scratch) {
  AxisSamples(src_w, out_w, scratch.x0, scratch.x1, scratch.wx);
  AxisSamples(src_h, out_h, scratch.y0, scratch.y1, scratch.wy);
  const std::size_t row_floats = std::size_t{out_w} * channels;
  scratch.rows.resize(std::size_t{src_h} * row_floats);

  // Horizontal pass over every source row.
  for (std::uint32_t y = 0; y < src_h; ++y) {
    const float* in_row = src + y * src_stride;
    float* row = scratch.rows.data() + y * row_floats;
    for (std::uint32_t j = 0; j < out_w; ++j) {
      const float w1 = scratch.wx[j];
      const float w0 = 1.0f - w1;
      const float* a = in_row + std::size_t{scratch.x0[j]} * channels;
      const float* b = in_row + std::size_t{scratch.x1[j]} * channels;
      for (std::uint32_t c = 0; c < channels; ++c) {
        const float left = a[c] * w0;
        const float right = b[c] * w1;
        row[std::size_t{j} * channels + c] = left + right;
      }
    }
  }

  // Vertical pass.
  for (std::uint32_t i = 0; i < out_h; ++i) {
    const float w1 = scratch.wy[i];
    const float w0 = 1.0f - w1;
    const float* a = scratch.rows.data() + scratch.y0[i] * row_floats;
    const float* b = scratch.rows.data() + scratch.y1[i] * row_floats;
    float* out_row = out + i * row_floats;
    for (std::size_t k = 0; k < row_floats; ++k) {
      const float top = a[k] * w0;
      const float bottom = b[k] * w1;
      out_row[k] = top + bottom;
    }
  }
}

std::optional<ResizeTransform> Preprocessor::Run(const std::uint8_t* rgb, std::uint32_t width,
                                                 std::uint32_t height, ResizeMethod method, float* nchw_out) {
  const std::optional<ResizeTransform> transform = ComputeTransform(width, height, method);
  if (!transform) {
    return std::nullopt;
  }
  const ResizeTransform& t = *transform;

  const std::array<float, 256>& unit = UnitLut();
  const std::size_t src_floats = std::size_t{width} * height * 3;
  image_.resize(src_floats);
  for (std::size_t i = 0; i < src_floats; ++i) {
    image_[i] = unit[rgb[i]];
  }
  resized_.resize(std::size_t{t.new_w} * t.new_h * 3);
  ResizeBilinear(image_.data(), width, height, std::size_t{width} * 3, 3, resized_.data(), t.new_w, t.new_h,
                 scratch_);

  // Normalize into NCHW. The padding is the mean, so it becomes exactly 0.
  constexpr std::size_t plane = std::size_t{kNetSize} * kNetSize;
  for (std::uint32_t c = 0; c < 3; ++c) {
    const float pad_value = (kMean[c] - kMean[c]) / kStd[c];
    float* out_plane = nchw_out + c * plane;
    for (std::uint32_t y = 0; y < kNetSize; ++y) {
      float* out_row = out_plane + std::size_t{y} * kNetSize;
      const bool inside_y = y >= t.pad_y && y < t.pad_y + t.new_h;
      if (!inside_y) {
        std::fill(out_row, out_row + kNetSize, pad_value);
        continue;
      }
      std::fill(out_row, out_row + t.pad_x, pad_value);
      std::fill(out_row + t.pad_x + t.new_w, out_row + kNetSize, pad_value);
      const float* in_row = resized_.data() + std::size_t{y - t.pad_y} * t.new_w * 3;
      for (std::uint32_t x = 0; x < t.new_w; ++x) {
        const float centred = in_row[std::size_t{x} * 3 + c] - kMean[c];
        out_row[t.pad_x + x] = centred / kStd[c];
      }
    }
  }
  return transform;
}

void DepthToSource(const float* depth, const ResizeTransform& transform, std::vector<float>& out,
                   ResizeScratch& scratch) {
  out.resize(std::size_t{transform.src_w} * transform.src_h);
  const float* cropped = depth + std::size_t{transform.pad_y} * kNetSize + transform.pad_x;
  ResizeBilinear(cropped, transform.new_w, transform.new_h, kNetSize, 1, out.data(), transform.src_w,
                 transform.src_h, scratch);
}

}  // namespace relume
