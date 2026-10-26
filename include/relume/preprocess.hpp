#pragma once

// The camera-image → network-tensor preprocessing contract, in C++. It must
// match scripts/relume_model/preprocess.py, the single definition of the
// contract (docs/model.md), and is checked against
// tests/python/golden/preprocess_golden.txt within 1e-6.
//
// Every step is done in the same order and precision as the Python code:
//   /255 in float32 → separable bilinear resize (half-pixel centres, edge
//   clamped; horizontal pass, then vertical; sample positions in double,
//   weights in float32) → placed on a canvas filled with the ImageNet mean →
//   (x - mean) / std in float32 → NCHW.
// The source file is compiled without floating-point contraction (no fused
// multiply-add), because numpy rounds every product and sum separately.
//
// Portable C++, no platform or ONNX Runtime dependency.

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace relume {

inline constexpr std::uint32_t kNetSize = 448;
inline constexpr std::size_t kNetTensorFloats = 3u * kNetSize * kNetSize;  // 1x3x448x448
inline constexpr std::size_t kNetDepthFloats = std::size_t{kNetSize} * kNetSize;  // 1x448x448

enum class ResizeMethod : std::uint8_t {
  kStretch,    // default (approved 2026-10-26): resize to 448x448, aspect ratio not kept
  kLetterbox,  // supported: keep the aspect ratio, pad with the ImageNet mean
};
inline constexpr ResizeMethod kDefaultResizeMethod = ResizeMethod::kStretch;

std::string_view ToString(ResizeMethod method);
std::optional<ResizeMethod> ParseResizeMethod(std::string_view name);

// Where a source image landed inside the 448x448 tensor.
struct ResizeTransform {
  ResizeMethod method = kDefaultResizeMethod;
  std::uint32_t src_w = 0;
  std::uint32_t src_h = 0;
  std::uint32_t new_w = 0;  // size of the resized image inside the tensor
  std::uint32_t new_h = 0;
  std::uint32_t pad_x = 0;  // offset of the resized image inside the tensor
  std::uint32_t pad_y = 0;

  double sx() const { return static_cast<double>(new_w) / src_w; }
  double sy() const { return static_cast<double>(new_h) / src_h; }
  // Continuous coordinates (pixel i covers [i, i+1)): source → tensor.
  double SrcToNetX(double x) const { return x * sx() + pad_x; }
  double SrcToNetY(double y) const { return y * sy() + pad_y; }
  // Tensor → source, the exact inverse.
  double NetToSrcX(double x) const { return (x - pad_x) / sx(); }
  double NetToSrcY(double y) const { return (y - pad_y) / sy(); }

  bool operator==(const ResizeTransform&) const = default;
};

// Placement of a src_w x src_h image. Letterbox: s = 448 / max(W, H),
// new = max(1, round(dim * s)) with round half to even, centred with
// pad = (448 - new) / 2. Stretch: new = 448, pad = 0.
// Returns nullopt for an empty size.
std::optional<ResizeTransform> ComputeTransform(std::uint32_t src_w, std::uint32_t src_h,
                                                ResizeMethod method = kDefaultResizeMethod);

// Working memory for ResizeBilinear, reused between calls so that repeated
// resizes of the same sizes do not allocate.
struct ResizeScratch {
  std::vector<float> rows;  // horizontal pass: src_h x out_w x channels
  std::vector<std::uint32_t> x0, x1, y0, y1;
  std::vector<float> wx, wy;
};

// Bilinear resize of a float image with `channels` interleaved channels.
// `src_stride` is the distance between source rows, in floats. Sizes must be
// at least 1.
void ResizeBilinear(const float* src, std::uint32_t src_w, std::uint32_t src_h, std::size_t src_stride,
                    std::uint32_t channels, float* out, std::uint32_t out_w, std::uint32_t out_h,
                    ResizeScratch& scratch);

// Reusable preprocessing state: keeps its scratch buffers between frames, so
// steady-state calls do not allocate. Not thread-safe; one per thread.
class Preprocessor {
 public:
  // Tightly packed RGB (3 bytes per pixel, top row first) → a 1x3x448x448
  // float32 NCHW tensor in `nchw_out` (kNetTensorFloats floats, caller-owned).
  // Returns the transform, or nullopt for an empty image.
  std::optional<ResizeTransform> Run(const std::uint8_t* rgb, std::uint32_t width, std::uint32_t height,
                                     ResizeMethod method, float* nchw_out);

 private:
  std::vector<float> image_;    // source as float32 in [0, 1], HxWx3
  std::vector<float> resized_;  // new_h x new_w x 3
  ResizeScratch scratch_;
};

// Network depth (448x448, row-major) → source-resolution depth (src_h rows of
// src_w floats) in `out`: crops the letterbox padding, then resamples with the
// same bilinear rule. Values are not rescaled. `scratch` is reused.
void DepthToSource(const float* depth, const ResizeTransform& transform, std::vector<float>& out,
                   ResizeScratch& scratch);

}  // namespace relume
