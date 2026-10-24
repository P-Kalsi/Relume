#include "relume/camera.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <tuple>

namespace relume {

std::optional<FormatChoice> SelectFormat(std::span<const CameraFormat> formats, const CaptureRequest& request) {
  if (request.width == 0 || request.height == 0 || request.fps <= 0.0) {
    return std::nullopt;
  }
  const double target_aspect = static_cast<double>(request.width) / request.height;
  const double target_area = static_cast<double>(request.width) * request.height;

  std::optional<FormatChoice> best;
  // Lower is better, compared lexicographically.
  std::tuple<double, int, double, double> best_key;
  for (std::size_t i = 0; i < formats.size(); ++i) {
    const CameraFormat& format = formats[i];
    if (format.width == 0 || format.height == 0) {
      continue;
    }
    // The achievable rate closest to the request across this format's ranges.
    std::optional<double> fps;
    for (const FrameRateRange& range : format.frame_rates) {
      if (range.max_fps <= 0.0 || range.min_fps > range.max_fps) {
        continue;
      }
      const double candidate = std::clamp(request.fps, range.min_fps, range.max_fps);
      if (!fps || std::abs(candidate - request.fps) < std::abs(*fps - request.fps)) {
        fps = candidate;
      }
    }
    if (!fps) {
      continue;
    }
    const double aspect = static_cast<double>(format.width) / format.height;
    const double area = static_cast<double>(format.width) * format.height;
    const bool exact = format.width == request.width && format.height == request.height;
    // Rounded so that nearly identical aspect ratios fall through to area.
    const double aspect_error = std::round(std::abs(std::log(aspect / target_aspect)) * 1000.0);
    const auto key = std::make_tuple(std::abs(*fps - request.fps), exact ? 0 : 1, aspect_error,
                                     std::abs(std::log(area / target_area)));
    if (!best || key < best_key) {
      best = FormatChoice{i, *fps};
      best_key = key;
    }
  }
  return best;
}

std::optional<std::size_t> ChooseOutputPixelFormat(std::span<const PixelFormatOption> options,
                                                   PixelFormat preferred, std::string_view native_code) {
  auto find = [&](auto predicate) -> std::optional<std::size_t> {
    for (std::size_t i = 0; i < options.size(); ++i) {
      if (predicate(options[i])) {
        return i;
      }
    }
    return std::nullopt;
  };
  auto nv12 = [&]() -> std::optional<std::size_t> {
    if (auto native = find([&](const PixelFormatOption& o) {
          return o.format == PixelFormat::kNv12 && !native_code.empty() && o.code == native_code;
        })) {
      return native;
    }
    if (auto video = find([](const PixelFormatOption& o) {
          return o.format == PixelFormat::kNv12 && o.range == YuvRange::kVideo;
        })) {
      return video;
    }
    return find([](const PixelFormatOption& o) { return o.format == PixelFormat::kNv12; });
  };
  auto bgra = [&] { return find([](const PixelFormatOption& o) { return o.format == PixelFormat::kBgra8; }); };

  if (preferred == PixelFormat::kBgra8) {
    if (auto choice = bgra()) {
      return choice;
    }
    return nv12();
  }
  if (auto choice = nv12()) {
    return choice;
  }
  return bgra();
}

YuvMatrix FallbackYuvMatrix(std::uint32_t width, std::uint32_t height) {
  (void)width;
  return height >= 720 ? YuvMatrix::kBt709 : YuvMatrix::kBt601;
}

bool IsLikelyVirtualCamera(std::string_view name) {
  constexpr std::string_view kNeedle = "obs virtual camera";
  if (name.size() < kNeedle.size()) {
    return false;
  }
  for (std::size_t start = 0; start + kNeedle.size() <= name.size(); ++start) {
    bool match = true;
    for (std::size_t i = 0; i < kNeedle.size() && match; ++i) {
      match = std::tolower(static_cast<unsigned char>(name[start + i])) == kNeedle[i];
    }
    if (match) {
      return true;
    }
  }
  return false;
}

std::optional<std::size_t> ChooseDefaultDevice(std::span<const CameraDeviceInfo> devices,
                                               std::string_view preferred_id) {
  if (!preferred_id.empty()) {
    for (std::size_t i = 0; i < devices.size(); ++i) {
      if (devices[i].id == preferred_id && !devices[i].is_virtual) {
        return i;
      }
    }
  }
  for (std::size_t i = 0; i < devices.size(); ++i) {
    if (!devices[i].is_virtual) {
      return i;
    }
  }
  return std::nullopt;
}

}  // namespace relume
