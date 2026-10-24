#include "relume/image_io.hpp"

#include <cctype>
#include <cstring>
#include <fstream>
#include <iterator>

namespace relume {

namespace {

constexpr std::uint32_t kMaxDimension = 16384;

// Reads the next whitespace-separated header token, skipping '#' comments.
bool NextToken(std::istream& in, std::string& token) {
  token.clear();
  int c = in.get();
  while (c != EOF) {
    if (c == '#') {
      while (c != EOF && c != '\n') {
        c = in.get();
      }
    } else if (std::isspace(c) != 0) {
      c = in.get();
    } else {
      break;
    }
  }
  while (c != EOF && std::isspace(c) == 0 && c != '#') {
    token.push_back(static_cast<char>(c));
    c = in.get();
  }
  // `c` is the single whitespace byte that ends the token; for maxval it
  // separates the header from the pixel data.
  return !token.empty();
}

bool ParseUint(const std::string& token, std::uint32_t& value) {
  if (token.empty() || token.size() > 9) {
    return false;
  }
  value = 0;
  for (char ch : token) {
    if (ch < '0' || ch > '9') {
      return false;
    }
    value = value * 10 + static_cast<std::uint32_t>(ch - '0');
  }
  return true;
}

}  // namespace

std::optional<RgbImage> ReadPpm(const std::string& path, std::string& error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    error = "cannot open " + path;
    return std::nullopt;
  }
  std::string token;
  if (!NextToken(in, token) || token != "P6") {
    error = path + " is not a binary PPM (P6)";
    return std::nullopt;
  }
  RgbImage image;
  std::uint32_t maxval = 0;
  if (!NextToken(in, token) || !ParseUint(token, image.width) || !NextToken(in, token) ||
      !ParseUint(token, image.height) || !NextToken(in, token) || !ParseUint(token, maxval)) {
    error = path + " has a malformed PPM header";
    return std::nullopt;
  }
  if (maxval != 255) {
    error = path + ": only maxval 255 is supported";
    return std::nullopt;
  }
  if (image.width == 0 || image.height == 0 || image.width > kMaxDimension || image.height > kMaxDimension) {
    error = path + ": unsupported image size";
    return std::nullopt;
  }
  image.pixels.resize(std::size_t{image.width} * image.height * 3);
  in.read(reinterpret_cast<char*>(image.pixels.data()), static_cast<std::streamsize>(image.pixels.size()));
  if (in.gcount() != static_cast<std::streamsize>(image.pixels.size())) {
    error = path + ": pixel data is truncated";
    return std::nullopt;
  }
  return image;
}

bool WritePpm(const std::string& path, const RgbImage& image) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    return false;
  }
  out << "P6\n" << image.width << " " << image.height << "\n255\n";
  out.write(reinterpret_cast<const char*>(image.pixels.data()), static_cast<std::streamsize>(image.pixels.size()));
  return static_cast<bool>(out);
}

RgbImage MakeOrientationPattern(std::uint32_t width, std::uint32_t height) {
  RgbImage image;
  image.width = width;
  image.height = height;
  image.pixels.resize(std::size_t{width} * height * 3);
  const std::uint32_t mx0 = width / 16;
  const std::uint32_t my0 = height / 16;
  const std::uint32_t mx1 = mx0 + width / 8;
  const std::uint32_t my1 = my0 + height / 8;
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      const bool left = x < width / 2;
      const bool top = y < height / 2;
      const std::uint8_t* color = top ? (left ? pattern::kTopLeft : pattern::kTopRight)
                                      : (left ? pattern::kBottomLeft : pattern::kBottomRight);
      if (x >= mx0 && x < mx1 && y >= my0 && y < my1) {
        color = pattern::kMarker;
      }
      std::memcpy(&image.pixels[(std::size_t{y} * width + x) * 3], color, 3);
    }
  }
  return image;
}

}  // namespace relume
