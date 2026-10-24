#pragma once

#include "relume/file_source.hpp"

#include <cstdint>
#include <optional>
#include <string>

namespace relume {

struct AppOptions {
  enum class Source : std::uint8_t { kCamera, kFile };
  Source source = Source::kCamera;
  FileSourceOptions file;                 // for kFile
  std::string camera_id;                  // for kCamera; empty = choose automatically
  PixelFormat pixel_format = PixelFormat::kNv12;
  bool headless = false;                  // no windows; render offscreen and self-check
  std::uint64_t frames = 0;               // stop after this many composed frames (0 = run until closed)
  double seconds = 0.0;                   // stop after this long (0 = run until closed)
  double stats_interval_s = 5.0;          // stats log period (0 = off)
};

// Parses command-line arguments. Returns nullopt and sets `error` (which may
// be the usage text) on failure.
std::optional<AppOptions> ParseOptions(int argc, char** argv, std::string& error);

// Runs Relume. Returns the process exit code.
int RunApp(const AppOptions& options);

}  // namespace relume
