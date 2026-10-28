#pragma once

#include "app/window.hpp"
#include "relume/depth_worker.hpp"
#include "relume/file_source.hpp"
#include "relume/frame_saver.hpp"

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
  PresenterKind presenter = PresenterKind::kLayer;  // window presentation path
  bool trace = false;                     // log the phases of slow render-loop iterations
  bool occlusion_test = false;            // cover/uncover the windows on a schedule and check the loop
  FrameSaverOptions save;                 // --save-frames: off unless save.count > 0 (camera only)
  bool guided_capture = false;            // --guided-capture: shots taken from the Controls window
  bool guided_self_test = false;          // with --guided-capture --source file: script one still and one burst
  bool inference_switch_test = false;     // --inference-switch-test: scripted provider changes, loop measured
  std::string test_guide;                 // --test-guide <name>: timed on-screen steps for a manual test

  // Depth inference. Default on macOS: CoreML with CPUAndGPU compute units,
  // falling back to CPU if the CoreML session cannot be created (approved
  // 2026-10-27, provisional until the Step 8 camera checks; docs/decisions.md).
  // Elsewhere: CPU. Off for --headless, --guided-capture and --save-frames
  // unless --inference is given (headless then also checks that matching
  // depth arrives).
#if defined(__APPLE__)
  std::optional<ExecutionProvider> inference = ExecutionProvider::kCoreML;  // nullopt = off
#else
  std::optional<ExecutionProvider> inference = ExecutionProvider::kCpu;
#endif
  bool inference_set = false;             // --inference was given
  CoreMLUnits coreml_units = CoreMLUnits::kCPUAndGPU;
  bool coreml_strict = false;             // --coreml-strict: diagnostic; this model then fails to load on
                                          // CoreML, which exercises the automatic CPU fallback
  std::string model_path;                 // default: models/depth_anything_v2_small_448.onnx in the repo
  int threads = 4;                        // ONNX Runtime intra-op threads (measured choice; docs/decisions.md)
  ResizeMethod preprocess = kDefaultResizeMethod;
};

// The depth worker configuration for `options`, or nullopt when inference is off.
std::optional<DepthWorkerConfig> InferenceConfig(const AppOptions& options);

// Parses command-line arguments. Returns nullopt and sets `error` (which may
// be the usage text) on failure.
std::optional<AppOptions> ParseOptions(int argc, char** argv, std::string& error);

// Runs Relume. Returns the process exit code.
int RunApp(const AppOptions& options);

}  // namespace relume
