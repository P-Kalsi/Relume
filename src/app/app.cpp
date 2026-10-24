#include "app/app.hpp"

#include "app/window.hpp"
#include "output/clean_window.hpp"
#include "relume/app_state.hpp"
#include "relume/event_queue.hpp"
#include "relume/frame_queue.hpp"
#include "relume/image_io.hpp"
#include "relume/layout.hpp"
#include "relume/pixel_convert.hpp"
#include "relume/renderer.hpp"
#include "relume/stats.hpp"
#include "ui/control_window.hpp"

#include <GLFW/glfw3.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string_view>
#include <thread>

namespace relume {

namespace {

constexpr char kUsage[] =
    "usage: Relume [options]\n"
    "  --source camera|file      video source (default: camera)\n"
    "  --file <path.ppm>         still image for the file source (default: orientation pattern)\n"
    "  --pattern <W>x<H>         size of the generated pattern (default: 1280x720)\n"
    "  --pixel-format nv12|bgra|rgb\n"
    "                            camera: requested output format (default nv12, rgb not allowed)\n"
    "                            file: encoding of the frames (default bgra)\n"
    "  --camera <unique-id>      open this camera instead of choosing automatically\n"
    "  --headless                no windows: render offscreen and check the output (file source only)\n"
    "  --frames <N>              stop after N composed frames\n"
    "  --stats <seconds>         stats log interval, 0 to disable (default 5)\n";

using Clock = std::chrono::steady_clock;

double Seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }

// Everything between a frame slot and the windows. One instance per app.
struct VideoPipeline {
  explicit VideoPipeline(const GpuContext& gpu)
      : output(gpu), uploader(gpu), compositor(gpu, output), presenter(gpu, output) {
    compositor.ClearToBlack();  // the output is never uninitialized
  }

  // Uploads and composes the newest frame, if there is a new one.
  bool Update(LatestFrameSlot& slot) {
    std::shared_ptr<const VideoFrame> frame = slot.TakeIfNew(last_frame_id);
    if (!frame) {
      return false;
    }
    last_frame_id = frame->frame_id;
    if (!uploader.Upload(*frame)) {
      ++rejected;
      return false;
    }
    compositor.Compose(uploader.camera_texture());
    showing_video = true;
    ++composed;
    return true;
  }

  // Black output, e.g. while the camera is unavailable (no frozen frame).
  void ShowBlack() {
    if (showing_video) {
      compositor.ClearToBlack();
      showing_video = false;
    }
  }

  // A new source numbers its frames from 1 again.
  void ResetFrameIds() { last_frame_id = 0; }

  OutputTexture output;
  FrameUploader uploader;
  OutputCompositor compositor;
  Presenter presenter;
  std::uint64_t last_frame_id = 0;
  std::uint64_t composed = 0;
  std::uint64_t rejected = 0;
  bool showing_video = false;
};

bool ParseSize(std::string_view text, std::uint32_t& width, std::uint32_t& height) {
  const std::size_t x = text.find('x');
  if (x == std::string_view::npos) {
    return false;
  }
  const std::string w(text.substr(0, x));
  const std::string h(text.substr(x + 1));
  char* end = nullptr;
  const unsigned long wv = std::strtoul(w.c_str(), &end, 10);
  if (end == w.c_str() || *end != '\0') {
    return false;
  }
  const unsigned long hv = std::strtoul(h.c_str(), &end, 10);
  if (end == h.c_str() || *end != '\0' || wv == 0 || hv == 0 || wv > 8192 || hv > 8192) {
    return false;
  }
  width = static_cast<std::uint32_t>(wv);
  height = static_cast<std::uint32_t>(hv);
  return true;
}

// Logs capture/render rates and drop counters every `interval` seconds.
class StatsLogger {
 public:
  explicit StatsLogger(double interval_s) : interval_s_(interval_s), last_(Clock::now()) {}

  void OnRender(Clock::time_point now) {
    if (last_render_ != Clock::time_point{}) {
      render_interval_ms_.Record(std::chrono::duration<double, std::milli>(now - last_render_).count());
    }
    last_render_ = now;
    ++renders_;
  }

  void Maybe(Clock::time_point now, const ICameraSource* source, const LatestFrameSlot& slot,
             const VideoPipeline& pipeline) {
    const double elapsed = Seconds(now - last_);
    if (interval_s_ <= 0.0 || elapsed < interval_s_) {
      return;
    }
    const CaptureStats capture = source != nullptr ? source->stats() : CaptureStats{};
    const SlotStats slot_stats = slot.stats();
    std::printf(
        "[stats] capture %.1f fps (copy p50 %.2f p95 %.2f ms) | composed %.1f fps | render loop %.1f fps "
        "(interval p50 %.2f p95 %.2f ms) | dropped: pool %llu, unread %llu, driver %llu, invalid %llu, "
        "rejected %llu\n",
        static_cast<double>(capture.frames_received - last_received_) / elapsed, capture.copy_ms_p50,
        capture.copy_ms_p95, static_cast<double>(pipeline.composed - last_composed_) / elapsed,
        static_cast<double>(renders_ - last_renders_) / elapsed, render_interval_ms_.Percentile(0.5),
        render_interval_ms_.Percentile(0.95), static_cast<unsigned long long>(capture.dropped_pool),
        static_cast<unsigned long long>(slot_stats.dropped_unread),
        static_cast<unsigned long long>(capture.dropped_driver),
        static_cast<unsigned long long>(capture.dropped_invalid),
        static_cast<unsigned long long>(pipeline.rejected));
    std::fflush(stdout);
    last_ = now;
    last_received_ = capture.frames_received;
    last_composed_ = pipeline.composed;
    last_renders_ = renders_;
    render_interval_ms_.Reset();
  }

 private:
  double interval_s_;
  Clock::time_point last_;
  Clock::time_point last_render_{};
  LatencyHistogram render_interval_ms_;
  std::uint64_t renders_ = 0;
  std::uint64_t last_renders_ = 0;
  std::uint64_t last_received_ = 0;
  std::uint64_t last_composed_ = 0;
};

// --- Headless self-check ---

std::array<int, 3> RgbAt(const std::vector<std::uint8_t>& rgba, std::uint32_t width, std::uint32_t x,
                         std::uint32_t y) {
  const std::uint8_t* p = &rgba[(std::size_t{y} * width + x) * 4];
  return {p[0], p[1], p[2]};
}

bool Near(std::array<int, 3> actual, const std::uint8_t (&expected)[3], int tolerance) {
  for (std::size_t c = 0; c < 3; ++c) {
    if (std::abs(actual[c] - expected[c]) > tolerance) {
      return false;
    }
  }
  return true;
}

// Checks the composed output against what the orientation pattern must look
// like after aspect-fit. Returns the number of failed checks.
int CheckPatternOutput(const std::vector<std::uint8_t>& out, std::uint32_t src_w, std::uint32_t src_h,
                       int tolerance) {
  const std::uint32_t ow = OutputTexture::kWidth;
  const PixelRect fit = FitRect(src_w, src_h, ow, OutputTexture::kHeight);
  // Map a point given as a fraction of the source image into the output.
  auto at = [&](double fx, double fy) {
    return RgbAt(out, ow, fit.x + static_cast<std::uint32_t>(fx * fit.width),
                 fit.y + static_cast<std::uint32_t>(fy * fit.height));
  };
  struct Check {
    const char* name;
    double fx, fy;
    const std::uint8_t (&color)[3];
  };
  const Check checks[] = {{"marker", 0.125, 0.125, pattern::kMarker},
                          {"top-left", 0.03, 0.4, pattern::kTopLeft},
                          {"top-right", 0.97, 0.03, pattern::kTopRight},
                          {"bottom-left", 0.03, 0.97, pattern::kBottomLeft},
                          {"bottom-right", 0.97, 0.97, pattern::kBottomRight}};
  int failures = 0;
  for (const Check& check : checks) {
    const auto rgb = at(check.fx, check.fy);
    const bool ok = Near(rgb, check.color, tolerance);
    std::printf("  %-13s %3d,%3d,%3d  %s\n", check.name, rgb[0], rgb[1], rgb[2], ok ? "ok" : "MISMATCH");
    failures += ok ? 0 : 1;
  }
  const std::uint8_t black[3] = {0, 0, 0};
  if (fit.x > 0) {
    const bool ok = Near(RgbAt(out, ow, fit.x / 2, 360), black, 0) &&
                    Near(RgbAt(out, ow, fit.x + fit.width + fit.x / 2, 360), black, 0);
    std::printf("  pillarbox bars (x < %u, x >= %u): %s\n", fit.x, fit.x + fit.width, ok ? "black" : "NOT BLACK");
    failures += ok ? 0 : 1;
  }
  if (fit.y > 0) {
    const bool ok = Near(RgbAt(out, ow, 640, fit.y / 2), black, 0) &&
                    Near(RgbAt(out, ow, 640, fit.y + fit.height + fit.y / 2), black, 0);
    std::printf("  letterbox bars (y < %u, y >= %u): %s\n", fit.y, fit.y + fit.height, ok ? "black" : "NOT BLACK");
    failures += ok ? 0 : 1;
  }
  return failures;
}

int RunHeadless(const AppOptions& options) {
  std::string error;
  std::unique_ptr<GpuContext> gpu = GpuContext::Create(error);
  if (!gpu) {
    std::fprintf(stderr, "FAIL: %s\n", error.c_str());
    return 1;
  }
  std::printf("GPU: %s\n", gpu->adapter_name().c_str());
  VideoPipeline pipeline(*gpu);
  CameraEventQueue events;
  LatestFrameSlot slot;
  FileSourceOptions file = options.file;
  file.realtime = false;
  FileSource source(file, slot, events);
  source.Start();
  std::printf("%s\n", source.description().c_str());

  const std::uint64_t target = options.frames == 0 ? 30 : options.frames;
  const Clock::time_point deadline = Clock::now() + std::chrono::seconds(30);
  while (pipeline.composed < target && source.state() != SourceState::kFailed && Clock::now() < deadline) {
    if (!pipeline.Update(slot)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    gpu->instance().ProcessEvents();
  }
  source.Stop();
  while (auto event = events.Poll()) {
    if (event->kind == CameraEvent::Kind::kSourceFailed) {
      std::fprintf(stderr, "FAIL: %s\n", event->message.c_str());
      return 1;
    }
  }
  std::printf("Composed %llu frames (target %llu), rejected %llu\n",
              static_cast<unsigned long long>(pipeline.composed), static_cast<unsigned long long>(target),
              static_cast<unsigned long long>(pipeline.rejected));
  if (pipeline.composed < target) {
    std::fprintf(stderr, "FAIL: not enough frames composed\n");
    return 1;
  }

  std::vector<std::uint8_t> out;
  if (!ReadbackTexture(*gpu, pipeline.output.texture(), out)) {
    std::fprintf(stderr, "FAIL: could not read back the output texture\n");
    return 1;
  }
  std::printf("Output texture: %ux%u RGBA8\n", pipeline.output.texture().GetWidth(),
              pipeline.output.texture().GetHeight());
  int failures = 0;

  const std::shared_ptr<const VideoFrame> last = slot.Latest();
  if (options.file.path.empty() && last) {
    const int tolerance = last->format == PixelFormat::kNv12 ? 2 : 0;  // NV12 round-trip bound
    std::printf("Orientation and aspect checks (%ux%u %s source):\n", last->width, last->height,
                ToString(last->format).data());
    failures += CheckPatternOutput(out, last->width, last->height, tolerance);
    // At 1:1 the output must be exactly the frame: nothing else is drawn into it.
    if (last->width == OutputTexture::kWidth && last->height == OutputTexture::kHeight &&
        last->format != PixelFormat::kNv12) {
      std::vector<std::uint8_t> expected;
      ToRgba8(*last, expected);
      const bool exact = expected == out;
      std::printf("  output == source frame, every pixel: %s\n", exact ? "yes" : "NO");
      failures += exact ? 0 : 1;
    }
  }

  // What a window shows must be exactly the output texture (BGRA surface format).
  wgpu::TextureDescriptor target_desc{};
  target_desc.size = {OutputTexture::kWidth, OutputTexture::kHeight, 1};
  target_desc.format = wgpu::TextureFormat::BGRA8Unorm;
  target_desc.usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc;
  const wgpu::Texture window_like = gpu->device().CreateTexture(&target_desc);
  pipeline.presenter.Present(window_like.CreateView(), target_desc.format, OutputTexture::kWidth,
                             OutputTexture::kHeight);
  std::vector<std::uint8_t> presented;
  if (!ReadbackTexture(*gpu, window_like, presented)) {
    std::fprintf(stderr, "FAIL: could not read back the presented texture\n");
    return 1;
  }
  for (std::size_t i = 0; i < presented.size(); i += 4) {
    std::swap(presented[i], presented[i + 2]);  // BGRA -> RGBA
  }
  const bool same = presented == out;
  std::printf("Presented 1280x720 window image == output texture: %s\n", same ? "yes" : "NO");
  failures += same ? 0 : 1;

  if (gpu->errors().count() != 0) {
    std::fprintf(stderr, "FAIL: %u WebGPU errors, last: %s\n", gpu->errors().count(), gpu->errors().last().c_str());
    return 1;
  }
  if (failures != 0) {
    std::fprintf(stderr, "FAIL: %d checks failed\n", failures);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}

// --- Windowed app ---

int RunWindowed(const AppOptions& options) {
  if (options.source == AppOptions::Source::kCamera) {
    std::fprintf(stderr, "The camera source is not available in this build yet; use --source file\n");
    return 1;
  }
  GlfwLibrary glfw;
  if (!glfw.ok()) {
    std::fprintf(stderr, "FAIL: could not initialize GLFW\n");
    return 1;
  }
  std::string error;
  std::unique_ptr<GpuContext> gpu = GpuContext::Create(error);
  if (!gpu) {
    std::fprintf(stderr, "FAIL: %s\n", error.c_str());
    return 1;
  }
  std::printf("GPU: %s\n", gpu->adapter_name().c_str());
  VideoPipeline pipeline(*gpu);
  ControlWindow controls(*gpu, pipeline.output);
  CleanWindow output(*gpu);
  if (!controls.ok() || !output.ok()) {
    std::fprintf(stderr, "FAIL: %s\n", !controls.ok() ? controls.error().c_str() : output.error().c_str());
    return 1;
  }
  output.Show();

  CameraEventQueue events;
  LatestFrameSlot slot;
  std::unique_ptr<ICameraSource> source = std::make_unique<FileSource>(options.file, slot, events);
  source->Start();
  std::printf("%s\n", source->description().c_str());

  StatsLogger stats(options.stats_interval_s);
  CaptureStatus status;
  int exit_code = 0;
  // Vsync normally paces the loop (60 Hz here). macOS stops pacing occluded
  // windows, so this floor caps the loop at about 125 Hz instead of letting
  // it spin; it is below one 60 Hz frame so it never costs a vsync.
  constexpr auto kMinIteration = std::chrono::milliseconds(8);
  Clock::time_point iteration_start = Clock::now();
  while (!controls.should_close()) {
    std::this_thread::sleep_until(iteration_start + kMinIteration);
    iteration_start = Clock::now();
    glfwPollEvents();
    output.HandleCloseRequest();
    while (auto event = events.Poll()) {
      std::printf("[event] %s\n", event->message.c_str());
    }

    CaptureFacts facts;
    facts.permission = PermissionState::kAuthorized;  // a file needs no permission
    facts.any_device = facts.has_selection = facts.selected_present = true;
    facts.source = source->state();
    facts.frame_received = pipeline.showing_video;
    const CaptureStatus next = DeriveStatus(facts);
    if (next.state != status.state) {
      std::printf("[status] %s: %s\n", StatusLabel(next.state).data(), next.detail.c_str());
    }
    status = next;

    pipeline.Update(slot);
    if (status.state != CaptureState::kLive && status.state != CaptureState::kStarting) {
      pipeline.ShowBlack();
    }
    output.Render(pipeline.presenter);
    ControlsModel model;
    model.status = status;
    model.source_description = source->description();
    model.camera_selectable = false;
    model.output_visible = output.visible();
    const ControlsActions actions = controls.Render(model);
    if (actions.toggle_output) {
      output.visible() ? output.Hide() : output.Show();
    }
    gpu->instance().ProcessEvents();

    const Clock::time_point now = Clock::now();
    stats.OnRender(now);
    stats.Maybe(now, source.get(), slot, pipeline);
    if (gpu->errors().device_lost()) {
      std::fprintf(stderr, "FAIL: the GPU device was lost\n");
      exit_code = 1;
      break;
    }
    if (options.frames != 0 && pipeline.composed >= options.frames) {
      break;
    }
  }
  source->Stop();
  gpu->WaitIdle();
  return exit_code;
}

}  // namespace

std::optional<AppOptions> ParseOptions(int argc, char** argv, std::string& error) {
  AppOptions options;
  bool pixel_format_set = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    auto value = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
    if (arg == "--source") {
      const char* v = value();
      if (v != nullptr && std::strcmp(v, "camera") == 0) {
        options.source = AppOptions::Source::kCamera;
      } else if (v != nullptr && std::strcmp(v, "file") == 0) {
        options.source = AppOptions::Source::kFile;
      } else {
        error = "--source must be camera or file";
        return std::nullopt;
      }
    } else if (arg == "--file") {
      const char* v = value();
      if (v == nullptr) {
        error = "--file needs a path";
        return std::nullopt;
      }
      options.file.path = v;
    } else if (arg == "--pattern") {
      const char* v = value();
      if (v == nullptr || !ParseSize(v, options.file.pattern_width, options.file.pattern_height)) {
        error = "--pattern needs <W>x<H>";
        return std::nullopt;
      }
    } else if (arg == "--pixel-format") {
      const char* v = value();
      const std::string_view name = v != nullptr ? v : "";
      if (name == "nv12") {
        options.pixel_format = PixelFormat::kNv12;
      } else if (name == "bgra") {
        options.pixel_format = PixelFormat::kBgra8;
      } else if (name == "rgb") {
        options.pixel_format = PixelFormat::kRgb8;
      } else {
        error = "--pixel-format must be nv12, bgra or rgb";
        return std::nullopt;
      }
      pixel_format_set = true;
    } else if (arg == "--camera") {
      const char* v = value();
      if (v == nullptr) {
        error = "--camera needs a unique ID";
        return std::nullopt;
      }
      options.camera_id = v;
    } else if (arg == "--headless") {
      options.headless = true;
    } else if (arg == "--frames") {
      const char* v = value();
      char* end = nullptr;
      const unsigned long long n = v != nullptr ? std::strtoull(v, &end, 10) : 0;
      if (v == nullptr || end == v || *end != '\0') {
        error = "--frames needs a number";
        return std::nullopt;
      }
      options.frames = n;
    } else if (arg == "--stats") {
      const char* v = value();
      char* end = nullptr;
      const double s = v != nullptr ? std::strtod(v, &end) : -1.0;
      if (v == nullptr || end == v || *end != '\0' || s < 0.0) {
        error = "--stats needs a number of seconds";
        return std::nullopt;
      }
      options.stats_interval_s = s;
    } else if (arg.rfind("-psn_", 0) == 0) {
      // Process serial number that older macOS passes to apps launched by Finder.
    } else if (arg == "--help" || arg == "-h") {
      error = kUsage;
      return std::nullopt;
    } else {
      error = std::string("unknown option ") + std::string(arg) + "\n" + kUsage;
      return std::nullopt;
    }
  }
  if (options.source == AppOptions::Source::kFile) {
    options.file.format = pixel_format_set ? options.pixel_format : PixelFormat::kBgra8;
  } else if (options.pixel_format == PixelFormat::kRgb8) {
    error = "--pixel-format rgb is only available for the file source";
    return std::nullopt;
  }
  if (options.headless && options.source != AppOptions::Source::kFile) {
    error = "--headless needs --source file (it never opens the camera)";
    return std::nullopt;
  }
  return options;
}

int RunApp(const AppOptions& options) { return options.headless ? RunHeadless(options) : RunWindowed(options); }

}  // namespace relume
