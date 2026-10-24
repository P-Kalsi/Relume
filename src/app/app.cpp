#include "app/app.hpp"

#include "app/window.hpp"
#include "output/clean_window.hpp"
#include "platform/window_state.hpp"
#include "relume/app_state.hpp"
#include "relume/camera.hpp"
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
    "  --seconds <S>             stop after S seconds (windowed)\n"
    "  --stats <seconds>         stats log interval, 0 to disable (default 5)\n"
    "  --presenter layer|surface window presentation: IOSurface layer contents (default; never\n"
    "                            waits for a drawable) or the CAMetalLayer surface (fallback)\n"
    "  --trace                   log the phases of render-loop iterations slower than 50 ms\n"
    "  --occlusion-test          cover/uncover each window on a schedule, then check that the\n"
    "                            loop never stalled and the output kept updating (exit 0/1)\n";

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
      const double ms = std::chrono::duration<double, std::milli>(now - last_render_).count();
      render_interval_ms_.Record(ms);
      max_interval_ms_ = std::max(max_interval_ms_, ms);
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
    if (source != last_source_ || capture.frames_received < last_received_) {
      last_received_ = 0;  // a new source counts from zero
      last_source_ = source;
    }
    const SlotStats slot_stats = slot.stats();
    std::printf(
        "[stats] capture %.1f fps (timestamp interval p50 %.2f p95 %.2f ms; copy p50 %.2f p95 %.2f ms) | "
        "composed %.1f fps | render loop %.1f fps "
        "(interval p50 %.2f p95 %.2f max %.1f ms) | dropped: pool %llu, unread %llu, driver %llu, invalid %llu, "
        "rejected %llu\n",
        static_cast<double>(capture.frames_received - last_received_) / elapsed, capture.interval_ms_p50,
        capture.interval_ms_p95, capture.copy_ms_p50,
        capture.copy_ms_p95, static_cast<double>(pipeline.composed - last_composed_) / elapsed,
        static_cast<double>(renders_ - last_renders_) / elapsed, render_interval_ms_.Percentile(0.5),
        render_interval_ms_.Percentile(0.95), max_interval_ms_, static_cast<unsigned long long>(capture.dropped_pool),
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
    max_interval_ms_ = 0.0;
  }

 private:
  double interval_s_;
  Clock::time_point last_;
  Clock::time_point last_render_{};
  LatencyHistogram render_interval_ms_;
  double max_interval_ms_ = 0.0;  // catches single stalls that percentiles hide
  const ICameraSource* last_source_ = nullptr;
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

const char* PermissionName(PermissionState permission) {
  switch (permission) {
    case PermissionState::kNotDetermined: return "not determined";
    case PermissionState::kAuthorized: return "authorized";
    case PermissionState::kDenied: return "denied";
    case PermissionState::kRestricted: return "restricted";
  }
  return "unknown";
}



// Opens the windows next to each other so the Controls window does not cover
// the Output window (the most common way it used to get covered).
void ArrangeSideBySide(GLFWwindow* controls, GLFWwindow* output) {
  GLFWmonitor* monitor = glfwGetPrimaryMonitor();
  if (monitor == nullptr) {
    return;
  }
  int area_x = 0, area_y = 0, area_w = 0, area_h = 0;
  glfwGetMonitorWorkarea(monitor, &area_x, &area_y, &area_w, &area_h);
  int output_w = 0, output_h = 0, left = 0, top = 0, right = 0, bottom = 0;
  glfwGetWindowSize(output, &output_w, &output_h);
  glfwGetWindowFrameSize(output, &left, &top, &right, &bottom);
  constexpr int kMargin = 16;
  const int controls_w = std::clamp(area_w - output_w - left - right - 3 * kMargin, 520, 960);
  const int controls_h = std::clamp(area_h - top - 2 * kMargin, 400, 640);
  glfwSetWindowSize(controls, controls_w, controls_h);
  glfwSetWindowPos(controls, area_x + kMargin + left, area_y + kMargin + top);
  glfwSetWindowPos(output, area_x + area_w - output_w - right - kMargin, area_y + kMargin + top);
}

// Times the phases of one render-loop iteration (main thread).
class IterationTrace {
 public:
  void Begin() {
    count_ = 0;
    start_ = Clock::now();
  }
  // Time from the end of phase `from` to the end of phase `to` (by name).
  double Between(const char* from, const char* to) const {
    std::optional<Clock::time_point> a, b;
    for (std::size_t i = 0; i < count_; ++i) {
      if (std::string_view(phases_[i].first) == from) a = phases_[i].second;
      if (std::string_view(phases_[i].first) == to) b = phases_[i].second;
    }
    return a && b ? std::chrono::duration<double, std::milli>(*b - *a).count() : 0.0;
  }
  void Mark(const char* phase) {
    if (count_ < phases_.size()) {
      phases_[count_++] = {phase, Clock::now()};
    }
  }
  double TotalMs() const {
    return count_ == 0 ? 0.0 : std::chrono::duration<double, std::milli>(phases_[count_ - 1].second - start_).count();
  }
  void Print(double since_start_s, const char* extra) const {
    std::printf("[trace] t=%.2fs iteration %.1f ms:", since_start_s, TotalMs());
    Clock::time_point previous = start_;
    for (std::size_t i = 0; i < count_; ++i) {
      std::printf(" %s %.1f", phases_[i].first,
                  std::chrono::duration<double, std::milli>(phases_[i].second - previous).count());
      previous = phases_[i].second;
    }
    std::printf(" | %s\n", extra);
    std::fflush(stdout);
  }

 private:
  Clock::time_point start_;
  std::array<std::pair<const char*, Clock::time_point>, 16> phases_{};
  std::size_t count_ = 0;
};

// Logs when a window's on-screen visibility (macOS occlusion state) changes.
class OcclusionLog {
 public:
  OcclusionLog(const char* name, GLFWwindow* window) : name_(name), window_(window) {}
  bool Update(double since_start_s) {
    const bool on_screen = IsWindowOnScreen(window_);
    if (!known_ || on_screen != on_screen_) {
      std::printf("[occlusion] t=%.2fs %s is %s (app %s)\n", since_start_s, name_,
                  on_screen ? "on screen" : "NOT on screen", IsApplicationActive() ? "active" : "inactive");
      std::fflush(stdout);
      known_ = true;
      on_screen_ = on_screen;
    }
    return on_screen_;
  }

 private:
  const char* name_;
  GLFWwindow* window_;
  bool known_ = false;
  bool on_screen_ = false;
};

// --occlusion-test: an undecorated window of our own covers the Output and
// Controls windows in turn (and the Output window is minimized once), on a
// fixed schedule, while the loop's behaviour is measured. The window server
// computes occlusion the same way whichever app owns the covering window.
class OcclusionTester {
 public:
  struct Phase {
    double start_s;
    const char* name;
    enum Action : std::uint8_t { kNone, kCoverOutput, kCoverControls, kResizeOutput, kMinimizeOutput } action;
  };

  OcclusionTester(GLFWwindow* output, GLFWwindow* controls)
      : output_(output), controls_(controls), occluder_(400, 300, "Relume occlusion test", false, false) {
    MakeWindowOpaque(occluder_.get());  // an undrawn window is transparent and covers nothing
    // Side by side and not overlapping, so only the occluder covers them.
    glfwSetWindowSize(controls_, 600, 450);
    glfwSetWindowPos(controls_, 40, 80);
    glfwSetWindowPos(output_, 700, 80);
  }

  // Applies the phase for `t` and returns its index.
  std::size_t Update(double t) {
    std::size_t index = 0;
    while (index + 1 < kPhases.size() && t >= kPhases[index + 1].start_s) {
      ++index;
    }
    if (index != current_) {
      current_ = index;
      Apply(kPhases[index]);
    }
    if (kPhases[index].action == Phase::kResizeOutput) {
      // A new Output window size every 0.6 s, ending at the original size.
      static constexpr int kSizes[][2] = {{900, 506}, {400, 225}, {1000, 400}, {500, 700}, {760, 428}, {640, 360}};
      const auto step = static_cast<std::size_t>((t - kPhases[index].start_s) / 0.6);
      if (step != resize_step_ && step < std::size(kSizes)) {
        resize_step_ = step;
        glfwSetWindowSize(output_, kSizes[step][0], kSizes[step][1]);
      }
    }
    return index;
  }
  bool done(double t) const { return t >= kPhases.back().start_s; }
  GLFWwindow* occluder() const { return occluder_.get(); }
  static const Phase& phase(std::size_t i) { return kPhases[i]; }
  static std::size_t phase_count() { return kPhases.size(); }

 private:
  void Cover(GLFWwindow* target) {
    int x = 0, y = 0, w = 0, h = 0, left = 0, top = 0, right = 0, bottom = 0;
    glfwGetWindowPos(target, &x, &y);
    glfwGetWindowSize(target, &w, &h);
    glfwGetWindowFrameSize(target, &left, &top, &right, &bottom);
    // Cover the whole frame, title bar included, with a margin.
    glfwSetWindowPos(occluder_.get(), x - left - 20, y - top - 20);
    glfwSetWindowSize(occluder_.get(), w + left + right + 40, h + top + bottom + 40);
    glfwShowWindow(occluder_.get());
    glfwFocusWindow(occluder_.get());
  }
  void Apply(const Phase& phase) {
    std::printf("[occlusion-test] t=%.1fs phase: %s\n", phase.start_s, phase.name);
    std::fflush(stdout);
    glfwHideWindow(occluder_.get());
    if (glfwGetWindowAttrib(output_, GLFW_ICONIFIED) == GLFW_TRUE) {
      glfwRestoreWindow(output_);
    }
    switch (phase.action) {
      case Phase::kNone: break;
      case Phase::kCoverOutput: Cover(output_); break;
      case Phase::kCoverControls: Cover(controls_); break;
      case Phase::kResizeOutput: resize_step_ = static_cast<std::size_t>(-1); break;
      case Phase::kMinimizeOutput: glfwIconifyWindow(output_); break;
    }
  }

  static constexpr std::array<Phase, 14> kPhases = {{
      {0.0, "startup (not measured)", Phase::kNone},
      {3.0, "both windows visible", Phase::kNone},
      {7.0, "Output covered", Phase::kCoverOutput},
      {13.0, "uncovered", Phase::kNone},
      {17.0, "Controls covered", Phase::kCoverControls},
      {23.0, "uncovered", Phase::kNone},
      {27.0, "Output covered again", Phase::kCoverOutput},
      {33.0, "uncovered", Phase::kNone},
      {37.0, "Output resized", Phase::kResizeOutput},
      {41.0, "settled", Phase::kNone},
      {43.0, "Output minimized", Phase::kMinimizeOutput},
      {49.0, "restored", Phase::kNone},
      {53.0, "both visible (final)", Phase::kNone},
      {57.0, "end", Phase::kNone},
  }};

  GLFWwindow* output_;
  GLFWwindow* controls_;
  Window occluder_;
  std::size_t current_ = static_cast<std::size_t>(-1);
  std::size_t resize_step_ = static_cast<std::size_t>(-1);
};

// Per-phase measurements for the occlusion test.
struct PhaseStats {
  double seconds = 0.0;
  std::uint64_t composed = 0;
  std::uint64_t iterations = 0;
  double max_iteration_ms = 0.0;
  double max_present_ms = 0.0;  // compose + both windows + finishing frames
};

// Owns the video source and decides when it runs: permission, camera
// selection, disconnect and reconnect. Main thread only.
class CaptureController {
 public:
  CaptureController(const AppOptions& options, CameraEventQueue& events, LatestFrameSlot& slot)
      : options_(options), events_(events), slot_(slot) {
    if (options.source == AppOptions::Source::kFile) {
      CameraDeviceInfo file;
      file.id = FileSource::kDeviceId;
      file.name = options.file.path.empty() ? "Orientation test pattern" : options.file.path;
      devices_.push_back(file);
      selected_id_ = file.id;
      return;
    }
    system_ = CreatePlatformCameraSystem(events);
    permission_ = system_->permission();
    std::printf("Camera permission at launch: %s\n", PermissionName(permission_));
    if (permission_ == PermissionState::kNotDetermined) {
      std::printf("Requesting camera access (answer the macOS prompt)\n");
      system_->RequestPermission();
    }
    RefreshDevices();
    selected_id_ = options.camera_id;
    if (!selected_id_.empty()) {
      std::printf("Requested camera: %s\n", selected_id_.c_str());
    }
  }

  ~CaptureController() { StopSource(); }
  CaptureController(const CaptureController&) = delete;
  CaptureController& operator=(const CaptureController&) = delete;

  // Handles events and starts or stops the source. Returns true when the
  // source was replaced or stopped, so the caller resets its frame state.
  bool Update() {
    bool changed = false;
    while (std::optional<CameraEvent> event = events_.Poll()) {
      switch (event->kind) {
        case CameraEvent::Kind::kPermissionChanged:
          permission_ = system_ ? system_->permission() : permission_;
          std::printf("[camera] %s (now %s)\n", event->message.c_str(), PermissionName(permission_));
          RefreshDevices();
          break;
        case CameraEvent::Kind::kDevicesChanged:
          std::printf("[camera] %s\n", event->message.c_str());
          RefreshDevices();
          break;
        case CameraEvent::Kind::kSourceFailed:
          error_ = event->message;
          std::printf("[camera] error: %s\n", event->message.c_str());
          break;
        case CameraEvent::Kind::kSourceStarted:
          std::printf("[camera] started: %s\n", event->message.c_str());
          break;
        case CameraEvent::Kind::kSourceEnded:
          std::printf("[camera] %s\n", event->message.c_str());
          break;
        case CameraEvent::Kind::kWarning:
          warning_ = event->message;
          std::printf("[camera] warning: %s\n", event->message.c_str());
          break;
      }
    }
    if (system_ && selected_id_.empty() && permission_ == PermissionState::kAuthorized) {
      if (std::optional<std::size_t> index = ChooseDefaultDevice(devices_, "")) {
        selected_id_ = devices_[*index].id;
        std::printf("Selected camera: %s [%s]\n", devices_[*index].name.c_str(), selected_id_.c_str());
      }
    }
    // Disconnect: stop now and show black instead of a frozen frame.
    if (source_ && !SelectedPresent()) {
      std::printf("[camera] selected camera is gone; stopping capture\n");
      StopSource();
      changed = true;
    }
    if (ShouldStartCapture(Facts())) {
      changed = StartSource() || changed;
    }
    return changed;
  }

  // The user picked another camera (virtual cameras were confirmed in the UI).
  bool Select(const std::string& id) {
    if (id == selected_id_ || !system_) {
      return false;
    }
    StopSource();
    selected_id_ = id;
    std::printf("Selected camera: %s\n", id.c_str());
    return true;
  }

  CaptureFacts Facts() const {
    CaptureFacts facts;
    facts.permission = permission_;
    facts.any_device = !devices_.empty();
    facts.has_selection = !selected_id_.empty();
    facts.selected_present = SelectedPresent();
    facts.source = source_ ? source_->state() : SourceState::kIdle;
    facts.frame_received = facts.source == SourceState::kRunning;
    facts.error = error_;
    return facts;
  }

  void FillModel(ControlsModel& model) const {
    model.devices = devices_;
    model.selected_id = selected_id_;
    model.source_description = source_ ? source_->description() : std::string();
    model.warning = warning_;
    model.camera_selectable = system_ != nullptr;
  }

  const ICameraSource* source() const { return source_.get(); }

  void StopSource() {
    if (source_) {
      source_->Stop();
      source_.reset();
    }
    slot_.Clear();
  }

 private:
  void RefreshDevices() {
    if (!system_) {
      return;
    }
    devices_ = system_->EnumerateDevices();
    std::printf("Cameras (%zu):\n", devices_.size());
    for (const CameraDeviceInfo& device : devices_) {
      std::printf("  - %s [%s]%s, %zu formats\n", device.name.c_str(), device.id.c_str(),
                  device.is_virtual ? " (virtual: never selected automatically)" : "", device.formats.size());
    }
  }

  bool SelectedPresent() const {
    for (const CameraDeviceInfo& device : devices_) {
      if (device.id == selected_id_) {
        return true;
      }
    }
    return false;
  }

  bool StartSource() {
    error_.clear();
    warning_.clear();
    if (system_) {
      CaptureRequest request;
      request.pixel_format = options_.pixel_format;
      source_ = system_->CreateSource(selected_id_, request, slot_);
    } else {
      source_ = std::make_unique<FileSource>(options_.file, slot_, events_);
    }
    if (!source_) {
      error_ = "This platform has no camera support yet.";
      return false;
    }
    source_->Start();
    return true;
  }

  const AppOptions& options_;
  CameraEventQueue& events_;
  LatestFrameSlot& slot_;
  std::unique_ptr<ICameraSystem> system_;  // null for the file source
  PermissionState permission_ = PermissionState::kAuthorized;
  std::vector<CameraDeviceInfo> devices_;
  std::string selected_id_;
  std::string error_;
  std::string warning_;
  std::unique_ptr<ICameraSource> source_;  // declared last: stopped before the rest goes
};

int RunWindowed(const AppOptions& options) {
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
  ControlWindow controls(*gpu, pipeline.output, options.presenter);
  CleanWindow output(*gpu, options.presenter);
  if (!controls.ok() || !output.ok()) {
    std::fprintf(stderr, "FAIL: %s\n", !controls.ok() ? controls.error().c_str() : output.error().c_str());
    return 1;
  }
  std::printf("Presentation: Output '%s', Controls '%s'\n", output.presenter()->kind(), controls.presenter()->kind());
  ArrangeSideBySide(controls.window(), output.window());
  output.Show();

  CameraEventQueue events;
  LatestFrameSlot slot;
  CaptureController capture(options, events, slot);

  StatsLogger stats(options.stats_interval_s);
  CaptureStatus status;
  status.state = CaptureState::kStarting;
  int exit_code = 0;
  // Loop pacing. The layer presenter has no vsync, so the loop ticks at the
  // display's refresh rate. The CAMetalLayer surface is paced by vsync while
  // visible; macOS stops pacing covered windows, so an 8 ms floor (below one
  // 60 Hz frame, so it never costs a vsync) caps it at about 125 Hz.
  const bool layer_paced = output.presenter()->kind() == std::string_view("layer");
  const auto tick = layer_paced ? std::chrono::duration_cast<Clock::duration>(
                                      std::chrono::duration<double>(1.0 / DisplayRefreshHz(output.window())))
                                : std::chrono::duration_cast<Clock::duration>(std::chrono::milliseconds(8));
  std::printf("Loop pacing: %s, %.2f ms per iteration\n", layer_paced ? "display refresh" : "vsync with 8 ms floor",
              std::chrono::duration<double, std::milli>(tick).count());
  const Clock::time_point run_start = Clock::now();
  Clock::time_point iteration_start = run_start;
  Clock::time_point next_tick = run_start;
  IterationTrace trace;
  OcclusionLog output_occlusion(CleanWindow::kTitle, output.window());
  OcclusionLog controls_occlusion(ControlWindow::kTitle, controls.window());
  std::optional<OcclusionTester> tester;
  std::vector<PhaseStats> phase_stats(OcclusionTester::phase_count());
  std::optional<OcclusionLog> occluder_occlusion;
  if (options.occlusion_test) {
    tester.emplace(output.window(), controls.window());
    occluder_occlusion.emplace("occluder", tester->occluder());
  }
  while (!controls.should_close()) {
    next_tick += tick;
    if (next_tick < Clock::now()) {
      next_tick = Clock::now();  // fell behind (e.g. a modal AppKit loop): do not try to catch up
    }
    std::this_thread::sleep_until(next_tick);
    const Clock::time_point previous_start = iteration_start;
    iteration_start = Clock::now();
    const double t = Seconds(iteration_start - run_start);
    trace.Begin();
    std::size_t phase_index = 0;
    if (tester) {
      phase_index = tester->Update(t);
      if (tester->done(t)) {
        break;
      }
      PhaseStats& ps = phase_stats[phase_index];
      ps.iterations += 1;
      ps.max_iteration_ms =
          std::max(ps.max_iteration_ms, std::chrono::duration<double, std::milli>(iteration_start - previous_start).count());
    }
    const std::uint64_t composed_before = pipeline.composed;
    glfwPollEvents();
    trace.Mark("poll");
    const bool output_on_screen = output_occlusion.Update(t);
    const bool controls_on_screen = controls_occlusion.Update(t);
    if (occluder_occlusion) {
      occluder_occlusion->Update(t);
    }
    output.HandleCloseRequest();

    if (capture.Update()) {
      pipeline.ResetFrameIds();
      pipeline.ShowBlack();
    }
    const CaptureStatus next = DeriveStatus(capture.Facts());
    if (next.state != status.state || next.detail != status.detail) {
      std::printf("[status] %s: %s\n", StatusLabel(next.state).data(), next.detail.c_str());
      std::fflush(stdout);
    }
    status = next;

    trace.Mark("capture");
    if (status.state == CaptureState::kLive || status.state == CaptureState::kStarting) {
      pipeline.Update(slot);
    } else {
      pipeline.ShowBlack();  // never a frozen or uninitialized frame
    }
    trace.Mark("compose");
    output.Render(pipeline.presenter);
    trace.Mark("output");

    ControlsModel model;
    model.status = status;
    model.output_visible = output.visible();
    capture.FillModel(model);
    const ControlsActions actions = controls.Render(model);
    trace.Mark("controls");
    if (actions.toggle_output) {
      output.visible() ? output.Hide() : output.Show();
    }
    if (actions.select_device && capture.Select(*actions.select_device)) {
      pipeline.ResetFrameIds();
      pipeline.ShowBlack();
    }
    gpu->instance().ProcessEvents();
    output.FinishFrames();
    controls.FinishFrames();
    trace.Mark("events");
    if (tester) {
      PhaseStats& ps = phase_stats[phase_index];
      ps.composed += pipeline.composed - composed_before;
      ps.max_present_ms = std::max(ps.max_present_ms, trace.Between("capture", "events"));
    }
    if (options.trace && trace.TotalMs() > 50.0) {
      char extra[200];
      const WindowPresenter* os = output.presenter();
      const WindowPresenter* cs = controls.presenter();
      std::snprintf(extra, sizeof(extra),
                    "Output on screen %d acquire %.1f present %.1f | Controls on screen %d acquire %.1f present %.1f",
                    output_on_screen, os ? os->last_acquire_ms() : 0.0, os ? os->last_present_ms() : 0.0,
                    controls_on_screen, cs ? cs->last_acquire_ms() : 0.0, cs ? cs->last_present_ms() : 0.0);
      trace.Print(t, extra);
    }

    const Clock::time_point now = Clock::now();
    stats.OnRender(now);
    stats.Maybe(now, capture.source(), slot, pipeline);
    if (gpu->errors().device_lost()) {
      std::fprintf(stderr, "FAIL: the GPU device was lost\n");
      exit_code = 1;
      break;
    }
    if (options.frames != 0 && pipeline.composed >= options.frames) {
      break;
    }
    if (options.seconds > 0.0 && Seconds(now - run_start) >= options.seconds) {
      break;
    }
  }
  capture.StopSource();
  gpu->WaitIdle();
  if (gpu->errors().count() != 0) {
    std::fprintf(stderr, "%u WebGPU errors during the run; last: %s\n", gpu->errors().count(),
                 gpu->errors().last().c_str());
    exit_code = 1;
  }
  if (tester) {
    // Criteria (fixed before the first run): in every phase, compositing and
    // presenting both windows never takes over 50 ms in one iteration; outside
    // the minimize/restore phases, no iteration exceeds 100 ms and at least
    // 90% of the source frame rate is composed. Minimizing runs AppKit's
    // animation inside glfwPollEvents; that stall is reported, not judged.
    constexpr double kMaxPresentMs = 50.0;
    constexpr double kMaxIterationMs = 100.0;
    constexpr double kMinComposedFraction = 0.9;
    const double source_fps = options.source == AppOptions::Source::kFile ? options.file.fps : 30.0;
    int failures = 0;
    std::printf("Occlusion test results (presenter '%s'):\n", output.presenter()->kind());
    for (std::size_t i = 1; i + 1 < OcclusionTester::phase_count(); ++i) {
      const auto& phase = OcclusionTester::phase(i);
      const double seconds = OcclusionTester::phase(i + 1).start_s - phase.start_s;
      const PhaseStats& ps = phase_stats[i];
      const double fps = static_cast<double>(ps.composed) / seconds;
      const bool appkit = phase.action == OcclusionTester::Phase::kMinimizeOutput ||
                          (i > 0 && OcclusionTester::phase(i - 1).action == OcclusionTester::Phase::kMinimizeOutput);
      const bool ok = ps.max_present_ms <= kMaxPresentMs &&
                      (appkit || (ps.max_iteration_ms <= kMaxIterationMs && fps >= kMinComposedFraction * source_fps));
      std::printf("  %-24s composed %5.1f fps, loop %5.1f it/s, max iteration %6.1f ms, max present %5.1f ms%s  %s\n",
                  phase.name, fps, static_cast<double>(ps.iterations) / seconds, ps.max_iteration_ms,
                  ps.max_present_ms, appkit ? " (AppKit minimize animation: iteration not judged)" : "",
                  ok ? "ok" : "FAIL");
      failures += ok ? 0 : 1;
    }
    std::printf(failures == 0 ? "OCCLUSION TEST PASS\n" : "OCCLUSION TEST FAIL (%d phases)\n", failures);
    if (failures != 0) {
      exit_code = 1;
    }
  }
  std::printf("Relume exited cleanly (exit code %d)\n", exit_code);
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
    } else if (arg == "--presenter") {
      const char* v = value();
      const std::string_view name = v != nullptr ? v : "";
      if (name == "layer") {
        options.presenter = PresenterKind::kLayer;
      } else if (name == "surface") {
        options.presenter = PresenterKind::kSurface;
      } else {
        error = "--presenter must be layer or surface";
        return std::nullopt;
      }
    } else if (arg == "--trace") {
      options.trace = true;
    } else if (arg == "--occlusion-test") {
      options.occlusion_test = true;
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
    } else if (arg == "--seconds") {
      const char* v = value();
      char* end = nullptr;
      const double sec = v != nullptr ? std::strtod(v, &end) : -1.0;
      if (v == nullptr || end == v || *end != '\0' || sec < 0.0) {
        error = "--seconds needs a number";
        return std::nullopt;
      }
      options.seconds = sec;
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
