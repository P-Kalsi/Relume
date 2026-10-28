#include "app/app.hpp"

#include "app/window.hpp"
#include "output/clean_window.hpp"
#include "platform/window_state.hpp"
#include "relume/app_state.hpp"
#include "relume/camera.hpp"
#include "relume/event_queue.hpp"
#include "relume/frame_queue.hpp"
#include "relume/guided_capture.hpp"
#include "relume/image_io.hpp"
#include "relume/inference_controller.hpp"
#include "relume/layout.hpp"
#include "relume/pixel_convert.hpp"
#include "relume/renderer.hpp"
#include "relume/stats.hpp"
#include "relume/test_guide.hpp"
#include "ui/control_window.hpp"

#include <GLFW/glfw3.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
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
    "                            loop never stalled and the output kept updating (exit 0/1)\n"
    "  --save-frames <N>         camera only, off by default: save N camera frames (1-60) as\n"
    "                            lossless RGB PPM for local tests, then quit. Frames stay local.\n"
    "  --save-interval <S>       seconds between saved frames (default 1; 0 = consecutive frames)\n"
    "  --save-delay <S>          seconds to wait before the first saved frame (default 5)\n"
    "  --save-dir <path>         where to save (default: third_party/assets/local_frames, git-ignored)\n"
    "  --guided-capture          off by default: take 12 stills and 2 bursts one at a time from the\n"
    "                            Controls window, saved to the --save-dir folder (file source: UI\n"
    "                            testing only, and only with an explicit --save-dir)\n"
    "  --inference off|cpu|coreml depth inference (default: coreml, CPU fallback; off for --headless,\n"
    "                            --guided-capture and --save-frames unless given; headless with\n"
    "                            --inference also checks that matching depth arrives)\n"
    "  --coreml-units ALL|CPUOnly|CPUAndGPU|CPUAndNeuralEngine  CoreML compute units (default CPUAndGPU)\n"
    "  --coreml-strict           diagnostic: refuse CPU nodes inside the CoreML session; with this\n"
    "                            model the session fails, so the run shows the automatic CPU fallback\n"
    "  --model <path.onnx>       depth model (default: models/depth_anything_v2_small_448.onnx)\n"
    "  --threads <N>             ONNX Runtime CPU threads, 1-16 (default 4)\n"
    "  --preprocess stretch|letterbox  resize policy (default stretch)\n"
    "  --test-guide switching    show timed steps for the manual switching test in the Controls window\n"
    "                            (instructions only; you do each step; step changes are logged)\n"
    "  --inference-switch-test   switch CPU -> CoreML -> CPU -> off -> on and 10 rapid changes while\n"
    "                            measuring the loop; exit 0 only if no iteration exceeds 50 ms\n";

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
    std::uint64_t generation = 0;
    std::shared_ptr<const VideoFrame> frame = slot.TakeIfNew(last_frame_id, &generation);
    if (!frame) {
      return false;
    }
    last_frame_id = frame->frame_id;
    camera_generation = generation;
    frame_width = frame->width;
    frame_height = frame->height;
    if (!uploader.Upload(*frame)) {
      ++rejected;
      return false;
    }
    if (on_frame) {
      on_frame(*frame);
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
  std::uint64_t camera_generation = 0;  // LatestFrameSlot generation of the frame being shown
  std::uint32_t frame_width = 0;
  std::uint32_t frame_height = 0;
  std::uint64_t composed = 0;
  std::uint64_t rejected = 0;
  bool showing_video = false;
  std::function<void(const VideoFrame&)> on_frame;  // only with --save-frames or --guided-capture
};

// The main thread's view of depth: the newest result of the active worker,
// whether it belongs to the camera image being shown, and how old it is.
// Depth is never presented as in sync unless its frame is the one shown.
class DepthTracker {
 public:
  explicit DepthTracker(ResizeMethod method) : method_(method) {}

  // Takes the newest result of the active worker, if there is a new one.
  void Update(LatestDepthSlot& slot, const InferenceController& inference, const VideoPipeline& pipeline,
              Clock::time_point now) {
    const std::uint64_t generation = inference.active_generation();
    if (generation != worker_generation_) {
      worker_generation_ = generation;  // a new worker: forget the old one's depth
      last_sequence_ = 0;
      current_.reset();
    }
    if (std::shared_ptr<const DepthResult> result = slot.TakeIfNew(generation, last_sequence_)) {
      last_sequence_ = result->sequence;
      current_ = std::move(result);
      ++taken_;
      if (Match(pipeline) == DepthMatch::kMatch) {
        age_ms_.Record(std::chrono::duration<double, std::milli>(now - current_->arrival_time).count());
        ++lag_counts_[std::min<std::uint64_t>(Lag(pipeline), kLagBuckets - 1)];
      } else {
        ++mismatched_;
      }
    }
  }

  DepthMatch Match(const VideoPipeline& pipeline) const {
    // While the output is black (camera stopped, unavailable or switching),
    // the last frame shown is gone: no depth matches it.
    if (!pipeline.showing_video) {
      return current_ ? DepthMatch::kNoImage : DepthMatch::kNoDepth;
    }
    return MatchDepth(current_.get(), pipeline.camera_generation, pipeline.frame_width, pipeline.frame_height,
                      method_);
  }
  // Camera frames shown since the depth's frame (0 = in sync).
  std::uint64_t Lag(const VideoPipeline& pipeline) const {
    return current_ && pipeline.last_frame_id >= current_->frame_id ? pipeline.last_frame_id - current_->frame_id : 0;
  }
  const DepthResult* current() const { return current_.get(); }

  // Stats since the last Reset.
  std::uint64_t taken() const { return taken_; }
  std::uint64_t mismatched() const { return mismatched_; }
  double AgePercentile(double p) const { return age_ms_.Percentile(p); }
  std::uint64_t LagPercentile(double p) const {
    std::uint64_t total = 0;
    for (const std::uint64_t c : lag_counts_) {
      total += c;
    }
    const auto rank = static_cast<std::uint64_t>(std::ceil(p * static_cast<double>(total)));
    std::uint64_t seen = 0;
    for (std::size_t i = 0; i < kLagBuckets; ++i) {
      seen += lag_counts_[i];
      if (total > 0 && seen >= rank) {
        return i;
      }
    }
    return 0;
  }
  void ResetStats() {
    taken_ = 0;
    mismatched_ = 0;
    age_ms_.Reset();
    lag_counts_.fill(0);
  }

 private:
  static constexpr std::size_t kLagBuckets = 64;  // the last bucket means "63 or more"
  ResizeMethod method_;
  std::uint64_t worker_generation_ = 0;
  std::uint64_t last_sequence_ = 0;
  std::shared_ptr<const DepthResult> current_;
  std::uint64_t taken_ = 0;
  std::uint64_t mismatched_ = 0;
  WideLatencyHistogram age_ms_;
  std::array<std::uint64_t, kLagBuckets> lag_counts_{};
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
             const VideoPipeline& pipeline, const WindowPresenter* output = nullptr,
             const WindowPresenter* controls = nullptr, InferenceController* inference = nullptr,
             DepthTracker* depth = nullptr, const LatestDepthSlot* depth_slot = nullptr) {
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
        "rejected %llu | memory %.1f MiB\n",
        static_cast<double>(capture.frames_received - last_received_) / elapsed, capture.interval_ms_p50,
        capture.interval_ms_p95, capture.copy_ms_p50,
        capture.copy_ms_p95, static_cast<double>(pipeline.composed - last_composed_) / elapsed,
        static_cast<double>(renders_ - last_renders_) / elapsed, render_interval_ms_.Percentile(0.5),
        render_interval_ms_.Percentile(0.95), max_interval_ms_, static_cast<unsigned long long>(capture.dropped_pool),
        static_cast<unsigned long long>(slot_stats.dropped_unread),
        static_cast<unsigned long long>(capture.dropped_driver),
        static_cast<unsigned long long>(capture.dropped_invalid),
        static_cast<unsigned long long>(pipeline.rejected), ProcessFootprintMiB());
    if (output != nullptr && controls != nullptr) {
      auto rate = [&](std::uint64_t now_count, std::uint64_t& last) {
        const double r = static_cast<double>(now_count - last) / elapsed;
        last = now_count;
        return r;
      };
      const double op = rate(output->presented_frames(), last_out_presented_);
      const double os = rate(output->skipped_frames(), last_out_skipped_);
      const double cp = rate(controls->presented_frames(), last_ctl_presented_);
      const double cs = rate(controls->skipped_frames(), last_ctl_skipped_);
      std::printf("[stats] windows: Output presented %.1f/s skipped %.1f/s | Controls presented %.1f/s skipped %.1f/s\n",
                  op, os, cp, cs);
    }
    if (inference != nullptr && depth != nullptr && depth_slot != nullptr) {
      PrintDepth(elapsed, *inference, *depth, *depth_slot, pipeline);
    }
    std::fflush(stdout);
    last_ = now;
    last_received_ = capture.frames_received;
    last_composed_ = pipeline.composed;
    last_renders_ = renders_;
    render_interval_ms_.Reset();
    max_interval_ms_ = 0.0;
  }

 private:
  void PrintDepth(double elapsed, InferenceController& inference, DepthTracker& depth,
                  const LatestDepthSlot& depth_slot, const VideoPipeline& pipeline) {
    const InferenceStatus status = inference.status();
    DepthWorker* worker = inference.active();
    if (worker == nullptr) {
      std::printf("[stats] depth: %s\n", status.label.c_str());
      return;
    }
    const WorkerSnapshot snapshot = worker->snapshot();
    if (worker != last_worker_) {
      last_worker_ = worker;
      last_results_ = 0;
    }
    const SlotStats slot_stats = depth_slot.stats();
    // Inference stage times are measured on the worker; age (now minus the
    // source frame's arrival, steady clock) and lag (camera frames shown since
    // the depth's frame) when the main thread takes a result that matches the
    // frame being shown.
    std::printf(
        "[stats] depth: %s | %.1f results/s | convert p50 %.0f p95 %.0f, preprocess p50 %.0f p95 %.0f, "
        "run p50 %.0f p95 %.0f, total p50 %.0f p95 %.0f ms | age p50 %.0f p95 %.0f ms | lag p50 %llu p95 %llu "
        "frames | now: %s, lag %llu | unread-dropped %llu, mismatched %llu, invalid range %llu, bad frames %llu\n",
        status.label.c_str(), static_cast<double>(snapshot.results - last_results_) / elapsed,
        worker->convert_ms.Percentile(0.5), worker->convert_ms.Percentile(0.95), worker->preprocess_ms.Percentile(0.5),
        worker->preprocess_ms.Percentile(0.95), worker->run_ms.Percentile(0.5), worker->run_ms.Percentile(0.95),
        worker->total_ms.Percentile(0.5), worker->total_ms.Percentile(0.95), depth.AgePercentile(0.5),
        depth.AgePercentile(0.95), static_cast<unsigned long long>(depth.LagPercentile(0.5)),
        static_cast<unsigned long long>(depth.LagPercentile(0.95)), ToString(depth.Match(pipeline)).data(),
        static_cast<unsigned long long>(depth.Lag(pipeline)), static_cast<unsigned long long>(slot_stats.dropped_unread),
        static_cast<unsigned long long>(depth.mismatched()), static_cast<unsigned long long>(snapshot.invalid_ranges),
        static_cast<unsigned long long>(snapshot.invalid_frames));
    last_results_ = snapshot.results;
    for (WideLatencyHistogram* h : {&worker->convert_ms, &worker->preprocess_ms, &worker->run_ms, &worker->total_ms}) {
      h->Reset();
    }
    depth.ResetStats();
  }

  double interval_s_;
  const DepthWorker* last_worker_ = nullptr;
  std::uint64_t last_results_ = 0;
  Clock::time_point last_;
  Clock::time_point last_render_{};
  LatencyHistogram render_interval_ms_;
  double max_interval_ms_ = 0.0;  // catches single stalls that percentiles hide
  const ICameraSource* last_source_ = nullptr;
  std::uint64_t last_out_presented_ = 0, last_out_skipped_ = 0, last_ctl_presented_ = 0, last_ctl_skipped_ = 0;
  std::uint64_t renders_ = 0;
  std::uint64_t last_renders_ = 0;
  std::uint64_t last_received_ = 0;
  std::uint64_t last_composed_ = 0;
};

// The depth debug view (Controls window only): uploads depth that matches
// the camera image being shown and draws it, aligned with the output, into
// the DepthDebugTexture. Anything that does not match shows black ("no
// depth"), never stale or misaligned depth. The display range is the worker's
// 2nd-98th percentile of the valid region, smoothed across results so the
// view does not flicker; it restarts when the camera or worker changes.
class DepthView {
 public:
  explicit DepthView(const GpuContext& gpu) : texture(gpu), uploader_(gpu), visualizer_(gpu, texture) {
    visualizer_.Clear();
  }

  void Update(const DepthTracker& depth, const VideoPipeline& pipeline, bool overlay) {
    const DepthResult* result = depth.current();
    if (depth.Match(pipeline) != DepthMatch::kMatch) {
      if (showing_) {
        visualizer_.Clear();
        showing_ = false;
      }
      return;
    }
    const bool new_result = result->worker_generation != worker_generation_ || result->sequence != sequence_;
    if (new_result) {
      if (result->worker_generation != worker_generation_ || result->camera_generation != camera_generation_) {
        range_ = {};  // restart smoothing for a new worker or camera
      }
      worker_generation_ = result->worker_generation;
      camera_generation_ = result->camera_generation;
      sequence_ = result->sequence;
      uploader_.Upload(result->depth.data(), result->transform);
      if (result->range.valid) {
        constexpr float kSmoothing = 0.3f;  // weight of the newest result
        range_.lo = range_.valid ? range_.lo + kSmoothing * (result->range.lo - range_.lo) : result->range.lo;
        range_.hi = range_.valid ? range_.hi + kSmoothing * (result->range.hi - range_.hi) : result->range.hi;
        range_.valid = range_.hi > range_.lo;
      }
    }
    // Redraw for new depth, a new camera frame (the overlay shows it) or an
    // overlay change.
    if (new_result || pipeline.composed != composed_ || overlay != overlay_ || !showing_) {
      DepthVisParams params;
      params.lo = range_.lo;
      params.hi = range_.hi;
      params.valid = range_.valid;
      params.camera_weight = overlay ? 0.5f : 0.0f;
      visualizer_.Draw(uploader_, result->transform, params, pipeline.uploader.camera_texture());
      composed_ = pipeline.composed;
      overlay_ = overlay;
      showing_ = true;
    }
  }

  DepthDebugTexture texture;

 private:
  DepthUploader uploader_;
  DepthVisualizer visualizer_;
  DisplayRange range_;
  std::uint64_t worker_generation_ = 0;
  std::uint64_t camera_generation_ = 0;
  std::uint64_t sequence_ = 0;
  std::uint64_t composed_ = 0;
  bool overlay_ = false;
  bool showing_ = false;
};

// A label that never presents old depth as current: "in sync" only when the
// depth's frame is the one being shown.
std::string DepthLabel(const DepthTracker& depth, const VideoPipeline& pipeline, Clock::time_point now) {
  const DepthResult* result = depth.current();
  const DepthMatch match = depth.Match(pipeline);
  if (result == nullptr) {
    return "No depth yet";
  }
  if (match != DepthMatch::kMatch) {
    return std::string("No depth for this image (") + ToString(match).data() + ")";
  }
  const double age_ms = std::chrono::duration<double, std::milli>(now - result->arrival_time).count();
  const std::uint64_t lag = depth.Lag(pipeline);
  char text[128];
  if (lag == 0) {
    std::snprintf(text, sizeof(text), "In sync with the shown frame (frame %llu, %.0f ms old)",
                  static_cast<unsigned long long>(result->frame_id), age_ms);
  } else {
    std::snprintf(text, sizeof(text), "Depth from %llu frame%s ago (%.0f ms old)", static_cast<unsigned long long>(lag),
                  lag == 1 ? "" : "s", age_ms);
  }
  return text;
}

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

  // With --inference: the real depth pipeline runs too, and the run waits
  // for a depth result that matches the frame being shown.
  LatestDepthSlot depth_slot;
  InferenceController inference(MakeOnnxDepthEstimator, slot, depth_slot);
  DepthTracker depth(options.preprocess);
  const std::optional<DepthWorkerConfig> inference_config = InferenceConfig(options);
  if (inference_config) {
    inference.Request(inference_config);
  }
  const auto depth_done = [&] {
    return !inference_config || depth.Match(pipeline) == DepthMatch::kMatch ||
           inference.status().state == InferenceState::kError;
  };

  const std::uint64_t target = options.frames == 0 ? 30 : options.frames;
  const Clock::time_point start = Clock::now();
  const Clock::time_point deadline = start + std::chrono::seconds(inference_config ? 90 : 30);
  while ((pipeline.composed < target || !depth_done()) && source.state() != SourceState::kFailed &&
         Clock::now() < deadline) {
    if (!pipeline.Update(slot)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    inference.Update();
    depth.Update(depth_slot, inference, pipeline, Clock::now());
    gpu->instance().ProcessEvents();
  }
  source.Stop();
  int failures = 0;
  if (inference_config) {
    const InferenceStatus status = inference.status();
    const DepthResult* result = depth.current();
    const DepthMatch match = depth.Match(pipeline);
    std::printf("Depth: %s | %s\n", status.label.c_str(), status.detail.c_str());
    if (result != nullptr) {
      const ResizeTransform& t = result->transform;
      std::printf("  first matching result after %.0f ms: frame %llu (showing %llu, lag %llu), %ux%u %s -> %ux%u at "
                  "(%u,%u); convert %.1f, preprocess %.1f, run %.1f ms; display range %s [%.3f, %.3f]\n",
                  std::chrono::duration<double, std::milli>(result->finished_time - start).count(),
                  static_cast<unsigned long long>(result->frame_id),
                  static_cast<unsigned long long>(pipeline.last_frame_id),
                  static_cast<unsigned long long>(depth.Lag(pipeline)), t.src_w, t.src_h, ToString(t.method).data(),
                  t.new_w, t.new_h, t.pad_x, t.pad_y, result->convert_ms, result->preprocess_ms, result->run_ms,
                  result->range.valid ? "valid" : "INVALID", result->range.lo, result->range.hi);
    }
    // Ready, or CPU fallback when CoreML could not be used (the label says
    // which; the fallback CTest checks for it explicitly).
    const bool running = status.state == InferenceState::kReady || status.state == InferenceState::kCpuFallback;
    const bool ok = running && match == DepthMatch::kMatch && result != nullptr &&
                    result->range.valid && result->frame_id <= pipeline.last_frame_id &&
                    result->camera_generation == slot.generation();
    std::printf("  depth matches the shown camera image (generation, size, transform): %s (%s)\n", ok ? "yes" : "NO",
                ToString(match).data());
    failures += ok ? 0 : 1;
  }
  if (inference_config) {
    std::printf("Inference shutdown: joined in %.1f ms\n", inference.Shutdown());
  }
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
  std::uint64_t output_presented = 0, output_skipped = 0, controls_presented = 0, controls_skipped = 0;
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
  DepthView depth_view(*gpu);
  ControlWindow controls(*gpu, pipeline.output, options.presenter);
  CleanWindow output(*gpu, options.presenter);
  if (!controls.ok() || !output.ok()) {
    std::fprintf(stderr, "FAIL: %s\n", !controls.ok() ? controls.error().c_str() : output.error().c_str());
    return 1;
  }
  std::printf("Presentation: Output '%s', Controls '%s'\n", output.presenter()->kind(), controls.presenter()->kind());
  ArrangeSideBySide(controls.window(), output.window());
  output.Show();

  std::optional<FrameSaver> saver;
  if (options.save.count > 0) {
    saver.emplace(options.save);
    if (!saver->error().empty()) {
      std::fprintf(stderr, "FAIL: --save-frames: %s\n", saver->error().c_str());
      return 1;
    }
    std::printf("Saving %u camera frames to %s (interval %.2f s, first after %.1f s); nothing leaves this Mac\n",
                options.save.count, options.save.directory.c_str(), options.save.interval_s, options.save.delay_s);
    pipeline.on_frame = [&saver](const VideoFrame& frame) { saver->Offer(frame, Clock::now()); };
  }
  std::optional<GuidedCapture> guided;
  if (options.guided_capture) {
    FrameSaverOptions manual = options.save;
    manual.count = 0;
    saver.emplace(manual);
    if (!saver->error().empty()) {
      std::fprintf(stderr, "FAIL: --guided-capture: %s\n", saver->error().c_str());
      return 1;
    }
    guided.emplace(*saver, DefaultCapturePlan());
    std::printf("Guided capture: shots are saved in %s; nothing leaves this Mac\n", options.save.directory.c_str());
#ifndef NDEBUG
    std::printf("Guided capture: warning: this is a Debug build; converting a 720p NV12 frame takes about 50 ms, "
                "so 30 fps bursts will fail. Use the Release build for real captures.\n");
#endif
    pipeline.on_frame = [&guided](const VideoFrame& frame) { guided->OnFrame(frame, Clock::now()); };
  }

  CameraEventQueue events;
  LatestFrameSlot slot;
  CaptureController capture(options, events, slot);
  // Depth inference on its own thread; declared after the frame slot it
  // reads, so it is destroyed (and its workers joined) first.
  LatestDepthSlot depth_slot;
  InferenceController inference(MakeOnnxDepthEstimator, slot, depth_slot);
  DepthTracker depth(options.preprocess);
  AppOptions inference_options = options;  // changed by the Controls window's inference choice
  inference.Request(InferenceConfig(inference_options));
  InferenceStatus inference_status;
  bool depth_overlay = false;
  const auto choice_index = [](const AppOptions& o) {
    for (int i = 0; i < static_cast<int>(std::size(kInferenceChoices)); ++i) {
      const InferenceChoice& c = kInferenceChoices[i];
      if (c.provider == o.inference && (o.inference != ExecutionProvider::kCoreML || c.units == o.coreml_units)) {
        return i;
      }
    }
    return 0;
  };

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
  // --guided-self-test: one still, then the first burst, clicked by script.
  // Criteria (fixed in advance): both shots are saved, the burst folder holds
  // all of its frames, and no loop iteration takes over 50 ms while a shot is
  // being captured or written (the loop never waits for the disk).
  enum class SelfTest : std::uint8_t { kIdle, kStill, kBurst, kDone } self_test = SelfTest::kIdle;
  // --inference-switch-test: each step is requested once the previous one
  // has settled (Ready, CPU fallback or Off). Criteria (fixed in advance):
  // every step settles within 120 s, no loop iteration takes over 50 ms after
  // the first 2 s, and every worker is joined at the end.
  struct SwitchStep {
    const char* name;
    std::optional<ExecutionProvider> provider;
    CoreMLUnits units;
    int rapid;  // extra rapid requests before this one (coalescing)
  };
  static constexpr SwitchStep kSwitchSteps[] = {
      {"CPU -> CoreML ALL", ExecutionProvider::kCoreML, CoreMLUnits::kAll, 0},
      {"CoreML -> CPU", ExecutionProvider::kCpu, CoreMLUnits::kAll, 0},
      {"CPU -> off", std::nullopt, CoreMLUnits::kAll, 0},
      {"off -> CPU", ExecutionProvider::kCpu, CoreMLUnits::kAll, 0},
      {"10 rapid changes, ending on CPU", ExecutionProvider::kCpu, CoreMLUnits::kAll, 10},
  };
  std::optional<TestGuide> test_guide;
  if (options.test_guide == "switching") {
    test_guide.emplace(SwitchingTestGuide(), run_start);
    std::printf("[guide] t=0.0s step 1/%zu: %s\n", test_guide->size(), test_guide->step(0).title.c_str());
  }
  std::size_t switch_step = 0;
  Clock::time_point switch_requested{};
  Clock::time_point switch_settled = run_start;
  double switch_max_iteration_ms = 0.0;  // max of start-to-start and end-to-end iteration intervals
  Clock::time_point last_iteration_end{};
  bool switch_failed = false;
  double self_test_max_iteration_ms = 0.0;
  bool self_test_failed = false;
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
    const std::uint64_t out_p0 = output.presenter()->presented_frames(), out_s0 = output.presenter()->skipped_frames();
    const std::uint64_t ctl_p0 = controls.presenter()->presented_frames(), ctl_s0 = controls.presenter()->skipped_frames();
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
    inference.Update();  // never waits for a worker
    depth.Update(depth_slot, inference, pipeline, Clock::now());
    const InferenceStatus next_inference = inference.status();
    if (next_inference.label != inference_status.label || next_inference.detail != inference_status.detail) {
      std::printf("[inference] %s: %s\n", next_inference.label.c_str(), next_inference.detail.c_str());
      std::fflush(stdout);
    }
    inference_status = next_inference;
    depth_view.Update(depth, pipeline, depth_overlay);
    trace.Mark("compose");
    output.Render(pipeline.presenter);
    trace.Mark("output");

    ControlsModel model;
    if (guided) {
      guided->Update(Clock::now());
      model.guided = &*guided;
      model.now = Clock::now();
    }
    if (test_guide) {
      if (test_guide->Update(Clock::now())) {
        std::printf("[guide] t=%.1fs step %zu/%zu: %s\n", test_guide->elapsed(Clock::now()), test_guide->current() + 1,
                    test_guide->size(), test_guide->step(test_guide->current()).title.c_str());
        std::fflush(stdout);
      }
      model.test_guide = &*test_guide;
      model.now = Clock::now();
    }
    model.status = status;
    model.output_visible = output.visible();
    capture.FillModel(model);
    model.inference = inference_status;
    model.inference_choice = choice_index(inference_options);
    model.depth_view = &depth_view.texture;
    model.depth_overlay = depth_overlay;
    if (inference.active() != nullptr) {
      model.depth_label = DepthLabel(depth, pipeline, Clock::now());
      const DepthWorker& worker = *inference.active();
      if (worker.run_ms.count() > 0) {
        char timing[96];
        std::snprintf(timing, sizeof(timing), "run p50 %.0f ms, p95 %.0f ms (since the last stats line)",
                      worker.run_ms.Percentile(0.5), worker.run_ms.Percentile(0.95));
        model.inference_timing = timing;
      }
    }
    const ControlsActions actions = controls.Render(model);
    trace.Mark("controls");
    if (guided && actions.guided_select) {
      guided->Select(*actions.guided_select);
    }
    if (guided && actions.guided_capture) {
      guided->StartCapture(Clock::now());
      std::printf("[guided] %s: countdown started\n", guided->step(guided->current()).name.c_str());
    }
    if (actions.toggle_output) {
      output.visible() ? output.Hide() : output.Show();
    }
    if (test_guide && actions.test_guide_skip) {
      test_guide->Skip(Clock::now());
      std::printf("[guide] t=%.1fs step %zu/%zu (skipped to): %s\n", test_guide->elapsed(Clock::now()),
                  test_guide->current() + 1, test_guide->size(), test_guide->step(test_guide->current()).title.c_str());
      std::fflush(stdout);
    }
    if (actions.toggle_depth_overlay) {
      depth_overlay = !depth_overlay;
    }
    if (actions.inference_choice) {
      const InferenceChoice& choice = kInferenceChoices[*actions.inference_choice];
      inference_options.inference = choice.provider;
      inference_options.coreml_units = choice.units;
      std::printf("[inference] requested: %s\n", choice.label);
      inference.Request(InferenceConfig(inference_options));  // applied by Update(); never waits
    }
    if (guided && options.guided_self_test) {
      if (guided->busy()) {
        self_test_max_iteration_ms = std::max(
            self_test_max_iteration_ms, std::chrono::duration<double, std::milli>(iteration_start - previous_start).count());
      }
      std::size_t burst = 0;
      while (burst < guided->size() && guided->step(burst).kind != CaptureStep::Kind::kBurst) {
        ++burst;
      }
      if (self_test == SelfTest::kIdle && t >= 1.0) {
        guided->Select(0);
        guided->StartCapture(Clock::now());
        self_test = SelfTest::kStill;
      } else if (self_test == SelfTest::kStill && !guided->busy()) {
        guided->Select(burst);
        guided->StartCapture(Clock::now());
        self_test = SelfTest::kBurst;
      } else if (self_test == SelfTest::kBurst && !guided->busy()) {
        self_test = SelfTest::kDone;
        break;
      }
      if (t > 30.0) {
        std::fprintf(stderr, "FAIL: guided self-test timed out\n");
        self_test_failed = true;
        break;
      }
    }
    if (options.inference_switch_test) {
      if (t >= 2.0) {
        switch_max_iteration_ms = std::max(
            switch_max_iteration_ms, std::chrono::duration<double, std::milli>(iteration_start - previous_start).count());
      }
      const InferenceState state = inference.status().state;
      const bool settled = !inference.pending() && !inference.retiring() &&
                           (state == InferenceState::kReady || state == InferenceState::kCpuFallback ||
                            state == InferenceState::kOff || state == InferenceState::kError);
      const Clock::time_point now_switch = Clock::now();
      if (switch_requested != Clock::time_point{} && settled) {
        std::printf("[switch-test] %s: settled as '%s' after %.0f ms\n", kSwitchSteps[switch_step - 1].name,
                    inference.status().label.c_str(),
                    std::chrono::duration<double, std::milli>(now_switch - switch_requested).count());
        switch_requested = {};
        switch_settled = now_switch;
        if (state == InferenceState::kError) {
          switch_failed = true;
        }
      }
      if (switch_requested == Clock::time_point{} && settled && Seconds(now_switch - switch_settled) >= 3.0) {
        if (switch_step == std::size(kSwitchSteps)) {
          break;
        }
        const SwitchStep& step = kSwitchSteps[switch_step++];
        for (int i = 0; i < step.rapid; ++i) {
          inference_options.inference = i % 2 == 0 ? ExecutionProvider::kCoreML : ExecutionProvider::kCpu;
          inference_options.threads = 2 + i % 3;
          inference.Request(InferenceConfig(inference_options));
          inference.Update();
        }
        inference_options.inference = step.provider;
        inference_options.coreml_units = step.units;
        inference_options.threads = options.threads;
        inference.Request(InferenceConfig(inference_options));
        switch_requested = now_switch;
        std::printf("[switch-test] t=%.1fs requesting: %s\n", t, step.name);
        std::fflush(stdout);
      }
      if (switch_requested != Clock::time_point{} && Seconds(now_switch - switch_requested) > 120.0) {
        std::fprintf(stderr, "FAIL: %s did not settle within 120 s\n", kSwitchSteps[switch_step - 1].name);
        switch_failed = true;
        break;
      }
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
      ps.output_presented += output.presenter()->presented_frames() - out_p0;
      ps.output_skipped += output.presenter()->skipped_frames() - out_s0;
      ps.controls_presented += controls.presenter()->presented_frames() - ctl_p0;
      ps.controls_skipped += controls.presenter()->skipped_frames() - ctl_s0;
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
    if (options.inference_switch_test && t >= 2.0 && last_iteration_end != Clock::time_point{}) {
      // The stats log's measure (end to end), in addition to start to start.
      switch_max_iteration_ms =
          std::max(switch_max_iteration_ms, std::chrono::duration<double, std::milli>(now - last_iteration_end).count());
    }
    last_iteration_end = now;
    stats.OnRender(now);
    stats.Maybe(now, capture.source(), slot, pipeline, output.presenter(), controls.presenter(), &inference, &depth,
                &depth_slot);
    if (gpu->errors().device_lost()) {
      std::fprintf(stderr, "FAIL: the GPU device was lost\n");
      exit_code = 1;
      break;
    }
    if (options.frames != 0 && pipeline.composed >= options.frames) {
      break;
    }
    if (saver && !guided && saver->done()) {
      break;
    }
    if (options.seconds > 0.0 && Seconds(now - run_start) >= options.seconds) {
      break;
    }
  }
  // Shutdown order: stop capture, hide the windows, then stop and join the
  // depth workers. A join can wait for a CoreML compile in progress; it
  // happens after the UI is gone and its duration is logged.
  capture.StopSource();
  pipeline.on_frame = nullptr;
  glfwHideWindow(output.window());
  glfwHideWindow(controls.window());
  const double shutdown_ms = inference.Shutdown();
  std::printf("Inference shutdown: %llu workers started, %llu joined; final join took %.1f ms\n",
              static_cast<unsigned long long>(inference.workers_started()),
              static_cast<unsigned long long>(inference.workers_joined()), shutdown_ms);
  if (options.inference_switch_test) {
    const bool all_steps = switch_step == std::size(kSwitchSteps) && switch_requested == Clock::time_point{};
    const bool joined = inference.workers_joined() == inference.workers_started();
    const bool timing_ok = switch_max_iteration_ms <= 50.0;
    std::printf("Inference switch test: %zu of %zu steps settled, workers started %llu joined %llu, max iteration "
                "%.1f ms (limit 50)\n",
                switch_step - (switch_requested == Clock::time_point{} ? 0 : 1), std::size(kSwitchSteps),
                static_cast<unsigned long long>(inference.workers_started()),
                static_cast<unsigned long long>(inference.workers_joined()), switch_max_iteration_ms);
    const bool pass = !switch_failed && all_steps && joined && timing_ok;
    std::printf(pass ? "INFERENCE SWITCH TEST PASS\n" : "INFERENCE SWITCH TEST FAIL\n");
    if (!pass) {
      exit_code = 1;
    }
  }
  if (guided) {
    std::printf("Guided capture: %u of %u stills and %u of %u bursts on disk; %u files written this session\n",
                guided->saved_stills(), guided->stills_total(), guided->saved_bursts(), guided->bursts_total(),
                guided->files_written());
    for (std::size_t i = 0; i < guided->size(); ++i) {
      std::printf("  %-9s %-9s %s\n", guided->step(i).name.c_str(), ToString(guided->status(i)).data(),
                  guided->message(i).c_str());
    }
    if (options.guided_self_test) {
      std::size_t burst = 0;
      while (burst < guided->size() && guided->step(burst).kind != CaptureStep::Kind::kBurst) {
        ++burst;
      }
      const bool still_ok = guided->status(0) == ShotStatus::kSaved;
      const bool burst_ok = guided->status(burst) == ShotStatus::kSaved;
      const bool files_ok = guided->files_written() == 1 + guided->step(burst).frames;
      const bool timing_ok = self_test_max_iteration_ms <= 50.0;
      std::printf("Guided self-test: still %s, burst %s, files %u (expected %u), max iteration while capturing "
                  "%.1f ms (limit 50), skipped by busy writer %u\n",
                  still_ok ? "saved" : "NOT saved", burst_ok ? "saved" : "NOT saved", guided->files_written(),
                  1 + guided->step(burst).frames, self_test_max_iteration_ms, saver->skipped_busy());
      const bool pass = !self_test_failed && self_test == SelfTest::kDone && still_ok && burst_ok && files_ok && timing_ok;
      std::printf(pass ? "GUIDED SELF-TEST PASS\n" : "GUIDED SELF-TEST FAIL\n");
      if (!pass) {
        exit_code = 1;
      }
    }
  } else if (saver) {
    for (const std::string& path : saver->paths()) {
      std::printf("Saved %s\n", path.c_str());
    }
    std::printf("Saved %u of %u frames (%u failed, %u skipped while the writer was busy)%s%s\n", saver->written(),
                options.save.count, saver->failed(), saver->skipped_busy(), saver->error().empty() ? "" : "; last error: ",
                saver->error().c_str());
    if (saver->written() < options.save.count) {
      exit_code = 1;
    }
  }
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
    // 90% of the source frame rate is composed; a fully visible window skips
    // at most 5% and presents at least 50% of iterations. Minimizing runs AppKit's
    // animation inside glfwPollEvents; that stall is reported, not judged.
    constexpr double kMaxPresentMs = 50.0;
    constexpr double kMaxIterationMs = 100.0;
    constexpr double kMinComposedFraction = 0.9;
    constexpr double kMinPresentFraction = 0.5;
    constexpr double kMaxSkipFraction = 0.05;
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
      // A fully visible window (not covered, minimized or being resized)
      // must skip at most 5% of iterations and present at least half of them
      // (30 frames/s at 60 Hz, every frame of a 30 fps source). Frames that
      // finish in the same tick as a newer one are superseded, not skipped.
      const bool all_visible = phase.action == OcclusionTester::Phase::kNone && !appkit;
      const double iterations = static_cast<double>(std::max<std::uint64_t>(ps.iterations, 1));
      auto window_ok = [&](std::uint64_t presented, std::uint64_t skipped) {
        return static_cast<double>(skipped) <= kMaxSkipFraction * iterations &&
               static_cast<double>(presented) >= kMinPresentFraction * iterations;
      };
      const bool presenting_ok = !all_visible || (window_ok(ps.output_presented, ps.output_skipped) &&
                                                  window_ok(ps.controls_presented, ps.controls_skipped));
      const bool ok = ps.max_present_ms <= kMaxPresentMs && presenting_ok &&
                      (appkit || (ps.max_iteration_ms <= kMaxIterationMs && fps >= kMinComposedFraction * source_fps));
      std::printf("  %-24s composed %5.1f fps, loop %5.1f it/s, max iteration %6.1f ms, max present %5.1f ms | "
                  "Output presented %4llu skipped %4llu, Controls presented %4llu skipped %4llu%s  %s\n",
                  phase.name, fps, static_cast<double>(ps.iterations) / seconds, ps.max_iteration_ms,
                  ps.max_present_ms, static_cast<unsigned long long>(ps.output_presented),
                  static_cast<unsigned long long>(ps.output_skipped),
                  static_cast<unsigned long long>(ps.controls_presented),
                  static_cast<unsigned long long>(ps.controls_skipped),
                  appkit ? " (AppKit minimize animation: iteration not judged)" : "", ok ? "ok" : "FAIL");
      failures += ok ? 0 : 1;
    }
    for (const WindowPresenter* p : {output.presenter(), controls.presenter()}) {
      if (!p->diagnostics().empty()) {
        std::printf("  %s presenter: %s\n", p == output.presenter() ? CleanWindow::kTitle : ControlWindow::kTitle,
                    p->diagnostics().c_str());
      }
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
  options.save.directory = RELUME_DEFAULT_FRAME_DIR;
  options.model_path = RELUME_DEFAULT_MODEL;
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
    } else if (arg == "--save-frames") {
      const char* v = value();
      char* end = nullptr;
      const unsigned long n = v != nullptr ? std::strtoul(v, &end, 10) : 0;
      if (v == nullptr || end == v || *end != '\0' || n < 1 || n > FrameSaver::kMaxFrames) {
        error = "--save-frames needs a number from 1 to " + std::to_string(FrameSaver::kMaxFrames);
        return std::nullopt;
      }
      options.save.count = static_cast<std::uint32_t>(n);
    } else if (arg == "--save-interval" || arg == "--save-delay") {
      const char* v = value();
      char* end = nullptr;
      const double sec = v != nullptr ? std::strtod(v, &end) : -1.0;
      if (v == nullptr || end == v || *end != '\0' || !(sec >= 0.0) || sec > 600.0) {
        error = std::string(arg) + " needs a number of seconds (0-600)";
        return std::nullopt;
      }
      (arg == "--save-interval" ? options.save.interval_s : options.save.delay_s) = sec;
    } else if (arg == "--guided-capture") {
      options.guided_capture = true;
    } else if (arg == "--guided-self-test") {
      options.guided_self_test = true;
    } else if (arg == "--save-dir") {
      const char* v = value();
      if (v == nullptr || *v == '\0') {
        error = "--save-dir needs a path";
        return std::nullopt;
      }
      options.save.directory = v;
    } else if (arg == "--test-guide") {
      const char* v = value();
      if (v == nullptr || std::string_view(v) != "switching") {
        error = "--test-guide must be: switching";
        return std::nullopt;
      }
      options.test_guide = v;
    } else if (arg == "--inference-switch-test") {
      options.inference_switch_test = true;
    } else if (arg == "--inference") {
      const char* v = value();
      const std::string_view name = v != nullptr ? v : "";
      if (name == "off") {
        options.inference = std::nullopt;
      } else if (const std::optional<ExecutionProvider> provider = ParseExecutionProvider(name)) {
        options.inference = provider;
      } else {
        error = "--inference must be off, cpu or coreml";
        return std::nullopt;
      }
      options.inference_set = true;
    } else if (arg == "--coreml-units") {
      const char* v = value();
      const std::optional<CoreMLUnits> units = ParseCoreMLUnits(v != nullptr ? v : "");
      if (!units) {
        error = "--coreml-units must be ALL, CPUOnly, CPUAndGPU or CPUAndNeuralEngine";
        return std::nullopt;
      }
      options.coreml_units = *units;
    } else if (arg == "--coreml-strict") {
      options.coreml_strict = true;
    } else if (arg == "--model") {
      const char* v = value();
      if (v == nullptr || *v == '\0') {
        error = "--model needs a path";
        return std::nullopt;
      }
      options.model_path = v;
    } else if (arg == "--threads") {
      const char* v = value();
      char* end = nullptr;
      const long n = v != nullptr ? std::strtol(v, &end, 10) : 0;
      if (v == nullptr || end == v || *end != '\0' || n < 1 || n > 16) {
        error = "--threads needs a number from 1 to 16";
        return std::nullopt;
      }
      options.threads = static_cast<int>(n);
    } else if (arg == "--preprocess") {
      const char* v = value();
      const std::optional<ResizeMethod> method = ParseResizeMethod(v != nullptr ? v : "");
      if (!method) {
        error = "--preprocess must be stretch or letterbox";
        return std::nullopt;
      }
      options.preprocess = *method;
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
  if (options.guided_capture && options.save.count > 0) {
    error = "use either --guided-capture or --save-frames, not both";
    return std::nullopt;
  }
  if (options.guided_self_test && (!options.guided_capture || options.source != AppOptions::Source::kFile)) {
    error = "--guided-self-test needs --guided-capture --source file";
    return std::nullopt;
  }
  if (options.guided_capture && options.headless) {
    error = "--guided-capture needs the Controls window";
    return std::nullopt;
  }
  // The file source is allowed for testing the UI, but never into the default
  // folder, which must hold only real webcam frames for the comparison.
  if (options.guided_capture && options.source == AppOptions::Source::kFile &&
      options.save.directory == RELUME_DEFAULT_FRAME_DIR) {
    error = "--guided-capture with --source file needs an explicit --save-dir outside local_frames";
    return std::nullopt;
  }
  if (options.save.count > 0 && (options.source != AppOptions::Source::kCamera || options.headless)) {
    error = "--save-frames saves camera frames; it needs --source camera and windows";
    return std::nullopt;
  }
  if (options.headless && options.source != AppOptions::Source::kFile) {
    error = "--headless needs --source file (it never opens the camera)";
    return std::nullopt;
  }
  // Headless runs and the frame-capture modes do not need depth; capture in
  // particular should not compete with inference for CPU (a Debug build's
  // burst writer could not keep up). --inference turns it on explicitly.
  if ((options.headless || options.guided_capture || options.save.count > 0) && !options.inference_set) {
    options.inference = std::nullopt;
  }
  if (!options.test_guide.empty() && options.headless) {
    error = "--test-guide needs the Controls window";
    return std::nullopt;
  }
  if (options.inference_switch_test && (options.headless || !options.inference)) {
    error = "--inference-switch-test needs windows and inference on";
    return std::nullopt;
  }
  return options;
}

namespace {

// Where CoreML keeps the models it compiles from ours (compiled models only,
// never frames): ~/Library/Caches/dev.relume.app/coreml/<model>/<units>.
// ONNX Runtime does not notice a changed model in its cache (its key is the
// model path), so the folder name includes the model file's size and
// modification time; a new export gets a fresh cache.
std::string CoreMLCacheDirectory(const std::string& model_path, CoreMLUnits units) {
  const std::string caches = UserCachesDirectory();
  std::error_code ec;
  const std::filesystem::path model(model_path);
  const auto size = std::filesystem::file_size(model, ec);
  if (caches.empty() || ec) {
    return {};
  }
  const long long modified = static_cast<long long>(std::filesystem::last_write_time(model, ec).time_since_epoch().count());
  if (ec) {
    return {};
  }
  const std::string key = model.stem().string() + "-" + std::to_string(static_cast<unsigned long long>(size)) + "-" +
                          std::to_string(modified);
  return (std::filesystem::path(caches) / "dev.relume.app" / "coreml" / key / std::string(ToString(units))).string();
}

}  // namespace

std::optional<DepthWorkerConfig> InferenceConfig(const AppOptions& options) {
  if (!options.inference) {
    return std::nullopt;
  }
  DepthWorkerConfig config;
  config.estimator.model_path = options.model_path;
  config.estimator.provider = *options.inference;
  config.estimator.coreml_units = options.coreml_units;
  if (config.estimator.provider == ExecutionProvider::kCoreML) {
    config.estimator.coreml_cache_dir = CoreMLCacheDirectory(options.model_path, options.coreml_units);
    config.estimator.coreml_strict = options.coreml_strict;
  }
  config.estimator.intra_op_threads = options.threads;
  config.method = options.preprocess;
  return config;
}

int RunApp(const AppOptions& options) { return options.headless ? RunHeadless(options) : RunWindowed(options); }

}  // namespace relume
