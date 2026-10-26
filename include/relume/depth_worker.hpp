#pragma once

// Depth inference on a dedicated worker thread.
//
// DepthWorker creates its estimator on its own thread (so a slow session
// creation never blocks the UI), warms it up, then repeatedly takes the
// newest camera frame from the LatestFrameSlot, converts it to RGB, releases
// the camera frame at once (it never holds a pool frame during inference),
// preprocesses, runs the model and publishes a DepthResult to a
// LatestDepthSlot. Frames that arrive while a run is in progress are never
// queued: the next run takes whatever is newest.
//
// Results come from a small recycling pool, so steady state allocates
// nothing in Relume. Each result records the frame it came from (ID, camera
// generation, arrival time, size and transform) so the renderer can refuse
// depth that does not belong to the current camera image.
//
// Threads are always joined, never detached. No exception leaves the
// worker thread; a failure becomes the kFailed phase with a message.
// Portable C++: the estimator comes from a factory (ONNX Runtime in the app,
// fakes in tests).

#include "relume/depth_estimator.hpp"
#include "relume/frame_queue.hpp"
#include "relume/preprocess.hpp"
#include "relume/stats.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace relume {

// Display normalization range: the 2nd and 98th percentiles of the valid
// (unpadded) region. `valid` is false for an empty region, a non-finite value
// or a constant map (range below 1e-6 of its magnitude); the view then shows
// a neutral image instead of dividing by zero.
struct DisplayRange {
  float lo = 0.0f;
  float hi = 0.0f;
  bool valid = false;
};

// Percentiles over depth[pad_y : pad_y+new_h, pad_x : pad_x+new_w] of the
// 448x448 tensor only; letterbox padding is never included. `scratch` is
// reused between calls.
DisplayRange ComputeDisplayRange(const float* depth, const ResizeTransform& transform, std::vector<float>& scratch);

struct DepthResult {
  using Clock = std::chrono::steady_clock;

  std::uint64_t sequence = 0;           // per worker, from 1
  std::uint64_t worker_generation = 0;  // the worker that produced it
  std::uint64_t camera_generation = 0;  // LatestFrameSlot generation of the source frame
  std::uint64_t frame_id = 0;           // the source frame
  std::int64_t capture_time_ns = 0;
  Clock::time_point arrival_time{};     // the source frame's arrival (steady clock)
  Clock::time_point finished_time{};
  ResizeTransform transform;            // includes the source size
  std::vector<float> depth;             // raw model output, 448x448 row-major
  DisplayRange range;
  double convert_ms = 0.0;     // camera frame -> RGB
  double preprocess_ms = 0.0;  // RGB -> tensor
  double run_ms = 0.0;         // the model
  double total_ms = 0.0;       // taking the frame -> result ready
};

// Whether a depth result belongs to the camera image being shown: same
// camera session (slot generation), same source size, and the transform the
// current preprocessing would use. Anything else is never displayed.
enum class DepthMatch : std::uint8_t { kMatch, kNoDepth, kOtherCamera, kOtherSize, kOtherTransform };
DepthMatch MatchDepth(const DepthResult* result, std::uint64_t camera_generation, std::uint32_t width,
                      std::uint32_t height, ResizeMethod method);
std::string_view ToString(DepthMatch match);

// The newest depth result. Same rules as LatestFrameSlot: publishing never
// blocks, consumers keep their own last-seen sequence, and a result replaced
// before anyone took it counts as dropped.
class LatestDepthSlot {
 public:
  void Publish(std::shared_ptr<const DepthResult> result);
  std::shared_ptr<const DepthResult> TakeIfNew(std::uint64_t worker_generation, std::uint64_t last_seen_sequence);
  void Clear();
  SlotStats stats() const;

 private:
  mutable std::mutex mutex_;
  std::shared_ptr<const DepthResult> result_;
  bool taken_ = false;
  SlotStats stats_;
};

struct DepthWorkerConfig {
  EstimatorConfig estimator;
  // If a CoreML session cannot be created, run on CPU instead (the status
  // then says "CPU fallback").
  bool cpu_fallback = true;
  ResizeMethod method = kDefaultResizeMethod;
  int warmup_runs = 2;
  std::uint64_t generation = 0;  // set by InferenceController
  std::chrono::milliseconds wait_timeout{50};

  // Settings that need a new worker when they change.
  bool SameSettings(const DepthWorkerConfig& other) const;
};

using EstimatorFactory = std::function<std::unique_ptr<IDepthEstimator>(const EstimatorConfig&)>;

enum class WorkerPhase : std::uint8_t {
  kLoading,    // creating the session
  kWarmingUp,  // warm-up runs
  kRunning,    // ready; producing results when frames arrive
  kFailed,     // load, warm-up or a run failed; see `error`
  kStopped,    // stopped on request
};

std::string_view ToString(WorkerPhase phase);

struct WorkerSnapshot {
  WorkerPhase phase = WorkerPhase::kLoading;
  ExecutionProvider active_provider = ExecutionProvider::kCpu;  // what the session was created with
  bool fell_back = false;      // CoreML was requested; CPU is running
  std::string fallback_reason;
  std::string description;     // from the estimator
  std::string error;
  double load_ms = 0.0;
  std::uint64_t results = 0;          // published
  std::uint64_t invalid_frames = 0;   // camera frames with a bad layout (skipped)
  std::uint64_t invalid_ranges = 0;   // results with no usable display range
};

class DepthWorker {
 public:
  using Clock = std::chrono::steady_clock;

  // Starts the thread at once.
  DepthWorker(DepthWorkerConfig config, EstimatorFactory factory, LatestFrameSlot& frames, LatestDepthSlot& out);
  // Requests a stop and joins. This waits for a session creation in
  // progress; the app joins only workers whose finished() is true, or at
  // shutdown after the windows are gone.
  ~DepthWorker();
  DepthWorker(const DepthWorker&) = delete;
  DepthWorker& operator=(const DepthWorker&) = delete;

  // Non-blocking: sets the stop flag, interrupts a run in progress
  // (IDepthEstimator::Terminate) and wakes a frame wait.
  void RequestStop();
  // The thread has exited; Join() will return at once.
  bool finished() const { return finished_.load(std::memory_order_acquire); }
  void Join();

  const DepthWorkerConfig& config() const { return config_; }
  WorkerSnapshot snapshot() const;

  // Timings in ms, recorded by the worker; read and reset by the stats log.
  WideLatencyHistogram convert_ms;
  WideLatencyHistogram preprocess_ms;
  WideLatencyHistogram run_ms;
  WideLatencyHistogram total_ms;

 private:
  void ThreadMain();
  void Loop(IDepthEstimator& estimator);
  bool CreateEstimator(std::unique_ptr<IDepthEstimator>& estimator);
  std::shared_ptr<DepthResult> AcquireResult();
  void SetPhase(WorkerPhase phase, const std::string& error = {});

  const DepthWorkerConfig config_;
  EstimatorFactory factory_;
  LatestFrameSlot& frames_;
  LatestDepthSlot& out_;
  std::atomic<bool> stop_{false};
  std::atomic<bool> finished_{false};

  mutable std::mutex mutex_;
  WorkerSnapshot snapshot_;
  IDepthEstimator* running_estimator_ = nullptr;  // for Terminate(); guarded by mutex_

  // Recycled results. A result returns to the pool when the last reference
  // to it is released, on whichever thread that happens; it is freed instead
  // if the worker is gone by then.
  struct ResultPool;
  std::shared_ptr<ResultPool> pool_;

  std::thread thread_;  // declared last: started after, joined before the rest goes
};

}  // namespace relume
