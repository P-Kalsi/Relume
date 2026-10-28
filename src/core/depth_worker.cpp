#include "relume/depth_worker.hpp"

#include "relume/pixel_convert.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>

namespace relume {

namespace {

constexpr std::size_t kResultCapacity = 4;  // worker + slot + renderer + one spare

double Ms(DepthWorker::Clock::time_point a, DepthWorker::Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

bool UsableRange(float lo, float hi) { return hi - lo > 1e-6f * std::max(1.0f, std::abs(hi)); }

}  // namespace

DisplayRange ComputeDisplayRange(const float* depth, const ResizeTransform& transform, std::vector<float>& scratch) {
  DisplayRange range;
  const std::size_t count = std::size_t{transform.new_w} * transform.new_h;
  if (count == 0 || transform.pad_x + transform.new_w > kNetSize || transform.pad_y + transform.new_h > kNetSize) {
    return range;
  }
  scratch.resize(count);
  for (std::uint32_t y = 0; y < transform.new_h; ++y) {
    const float* row = depth + std::size_t{transform.pad_y + y} * kNetSize + transform.pad_x;
    std::copy(row, row + transform.new_w, scratch.begin() + static_cast<std::ptrdiff_t>(std::size_t{y} * transform.new_w));
  }
  if (!std::all_of(scratch.begin(), scratch.end(), [](float v) { return std::isfinite(v); })) {
    return range;
  }
  const auto at = [&](double p) {
    const auto index = static_cast<std::size_t>(std::lround(p * static_cast<double>(count - 1)));
    std::nth_element(scratch.begin(), scratch.begin() + static_cast<std::ptrdiff_t>(index), scratch.end());
    return scratch[index];
  };
  float lo = at(0.02);
  float hi = at(0.98);
  if (!UsableRange(lo, hi)) {
    // Mostly flat (e.g. large exact-zero areas after the model's ReLU): fall
    // back to the full range before giving up.
    const auto [min_it, max_it] = std::minmax_element(scratch.begin(), scratch.end());
    lo = *min_it;
    hi = *max_it;
  }
  if (UsableRange(lo, hi)) {
    range.lo = lo;
    range.hi = hi;
    range.valid = true;
  }
  return range;
}

DepthMatch MatchDepth(const DepthResult* result, std::uint64_t camera_generation, std::uint32_t width,
                      std::uint32_t height, ResizeMethod method) {
  if (result == nullptr) {
    return DepthMatch::kNoDepth;
  }
  if (result->camera_generation != camera_generation) {
    return DepthMatch::kOtherCamera;
  }
  if (result->transform.src_w != width || result->transform.src_h != height) {
    return DepthMatch::kOtherSize;
  }
  const std::optional<ResizeTransform> expected = ComputeTransform(width, height, method);
  if (!expected || *expected != result->transform) {
    return DepthMatch::kOtherTransform;
  }
  return DepthMatch::kMatch;
}

std::string_view ToString(DepthMatch match) {
  switch (match) {
    case DepthMatch::kMatch: return "match";
    case DepthMatch::kNoDepth: return "no depth";
    case DepthMatch::kNoImage: return "no camera image";
    case DepthMatch::kOtherCamera: return "other camera";
    case DepthMatch::kOtherSize: return "other size";
    case DepthMatch::kOtherTransform: return "other transform";
  }
  return "?";
}

void LatestDepthSlot::Publish(std::shared_ptr<const DepthResult> result) {
  {
    std::lock_guard lock(mutex_);
    if (result_ && !taken_) {
      ++stats_.dropped_unread;
    }
    ++stats_.published;
    std::swap(result_, result);
    taken_ = false;
  }
  // `result` holds the replaced one and is released outside the lock.
}

std::shared_ptr<const DepthResult> LatestDepthSlot::TakeIfNew(std::uint64_t worker_generation,
                                                              std::uint64_t last_seen_sequence) {
  std::lock_guard lock(mutex_);
  if (!result_ || result_->worker_generation != worker_generation || result_->sequence == last_seen_sequence) {
    return nullptr;
  }
  taken_ = true;
  return result_;
}

void LatestDepthSlot::Clear() {
  std::shared_ptr<const DepthResult> released;
  std::lock_guard lock(mutex_);
  std::swap(result_, released);
  taken_ = false;
}

SlotStats LatestDepthSlot::stats() const {
  std::lock_guard lock(mutex_);
  return stats_;
}

bool DepthWorkerConfig::SameSettings(const DepthWorkerConfig& other) const {
  const EstimatorConfig& a = estimator;
  const EstimatorConfig& b = other.estimator;
  return a.model_path == b.model_path && a.provider == b.provider && a.coreml_units == b.coreml_units &&
         a.coreml_strict == b.coreml_strict && a.coreml_cache_dir == b.coreml_cache_dir &&
         a.coreml_profile_compute_plan == b.coreml_profile_compute_plan &&
         a.intra_op_threads == b.intra_op_threads && a.profile_prefix == b.profile_prefix &&
         a.verbose_log == b.verbose_log && cpu_fallback == other.cpu_fallback && method == other.method &&
         warmup_runs == other.warmup_runs;
}

std::string_view ToString(WorkerPhase phase) {
  switch (phase) {
    case WorkerPhase::kLoading: return "loading";
    case WorkerPhase::kWarmingUp: return "warming up";
    case WorkerPhase::kRunning: return "running";
    case WorkerPhase::kFailed: return "failed";
    case WorkerPhase::kStopped: return "stopped";
  }
  return "?";
}

// A small set of DepthResult buffers, recycled like FramePool's frames.
struct DepthWorker::ResultPool {
  ResultPool() { free.reserve(kResultCapacity); }  // returning never allocates
  std::mutex mutex;
  std::vector<std::unique_ptr<DepthResult>> free;
  std::size_t outstanding = 0;
  std::uint64_t exhausted = 0;
};

DepthWorker::DepthWorker(DepthWorkerConfig config, EstimatorFactory factory, LatestFrameSlot& frames,
                         LatestDepthSlot& out)
    : config_(std::move(config)), factory_(std::move(factory)), frames_(frames), out_(out),
      pool_(std::make_shared<ResultPool>()) {
  snapshot_.active_provider = config_.estimator.provider;
  thread_ = std::thread([this] { ThreadMain(); });
}

DepthWorker::~DepthWorker() {
  RequestStop();
  Join();
}

void DepthWorker::RequestStop() {
  stop_.store(true, std::memory_order_release);
  {
    std::lock_guard lock(mutex_);
    if (running_estimator_ != nullptr) {
      running_estimator_->Terminate();  // thread-safe by contract; does not wait
    }
  }
  frames_.WakeWaiters();
}

void DepthWorker::Join() {
  if (thread_.joinable()) {
    thread_.join();
  }
}

WorkerSnapshot DepthWorker::snapshot() const {
  std::lock_guard lock(mutex_);
  return snapshot_;
}

void DepthWorker::SetPhase(WorkerPhase phase, const std::string& error) {
  std::lock_guard lock(mutex_);
  snapshot_.phase = phase;
  if (!error.empty()) {
    snapshot_.error = error;
  }
}

std::shared_ptr<DepthResult> DepthWorker::AcquireResult() {
  std::unique_ptr<DepthResult> result;
  {
    std::lock_guard lock(pool_->mutex);
    if (!pool_->free.empty()) {
      result = std::move(pool_->free.back());
      pool_->free.pop_back();
    } else if (pool_->outstanding >= kResultCapacity) {
      ++pool_->exhausted;
      return nullptr;
    }
    ++pool_->outstanding;
  }
  if (!result) {
    result = std::make_unique<DepthResult>();  // grows lazily up to capacity
  }
  auto give_back = [pool = std::weak_ptr<ResultPool>(pool_)](DepthResult* raw) noexcept {
    std::unique_ptr<DepthResult> owned(raw);
    const std::shared_ptr<ResultPool> shared = pool.lock();
    if (!shared) {
      return;  // the worker is gone: `owned` frees the result
    }
    std::lock_guard lock(shared->mutex);
    shared->free.push_back(std::move(owned));
    --shared->outstanding;
  };
  return std::shared_ptr<DepthResult>(result.release(), std::move(give_back));
}

void DepthWorker::ThreadMain() {
  try {
    std::unique_ptr<IDepthEstimator> estimator;
    // Declared after `estimator`, so it is destroyed first: RequestStop never
    // sees a destroyed estimator, even during stack unwinding.
    struct Unregister {
      DepthWorker& worker;
      ~Unregister() {
        std::lock_guard lock(worker.mutex_);
        worker.running_estimator_ = nullptr;
      }
    } unregister{*this};
    if (CreateEstimator(estimator)) {
      Loop(*estimator);
    }
  } catch (const std::exception& e) {
    SetPhase(WorkerPhase::kFailed, std::string("unexpected error: ") + e.what());
  } catch (...) {
    SetPhase(WorkerPhase::kFailed, "unexpected error");
  }
  {
    std::lock_guard lock(mutex_);
    if (snapshot_.phase != WorkerPhase::kFailed) {
      snapshot_.phase = WorkerPhase::kStopped;
    }
  }
  finished_.store(true, std::memory_order_release);
}

bool DepthWorker::CreateEstimator(std::unique_ptr<IDepthEstimator>& estimator) {
  SetPhase(WorkerPhase::kLoading);
  const auto load = [&](const EstimatorConfig& config, std::string& error) -> std::unique_ptr<IDepthEstimator> {
    std::unique_ptr<IDepthEstimator> candidate = factory_(config);
    if (!candidate) {
      error = "no depth estimator is available";
      return nullptr;
    }
    {
      std::lock_guard lock(mutex_);
      running_estimator_ = candidate.get();
    }
    const bool ok = candidate->Load(error);
    std::lock_guard lock(mutex_);
    running_estimator_ = ok ? candidate.get() : nullptr;
    return ok ? std::move(candidate) : nullptr;
  };

  std::string error;
  if (!config_.estimator.coreml_cache_dir.empty()) {
    std::error_code ec;  // a missing cache only costs a recompile
    std::filesystem::create_directories(config_.estimator.coreml_cache_dir, ec);
  }
  estimator = load(config_.estimator, error);
  if (!estimator && !stop_.load(std::memory_order_acquire) && config_.cpu_fallback &&
      config_.estimator.provider == ExecutionProvider::kCoreML) {
    {
      std::lock_guard lock(mutex_);
      snapshot_.fell_back = true;
      snapshot_.fallback_reason = error;
      snapshot_.active_provider = ExecutionProvider::kCpu;
    }
    EstimatorConfig cpu = config_.estimator;
    cpu.provider = ExecutionProvider::kCpu;
    error.clear();
    estimator = load(cpu, error);
  }
  if (stop_.load(std::memory_order_acquire)) {
    return false;
  }
  if (!estimator) {
    SetPhase(WorkerPhase::kFailed, error);
    return false;
  }
  {
    std::lock_guard lock(mutex_);
    snapshot_.description = estimator->info().description;
    snapshot_.load_ms = estimator->info().load_ms;
    snapshot_.phase = WorkerPhase::kWarmingUp;
  }
  // Warm-up on a neutral tensor (the ImageNet mean, i.e. zeros).
  std::fill(estimator->input(), estimator->input() + kNetTensorFloats, 0.0f);
  for (int i = 0; i < config_.warmup_runs && !stop_.load(std::memory_order_acquire); ++i) {
    if (!estimator->Run(error)) {
      if (!stop_.load(std::memory_order_acquire)) {
        SetPhase(WorkerPhase::kFailed, "warm-up run failed: " + error);
      }
      return false;
    }
  }
  if (stop_.load(std::memory_order_acquire)) {
    return false;
  }
  SetPhase(WorkerPhase::kRunning);
  return true;
}

void DepthWorker::Loop(IDepthEstimator& estimator) {
  Preprocessor preprocessor;
  std::vector<std::uint8_t> rgb;
  std::vector<float> range_scratch;
  std::uint64_t last_frame_id = 0;
  std::uint64_t last_generation = 0;
  std::uint64_t sequence = 0;
  while (!stop_.load(std::memory_order_acquire)) {
    std::uint64_t camera_generation = 0;
    std::shared_ptr<const VideoFrame> frame =
        frames_.WaitTakeIfNew(last_frame_id, last_generation, config_.wait_timeout, &camera_generation);
    if (!frame) {
      continue;
    }
    const Clock::time_point t0 = Clock::now();
    last_frame_id = frame->frame_id;
    last_generation = camera_generation;
    const std::uint64_t frame_id = frame->frame_id;
    const std::int64_t capture_time_ns = frame->capture_time_ns;
    const Clock::time_point arrival = frame->arrival_time;
    const std::uint32_t width = frame->width;
    const std::uint32_t height = frame->height;
    const bool converted = ToRgb8(*frame, rgb);
    frame.reset();  // back to the camera's pool before the (long) model run
    if (!converted) {
      std::lock_guard lock(mutex_);
      ++snapshot_.invalid_frames;
      continue;
    }
    const Clock::time_point t1 = Clock::now();
    const std::optional<ResizeTransform> transform =
        preprocessor.Run(rgb.data(), width, height, config_.method, estimator.input());
    if (!transform) {
      std::lock_guard lock(mutex_);
      ++snapshot_.invalid_frames;
      continue;
    }
    const Clock::time_point t2 = Clock::now();
    std::string error;
    if (!estimator.Run(error)) {
      if (!stop_.load(std::memory_order_acquire)) {
        SetPhase(WorkerPhase::kFailed, error);
      }
      return;
    }
    const Clock::time_point t3 = Clock::now();

    std::shared_ptr<DepthResult> result = AcquireResult();
    if (!result) {
      continue;  // every buffer is held by consumers; this result is skipped
    }
    result->sequence = ++sequence;
    result->worker_generation = config_.generation;
    result->camera_generation = camera_generation;
    result->frame_id = frame_id;
    result->capture_time_ns = capture_time_ns;
    result->arrival_time = arrival;
    result->transform = *transform;
    result->depth.assign(estimator.output(), estimator.output() + kNetDepthFloats);  // reuses capacity
    result->range = ComputeDisplayRange(result->depth.data(), *transform, range_scratch);
    const Clock::time_point t4 = Clock::now();
    result->finished_time = t4;
    result->convert_ms = Ms(t0, t1);
    result->preprocess_ms = Ms(t1, t2);
    result->run_ms = Ms(t2, t3);
    result->total_ms = Ms(t0, t4);
    convert_ms.Record(result->convert_ms);
    preprocess_ms.Record(result->preprocess_ms);
    run_ms.Record(result->run_ms);
    total_ms.Record(result->total_ms);
    {
      std::lock_guard lock(mutex_);
      ++snapshot_.results;
      snapshot_.invalid_ranges += result->range.valid ? 0 : 1;
    }
    out_.Publish(std::move(result));
  }
}

}  // namespace relume
