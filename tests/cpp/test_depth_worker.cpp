// DepthWorker, LatestDepthSlot, ComputeDisplayRange and InferenceController
// with a fake estimator (no ONNX Runtime). The threaded tests are meant to be
// run under the tsan and asan presets too.

#include "relume/depth_worker.hpp"
#include "relume/inference_controller.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>

namespace relume {
namespace {

using Clock = std::chrono::steady_clock;
using std::chrono::milliseconds;

double MsSince(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Waits up to `limit` for `condition`, polling. Test helper only.
template <typename F>
bool WaitFor(F condition, milliseconds limit = milliseconds(5000)) {
  const auto end = Clock::now() + limit;
  while (Clock::now() < end) {
    if (condition()) {
      return true;
    }
    std::this_thread::sleep_for(milliseconds(1));
  }
  return condition();
}

struct FakeBehaviour {
  milliseconds load_delay{0};   // not interruptible, like ONNX Runtime session creation
  milliseconds run_delay{0};    // interruptible by Terminate, like RunOptions::SetTerminate
  bool fail_load = false;
  bool fail_coreml_load = false;
  bool throw_in_run = false;
  bool fail_run = false;
  std::function<void()> on_run;  // called at the start of every Run
};

struct FakeCounters {
  std::atomic<int> live{0};
  std::atomic<int> max_live{0};
  std::atomic<int> created{0};
  std::atomic<int> runs{0};
};

class FakeEstimator final : public IDepthEstimator {
 public:
  FakeEstimator(const EstimatorConfig& config, const FakeBehaviour& behaviour, FakeCounters& counters)
      : behaviour_(behaviour), counters_(counters), input_(kNetTensorFloats), output_(kNetDepthFloats) {
    info_.requested = config.provider;
    info_.description = config.provider == ExecutionProvider::kCoreML ? "CoreML (fake)" : "CPU (fake)";
    provider_ = config.provider;
    ++counters_.created;
    const int live = ++counters_.live;
    int seen = counters_.max_live.load();
    while (live > seen && !counters_.max_live.compare_exchange_weak(seen, live)) {
    }
  }
  ~FakeEstimator() override { --counters_.live; }

  bool Load(std::string& error) override {
    std::this_thread::sleep_for(behaviour_.load_delay);
    if (behaviour_.fail_load || (behaviour_.fail_coreml_load && provider_ == ExecutionProvider::kCoreML)) {
      error = "fake load failure";
      return false;
    }
    return true;
  }
  float* input() override { return input_.data(); }
  const float* output() const override { return output_.data(); }
  bool Run(std::string& error) override {
    ++counters_.runs;
    if (behaviour_.on_run) {
      behaviour_.on_run();
    }
    const auto end = Clock::now() + behaviour_.run_delay;
    while (Clock::now() < end) {
      if (terminated_.load()) {
        break;
      }
      std::this_thread::sleep_for(milliseconds(2));
    }
    if (terminated_.load()) {
      error = "stopped";
      return false;
    }
    if (behaviour_.throw_in_run) {
      throw std::runtime_error("fake exception in Run");
    }
    if (behaviour_.fail_run) {
      error = "the model output has a non-finite value";
      return false;
    }
    for (std::size_t i = 0; i < output_.size(); ++i) {
      output_[i] = static_cast<float>(i % kNetSize);  // a usable, non-constant map
    }
    return true;
  }
  void Terminate() override { terminated_.store(true); }
  const EstimatorInfo& info() const override { return info_; }

 private:
  FakeBehaviour behaviour_;
  FakeCounters& counters_;
  ExecutionProvider provider_ = ExecutionProvider::kCpu;
  std::vector<float> input_;
  std::vector<float> output_;
  std::atomic<bool> terminated_{false};
  EstimatorInfo info_;
};

EstimatorFactory FakeFactory(const FakeBehaviour& behaviour, FakeCounters& counters) {
  return [behaviour, &counters](const EstimatorConfig& config) -> std::unique_ptr<IDepthEstimator> {
    return std::make_unique<FakeEstimator>(config, behaviour, counters);
  };
}

DepthWorkerConfig Config(ExecutionProvider provider = ExecutionProvider::kCpu) {
  DepthWorkerConfig config;
  config.estimator.provider = provider;
  config.warmup_runs = 1;
  config.wait_timeout = milliseconds(20);
  return config;
}

std::shared_ptr<VideoFrame> RgbFrame(FramePool& pool, std::uint64_t id, std::uint32_t w = 64, std::uint32_t h = 36) {
  std::shared_ptr<VideoFrame> frame = pool.Acquire();
  if (frame) {
    AllocateFrame(*frame, PixelFormat::kRgb8, w, h);
    frame->frame_id = id;
    frame->arrival_time = Clock::now();
    std::fill(frame->data.begin(), frame->data.end(), std::byte{static_cast<unsigned char>(id)});
  }
  return frame;
}

// --- LatestFrameSlot additions ---

TEST(FrameSlotWait, WaitTakeIfNewDoesNotTakeTheFrameFromOtherConsumers) {
  FramePool pool(4);
  LatestFrameSlot slot;
  slot.Publish(RgbFrame(pool, 1));
  const auto worker = slot.WaitTakeIfNew(0, 0, milliseconds(10));
  const auto renderer = slot.TakeIfNew(0);
  ASSERT_TRUE(worker);
  ASSERT_TRUE(renderer);
  EXPECT_EQ(worker->frame_id, 1u);
  EXPECT_EQ(renderer->frame_id, 1u);
  EXPECT_FALSE(slot.WaitTakeIfNew(1, 0, milliseconds(5))) << "already seen: times out";
}

TEST(FrameSlotWait, WakesOnPublishAndOnWakeWaiters) {
  FramePool pool(4);
  LatestFrameSlot slot;
  std::atomic<std::uint64_t> got{0};
  std::thread waiter([&] {
    if (auto frame = slot.WaitTakeIfNew(0, 0, milliseconds(5000))) {
      got = frame->frame_id;
    }
  });
  std::this_thread::sleep_for(milliseconds(20));
  slot.Publish(RgbFrame(pool, 7));
  waiter.join();
  EXPECT_EQ(got.load(), 7u);

  const auto start = Clock::now();
  std::thread stopped([&] { EXPECT_FALSE(slot.WaitTakeIfNew(7, 0, milliseconds(5000))); });
  std::this_thread::sleep_for(milliseconds(20));
  slot.WakeWaiters();
  stopped.join();
  EXPECT_LT(MsSince(start), 1000.0);
}

TEST(FrameSlotWait, GenerationCountsClears) {
  FramePool pool(4);
  LatestFrameSlot slot;
  std::uint64_t generation = 99;
  slot.Publish(RgbFrame(pool, 1));
  ASSERT_TRUE(slot.TakeIfNew(0, &generation));
  EXPECT_EQ(generation, 0u);
  slot.Clear();
  slot.Publish(RgbFrame(pool, 1));  // a new source numbers from 1 again
  ASSERT_TRUE(slot.WaitTakeIfNew(1, 0, milliseconds(10), &generation)) << "same ID, new generation: new";
  EXPECT_EQ(generation, 1u);
  EXPECT_EQ(slot.generation(), 1u);
  EXPECT_FALSE(slot.WaitTakeIfNew(1, 1, milliseconds(5)));
}

// --- Display range ---

TEST(DisplayRange, IgnoresLetterboxPadding) {
  const ResizeTransform t = *ComputeTransform(1280, 720, ResizeMethod::kLetterbox);
  std::vector<float> depth(kNetDepthFloats, 1e9f);  // extreme padding values
  for (std::uint32_t y = 0; y < t.new_h; ++y) {
    for (std::uint32_t x = 0; x < t.new_w; ++x) {
      depth[std::size_t{t.pad_y + y} * kNetSize + t.pad_x + x] = static_cast<float>(x % 100);
    }
  }
  std::vector<float> scratch;
  const DisplayRange range = ComputeDisplayRange(depth.data(), t, scratch);
  ASSERT_TRUE(range.valid);
  EXPECT_NEAR(range.lo, 2.0f, 1.0f);
  EXPECT_NEAR(range.hi, 97.0f, 1.0f);
}

TEST(DisplayRange, ConstantNanAndEmptyAreRejected) {
  const ResizeTransform t = *ComputeTransform(1280, 720, ResizeMethod::kStretch);
  std::vector<float> scratch;
  std::vector<float> depth(kNetDepthFloats, 3.0f);
  EXPECT_FALSE(ComputeDisplayRange(depth.data(), t, scratch).valid) << "constant";
  depth[1234] = std::numeric_limits<float>::quiet_NaN();
  depth[4321] = 5.0f;
  EXPECT_FALSE(ComputeDisplayRange(depth.data(), t, scratch).valid) << "NaN";
  ResizeTransform empty = t;
  empty.new_w = 0;
  EXPECT_FALSE(ComputeDisplayRange(depth.data(), empty, scratch).valid) << "empty region";
}

TEST(DisplayRange, MostlyZeroMapFallsBackToFullRange) {
  // After the model's ReLU, far regions are exactly 0; when over 98% of the
  // map is 0, the percentile range collapses and the full range is used.
  const ResizeTransform t = *ComputeTransform(448, 448, ResizeMethod::kStretch);
  std::vector<float> depth(kNetDepthFloats, 0.0f);
  depth[10] = 4.0f;
  std::vector<float> scratch;
  const DisplayRange range = ComputeDisplayRange(depth.data(), t, scratch);
  ASSERT_TRUE(range.valid);
  EXPECT_EQ(range.lo, 0.0f);
  EXPECT_EQ(range.hi, 4.0f);
}

// --- DepthWorker ---

TEST(DepthWorker, NewestFrameWinsAndStaleFramesAreSkipped) {
  FakeCounters counters;
  FakeBehaviour behaviour;
  behaviour.run_delay = milliseconds(60);
  FramePool pool(4);
  LatestFrameSlot frames;
  LatestDepthSlot depth;
  DepthWorkerConfig config = Config();
  config.generation = 5;
  DepthWorker worker(config, FakeFactory(behaviour, counters), frames, depth);
  ASSERT_TRUE(WaitFor([&] { return worker.snapshot().phase == WorkerPhase::kRunning; }));

  std::vector<std::uint64_t> seen;
  std::uint64_t last_sequence = 0;
  for (std::uint64_t id = 1; id <= 40; ++id) {
    frames.Publish(RgbFrame(pool, id));
    std::this_thread::sleep_for(milliseconds(5));  // 200 fps: far faster than the worker
    if (auto result = depth.TakeIfNew(5, last_sequence)) {
      last_sequence = result->sequence;
      seen.push_back(result->frame_id);
    }
  }
  ASSERT_TRUE(WaitFor([&] {
    if (auto result = depth.TakeIfNew(5, last_sequence)) {
      last_sequence = result->sequence;
      seen.push_back(result->frame_id);
    }
    return !seen.empty() && seen.back() == 40;
  })) << "the newest frame is processed last";
  EXPECT_TRUE(std::is_sorted(seen.begin(), seen.end()));
  EXPECT_LT(worker.snapshot().results, 20u) << "stale frames are skipped, not queued";
  EXPECT_EQ(depth.TakeIfNew(4, 0), nullptr) << "another worker generation is never handed out";
}

TEST(DepthWorker, ReleasesTheCameraFrameBeforeRunningAndNeverBlocksTheProducer) {
  FakeCounters counters;
  FramePool pool(4);
  LatestFrameSlot frames;
  LatestDepthSlot depth;
  std::weak_ptr<const VideoFrame> watched;
  std::atomic<long> holders_during_run{-1};
  FakeBehaviour behaviour;
  behaviour.run_delay = milliseconds(300);
  behaviour.on_run = [&] {
    if (holders_during_run.load() == -1 && !watched.expired()) {
      holders_during_run = watched.use_count();  // only the slot should hold it
    }
  };
  DepthWorker worker(Config(), FakeFactory(behaviour, counters), frames, depth);
  ASSERT_TRUE(WaitFor([&] { return worker.snapshot().phase == WorkerPhase::kRunning; }));
  holders_during_run = -1;
  {
    std::shared_ptr<VideoFrame> frame = RgbFrame(pool, 1);
    watched = frame;
    frames.Publish(std::move(frame));
  }
  ASSERT_TRUE(WaitFor([&] { return holders_during_run.load() != -1; }));
  EXPECT_EQ(holders_during_run.load(), 1) << "the worker must not hold the camera frame during Run";

  // While Run is in progress, publishing (the camera callback) never waits.
  double max_publish_ms = 0.0;
  for (std::uint64_t id = 2; id < 30; ++id) {
    std::shared_ptr<VideoFrame> frame = RgbFrame(pool, id);
    ASSERT_TRUE(frame) << "the pool never runs out: the worker holds no frames";
    const auto start = Clock::now();
    frames.Publish(std::move(frame));
    max_publish_ms = std::max(max_publish_ms, MsSince(start));
  }
  EXPECT_LT(max_publish_ms, 5.0);
  EXPECT_LE(pool.outstanding(), 1u);
}

TEST(DepthWorker, LoadFailureIsAnError) {
  FakeCounters counters;
  FakeBehaviour behaviour;
  behaviour.fail_load = true;
  LatestFrameSlot frames;
  LatestDepthSlot depth;
  DepthWorker worker(Config(), FakeFactory(behaviour, counters), frames, depth);
  ASSERT_TRUE(WaitFor([&] { return worker.finished(); }));
  const WorkerSnapshot s = worker.snapshot();
  EXPECT_EQ(s.phase, WorkerPhase::kFailed);
  EXPECT_EQ(s.error, "fake load failure");
}

TEST(DepthWorker, ExceptionInRunIsAnErrorNotACrash) {
  FakeCounters counters;
  FakeBehaviour behaviour;
  behaviour.throw_in_run = true;
  LatestFrameSlot frames;
  LatestDepthSlot depth;
  DepthWorker worker(Config(), FakeFactory(behaviour, counters), frames, depth);
  ASSERT_TRUE(WaitFor([&] { return worker.finished(); }));
  EXPECT_EQ(worker.snapshot().phase, WorkerPhase::kFailed);
  EXPECT_NE(worker.snapshot().error.find("fake exception"), std::string::npos);
}

TEST(DepthWorker, RejectedOutputPublishesNothing) {
  FakeCounters counters;
  FakeBehaviour behaviour;
  behaviour.fail_run = true;  // e.g. a non-finite output
  FramePool pool(4);
  LatestFrameSlot frames;
  LatestDepthSlot depth;
  DepthWorkerConfig config = Config();
  config.warmup_runs = 0;
  DepthWorker worker(config, FakeFactory(behaviour, counters), frames, depth);
  frames.Publish(RgbFrame(pool, 1));
  ASSERT_TRUE(WaitFor([&] { return worker.finished(); }));
  EXPECT_EQ(worker.snapshot().phase, WorkerPhase::kFailed);
  EXPECT_EQ(depth.stats().published, 0u);
}

TEST(DepthWorker, StopDuringRunJoinsPromptly) {
  FakeCounters counters;
  FakeBehaviour behaviour;
  behaviour.run_delay = milliseconds(5000);
  FramePool pool(4);
  LatestFrameSlot frames;
  LatestDepthSlot depth;
  DepthWorkerConfig config = Config();
  config.warmup_runs = 0;
  auto worker = std::make_unique<DepthWorker>(config, FakeFactory(behaviour, counters), frames, depth);
  frames.Publish(RgbFrame(pool, 1));
  ASSERT_TRUE(WaitFor([&] { return counters.runs.load() == 1; }));
  const auto start = Clock::now();
  worker->RequestStop();
  ASSERT_TRUE(WaitFor([&] { return worker->finished(); }, milliseconds(1000)));
  worker.reset();
  EXPECT_LT(MsSince(start), 500.0);
  EXPECT_EQ(counters.live.load(), 0) << "the estimator is destroyed on the worker thread";
}

TEST(DepthWorker, StopDuringLoadWaitsForLoadThenStops) {
  FakeCounters counters;
  FakeBehaviour behaviour;
  behaviour.load_delay = milliseconds(300);  // cannot be interrupted
  LatestFrameSlot frames;
  LatestDepthSlot depth;
  DepthWorker worker(Config(), FakeFactory(behaviour, counters), frames, depth);
  std::this_thread::sleep_for(milliseconds(20));
  const auto start = Clock::now();
  worker.RequestStop();
  EXPECT_FALSE(worker.finished());
  EXPECT_LT(MsSince(start), 5.0) << "RequestStop never waits";
  ASSERT_TRUE(WaitFor([&] { return worker.finished(); }));
  EXPECT_EQ(worker.snapshot().phase, WorkerPhase::kStopped);
  EXPECT_EQ(counters.runs.load(), 0) << "no warm-up after a stop request";
}

TEST(DepthWorker, CoreMLFailureFallsBackToCpu) {
  FakeCounters counters;
  FakeBehaviour behaviour;
  behaviour.fail_coreml_load = true;
  LatestFrameSlot frames;
  LatestDepthSlot depth;
  DepthWorker worker(Config(ExecutionProvider::kCoreML), FakeFactory(behaviour, counters), frames, depth);
  ASSERT_TRUE(WaitFor([&] { return worker.snapshot().phase == WorkerPhase::kRunning; }));
  const WorkerSnapshot s = worker.snapshot();
  EXPECT_TRUE(s.fell_back);
  EXPECT_EQ(s.active_provider, ExecutionProvider::kCpu);
  EXPECT_EQ(s.fallback_reason, "fake load failure");
  EXPECT_EQ(s.description, "CPU (fake)");

  InferenceFacts facts;
  facts.requested_on = true;
  facts.requested = ExecutionProvider::kCoreML;
  facts.active = s;
  EXPECT_EQ(DeriveInferenceStatus(facts).state, InferenceState::kCpuFallback);
}

TEST(DepthWorker, ResultsRecordTheirCameraGeneration) {
  FakeCounters counters;
  FakeBehaviour behaviour;
  FramePool pool(4);
  LatestFrameSlot frames;
  LatestDepthSlot depth;
  DepthWorker worker(Config(), FakeFactory(behaviour, counters), frames, depth);
  frames.Publish(RgbFrame(pool, 1, 64, 36));
  std::shared_ptr<const DepthResult> first;
  ASSERT_TRUE(WaitFor([&] { return (first = depth.TakeIfNew(0, 0)) != nullptr; }));
  EXPECT_EQ(first->camera_generation, 0u);
  EXPECT_EQ(first->transform, *ComputeTransform(64, 36));

  // The camera changes (the slot is cleared) while results are in flight.
  frames.Clear();
  frames.Publish(RgbFrame(pool, 1, 48, 64));  // new source, numbered from 1 again
  std::shared_ptr<const DepthResult> next;
  ASSERT_TRUE(WaitFor([&] {
    next = depth.TakeIfNew(0, first->sequence);
    return next && next->camera_generation == 1u;
  }));
  EXPECT_EQ(next->transform.src_w, 48u);
  EXPECT_EQ(next->transform.src_h, 64u);
  EXPECT_TRUE(next->range.valid);
}

TEST(DepthMatch, RejectsDepthFromAnotherCameraSizeOrTransform) {
  DepthResult result;
  result.camera_generation = 3;
  result.transform = *ComputeTransform(1280, 720, ResizeMethod::kStretch);
  EXPECT_EQ(MatchDepth(nullptr, 3, 1280, 720, ResizeMethod::kStretch), DepthMatch::kNoDepth);
  EXPECT_EQ(MatchDepth(&result, 3, 1280, 720, ResizeMethod::kStretch), DepthMatch::kMatch);
  EXPECT_EQ(MatchDepth(&result, 4, 1280, 720, ResizeMethod::kStretch), DepthMatch::kOtherCamera);
  EXPECT_EQ(MatchDepth(&result, 3, 1920, 1080, ResizeMethod::kStretch), DepthMatch::kOtherSize);
  EXPECT_EQ(MatchDepth(&result, 3, 1280, 720, ResizeMethod::kLetterbox), DepthMatch::kOtherTransform);
}

// --- InferenceController ---

TEST(InferenceController, RapidChangesAreCoalescedAndNeverBlock) {
  FakeCounters counters;
  FakeBehaviour behaviour;
  behaviour.load_delay = milliseconds(150);
  behaviour.run_delay = milliseconds(30);
  FramePool pool(4);
  LatestFrameSlot frames;
  LatestDepthSlot depth;
  InferenceController controller(FakeFactory(behaviour, counters), frames, depth);
  double max_call_ms = 0.0;
  for (int i = 0; i < 20; ++i) {
    DepthWorkerConfig config = Config(i % 2 == 0 ? ExecutionProvider::kCpu : ExecutionProvider::kCoreML);
    config.estimator.intra_op_threads = i;
    const auto start = Clock::now();
    controller.Request(config);
    controller.Update();
    max_call_ms = std::max(max_call_ms, MsSince(start));
    std::this_thread::sleep_for(milliseconds(10));
  }
  EXPECT_LT(max_call_ms, 1.0) << "Request + Update never wait for a worker";
  ASSERT_TRUE(WaitFor([&] {
    const auto start = Clock::now();
    controller.Update();
    max_call_ms = std::max(max_call_ms, MsSince(start));
    return !controller.pending() && !controller.retiring() && controller.active() &&
           controller.active()->snapshot().phase == WorkerPhase::kRunning;
  }));
  EXPECT_LT(max_call_ms, 1.0);
  EXPECT_EQ(controller.active()->config().estimator.intra_op_threads, 19) << "the newest request wins";
  EXPECT_LT(controller.workers_started(), 20u) << "intermediate requests were coalesced";
  EXPECT_LE(counters.max_live.load(), 2) << "at most one active and one retiring session";
  EXPECT_EQ(controller.status().state, InferenceState::kReady);
  controller.Shutdown();
  EXPECT_EQ(controller.workers_joined(), controller.workers_started()) << "no leaked workers";
  EXPECT_EQ(counters.live.load(), 0);
}

TEST(InferenceController, TurningOffWhileLoading) {
  FakeCounters counters;
  FakeBehaviour behaviour;
  behaviour.load_delay = milliseconds(200);
  LatestFrameSlot frames;
  LatestDepthSlot depth;
  InferenceController controller(FakeFactory(behaviour, counters), frames, depth);
  controller.Request(Config());
  controller.Update();
  EXPECT_EQ(controller.status().state, InferenceState::kLoading);
  controller.Request(std::nullopt);
  const auto start = Clock::now();
  controller.Update();
  EXPECT_LT(MsSince(start), 1.0);
  EXPECT_EQ(controller.status().state, InferenceState::kOff);
  ASSERT_TRUE(WaitFor([&] {
    controller.Update();
    return !controller.retiring() && !controller.pending();
  }));
  EXPECT_EQ(controller.active(), nullptr);
  EXPECT_EQ(controller.workers_joined(), 1u);
  EXPECT_EQ(counters.live.load(), 0);
}

TEST(InferenceController, ShutdownDuringLoadJoins) {
  FakeCounters counters;
  FakeBehaviour behaviour;
  behaviour.load_delay = milliseconds(200);
  LatestFrameSlot frames;
  LatestDepthSlot depth;
  InferenceController controller(FakeFactory(behaviour, counters), frames, depth);
  controller.Request(Config());
  controller.Update();
  const double waited_ms = controller.Shutdown();
  EXPECT_GT(waited_ms, 50.0) << "the join waits out the uninterruptible load (recorded, not hidden)";
  EXPECT_EQ(counters.live.load(), 0);
  EXPECT_EQ(controller.active(), nullptr);
}

TEST(InferenceController, ErrorAndRetry) {
  FakeCounters counters;
  FakeBehaviour behaviour;
  behaviour.fail_load = true;
  LatestFrameSlot frames;
  LatestDepthSlot depth;
  InferenceController controller(FakeFactory(behaviour, counters), frames, depth);
  controller.Request(Config());
  controller.Update();
  ASSERT_TRUE(WaitFor([&] { return controller.status().state == InferenceState::kError; }));
  EXPECT_NE(controller.status().detail.find("passthrough"), std::string::npos);
  // Requesting the same settings again restarts the failed worker.
  controller.Request(Config());
  ASSERT_TRUE(WaitFor([&] {
    controller.Update();
    return controller.workers_started() == 2u;
  }));
}

TEST(InferenceStatus, Labels) {
  InferenceFacts facts;
  EXPECT_EQ(DeriveInferenceStatus(facts).label, "Off");
  facts.requested_on = true;
  EXPECT_EQ(DeriveInferenceStatus(facts).state, InferenceState::kLoading);
  WorkerSnapshot worker;
  worker.phase = WorkerPhase::kRunning;
  worker.description = "CPU, 4 threads";
  facts.active = worker;
  EXPECT_EQ(DeriveInferenceStatus(facts).label, "Ready (CPU, 4 threads)");
  facts.switching = true;
  EXPECT_EQ(DeriveInferenceStatus(facts).state, InferenceState::kLoading);
}

}  // namespace
}  // namespace relume
