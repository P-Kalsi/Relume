#pragma once

// Owns the depth workers and decides when they run. Main thread only; no
// call ever waits for a worker, except Shutdown().
//
// Changing the configuration (provider, compute units, threads...) or turning
// inference off asks the current worker to stop and moves it to the retiring
// slot. A retiring worker is joined only once it has finished, which the
// main thread checks each frame, so the join returns at once. Session
// creation cannot be interrupted inside ONNX Runtime: a retiring worker that
// is still loading finishes loading, destroys its session on its own thread
// and exits.
//
// Rapid changes are coalesced: there is at most one active and one retiring
// worker, and only the newest requested configuration is kept as pending. A
// pending configuration starts once nothing is retiring, so at most two
// sessions exist at any time.
//
// Results carry their worker's generation; the depth slot only hands out
// results of the active worker, so a stopped worker's depth is never shown.

#include "relume/depth_worker.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace relume {

enum class InferenceState : std::uint8_t {
  kOff,          // turned off
  kLoading,      // creating or warming up a session (or switching)
  kReady,        // running with the requested provider
  kCpuFallback,  // CoreML was requested but could not be used; running on CPU
  kError,        // no session; the output stays passthrough
};

struct InferenceStatus {
  InferenceState state = InferenceState::kOff;
  std::string label;   // "Off", "Loading", "Ready (CPU, 4 threads)", "CPU fallback", "Error"
  std::string detail;  // a sentence for the Controls window
};

// What the controller knows, from which the status is derived.
struct InferenceFacts {
  bool requested_on = false;
  bool switching = false;  // a configuration is pending or a worker is retiring
  ExecutionProvider requested = ExecutionProvider::kCpu;
  std::optional<WorkerSnapshot> active;
};

InferenceStatus DeriveInferenceStatus(const InferenceFacts& facts);

class InferenceController {
 public:
  InferenceController(EstimatorFactory factory, LatestFrameSlot& frames, LatestDepthSlot& depth);
  ~InferenceController();  // Shutdown()
  InferenceController(const InferenceController&) = delete;
  InferenceController& operator=(const InferenceController&) = delete;

  // Requests a configuration, or off (nullopt). Never blocks; applied by Update().
  void Request(std::optional<DepthWorkerConfig> config);
  // Joins finished retiring workers, retires the active one if a different
  // configuration is pending, and starts the pending one when possible.
  // Never blocks.
  void Update();
  // Stops every worker and joins them (this can wait for a session creation
  // in progress). Returns the time spent waiting, in ms.
  double Shutdown();

  const DepthWorker* active() const { return active_.get(); }
  DepthWorker* active() { return active_.get(); }
  // The generation of the active worker; 0 when none.
  std::uint64_t active_generation() const { return active_ ? active_->config().generation : 0; }
  bool retiring() const { return retiring_ != nullptr; }
  bool pending() const { return pending_.has_value(); }
  InferenceStatus status() const;

  std::uint64_t workers_started() const { return started_; }
  std::uint64_t workers_joined() const { return joined_; }

 private:
  void Retire();

  EstimatorFactory factory_;
  LatestFrameSlot& frames_;
  LatestDepthSlot& depth_;
  // The latest request: nullopt = nothing pending; a pending nullopt config
  // means "off".
  std::optional<std::optional<DepthWorkerConfig>> pending_;
  std::optional<DepthWorkerConfig> wanted_;  // the configuration that should be running
  std::uint64_t next_generation_ = 1;
  std::uint64_t started_ = 0;
  std::uint64_t joined_ = 0;
  std::unique_ptr<DepthWorker> retiring_;
  std::unique_ptr<DepthWorker> active_;  // declared last: destroyed first
};

}  // namespace relume
