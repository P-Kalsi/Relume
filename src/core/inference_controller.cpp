#include "relume/inference_controller.hpp"

#include <chrono>

namespace relume {

InferenceStatus DeriveInferenceStatus(const InferenceFacts& facts) {
  InferenceStatus status;
  if (!facts.requested_on) {
    status.state = InferenceState::kOff;
    status.label = "Off";
    status.detail = "Depth inference is off; the output is camera passthrough.";
    return status;
  }
  if (!facts.active || facts.switching) {
    status.state = InferenceState::kLoading;
    status.label = "Loading";
    status.detail = facts.switching ? "Switching depth settings; the previous session is being released."
                                    : "Starting depth inference.";
    return status;
  }
  const WorkerSnapshot& worker = *facts.active;
  switch (worker.phase) {
    case WorkerPhase::kLoading:
      status.state = InferenceState::kLoading;
      status.label = "Loading";
      status.detail = facts.requested == ExecutionProvider::kCoreML
                          ? "Creating the session; CoreML may compile the model first, which can take a while."
                          : "Creating the ONNX Runtime session.";
      break;
    case WorkerPhase::kWarmingUp:
      status.state = InferenceState::kLoading;
      status.label = "Loading";
      status.detail = "Warming up (" + worker.description + ").";
      break;
    case WorkerPhase::kRunning:
      if (worker.fell_back) {
        status.state = InferenceState::kCpuFallback;
        status.label = "CPU fallback";
        status.detail = "CoreML could not be used (" + worker.fallback_reason + "); running on " +
                        worker.description + ".";
      } else {
        status.state = InferenceState::kReady;
        status.label = "Ready (" + worker.description + ")";
        status.detail = "Depth inference is running.";
      }
      break;
    case WorkerPhase::kFailed:
      status.state = InferenceState::kError;
      status.label = "Error";
      status.detail = worker.error + " The output stays camera passthrough.";
      break;
    case WorkerPhase::kStopped:
      status.state = InferenceState::kOff;
      status.label = "Off";
      status.detail = "Depth inference stopped.";
      break;
  }
  return status;
}

InferenceController::InferenceController(EstimatorFactory factory, LatestFrameSlot& frames, LatestDepthSlot& depth)
    : factory_(std::move(factory)), frames_(frames), depth_(depth) {}

InferenceController::~InferenceController() { Shutdown(); }

void InferenceController::Request(std::optional<DepthWorkerConfig> config) { pending_ = std::move(config); }

void InferenceController::Retire() {
  active_->RequestStop();  // does not wait
  retiring_ = std::move(active_);
}

void InferenceController::Update() {
  if (retiring_ && retiring_->finished()) {
    retiring_->Join();  // the thread has exited: returns at once
    retiring_.reset();
    ++joined_;
  }
  if (!pending_) {
    return;
  }
  const std::optional<DepthWorkerConfig>& want = *pending_;
  if (active_ && want && active_->config().SameSettings(*want) &&
      active_->snapshot().phase != WorkerPhase::kFailed) {
    pending_.reset();  // already running (a failed worker is restarted instead)
    return;
  }
  if (active_) {
    if (retiring_) {
      return;  // at most one retiring worker: wait for it first
    }
    Retire();
  }
  if (retiring_) {
    return;  // the pending configuration starts once nothing is retiring
  }
  wanted_ = want;
  pending_.reset();
  if (wanted_) {
    DepthWorkerConfig config = *wanted_;
    config.generation = next_generation_++;
    depth_.Clear();
    active_ = std::make_unique<DepthWorker>(std::move(config), factory_, frames_, depth_);
    ++started_;
  }
}

double InferenceController::Shutdown() {
  const auto start = std::chrono::steady_clock::now();
  for (DepthWorker* worker : {active_.get(), retiring_.get()}) {
    if (worker != nullptr) {
      worker->RequestStop();
    }
  }
  for (std::unique_ptr<DepthWorker>* worker : {&active_, &retiring_}) {
    if (*worker) {
      (*worker)->Join();
      worker->reset();
      ++joined_;
    }
  }
  pending_.reset();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

InferenceStatus InferenceController::status() const {
  InferenceFacts facts;
  const std::optional<DepthWorkerConfig>& wanted = pending_ ? *pending_ : wanted_;
  facts.requested_on = wanted.has_value();
  facts.requested = wanted ? wanted->estimator.provider : ExecutionProvider::kCpu;
  facts.switching = pending_.has_value() || retiring_ != nullptr;
  if (active_) {
    facts.active = active_->snapshot();
  }
  return DeriveInferenceStatus(facts);
}

}  // namespace relume
