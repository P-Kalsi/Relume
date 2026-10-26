#pragma once

// Depth estimation behind a portable interface. The ONNX Runtime
// implementation lives in src/inference/onnx_depth.cpp; tests use fakes.
//
// An estimator owns its input and output buffers: the caller writes the
// preprocessed 1x3x448x448 tensor into input(), calls Run(), and reads the
// 1x448x448 depth from output(). The ONNX Runtime tensors are created once
// over these buffers, so Relume allocates nothing per run (ONNX Runtime's own
// internal allocations are not claimed to be zero; see docs/architecture.md).
//
// Not thread-safe, except Terminate(): one thread loads and runs an
// estimator, and destroys it.

#include "relume/preprocess.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace relume {

enum class ExecutionProvider : std::uint8_t { kCpu, kCoreML };
// CoreML MLComputeUnits.
enum class CoreMLUnits : std::uint8_t { kAll, kCPUOnly, kCPUAndGPU, kCPUAndNeuralEngine };

std::string_view ToString(ExecutionProvider provider);
std::string_view ToString(CoreMLUnits units);
std::optional<ExecutionProvider> ParseExecutionProvider(std::string_view name);
std::optional<CoreMLUnits> ParseCoreMLUnits(std::string_view name);

struct EstimatorConfig {
  std::string model_path;
  ExecutionProvider provider = ExecutionProvider::kCpu;
  CoreMLUnits coreml_units = CoreMLUnits::kAll;
  // Diagnostic only: fail session creation unless CoreML takes every node
  // (session.disable_cpu_ep_fallback). Never the deployment configuration.
  bool coreml_strict = false;
  std::string coreml_cache_dir;       // compiled-model cache; empty = none
  bool coreml_profile_compute_plan = false;  // log CoreML's planned device per op
  int intra_op_threads = 0;           // 0 = ONNX Runtime's default
  std::string profile_prefix;         // ONNX Runtime profiling JSON prefix; empty = off
  bool verbose_log = false;           // ONNX Runtime verbose log to stderr (diagnostics)
};

struct TensorSignature {
  std::string name;
  std::string type;  // "float" for the expected model
  std::vector<std::int64_t> shape;
};

// What was requested and what the session reports. "Registered" or listed is
// never proof that a provider executed anything (see docs/model.md).
struct EstimatorInfo {
  std::string ort_version;
  ExecutionProvider requested = ExecutionProvider::kCpu;
  std::string description;  // e.g. "CPU, 4 threads" or "CoreML (CPUAndNeuralEngine)"
  TensorSignature input;
  TensorSignature output;
  double load_ms = 0.0;  // session creation
};

class IDepthEstimator {
 public:
  virtual ~IDepthEstimator() = default;

  // Creates the session (may take seconds) and checks the model's signature
  // against the expected tensors. Returns false and sets `error` on failure.
  virtual bool Load(std::string& error) = 0;
  // kNetTensorFloats floats; valid after a successful Load.
  virtual float* input() = 0;
  // kNetDepthFloats floats; valid after a successful Run.
  virtual const float* output() const = 0;
  // Runs the model on input(). Fails (never throws) on an ONNX Runtime error,
  // after Terminate(), or if the output has a non-finite value.
  virtual bool Run(std::string& error) = 0;
  // Thread-safe: makes a Run in progress (and later ones) stop early.
  virtual void Terminate() = 0;
  virtual const EstimatorInfo& info() const = 0;
  // Ends ONNX Runtime profiling (if enabled) and returns the profile file;
  // empty if profiling was off. Profiling also ends on destruction.
  virtual std::string EndProfiling() { return {}; }
};

// The model's expected signature (checked by Load, and against the model
// manifest by scripts/check_native_parity.py).
inline constexpr std::string_view kModelInputName = "image";
inline constexpr std::string_view kModelOutputName = "depth";

std::unique_ptr<IDepthEstimator> MakeOnnxDepthEstimator(EstimatorConfig config);

}  // namespace relume
