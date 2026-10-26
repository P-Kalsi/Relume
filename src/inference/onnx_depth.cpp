// ONNX Runtime implementation of IDepthEstimator. Portable C++: the only
// dependency is the ONNX Runtime C++ API. Every Ort:: object is owned here
// (RAII) and used only on the thread that calls Load/Run.

#include "relume/depth_estimator.hpp"

#include <onnxruntime_cxx_api.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <unordered_map>

namespace relume {

std::string_view ToString(ExecutionProvider provider) {
  switch (provider) {
    case ExecutionProvider::kCpu: return "cpu";
    case ExecutionProvider::kCoreML: return "coreml";
  }
  return "?";
}

std::string_view ToString(CoreMLUnits units) {
  switch (units) {
    case CoreMLUnits::kAll: return "ALL";
    case CoreMLUnits::kCPUOnly: return "CPUOnly";
    case CoreMLUnits::kCPUAndGPU: return "CPUAndGPU";
    case CoreMLUnits::kCPUAndNeuralEngine: return "CPUAndNeuralEngine";
  }
  return "?";
}

std::optional<ExecutionProvider> ParseExecutionProvider(std::string_view name) {
  if (name == "cpu") {
    return ExecutionProvider::kCpu;
  }
  if (name == "coreml") {
    return ExecutionProvider::kCoreML;
  }
  return std::nullopt;
}

std::optional<CoreMLUnits> ParseCoreMLUnits(std::string_view name) {
  for (const CoreMLUnits units :
       {CoreMLUnits::kAll, CoreMLUnits::kCPUOnly, CoreMLUnits::kCPUAndGPU, CoreMLUnits::kCPUAndNeuralEngine}) {
    if (name == ToString(units)) {
      return units;
    }
  }
  return std::nullopt;
}

namespace {

std::string ElementTypeName(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return "double";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return "uint8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
    default: return "type " + std::to_string(static_cast<int>(type));
  }
}

std::string ShapeText(const std::vector<std::int64_t>& shape) {
  std::string text = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    text += (i ? "," : "") + std::to_string(shape[i]);
  }
  return text + "]";
}

std::string SignatureText(const TensorSignature& s) { return s.name + " " + s.type + " " + ShapeText(s.shape); }

class OnnxDepthEstimator final : public IDepthEstimator {
 public:
  explicit OnnxDepthEstimator(EstimatorConfig config) : config_(std::move(config)) {
    info_.ort_version = Ort::GetVersionString();
    info_.requested = config_.provider;
    input_.resize(kNetTensorFloats);
    output_.resize(kNetDepthFloats);
  }

  ~OnnxDepthEstimator() override { EndProfiling(); }

  std::string EndProfiling() override {
    if (!session_ || config_.profile_prefix.empty() || profiling_ended_) {
      return {};
    }
    profiling_ended_ = true;
    // Never let an exception escape (this also runs in the destructor).
    try {
      Ort::AllocatorWithDefaultOptions allocator;
      return session_->EndProfilingAllocated(allocator).get();
    } catch (...) {
      return {};
    }
  }

  bool Load(std::string& error) override {
    try {
      return LoadImpl(error);
    } catch (const Ort::Exception& e) {
      error = std::string("ONNX Runtime: ") + e.what();
    } catch (const std::exception& e) {
      error = e.what();
    }
    session_.reset();
    return false;
  }

  float* input() override { return input_.data(); }
  const float* output() const override { return output_.data(); }

  bool Run(std::string& error) override {
    if (!session_) {
      error = "the model is not loaded";
      return false;
    }
    if (terminated_.load(std::memory_order_acquire)) {
      error = "stopped";
      return false;
    }
    try {
      const char* input_names[] = {input_name_.c_str()};
      const char* output_names[] = {output_name_.c_str()};
      session_->Run(run_options_, input_names, &input_value_, 1, output_names, &output_value_, 1);
    } catch (const Ort::Exception& e) {
      error = terminated_.load(std::memory_order_acquire) ? "stopped" : std::string("ONNX Runtime: ") + e.what();
      return false;
    } catch (const std::exception& e) {
      error = e.what();
      return false;
    }
    for (const float v : output_) {
      if (!std::isfinite(v)) {
        error = "the model output has a non-finite value";
        return false;
      }
    }
    return true;
  }

  void Terminate() override {
    terminated_.store(true, std::memory_order_release);
    // RunOptionsSetTerminate is safe to call from another thread while Run
    // is in progress.
    try {
      run_options_.SetTerminate();
    } catch (...) {
    }
  }

  const EstimatorInfo& info() const override { return info_; }

 private:
  bool LoadImpl(std::string& error) {
    const auto start = std::chrono::steady_clock::now();
    env_.emplace(config_.verbose_log ? ORT_LOGGING_LEVEL_VERBOSE : ORT_LOGGING_LEVEL_WARNING, "relume");

    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (config_.intra_op_threads > 0) {
      options.SetIntraOpNumThreads(config_.intra_op_threads);
    }
    if (config_.verbose_log) {
      options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_VERBOSE);
    }
    if (!config_.profile_prefix.empty()) {
      options.EnableProfiling(config_.profile_prefix.c_str());
    }

    std::string description;
    if (config_.provider == ExecutionProvider::kCoreML) {
      std::unordered_map<std::string, std::string> coreml = {
          {"ModelFormat", "MLProgram"},
          {"MLComputeUnits", std::string(ToString(config_.coreml_units))},
          {"RequireStaticInputShapes", "1"},
      };
      if (!config_.coreml_cache_dir.empty()) {
        coreml["ModelCacheDirectory"] = config_.coreml_cache_dir;
      }
      if (config_.coreml_profile_compute_plan) {
        coreml["ProfileComputePlan"] = "1";
      }
      options.AppendExecutionProvider("CoreML", coreml);
      if (config_.coreml_strict) {
        options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
      }
      description = "CoreML (" + std::string(ToString(config_.coreml_units)) + ")" +
                    (config_.coreml_strict ? ", strict" : ", CPU EP for unsupported nodes");
    } else {
      description = "CPU";
    }
    description += config_.intra_op_threads > 0 ? ", " + std::to_string(config_.intra_op_threads) + " threads"
                                                 : ", default threads";

    session_.emplace(*env_, config_.model_path.c_str(), options);

    if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1) {
      error = "expected 1 input and 1 output, the model has " + std::to_string(session_->GetInputCount()) + " and " +
              std::to_string(session_->GetOutputCount());
      return false;
    }
    Ort::AllocatorWithDefaultOptions allocator;
    info_.input.name = session_->GetInputNameAllocated(0, allocator).get();
    info_.output.name = session_->GetOutputNameAllocated(0, allocator).get();
    {
      const Ort::TypeInfo type = session_->GetInputTypeInfo(0);
      const auto tensor = type.GetTensorTypeAndShapeInfo();
      info_.input.type = ElementTypeName(tensor.GetElementType());
      info_.input.shape = tensor.GetShape();
    }
    {
      const Ort::TypeInfo type = session_->GetOutputTypeInfo(0);
      const auto tensor = type.GetTensorTypeAndShapeInfo();
      info_.output.type = ElementTypeName(tensor.GetElementType());
      info_.output.shape = tensor.GetShape();
    }
    const TensorSignature expected_in{std::string(kModelInputName), "float", {1, 3, kNetSize, kNetSize}};
    const TensorSignature expected_out{std::string(kModelOutputName), "float", {1, kNetSize, kNetSize}};
    if (info_.input.name != expected_in.name || info_.input.type != expected_in.type ||
        info_.input.shape != expected_in.shape || info_.output.name != expected_out.name ||
        info_.output.type != expected_out.type || info_.output.shape != expected_out.shape) {
      error = "unexpected model signature: " + SignatureText(info_.input) + " -> " + SignatureText(info_.output) +
              " (expected " + SignatureText(expected_in) + " -> " + SignatureText(expected_out) + ")";
      return false;
    }
    input_name_ = info_.input.name;
    output_name_ = info_.output.name;

    // Tensors over our own buffers, created once.
    const Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    input_value_ = Ort::Value::CreateTensor<float>(memory, input_.data(), input_.size(), info_.input.shape.data(),
                                                   info_.input.shape.size());
    output_value_ = Ort::Value::CreateTensor<float>(memory, output_.data(), output_.size(),
                                                    info_.output.shape.data(), info_.output.shape.size());

    info_.description = description;
    info_.load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return true;
  }

  const EstimatorConfig config_;
  EstimatorInfo info_;
  std::vector<float> input_;
  std::vector<float> output_;
  std::string input_name_;
  std::string output_name_;
  std::atomic<bool> terminated_{false};
  bool profiling_ended_ = false;

  // Declared in dependency order, destroyed in reverse: values and the
  // session before the environment.
  std::optional<Ort::Env> env_;
  Ort::RunOptions run_options_;
  std::optional<Ort::Session> session_;
  Ort::Value input_value_{nullptr};
  Ort::Value output_value_{nullptr};
};

}  // namespace

std::unique_ptr<IDepthEstimator> MakeOnnxDepthEstimator(EstimatorConfig config) {
  return std::make_unique<OnnxDepthEstimator>(std::move(config));
}

}  // namespace relume
