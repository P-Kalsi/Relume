#include "relume/depth_estimator.hpp"

// Names used on the command line and in the UI; the ONNX Runtime
// implementation is in src/inference/onnx_depth.cpp.

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

}  // namespace relume
