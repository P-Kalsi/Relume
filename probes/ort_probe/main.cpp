// Phase 0 probe: which ONNX Runtime execution providers (EPs) exist in this
// build, and does the CoreML EP accept registration on a session?
//
// Important limit: a provider being listed or accepted on SessionOptions does
// NOT mean any model node runs on CoreML. That needs a real model, ORT
// profiling and a parity check, which is Phase 1+ work.

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool contains(const std::vector<std::string>& items, const std::string& value) {
  return std::find(items.begin(), items.end(), value) != items.end();
}

}  // namespace

int main(int argc, char** argv) {
  const bool require_coreml = argc > 1 && std::strcmp(argv[1], "--require-coreml") == 0;

  try {
    std::cout << "ONNX Runtime version: " << Ort::GetVersionString() << '\n';

    const std::vector<std::string> providers = Ort::GetAvailableProviders();
    std::cout << "Available execution providers (" << providers.size() << "):\n";
    for (const std::string& name : providers) {
      std::cout << "  - " << name << '\n';
    }

    const bool has_cpu = contains(providers, "CPUExecutionProvider");
    const bool has_coreml = contains(providers, "CoreMLExecutionProvider");

    // Ort::Env owns the runtime's global state; it is released when it goes
    // out of scope (RAII), so there is no manual cleanup below.
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "relume_ort_probe");

    bool coreml_registered = false;
    if (has_coreml) {
      Ort::SessionOptions options;
      try {
        options.AppendExecutionProvider("CoreML", {{"ModelFormat", "MLProgram"}});
        coreml_registered = true;
        std::cout << "CoreML EP: registered on SessionOptions "
                     "(no model has been run; CoreML execution is NOT yet verified)\n";
      } catch (const Ort::Exception& e) {
        std::cout << "CoreML EP: listed but registration failed: " << e.what() << '\n';
      }
    } else {
      std::cout << "CoreML EP: not available in this ONNX Runtime build\n";
    }

    if (!has_cpu) {
      std::cerr << "FAIL: CPUExecutionProvider missing\n";
      return 1;
    }
    if (require_coreml && !coreml_registered) {
      std::cerr << "FAIL: CoreML EP could not be registered\n";
      return 1;
    }
    std::cout << "PASS\n";
    return 0;
  } catch (const Ort::Exception& e) {
    std::cerr << "FAIL: ONNX Runtime error: " << e.what() << '\n';
    return 1;
  }
}
