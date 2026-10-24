// Relume: live, on-device webcam relighting. Phase 2: camera passthrough into
// the Controls and Output windows.

#include "app/app.hpp"

#include <cstdio>
#include <exception>

int main(int argc, char** argv) {
  std::string error;
  const std::optional<relume::AppOptions> options = relume::ParseOptions(argc, argv, error);
  if (!options) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 2;
  }
  try {
    return relume::RunApp(*options);
  } catch (const std::exception& e) {
    // Last resort: report instead of terminating with an unexplained crash.
    std::fprintf(stderr, "Relume stopped because of an internal error: %s\n", e.what());
    return 1;
  }
}
