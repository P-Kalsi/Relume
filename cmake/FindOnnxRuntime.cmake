# Locates the prebuilt ONNX Runtime package installed by scripts/bootstrap_deps.sh
# and defines the imported target OnnxRuntime::OnnxRuntime.
#
# We do not use the package's bundled onnxruntimeConfig.cmake: in the 1.31.0
# osx-arm64 archive it points at include/onnxruntime/, but the headers are
# shipped directly in include/ (see docs/decisions.md).
find_path(OnnxRuntime_INCLUDE_DIR onnxruntime_cxx_api.h PATH_SUFFIXES include)
find_library(OnnxRuntime_LIBRARY NAMES onnxruntime PATH_SUFFIXES lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OnnxRuntime
  REQUIRED_VARS OnnxRuntime_LIBRARY OnnxRuntime_INCLUDE_DIR
  REASON_FAILURE_MESSAGE "Run scripts/bootstrap_deps.sh ort, then configure with a CMake preset")

if(OnnxRuntime_FOUND AND NOT TARGET OnnxRuntime::OnnxRuntime)
  add_library(OnnxRuntime::OnnxRuntime SHARED IMPORTED)
  set_target_properties(OnnxRuntime::OnnxRuntime PROPERTIES
    IMPORTED_LOCATION "${OnnxRuntime_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${OnnxRuntime_INCLUDE_DIR}")
endif()
