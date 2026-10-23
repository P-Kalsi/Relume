cmake_minimum_required(VERSION 3.25)

# ONNX Runtime 1.31.0's prebuilt dylib declares minos 14.0, so that is the
# lowest macOS we can target. Must be set before project().
set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum macOS version")

project(Relume VERSION 0.0.0 LANGUAGES C CXX)
if(APPLE)
  # OBJC must be enabled too: with only OBJCXX, CMake compiles third-party
  # .m files (e.g. GLFW's) as Objective-C++, which they are not.
  enable_language(OBJC OBJCXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Sanitizers)
include(Warnings)

option(RELUME_BUILD_ORT_PROBE "Build the ONNX Runtime provider probe" ON)
option(RELUME_BUILD_DAWN_PROBE "Build the Dawn textured-quad probe" ON)
option(RELUME_BUILD_CAMERA_PROBE "Build the AVFoundation camera probe (macOS only)" ${APPLE})

enable_testing()

# Phase 0 contains only feasibility probes; the application itself starts in
# later phases (see RELUME_PRD_REVISED.md §8).
if(RELUME_BUILD_ORT_PROBE)
  add_subdirectory(probes/ort_probe)
endif()
if(RELUME_BUILD_DAWN_PROBE)
  add_subdirectory(probes/dawn_hello)
endif()
if(RELUME_BUILD_CAMERA_PROBE)
  if(NOT APPLE)
    message(FATAL_ERROR "The camera probe uses AVFoundation and only builds on macOS")
  endif()
  add_subdirectory(probes/camera_probe)
endif()
