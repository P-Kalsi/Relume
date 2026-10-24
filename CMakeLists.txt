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
include(Dependencies)

option(RELUME_BUILD_ORT_PROBE "Build the ONNX Runtime provider probe" ON)
option(RELUME_BUILD_DAWN_PROBE "Build the Dawn textured-quad probe" ON)
option(RELUME_BUILD_CAMERA_PROBE "Build the AVFoundation camera probe (macOS only)" ${APPLE})
option(RELUME_BUILD_TESTS "Build the native C++ unit tests (GoogleTest)" ON)

enable_testing()

if(RELUME_BUILD_TESTS)
  add_subdirectory(tests/cpp)
endif()

# Phase 0 feasibility probes.
if(RELUME_BUILD_ORT_PROBE)
  add_subdirectory(probes/ort_probe)
endif()
if(RELUME_BUILD_DAWN_PROBE)
  add_subdirectory(probes/dawn_hello)
endif()
# Python model-tooling tests (Phase 1). They need the venv created by
# scripts/setup_python.sh; without it they are not registered.
set(RELUME_VENV_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
if(EXISTS "${RELUME_VENV_PYTHON}")
  set(_relume_python_tests_default ON)
else()
  set(_relume_python_tests_default OFF)
endif()
option(RELUME_PYTHON_TESTS "Register the Python model-tooling tests with CTest" ${_relume_python_tests_default})
if(RELUME_PYTHON_TESTS)
  if(NOT EXISTS "${RELUME_VENV_PYTHON}")
    message(FATAL_ERROR "RELUME_PYTHON_TESTS=ON but ${RELUME_VENV_PYTHON} is missing; run scripts/setup_python.sh")
  endif()
  add_test(NAME python_preprocess
    COMMAND "${RELUME_VENV_PYTHON}" -m pytest -q -p no:cacheprovider tests/python/test_preprocess.py
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
  # Fails (with instructions) if scripts/export.py has not been run.
  add_test(NAME python_model_artifacts
    COMMAND "${RELUME_VENV_PYTHON}" -m pytest -q -p no:cacheprovider tests/python/test_model_artifacts.py
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
  set_tests_properties(python_preprocess python_model_artifacts PROPERTIES TIMEOUT 300)
else()
  message(STATUS "Python model tests not registered (run scripts/setup_python.sh to enable)")
endif()

if(RELUME_BUILD_CAMERA_PROBE)
  if(NOT APPLE)
    message(FATAL_ERROR "The camera probe uses AVFoundation and only builds on macOS")
  endif()
  add_subdirectory(probes/camera_probe)
endif()
