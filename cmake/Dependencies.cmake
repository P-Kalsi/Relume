# Source dependencies fetched at configure time with FetchContent. Every
# download is pinned by URL and SHA-256; versions are recorded in
# docs/dependencies.md. Prebuilt dependencies (Dawn, ONNX Runtime) come from
# scripts/bootstrap_deps.sh instead.
include(FetchContent)

# GLFW: windowing for the Dawn probe and the app.
function(relume_fetch_glfw)
  if(TARGET glfw)
    return()
  endif()
  FetchContent_Declare(glfw
    URL https://github.com/glfw/glfw/releases/download/3.4/glfw-3.4.zip
    URL_HASH SHA256=b5ec004b2712fd08e8861dc271428f048775200a2df719ccf575143ba749a3e9)
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(glfw)
endfunction()

# Dear ImGui with its GLFW and WebGPU (Dawn) backends, built as the static
# library relume_imgui. ImGui ships no CMake build, so the sources are only
# downloaded and compiled here (third-party flags, not relume_set_warnings).
# GitHub publishes no digest for tag archives; the hash was computed on first
# download.
function(relume_fetch_imgui)
  if(TARGET relume_imgui)
    return()
  endif()
  FetchContent_Declare(imgui
    URL https://github.com/ocornut/imgui/archive/refs/tags/v1.93.0.tar.gz
    URL_HASH SHA256=f7bf6bc76e5d79d40d751096663e6c5d18daf08129dd52e7f7b09e4251c8eeee
    SOURCE_SUBDIR no-cmake-build)  # populate only
  FetchContent_MakeAvailable(imgui)
  set(dir "${imgui_SOURCE_DIR}")
  add_library(relume_imgui STATIC
    "${dir}/imgui.cpp" "${dir}/imgui_draw.cpp" "${dir}/imgui_tables.cpp" "${dir}/imgui_widgets.cpp"
    "${dir}/backends/imgui_impl_glfw.cpp" "${dir}/backends/imgui_impl_wgpu.cpp")
  target_include_directories(relume_imgui PUBLIC "${dir}" "${dir}/backends")
  target_compile_definitions(relume_imgui PUBLIC IMGUI_IMPL_WEBGPU_BACKEND_DAWN IMGUI_DISABLE_OBSOLETE_FUNCTIONS)
  target_link_libraries(relume_imgui PUBLIC dawn::webgpu_dawn glfw)
  if(APPLE)
    # The WebGPU backend's macOS surface helper uses Cocoa, so upstream
    # requires compiling it as Objective-C++ (manual reference counting).
    set_source_files_properties("${dir}/backends/imgui_impl_wgpu.cpp" PROPERTIES COMPILE_FLAGS "-x objective-c++")
    target_link_libraries(relume_imgui PUBLIC "-framework Cocoa" "-framework QuartzCore")
  endif()
endfunction()

# GoogleTest: native unit tests. GitHub publishes no digest for this asset;
# the hash was computed on first download.
function(relume_fetch_googletest)
  if(TARGET GTest::gtest_main)
    return()
  endif()
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
    URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)  # MSVC only; harmless elsewhere
  FetchContent_MakeAvailable(googletest)
endfunction()
