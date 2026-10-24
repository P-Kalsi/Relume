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
