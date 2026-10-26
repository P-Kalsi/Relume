# Threads: relume_core starts std::threads, and DawnConfig.cmake needs the
# target but does not find it.
find_package(Threads REQUIRED)

# relume_core: portable C++ with no platform or GPU dependencies.
add_library(relume_core STATIC
  core/video_frame.cpp
  core/pixel_convert.cpp
  core/frame_queue.cpp
  core/camera_format.cpp
  core/layout.cpp
  core/app_state.cpp
  core/stats.cpp
  core/event_queue.cpp
  core/image_io.cpp
  core/frame_saver.cpp
  core/guided_capture.cpp)
target_include_directories(relume_core PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(relume_core PUBLIC Threads::Threads)
relume_set_warnings(relume_core)

# relume_camera: camera sources. The file source is portable.
add_library(relume_camera STATIC
  camera/file_source.cpp)
target_link_libraries(relume_camera PUBLIC relume_core)
if(APPLE)
  # AVFoundation adapter (Objective-C++, ARC). Only this file is
  # platform-specific; everything above it uses the portable interfaces.
  target_sources(relume_camera PRIVATE camera/avfoundation.mm)
  target_compile_options(relume_camera PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_libraries(relume_camera PUBLIC
    "-framework Foundation" "-framework AVFoundation" "-framework CoreMedia" "-framework CoreVideo")
else()
  target_sources(relume_camera PRIVATE camera/camera_stub.cpp)
endif()
relume_set_warnings(relume_camera)

# relume_render: Dawn/WebGPU upload, compositing and presentation. WGSL
# sources are embedded into generated headers at build time.
find_package(Dawn REQUIRED)
set(_shader_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/shaders")
set(_shader_headers)
foreach(_shader convert blit)
  set(_input "${PROJECT_SOURCE_DIR}/shaders/${_shader}.wgsl")
  set(_output "${_shader_dir}/${_shader}_wgsl.hpp")
  string(SUBSTRING "${_shader}" 0 1 _first)
  string(TOUPPER "${_first}" _first)
  string(SUBSTRING "${_shader}" 1 -1 _rest)
  add_custom_command(OUTPUT "${_output}"
    COMMAND "${CMAKE_COMMAND}" -DINPUT=${_input} -DOUTPUT=${_output} -DNAME=k${_first}${_rest}Wgsl
            -P "${PROJECT_SOURCE_DIR}/cmake/EmbedShader.cmake"
    DEPENDS "${_input}" "${PROJECT_SOURCE_DIR}/cmake/EmbedShader.cmake"
    COMMENT "Embedding shaders/${_shader}.wgsl"
    VERBATIM)
  list(APPEND _shader_headers "${_output}")
endforeach()
add_library(relume_render STATIC
  render/renderer.cpp
  ${_shader_headers})
target_include_directories(relume_render PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(relume_render PUBLIC relume_core dawn::webgpu_dawn)
relume_set_warnings(relume_render)

# The Relume application.
relume_fetch_glfw()
relume_fetch_imgui()
add_executable(Relume
  main.cpp
  app/app.cpp
  app/window.cpp
  output/clean_window.cpp
  ui/control_window.cpp)
if(APPLE)
  target_sources(Relume PRIVATE
    platform/macos/surface.mm platform/macos/window_state.mm platform/macos/layer_presenter.mm)
  target_compile_options(Relume PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_libraries(Relume PRIVATE "-framework Cocoa" "-framework QuartzCore" "-framework IOSurface")
  # An .app bundle, so macOS privacy (TCC) attributes the camera request to
  # Relume and reads NSCameraUsageDescription. Launch it with `open`.
  set_target_properties(Relume PROPERTIES
    MACOSX_BUNDLE ON
    MACOSX_BUNDLE_INFO_PLIST "${PROJECT_SOURCE_DIR}/app/Info.plist")
  relume_codesign(Relume)
endif()
target_include_directories(Relume PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
# Default folder for --save-frames (git-ignored). Absolute, because `open`
# starts the app in /.
target_compile_definitions(Relume PRIVATE
  RELUME_DEFAULT_FRAME_DIR="${PROJECT_SOURCE_DIR}/third_party/assets/local_frames")
target_link_libraries(Relume PRIVATE relume_camera relume_render relume_imgui glfw)
relume_set_warnings(Relume)

# Headless smoke tests: the real pipeline from a file source to the output
# texture and a window-sized target, checked pixel by pixel. No camera.
add_test(NAME app_headless_smoke_bgra_720p
  COMMAND Relume --headless --source file --pattern 1280x720 --pixel-format bgra --frames 30)
add_test(NAME app_headless_smoke_nv12_4x3
  COMMAND Relume --headless --source file --pattern 640x480 --pixel-format nv12 --frames 30)
add_test(NAME app_headless_smoke_rgb_tall
  COMMAND Relume --headless --source file --pattern 480x640 --pixel-format rgb --frames 10)
set_tests_properties(app_headless_smoke_bgra_720p app_headless_smoke_nv12_4x3 app_headless_smoke_rgb_tall
  PROPERTIES TIMEOUT 60)

# Window occlusion regression test: opens real windows, covers, resizes and
# minimizes them for about a minute, and fails if presentation ever blocks.
# It needs a logged-in GUI session, so it carries the `display` label, which
# the default test presets exclude (run it with `ctest --preset display`).
add_test(NAME app_window_occlusion
  COMMAND Relume --source file --pixel-format nv12 --occlusion-test --stats 0)
set_tests_properties(app_window_occlusion PROPERTIES TIMEOUT 180 LABELS display)

# Guided capture in the real app with the file source (never the camera, and
# never the local_frames folder): scripted Capture of one still and one burst.
# RGB frames, because a Debug build needs about 51 ms to convert a 720p NV12
# frame, too slow for a 30 fps burst (Release: about 5 ms). Opens windows, so
# it is in the `display` set too.
add_test(NAME app_guided_capture_self_test
  COMMAND Relume --source file --pixel-format rgb --guided-capture --guided-self-test --stats 0
          --save-dir "${CMAKE_CURRENT_BINARY_DIR}/guided_self_test")
set_tests_properties(app_guided_capture_self_test PROPERTIES TIMEOUT 60 LABELS display)
