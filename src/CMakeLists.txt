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
  core/image_io.cpp)
target_include_directories(relume_core PUBLIC "${PROJECT_SOURCE_DIR}/include")
relume_set_warnings(relume_core)

# relume_camera: camera sources. The file source is portable.
add_library(relume_camera STATIC
  camera/file_source.cpp)
target_link_libraries(relume_camera PUBLIC relume_core)
relume_set_warnings(relume_camera)

# relume_render: Dawn/WebGPU upload, compositing and presentation. WGSL
# sources are embedded into generated headers at build time.
find_package(Threads REQUIRED)  # DawnConfig.cmake needs it but does not find it
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
  target_sources(Relume PRIVATE platform/macos/surface.mm)
  target_compile_options(Relume PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_libraries(Relume PRIVATE "-framework Cocoa" "-framework QuartzCore")
endif()
target_include_directories(Relume PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
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
