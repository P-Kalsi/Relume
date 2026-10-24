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
