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
