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
  core/event_queue.cpp
  core/image_io.cpp
  core/frame_saver.cpp
  core/guided_capture.cpp
  core/preprocess.cpp
  core/depth_estimator.cpp
  core/depth_worker.cpp
  core/inference_controller.cpp
  core/test_guide.cpp
  core/lighting.cpp
  core/crossfade.cpp)
target_include_directories(relume_core PUBLIC "${PROJECT_SOURCE_DIR}/include")
# The preprocessing contract is matched against numpy, which rounds every
# product and sum separately: no fused multiply-add.
set_source_files_properties(core/preprocess.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
target_link_libraries(relume_core PUBLIC Threads::Threads)
relume_set_warnings(relume_core)

# relume_inference: depth estimation with ONNX Runtime behind the portable
# IDepthEstimator interface. Portable C++; no Objective-C++.
find_package(OnnxRuntime REQUIRED)
add_library(relume_inference STATIC
  inference/onnx_depth.cpp)
target_link_libraries(relume_inference PUBLIC relume_core PRIVATE OnnxRuntime::OnnxRuntime)
relume_set_warnings(relume_inference)

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
foreach(_shader convert blit depth_vis normals relight)
  set(_input "${PROJECT_SOURCE_DIR}/shaders/${_shader}.wgsl")
  set(_output "${_shader_dir}/${_shader}_wgsl.hpp")
  # depth_vis -> kDepthVisWgsl
  set(_name "k")
  string(REPLACE "_" ";" _words "${_shader}")
  foreach(_word IN LISTS _words)
    string(SUBSTRING "${_word}" 0 1 _first)
    string(TOUPPER "${_first}" _first)
    string(SUBSTRING "${_word}" 1 -1 _rest)
    string(APPEND _name "${_first}${_rest}")
  endforeach()
  add_custom_command(OUTPUT "${_output}"
    COMMAND "${CMAKE_COMMAND}" -DINPUT=${_input} -DOUTPUT=${_output} -DNAME=${_name}Wgsl
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
  RELUME_DEFAULT_FRAME_DIR="${PROJECT_SOURCE_DIR}/third_party/assets/local_frames"
  RELUME_DEFAULT_MODEL="${PROJECT_SOURCE_DIR}/models/depth_anything_v2_small_448.onnx")
target_link_libraries(Relume PRIVATE relume_camera relume_render relume_inference relume_imgui glfw)
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
# The same, with the real depth pipeline (ONNX Runtime CPU): a depth result
# must arrive and match the shown frame's camera generation, size and
# transform. Registered only when the exported model exists.
if(EXISTS "${PROJECT_SOURCE_DIR}/models/depth_anything_v2_small_448.onnx")
  add_test(NAME app_headless_inference_nv12_720p
    COMMAND Relume --headless --source file --pattern 1280x720 --pixel-format nv12 --frames 30 --inference cpu)
  add_test(NAME app_headless_inference_portrait_letterbox
    COMMAND Relume --headless --source file --pattern 480x640 --pixel-format bgra --frames 10 --inference cpu
            --preprocess letterbox)
  add_test(NAME app_headless_inference_missing_model
    COMMAND Relume --headless --source file --frames 10 --inference cpu --model /nonexistent/model.onnx)
  # The default provider (CoreML CPUAndGPU) end to end; the first run compiles
  # the model into the user's CoreML cache (about 90 s), later runs about 16 s.
  add_test(NAME app_headless_inference_coreml_default
    COMMAND Relume --headless --source file --pattern 1280x720 --pixel-format nv12 --frames 30 --inference coreml)
  # Automatic CPU fallback: strict mode makes the CoreML session fail on this
  # model (36 Gather nodes need CPU), so the worker must fall back to CPU and
  # still deliver matching depth.
  add_test(NAME app_headless_inference_cpu_fallback
    COMMAND Relume --headless --source file --pattern 1280x720 --pixel-format nv12 --frames 30 --inference coreml
            --coreml-strict)
  set_tests_properties(app_headless_inference_nv12_720p app_headless_inference_portrait_letterbox
    PROPERTIES TIMEOUT 120)
  set_tests_properties(app_headless_inference_coreml_default PROPERTIES TIMEOUT 400
    PASS_REGULAR_EXPRESSION "Depth: Ready [(]CoreML [(]CPUAndGPU[)].*\n.*\n.*depth matches the shown camera image.*: yes")
  set_tests_properties(app_headless_inference_cpu_fallback PROPERTIES TIMEOUT 400
    PASS_REGULAR_EXPRESSION "Depth: CPU fallback [|].*running on CPU.*\n.*\n.*depth matches the shown camera image.*: yes")
  # A missing model must be an error state, not a crash or a hang: the run
  # reports the Error status and passthrough, and refuses to call it a match.
  set_tests_properties(app_headless_inference_missing_model PROPERTIES TIMEOUT 60
    PASS_REGULAR_EXPRESSION "Depth: Error [|].*passthrough.*\n.*depth matches the shown camera image.*: NO")
endif()

# Window occlusion regression test: opens real windows, covers, resizes and
# minimizes them for about a minute, and fails if presentation ever blocks.
# It needs a logged-in GUI session, so it carries the `display` label, which
# the default test presets exclude (run it with `ctest --preset display`).
add_test(NAME app_window_occlusion
  COMMAND Relume --source file --pixel-format nv12 --occlusion-test --stats 0)
set_tests_properties(app_window_occlusion PROPERTIES TIMEOUT 180 LABELS display)

# Guided capture in the real app with the file source (never the camera, and
# never the local_frames folder): scripted Capture of one still and one burst.
# RGB frames at 640x360: this checks the capture logic and that the loop
# never waits for the disk, not the writer's throughput. A Debug build needs
# about 51 ms to convert a 720p NV12 frame, and even 720p RGB passed only 9 of
# 10 runs (the burst writer barely keeps up); 640x360 passed 10 of 10. Release
# throughput is checked by hand at 720p NV12 (docs/decisions.md). Opens
# windows, so it is in the `display` set too.
add_test(NAME app_guided_capture_self_test
  COMMAND Relume --source file --pattern 640x360 --pixel-format rgb --guided-capture --guided-self-test --stats 0
          --save-dir "${CMAKE_CURRENT_BINARY_DIR}/guided_self_test")
set_tests_properties(app_guided_capture_self_test PROPERTIES TIMEOUT 60 LABELS display)

# Live provider switching with the real ONNX Runtime (CPU and CoreML), file
# source: CPU -> CoreML -> CPU -> off -> on, then 10 rapid changes. Fails if a
# step does not settle, a worker is not joined, or any loop iteration takes
# over 50 ms. Opens windows; the first CoreML compile can take a while.
if(EXISTS "${PROJECT_SOURCE_DIR}/models/depth_anything_v2_small_448.onnx")
  add_test(NAME app_inference_switch_test
    COMMAND Relume --source file --pixel-format nv12 --inference cpu --inference-switch-test --stats 5)
  set_tests_properties(app_inference_switch_test PROPERTIES TIMEOUT 600 LABELS display)
endif()
