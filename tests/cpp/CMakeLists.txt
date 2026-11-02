# Native unit tests (GoogleTest). None of them needs a camera; GPU tests render
# offscreen with Dawn.
relume_fetch_googletest()
include(GoogleTest)

# relume_add_gtest(<name> SOURCES <files...> LIBS <targets...>)
function(relume_add_gtest name)
  cmake_parse_arguments(ARG "" "" "SOURCES;LIBS" ${ARGN})
  add_executable(${name} ${ARG_SOURCES})
  target_link_libraries(${name} PRIVATE ${ARG_LIBS} GTest::gtest_main)
  relume_set_warnings(${name})
  gtest_discover_tests(${name} DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 120)
endfunction()

relume_add_gtest(test_pixel_convert SOURCES test_pixel_convert.cpp LIBS relume_core)
relume_add_gtest(test_frame_queue SOURCES test_frame_queue.cpp LIBS relume_core)
relume_add_gtest(test_camera_format SOURCES test_camera_format.cpp LIBS relume_core)
relume_add_gtest(test_layout SOURCES test_layout.cpp LIBS relume_core)
relume_add_gtest(test_app_state SOURCES test_app_state.cpp LIBS relume_core)
relume_add_gtest(test_file_source SOURCES test_file_source.cpp LIBS relume_camera)
relume_add_gtest(test_gpu_upload SOURCES test_gpu_upload.cpp LIBS relume_render)
relume_add_gtest(test_compositor_reuse SOURCES test_compositor_reuse.cpp LIBS relume_core)
relume_add_gtest(test_frame_saver SOURCES test_frame_saver.cpp LIBS relume_core)
relume_add_gtest(test_guided_capture SOURCES test_guided_capture.cpp LIBS relume_core)
relume_add_gtest(test_preprocess SOURCES test_preprocess.cpp LIBS relume_core)
target_compile_definitions(test_preprocess PRIVATE
  RELUME_GOLDEN_TEXT="${PROJECT_SOURCE_DIR}/tests/python/golden/preprocess_golden.txt")
relume_add_gtest(test_onnx_depth SOURCES test_onnx_depth.cpp LIBS relume_inference)
target_compile_definitions(test_onnx_depth PRIVATE
  RELUME_TEST_MODEL="${PROJECT_SOURCE_DIR}/models/depth_anything_v2_small_448.onnx")
relume_add_gtest(test_depth_worker SOURCES test_depth_worker.cpp LIBS relume_core)
relume_add_gtest(test_depth_vis SOURCES test_depth_vis.cpp LIBS relume_render)
relume_add_gtest(test_test_guide SOURCES test_test_guide.cpp LIBS relume_core)
relume_add_gtest(test_lighting_math SOURCES test_lighting_math.cpp LIBS relume_core)
