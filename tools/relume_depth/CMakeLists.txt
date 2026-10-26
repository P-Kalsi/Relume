# relume_depth: native preprocessing + ONNX Runtime from the command line,
# used by scripts/check_native_parity.py and for benchmarks.
add_executable(relume_depth main.cpp)
target_link_libraries(relume_depth PRIVATE relume_inference)
# Absolute default model path (dev builds; bundling the model is Phase 8).
target_compile_definitions(relume_depth PRIVATE
  RELUME_DEFAULT_MODEL="${PROJECT_SOURCE_DIR}/models/depth_anything_v2_small_448.onnx")
relume_set_warnings(relume_depth)
