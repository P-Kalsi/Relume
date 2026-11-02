# relume_relight: the Phase 4 temporal-stabilization ablation (metrics only;
# it writes nothing but the optional --json summary).
add_executable(relume_relight main.cpp)
target_link_libraries(relume_relight PRIVATE relume_inference)
target_compile_definitions(relume_relight PRIVATE
  RELUME_DEFAULT_MODEL="${PROJECT_SOURCE_DIR}/models/depth_anything_v2_small_448.onnx")
relume_set_warnings(relume_relight)
