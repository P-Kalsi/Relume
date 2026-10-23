find_package(OnnxRuntime REQUIRED)

add_executable(relume_ort_probe main.cpp)
target_link_libraries(relume_ort_probe PRIVATE OnnxRuntime::OnnxRuntime)
relume_set_warnings(relume_ort_probe)

# Fails if the CPU provider is missing or the CoreML provider cannot be
# registered on session options. Does NOT prove a model runs on CoreML.
add_test(NAME ort_provider_probe COMMAND relume_ort_probe --require-coreml)
set_tests_properties(ort_provider_probe PROPERTIES TIMEOUT 60)
