# DawnConfig.cmake references Threads::Threads without calling
# find_dependency(Threads), so the consumer has to provide it first.
find_package(Threads REQUIRED)
find_package(Dawn REQUIRED)
relume_fetch_glfw()

add_executable(relume_dawn_hello main.cpp metal_layer.mm)
target_compile_options(relume_dawn_hello PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
target_link_libraries(relume_dawn_hello PRIVATE dawn::webgpu_dawn glfw "-framework Cocoa" "-framework QuartzCore")
relume_set_warnings(relume_dawn_hello)

# Headless pixel check. The windowed mode needs a display and is run manually.
add_test(NAME dawn_offscreen_quad COMMAND relume_dawn_hello --offscreen)
set_tests_properties(dawn_offscreen_quad PROPERTIES TIMEOUT 60)
