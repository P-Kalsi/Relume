# DawnConfig.cmake references Threads::Threads without calling
# find_dependency(Threads), so the consumer has to provide it first.
find_package(Threads REQUIRED)
find_package(Dawn REQUIRED)

include(FetchContent)
FetchContent_Declare(glfw
  URL https://github.com/glfw/glfw/releases/download/3.4/glfw-3.4.zip
  URL_HASH SHA256=b5ec004b2712fd08e8861dc271428f048775200a2df719ccf575143ba749a3e9)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

add_executable(relume_dawn_hello main.cpp metal_layer.mm)
target_compile_options(relume_dawn_hello PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
target_link_libraries(relume_dawn_hello PRIVATE dawn::webgpu_dawn glfw "-framework Cocoa" "-framework QuartzCore")
relume_set_warnings(relume_dawn_hello)

# Headless pixel check. The windowed mode needs a display and is run manually.
add_test(NAME dawn_offscreen_quad COMMAND relume_dawn_hello --offscreen)
set_tests_properties(dawn_offscreen_quad PROPERTIES TIMEOUT 60)
