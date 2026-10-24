#pragma once

#include <webgpu/webgpu_cpp.h>

struct GLFWwindow;

namespace relume {

// Creates a WebGPU surface for a GLFW window using the platform's native
// layer (a CAMetalLayer on macOS). Returns null if the platform is not
// supported. Implemented per platform (src/platform/macos/surface.mm).
wgpu::Surface CreateSurfaceForWindow(const wgpu::Instance& instance, GLFWwindow* window);

}  // namespace relume
