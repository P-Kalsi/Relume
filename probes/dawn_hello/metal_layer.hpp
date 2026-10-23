#pragma once

struct GLFWwindow;

// Attaches a CAMetalLayer to the window's content view and returns it as an
// opaque pointer for wgpu::SurfaceSourceMetalLayer. The view owns the layer;
// the pointer stays valid for as long as the window exists.
void* AttachMetalLayer(GLFWwindow* window);
