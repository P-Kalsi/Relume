#include "platform/surface.hpp"

#import <Cocoa/Cocoa.h>
#import <QuartzCore/CAMetalLayer.h>

#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

namespace relume {

wgpu::Surface CreateSurfaceForWindow(const wgpu::Instance& instance, GLFWwindow* window) {
  NSWindow* ns_window = glfwGetCocoaWindow(window);
  if (ns_window == nil) {
    return nullptr;
  }
  // The content view owns the layer for as long as the window exists.
  CAMetalLayer* layer = [CAMetalLayer layer];
  layer.opaque = YES;
  ns_window.contentView.wantsLayer = YES;
  ns_window.contentView.layer = layer;

  wgpu::SurfaceSourceMetalLayer metal_source{};
  metal_source.layer = (__bridge void*)layer;
  wgpu::SurfaceDescriptor desc{};
  desc.nextInChain = &metal_source;
  return instance.CreateSurface(&desc);
}

}  // namespace relume
