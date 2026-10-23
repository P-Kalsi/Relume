#include "metal_layer.hpp"

#import <Cocoa/Cocoa.h>
#import <QuartzCore/CAMetalLayer.h>

#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

void* AttachMetalLayer(GLFWwindow* window) {
  NSWindow* ns_window = glfwGetCocoaWindow(window);
  if (ns_window == nil) {
    return nullptr;
  }
  CAMetalLayer* layer = [CAMetalLayer layer];
  ns_window.contentView.wantsLayer = YES;
  ns_window.contentView.layer = layer;
  return (__bridge void*)layer;
}
