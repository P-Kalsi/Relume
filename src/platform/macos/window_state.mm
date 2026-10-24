#include "platform/window_state.hpp"

#import <Cocoa/Cocoa.h>

#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

namespace relume {

bool IsWindowOnScreen(GLFWwindow* window) {
  NSWindow* ns_window = glfwGetCocoaWindow(window);
  if (ns_window == nil) {
    return false;
  }
  return (ns_window.occlusionState & NSWindowOcclusionStateVisible) != 0;
}

void MakeWindowOpaque(GLFWwindow* window) {
  NSWindow* ns_window = glfwGetCocoaWindow(window);
  ns_window.opaque = YES;
  ns_window.backgroundColor = [NSColor colorWithCalibratedRed:0.5 green:0.1 blue:0.5 alpha:1.0];
  ns_window.contentView.wantsLayer = YES;
  ns_window.contentView.layer.backgroundColor = ns_window.backgroundColor.CGColor;
  ns_window.contentView.layer.opaque = YES;
}

double DisplayRefreshHz(GLFWwindow* window) {
  NSWindow* ns_window = glfwGetCocoaWindow(window);
  NSScreen* screen = ns_window != nil && ns_window.screen != nil ? ns_window.screen : NSScreen.mainScreen;
  const NSInteger fps = screen != nil ? screen.maximumFramesPerSecond : 0;
  return fps > 0 ? static_cast<double>(fps) : 60.0;
}

bool IsApplicationActive() { return NSApp.isActive; }

}  // namespace relume
