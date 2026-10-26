#include "platform/window_state.hpp"

#import <Cocoa/Cocoa.h>
#include <mach/mach.h>

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

double ProcessFootprintMiB() {
  task_vm_info_data_t info{};
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) {
    return -1.0;
  }
  return static_cast<double>(info.phys_footprint) / (1024.0 * 1024.0);
}

bool IsApplicationActive() { return NSApp.isActive; }

std::string UserCachesDirectory() {
  NSArray<NSString*>* paths = NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES);
  return paths.count > 0 ? std::string(paths.firstObject.fileSystemRepresentation) : std::string();
}

}  // namespace relume
