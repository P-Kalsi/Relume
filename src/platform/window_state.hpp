#pragma once

struct GLFWwindow;

namespace relume {

// True if any part of the window is currently visible on screen, as the
// window server reports it (NSWindow.occlusionState on macOS). A covered,
// minimized, hidden or off-screen window reports false. Platforms without
// occlusion information always report true.
bool IsWindowOnScreen(GLFWwindow* window);

// Gives a window an opaque solid background so it hides what is behind it
// (used by the occlusion test's covering window, which draws nothing).
void MakeWindowOpaque(GLFWwindow* window);

// Refresh rate of the display showing the window, in Hz (60 if unknown).
double DisplayRefreshHz(GLFWwindow* window);

// The process's physical memory footprint in MiB, as Activity Monitor's
// "Memory" column reports it (task_vm_info phys_footprint); -1 if unknown.
double ProcessFootprintMiB();

// True if this application is the active (frontmost) app.
bool IsApplicationActive();

}  // namespace relume
