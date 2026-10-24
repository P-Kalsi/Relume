#pragma once

#include "app/window.hpp"

#include <memory>
#include <string>

struct GLFWwindow;

namespace relume {

// A WindowPresenter that never acquires a CAMetalDrawable: frames are drawn
// into IOSurface-backed textures (Dawn SharedTextureMemory) and shown as the
// window layer's contents (src/platform/macos/layer_presenter.mm). Returns
// null and sets `error` where this is unavailable (no macOS, or the GPU lacks
// the SharedTextureMemoryIOSurface / SharedFenceMTLSharedEvent features).
std::unique_ptr<WindowPresenter> CreateLayerPresenter(const GpuContext& gpu, GLFWwindow* window,
                                                      std::string& error);

}  // namespace relume
