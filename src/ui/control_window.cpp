#include "ui/control_window.hpp"

#include "relume/layout.hpp"

#include <GLFW/glfw3.h>

#include <cstdio>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_wgpu.h>

#include <cstdint>

namespace relume {

namespace {

constexpr const char* kVirtualPopup = "Use a virtual camera?";

ImVec4 StatusColor(CaptureState state) {
  switch (state) {
    case CaptureState::kLive: return {0.35f, 0.85f, 0.45f, 1.0f};
    case CaptureState::kStarting:
    case CaptureState::kPermissionPending: return {0.95f, 0.80f, 0.30f, 1.0f};
    case CaptureState::kPermissionDenied:
    case CaptureState::kNoCamera:
    case CaptureState::kCameraUnavailable:
    case CaptureState::kError: return {0.95f, 0.40f, 0.35f, 1.0f};
  }
  return {1.0f, 1.0f, 1.0f, 1.0f};
}

std::string DeviceLabel(const CameraDeviceInfo& device) {
  return device.is_virtual ? device.name + "  (virtual camera)" : device.name;
}

}  // namespace

ControlWindow::ControlWindow(const GpuContext& gpu, const OutputTexture& output, PresenterKind kind)
    : gpu_(gpu), output_(output), window_(960, 640, kTitle, true) {
  if (window_.get() == nullptr) {
    error_ = "could not create the Controls window";
    return;
  }
  std::string note;
  presenter_ = CreateWindowPresenter(gpu, window_.get(), kind, note);
  if (!note.empty()) {
    std::printf("%s: %s\n", kTitle, note.c_str());
  }
  if (!presenter_->ok()) {
    error_ = presenter_->error();
    return;
  }
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.IniFilename = nullptr;  // never write imgui.ini (or anything else) to disk
  io.LogFilename = nullptr;
  ImGui::StyleColorsDark();
  // ImGui's callbacks go on this window only; the Output window has none.
  imgui_glfw_ = ImGui_ImplGlfw_InitForOther(window_.get(), true);
  ImGui_ImplWGPU_InitInfo info;
  info.Device = gpu.device().Get();
  info.RenderTargetFormat = static_cast<WGPUTextureFormat>(presenter_->format());
  imgui_wgpu_ = imgui_glfw_ && ImGui_ImplWGPU_Init(&info);
  if (!imgui_wgpu_) {
    error_ = "could not initialize Dear ImGui";
  }
}

ControlWindow::~ControlWindow() {
  if (imgui_wgpu_) {
    ImGui_ImplWGPU_Shutdown();
  }
  if (imgui_glfw_) {
    ImGui_ImplGlfw_Shutdown();
  }
  if (ImGui::GetCurrentContext() != nullptr) {
    ImGui::DestroyContext();
  }
}

bool ControlWindow::should_close() const { return glfwWindowShouldClose(window_.get()) == GLFW_TRUE; }

ControlsActions ControlWindow::BuildUi(const ControlsModel& model) {
  ControlsActions actions;
  const ImGuiViewport* viewport = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(viewport->WorkPos);
  ImGui::SetNextWindowSize(viewport->WorkSize);
  ImGui::Begin("Relume", nullptr,
               ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                   ImGuiWindowFlags_NoBringToFrontOnFocus);

  // Status.
  ImGui::TextColored(StatusColor(model.status.state), "%s", StatusLabel(model.status.state).data());
  ImGui::SameLine();
  ImGui::TextWrapped("%s", model.status.detail.c_str());
  if (!model.source_description.empty()) {
    ImGui::TextDisabled("%s", model.source_description.c_str());
  }
  if (!model.warning.empty()) {
    ImGui::TextColored({0.95f, 0.80f, 0.30f, 1.0f}, "%s", model.warning.c_str());
  }

  // Camera selection.
  const CameraDeviceInfo* selected = nullptr;
  for (const CameraDeviceInfo& device : model.devices) {
    if (device.id == model.selected_id) {
      selected = &device;
    }
  }
  const std::string preview = selected != nullptr ? DeviceLabel(*selected) : std::string("(none)");
  ImGui::BeginDisabled(!model.camera_selectable);
  ImGui::SetNextItemWidth(360.0f);
  if (ImGui::BeginCombo("Camera", preview.c_str())) {
    for (const CameraDeviceInfo& device : model.devices) {
      const bool is_selected = device.id == model.selected_id;
      if (ImGui::Selectable(DeviceLabel(device).c_str(), is_selected) && !is_selected) {
        if (device.is_virtual) {
          pending_virtual_id_ = device.id;  // ask first: it may be Relume's own output
        } else {
          actions.select_device = device.id;
        }
      }
      if (is_selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }
  ImGui::EndDisabled();
  if (!pending_virtual_id_.empty() && !ImGui::IsPopupOpen(kVirtualPopup)) {
    ImGui::OpenPopup(kVirtualPopup);
  }
  if (ImGui::BeginPopupModal(kVirtualPopup, nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::TextUnformatted(
        "This looks like a virtual camera such as OBS Virtual Camera.\n"
        "If it carries Relume's own output, you will create a feedback loop.");
    if (ImGui::Button("Use it anyway")) {
      actions.select_device = pending_virtual_id_;
      pending_virtual_id_.clear();
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
      pending_virtual_id_.clear();
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }

  // Output window.
  ImGui::SameLine();
  if (ImGui::Button(model.output_visible ? "Hide Output window" : "Show Output window")) {
    actions.toggle_output = true;
  }

  // Preview of the clean output, aspect-fit into the remaining space.
  ImGui::Separator();
  const ImVec2 avail = ImGui::GetContentRegionAvail();
  if (avail.x >= 1.0f && avail.y >= 1.0f) {
    const PixelRect fit = FitRect(OutputTexture::kWidth, OutputTexture::kHeight, static_cast<std::uint32_t>(avail.x),
                                  static_cast<std::uint32_t>(avail.y));
    const ImVec2 cursor = ImGui::GetCursorPos();
    ImGui::SetCursorPos({cursor.x + static_cast<float>(fit.x), cursor.y + static_cast<float>(fit.y)});
    ImGui::Image(static_cast<ImTextureID>(reinterpret_cast<std::uintptr_t>(output_.view().Get())),
                 {static_cast<float>(fit.width), static_cast<float>(fit.height)});
  }
  ImGui::End();
  return actions;
}

ControlsActions ControlWindow::Render(const ControlsModel& model) {
  if (!ok()) {
    return {};
  }
  ImGui_ImplWGPU_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  const ControlsActions actions = BuildUi(model);
  ImGui::Render();

  std::optional<WindowPresenter::Frame> frame = presenter_->Acquire();
  if (!frame) {
    return actions;  // minimized: the UI state still advanced
  }
  wgpu::RenderPassColorAttachment attachment{};
  attachment.view = frame->view;
  attachment.loadOp = wgpu::LoadOp::Clear;
  attachment.storeOp = wgpu::StoreOp::Store;
  attachment.clearValue = {0.08, 0.08, 0.09, 1.0};
  wgpu::RenderPassDescriptor pass_desc{};
  pass_desc.colorAttachmentCount = 1;
  pass_desc.colorAttachments = &attachment;
  const wgpu::CommandEncoder encoder = gpu_.device().CreateCommandEncoder();
  const wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&pass_desc);
  ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), pass.Get());
  pass.End();
  const wgpu::CommandBuffer commands = encoder.Finish();
  gpu_.queue().Submit(1, &commands);
  presenter_->Present();
  return actions;
}

}  // namespace relume
