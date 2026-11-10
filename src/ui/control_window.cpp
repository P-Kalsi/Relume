#include "ui/control_window.hpp"

#include "relume/layout.hpp"

#include <GLFW/glfw3.h>

#include <cstdio>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_wgpu.h>

#include <algorithm>
#include <cmath>
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

ImVec4 InferenceColor(InferenceState state) {
  switch (state) {
    case InferenceState::kReady: return {0.35f, 0.85f, 0.45f, 1.0f};
    case InferenceState::kLoading:
    case InferenceState::kCpuFallback: return {0.95f, 0.80f, 0.30f, 1.0f};
    case InferenceState::kError: return {0.95f, 0.40f, 0.35f, 1.0f};
    case InferenceState::kOff: return {0.6f, 0.6f, 0.6f, 1.0f};
  }
  return {1.0f, 1.0f, 1.0f, 1.0f};
}

// Draws `view` aspect-fit (1280x720) into a `width` x `height` area at the
// cursor. Returns where it landed on screen.
PreviewGeometry FittedImage(const wgpu::TextureView& view, float width, float height) {
  PreviewGeometry geometry;
  geometry.output_width = OutputTexture::kWidth;
  geometry.output_height = OutputTexture::kHeight;
  if (width < 1.0f || height < 1.0f) {
    return geometry;
  }
  const PixelRect fit = FitRect(OutputTexture::kWidth, OutputTexture::kHeight, static_cast<std::uint32_t>(width),
                                static_cast<std::uint32_t>(height));
  const ImVec2 cursor = ImGui::GetCursorPos();
  ImGui::SetCursorPos({cursor.x + static_cast<float>(fit.x), cursor.y + static_cast<float>(fit.y)});
  ImGui::Image(static_cast<ImTextureID>(reinterpret_cast<std::uintptr_t>(view.Get())),
               {static_cast<float>(fit.width), static_cast<float>(fit.height)});
  const ImVec2 min = ImGui::GetItemRectMin();
  geometry.x = min.x;
  geometry.y = min.y;
  geometry.width = static_cast<float>(fit.width);
  geometry.height = static_cast<float>(fit.height);
  return geometry;
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

  if (model.test_guide != nullptr) {
    BuildTestGuide(*model.test_guide, model.now, actions);
  }
  if (model.guided != nullptr) {
    BuildGuidedCapture(*model.guided, model.now, actions);
  }
  BuildInference(model, actions);
  BuildLighting(model, actions);

  // Preview of the clean output and, beside it, the depth or lighting debug
  // view, each aspect-fit into half of the remaining space. The light is
  // dragged on the output preview; its marker is drawn here only.
  ImGui::Separator();
  const ImVec2 avail = ImGui::GetContentRegionAvail();
  const char* output_caption = model.relit ? "Output (relit; drag to move the light)" : "Output (camera passthrough)";
  if (model.depth_view == nullptr) {
    ImGui::TextDisabled("%s", output_caption);
    const PreviewGeometry output_preview =
        FittedImage(output_.view(), avail.x, avail.y - ImGui::GetTextLineHeightWithSpacing());
    HandleLightDrag(model, output_preview, actions);
  } else {
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float half = (avail.x - spacing) / 2.0f;
    const float label_height = ImGui::GetTextLineHeightWithSpacing();
    const ImVec2 start = ImGui::GetCursorPos();
    ImGui::TextDisabled("%s", output_caption);
    const PreviewGeometry output_preview = FittedImage(output_.view(), half, avail.y - label_height);
    HandleLightDrag(model, output_preview, actions);
    ImGui::SetCursorPos({start.x + half + spacing, start.y});
    const int view = std::clamp(model.relight.view, 0, static_cast<int>(std::size(kRelightViews)) - 1);
    const bool raw_depth = !kRelightViews[view].view.has_value();
    ImGui::TextDisabled("%s", raw_depth && model.depth_overlay ? "Depth over camera (brighter = nearer)"
                                                               : kRelightViews[view].caption);
    ImGui::SetCursorPos({start.x + half + spacing, start.y + label_height});
    FittedImage(model.depth_view->view(), half, avail.y - label_height);
  }
  ImGui::End();
  return actions;
}

// Timed steps for a manual test. Text only: the person does every action.
void ControlWindow::BuildTestGuide(const TestGuide& guide, TestGuide::Clock::time_point now,
                                   ControlsActions& actions) {
  const auto clock = [](double seconds) {
    const int s = static_cast<int>(seconds);
    char text[16];
    std::snprintf(text, sizeof(text), "%d:%02d", s / 60, s % 60);
    return std::string(text);
  };
  ImGui::Separator();
  const std::size_t index = guide.current();
  const GuideStep& step = guide.step(index);
  ImGui::TextColored({0.55f, 0.75f, 1.0f, 1.0f}, "Test guide  |  elapsed %s  |  step %zu of %zu",
                     clock(guide.elapsed(now)).c_str(), index + 1, guide.size());
  ImGui::PushFont(nullptr, ImGui::GetStyle().FontSizeBase * 1.3f);
  ImGui::TextUnformatted(step.title.c_str());
  ImGui::PopFont();
  ImGui::TextWrapped("%s", step.instruction.c_str());
  if (!guide.finished()) {
    ImGui::TextColored({0.95f, 0.80f, 0.30f, 1.0f}, "Next step in %.0f s  (this step %s-%s)", guide.remaining(now),
                       clock(guide.step_start(index)).c_str(), clock(guide.step_start(index + 1)).c_str());
    ImGui::SameLine();
    if (ImGui::Button("Skip to next step")) {
      actions.test_guide_skip = true;
    }
  }
  if (ImGui::TreeNode("All steps")) {
    for (std::size_t i = 0; i < guide.size(); ++i) {
      const ImVec4 color = i < index    ? ImVec4{0.5f, 0.5f, 0.5f, 1.0f}
                           : i == index ? ImVec4{0.35f, 0.85f, 0.45f, 1.0f}
                                        : ImVec4{0.85f, 0.85f, 0.85f, 1.0f};
      ImGui::TextColored(color, "%s  %2zu. %s", clock(guide.step_start(i)).c_str(), i + 1,
                         guide.step(i).title.c_str());
    }
    ImGui::TreePop();
  }
}

// Inference status and settings. The depth label always says how old the
// depth is; it never implies frame-exact sync unless the lag is 0.
void ControlWindow::BuildInference(const ControlsModel& model, ControlsActions& actions) {
  ImGui::Separator();
  ImGui::TextUnformatted("Depth:");
  ImGui::SameLine();
  ImGui::TextColored(InferenceColor(model.inference.state), "%s", model.inference.label.c_str());
  ImGui::SameLine();
  ImGui::TextWrapped("%s", model.inference.detail.c_str());
  ImGui::SetNextItemWidth(260.0f);
  const int current = std::clamp(model.inference_choice, 0, static_cast<int>(std::size(kInferenceChoices)) - 1);
  if (ImGui::BeginCombo("Inference", kInferenceChoices[current].label)) {
    for (int i = 0; i < static_cast<int>(std::size(kInferenceChoices)); ++i) {
      if (ImGui::Selectable(kInferenceChoices[i].label, i == current) && i != current) {
        actions.inference_choice = i;
      }
    }
    ImGui::EndCombo();
  }
  ImGui::SameLine();
  bool overlay = model.depth_overlay;
  if (ImGui::Checkbox("Overlay on camera", &overlay)) {
    actions.toggle_depth_overlay = true;
  }
  if (!model.depth_label.empty()) {
    ImGui::TextDisabled("%s", model.depth_label.c_str());
  }
  if (!model.inference_timing.empty()) {
    ImGui::SameLine();
    ImGui::TextDisabled("|  %s", model.inference_timing.c_str());
  }
}

// Relighting engineering controls (Phase 4). Every change is applied on the
// next frame; nothing here waits for the camera or the depth worker.
void ControlWindow::BuildLighting(const ControlsModel& model, ControlsActions& actions) {
  ImGui::Separator();
  RelightControls c = model.relight;
  ImGui::TextUnformatted("Relighting:");
  ImGui::SameLine();
  ImGui::TextDisabled("%s", model.relight_label.c_str());
  bool changed = ImGui::Checkbox("Bypass (A/B)", &c.bypass);
  ImGui::SameLine();
  ImGui::TextDisabled("| Temporal:");
  ImGui::SameLine();
  changed |= ImGui::Checkbox("A: align", &c.align);
  ImGui::SameLine();
  changed |= ImGui::Checkbox("B: EMA", &c.ema);
  ImGui::SameLine();
  changed |= ImGui::Checkbox("C: crossfade", &c.crossfade);
  ImGui::SameLine();
  ImGui::SetNextItemWidth(200.0f);
  const int current = std::clamp(c.view, 0, static_cast<int>(std::size(kRelightViews)) - 1);
  if (ImGui::BeginCombo("Right view", kRelightViews[current].label)) {
    for (int i = 0; i < static_cast<int>(std::size(kRelightViews)); ++i) {
      if (ImGui::Selectable(kRelightViews[i].label, i == current) && i != current) {
        c.view = i;
        changed = true;
      }
    }
    ImGui::EndCombo();
  }
  if (ImGui::TreeNodeEx("Light", ImGuiTreeNodeFlags_DefaultOpen)) {
    LightSettings& l = c.light;
    const float width = 220.0f;
    ImGui::SetNextItemWidth(width);
    changed |= ImGui::SliderFloat("X (left-right)", &l.x, 0.0f, 1.0f, "%.2f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(width);
    changed |= ImGui::SliderFloat("Y (top-bottom)", &l.y, 0.0f, 1.0f, "%.2f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(width);
    changed |= ImGui::SliderFloat("Z (distance)", &l.z, 0.2f, 4.0f, "%.2f");
    ImGui::SetNextItemWidth(width);
    changed |= ImGui::SliderFloat("Diffuse intensity", &l.intensity, 0.0f, 1.5f, "%.2f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(width);
    changed |= ImGui::SliderFloat("Ambient", &l.ambient, 0.2f, 1.2f, "%.2f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(width);
    changed |= ImGui::SliderFloat("Specular", &l.specular, 0.0f, 0.4f, "%.2f");
    ImGui::SetNextItemWidth(width);
    changed |= ImGui::SliderFloat("Tint (cool - warm)", &l.tint, -1.0f, 1.0f, "%.2f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(width);
    changed |= ImGui::SliderFloat("Relief (heuristic)", &l.relief, 0.0f, 1.5f, "%.2f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(width);
    changed |= ImGui::SliderFloat("Depth falloff", &l.depth_falloff, 0.0f, 1.0f, "%.2f");
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("How much depth changes the light's distance. 0 avoids a seam at a lagging silhouette.");
    }
    ImGui::TreePop();
  }
  if (changed) {
    c.light = c.light.Clamped();
    actions.relight = c;
  }
}

void ControlWindow::HandleLightDrag(const ControlsModel& model, const PreviewGeometry& base, ControlsActions& actions) {
  if (base.width <= 0.0f || !model.has_image) {
    dragging_light_ = false;
    return;
  }
  PreviewGeometry preview = base;
  preview.image = model.image_rect;
  if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
    dragging_light_ = true;
  }
  if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
    dragging_light_ = false;
  }
  RelightControls c = actions.relight.value_or(model.relight);
  if (dragging_light_) {
    const ImVec2 mouse = ImGui::GetMousePos();
    const ImageUv uv = PreviewToImageUv(mouse.x, mouse.y, preview);
    if (uv.u != c.light.x || uv.v != c.light.y) {
      c.light.x = uv.u;
      c.light.y = uv.v;
      actions.relight = c;
    }
  }
  // The marker: over the preview in this window only, never in the output.
  float mx = 0.0f, my = 0.0f;
  ImageUvToPreview({c.light.x, c.light.y}, preview, mx, my);
  ImDrawList* draw = ImGui::GetWindowDrawList();
  const ImU32 colour = dragging_light_ ? IM_COL32(255, 220, 90, 255) : IM_COL32(255, 255, 255, 200);
  draw->AddCircle({mx, my}, 9.0f, IM_COL32(0, 0, 0, 160), 0, 4.0f);
  draw->AddCircle({mx, my}, 9.0f, colour, 0, 2.0f);
  draw->AddLine({mx - 4.0f, my}, {mx + 4.0f, my}, colour, 1.5f);
  draw->AddLine({mx, my - 4.0f}, {mx, my + 4.0f}, colour, 1.5f);
}

// Guided capture: instructions, countdown and progress for local test shots.
// Everything here is drawn in the Controls window only.
void ControlWindow::BuildGuidedCapture(const GuidedCapture& guided, GuidedCapture::Clock::time_point now,
                                       ControlsActions& actions) {
  ImGui::Separator();
  const std::size_t index = guided.current();
  const CaptureStep& step = guided.step(index);
  const ShotStatus status = guided.status(index);
  const bool still = step.kind == CaptureStep::Kind::kStill;
  const bool busy = guided.busy();

  ImGui::TextColored({0.55f, 0.75f, 1.0f, 1.0f}, "Guided capture  |  shot %zu of %zu", index + 1, guided.size());
  ImGui::SameLine();
  ImGui::TextDisabled("(saved locally in %s; nothing is uploaded)", guided.directory().c_str());
  ImGui::BeginDisabled(busy);
  ImGui::SetNextItemWidth(360.0f);
  char label[160];
  std::snprintf(label, sizeof(label), "%02zu  %s  [%s]", index + 1, step.title.c_str(), ToString(status).data());
  if (ImGui::BeginCombo("Shot", label)) {
    for (std::size_t i = 0; i < guided.size(); ++i) {
      std::snprintf(label, sizeof(label), "%02zu  %s  [%s]", i + 1, guided.step(i).title.c_str(),
                    ToString(guided.status(i)).data());
      if (ImGui::Selectable(label, i == index)) {
        actions.guided_select = i;
      }
    }
    ImGui::EndCombo();
  }
  ImGui::SameLine();
  if (ImGui::Button("< Previous") && index > 0) {
    actions.guided_select = index - 1;
  }
  ImGui::SameLine();
  if (ImGui::Button("Next >") && index + 1 < guided.size()) {
    actions.guided_select = index + 1;
  }
  ImGui::EndDisabled();

  ImGui::TextUnformatted(step.title.c_str());
  ImGui::TextWrapped("%s", step.instruction.c_str());

  const bool taken = status == ShotStatus::kSaved || status == ShotStatus::kFailed;
  const char* button = still ? (taken ? "Retake" : "Capture Frame") : (taken ? "Retake burst" : "Start burst");
  ImGui::BeginDisabled(busy);
  if (ImGui::Button(button, {160.0f, 0.0f})) {
    actions.guided_capture = true;
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  switch (status) {
    case ShotStatus::kCountdown: {
      ImGui::PushFont(nullptr, ImGui::GetStyle().FontSizeBase * 1.6f);
      ImGui::TextColored({0.95f, 0.80f, 0.30f, 1.0f}, "Capturing in %.0f ...",
                         std::max(1.0, std::ceil(guided.countdown_remaining(now))));
      ImGui::PopFont();
      break;
    }
    case ShotStatus::kCapturing:
      if (still) {
        ImGui::TextColored({0.95f, 0.80f, 0.30f, 1.0f}, "Capturing ...");
      } else {
        ImGui::TextColored({0.95f, 0.80f, 0.30f, 1.0f}, "Capturing: hold still (%u of %u frames)",
                           guided.burst_progress(), step.frames);
      }
      break;
    case ShotStatus::kWriting: ImGui::TextColored({0.95f, 0.80f, 0.30f, 1.0f}, "Writing to disk ..."); break;
    case ShotStatus::kSaved: ImGui::TextColored({0.35f, 0.85f, 0.45f, 1.0f}, "%s", guided.message(index).c_str()); break;
    case ShotStatus::kFailed: ImGui::TextColored({0.95f, 0.40f, 0.35f, 1.0f}, "%s", guided.message(index).c_str()); break;
    case ShotStatus::kNotTaken: ImGui::TextDisabled("Not taken yet"); break;
  }
  ImGui::Text("Saved on disk: %u of %u stills, %u of %u bursts  |  files written this session: %u",
              guided.saved_stills(), guided.stills_total(), guided.saved_bursts(), guided.bursts_total(),
              guided.files_written());
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
