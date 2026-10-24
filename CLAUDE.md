# Relume — instructions for Claude Code

Relume is a native macOS (Apple Silicon) app for live, on-device relighting of webcam video, guided by estimated depth.
- **Stack:** C++20, CMake + Ninja, Dawn/WebGPU + WGSL, ONNX Runtime C++ (CPU, then CoreML EP), AVFoundation in Objective-C++ (`.mm`), GLFW + Dear ImGui.
- **Model:** Depth Anything V2 **Small** exported to a fixed 448×448 ONNX model by Python tooling.
- **Output:** reaches video calls through OBS window capture and the OBS Virtual Camera.

## Source of truth
- **`RELUME_PRD_REVISED.md`** defines the requirements, phases and acceptance criteria. **Never modify it** without explicit approval.
- `docs/SESSION_HANDOFF.md`: current state and next steps. Read it first in a new session.
- `docs/decisions.md`: decisions and feasibility log, with toolchain pitfalls.
- `docs/model.md`: model provenance, preprocessing contract, parity results.
- `docs/build.md`: exact build commands. `docs/dependencies.md`: pinned versions and hashes.

## Working rules
- Implement **one PRD phase at a time**, in order. Don't add future-phase features. Stop at the end of a phase and report: files changed, commands, real results, risks, next step.
- **Never claim something works without running it.** Report actual command output. Untested means "not verified", and failures are reported as failures. CoreML being listed or registered is **not** proof that CoreML executes a model.
- **Don't change major dependencies, pinned versions, the toolchain or the stack without approval.** That covers Dawn, ONNX Runtime, GLFW, the model and the Python pins. If something is blocked, report the command, the error, the likely cause and a recommended fix, then stop.
- Never relax test tolerances to make tests pass. The hard ceilings in `scripts/relume_model/tolerances.py` are fixed.
- Never commit large or external assets: checkpoints, `.onnx` files, upstream demo images or webcam frames. `third_party/`, `models/*` (except the README and manifests), `.venv/` and `build/` are git-ignored.
- Privacy: no network calls or telemetry at runtime, and raw frames are not saved by default. Never bypass macOS privacy prompts (TCC). Ask the user to approve camera access when it's needed.

## Git
- Make small, meaningful **Conventional Commits**, and only for verified changes.
- Use the user's configured Git identity. **No `Co-Authored-By: Claude` trailers or other AI attribution.**
- **Never push** without explicit approval. Never rewrite published history.

## Code conventions
- C++20 with no compiler extensions. RAII owns every resource (wgpu handles, `Ort::` objects, ObjC under ARC). No raw owning pointers, ad-hoc globals or detached threads. No exceptions may escape callbacks or threads.
- Platform-specific code lives only in `.mm` files behind portable C++ interfaces, so Windows support can come later.
- Our own targets use `relume_set_warnings()` (`cmake/Warnings.cmake`), and new code must build with no warnings.
- Python tooling lives in `scripts/` (with the shared package `scripts/relume_model/`) and runs only in `.venv`. The preprocessing contract has **one** implementation, `scripts/relume_model/preprocess.py`. C++ must match `tests/python/golden/preprocess_golden.json`.
- Match the surrounding code's style and comment density.
- Windows present only the fixed 1280×720 `OutputTexture`, through `WindowPresenter`. On macOS the default `LayerPresenter` (IOSurface `CALayer.contents`) must never block. Don't add `wgpu::Surface::GetCurrentTexture` calls to the main loop (it can block for 1 s when a window is covered). Reuse IOSurfaces only through `CompositorReuseTracker`. See `docs/architecture.md`.

## Build and test
Toolchain: macOS 14+ (minimum deployment target 14.0), **Apple clang 16+** (Command Line Tools 16.2). Apple clang 15 cannot build the pinned Dawn.
```sh
scripts/bootstrap_deps.sh                  # once: ONNX Runtime 1.31.0 + build pinned Dawn (Metal only)
scripts/setup_python.sh                    # once: .venv (3.14, else 3.12), checkpoint + assets (SHA-256 checked)
.venv/bin/python scripts/export.py         # ONNX model + manifest (needed by the CTest python_model_artifacts test)
cmake --preset debug && cmake --build --preset debug && ctest --preset debug
# other presets: release, asan (ASan+UBSan), tsan (optional)
ctest --preset display                     # window occlusion test: opens windows (~1 min), needs a GUI session; run after presentation changes
.venv/bin/python scripts/check_parity.py   # PyTorch vs ONNX Runtime parity; non-zero exit on failure
.venv/bin/python -m pytest tests/python -v
```
The camera probe needs a person to approve the permission prompt. Launch it with `open` (see `docs/build.md`), never as a raw binary.
