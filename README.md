# Relume

Real-time, on-device AI relighting for video calls.

Relume uses monocular depth estimation and GPU shaders to apply adjustable virtual lighting to a webcam feed.

## Technology

- C++20
- Dawn / WebGPU
- WGSL
- ONNX Runtime
- CoreML on macOS
- CMake

## Current status

- **Phase 0 (done):** environment and feasibility probes for the Dawn textured quad, the ONNX Runtime provider list and the AVFoundation camera.
- **Phase 1 (done):** reproducible Depth Anything V2 Small → fixed 448×448 ONNX export, with a manifest, a preprocessing contract and PyTorch↔ONNX parity tests. See `docs/model.md`.

There are no application features yet.

## Development platform

Apple Silicon macOS 14+.

## Building

See `docs/build.md`. Dependency versions are in `docs/dependencies.md`, and feasibility results and decisions are in `docs/decisions.md`.

## Roadmap

See `RELUME_PRD_REVISED.md` for the implementation plan.
