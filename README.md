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

Phase 0 — Environment and feasibility probes (Dawn textured quad, ONNX Runtime provider probe, AVFoundation camera probe). No application features yet.

## Development platform

Apple Silicon macOS 14+.

## Building

See `docs/build.md`. Dependency versions are in `docs/dependencies.md`, and feasibility results and decisions are in `docs/decisions.md`.

## Roadmap

See `RELUME_PRD_REVISED.md` for the implementation plan.
