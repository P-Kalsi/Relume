# Building Relume

## Requirements

- Apple Silicon Mac, macOS 14.0 or later
- Command Line Tools 16 or later (Apple clang 16+). Apple clang 15 cannot
  compile the pinned Dawn revision; see `docs/decisions.md`.
- CMake 3.25+ and Ninja
- Python 3 and Git (used by Dawn's dependency fetcher)
- About 2 GB of free disk space for `third_party/`

## 1. Fetch and build dependencies (once)

```sh
scripts/bootstrap_deps.sh          # ONNX Runtime + Dawn
scripts/bootstrap_deps.sh ort      # only ONNX Runtime
scripts/bootstrap_deps.sh dawn     # only Dawn; DAWN_JOBS=4 by default
```

The Dawn build compiles about 940 files and is the slow step. Re-running the
script skips anything already installed.

## 2. Configure, build and test

```sh
cmake --preset debug   && cmake --build --preset debug   && ctest --preset debug
cmake --preset release && cmake --build --preset release && ctest --preset release
cmake --preset asan    && cmake --build --preset asan    && ctest --preset asan   # ASan + UBSan
cmake --preset tsan    && cmake --build --preset tsan    && ctest --preset tsan   # optional
```

CTest runs:

- `ort_provider_probe`: lists ORT execution providers. It fails if the CPU EP
  is missing or the CoreML EP cannot be registered.
- `dawn_offscreen_quad`: renders the textured quad offscreen and checks the
  pixels.

## 3. Manual probes

```sh
# Dawn window with the textured quad (close the window to exit)
./build/debug/probes/dawn_hello/relume_dawn_hello

# Camera permission + capture. Launch with `open` so macOS asks for permission
# on behalf of the app; approve the prompt when it appears.
open -W -n --stdout "$PWD/camera.log" --stderr "$PWD/camera.log" \
  build/debug/probes/camera_probe/RelumeCameraProbe.app && cat camera.log
```

If camera access was denied earlier, re-enable **Relume Camera Probe** in
System Settings → Privacy & Security → Camera.
