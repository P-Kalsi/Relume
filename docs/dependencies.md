# Dependencies

Native dependencies are pinned and fetched by `scripts/bootstrap_deps.sh` into
`third_party/` (git-ignored). Nothing here is committed to the repository.

| Dependency | Version / revision | Source | Integrity | How it is used |
|---|---|---|---|---|
| Dawn (WebGPU) | commit `537308541a072a4b348a6bdd4d54e808f4239a7a` (release `v20261022.205350`) | https://dawn.googlesource.com/dawn | Git commit hash | Built from source (Release, Metal backend only), installed to `third_party/install/dawn`, consumed with `find_package(Dawn)` → `dawn::webgpu_dawn` |
| ONNX Runtime | 1.31.0, `onnxruntime-osx-arm64-1.31.0.tgz` | https://github.com/microsoft/onnxruntime/releases/tag/v1.31.0 | SHA-256 `1031b36dd3bdaa8d976a6130d9cf2a05e9be320b93ae804f8427b91dc3432afc` (matches GitHub's published digest) | Prebuilt arm64 dylib (minos 14.0) with CoreML EP; `cmake/FindOnnxRuntime.cmake` |
| GLFW | 3.4, `glfw-3.4.zip` | https://github.com/glfw/glfw/releases/tag/3.4 | SHA-256 `b5ec004b2712fd08e8861dc271428f048775200a2df719ccf575143ba749a3e9` (computed on first download; GitHub publishes no digest for this asset) | CMake `FetchContent`, windowing for the Dawn probe |

## Model tooling (Phase 1)

| Dependency | Pin | Source / integrity |
|---|---|---|
| Python | 3.14.7 in `.venv`; 3.12 is a supported fallback | `scripts/setup_python.sh`. The system Python is never modified. |
| Python packages | Exact pins in `scripts/requirements.txt`; full resolved set in `scripts/requirements-lock.txt` | torch 2.14.1, torchvision 0.29.1, onnx 1.23.2, onnxscript 0.7.2, onnxruntime 1.31.0 (same as C++), numpy 2.5.4, opencv-python-headless 5.0.0.93, pillow 13.0.0, pytest 9.1.1 |
| Depth-Anything-V2 source | commit `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf` | https://github.com/DepthAnything/Depth-Anything-V2 (Apache-2.0), fetched into `third_party/src/depth_anything_v2` |
| Depth-Anything-V2-Small weights | HF revision `03876f8651c73a60fe4c2c48294e09fcb6838fcf` | SHA-256 `715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378` (Apache-2.0) |
| Upstream demo images | 5 files at the same commit | SHA-256 values in `scripts/assets.json`. Used for local validation only; never committed, because their redistribution terms are unstated. |

opencv and torchvision are installed only because upstream `dpt.py` imports
them. Relume's preprocessing does not use OpenCV.

Dawn's own third-party dependencies are fetched by Dawn's
`tools/fetch_dawn_dependencies.py` (`-DDAWN_FETCH_DEPENDENCIES=ON`) at the
revisions pinned in Dawn's `DEPS` file for the commit above.

## Licenses

Dawn (BSD-3-Clause), ONNX Runtime (MIT) and GLFW (zlib) notices must be
included before any binary is distributed (PRD §10). The
Depth-Anything-V2-Small notice is already in
`LICENSES/depth-anything-v2-small.txt`. Nothing is distributed yet.
