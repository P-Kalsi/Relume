# Dependencies

Native dependencies are pinned and fetched by `scripts/bootstrap_deps.sh` into
`third_party/` (git-ignored). Nothing here is committed to the repository.

| Dependency | Version / revision | Source | Integrity | How it is used |
|---|---|---|---|---|
| Dawn (WebGPU) | commit `537308541a072a4b348a6bdd4d54e808f4239a7a` (release `v20261022.205350`) | https://dawn.googlesource.com/dawn | Git commit hash | Built from source (Release, Metal backend only), installed to `third_party/install/dawn`, consumed with `find_package(Dawn)` → `dawn::webgpu_dawn` |
| ONNX Runtime | 1.31.0, `onnxruntime-osx-arm64-1.31.0.tgz` | https://github.com/microsoft/onnxruntime/releases/tag/v1.31.0 | SHA-256 `1031b36dd3bdaa8d976a6130d9cf2a05e9be320b93ae804f8427b91dc3432afc` (matches GitHub's published digest) | Prebuilt arm64 dylib (minos 14.0) with CoreML EP; `cmake/FindOnnxRuntime.cmake` |
| GLFW | 3.4, `glfw-3.4.zip` | https://github.com/glfw/glfw/releases/tag/3.4 | SHA-256 `b5ec004b2712fd08e8861dc271428f048775200a2df719ccf575143ba749a3e9` (computed on first download; GitHub publishes no digest for this asset) | CMake `FetchContent`, windowing for the Dawn probe |

Dawn's own third-party dependencies are fetched by Dawn's
`tools/fetch_dawn_dependencies.py` (`-DDAWN_FETCH_DEPENDENCIES=ON`) at the
revisions pinned in Dawn's `DEPS` file for the commit above.

## Licenses

Dawn (BSD-3-Clause), ONNX Runtime (MIT) and GLFW (zlib) notices must be
included before any binary is distributed (PRD §10). Phase 0 distributes
nothing.
