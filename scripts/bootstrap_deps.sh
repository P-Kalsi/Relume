#!/usr/bin/env bash
# Fetch, verify and build Relume's pinned native dependencies into third_party/.
#
# Usage: scripts/bootstrap_deps.sh [ort|dawn|all]   (default: all)
#
# Everything lands in third_party/ (git-ignored). Re-running skips work that is
# already done. Versions and hashes are recorded in docs/dependencies.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="${ROOT}/third_party"
DOWNLOADS="${TP}/downloads"
INSTALL="${TP}/install"

ORT_VERSION="1.31.0"
ORT_ARCHIVE="onnxruntime-osx-arm64-${ORT_VERSION}.tgz"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_ARCHIVE}"
ORT_SHA256="1031b36dd3bdaa8d976a6130d9cf2a05e9be320b93ae804f8427b91dc3432afc"

DAWN_REPO="https://dawn.googlesource.com/dawn"
DAWN_COMMIT="537308541a072a4b348a6bdd4d54e808f4239a7a" # release v20261022.205350
# Dawn is a large C++ build; cap parallelism so a 16 GB laptop stays usable.
DAWN_JOBS="${DAWN_JOBS:-4}"

# Must match CMAKE_OSX_DEPLOYMENT_TARGET in CMakeLists.txt. ONNX Runtime 1.31.0's
# prebuilt dylib requires macOS 14.0 (vtool minos), so 14.0 is our floor.
MACOS_DEPLOYMENT_TARGET="14.0"

log() { printf '[bootstrap] %s\n' "$*"; }

fetch_ort() {
  local dest="${INSTALL}/onnxruntime"
  if [[ -f "${dest}/include/onnxruntime_cxx_api.h" ]]; then
    log "ONNX Runtime ${ORT_VERSION} already installed at ${dest}"
    return
  fi
  mkdir -p "${DOWNLOADS}"
  local archive="${DOWNLOADS}/${ORT_ARCHIVE}"
  if [[ ! -f "${archive}" ]]; then
    log "Downloading ${ORT_URL}"
    curl --fail --location --silent --show-error -o "${archive}.part" "${ORT_URL}"
    mv "${archive}.part" "${archive}"
  fi
  log "Verifying SHA-256"
  echo "${ORT_SHA256}  ${archive}" | shasum -a 256 -c -
  local staging="${TP}/staging/onnxruntime"
  rm -rf "${staging}" && mkdir -p "${staging}"
  # The archive contains a single top-level onnxruntime-osx-arm64-<ver>/ folder.
  tar -xzf "${archive}" -C "${staging}"
  rm -rf "${dest}" && mkdir -p "${INSTALL}"
  mv "${staging}/onnxruntime-osx-arm64-${ORT_VERSION}" "${dest}"
  rm -rf "${staging}"
  log "ONNX Runtime installed at ${dest}"
}

build_dawn() {
  local src="${TP}/src/dawn"
  local build="${TP}/build/dawn"
  local dest="${INSTALL}/dawn"
  if [[ -f "${dest}/lib/cmake/Dawn/DawnConfig.cmake" ]]; then
    log "Dawn already installed at ${dest}"
    return
  fi
  if [[ ! -d "${src}/.git" ]]; then
    log "Fetching Dawn ${DAWN_COMMIT}"
    mkdir -p "${src}"
    git -C "${src}" init --quiet
    git -C "${src}" remote add origin "${DAWN_REPO}"
  fi
  if [[ "$(git -C "${src}" rev-parse HEAD 2>/dev/null || true)" != "${DAWN_COMMIT}" ]]; then
    git -C "${src}" fetch --depth 1 origin "${DAWN_COMMIT}"
    git -C "${src}" checkout --quiet --detach FETCH_HEAD
  fi

  # Documented route: docs/quickstart-cmake.md. Only the Metal backend is
  # built; samples, tests, tools and other backends are disabled to save time
  # and disk space.
  log "Configuring Dawn"
  cmake -S "${src}" -B "${build}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOS_DEPLOYMENT_TARGET}" \
    -DDAWN_FETCH_DEPENDENCIES=ON \
    -DDAWN_ENABLE_INSTALL=ON \
    -DDAWN_BUILD_SAMPLES=OFF \
    -DDAWN_BUILD_TESTS=OFF \
    -DDAWN_BUILD_BENCHMARKS=OFF \
    -DDAWN_BUILD_PROTOBUF=OFF \
    -DDAWN_USE_GLFW=OFF \
    -DDAWN_ENABLE_METAL=ON \
    -DDAWN_ENABLE_NULL=OFF \
    -DDAWN_ENABLE_VULKAN=OFF \
    -DDAWN_ENABLE_DESKTOP_GL=OFF \
    -DDAWN_ENABLE_OPENGLES=OFF \
    -DDAWN_ENABLE_SWIFTSHADER=OFF \
    -DTINT_BUILD_CMD_TOOLS=OFF \
    -DTINT_BUILD_TESTS=OFF \
    -DTINT_BUILD_BENCHMARKS=OFF

  log "Building Dawn with ${DAWN_JOBS} jobs (this takes a while)"
  cmake --build "${build}" --parallel "${DAWN_JOBS}"
  cmake --install "${build}" --prefix "${dest}"
  log "Dawn installed at ${dest}"
}

case "${1:-all}" in
  ort) fetch_ort ;;
  dawn) build_dawn ;;
  all) fetch_ort; build_dawn ;;
  *) echo "usage: $0 [ort|dawn|all]" >&2; exit 2 ;;
esac
