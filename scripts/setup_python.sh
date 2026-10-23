#!/usr/bin/env bash
# Create Relume's isolated Python environment for model tooling and fetch the
# pinned model assets. The system Python is never modified.
#
# Usage: scripts/setup_python.sh [--write-lock]
#   RELUME_PYTHON=python3.12 scripts/setup_python.sh   # force an interpreter
#
# Interpreter choice: $RELUME_PYTHON if set, otherwise python3.14, then
# python3.12. Both versions use the same pins (all have cp312/cp314 wheels).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ROOT}/.venv"
LOCK="${ROOT}/scripts/requirements-lock.txt"
UPSTREAM_DIR="${ROOT}/third_party/src/depth_anything_v2"
UPSTREAM_REPO="https://github.com/DepthAnything/Depth-Anything-V2.git"
UPSTREAM_COMMIT="a561b849ebae10a6f5ef49e26c83cbbcd36c71bf" # keep in sync with scripts/assets.json

log() { printf '[setup_python] %s\n' "$*"; }

pick_python() {
  if [[ -n "${RELUME_PYTHON:-}" ]]; then
    echo "${RELUME_PYTHON}"
    return
  fi
  for candidate in python3.14 python3.12; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      echo "${candidate}"
      return
    fi
  done
  echo "error: need python3.14 or python3.12 (or set RELUME_PYTHON)" >&2
  exit 1
}

PYTHON="$(pick_python)"
VERSION="$("${PYTHON}" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
case "${VERSION}" in
  3.12|3.14) ;;
  *) echo "error: ${PYTHON} is Python ${VERSION}; supported: 3.12, 3.14" >&2; exit 1 ;;
esac
log "Using ${PYTHON} (Python ${VERSION})"

if [[ -x "${VENV}/bin/python" ]]; then
  existing="$("${VENV}/bin/python" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
  if [[ "${existing}" != "${VERSION}" ]]; then
    echo "error: ${VENV} uses Python ${existing}, not ${VERSION}. Remove it first: rm -rf .venv" >&2
    exit 1
  fi
else
  log "Creating ${VENV}"
  "${PYTHON}" -m venv "${VENV}"
fi

"${VENV}/bin/python" -m pip install --quiet --upgrade pip
if [[ -f "${LOCK}" && "${1:-}" != "--write-lock" ]]; then
  log "Installing from requirements-lock.txt"
  "${VENV}/bin/python" -m pip install --quiet -r "${LOCK}"
else
  log "Installing from requirements.txt"
  "${VENV}/bin/python" -m pip install --quiet -r "${ROOT}/scripts/requirements.txt"
  if [[ "${1:-}" == "--write-lock" ]]; then
    "${VENV}/bin/python" -m pip freeze --exclude pip > "${LOCK}"
    log "Wrote ${LOCK}"
  fi
fi

if [[ "$(git -C "${UPSTREAM_DIR}" rev-parse HEAD 2>/dev/null || true)" != "${UPSTREAM_COMMIT}" ]]; then
  log "Fetching upstream Depth-Anything-V2 ${UPSTREAM_COMMIT}"
  mkdir -p "${UPSTREAM_DIR}"
  git -C "${UPSTREAM_DIR}" init --quiet
  git -C "${UPSTREAM_DIR}" remote remove origin 2>/dev/null || true
  git -C "${UPSTREAM_DIR}" remote add origin "${UPSTREAM_REPO}"
  git -C "${UPSTREAM_DIR}" fetch --quiet --depth 1 origin "${UPSTREAM_COMMIT}"
  git -C "${UPSTREAM_DIR}" checkout --quiet --detach FETCH_HEAD
fi

log "Fetching and verifying model assets"
"${VENV}/bin/python" "${ROOT}/scripts/fetch_assets.py"

log "Running import smoke test"
"${VENV}/bin/python" "${ROOT}/scripts/smoke_test.py"
