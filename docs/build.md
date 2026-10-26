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

## 1b. Model tooling: Python environment, checkpoint and ONNX export (once)

```sh
scripts/setup_python.sh                        # .venv (python3.14, else python3.12), pinned packages,
                                               # upstream source, checkpoint and demo images (SHA-256 checked),
                                               # import smoke test
RELUME_PYTHON=python3.12 scripts/setup_python.sh   # force an interpreter
.venv/bin/python scripts/export.py             # -> models/depth_anything_v2_small_448.onnx + manifest
.venv/bin/python scripts/check_parity.py       # PyTorch vs ONNX Runtime CPU; exits non-zero on failure
.venv/bin/python scripts/compare_preprocessing.py   # informational letterbox/stretch/upstream comparison
.venv/bin/python -m pytest tests/python -v
```

Downloads: about 100 MB of checkpoint, 2 MB of demo images, and Python wheels
(torch is the largest). The exported ONNX file is about 100 MB. None of these
are committed.

**Fresh-clone check:** after the steps above, `git status` must show the
manifest unchanged except for its `environment` block, and `check_parity.py`
must pass.

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
- `python_preprocess` and `python_model_artifacts`: the pytest suites. They are registered
  only when `.venv` exists (`RELUME_PYTHON_TESTS`). The artifact suite
  fails, with instructions, if `scripts/export.py` has not been run.

CTest also runs the native GoogleTest suites (`tests/cpp/`). They cover
frame queue, pixel conversion, format negotiation, layout, status and file
source; offscreen Dawn tests compare GPU output against the CPU reference.
It also runs three `app_headless_smoke_*` tests: the real pipeline from the
file source, checked pixel by pixel. None of these needs a camera.

Window test (opens real windows for about a minute; needs a logged-in GUI
session; excluded from the presets above):

```sh
ctest --preset display      # app_window_occlusion: cover/resize/minimize, fails if presentation blocks
                            # app_guided_capture_self_test: scripted guided still + burst (file source)
```

## 3. Run Relume

```sh
# Camera (launch the bundle with `open` so macOS attributes the camera
# prompt to Relume; approve it). Logs go to relume.log.
open -W -n --stdout "$PWD/relume.log" --stderr "$PWD/relume.log" \
  build/debug/src/Relume.app --args --seconds 60 && cat relume.log

# Useful options (pass after --args):
#   --pixel-format nv12|bgra   capture output format (default nv12)
#   --camera <unique-id>       a specific camera (IDs are listed in the log)
#   --seconds <S> / --frames <N> / --stats <interval-s>
#   --presenter layer|surface  window presentation (default layer; surface is the
#                              CAMetalLayer fallback, which can freeze ~1 s on cover)
#   --trace                    log the phases of render-loop iterations over 50 ms

# Local webcam test frames for the preprocessing comparison (opt-in; frames
# stay on this Mac in the git-ignored third_party/assets/local_frames/).
# Use the Release build: a Debug build converts a 720p frame in about 50 ms,
# too slow for 30 fps bursts (Release: about 5 ms).
open -W -n --stdout "$PWD/relume.log" --stderr "$PWD/relume.log" \
  build/release/src/Relume.app --args --guided-capture && cat relume.log
#   Guided: 12 stills and 2 bursts of 20 consecutive frames, taken one at a time
#   from the Controls window (Capture Frame, 3 s countdown, Retake). A shot counts
#   as saved only after the file is written; a broken burst keeps nothing.
#   --save-frames <N> [--save-interval S] [--save-delay S] saves on a timer instead.

# File source (no camera, no permission; can be run directly)
build/debug/src/Relume.app/Contents/MacOS/Relume --source file --pixel-format nv12
build/debug/src/Relume.app/Contents/MacOS/Relume --headless --source file --pattern 640x480 --pixel-format nv12
```

Two windows open: **Relume Controls** (camera choice, status, Output toggle
and preview) and **Relume Output** (video only, for OBS window capture in
Phase 7). Closing Controls quits; closing Output only hides it.

Relume.app is signed ad-hoc by default, so macOS asks for camera permission
again after each rebuild. See `docs/signing.md` for an optional stable local
identity (`-DRELUME_CODESIGN_IDENTITY=...`).

## 4. Manual probes

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
