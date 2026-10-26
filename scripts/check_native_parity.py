"""Native (C++) pipeline parity against the Python reference.

Usage:
  .venv/bin/python scripts/check_native_parity.py [--cli build/release/tools/relume_depth/relume_depth]

For every fixture (6 generated + 5 pinned demo images) and both methods
(stretch, the default, and letterbox):
  1. C++ preprocessing vs Python preprocessing: the transform must be equal
     and every tensor value within 1e-6 (the preprocessing contract).
  2. C++ ONNX Runtime CPU vs Python ONNX Runtime CPU, on the identical Python
     tensor: within WORKING_TOLERANCES.
  3. C++ end to end (C++ preprocessing + C++ ONNX Runtime) vs PyTorch on the
     Python tensor: within WORKING_TOLERANCES.
It also checks that the tensor names and shapes the C++ session reports match
the model manifest. Tolerances come from relume_model/tolerances.py and are
never changed here. Exits non-zero on any failure.

Writes docs/native_parity_results.json. Working files go to
build/native_parity/ (git-ignored). Local webcam frames are not used.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_parity import compare  # noqa: E402
from relume_model import preprocess as pp  # noqa: E402
from relume_model.fixtures import GENERATED  # noqa: E402
from relume_model.images import demo_images, load_rgb  # noqa: E402
from relume_model.tolerances import HARD_CEILINGS, WORKING_TOLERANCES  # noqa: E402
from relume_model.upstream import load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH = ROOT / "models" / "depth_anything_v2_small_448.onnx"
MANIFEST_PATH = ROOT / "models" / "depth_anything_v2_small_448.manifest.json"
RESULTS_PATH = ROOT / "docs" / "native_parity_results.json"
WORK_DIR = ROOT / "build" / "native_parity"
DEFAULT_CLI = ROOT / "build" / "release" / "tools" / "relume_depth" / "relume_depth"
PREPROCESS_TOLERANCE = 1e-6  # the contract (docs/model.md)


def write_ppm(path: Path, rgb: np.ndarray) -> None:
    h, w = rgb.shape[:2]
    path.write_bytes(f"P6\n{w} {h}\n255\n".encode() + np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())


def run_cli(cli: Path, *args: str) -> subprocess.CompletedProcess:
    result = subprocess.run([str(cli), *args, "--model", str(ONNX_PATH)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{cli.name} {' '.join(args)} failed ({result.returncode}): {result.stderr.strip()}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH,
                        help="where to write the JSON results (CTest writes into the build tree)")
    parser.add_argument("--work-dir", type=Path, default=WORK_DIR, help="PPM and .npy working files")
    options = parser.parse_args()
    work_dir = options.work_dir
    if not options.cli.exists():
        print(f"FAIL: {options.cli} missing; build it with: cmake --build --preset release")
        return 1
    if not ONNX_PATH.exists():
        print(f"FAIL: {ONNX_PATH} missing; run scripts/export.py")
        return 1

    import onnxruntime as ort
    import torch

    work_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    # Signature reported by the C++ session vs the manifest.
    manifest = json.loads(MANIFEST_PATH.read_text())
    probe = run_cli(options.cli, "bench", "--warmup", "0", "--runs", "1")
    match = re.search(r"signature: (\S+) (\S+) (\S+) -> (\S+) (\S+) (\S+)", probe.stderr)
    native_signature = None
    if not match:
        failures.append("relume_depth did not print a signature")
    else:
        native_signature = {
            "inputs": [{"name": match[1], "dtype": match[2], "shape": json.loads(match[3])}],
            "outputs": [{"name": match[4], "dtype": match[5], "shape": json.loads(match[6])}],
        }
        expected = {"inputs": manifest["onnx"]["inputs"], "outputs": manifest["onnx"]["outputs"]}
        if native_signature != expected:
            failures.append(f"C++ signature {native_signature} != manifest {expected}")
    print(f"C++ session signature: {native_signature}")

    model = load_model()
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    inputs = [(name, "generated", factory()) for name, factory in GENERATED.items()]
    inputs += [(name, "upstream_demo", load_rgb(path)) for name, path, _ in demo_images()]

    rows = []
    for name, kind, rgb in inputs:
        stem = Path(name).stem
        image_path = work_dir / f"{stem}.ppm"
        write_ppm(image_path, rgb)
        for method in pp.METHODS:
            tensor_py, transform_py = pp.preprocess(rgb, method)
            prefix = work_dir / f"{stem}_{method}"

            # 1 + 3: C++ end to end, also writing its tensor.
            out = run_cli(options.cli, "run", "--image", str(image_path), "--method", method,
                          "--out-tensor", f"{prefix}_tensor.npy", "--out-depth", f"{prefix}_depth_e2e.npy")
            report = json.loads(out.stdout.strip().splitlines()[-1])
            tensor_cpp = np.load(f"{prefix}_tensor.npy")
            depth_e2e = np.load(f"{prefix}_depth_e2e.npy")
            transform_ok = report["transform"] == transform_py.to_dict()
            tensor_diff = float(np.abs(tensor_cpp.astype(np.float64) - tensor_py).max())

            # 2: C++ ONNX Runtime on the identical Python tensor.
            np.save(f"{prefix}_tensor_py.npy", tensor_py)
            run_cli(options.cli, "run-tensor", "--tensor", f"{prefix}_tensor_py.npy",
                    "--out-depth", f"{prefix}_depth_ort.npy")
            depth_cpp_ort = np.load(f"{prefix}_depth_ort.npy")
            depth_py_ort = session.run(["depth"], {"image": tensor_py})[0]
            with torch.inference_mode():
                depth_torch = model(torch.from_numpy(tensor_py)).numpy()

            row = {
                "fixture": name, "kind": kind, "method": method, "size": [int(rgb.shape[1]), int(rgb.shape[0])],
                "transform_equal": transform_ok,
                "preprocess_max_abs": tensor_diff,
                "cpp_ort_vs_python_ort": compare(depth_py_ort, depth_cpp_ort),
                "cpp_end_to_end_vs_pytorch": compare(depth_torch, depth_e2e),
            }
            rows.append(row)

            tag = f"{name}/{method}"
            if not transform_ok:
                failures.append(f"{tag}: transform {report['transform']} != {transform_py.to_dict()}")
            if not tensor_diff <= PREPROCESS_TOLERANCE:
                failures.append(f"{tag}: preprocessing max |diff| {tensor_diff:.3e} > {PREPROCESS_TOLERANCE:.0e}")
            for check in ("cpp_ort_vs_python_ort", "cpp_end_to_end_vs_pytorch"):
                metrics = row[check]
                if not metrics["finite"]:
                    failures.append(f"{tag}: {check} non-finite output")
                for key, limit in WORKING_TOLERANCES.items():
                    if metrics[key] > limit:
                        failures.append(f"{tag}: {check} {key} {metrics[key]:.3e} > tolerance {limit:.1e}")
                for key, ceiling in HARD_CEILINGS.items():
                    if metrics[key] > ceiling:
                        failures.append(f"{tag}: {check} {key} {metrics[key]:.3e} > hard ceiling {ceiling:.0e}")

    print(f"{'fixture':28} {'method':9} {'prep max':>9} {'ort max_abs':>11} {'ort rel':>9} "
          f"{'e2e max_abs':>11} {'e2e rel':>9} {'e2e mean/rng':>12}")
    for r in rows:
        a, b = r["cpp_ort_vs_python_ort"], r["cpp_end_to_end_vs_pytorch"]
        print(f"{r['fixture']:28} {r['method']:9} {r['preprocess_max_abs']:9.1e} {a['max_abs']:11.3e} "
              f"{a['rel_max']:9.2e} {b['max_abs']:11.3e} {b['rel_max']:9.2e} {b['mean_abs_over_range']:12.2e}")

    worst = {
        "preprocess_max_abs": max(r["preprocess_max_abs"] for r in rows),
        **{f"{check}.{key}": max(r[check][key] for r in rows)
           for check in ("cpp_ort_vs_python_ort", "cpp_end_to_end_vs_pytorch")
           for key in ("max_abs", "rel_max", "mean_abs_over_range")},
    }
    print(f"worst: {json.dumps(worst)}")

    options.results.parent.mkdir(parents=True, exist_ok=True)
    options.results.write_text(json.dumps({
        "description": "Native C++ pipeline (relume_depth) vs the Python reference: preprocessing, ONNX Runtime "
                       "CPU on identical tensors, and end to end vs PyTorch",
        "tolerances": {"preprocess": PREPROCESS_TOLERANCE, "working": WORKING_TOLERANCES,
                       "hard_ceilings": HARD_CEILINGS},
        "native_signature": native_signature,
        "worst": worst,
        "passed": not failures,
        "failures": failures,
        "fixtures": rows,
        "environment": {"python": platform.python_version(), "onnxruntime_python": ort.__version__,
                        "torch": torch.__version__, "os": platform.platform(), "machine": platform.machine()},
    }, indent=1) + "\n")
    print(f"wrote {options.results}")

    if failures:
        print("NATIVE PARITY FAIL:\n  " + "\n  ".join(failures))
        return 1
    print("NATIVE PARITY PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
