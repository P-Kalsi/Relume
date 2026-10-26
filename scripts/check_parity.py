"""PyTorch (reference) vs ONNX Runtime CPU parity for the exported model.

Usage:
  .venv/bin/python scripts/check_parity.py              # check against committed tolerances
  .venv/bin/python scripts/check_parity.py --calibrate  # one-time: propose working tolerances

Every fixture is preprocessed ONCE and the same float32 tensor is given to
both runtimes, for both the letterbox and stretch methods, so parity measures
the export, not the preprocessing choice. Exits non-zero if any metric exceeds
its working tolerance or hard ceiling, or if any output is non-finite.

Writes docs/parity_results.json and, for the demo images, display renders in
build/parity/ (git-ignored).
"""

from __future__ import annotations

import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from relume_model import preprocess as pp  # noqa: E402
from relume_model.fixtures import GENERATED  # noqa: E402
from relume_model.images import demo_images, depth_stats, load_rgb, render_depth, save_side_by_side  # noqa: E402
from relume_model.tolerances import HARD_CEILINGS, WORKING_TOLERANCES  # noqa: E402
from relume_model.upstream import load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH = ROOT / "models" / "depth_anything_v2_small_448.onnx"
RESULTS_PATH = ROOT / "docs" / "parity_results.json"
RENDER_DIR = ROOT / "build" / "parity"


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict:
    ref = reference.astype(np.float64)
    cand = candidate.astype(np.float64)
    finite = bool(np.isfinite(ref).all() and np.isfinite(cand).all())
    diff = np.abs(cand - ref)
    value_range = float(ref.max() - ref.min())
    pearson = float(np.corrcoef(ref.ravel(), cand.ravel())[0, 1]) if value_range > 0 else float("nan")
    return {
        "finite": finite,
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "p95_abs": float(np.percentile(diff, 95)),
        "range": value_range,
        "rel_max": float(diff.max() / value_range) if value_range > 0 else float("inf"),
        "mean_abs_over_range": float(diff.mean() / value_range) if value_range > 0 else float("inf"),
        "pearson": pearson,
        "argmax_abs_yx": [int(v) for v in np.unravel_index(int(diff.argmax()), diff.shape)[1:]],
    }


def round_up_1sf(x: float) -> float:
    if x <= 0:
        return 0.0
    exponent = math.floor(math.log10(x))
    return math.ceil(x / 10**exponent) * 10**exponent


def latency(fn, warmup: int = 5, runs: int = 50) -> dict:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return {"runs": runs, "warmup": warmup, "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95))}


def main() -> int:
    calibrate = "--calibrate" in sys.argv
    if not calibrate and WORKING_TOLERANCES is None:
        print("FAIL: working tolerances are not calibrated; run with --calibrate once and commit them")
        return 1
    if not ONNX_PATH.exists():
        print(f"FAIL: {ONNX_PATH} missing; run scripts/export.py")
        return 1

    import onnxruntime as ort
    import torch

    model = load_model()
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    def run_torch(tensor: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return model(torch.from_numpy(tensor)).numpy()

    def run_ort(tensor: np.ndarray) -> np.ndarray:
        return session.run(["depth"], {"image": tensor})[0]

    inputs = [(name, "generated", None, factory()) for name, factory in GENERATED.items()]
    inputs += [(name, "upstream_demo", digest, load_rgb(path)) for name, path, digest in demo_images()]

    rows = []
    for name, kind, digest, rgb in inputs:
        for method in pp.METHODS:
            tensor, transform = pp.preprocess(rgb, method)  # one tensor for both runtimes
            ref = run_torch(tensor)
            out = run_ort(tensor)
            metrics = compare(ref, out)
            row = {"fixture": name, "kind": kind, "method": method, "sha256": digest,
                   "size": [int(rgb.shape[1]), int(rgb.shape[0])], **metrics}
            if kind == "upstream_demo" and method == "letterbox":
                source_depth = pp.depth_to_source(out, transform)
                row["onnx_depth_stats"] = depth_stats(out[0, transform.pad_y:transform.pad_y + transform.new_h,
                                                          transform.pad_x:transform.pad_x + transform.new_w])
                save_side_by_side(RENDER_DIR / f"{Path(name).stem}_letterbox.png",
                                  [rgb, render_depth(source_depth)])
            rows.append(row)

    worst = {key: max(r[key] for r in rows) for key in ("max_abs", "rel_max", "mean_abs_over_range")}
    all_finite = all(r["finite"] for r in rows)

    print(f"{'fixture':28} {'method':9} {'max_abs':>10} {'mean_abs':>10} {'rel_max':>10} {'mean/rng':>10} {'pearson':>9}")
    for r in rows:
        print(f"{r['fixture']:28} {r['method']:9} {r['max_abs']:10.3e} {r['mean_abs']:10.3e} "
              f"{r['rel_max']:10.3e} {r['mean_abs_over_range']:10.3e} {r['pearson']:9.6f}")
    print(f"worst: {worst}  all finite: {all_finite}")

    failures = []
    if not all_finite:
        failures.append("non-finite output")
    for key, ceiling in HARD_CEILINGS.items():
        if worst[key] > ceiling:
            failures.append(f"{key} {worst[key]:.3e} exceeds hard ceiling {ceiling:.0e}")

    if calibrate:
        proposed = {
            "max_abs": round_up_1sf(10 * worst["max_abs"]),
            "rel_max": min(round_up_1sf(10 * worst["rel_max"]), HARD_CEILINGS["rel_max"]),
            "mean_abs_over_range": min(round_up_1sf(10 * worst["mean_abs_over_range"]),
                                       HARD_CEILINGS["mean_abs_over_range"]),
        }
        print(f"PROPOSED WORKING_TOLERANCES = {proposed}")
        if failures:
            print("FAIL: " + "; ".join(failures))
            return 1
        print("Calibration run within hard ceilings. Commit the proposed values to tolerances.py.")
        return 0

    for key, limit in WORKING_TOLERANCES.items():
        over = [r for r in rows if r[key] > limit]
        for r in over:
            failures.append(f"{r['fixture']}/{r['method']}: {key} {r[key]:.3e} > tolerance {limit:.1e} "
                            f"(max |diff| at y,x={r['argmax_abs_yx']})")

    torch_threads = torch.get_num_threads()
    bench_tensor, _ = pp.preprocess(GENERATED["smooth_1280x720"]())
    results = {
        "description": "PyTorch CPU FP32 (reference) vs ONNX Runtime CPUExecutionProvider FP32 on identical "
                       "preprocessed tensors",
        "tolerances": {"hard_ceilings": HARD_CEILINGS, "working": WORKING_TOLERANCES},
        "worst": worst,
        "passed": not failures,
        "fixtures": rows,
        "latency_cpu_informational": {
            "input": f"smooth_1280x720 {pp.DEFAULT_METHOD} tensor, 1x3x448x448",
            "pytorch_cpu": latency(lambda: run_torch(bench_tensor)),
            "onnxruntime_cpu": latency(lambda: run_ort(bench_tensor)),
            "torch_threads": torch_threads,
            "ort_intra_op_threads": "ORT default (0 = automatic)",
            "note": "CPU only; CoreML is not measured in Phase 1. Laptop timing, not a controlled benchmark.",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
            "os": platform.platform(),
            "machine": platform.machine(),
            "cpu": platform.processor() or "unknown",
        },
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")
    lat = results["latency_cpu_informational"]
    print(f"latency p50/p95 ms: torch {lat['pytorch_cpu']['p50_ms']:.1f}/{lat['pytorch_cpu']['p95_ms']:.1f}, "
          f"ORT {lat['onnxruntime_cpu']['p50_ms']:.1f}/{lat['onnxruntime_cpu']['p95_ms']:.1f}")
    print(f"wrote {RESULTS_PATH.relative_to(ROOT)}; renders in {RENDER_DIR.relative_to(ROOT)}/")

    if failures:
        print("PARITY FAIL:\n  " + "\n  ".join(failures))
        return 1
    print("PARITY PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
