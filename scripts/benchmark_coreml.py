"""CoreML execution-provider matrix: evidence, latency and FP32 parity.

Usage:
  .venv/bin/python scripts/benchmark_coreml.py [--cli build/release/tools/relume_depth/relume_depth]
                                               [--runs 200] [--units CPUOnly,CPUAndGPU,CPUAndNeuralEngine,ALL]

Everything runs through the native relume_depth tool (C++ ONNX Runtime 1.31.0).
For CPU (the reference) and each CoreML compute-unit setting:

  Evidence, recorded per level (none of them is proof of the others):
    (a) ORT assignment: strict mode (session.disable_cpu_ep_fallback, a
        diagnostic only), the CoreML GetCapability line (partitions, nodes
        supported) and the nodes ORT placed on each provider (verbose log).
    (b) Execution at run time: ORT profiling, kernel time per provider per run
        (warm-ups excluded).
    (c) Planned hardware: CoreML's compute plan (ProfileComputePlan), the
        device it plans for each op. This is the planned placement, not proof
        of what ran on each inference.
  Latency: session creation cold (empty compile cache, with verbose logging)
  and warm (cache filled), then Run p50/p95 over --runs runs after 10 warm-ups,
  and the process's peak resident memory.
  Parity against C++ CPU (bit-identical to Python ONNX Runtime CPU, see
  docs/native_parity_results.json) on identical tensors, with the unchanged
  FP32 metrics and tolerances, on: the 11 fixtures x 2 methods, the 12 local
  webcam stills and both 20-frame bursts (stretch, the default). Also error at
  depth edges vs elsewhere and on the subject (nearest 25%), and frame-to-frame
  flicker on the bursts relative to CPU. Output differences and latency are
  never treated as proof of CoreML or Neural Engine execution.

Writes docs/coreml_results.json (metrics only; local frames by name only).
Working files, logs and comparison panels stay in build/coreml_bench/
(git-ignored); local webcam frames never leave this Mac.
"""

from __future__ import annotations

import argparse
import collections
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_parity import compare  # noqa: E402
from relume_model import preprocess as pp  # noqa: E402
from relume_model.fixtures import GENERATED  # noqa: E402
from relume_model.images import demo_images, load_rgb, save_side_by_side  # noqa: E402
from relume_model.tolerances import HARD_CEILINGS, WORKING_TOLERANCES  # noqa: E402
from summarize_ort_profile import summarize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH = ROOT / "models" / "depth_anything_v2_small_448.onnx"
LOCAL_FRAMES = ROOT / "third_party" / "assets" / "local_frames"
RESULTS_PATH = ROOT / "docs" / "coreml_results.json"
WORK = ROOT / "build" / "coreml_bench"
DEFAULT_CLI = ROOT / "build" / "release" / "tools" / "relume_depth" / "relume_depth"
ALL_UNITS = ["CPUOnly", "CPUAndGPU", "CPUAndNeuralEngine", "ALL"]


def run(cmd: list[str], log: Path) -> subprocess.CompletedProcess:
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    log.write_text(f"$ {' '.join(cmd)}\n# exit {result.returncode}, {time.perf_counter() - start:.1f} s\n"
                   f"--- stdout\n{result.stdout}\n--- stderr\n{result.stderr}")
    return result


def bench_json(stdout: str) -> dict | None:
    for line in stdout.splitlines():
        if line.startswith("BENCH_JSON "):
            return json.loads(line[len("BENCH_JSON "):])
    return None


def parse_assignment(log: str) -> dict:
    out: dict = {}
    m = re.search(r"number of partitions supported by CoreML: (\d+) number of nodes in the graph: (\d+) "
                  r"number of nodes supported by CoreML: (\d+)", log)
    if m:
        out["coreml_partitions"], out["graph_nodes"], out["coreml_supported_nodes"] = map(int, m.groups())
    placed: dict[str, dict] = {}
    lines = log.splitlines()
    for i, line in enumerate(lines):
        m = re.search(r"Node\(s\) placed on \[(\w+)\]\. Number of nodes: (\d+)", line)
        if not m:
            continue
        provider, count = m[1], int(m[2])
        # The node list follows as continuation lines: "  <OpType> (<node name>)".
        ops = collections.Counter(
            follow.split("] ")[-1].strip().split(" (")[0] for follow in lines[i + 1:i + 1 + count] if " (" in follow)
        if provider == "CoreMLExecutionProvider":
            ops = collections.Counter({"fused CoreML partition": count})
        placed[provider] = {"nodes": count, "op_types": dict(ops.most_common())}
    out["placed"] = placed
    return out


def parse_compute_plan(log: str) -> dict:
    devices = collections.Counter(re.findall(r"Device Usage: <(ML\w+ComputeDevice)", log))
    per_op = collections.Counter(
        f"{dev}:{op}" for op, dev in re.findall(r"Operation: (\S+), Device Usage: <(ML\w+ComputeDevice)", log))
    return {"ops_planned": sum(devices.values()), "devices": dict(devices),
            "by_op": dict(per_op.most_common())}


def robust_unit(depth: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(depth, [2, 98])
    return (depth - lo) / max(hi - lo, 1e-12)


def detail_metrics(ref: np.ndarray, cand: np.ndarray) -> dict:
    """Edge / subject / elsewhere error, as fractions of the reference range."""
    r = ref[0].astype(np.float64)
    c = cand[0].astype(np.float64)
    value_range = max(float(r.max() - r.min()), 1e-12)
    diff = np.abs(c - r) / value_range
    gy, gx = np.gradient(r)
    grad = np.hypot(gx, gy)
    edge = grad >= np.percentile(grad, 90)
    subject = r >= np.percentile(r, 75)  # larger = nearer
    return {
        "edge_mean_abs_over_range": float(diff[edge].mean()),
        "non_edge_mean_abs_over_range": float(diff[~edge].mean()),
        "subject_mean_abs_over_range": float(diff[subject].mean()),
        "max_abs_over_range": float(diff.max()),
    }


def flicker(depths: list[np.ndarray]) -> float:
    units = [robust_unit(d[0]) for d in depths]
    return float(np.mean([np.abs(b - a).mean() for a, b in zip(units, units[1:])]))


def render(depth: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.round(np.clip((depth - lo) / max(hi - lo, 1e-12), 0, 1) * 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--units", default=",".join(ALL_UNITS))
    options = parser.parse_args()
    if not options.cli.exists() or not ONNX_PATH.exists():
        print(f"FAIL: need {options.cli} (cmake --build --preset release) and {ONNX_PATH} (scripts/export.py)")
        return 1
    cli = str(options.cli)
    model = ["--model", str(ONNX_PATH)]
    WORK.mkdir(parents=True, exist_ok=True)
    logs = WORK / "logs"
    logs.mkdir(exist_ok=True)

    # Inputs: Python-preprocessed tensors (the contract), written once.
    tensors_dir = WORK / "tensors"
    tensors_dir.mkdir(exist_ok=True)
    inputs: list[dict] = []
    for name, factory in GENERATED.items():
        for method in pp.METHODS:
            inputs.append({"name": f"{name}/{method}", "group": "fixtures", "rgb": factory, "method": method})
    for name, path, _ in demo_images():
        for method in pp.METHODS:
            inputs.append({"name": f"{name}/{method}", "group": "fixtures", "rgb": path, "method": method})
    stills = sorted(LOCAL_FRAMES.glob("still_*.ppm")) if LOCAL_FRAMES.exists() else []
    for path in stills:
        inputs.append({"name": path.stem, "group": "local_stills", "rgb": path, "method": pp.DEFAULT_METHOD})
    bursts = {}
    for burst in sorted(p for p in LOCAL_FRAMES.glob("burst_*") if p.is_dir() and p.suffix != ".partial") \
            if LOCAL_FRAMES.exists() else []:
        frames = sorted(burst.glob("*.ppm"))
        bursts[burst.name] = [f"{burst.name}/{f.stem}" for f in frames]
        for f in frames:
            inputs.append({"name": f"{burst.name}/{f.stem}", "group": "local_bursts", "rgb": f,
                           "method": pp.DEFAULT_METHOD})
    for item in inputs:
        item["tensor"] = tensors_dir / (item["name"].replace("/", "__") + ".npy")
        if not item["tensor"].exists():
            src = item["rgb"]
            rgb = src() if callable(src) else load_rgb(src)
            np.save(item["tensor"], pp.preprocess(rgb, item["method"])[0])
    print(f"inputs: {sum(i['group'] == 'fixtures' for i in inputs)} fixture tensors, {len(stills)} local stills, "
          f"{sum(len(v) for v in bursts.values())} burst frames")

    configs = [("cpu", None)] + [("coreml", u) for u in options.units.split(",")]
    results: dict = {}
    outputs: dict[str, dict[str, np.ndarray]] = {}
    for provider, units in configs:
        label = "CPU" if units is None else f"CoreML {units}"
        key = "cpu" if units is None else units
        print(f"\n=== {label}")
        common = model + ["--provider", provider] + (["--units", units] if units else []) + ["--threads", "4"]
        entry: dict = {"label": label}
        cache = WORK / "cache" / key

        if units is not None:
            strict = run([cli, "bench", *common, "--strict", "--warmup", "0", "--runs", "1",
                          "--cache", str(WORK / "cache_strict" / key)], logs / f"{key}_strict.txt")
            failure = next((l for l in strict.stderr.splitlines() if l.startswith("load failed")), "")
            entry["strict_all_nodes_on_coreml"] = {
                "session_created": strict.returncode == 0,
                "error": failure[:400] if failure else "",
                "note": "diagnostic only; not the deployment configuration",
            }
            print(f"  strict: {'session created' if strict.returncode == 0 else 'refused'}")

        # Cold creation and evidence (verbose log, compute plan, profiling).
        shutil.rmtree(cache, ignore_errors=True)
        profile_prefix = WORK / f"profile_{key}"
        for old in WORK.glob(f"profile_{key}_*.json"):
            old.unlink()
        evidence_args = [cli, "bench", *common, "--verbose", "--profile", str(profile_prefix), "--warmup", "2",
                         "--runs", "10"]
        if units is not None:
            evidence_args += ["--compute-plan", "--cache", str(cache)]
        evidence = run(evidence_args, logs / f"{key}_evidence.txt")
        if evidence.returncode != 0:
            entry["error"] = evidence.stderr.strip().splitlines()[-1][:400]
            print(f"  FAILED: {entry['error']}")
            results[key] = entry
            continue
        cold = bench_json(evidence.stdout) or {}
        entry["assignment"] = parse_assignment(evidence.stderr)
        if units is not None:
            entry["compute_plan"] = parse_compute_plan(evidence.stderr)
        profile_files = sorted(WORK.glob(f"profile_{key}_*.json"))
        entry["runtime_profile"] = summarize(profile_files[-1], skip=2) if profile_files else None
        entry["session_create_cold_ms"] = cold.get("load_ms")
        a = entry["assignment"]
        if units is not None:
            entry["coverage_label"] = (
                f"partial CoreML ({a.get('coreml_supported_nodes')} of {a.get('graph_nodes')} nodes, "
                f"{a.get('coreml_partitions')} partitions)"
                if a.get("coreml_supported_nodes") != a.get("graph_nodes") else "all nodes assigned to CoreML")
            print(f"  assignment: {entry['coverage_label']}; compute plan: {entry['compute_plan']['devices']}")
        print(f"  cold session creation {entry['session_create_cold_ms']:.0f} ms; profile: "
              f"{json.dumps(entry['runtime_profile']['providers']) if entry['runtime_profile'] else 'none'}")

        # Warm creation and latency.
        timing_args = [cli, "bench", *common, "--warmup", "10", "--runs", str(options.runs)]
        if units is not None:
            timing_args += ["--cache", str(cache)]
        timing = run(timing_args, logs / f"{key}_timing.txt")
        warm = bench_json(timing.stdout) or {}
        entry["session_create_warm_ms"] = warm.get("load_ms")
        entry["latency"] = {k: warm.get(k) for k in ("runs", "warmup", "run_p50_ms", "run_p95_ms", "run_min_ms",
                                                      "run_max_ms", "preprocess_p50_ms", "total_p50_ms",
                                                      "total_p95_ms")}
        entry["peak_resident_mib"] = warm.get("peak_resident_mib")
        print(f"  warm creation {warm.get('load_ms', 0):.0f} ms; run p50 {warm.get('run_p50_ms', 0):.1f} "
              f"p95 {warm.get('run_p95_ms', 0):.1f} ms; peak RSS {warm.get('peak_resident_mib', 0):.0f} MiB")

        # Outputs on every input, one session.
        out_dir = WORK / "outputs" / key
        out_dir.mkdir(parents=True, exist_ok=True)
        jobs = WORK / f"jobs_{key}.txt"
        jobs.write_text("".join(f"{i['tensor']} {out_dir / i['tensor'].name}\n" for i in inputs))
        outs_args = [cli, "run-tensors", *common, "--list", str(jobs)]
        if units is not None:
            outs_args += ["--cache", str(cache)]
        outs = run(outs_args, logs / f"{key}_outputs.txt")
        if outs.returncode != 0:
            entry["error"] = outs.stderr.strip()[-400:]
            results[key] = entry
            continue
        outputs[key] = {i["name"]: np.load(out_dir / i["tensor"].name) for i in inputs}
        results[key] = entry

    # Parity against C++ CPU, unchanged FP32 tolerances.
    cpu = outputs.get("cpu")
    panels = WORK / "panels"
    for key, entry in results.items():
        if key == "cpu" or key not in outputs or cpu is None:
            continue
        out = outputs[key]
        groups: dict[str, list] = collections.defaultdict(list)
        failures = []
        for item in inputs:
            m = compare(cpu[item["name"]], out[item["name"]])
            m.update(detail_metrics(cpu[item["name"]], out[item["name"]]))
            m["name"] = item["name"]
            groups[item["group"]].append(m)
            for tol_key, limit in WORKING_TOLERANCES.items():
                if not m["finite"] or m[tol_key] > limit:
                    failures.append(f"{item['name']}: {tol_key} {m[tol_key]:.3e} > {limit:.1e}")
            for tol_key, ceiling in HARD_CEILINGS.items():
                if m[tol_key] > ceiling:
                    failures.append(f"{item['name']}: {tol_key} {m[tol_key]:.3e} > hard ceiling {ceiling:.0e}")
        summary = {}
        for group, rows in groups.items():
            summary[group] = {
                "count": len(rows),
                "worst": {k: max(r[k] for r in rows) for k in ("max_abs", "rel_max", "mean_abs_over_range")},
                "mean": {k: float(np.mean([r[k] for r in rows])) for k in
                         ("rel_max", "mean_abs_over_range", "edge_mean_abs_over_range",
                          "non_edge_mean_abs_over_range", "subject_mean_abs_over_range")},
                "min_pearson": min(r["pearson"] for r in rows),
                "within_working_tolerances": sum(
                    all(r[k] <= v for k, v in WORKING_TOLERANCES.items()) for r in rows),
            }
        entry["parity_vs_cpu"] = summary
        entry["parity_rows_local"] = [
            {k: r[k] for k in ("name", "rel_max", "mean_abs_over_range", "edge_mean_abs_over_range",
                               "subject_mean_abs_over_range", "pearson")}
            for r in groups.get("local_stills", [])]
        entry["temporal"] = {
            burst: {"cpu_flicker": flicker([cpu[n] for n in names]), "flicker": flicker([out[n] for n in names])}
            for burst, names in bursts.items()}
        for burst in entry["temporal"].values():
            burst["ratio_to_cpu"] = burst["flicker"] / max(burst["cpu_flicker"], 1e-12)
        entry["fp32_parity"] = "passes FP32 parity: eligible for default" if not failures else \
            "not yet approved for default use (exceeds the FP32 tolerances)"
        entry["fp32_failures"] = {"count": len(failures), "first": failures[:10]}
        print(f"\n{entry['label']}: {entry['fp32_parity']} ({len(failures)} metric failures)")
        for group, s in summary.items():
            print(f"  {group:13} n={s['count']:3} worst rel_max {s['worst']['rel_max']:.2e} mean/range "
                  f"{s['worst']['mean_abs_over_range']:.2e} | mean edge {s['mean']['edge_mean_abs_over_range']:.2e} "
                  f"non-edge {s['mean']['non_edge_mean_abs_over_range']:.2e} subject "
                  f"{s['mean']['subject_mean_abs_over_range']:.2e} | min pearson {s['min_pearson']:.6f} | "
                  f"within tolerances {s['within_working_tolerances']}/{s['count']}")
        for burst, t in entry["temporal"].items():
            print(f"  {burst}: flicker {t['flicker']:.4f} vs CPU {t['cpu_flicker']:.4f} (x{t['ratio_to_cpu']:.2f})")

        # Local panels: CPU | CoreML | |diff| x20 (same display range). Never uploaded.
        target = panels / key
        target.mkdir(parents=True, exist_ok=True)
        for item in inputs:
            if item["group"] == "fixtures" and "demo" not in item["name"]:
                continue
            if item["group"] == "local_bursts" and not item["name"].endswith("01"):
                continue
            ref = cpu[item["name"]][0]
            cand = out[item["name"]][0]
            lo, hi = np.percentile(ref, [2, 98])
            diff = np.clip(np.abs(cand - ref) / max(hi - lo, 1e-12) * 20, 0, 1)
            save_side_by_side(target / (item["name"].replace("/", "__") + ".png"),
                              [render(ref, lo, hi), render(cand, lo, hi), np.round(diff * 255).astype(np.uint8)],
                              height=320)

    RESULTS_PATH.write_text(json.dumps({
        "description": "CoreML EP matrix on the native pipeline (relume_depth, ONNX Runtime 1.31.0): evidence per "
                       "level, latency and FP32 parity against C++ CPU on identical tensors",
        "evidence_levels": {
            "a_assignment": "strict mode (diagnostic), GetCapability partitions/nodes, nodes placed per provider",
            "b_execution": "ORT profiling: kernel time per provider per run, warm-ups excluded",
            "c_planned_hardware": "CoreML compute plan (ProfileComputePlan): planned device per op; not proof of "
                                  "what ran on each inference",
            "not_proof": "output differences and latency are never treated as proof of CoreML or ANE execution",
        },
        "tolerances": {"working": WORKING_TOLERANCES, "hard_ceilings": HARD_CEILINGS},
        "inputs": {"fixtures": sum(i["group"] == "fixtures" for i in inputs), "local_stills": len(stills),
                   "local_bursts": {k: len(v) for k, v in bursts.items()}, "local_method": pp.DEFAULT_METHOD},
        "configs": results,
        "environment": {"os": platform.platform(), "machine": platform.machine(), "threads": 4},
    }, indent=1) + "\n")
    print(f"\nwrote {RESULTS_PATH.relative_to(ROOT)}; logs and local panels in {WORK.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
