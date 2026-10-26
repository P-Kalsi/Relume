"""Summarize an ONNX Runtime profiling JSON by execution provider.

Usage:
  .venv/bin/python scripts/summarize_ort_profile.py <profile.json> [--skip N]

Kernel events are grouped into the model runs they fall in ("model_run"
session events); the first N runs (warm-ups) are skipped. For each provider
it reports the kernel events per run and the kernel time per run. A provider
with kernel time is evidence that it executed nodes at run time (level (b));
it says nothing about which hardware CoreML used.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def summarize(path: Path, skip: int = 0) -> dict:
    events = json.loads(path.read_text())
    runs = sorted((e["ts"], e["ts"] + e["dur"]) for e in events
                  if e.get("cat") == "Session" and e.get("name") == "model_run")
    measured = runs[skip:]
    per_provider: dict[str, dict[str, float]] = defaultdict(lambda: {"events": 0, "kernel_us": 0.0})
    for e in events:
        if e.get("cat") != "Node" or not e.get("name", "").endswith("_kernel_time"):
            continue
        ts = e["ts"]
        if not any(start <= ts <= end for start, end in measured):
            continue
        provider = e.get("args", {}).get("provider", "unknown")
        per_provider[provider]["events"] += 1
        per_provider[provider]["kernel_us"] += e["dur"]
    count = max(len(measured), 1)
    run_ms = [(end - start) / 1000 for start, end in measured]
    return {
        "file": path.name,
        "runs_total": len(runs),
        "runs_measured": len(measured),
        "model_run_ms_mean": sum(run_ms) / count if measured else 0.0,
        "providers": {
            p: {"kernel_events_per_run": v["events"] / count, "kernel_ms_per_run": v["kernel_us"] / 1000 / count}
            for p, v in sorted(per_provider.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    parser.add_argument("--skip", type=int, default=0, help="warm-up runs to leave out")
    options = parser.parse_args()
    print(json.dumps(summarize(options.profile, options.skip), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
