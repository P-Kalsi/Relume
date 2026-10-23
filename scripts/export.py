"""Export Depth Anything V2 Small to a fixed 1x3x448x448 ONNX model.

Usage: .venv/bin/python scripts/export.py

Export is declared successful only after:
  1. onnx.checker (full_check) passes on the written file;
  2. the graph's actual opset, IR version and input/output signature are read
     back and match the expected fixed signature exactly;
  3. ONNX Runtime 1.31.0 (CPUExecutionProvider) loads it and produces a finite
     output of the expected shape.
Then the manifest models/depth_anything_v2_small_448.manifest.json is written.
"""

from __future__ import annotations

import collections
import hashlib
import json
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from relume_model import preprocess as pp  # noqa: E402
from relume_model.fixtures import GENERATED  # noqa: E402
from relume_model.tolerances import HARD_CEILINGS, WORKING_TOLERANCES  # noqa: E402
from relume_model.upstream import ASSETS, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH = ROOT / "models" / "depth_anything_v2_small_448.onnx"
MANIFEST_PATH = ROOT / "models" / "depth_anything_v2_small_448.manifest.json"

OPSET = 18
INPUT_NAME = "image"
OUTPUT_NAME = "depth"
INPUT_SHAPE = [1, 3, pp.NET_SIZE, pp.NET_SIZE]
OUTPUT_SHAPE = [1, pp.NET_SIZE, pp.NET_SIZE]
REQUIRED_ORT_VERSION = "1.31.0"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_to(model, example, path: Path) -> None:
    import torch

    with torch.inference_mode():
        torch.onnx.export(
            model,
            (example,),
            str(path),
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            opset_version=OPSET,
            dynamo=True,
            external_data=False,  # one self-contained file (~100 MB, < 2 GB limit)
            optimize=True,
        )
    strip_local_paths(path)


# torch.onnx records a Python stack trace per node, containing absolute paths
# of the checkout. That makes the file (and its hash) depend on where the repo
# lives and leaks local paths, so it is removed. Weights, graph structure and
# all other metadata are untouched.
STRIPPED_NODE_METADATA = ("pkg.torch.onnx.stack_trace",)


def strip_local_paths(path: Path) -> None:
    import onnx

    model = onnx.load(str(path))
    for node in model.graph.node:
        kept = [p for p in node.metadata_props if p.key not in STRIPPED_NODE_METADATA]
        del node.metadata_props[:]
        node.metadata_props.extend(kept)
    onnx.save(model, str(path))
    if str(ROOT).encode() in path.read_bytes():
        raise SystemExit(f"FAIL: {path} still contains the local checkout path {ROOT}")


def tensor_signature(value_info) -> dict:
    t = value_info.type.tensor_type
    dims = [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in t.shape.dim]
    import onnx

    return {"name": value_info.name, "dtype": onnx.TensorProto.DataType.Name(t.elem_type).lower(),
            "shape": dims}


def validate(path: Path) -> dict:
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(str(path), full_check=True)
    print("onnx.checker (full_check): OK")

    model = onnx.load(str(path))
    opsets = {o.domain or "ai.onnx": o.version for o in model.opset_import}
    inputs = [tensor_signature(v) for v in model.graph.input]
    outputs = [tensor_signature(v) for v in model.graph.output]
    print(f"IR version {model.ir_version}, opsets {opsets}")
    print(f"inputs  {inputs}")
    print(f"outputs {outputs}")

    expected_in = [{"name": INPUT_NAME, "dtype": "float", "shape": INPUT_SHAPE}]
    expected_out = [{"name": OUTPUT_NAME, "dtype": "float", "shape": OUTPUT_SHAPE}]
    if inputs != expected_in or outputs != expected_out:
        raise SystemExit(f"FAIL: graph signature mismatch\n  expected {expected_in} -> {expected_out}")
    if opsets.get("ai.onnx") != OPSET:
        raise SystemExit(f"FAIL: default-domain opset is {opsets.get('ai.onnx')}, expected {OPSET}")

    if ort.__version__ != REQUIRED_ORT_VERSION:
        raise SystemExit(f"FAIL: onnxruntime {ort.__version__}, expected {REQUIRED_ORT_VERSION}")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise SystemExit(f"FAIL: unexpected providers {session.get_providers()}")
    tensor, _ = pp.preprocess(GENERATED["smooth_1280x720"]())
    (depth,) = session.run([OUTPUT_NAME], {INPUT_NAME: tensor})
    if list(depth.shape) != OUTPUT_SHAPE or depth.dtype != np.float32 or not np.isfinite(depth).all():
        raise SystemExit(f"FAIL: ORT output {depth.dtype} {depth.shape}, finite={np.isfinite(depth).all()}")
    print(f"ORT {ort.__version__} CPUExecutionProvider run: OK, output {depth.dtype} {list(depth.shape)}, "
          f"range [{depth.min():.4f}, {depth.max():.4f}]")

    ops = collections.Counter(node.op_type for node in model.graph.node)
    return {
        "ir_version": model.ir_version,
        "opset_imports": opsets,
        "inputs": inputs,
        "outputs": outputs,
        "node_count": len(model.graph.node),
        "op_histogram": dict(sorted(ops.items())),
    }


def main() -> int:
    import onnx
    import onnxruntime as ort
    import torch

    model = load_model()
    example = torch.from_numpy(pp.preprocess(GENERATED["smooth_1280x720"]())[0])

    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to {ONNX_PATH.relative_to(ROOT)} (dynamo exporter, opset {OPSET})")
    export_to(model, example, ONNX_PATH)
    graph_info = validate(ONNX_PATH)
    onnx_sha = sha256_file(ONNX_PATH)

    # Determinism check: a second export must produce identical bytes, or the
    # difference is recorded and parity becomes the authority.
    with tempfile.TemporaryDirectory() as tmp:
        second = Path(tmp) / "second.onnx"
        export_to(model, example, second)
        second_sha = sha256_file(second)
    deterministic = second_sha == onnx_sha
    print(f"ONNX SHA-256 {onnx_sha} (second export {'identical' if deterministic else 'DIFFERS: ' + second_sha})")

    ckpt = ASSETS["checkpoint"]
    manifest = {
        "model": {
            "name": "Depth-Anything-V2-Small",
            "license": ckpt["license"],
            "upstream_repository": ASSETS["upstream"]["repository"],
            "upstream_commit": ASSETS["upstream"]["commit"],
            "huggingface_revision": ckpt["huggingface_revision"],
            "checkpoint_file": Path(ckpt["destination"]).name,
            "checkpoint_sha256": ckpt["sha256"],
            "checkpoint_size": ckpt["size"],
            "config": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        },
        "onnx": {
            "file": ONNX_PATH.name,
            "sha256": onnx_sha,
            "size": ONNX_PATH.stat().st_size,
            "export_deterministic": deterministic,
            "exporter": "torch.onnx.export(dynamo=True, optimize=True, external_data=False)",
            "post_export": "removed node metadata " + ", ".join(STRIPPED_NODE_METADATA)
                           + " (absolute local paths); weights and graph unchanged",
            "opset": OPSET,
            **graph_info,
        },
        "output_semantics": {
            "tensor": OUTPUT_NAME,
            "shape": OUTPUT_SHAPE,
            "dtype": "float32",
            "meaning": "relative, disparity-like depth; no metric units; scale and offset vary per image",
            "range": "non-negative (in-graph ReLU); no fixed upper bound",
            "direction": "larger_is_nearer",
            "direction_evidence": "visual check of 5 pinned upstream demo images (demo01/03/07/10/16), "
                                  "confirmed by the developer on 2026-10-23; not a metric measurement",
            "in_graph_postprocessing": "ReLU only",
            "postprocessing_outside_graph": "crop letterbox padding, then bilinear (half-pixel) resize to source size; "
                                            "display normalization (2nd-98th percentile) for visualization only",
        },
        "preprocessing": pp.contract_description(),
        "parity": {"hard_ceilings": HARD_CEILINGS, "working_tolerances": WORKING_TOLERANCES},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
            "os": platform.platform(),
            "machine": platform.machine(),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {MANIFEST_PATH.relative_to(ROOT)}")
    print("EXPORT OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
