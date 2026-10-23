"""Checks on the exported ONNX model against the committed manifest.

Requires scripts/setup_python.sh and scripts/export.py to have been run; a
missing model is a test failure, not a skip.
"""

import hashlib
import json

import numpy as np
import pytest

from conftest import ROOT
from relume_model.fixtures import GENERATED
from relume_model.preprocess import preprocess

ONNX_PATH = ROOT / "models" / "depth_anything_v2_small_448.onnx"
MANIFEST_PATH = ROOT / "models" / "depth_anything_v2_small_448.manifest.json"


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST_PATH.read_text())


@pytest.fixture(scope="module")
def onnx_path():
    if not ONNX_PATH.exists():
        pytest.fail(f"{ONNX_PATH} is missing: run scripts/setup_python.sh, then .venv/bin/python scripts/export.py")
    return ONNX_PATH


@pytest.fixture(scope="module")
def session(onnx_path):
    import onnxruntime as ort

    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def test_onnx_hash_matches_manifest(onnx_path, manifest):
    digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    assert digest == manifest["onnx"]["sha256"], "model differs from the manifest; re-run scripts/export.py"


def test_no_local_paths_embedded(onnx_path):
    """The model must not depend on, or leak, where the repo is checked out."""
    data = onnx_path.read_bytes()
    assert str(ROOT).encode() not in data
    assert b"pkg.torch.onnx.stack_trace" not in data


def test_checker_and_signature(onnx_path, manifest):
    import onnx

    onnx.checker.check_model(str(onnx_path), full_check=True)
    model = onnx.load(str(onnx_path))
    opsets = {o.domain or "ai.onnx": o.version for o in model.opset_import}
    assert opsets == manifest["onnx"]["opset_imports"]
    assert opsets["ai.onnx"] == 18

    def sig(v):
        t = v.type.tensor_type
        return {"name": v.name, "dtype": onnx.TensorProto.DataType.Name(t.elem_type).lower(),
                "shape": [d.dim_value for d in t.shape.dim]}

    assert [sig(v) for v in model.graph.input] == [{"name": "image", "dtype": "float", "shape": [1, 3, 448, 448]}]
    assert [sig(v) for v in model.graph.output] == [{"name": "depth", "dtype": "float", "shape": [1, 448, 448]}]
    assert [sig(v) for v in model.graph.input] == manifest["onnx"]["inputs"]
    assert [sig(v) for v in model.graph.output] == manifest["onnx"]["outputs"]


def test_runtime_version_matches_cpp():
    import onnxruntime as ort

    assert ort.__version__ == "1.31.0"


@pytest.fixture(scope="module")
def torch_model():
    from relume_model.upstream import load_model

    return load_model()


@pytest.mark.parametrize("method", ["letterbox", "stretch"])
@pytest.mark.parametrize("name", sorted(GENERATED))
def test_parity_with_pytorch(session, torch_model, name, method):
    """Same tensor into both runtimes; fails above the committed tolerances."""
    import torch

    from relume_model.tolerances import HARD_CEILINGS, WORKING_TOLERANCES

    assert WORKING_TOLERANCES is not None, "tolerances not calibrated"
    for key, value in WORKING_TOLERANCES.items():
        if key in HARD_CEILINGS:
            assert value <= HARD_CEILINGS[key], f"{key} tolerance above its hard ceiling"

    tensor, _ = preprocess(GENERATED[name](), method)
    with torch.inference_mode():
        ref = torch_model(torch.from_numpy(tensor)).numpy().astype(np.float64)
    out = session.run(["depth"], {"image": tensor})[0].astype(np.float64)
    assert np.isfinite(ref).all() and np.isfinite(out).all()
    diff = np.abs(out - ref)
    value_range = ref.max() - ref.min()
    assert diff.max() <= WORKING_TOLERANCES["max_abs"]
    assert diff.max() / value_range <= WORKING_TOLERANCES["rel_max"]
    assert diff.mean() / value_range <= WORKING_TOLERANCES["mean_abs_over_range"]


@pytest.mark.parametrize("name", sorted(GENERATED))
def test_cpu_inference_output(session, name):
    tensor, _ = preprocess(GENERATED[name]())
    (depth,) = session.run(["depth"], {"image": tensor})
    assert depth.shape == (1, 448, 448)
    assert depth.dtype == np.float32
    assert np.isfinite(depth).all()
    assert depth.min() >= 0.0  # in-graph ReLU
