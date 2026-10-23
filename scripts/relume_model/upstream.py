"""Load the pinned official Depth Anything V2 Small model (PyTorch reference)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = json.loads((ROOT / "scripts" / "assets.json").read_text())
CHECKPOINT = ROOT / ASSETS["checkpoint"]["destination"]
UPSTREAM_DIR = ROOT / ASSETS["upstream"]["destination"]

# Upstream README, model_configs['vits'].
VITS_CONFIG = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}


def verify_checkpoint() -> None:
    if not CHECKPOINT.exists():
        raise SystemExit(f"Checkpoint missing: {CHECKPOINT}. Run scripts/setup_python.sh")
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    if digest != ASSETS["checkpoint"]["sha256"]:
        raise SystemExit(f"Checkpoint SHA-256 mismatch: {digest}")


def import_upstream():
    """Returns the upstream DepthAnythingV2 class from the pinned checkout."""
    if not (UPSTREAM_DIR / "depth_anything_v2" / "dpt.py").exists():
        raise SystemExit(f"Upstream source missing: {UPSTREAM_DIR}. Run scripts/setup_python.sh")
    if str(UPSTREAM_DIR) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_DIR))
    from depth_anything_v2.dpt import DepthAnythingV2

    return DepthAnythingV2


def load_model():
    """Hash-checked checkpoint loaded with strict keys, eval mode, CPU, FP32."""
    import torch

    verify_checkpoint()
    model = import_upstream()(**VITS_CONFIG)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device="cpu", dtype=torch.float32)
