# models/

Model weights are **never committed**. This folder holds them locally; only
this README and `*.manifest.json` files are tracked.

| File | Tracked | How to get it |
|---|---|---|
| `depth_anything_v2_vits.pth` | no | `scripts/setup_python.sh`. It downloads the official Depth-Anything-V2-Small checkpoint (Apache-2.0) at a pinned Hugging Face revision and verifies its SHA-256. |
| `depth_anything_v2_small_448.onnx` | no | `.venv/bin/python scripts/export.py` (fixed 1×3×448×448, opset 18) |
| `depth_anything_v2_small_448.manifest.json` | **yes** | Written by `export.py`. It records provenance, hashes, the graph signature, the preprocessing contract, output semantics and parity limits. |

Pins and hashes live in `scripts/assets.json`. See `docs/model.md` for details.
Only the **Small** checkpoint may be used. The Base, Large and Giant checkpoints are
CC-BY-NC-4.0 and are not drop-in substitutes.
