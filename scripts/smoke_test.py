"""Import smoke test for the model tooling environment.

Fails (non-zero exit) unless every pinned package imports, ONNX Runtime is
1.31.0 with the CPU provider, the pinned upstream code imports, and the
verified checkpoint loads into DepthAnythingV2-Small with strict key matching.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    import cv2
    import numpy
    import onnx
    import onnxruntime
    import onnxscript
    import PIL
    import torch
    import torchvision

    print(f"python       {platform.python_version()}")
    for module in (torch, torchvision, onnx, onnxscript, onnxruntime, numpy, cv2, PIL):
        print(f"{module.__name__:12} {module.__version__}")

    if onnxruntime.__version__ != "1.31.0":
        print(f"FAIL: onnxruntime {onnxruntime.__version__} != 1.31.0 (must match the C++ runtime)")
        return 1
    providers = onnxruntime.get_available_providers()
    print(f"ORT providers {providers}")
    if "CPUExecutionProvider" not in providers:
        print("FAIL: CPUExecutionProvider missing")
        return 1

    from relume_model.upstream import load_model

    model = load_model()
    params = sum(p.numel() for p in model.parameters())
    print(f"DepthAnythingV2-Small loaded (strict), {params / 1e6:.1f}M parameters")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
