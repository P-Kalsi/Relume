"""Golden vectors that pin the preprocessing contract.

`python scripts/relume_model/golden.py --write` regenerates
tests/python/golden/preprocess_golden.json. Regenerate only when the contract
is intentionally changed; the tests compare the live implementation against
the committed file, and Phase 3 C++ will check itself against it too.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from relume_model.preprocess import METHODS, preprocess  # noqa: E402
else:
    from .preprocess import METHODS, preprocess

GOLDEN_PATH = Path(__file__).resolve().parents[2] / "tests" / "python" / "golden" / "preprocess_golden.json"

# (width, height) of deterministic random images; small odd, portrait, 16:9.
CASES = [(3, 2), (5, 7), (1, 1), (17, 9), (9, 17), (64, 36)]
SAMPLE_POINTS = 24  # tensor positions recorded per case


def _image(w: int, h: int) -> np.ndarray:
    rng = np.random.default_rng(1000 * w + h)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def build() -> dict:
    cases = []
    for method in METHODS:
        for w, h in CASES:
            image = _image(w, h)
            tensor, transform = preprocess(image, method)
            rng = np.random.default_rng(7 * w + h)
            # Sample inside the image region and anywhere (includes padding).
            ys = np.concatenate([
                rng.integers(transform.pad_y, transform.pad_y + transform.new_h, SAMPLE_POINTS // 2),
                rng.integers(0, 448, SAMPLE_POINTS // 2)])
            xs = np.concatenate([
                rng.integers(transform.pad_x, transform.pad_x + transform.new_w, SAMPLE_POINTS // 2),
                rng.integers(0, 448, SAMPLE_POINTS // 2)])
            samples = [
                {"y": int(y), "x": int(x), "rgb": [float(v) for v in tensor[0, :, y, x]]}
                for y, x in zip(ys, xs)
            ]
            cases.append({
                "method": method,
                "width": w,
                "height": h,
                # Row-major HxWx3 RGB bytes, hex encoded.
                "image_rgb_hex": image.tobytes().hex(),
                "transform": transform.to_dict(),
                "samples": samples,
            })
    return {"description": "Relume preprocessing golden vectors; tensor values are float32 NCHW",
            "cases": cases}


if __name__ == "__main__":
    if "--write" not in sys.argv:
        raise SystemExit("usage: golden.py --write")
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {GOLDEN_PATH}")
