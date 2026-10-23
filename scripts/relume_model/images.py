"""Image loading and depth visualization helpers (model tooling only)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def load_rgb(path: Path) -> np.ndarray:
    """File → HxWx3 RGB uint8 (Pillow; alpha dropped, palette/greyscale expanded)."""
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8).copy()


def demo_images() -> list[tuple[str, Path, str]]:
    """(name, path, sha256) for each pinned upstream demo image, hash-checked."""
    assets = json.loads((ROOT / "scripts" / "assets.json").read_text())["demo_images"]
    out = []
    for name, digest in assets["files"].items():
        path = ROOT / assets["destination"] / name
        if not path.exists():
            raise SystemExit(f"{path} missing: run scripts/setup_python.sh")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise SystemExit(f"{path} SHA-256 mismatch: {actual}")
        out.append((name, path, digest))
    return out


def depth_stats(depth: np.ndarray) -> dict:
    d = np.asarray(depth, dtype=np.float64)
    p2, p50, p98 = np.percentile(d, [2, 50, 98])
    return {"min": float(d.min()), "p2": float(p2), "median": float(p50), "p98": float(p98),
            "max": float(d.max())}


def render_depth(depth: np.ndarray) -> np.ndarray:
    """Display-only: clip to the 2nd-98th percentile and map to 8-bit grey
    (brighter = larger value). The raw depth array is not modified."""
    lo, hi = np.percentile(depth, [2, 98])
    scaled = (np.clip(depth, lo, hi) - lo) / max(hi - lo, 1e-12)
    return np.round(scaled * 255).astype(np.uint8)


def save_side_by_side(path: Path, panels: list[np.ndarray], height: int = 360) -> None:
    """Save RGB / greyscale panels next to each other, scaled to a common height."""
    images = []
    for panel in panels:
        img = Image.fromarray(panel if panel.ndim == 3 else np.stack([panel] * 3, axis=-1))
        width = max(1, round(img.width * height / img.height))
        images.append(img.resize((width, height), Image.Resampling.BILINEAR))
    canvas = Image.new("RGB", (sum(i.width for i in images) + 8 * (len(images) - 1), height), "white")
    x = 0
    for img in images:
        canvas.paste(img, (x, 0))
        x += img.width + 8
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
