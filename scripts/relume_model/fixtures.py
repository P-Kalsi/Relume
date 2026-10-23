"""Deterministic generated test images.

These exercise preprocessing and PyTorch/ONNX numeric agreement only. They
are NOT depth ground truth: a monocular model's output on them is not
physically meaningful and is never checked for plausibility.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .preprocess import MEAN


def _gradient(w: int = 640, h: int = 480) -> np.ndarray:
    x = np.linspace(0, 255, w, dtype=np.float64)
    y = np.linspace(0, 255, h, dtype=np.float64)
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[..., 0] = np.round(x)[np.newaxis, :]
    img[..., 1] = np.round(y)[:, np.newaxis]
    img[..., 2] = 96
    return img


def _checker_portrait(w: int = 333, h: int = 517, cell: int = 16) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    on = ((xx // cell) + (yy // cell)) % 2 == 0
    img = np.where(on[..., np.newaxis], np.array([230, 40, 20]), np.array([20, 60, 210]))
    return img.astype(np.uint8)


def _noise_16x9(w: int = 1280, h: int = 720) -> np.ndarray:
    rng = np.random.default_rng(20261023)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _smooth_16x9(w: int = 1280, h: int = 720) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r = 128 + 100 * np.sin(xx / 97.0) * np.cos(yy / 71.0)
    g = 128 + 100 * np.sin((xx + yy) / 131.0)
    b = 128 + 100 * np.cos(np.hypot(xx - w / 2, yy - h / 2) / 53.0)
    return np.round(np.stack([r, g, b], axis=-1)).astype(np.uint8)


def _solid_mean(w: int = 448, h: int = 448) -> np.ndarray:
    return np.broadcast_to(np.round(MEAN * 255).astype(np.uint8), (h, w, 3)).copy()


def _single_pixel() -> np.ndarray:
    return np.array([[[200, 100, 50]]], dtype=np.uint8)


GENERATED: dict[str, Callable[[], np.ndarray]] = {
    "gradient_640x480": _gradient,
    "checker_333x517_portrait": _checker_portrait,
    "noise_1280x720": _noise_16x9,
    "smooth_1280x720": _smooth_16x9,
    "solid_mean_448x448": _solid_mean,
    "pixel_1x1": _single_pixel,
}
