"""Relume's camera-image → network-tensor preprocessing contract.

This module is the single source of truth for preprocessing. PyTorch reference
inference, ONNX Runtime inference and the tests all call it, and the Phase 3
C++ implementation must reproduce it (checked against
tests/python/golden/preprocess_golden.json).

Coordinate convention (used by both the resampler and the point mapping):
  * Continuous image coordinates; pixel i covers [i, i+1) and its centre is
    i + 0.5.
  * A resize from width W to width w has per-axis scale sx = w / W, so the
    continuous point X maps to X * sx (+ letterbox offset). In pixel indices:
    t = (i + 0.5) * sx - 0.5 + pad_x.
  * Bilinear resampling evaluates output pixel j at source position
    u = (j + 0.5) / sx - 0.5, which is the same mapping inverted, clamped to
    [0, W - 1] (edge pixels repeat).

Status: the letterbox policy is PROVISIONAL. On the pinned demo images,
direct resizing ("stretch") agreed more closely with the official upstream
preprocessing (docs/preprocessing_comparison.json); the choice will be revisited
with real webcam frames containing faces before Phase 3.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

NET_SIZE = 448
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet, RGB order
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

METHODS = ("letterbox", "stretch")


@dataclass(frozen=True)
class ResizeTransform:
    """Where a source image landed inside the NET_SIZE x NET_SIZE tensor."""

    method: str
    src_w: int
    src_h: int
    new_w: int  # size of the resized image inside the tensor
    new_h: int
    pad_x: int  # offset of the resized image inside the tensor
    pad_y: int

    @property
    def sx(self) -> float:
        return self.new_w / self.src_w

    @property
    def sy(self) -> float:
        return self.new_h / self.src_h

    def src_to_net(self, x: np.ndarray | float, y: np.ndarray | float):
        """Continuous source coordinates → continuous tensor coordinates."""
        return np.asarray(x) * self.sx + self.pad_x, np.asarray(y) * self.sy + self.pad_y

    def net_to_src(self, x: np.ndarray | float, y: np.ndarray | float):
        """Continuous tensor coordinates → continuous source coordinates (exact inverse)."""
        return (np.asarray(x) - self.pad_x) / self.sx, (np.asarray(y) - self.pad_y) / self.sy

    def to_dict(self) -> dict:
        return asdict(self)


def compute_transform(src_w: int, src_h: int, method: str = "letterbox") -> ResizeTransform:
    """Placement of a src_w x src_h image in the network tensor.

    letterbox: scale s = NET_SIZE / max(W, H), preserving aspect ratio;
               new = max(1, round(dim * s)) using round-half-to-even, centred
               with pad = (NET_SIZE - new) // 2.
    stretch:   resize directly to NET_SIZE x NET_SIZE (aspect ratio not kept).
    """
    if src_w < 1 or src_h < 1:
        raise ValueError(f"invalid image size {src_w}x{src_h}")
    if method == "letterbox":
        scale = NET_SIZE / max(src_w, src_h)
        new_w = min(NET_SIZE, max(1, round(src_w * scale)))
        new_h = min(NET_SIZE, max(1, round(src_h * scale)))
        return ResizeTransform(method, src_w, src_h, new_w, new_h,
                               (NET_SIZE - new_w) // 2, (NET_SIZE - new_h) // 2)
    if method == "stretch":
        return ResizeTransform(method, src_w, src_h, NET_SIZE, NET_SIZE, 0, 0)
    raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")


def _axis_samples(src_len: int, dst_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lower index, upper index and upper weight for each output position."""
    u = (np.arange(dst_len, dtype=np.float64) + 0.5) * (src_len / dst_len) - 0.5
    u = np.clip(u, 0.0, src_len - 1)
    i0 = np.floor(u).astype(np.int64)
    i1 = np.minimum(i0 + 1, src_len - 1)
    w1 = (u - i0).astype(np.float32)
    return i0, i1, w1


def resize_bilinear(image: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Half-pixel-centred, edge-clamped bilinear resize of an HxW[xC] float32 array.

    Evaluated separably: horizontal pass first, then vertical, in float32.
    Implementations must follow the same order to match within 1e-6.
    """
    if image.dtype != np.float32:
        raise TypeError("resize_bilinear expects float32")
    src_h, src_w = image.shape[:2]
    x0, x1, wx = _axis_samples(src_w, out_w)
    y0, y1, wy = _axis_samples(src_h, out_h)
    extra = (1,) * (image.ndim - 2)
    wx = wx.reshape((1, out_w) + extra)
    rows = image[:, x0] * (np.float32(1) - wx) + image[:, x1] * wx
    wy = wy.reshape((out_h, 1) + extra)
    return rows[y0] * (np.float32(1) - wy) + rows[y1] * wy


def preprocess(rgb: np.ndarray, method: str = "letterbox") -> tuple[np.ndarray, ResizeTransform]:
    """HxWx3 RGB uint8 image → (1x3xNET_SIZExNET_SIZE float32 NCHW tensor, transform).

    Steps: /255 → resize (bilinear, half-pixel) → place on a canvas filled
    with the ImageNet mean (letterbox) → (x - mean) / std → NCHW.
    Padding is therefore exactly 0 after normalization.
    Callers convert BGR/BGRA/NV12 camera data to RGB first.
    """
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 uint8 RGB, got {rgb.dtype} {rgb.shape}")
    src_h, src_w = rgb.shape[:2]
    transform = compute_transform(src_w, src_h, method)

    image = rgb.astype(np.float32) / np.float32(255)
    resized = resize_bilinear(image, transform.new_w, transform.new_h)

    canvas = np.empty((NET_SIZE, NET_SIZE, 3), dtype=np.float32)
    canvas[:] = MEAN
    canvas[transform.pad_y:transform.pad_y + transform.new_h,
           transform.pad_x:transform.pad_x + transform.new_w] = resized

    normalized = (canvas - MEAN) / STD
    tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[np.newaxis])
    return tensor, transform


def depth_to_source(depth: np.ndarray, transform: ResizeTransform) -> np.ndarray:
    """Network depth (448x448 or 1x448x448) → source-resolution depth (src_h x src_w).

    Crops the padding, then resamples with the same bilinear convention; this
    is the exact inverse of the forward placement. Values are not rescaled.
    """
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[0]
    if depth.shape != (NET_SIZE, NET_SIZE):
        raise ValueError(f"expected {NET_SIZE}x{NET_SIZE} depth, got {depth.shape}")
    cropped = depth[transform.pad_y:transform.pad_y + transform.new_h,
                    transform.pad_x:transform.pad_x + transform.new_w]
    return resize_bilinear(np.ascontiguousarray(cropped), transform.src_w, transform.src_h)


def contract_description() -> dict:
    """Machine-readable summary of the contract, stored in the model manifest."""
    return {
        "status": "provisional - letterbox kept for Phase 1; before Phase 3, compare letterbox vs stretch "
                  "on local webcam frames with faces, hair, shoulders and varied lighting "
                  "(see docs/preprocessing_comparison.json)",
        "input": "HxWx3 RGB uint8 (callers convert BGR/BGRA/NV12 to RGB; alpha dropped)",
        "scale": "float32 value / 255 -> [0, 1]",
        "aspect_policy": "letterbox: s = 448 / max(W, H); new = max(1, round(dim * s)) "
                         "(round half to even); pad = (448 - new) // 2, centred",
        "resize": "bilinear, half-pixel centres, edge clamped; source u = (j + 0.5) * W / new_w - 0.5; "
                  "separable, horizontal pass then vertical, float32",
        "pad_value_rgb": MEAN.tolist(),
        "normalize": {"mean": MEAN.tolist(), "std": STD.tolist(), "formula": "(x - mean) / std"},
        "layout": "NCHW, contiguous, float32, channel order R,G,B",
        "tensor_shape": [1, 3, NET_SIZE, NET_SIZE],
        "inverse": "crop [pad_y:pad_y+new_h, pad_x:pad_x+new_w] then the same bilinear resize to src size",
        "coordinates": "continuous, pixel i centre at i + 0.5; net = src * (new/src) + pad",
    }
