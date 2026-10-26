"""Tests for the preprocessing contract (no model required)."""

import json

import numpy as np
import pytest

from relume_model.fixtures import GENERATED
from relume_model.golden import GOLDEN_PATH
from relume_model.preprocess import (
    DEFAULT_METHOD,
    MEAN,
    NET_SIZE,
    STD,
    _axis_samples,
    compute_transform,
    contract_description,
    depth_to_source,
    preprocess,
    resize_bilinear,
)

# (width, height): 16:9, portrait 16:9, odd portrait, odd landscape, square, tiny.
SIZES = [(1280, 720), (720, 1280), (333, 517), (517, 333), (448, 448), (1, 1), (1920, 1080), (17, 9)]


@pytest.mark.parametrize(
    "w,h,expected",
    [
        ((1280), 720, (448, 252, 0, 98)),
        (720, 1280, (252, 448, 98, 0)),
        (1920, 1080, (448, 252, 0, 98)),
        (333, 517, (289, 448, 79, 0)),  # 333 * 448 / 517 = 288.56 -> 289
        (517, 333, (448, 289, 0, 79)),
        (1, 1, (448, 448, 0, 0)),
        (448, 448, (448, 448, 0, 0)),
    ],
)
def test_letterbox_placement(w, h, expected):
    t = compute_transform(w, h, "letterbox")
    assert (t.new_w, t.new_h, t.pad_x, t.pad_y) == expected


def test_stretch_placement():
    t = compute_transform(1280, 720, "stretch")
    assert (t.new_w, t.new_h, t.pad_x, t.pad_y) == (448, 448, 0, 0)
    assert t.sx == pytest.approx(448 / 1280) and t.sy == pytest.approx(448 / 720)


def test_default_method_is_stretch():
    # Approved 2026-10-26 (docs/decisions.md); letterbox stays supported.
    assert DEFAULT_METHOD == "stretch"
    assert compute_transform(1280, 720).method == "stretch"
    assert preprocess(np.zeros((9, 17, 3), dtype=np.uint8))[1].method == "stretch"
    assert contract_description()["default_method"] == "stretch"


@pytest.mark.parametrize("method", ["letterbox", "stretch"])
@pytest.mark.parametrize("name", sorted(GENERATED))
def test_tensor_layout(name, method):
    tensor, _ = preprocess(GENERATED[name](), method)
    assert tensor.shape == (1, 3, NET_SIZE, NET_SIZE)
    assert tensor.dtype == np.float32
    assert tensor.flags["C_CONTIGUOUS"]
    assert np.isfinite(tensor).all()


def test_padding_is_exactly_zero():
    tensor, t = preprocess(GENERATED["noise_1280x720"](), "letterbox")
    assert t.pad_y == 98
    assert np.all(tensor[:, :, : t.pad_y, :] == 0)
    assert np.all(tensor[:, :, t.pad_y + t.new_h :, :] == 0)


def test_channel_order_is_rgb():
    red = np.zeros((10, 20, 3), dtype=np.uint8)
    red[..., 0] = 255
    tensor, t = preprocess(red)
    centre = tensor[0, :, NET_SIZE // 2, NET_SIZE // 2]
    expected = (np.array([1, 0, 0], dtype=np.float32) - MEAN) / STD
    np.testing.assert_allclose(centre, expected, rtol=0, atol=1e-6)


def test_rejects_non_rgb_uint8():
    with pytest.raises(ValueError):
        preprocess(np.zeros((4, 4, 4), dtype=np.uint8))  # BGRA must be converted first
    with pytest.raises(ValueError):
        preprocess(np.zeros((4, 4, 3), dtype=np.float32))


def test_bilinear_closed_form():
    # 2 pixels [0, 1] upsampled to 4: u = (j + 0.5) * 0.5 - 0.5 = -0.25, 0.25, 0.75, 1.25
    # -> clamped to 0, 0.25, 0.75, 1.
    row = np.array([[0.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(resize_bilinear(row, 4, 1)[0], [0.0, 0.25, 0.75, 1.0], atol=1e-7)


def test_bilinear_identity():
    image = GENERATED["gradient_640x480"]().astype(np.float32)
    np.testing.assert_array_equal(resize_bilinear(image, 640, 480), image)


@pytest.mark.parametrize("w,h", SIZES)
@pytest.mark.parametrize("method", ["letterbox", "stretch"])
def test_point_mapping_round_trip(w, h, method):
    t = compute_transform(w, h, method)
    rng = np.random.default_rng(w * 31 + h)
    x = rng.uniform(0, w, 1000)
    y = rng.uniform(0, h, 1000)
    nx, ny = t.src_to_net(x, y)
    bx, by = t.net_to_src(nx, ny)
    assert np.max(np.abs(bx - x)) <= 1e-9 and np.max(np.abs(by - y)) <= 1e-9
    # Source image corners land on the edges of the placed region.
    assert t.src_to_net(0, 0) == pytest.approx((t.pad_x, t.pad_y))
    assert t.src_to_net(w, h) == pytest.approx((t.pad_x + t.new_w, t.pad_y + t.new_h))


@pytest.mark.parametrize("w,h", [s for s in SIZES if min(s) > 1])
def test_resampler_matches_point_mapping(w, h):
    """The resize samples exactly where net_to_src says each tensor pixel comes from."""
    t = compute_transform(w, h, "letterbox")
    j = np.arange(t.new_w)
    u_resampler = _axis_samples(w, t.new_w)
    lo, hi, frac = u_resampler
    u_from_resampler = lo + frac.astype(np.float64)
    # Pixel centre of tensor column (pad_x + j), mapped back, in pixel-index units.
    u_from_mapping, _ = t.net_to_src(t.pad_x + j + 0.5, 0)
    u_from_mapping = np.clip(u_from_mapping - 0.5, 0, w - 1)
    np.testing.assert_allclose(u_from_resampler, u_from_mapping, atol=1e-6)

    # A ramp whose value is its own x index is reproduced exactly by bilinear
    # interpolation, so the resized ramp must equal the mapped coordinates.
    ramp = np.broadcast_to(np.arange(w, dtype=np.float32), (h, w)).copy()
    resized = resize_bilinear(ramp, t.new_w, t.new_h)
    np.testing.assert_allclose(resized[0], u_from_mapping, atol=1e-4)


@pytest.mark.parametrize("name", ["smooth_1280x720", "gradient_640x480", "checker_333x517_portrait"])
def test_image_round_trip(name):
    """letterbox → inverse returns a smooth image close to the original."""
    image = GENERATED[name]()
    tensor, t = preprocess(image, "letterbox")
    back = np.stack(
        [depth_to_source(tensor[0, c] * STD[c] + MEAN[c], t) for c in range(3)], axis=-1)
    assert back.shape == image.shape
    error = np.abs(back - image.astype(np.float32) / 255)
    # The checker has hard edges that a downscale cannot preserve, so only
    # smooth content gets the strict bound.
    limit = 0.05 if name.startswith("checker") else 0.01
    assert error.mean() <= limit, f"mean abs error {error.mean():.4f}"


def test_depth_to_source_shapes():
    t = compute_transform(1280, 720)
    out = depth_to_source(np.ones((1, NET_SIZE, NET_SIZE), dtype=np.float32), t)
    assert out.shape == (720, 1280)
    np.testing.assert_allclose(out, 1.0)


def test_golden_vectors():
    golden = json.loads(GOLDEN_PATH.read_text())
    for case in golden["cases"]:
        image = np.frombuffer(bytes.fromhex(case["image_rgb_hex"]), dtype=np.uint8).reshape(
            case["height"], case["width"], 3)
        tensor, t = preprocess(image, case["method"])
        assert t.to_dict() == case["transform"], case["method"]
        for s in case["samples"]:
            np.testing.assert_allclose(tensor[0, :, s["y"], s["x"]], s["rgb"], rtol=0, atol=1e-6)
