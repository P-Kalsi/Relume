"""Compare preprocessing methods on realistic images (informational, not a gate).

Usage: .venv/bin/python scripts/compare_preprocessing.py

Methods:
  A  letterbox  - Relume's proposed contract, fixed 448x448 ONNX model
  B  stretch    - direct resize to 448x448 ignoring aspect, same ONNX model
  C  upstream   - official Depth-Anything-V2 image2tensor (518 lower bound,
                  multiple of 14, bicubic, variable shape), PyTorch, output
                  upsampled with align_corners=True as in infer_image()
C is the official reference behaviour, NOT ground truth. The script reports
agreement and differences only; it makes no accuracy claims.

Images: each pinned upstream demo image at its original aspect, plus a
webcam-like 1280x720 (16:9) centre crop of it, plus any local frames you put
in third_party/assets/local_frames/ (git-ignored, optional; Relume's
--save-frames writes them there). Image files directly in that folder are
stills; each subfolder is a burst of consecutive frames (sorted by name),
used for the frame-to-frame stability metric.

Pre-registered decision rule (fixed before any local frame was compared):
recommend stretch only if, on the local stills, B beats A against C on both
Pearson and Spearman in at least 75% of frames, and B's mean normalized MAD
is at least 10% lower than A's; and the developer's visual review of faces,
hair and edges finds B no worse. Otherwise keep letterbox. The script
evaluates the numeric part only.

Writes docs/preprocessing_comparison.json and panels in build/preproc_compare/.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from relume_model import preprocess as pp  # noqa: E402
from relume_model.images import ROOT, demo_images, load_rgb, render_depth, save_side_by_side  # noqa: E402
from relume_model.upstream import UPSTREAM_DIR, load_model  # noqa: E402

ONNX_PATH = ROOT / "models" / "depth_anything_v2_small_448.onnx"
RESULTS_PATH = ROOT / "docs" / "preprocessing_comparison.json"
PANEL_DIR = ROOT / "build" / "preproc_compare"
LOCAL_FRAMES = ROOT / "third_party" / "assets" / "local_frames"


def webcam_crop(rgb: np.ndarray, w: int = 1280, h: int = 720) -> np.ndarray:
    """Largest centred 16:9 crop, resized to w x h (deterministic, our bilinear)."""
    src_h, src_w = rgb.shape[:2]
    crop_w = min(src_w, round(src_h * w / h))
    crop_h = min(src_h, round(crop_w * h / w))
    x0, y0 = (src_w - crop_w) // 2, (src_h - crop_h) // 2
    crop = rgb[y0:y0 + crop_h, x0:x0 + crop_w].astype(np.float32)
    return np.clip(np.round(pp.resize_bilinear(crop, w, h)), 0, 255).astype(np.uint8)


def robust_unit(depth: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(depth, [2, 98])
    return np.clip((depth - lo) / max(hi - lo, 1e-12), 0, 1)


def ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    r = np.empty_like(order, dtype=np.float64)
    r[order] = np.arange(x.size, dtype=np.float64)
    return r


def boundary_mask(reference: np.ndarray) -> np.ndarray:
    """Depth boundaries of the reference: the top 10% of its normalized gradient magnitude."""
    gy, gx = np.gradient(robust_unit(reference))
    magnitude = np.hypot(gx, gy)
    return magnitude >= np.percentile(magnitude, 90)


def subject_mask(reference: np.ndarray) -> np.ndarray:
    """Nearest quarter of the reference (larger = nearer): a proxy for the person in a webcam frame."""
    return reference >= np.percentile(reference, 75)


def agreement(a: np.ndarray, c: np.ndarray) -> dict:
    """a, c: same-shape source-resolution depth maps. Subsampled for speed."""
    a_s, c_s = a[::4, ::4].ravel(), c[::4, ::4].ravel()
    ua, uc = robust_unit(a), robust_unit(c)
    diff = np.abs(ua - uc)
    band = max(1, a.shape[0] // 10)  # top and bottom 10% of rows
    edge = np.concatenate([diff[:band], diff[-band:]])
    interior = diff[band:-band]
    boundary = boundary_mask(c)
    subject = subject_mask(c)
    sub_a, sub_c = a[subject][::16], c[subject][::16]
    return {
        "pearson": float(np.corrcoef(a_s, c_s)[0, 1]),
        "spearman": float(np.corrcoef(ranks(a_s), ranks(c_s))[0, 1]),
        "mean_abs_diff_normalized": float(diff.mean()),
        "edge_band_mean_abs_diff": float(edge.mean()),
        "interior_mean_abs_diff": float(interior.mean()),
        "boundary_mean_abs_diff": float(diff[boundary].mean()),
        "non_boundary_mean_abs_diff": float(diff[~boundary].mean()),
        "subject_mean_abs_diff": float(diff[subject].mean()),
        "subject_spearman": float(np.corrcoef(ranks(sub_a), ranks(sub_c))[0, 1]),
    }


def flicker(depths: list[np.ndarray]) -> dict:
    """Frame-to-frame change of consecutive normalized depth maps (whole frame and subject region)."""
    units = [robust_unit(d) for d in depths]
    whole = [float(np.abs(u1 - u0).mean()) for u0, u1 in zip(units, units[1:])]
    subject = [float(np.abs(u1 - u0)[subject_mask(d1)].mean()) for u0, u1, d1 in zip(units, units[1:], depths[1:])]
    return {"mean_abs_frame_diff": float(np.mean(whole)), "p95_abs_frame_diff": float(np.percentile(whole, 95)),
            "subject_mean_abs_frame_diff": float(np.mean(subject))}


def local_frames() -> tuple[list[Path], dict[str, list[Path]]]:
    """Stills (files directly in LOCAL_FRAMES) and bursts (one per subfolder)."""
    def images_in(folder: Path) -> list[Path]:
        return sorted(p for p in folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".ppm"})
    if not LOCAL_FRAMES.is_dir():
        return [], {}
    bursts = {d.name: images_in(d) for d in sorted(LOCAL_FRAMES.iterdir()) if d.is_dir()}
    return images_in(LOCAL_FRAMES), {k: v for k, v in bursts.items() if len(v) >= 2}


def decision_rule(rows: list[dict]) -> dict | None:
    """Numeric part of the pre-registered rule, on the local stills."""
    local = [r for r in rows if r["kind"] == "local_frame"]
    if not local:
        return None
    a = [r["A_letterbox_vs_C_upstream"] for r in local]
    b = [r["B_stretch_vs_C_upstream"] for r in local]
    wins = sum(1 for x, y in zip(a, b) if y["pearson"] > x["pearson"] and y["spearman"] > x["spearman"])
    mad_a = float(np.mean([x["mean_abs_diff_normalized"] for x in a]))
    mad_b = float(np.mean([y["mean_abs_diff_normalized"] for y in b]))
    win_fraction = wins / len(local)
    mad_reduction = 1.0 - mad_b / mad_a if mad_a > 0 else 0.0
    return {
        "frames": len(local),
        "B_wins_both_correlations": wins,
        "B_win_fraction": win_fraction,
        "mean_mad_A": mad_a,
        "mean_mad_B": mad_b,
        "B_mad_reduction": mad_reduction,
        "numeric_rule_met": bool(win_fraction >= 0.75 and mad_reduction >= 0.10),
        "note": "Numeric part only; stretch also requires the developer's visual review to find B no worse.",
    }


def main() -> int:
    import cv2
    import onnxruntime as ort
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, str(UPSTREAM_DIR))
    from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize
    from torchvision.transforms import Compose

    model = load_model()
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    upstream_transform = Compose([
        Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14,
               resize_method="lower_bound", image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    def method_ab(rgb: np.ndarray, method: str) -> np.ndarray:
        tensor, transform = pp.preprocess(rgb, method)
        out = session.run(["depth"], {"image": tensor})[0]
        return pp.depth_to_source(out, transform)

    def method_c(rgb: np.ndarray) -> tuple[np.ndarray, list[int]]:
        # Same steps as upstream infer_image(), kept on the CPU. Upstream takes
        # BGR and converts to RGB / 255; we start from RGB directly.
        h, w = rgb.shape[:2]
        image = upstream_transform({"image": rgb.astype(np.float64) / 255.0})["image"]
        tensor = torch.from_numpy(image).unsqueeze(0)
        with torch.inference_mode():
            depth = model(tensor)
            depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        return depth.numpy(), list(tensor.shape)

    def torch_latency(tensor: np.ndarray | torch.Tensor, runs: int = 10) -> float:
        t = torch.as_tensor(tensor)
        with torch.inference_mode():
            model(t)
            times = []
            for _ in range(runs):
                start = time.perf_counter()
                model(t)
                times.append((time.perf_counter() - start) * 1000)
        return float(np.median(times))

    images = []
    for name, path, digest in demo_images():
        rgb = load_rgb(path)
        stem = Path(name).stem
        images.append((f"{stem}_original", "upstream_demo", digest, rgb))
        images.append((f"{stem}_webcam16x9", "upstream_demo_cropped_16x9", digest, webcam_crop(rgb)))
    stills, bursts = local_frames()
    for path in stills:
        images.append((path.stem, "local_frame", None, load_rgb(path)))

    rows = []
    for name, kind, digest, rgb in images:
        depth_a = method_ab(rgb, "letterbox")
        depth_b = method_ab(rgb, "stretch")
        depth_c, c_shape = method_c(rgb)
        rows.append({
            "image": name,
            "kind": kind,
            "source_sha256": digest,
            "size": [int(rgb.shape[1]), int(rgb.shape[0])],
            "upstream_network_input": c_shape,
            "A_letterbox_vs_C_upstream": agreement(depth_a, depth_c),
            "B_stretch_vs_C_upstream": agreement(depth_b, depth_c),
            "A_letterbox_vs_B_stretch": agreement(depth_a, depth_b),
        })
        ua, ub, uc = robust_unit(depth_a), robust_unit(depth_b), robust_unit(depth_c)
        to8 = lambda x: np.round(np.clip(x, 0, 1) * 255).astype(np.uint8)  # noqa: E731
        save_side_by_side(PANEL_DIR / f"{name}.png",
                          [rgb, render_depth(depth_a), render_depth(depth_b), render_depth(depth_c),
                           to8(np.abs(ua - uc) * 4), to8(np.abs(ub - uc) * 4)], height=300)
        r = rows[-1]
        print(f"{name:22} A~C pearson {r['A_letterbox_vs_C_upstream']['pearson']:.4f} "
              f"spearman {r['A_letterbox_vs_C_upstream']['spearman']:.4f} "
              f"mad {r['A_letterbox_vs_C_upstream']['mean_abs_diff_normalized']:.4f} | "
              f"B~C pearson {r['B_stretch_vs_C_upstream']['pearson']:.4f} "
              f"spearman {r['B_stretch_vs_C_upstream']['spearman']:.4f} "
              f"mad {r['B_stretch_vs_C_upstream']['mean_abs_diff_normalized']:.4f} | C input {c_shape[2]}x{c_shape[3]}")

    stability = {}
    for burst, paths in bursts.items():
        frames = [load_rgb(p) for p in paths]
        per_method = {"A_letterbox": [method_ab(f, "letterbox") for f in frames],
                      "B_stretch": [method_ab(f, "stretch") for f in frames],
                      "C_upstream": [method_c(f)[0] for f in frames]}
        stability[burst] = {"frames": len(frames), **{k: flicker(v) for k, v in per_method.items()}}
        print(f"burst {burst}: " + " | ".join(f"{k} flicker {v['mean_abs_frame_diff']:.4f}"
                                             for k, v in stability[burst].items() if k != "frames"))
        save_side_by_side(PANEL_DIR / f"burst_{burst}.png",
                          [frames[0]] + [render_depth(per_method[k][0]) for k in per_method], height=300)

    webcam_rgb = images[1][3]
    tensor_a, _ = pp.preprocess(webcam_rgb, "letterbox")
    tensor_c = torch.from_numpy(upstream_transform({"image": webcam_rgb.astype(np.float64) / 255.0})["image"]).unsqueeze(0)
    lat = {
        "note": "PyTorch CPU median of 10 runs on one 1280x720 frame; same runtime for both, so only the "
                "input size differs. Informational laptop timing.",
        "A_or_B_448x448_ms": torch_latency(tensor_a),
        "C_upstream_ms": torch_latency(tensor_c),
        "C_upstream_input_shape": list(tensor_c.shape),
    }
    print(f"latency (PyTorch CPU): 448x448 {lat['A_or_B_448x448_ms']:.0f} ms, upstream "
          f"{tensor_c.shape[2]}x{tensor_c.shape[3]} {lat['C_upstream_ms']:.0f} ms")

    summary = {}
    for key in ("A_letterbox_vs_C_upstream", "B_stretch_vs_C_upstream"):
        for subset in ("upstream_demo", "upstream_demo_cropped_16x9", "local_frame"):
            sel = [r[key] for r in rows if r["kind"] == subset]
            if not sel:
                continue
            summary[f"{key}/{subset}"] = {m: float(np.mean([s[m] for s in sel])) for m in sel[0]}
    results = {
        "description": "Agreement between preprocessing methods; C (official upstream) is a reference "
                       "behaviour, not ground truth. Normalized maps use the 2nd-98th percentile per image.",
        "methods": {
            "A": "letterbox 448 (Relume proposed), ONNX Runtime CPU",
            "B": "stretch to 448x448, ONNX Runtime CPU",
            "C": "upstream image2tensor 518 lower-bound, multiple of 14, bicubic; PyTorch CPU; "
                 "bilinear align_corners=True upsample",
        },
        "mean_by_subset": summary,
        "decision_rule": decision_rule(rows),
        "temporal_stability": stability,
        "images": rows,
        "latency": lat,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "onnxruntime": ort.__version__, "machine": platform.machine()},
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(summary, indent=1))
    print("decision rule (numeric part):", json.dumps(results["decision_rule"], indent=1))
    print(f"wrote {RESULTS_PATH.relative_to(ROOT)}; panels in {PANEL_DIR.relative_to(ROOT)}/ "
          "(source | A letterbox | B stretch | C upstream | |A-C|x4 | |B-C|x4)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
