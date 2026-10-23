# Depth model: provenance, contract and validation

Phase 1 record for the depth network. The machine-readable source of truth is
`models/depth_anything_v2_small_448.manifest.json`, written by
`scripts/export.py`.

## Provenance

| Item | Value |
|---|---|
| Model | Depth-Anything-V2-**Small** (ViT-S encoder), 24.8M parameters |
| License | Apache-2.0 (code and Small weights). Base/Large/Giant are CC-BY-NC-4.0 and are not used. Notice: `LICENSES/depth-anything-v2-small.txt` |
| Upstream code | https://github.com/DepthAnything/Depth-Anything-V2 @ `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf` (fetched, not vendored) |
| Checkpoint | HF `depth-anything/Depth-Anything-V2-Small` @ `03876f8651c73a60fe4c2c48294e09fcb6838fcf`, `depth_anything_v2_vits.pth`, 99,218,434 B, SHA-256 `715fade1…bf57860e1378` |
| Config | `encoder='vits', features=64, out_channels=[48, 96, 192, 384]`, loaded with `strict=True` |
| ONNX | `models/depth_anything_v2_small_448.onnx`, 99,345,329 B, SHA-256 `6ca89260de8ed5cb08571d197f81a04cf54cb04857c3a8030759b1aa8a8aa42c`, IR 10, opset 18, 466 nodes |
| Exporter | `torch.onnx.export(dynamo=True, opset_version=18, optimize=True, external_data=False)`, torch 2.14.1. After export, the per-node `pkg.torch.onnx.stack_trace` metadata is removed. |
| Determinism | Byte-identical across repeated exports **and across checkout locations**: the working copy and a fresh clone in another directory produce the same SHA-256. |

**Why the stack traces are removed:** the exporter stores a Python stack trace on every node, and those traces contain absolute paths to the checkout. Before they were removed, a fresh clone produced a different hash (`8ee33eb9…` vs `8d661548…`) even though the weights, ops, connections and attributes were bit-identical, and the file contained the developer's home-folder path. Removing only that key leaves weights and graph unchanged, and parity results were identical before and after. A test checks that no local path is embedded.

Neither the weights nor the ONNX file is in git. `scripts/setup_python.sh` downloads the weights and checks their hash, and `scripts/export.py` regenerates the ONNX file.

## Network signature

| | Name | dtype | Shape | Notes |
|---|---|---|---|---|
| Input | `image` | float32 | [1, 3, 448, 448] | NCHW, RGB, normalized as described below |
| Output | `depth` | float32 | [1, 448, 448] | Relative depth, see "Output semantics" |

All dimensions are fixed; there are no dynamic axes. The op types are Add, Concat, Conv,
ConvTranspose, Div, Erf, Gather, LayerNormalization, MatMul, Mul, Relu,
Reshape, Resize, Slice, Softmax, Squeeze, Transpose and Unsqueeze. Whether CoreML supports all of them is a Phase 3 question; it has not been tested.

## Preprocessing contract (status: provisional)

Implemented once in `scripts/relume_model/preprocess.py` and used by every
Python path. Phase 3 C++ must reproduce it and match
`tests/python/golden/preprocess_golden.json` within 1e-6.

1. **Input:** an H×W×3 **RGB** uint8 image of any size of at least 1×1. Callers convert
   BGR/BGRA/NV12 to RGB first, dropping any alpha channel.
2. **Scale:** convert to float32 and divide by 255, giving values in [0, 1].
3. **Aspect ratio (letterbox):**
   - `s = 448 / max(W, H)`
   - `new_w = max(1, round(W·s))` and `new_h = max(1, round(H·s))`, rounding half to even
   - `pad_x = (448 − new_w) // 2` and `pad_y = (448 − new_h) // 2`
   - Example: 1280×720 becomes 448×252 with `pad_y = 98`.
4. **Resize:** bilinear with half-pixel centres and clamped edges.
   - Output pixel `j` samples the source at `u = (j + 0.5)·W/new_w − 0.5`, clamped to `[0, W−1]`.
   - It is computed separably, horizontal pass first, in float32.
5. **Pad:** fill the canvas with the ImageNet mean RGB `(0.485, 0.456, 0.406)`, which becomes exactly 0 after normalization.
6. **Normalize:** `(x − mean) / std`, with mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`.
7. **Layout:** a contiguous NCHW float32 tensor of shape [1, 3, 448, 448], channels in R, G, B order.

**Coordinates.** Coordinates are continuous, with pixel *i*'s centre at *i* + 0.5.
- Forward: `X_net = X_src · (new_w / W) + pad_x`, and the same for y.
- Pixel-index form: `t = (i + 0.5)·sx − 0.5 + pad_x`, which is exactly where the resampler samples.
- Inverse (`depth_to_source`): crop `[pad_y : pad_y+new_h, pad_x : pad_x+new_w]`, then apply the same bilinear resize to W×H.
- Tests check that point mappings round-trip to within 1e-9 px, that the resampler and the mapping agree, and that images round-trip.

**Differences from upstream `image2tensor`:**
- Upstream uses a lower-bound resize to **518**, keeps the aspect ratio with each side a multiple of 14 (for example 924×518 for 16:9), uses **bicubic** interpolation (OpenCV), and adds no padding.
- Its output is upsampled with `align_corners=True`.
- Relume uses a fixed 448 input, bilinear interpolation with half-pixel centres, and mean padding. The trade-off is a fixed ONNX shape, lower cost and exact reproducibility in C++.

## Preprocessing comparison (informational)

`scripts/compare_preprocessing.py` compares three methods. Full numbers are in
`docs/preprocessing_comparison.json`; panels are in `build/preproc_compare/` (local only).

| Method | Description |
|---|---|
| A | Letterbox 448 (the current contract), ONNX Runtime CPU |
| B | Stretch to 448×448, ignoring aspect ratio, ONNX Runtime CPU |
| C | Official upstream preprocessing at 518, PyTorch CPU. **This is the reference behaviour, not ground truth.** |

**Images:** 5 pinned upstream demo images, at their original aspect ratio (about 3:2) and as 1280×720 centre crops (16:9).

| Mean over images | Pearson A~C | Pearson B~C | Spearman A~C | Spearman B~C | Normalized mean abs diff A~C | Normalized mean abs diff B~C |
|---|---|---|---|---|---|---|
| Original aspect | 0.975 | 0.984 | 0.965 | 0.975 | 0.035 | 0.031 |
| 16:9 crops | 0.967 | 0.981 | 0.952 | 0.973 | 0.053 | 0.040 |

**Observations (no accuracy claims):**
- On these images, B's output is closer to upstream's than A's in 9 of 10 cases; demo10 at its original aspect is the exception. A plausible cause, not established: for a 16:9 frame, letterbox uses only a 448×252 region (about 576 ViT patches), while stretch uses 1,024. That gives more vertical detail, at the cost of squeezing the content horizontally 1.78×.
- All three methods produce visibly similar depth structure. Differences sit mostly at object edges and thin structures (net strings, bottle outlines, bike frames). A's edges look softer and coarser.
- **B:** on demo16 (backlit), B places the left-side person and bike visibly nearer than C does.
- **A and B:** on demo10, both miss a near door frame at the right edge that C shows.
- **Letterbox boundary:** no artifact was seen. For A, the top and bottom 10% bands differ from C *less* than the interior does (0.042 vs 0.055 on 16:9).
- **Faces were not tested.** No demo image contains a webcam-style face, so the effect of B's horizontal squeeze on faces is unknown.
- **Latency:** PyTorch CPU takes 315 ms for 448×448 versus 1,042 ms for upstream's 518×924.

**Decision:** letterbox **remains provisional**. Before Phase 3, compare A and B on
real webcam frames with faces, hair, shoulders and varied lighting. Put those frames in
`third_party/assets/local_frames/`; they stay local and git-ignored, and
`compare_preprocessing.py` picks them up automatically. The contract changes
only if that comparison gives a clear reason.

## Output semantics

| Property | Value |
|---|---|
| Shape / dtype | [1, 448, 448] float32 |
| Meaning | **Relative, disparity-like** depth, not metres. Scale and offset vary per image. |
| Direction | **Larger = nearer.** Checked visually on 5 demo images and confirmed by the developer (2026-10-23). |
| Range | ≥ 0, because of the in-graph ReLU. There is no fixed upper bound: the maxima seen on the demo images range from 3.1 to 6.5. |
| Zeros | Far regions are often **exactly 0** after the ReLU (sky, blurred background): the 2nd percentile was 0 for 3 of 5 demo images. Later normalization must cope with large flat zero areas. |
| Postprocessing in the graph | ReLU only |
| Postprocessing outside the graph | Crop the letterbox padding, then resize back to source size with the same bilinear rule. Display rendering clips to the 2nd–98th percentile, **for visualization only**. The raw tensor is kept. |

Per-image observations (brighter = larger value = nearer):

| Image | Brightest | Darkest |
|---|---|---|
| demo01 (street) | Cars at bottom right, near road | Sky, distant tower |
| demo03 (still life) | Front cylinder and block, table edge | Back wall |
| demo07 (basketball) | Ball, net and rim | Blurred background (exactly 0) |
| demo10 (room) | Glass tabletop, chair back | Windows, far wall |
| demo16 (backlit street) | Ground in front | Sky; people mid-grey |

## PyTorch ↔ ONNX Runtime parity

- `scripts/check_parity.py` feeds the **same preprocessed float32 tensor** to PyTorch CPU and ONNX Runtime 1.31.0 CPU, both in FP32.
- It does this for both the letterbox and stretch methods, over 6 generated fixtures and 5 demo images (22 comparisons).
- Generated fixtures check numeric agreement only; they are never treated as depth ground truth.

| Limit | Hard ceiling (fixed beforehand) | Working tolerance (calibrated once at 10× observed) | Worst observed |
|---|---|---|---|
| max abs | — | 2e-3 | 1.27e-4 |
| max abs ÷ output range | 1e-3 | 4e-4 | 3.27e-5 |
| mean abs ÷ output range | 1e-4 | 2e-5 | 1.41e-6 |

- Pearson r was ≥ 0.999999 on every comparison, and every output was finite. Results: `docs/parity_results.json`.
- **Why these ceilings:** FP32 differences from operator fusion and reduction order are usually around 1e-5 of the output range. 1e-3 of the range is still below one 8-bit display step (1/255 ≈ 3.9e-3).
- The tolerances live in `scripts/relume_model/tolerances.py`. They are **never raised automatically**: a result above them is a failure to investigate.

## CPU latency (informational)

| | p50 | p95 |
|---|---|---|
| PyTorch CPU | 316 ms | 322 ms |
| ONNX Runtime CPU | 328 ms | 338 ms |

- **Conditions:** 1×3×448×448, 5 warm-up runs then 50 timed runs, Apple M2 (16 GB), macOS 14.8.9, Python 3.14.7, torch 2.14.1 (4 threads), ONNX Runtime 1.31.0 with the default thread count. Measured on a laptop, not a controlled benchmark.
- **About 3 inferences per second on the CPU is far below the 30 FPS goal.** CoreML (Phase 3) has not been measured, and nothing here predicts its speed.

## Limitations

- **Letterbox choice:** provisional. The comparison above has no faces or webcam frames.
- **Depth direction:** checked visually on 5 images; this is not a metric evaluation.
- **CoreML:** not exercised. Phase 0/1 show only that the CoreML provider is available.
- **Downscale aliasing:** bilinear downscaling by more than 2× aliases (for example 1920→448 is 4.3×). This is accepted for now and covered by the round-trip tests, not optimized.
- **Not yet done:** C++ preprocessing and inference are Phase 3. FP16 and INT8 are out of scope.
