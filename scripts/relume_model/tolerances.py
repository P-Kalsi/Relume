"""PyTorch vs ONNX Runtime CPU parity limits.

HARD_CEILINGS were fixed before any measurement and must never be raised:
  rel_max            max |onnx - torch| / (max(torch) - min(torch))
  mean_abs_over_range mean |onnx - torch| / (max(torch) - min(torch))
Rationale: FP32 differences from op fusion and reduction order are normally
around 1e-5 of the output range; 1e-3 of the range is still well below one
8-bit display step (1/255 ~ 3.9e-3).

WORKING_TOLERANCES are set once, from the first observed run, as
min(10 x worst observed, ceiling), and then committed. They are never
recomputed or raised automatically; a result above them is a failure to
investigate. None means "not yet calibrated" (check_parity.py --calibrate).
"""

HARD_CEILINGS = {
    "rel_max": 1e-3,
    "mean_abs_over_range": 1e-4,
}

# Calibrated 2026-10-23 from the first run (Apple M2, macOS 14.8.9, torch 2.14.1,
# onnxruntime 1.31.0; 11 fixtures x 2 methods). Worst observed: max_abs 1.27e-4,
# rel_max 3.27e-5, mean_abs_over_range 1.41e-6 -> 10x, rounded up to 1 s.f.,
# capped at the ceilings.
WORKING_TOLERANCES = {
    "max_abs": 2e-3,
    "rel_max": 4e-4,
    "mean_abs_over_range": 2e-5,
}
