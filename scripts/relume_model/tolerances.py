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

WORKING_TOLERANCES = None
