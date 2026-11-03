# Phase 4 relighting quality checks

The criteria are fixed before the camera session. The developer rates each
item; Claude never sees webcam pixels. Only the numbers (stats logs) and these
ratings are recorded. No frames are saved unless the developer passes
`--save-frames` or `--guided-capture`.

## Automated (already measured)

| Check | Criterion | Result |
|---|---|---|
| Lighting math (`test_lighting_math`) | orientation, light direction, identity, bounds, no NaN | 13/13 pass |
| GPU vs CPU (`test_relight_gpu`) | field ≤ 2e-3, relit ±2, neutral/bypass vs passthrough ±1, step ±1 px | field 1e-3, relit max 1, neutral 0 (16:9) / max 1 (4:3 upscaled), step within 1 px |
| Crossfade (`test_crossfade`, GPU sequence) | no jump at a new result (1e-6), monotonic weight, ownership rule | 11/11 + GPU sequence pass |
| Stabilizer (`test_depth_stabilizer`) | jitter ≤ 10%, noise flicker −40% or better, no drift, resets, no allocation | 2.4%, −59%, no drift, all pass |
| Ablation (`relume_relight`, `docs/phase4_ablation.json`) | F ≤ max(0.5 F_none, ε), S ≤ 1.5, R ≤ 6, G = 0 | A+B+C (α 0.5, T 0.5): F 0.0037 (none 0.0090), S 0.56, R 2, G 0 |

## Live camera (M5): ratings by the developer

Setup: Release build, `open build/release/src/Relume.app --args --stats 10`,
FaceTime HD camera, CoreML CPUAndGPU (default). Default light. Use the
Controls window's Bypass (A/B) and the A/B/C temporal toggles to compare.

Each item is rated **pass** or **fail**, with a note if useful. "Pass"
means:
- the light moves in the expected direction (left drag lights the left side);
- the face stays identifiable, with skin texture and baseline visibility kept;
- there are no neon highlights, no fully black shadows, no halos, and no
  NaN-like speckles;
- there are no visible jumps when depth updates (about 7 per second) and no
  visible ghosting.

| # | Item | Stabilization on (A+B+C) | Off (all three off) | Bypass is clean passthrough |
|---|---|---|---|---|
| 1 | Static: light front (X 0.5, Y 0.3) | | | |
| 2 | Static: light side (drag to X 0.1, then 0.9) | | | |
| 3 | Static: light from top (Y 0.05) and bottom (Y 0.95) | | | |
| 4 | Head: slow turn left and right | | | |
| 5 | Head: fast turn | | | |
| 6 | Hand: raised near the face | | | |
| 7 | Hand: moved in front of the face | | | |
| 8 | Edges: glasses (if available), hair, shoulders | | | |
| 9 | Cluttered background | | | |
| 10 | Sudden: leave the frame and come back | | | |
| 11 | Sudden: cover the camera, then uncover | | | |
| 12 | Switch camera or provider: hard reset to passthrough, no stale lighting | | | |
| 13 | Controls: sliders and drag respond immediately; video stays smooth | | | |
| 14 | Output window: video only (no marker, no text) | | | |

## Live camera numbers (from `[stats]`, 120 s after Ready)

| Measure | Requirement | Result |
|---|---|---|
| Capture / composed | 30.0 fps, 0 pool drops | |
| Render loop p95 | ≤ Phase 3 + 1 ms (17.4 → 18.4 ms) | |
| Relight pass GPU p95 | goal < 2 ms (a goal, not a claim) | |
| Depth field GPU p95 | reported | |
| Stabilize p95 (worker) | < 3 ms | |
| Depth age p50 | unchanged (about 168 ms) | |
| Memory | flat | |
