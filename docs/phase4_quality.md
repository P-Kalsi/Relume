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

## Session 1 (2026-11-02): paused after Section 2 at the developer's request

FaceTime HD, Release, CoreML CPUAndGPU, default controls. Ratings so far:

| Item | Result |
|---|---|
| 1a Status label (relit / bypassed) | pass |
| 1b Shape-following shading | pass, but subtle |
| 1c Facial detail kept | pass |
| 1d Halos | **fail (open issue 1):** during head motion, a ghost head outline on the wall where the head was, and a sharp vertical seam across the face. Seen in one screenshot the developer approved Claude to view (stays in the developer's Downloads; not copied). Likely cause, not yet measured: the distance term uses the depth height, which steps sharply at a silhouette; with about 180 ms of depth lag, the step lands on the old silhouette position. Confidence softens only the normals. |
| 1e Stable when still | pass |
| 1f Bypass instant, passthrough normal | pass |
| 2a Left/right follows the marker | pass |
| 2b Up/down follows the marker | pass |
| 2c Face-shaped shading | **fail (open issue 2):** reads as a brightness change over the whole face area, not face-shaped light |
| 2d Relief 0 vs 1.5 | weak: more light, not very different (consistent with issue 2) |
| 2e Normals view as described | pass (developer's remark "the input looks the same as the output views" to be clarified) |

Not yet done: Sections 3-8.

### Open issue 1: first measurements (2026-11-09)
- A **Depth falloff** control (0-1, default 1, unchanged lighting) sets how
  much depth height enters the light's distance.
- Across a depth edge, the gain steps by 0.18 at falloff 1 and 0 at 0
  (`test_lighting_math`).
- In the moving-head ablation scene, the sharpest gain step falls only from
  0.283 to 0.235 (`relume_relight --depth-falloff 0`). The distance falloff
  explains part of the seam; most of the rest comes from the normals at the
  head's own silhouette.
- **Silhouette confidence (2026-11-12):** normals now fade to camera-facing
  within 6 texels of a depth jump (range of h over 24 samples above 0.12).
  The seam metric drops from 0.283 to 0.215 at falloff 1, and to 0.176 with
  falloff 0. It is 24-38% lower but not gone: what remains is the
  head-vs-wall difference itself while the depth lags (about 180 ms). The
  face's interior shading is unchanged, and every approved temporal threshold
  still passes.
- Still to do: check it live (Section 1d again, plus head movement in
  Section 5). If the seam is still visible, the remaining options are a
  screen-space blur of the gain across silhouettes or motion compensation
  (Phase 5).

### Open issue 2: measurements (2026-11-13)
From `relume_relight` on the two local bursts (numbers only):
- **The face is not flat in the depth.** Surface tilt over the subject at the
  default relief: p50 17-18 deg, p90 29-34 deg (the synthetic person: 6 deg).
- **The lighting is dominated by a position gradient, not by shape.** At the
  default light, the subject-vs-background difference is small (0.018-0.030
  gain). The spread within the subject is 0.091-0.097, and with relief 0 (no
  shape at all) it is still 0.086-0.088. The shape term adds only about 5-10%;
  estimated separately, about 0.035 against 0.087 for the position term.
- **Conclusion:** "brightness over the face area" comes mostly from the
  light's distance and angle varying across the image, not from too little
  depth relief.
- **Candidate fixes (to choose with the developer):**
  - a more distant, directional key light (weaker attenuation and position
    dependence; Z 4.0 already approximates this);
  - less wrap in the Lambert term, for more shape contrast;
  - a relief gain targeted at the subject.

  Each is measurable with this split before a camera check.

Live numbers (session 1, stats intervals after Ready):
- capture and composed 30.0 fps; pool drops 0; render loop p95 17.1 ms, max 18.1 ms;
- depth 6.3-6.5 results/s, age p50 182 ms, lag p50 5 frames;
- relight GPU p50 0.7-0.8 ms, p95 2.2-2.65 ms; depth field GPU p50 0.8-0.95 ms;
- stabilize p50 2.3-2.35 ms, p95 2.5-2.75 ms (under 3 ms);
- memory 710-712 MiB;
- **662 scene changes out of 30,892 depth fields (2.1%)**, each a hard cut without a crossfade (open issue 3: check whether motion triggers them and whether they are visible). Since `9e76efc`/`1fe5c69` the stats log counts scene changes by rejection reason (too few pixels, flat, no inliers, few inliers, high residual, scale); the next camera session will show which check fires.

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
