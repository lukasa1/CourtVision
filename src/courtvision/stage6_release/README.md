# Stage 6 — Release-Frame Extraction

**Status:** stub — implementation in weeks 3-4 (needs own labeled releases for threshold tuning).

## Two implementations planned (head-to-head per Decision 8)

### `rule_forward.py` — 4-condition forward rule

Walk forward through the shot event; release frame is the first frame meeting all of:

1. Both wrists above shoulder line (in 3D world coords).
2. Ball within ε of dominant wrist (≤ 30 cm).
3. Ball's vertical velocity (`v_z`) transitions from positive (rising) to "released" — the frame where ball-wrist distance starts growing and `v_z` stays positive.
4. Elbow extension angle exceeds a learned threshold (≥ 160° as a starting point).

### `rule_backward.py` — backward-trace-from-apex

Find the ball's apex (max z over the event window), then walk backward to the first frame where the ball is within ε of the dominant wrist. This is more robust to wrist occlusion at release.

### Eval

For each event with a ground-truth release frame (hand-labeled own footage):
- Δframes between predicted and GT.
- % within ±2 frames (≈ 33 ms tolerance at 60 fps).

Per-method confidence flagged via `ReleaseFrame.confidence_flag` (high/low).
Fallback: if both rules disagree and confidence is low, use ball apex as a last resort (`method="fallback_apex"`).

## I/O contract (PRD §7.6)

Writes `releases_{session_id}.json`. Schema: `ReleaseFrame` (+ `ReleaseFrameSequence` wrapper).
