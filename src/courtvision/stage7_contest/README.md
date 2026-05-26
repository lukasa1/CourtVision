# Stage 7 — Contest Measurement

**Status:** stub — weeks 3-4 (needs Stage 4 + Stage 6 outputs).

## 5 measurements at release (Decision 9, PRD §9.7)

1. **3D wrist-to-trajectory distance** (headline novel metric): min Euclidean distance from each defender's near-side wrist to the ball's 3D trajectory polyline over the release-window frames. Signed (positive = obstruction side). Returns `None` if no defender within 3 m at release.

2. **Wrist height advantage**: defender's highest wrist z − shooter's release-wrist z at release frame. Negative ⇒ defender below ball release height.

3. **Shooter velocity, decomposed**: (forward, lateral, vertical) m/s averaged over a small window centered on release.

4. **Floor distance**: SportVU-style 2D footprint distance between defender and shooter at release. Kept for comparability with public NBA stats.

5. **Composite**: weighted blend of (1)-(4) into a single `composite_score`. Weights TBD via small calibration set.

**Flags** (`ContestMeasurement.flags`):
- `uncontested` — no defender within 3 m
- `id_swap` — track ID inconsistency around release
- `weak_triangulation` — `reprojection_error_px > threshold` at release

## I/O (PRD §7.7)

Writes `contest_{session_id}.json`. Schema: `ContestMeasurement` (+ `ContestMeasurementSequence` wrapper).
