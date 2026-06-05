"""Compare Stage 5 detection metrics across two frame_interval runs.

Used to evaluate F27 — the train/deploy fps mismatch fix. Compares the baseline
(frame_interval=1, native 60 fps inference) against the corrected run
(frame_interval=2, matching the 25 fps MultiSubjects training window).

Usage:
    uv run python scripts/eval_stage5_fi_compare.py --session 2026-06-02_1on1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make sibling scripts importable when this is run from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_session import evaluate_stage5  # noqa: E402


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _eval_pair(session: str, suffix: str, gt: dict, drift_slope: float, thr: float):
    out_dir = Path(f"outputs/sessions/{session}")
    sa_path = out_dir / f"shotevents_cam_a{suffix}.json"
    sb_path = out_dir / f"shotevents_cam_b{suffix}.json"
    if not sa_path.exists() or not sb_path.exists():
        return None
    sa, sb = _load(sa_path), _load(sb_path)
    return evaluate_stage5(sa, sb, drift_slope, gt, tolerance_frames=30)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--session", default="2026-06-02_1on1")
    args = p.parse_args()

    out_dir = Path(f"outputs/sessions/{args.session}")
    gt = _load(Path(f"data/sessions/{args.session}/shots_gt.json"))
    trim = _load(Path(f"data/sessions/{args.session}/trim_metadata.json"))
    drift_slope = trim["drift_slope_b_per_a"]

    print("=" * 78)
    print(f" STAGE 5 — frame_interval comparison (session={args.session})")
    print("=" * 78)

    print("\n  GT shots:", gt["n_shots"], "  drift_slope:", drift_slope)

    rows = []
    for suffix, label in [("", "fi=1 (baseline, 60 fps, ~1.07 s window)"),
                          ("_fi2", "fi=2 (F27 fix, ~2.13 s window)")]:
        res = _eval_pair(args.session, suffix, gt, drift_slope, thr=0.75)
        if res is None:
            print(f"\n  {label}: ✗ missing shotevents_cam_*_{suffix or 'baseline'}.json")
            continue
        s = res["summary"]
        rows.append((label, s))
        print(f"\n  {label}")
        print(f"    n_predicted_after_fusion : {s['n_predicted_events_after_fusion']}")
        print(f"    TP / FP / FN             : {s['true_positives']} / {s['false_positives']} / {s['false_negatives']}")
        print(f"    Precision                : {s['precision']:.3f}")
        print(f"    Recall                   : {s['recall']:.3f}")
        print(f"    F1                       : {s['f1']:.3f}")

    if len(rows) == 2:
        b, f = rows[0][1], rows[1][1]
        print("\n  Δ (fi=2 − fi=1):")
        print(f"    Precision : {f['precision'] - b['precision']:+.3f}")
        print(f"    Recall    : {f['recall'] - b['recall']:+.3f}")
        print(f"    F1        : {f['f1'] - b['f1']:+.3f}")
        print(f"    n_events  : {f['n_predicted_events_after_fusion'] - b['n_predicted_events_after_fusion']:+d}")
        print(f"    TP        : {f['true_positives'] - b['true_positives']:+d}")
        print(f"    FP        : {f['false_positives'] - b['false_positives']:+d}")
        print(f"    FN        : {f['false_negatives'] - b['false_negatives']:+d}")

    # Save the comparison JSON
    out_path = out_dir / "stage5_fi_comparison.json"
    out_path.write_text(json.dumps({
        "session": args.session,
        "runs": {(suf or "fi1"): s for (suf, s) in (
            ("fi1", rows[0][1] if len(rows) >= 1 else None),
            ("fi2", rows[1][1] if len(rows) >= 2 else None),
        ) if s is not None},
    }, indent=2))
    print(f"\n✓ Saved comparison → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
