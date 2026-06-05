"""Render each labeled shot as a side-by-side PNG for visual sanity check.

Reads one or more shots_gt[_partN].json files and produces:
    review/shot_NNN.png — Camera A | Camera B at the release frame, with
                          colored markers + labels for each clicked wrist:
                            red    = shooter (S-A / S-B)
                            blue   = defender wrist 1 (D1-A / D1-B)
                            magenta= defender wrist 2 (D2-A / D2-B)
                          And a caption summarizing release frames, make/miss,
                          shot type, contested/uncontested.
    review/index.html  — quick gallery to flip through all shots in a browser

Usage:
    # Review one part's labels:
    uv run python scripts/review_shots.py --gt data/sessions/2026-06-02_1on1/shots_gt_part1.json

    # Review the merged file:
    uv run python scripts/review_shots.py --gt data/sessions/2026-06-02_1on1/shots_gt.json

    # Review all parts in a folder (auto-globs shots_gt*.json, skipping the merged):
    uv run python scripts/review_shots.py --gt-dir data/sessions/2026-06-02_1on1/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


COLOR_SHOOTER = (0, 0, 255)     # red (BGR)
COLOR_DEF1 = (255, 128, 0)      # blue-ish
COLOR_DEF2 = (255, 0, 255)      # magenta


def read_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    return img if ok else None


def draw_click(img: np.ndarray, pt: list[float] | None, color: tuple[int, int, int],
               label: str) -> None:
    if pt is None:
        return
    x, y = int(round(pt[0])), int(round(pt[1]))
    # White halo + filled colored dot makes it visible on any background
    cv2.circle(img, (x, y), 14, (255, 255, 255), 2)
    cv2.circle(img, (x, y), 9, color, -1)
    cv2.circle(img, (x, y), 9, (255, 255, 255), 1)
    # Label with high-contrast text bg
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    tx, ty = x + 18, y - 12
    cv2.rectangle(img, (tx - 2, ty - th - 4), (tx + tw + 2, ty + 4), (0, 0, 0), -1)
    cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def render_shot(shot: dict[str, Any], cap_a: cv2.VideoCapture, cap_b: cv2.VideoCapture,
                out_path: Path) -> bool:
    fa = shot["release_frame_a"]
    fb = shot["release_frame_b"]
    img_a = read_frame(cap_a, fa)
    img_b = read_frame(cap_b, fb)
    if img_a is None or img_b is None:
        print(f"  ⚠ shot {shot['shot_id']}: could not read frame A={fa} or B={fb}")
        return False

    # Annotate Camera A
    draw_click(img_a, shot.get("shooter_wrist_a"), COLOR_SHOOTER, "S")
    draw_click(img_a, shot.get("defender_wrist1_a"), COLOR_DEF1, "D1")
    draw_click(img_a, shot.get("defender_wrist2_a"), COLOR_DEF2, "D2")
    # Annotate Camera B
    draw_click(img_b, shot.get("shooter_wrist_b"), COLOR_SHOOTER, "S")
    draw_click(img_b, shot.get("defender_wrist1_b"), COLOR_DEF1, "D1")
    draw_click(img_b, shot.get("defender_wrist2_b"), COLOR_DEF2, "D2")

    # Camera title bars
    def header(img: np.ndarray, text: str) -> None:
        cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    header(img_a, f"A  frame {fa}")
    header(img_b, f"B  frame {fb}")

    # Pad to equal heights (both should already be 1920x1080, but defensive)
    h = max(img_a.shape[0], img_b.shape[0])
    def pad_to(img: np.ndarray, target_h: int) -> np.ndarray:
        if img.shape[0] == target_h: return img
        pad = np.zeros((target_h - img.shape[0], img.shape[1], 3), dtype=img.dtype)
        return np.vstack([img, pad])
    img_a = pad_to(img_a, h)
    img_b = pad_to(img_b, h)

    side_by_side = np.hstack([img_a, img_b])

    # Caption strip at the bottom
    cap_h = 90
    caption = np.zeros((cap_h, side_by_side.shape[1], 3), dtype=np.uint8)
    miss_str = "MAKE" if shot.get("make") else "MISS"
    miss_color = (0, 255, 0) if shot.get("make") else (0, 0, 255)
    type_str = shot.get("shot_type", "?")
    cont_str = "UNCONTESTED" if shot.get("uncontested") else "contested"
    line1 = f"Shot {shot.get('shot_id', '?')}   {type_str}   {cont_str}"
    cv2.putText(caption, line1, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(caption, miss_str, (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.95, miss_color, 2)

    # Coverage summary on caption
    fields = [
        ("S-A", shot.get("shooter_wrist_a")),
        ("S-B", shot.get("shooter_wrist_b")),
        ("D1-A", shot.get("defender_wrist1_a")),
        ("D1-B", shot.get("defender_wrist1_b")),
        ("D2-A", shot.get("defender_wrist2_a")),
        ("D2-B", shot.get("defender_wrist2_b")),
    ]
    coverage = " ".join([f"{n}:{'Y' if v else '-'}" for n, v in fields])
    cv2.putText(caption, coverage, (220, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    final = np.vstack([side_by_side, caption])

    # Downscale to a reasonable file size if the source is large
    if final.shape[1] > 2400:
        scale = 2400 / final.shape[1]
        new_w = 2400
        new_h = int(final.shape[0] * scale)
        final = cv2.resize(final, (new_w, new_h), interpolation=cv2.INTER_AREA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), final)
    return True


def write_index_html(out_dir: Path, shots: list[dict[str, Any]], png_paths: list[Path]) -> None:
    """Simple gallery."""
    rows = []
    for shot, png in zip(shots, png_paths):
        miss = "MAKE" if shot.get("make") else "MISS"
        cont = "UNCONTESTED" if shot.get("uncontested") else "contested"
        rows.append(
            f'<div style="margin: 24px 0; padding: 8px; border: 1px solid #ccc;">'
            f'<div style="font-family: monospace; padding: 4px 0;">'
            f'Shot {shot["shot_id"]} — release_a={shot["release_frame_a"]} — '
            f'{shot.get("shot_type", "?")} — {miss} — {cont}</div>'
            f'<img src="{png.name}" style="max-width: 100%; height: auto;"/>'
            f'</div>'
        )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Shot label review</title></head>"
        f"<body style='font-family: -apple-system, system-ui, sans-serif; max-width: 1400px; margin: 24px auto;'>"
        f"<h1>Shot Label Review — {len(shots)} shots</h1>"
        + "".join(rows) +
        "</body></html>"
    )
    (out_dir / "index.html").write_text(html)


def load_shots(gt_paths: list[Path]) -> tuple[list[dict[str, Any]], Path, Path]:
    """Load and merge shot lists from one or more GT JSONs."""
    all_shots: list[dict[str, Any]] = []
    video_a = None; video_b = None
    for path in gt_paths:
        data = json.loads(path.read_text())
        if video_a is None:
            video_a = Path(data["video_a"])
            video_b = Path(data["video_b"])
        for s in data.get("shots", []):
            all_shots.append(s)
    if video_a is None or video_b is None:
        raise RuntimeError("No GT files contained video paths.")
    return all_shots, video_a, video_b


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--gt", type=Path, action="append", help="One or more GT JSON file(s). Repeat for multiple.")
    grp.add_argument("--gt-dir", type=Path, help="Folder containing shots_gt_part*.json — review all parts.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output folder for review PNGs (default: <gt-folder>/review/)")
    args = p.parse_args()

    if args.gt_dir:
        gt_paths = sorted(args.gt_dir.glob("shots_gt_part*.json"))
        if not gt_paths:
            print(f"⚠ No shots_gt_part*.json found in {args.gt_dir}")
            return 1
    else:
        gt_paths = args.gt

    print(f"Loading {len(gt_paths)} GT file(s):")
    for p_ in gt_paths:
        print(f"  - {p_}")
    shots, video_a, video_b = load_shots(gt_paths)
    print(f"Total shots: {len(shots)}")
    if not shots:
        print("No shots labeled yet."); return 0

    out_dir = args.out_dir or gt_paths[0].parent / "review"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output → {out_dir}")

    # Resolve videos relative to GT directory if necessary
    # The labeling script stores them as raw strings; try both as-is and relative to gt dir
    def resolve(v: Path) -> Path:
        if v.is_absolute() and v.exists(): return v
        if v.exists(): return v.resolve()
        candidate = gt_paths[0].parent / v
        if candidate.exists(): return candidate.resolve()
        return v

    video_a = resolve(video_a)
    video_b = resolve(video_b)
    cap_a = cv2.VideoCapture(str(video_a))
    cap_b = cv2.VideoCapture(str(video_b))
    if not cap_a.isOpened():
        print(f"⚠ cannot open {video_a}"); return 1
    if not cap_b.isOpened():
        print(f"⚠ cannot open {video_b}"); return 1

    png_paths = []
    for shot in shots:
        out_path = out_dir / f"shot_{shot['shot_id']:03d}.png"
        ok = render_shot(shot, cap_a, cap_b, out_path)
        if ok:
            png_paths.append(out_path)
            print(f"  ✓ shot {shot['shot_id']:3d} → {out_path.name}")
    cap_a.release(); cap_b.release()

    write_index_html(out_dir, shots, png_paths)
    print(f"\n✓ Rendered {len(png_paths)} shot reviews → {out_dir}")
    print(f"  Open the folder in Finder, or:")
    print(f"    open '{out_dir}/index.html'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
