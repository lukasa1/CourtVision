"""Upload one capture session's trimmed videos + calibration + GT to Modal.

Layout on the `courtvision-data` volume after this runs:

    /data/sessions/2026-06-02_1on1/
        A_trimmed.mp4
        B_trimmed.mp4
        intrinsics_A.json
        intrinsics_B.json
        extrinsics_A.json
        extrinsics_B.json
        sync.json
        trim_metadata.json
        shots_gt.json

All downstream Modal entrypoints (Stage 1, Stage 2, Stage 5) read from this path.

Usage:
    uv run python scripts/upload_session.py                       # default session
    uv run python scripts/upload_session.py --session 2026-06-02_1on1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import modal


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", default="2026-06-02_1on1")
    p.add_argument("--volume", default="courtvision-data")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    session = args.session
    session_dir = Path("data/sessions") / session
    if not session_dir.exists():
        print(f"⚠ Session dir not found: {session_dir}"); return 1

    files: list[tuple[Path, str]] = [
        (session_dir / "trimmed/A_trimmed.mp4",        f"sessions/{session}/A_trimmed.mp4"),
        (session_dir / "trimmed/B_trimmed.mp4",        f"sessions/{session}/B_trimmed.mp4"),
        (session_dir / "shots_gt.json",                f"sessions/{session}/shots_gt.json"),
        (session_dir / "sync.json",                    f"sessions/{session}/sync.json"),
        (session_dir / "trim_metadata.json",           f"sessions/{session}/trim_metadata.json"),
        (session_dir / "extrinsics_A.json",            f"sessions/{session}/extrinsics_A.json"),
        (session_dir / "extrinsics_B.json",            f"sessions/{session}/extrinsics_B.json"),
        (Path("data/calibration/intrinsics_A_checker.json"), f"sessions/{session}/intrinsics_A.json"),
        (Path("data/calibration/intrinsics_B_checker.json"), f"sessions/{session}/intrinsics_B.json"),
    ]

    print(f"\nSession: {session}")
    print(f"Volume:  {args.volume}\n")

    missing = [p for p, _ in files if not p.exists()]
    if missing:
        print("⚠ Missing local files:")
        for m in missing:
            print(f"  - {m}")
        return 1

    total_size_mb = sum(p.stat().st_size for p, _ in files) / 1e6
    print(f"Plan ({total_size_mb:.1f} MB total):")
    for local, remote in files:
        size_mb = local.stat().st_size / 1e6
        print(f"  {local}  →  {args.volume}:{remote}  ({size_mb:.1f} MB)")

    if args.dry_run:
        print("\n(--dry-run; not uploading.)"); return 0

    print(f"\nUploading...")
    vol = modal.Volume.from_name(args.volume, create_if_missing=False)
    with vol.batch_upload(force=True) as batch:
        for local, remote in files:
            print(f"  uploading {local.name}...")
            batch.put_file(str(local), remote)
    print(f"\n✓ Done. {len(files)} files uploaded to {args.volume}:sessions/{session}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
