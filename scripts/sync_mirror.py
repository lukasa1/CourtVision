"""Pull a tiny per-dataset subset onto the laptop for dataloader iteration.

PRD §8.4. Run once after Modal Volumes are populated. Re-run only if you
want to change which clips live locally. Wraps ``modal volume get``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from courtvision.paths import repo_root

# (dataset, relpath on Volume) → local dest
_MIRROR_FILES = [
    # Filled during Phase A user-test step once datasets land on Modal.
    # Example:
    # ("multisubjects", "subset_for_local_dev/clip_001.mp4"),
]


def main() -> int:
    if not _MIRROR_FILES:
        print("nothing to mirror — populate _MIRROR_FILES in scripts/sync_mirror.py")
        return 0
    mirror_root = repo_root() / "data" / "mirror"
    for dataset, relpath in _MIRROR_FILES:
        dest = mirror_root / dataset / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "modal", "volume", "get",
            "courtvision-data", f"{dataset}/{relpath}", str(dest),
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
