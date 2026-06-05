"""Run Stage 5 (PoseConv3D shot classifier) on both cameras of a session in parallel.

Server-side orchestrator (per flags.md F14) — local entrypoint just calls
``orchestrate.remote(...)`` once and downloads results.

Usage:
    uv run modal run --detach scripts/run_stage5_session.py::run \
        --session 2026-06-02_1on1 --shoot-threshold 0.75
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from courtvision.modal_app import VOLUMES, app, image
# The Stage 5 entrypoint we added below the existing action_overlay:
from scripts.stage5_modal import action_overlay_session  # noqa: F401 (imported for registration)


@app.function(image=image, volumes=VOLUMES, timeout=3600 * 2, cpu=1)
def orchestrate_stage5(
    session: str, shoot_threshold: float = 0.75, frame_interval: int = 1,
) -> dict[str, Any]:
    """Spawn Stage 5 inference + overlay on both cameras in parallel."""
    print(f"=== Stage 5 — spawning both cameras at threshold {shoot_threshold}, "
          f"frame_interval={frame_interval} ===")
    calls = {}
    for cam in ("cam_a", "cam_b"):
        calls[cam] = action_overlay_session.spawn(
            session, cam, shoot_threshold=shoot_threshold, frame_interval=frame_interval,
        )
        print(f"  spawned {cam}")
    results = {cam: call.get() for cam, call in calls.items()}
    for cam, r in results.items():
        print(f"  ✓ {cam}: events={r['events_relpath']}  overlay={r['overlay_relpath']}  "
              f"n_events={r['n_events']}")
    return {
        "session": session, "shoot_threshold": shoot_threshold,
        "frame_interval": frame_interval, "results": results,
    }


@app.local_entrypoint()
def run(
    session: str = "2026-06-02_1on1",
    shoot_threshold: float = 0.75,
    frame_interval: int = 1,
    tag: str = "",
) -> None:
    print(f"Kicking off Stage 5 orchestrator (session={session}, thr={shoot_threshold}, "
          f"frame_interval={frame_interval})...")
    result = orchestrate_stage5.remote(
        session, shoot_threshold=shoot_threshold, frame_interval=frame_interval,
    )

    out_dir = Path(f"outputs/sessions/{session}")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""

    pulls: list[tuple[str, Path]] = []
    for cam, r in result["results"].items():
        pulls.append((r["events_relpath"], out_dir / f"shotevents_{cam}{suffix}.json"))
        pulls.append((r["overlay_relpath"], out_dir / f"stage5_overlay_{cam}{suffix}.mp4"))

    print(f"\nDownloading {len(pulls)} files to {out_dir}...")
    for rel, dest in pulls:
        cmd = ["uv", "run", "modal", "volume", "get",
               "courtvision-artifacts", rel, str(dest), "--force"]
        print(f"  pulling {rel}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"    ⚠ failed: {proc.stderr.strip()}")
    print(f"\n✓ Stage 5 outputs in {out_dir}")
