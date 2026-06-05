"""Orchestrate Stage 1 + Stage 2 + overlays on Modal (server-side, parallel per camera).

Key design point — the orchestrator itself is a server-side ``@app.function``
because the local-client model (calling ``.spawn()`` + ``.get()`` from a local
entrypoint) breaks on long-running pipelines: the local heartbeat dies and
Modal terminates the app while we wait (per flags.md F14). Running the
orchestrator inside Modal makes its lifetime independent of the laptop.

Order:
    1. Stage 1 (YOLO+ByteTrack) on A and B in parallel              [GPU T4]
    2. Stage 2 (RTMPose-l via rtmlib/ONNX) on A and B in parallel   [CPU]
    3. Stage 1 + Stage 2 overlay videos on A and B in parallel      [CPU]

Outputs land on the ``courtvision-artifacts`` volume; the local entrypoint
downloads them to ``outputs/sessions/<session>/`` after the orchestrator
returns.

Usage:
    uv run modal run scripts/run_session_pipeline.py::run --session 2026-06-02_1on1
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from courtvision.modal_app import (
    VOLUMES,
    app,
    image,
    make_stage1_overlay,
    make_stage2_overlay,
    run_rtmpose,
    run_yolo_bytetrack,
)


_STAGE1_CONFIG = {
    "model": {
        "name": "yolov8m",
        "weights": "/artifacts/checkpoints/yolov8m.pt",
        "classes": [0],
        "conf": 0.4,
        "iou": 0.5,
        "imgsz": 1280,
    },
    "tracker": {"name": "bytetrack", "cfg": "bytetrack.yaml"},
}

_STAGE2_CONFIG = {
    "model": {
        "name": "rtmpose-l_halpe26",
        # The KEY rtmlib reads is `onnx_model` (not `weights`). Passing a local
        # path bypasses the openmmlab CDN download (the URL was unreachable on
        # 2026-06-05 — see the report's iPhone-session engineering log).
        "onnx_model": "/artifacts/checkpoints/rtmpose_l_halpe26.onnx",
        "input_size": [192, 256],
    },
}


@app.function(image=image, volumes=VOLUMES, timeout=3600 * 2, cpu=1)
def orchestrate_session(session: str, skip_stage1: bool = False, skip_stage2: bool = False) -> dict[str, Any]:
    """Server-side orchestrator: Stage 1 → Stage 2 → overlay rendering. Runs entirely
    inside Modal so the lifetime is independent of the local client.

    ``skip_stage1`` / ``skip_stage2`` reuse existing outputs on the artifacts volume
    instead of re-running. Useful when an earlier run completed Stage 1 before a
    later step crashed."""
    cams = [
        ("cam_a", f"sessions/{session}/A_trimmed.mp4"),
        ("cam_b", f"sessions/{session}/B_trimmed.mp4"),
    ]

    if skip_stage1:
        print(f"=== Stage 1 SKIPPED — reusing existing tracks on the volume ===")
        tracks_relpaths = {
            "cam_a": f"stage1/{session}/tracks_{session}_cam_a.json",
            "cam_b": f"stage1/{session}/tracks_{session}_cam_b.json",
        }
        for cam, rel in tracks_relpaths.items():
            print(f"  using /artifacts/{rel}")
    else:
        print(f"=== Stage 1 (tracking) — spawning both cameras ===")
        stage1_calls = {}
        for camera_id, video_relpath in cams:
            cfg = {**_STAGE1_CONFIG, "session_id": session, "camera_id": camera_id}
            stage1_calls[camera_id] = run_yolo_bytetrack.spawn(video_relpath, cfg)
            print(f"  spawned {camera_id}")
        tracks_relpaths = {cam: call.get() for cam, call in stage1_calls.items()}
        for cam, rel in tracks_relpaths.items():
            print(f"  ✓ {cam} tracks → /artifacts/{rel}")

    if skip_stage2:
        print(f"\n=== Stage 2 SKIPPED — reusing existing pose JSONs on the volume ===")
        pose_relpaths = {
            "cam_a": f"stage2/{session}/pose2d_{session}_cam_a.json",
            "cam_b": f"stage2/{session}/pose2d_{session}_cam_b.json",
        }
        for cam, rel in pose_relpaths.items():
            print(f"  using /artifacts/{rel}")
    else:
        print(f"\n=== Stage 2 (pose) — spawning both cameras ===")
        stage2_calls = {}
        for camera_id, video_relpath in cams:
            cfg = {**_STAGE2_CONFIG, "session_id": session, "camera_id": camera_id}
            stage2_calls[camera_id] = run_rtmpose.spawn(video_relpath, tracks_relpaths[camera_id], cfg)
            print(f"  spawned {camera_id}")
        pose_relpaths = {cam: call.get() for cam, call in stage2_calls.items()}
        for cam, rel in pose_relpaths.items():
            print(f"  ✓ {cam} pose → /artifacts/{rel}")

    print(f"\n=== Overlay videos — 4 jobs in parallel ===")
    overlay_calls: dict[str, Any] = {}
    for camera_id, video_relpath in cams:
        s1_out = f"stage1_overlays/{session}_{camera_id}.mp4"
        s2_out = f"stage2_overlays/{session}_{camera_id}.mp4"
        overlay_calls[f"s1_{camera_id}"] = make_stage1_overlay.spawn(
            video_relpath, tracks_relpaths[camera_id], None, s1_out,
        )
        overlay_calls[f"s2_{camera_id}"] = make_stage2_overlay.spawn(
            video_relpath, pose_relpaths[camera_id], None, s2_out,
        )
    overlay_relpaths = {key: call.get() for key, call in overlay_calls.items()}
    for key, rel in overlay_relpaths.items():
        print(f"  ✓ {key} → /artifacts/{rel}")

    return {
        "session": session,
        "tracks_relpaths": tracks_relpaths,
        "pose_relpaths": pose_relpaths,
        "overlay_relpaths": overlay_relpaths,
    }


@app.local_entrypoint()
def run(session: str = "2026-06-02_1on1", skip_stage1: bool = False, skip_stage2: bool = False) -> None:
    """Local entrypoint — one short-lived ``.remote()`` call into the server-side
    orchestrator, then download the artifacts."""
    print(f"Kicking off server-side orchestrator for session={session} "
          f"(skip_stage1={skip_stage1}, skip_stage2={skip_stage2})...")
    result = orchestrate_session.remote(session, skip_stage1=skip_stage1, skip_stage2=skip_stage2)
    print(f"\nOrchestrator returned. Downloading artifacts...")

    out_dir = Path(f"outputs/sessions/{session}")
    out_dir.mkdir(parents=True, exist_ok=True)

    to_pull: list[tuple[str, Path]] = []
    for cam, rel in result["tracks_relpaths"].items():
        to_pull.append((rel, out_dir / f"tracks_{cam}.json"))
    for cam, rel in result["pose_relpaths"].items():
        to_pull.append((rel, out_dir / f"pose2d_{cam}.json"))
    for key, rel in result["overlay_relpaths"].items():
        s_label = "stage1" if key.startswith("s1_") else "stage2"
        cam = key.split("_", 1)[1]
        to_pull.append((rel, out_dir / f"{s_label}_overlay_{cam}.mp4"))

    for rel, local_dest in to_pull:
        cmd = ["uv", "run", "modal", "volume", "get",
               "courtvision-artifacts", rel, str(local_dest), "--force"]
        print(f"  pulling {rel} → {local_dest.name}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"    ⚠ download failed: {r.stderr.strip()}")

    summary = {
        "session": session,
        "tracks": {cam: str(out_dir / f"tracks_{cam}.json") for cam in result["tracks_relpaths"]},
        "pose2d": {cam: str(out_dir / f"pose2d_{cam}.json") for cam in result["pose_relpaths"]},
        "stage1_overlays": {
            "cam_a": str(out_dir / "stage1_overlay_cam_a.mp4"),
            "cam_b": str(out_dir / "stage1_overlay_cam_b.mp4"),
        },
        "stage2_overlays": {
            "cam_a": str(out_dir / "stage2_overlay_cam_a.mp4"),
            "cam_b": str(out_dir / "stage2_overlay_cam_b.mp4"),
        },
    }
    (out_dir / "pipeline_outputs.json").write_text(json.dumps(summary, indent=2))
    print(f"\n✓ Outputs in {out_dir}")
    print(f"  Next: uv run python scripts/triangulate_session.py --session {session}")
