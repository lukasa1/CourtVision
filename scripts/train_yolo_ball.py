"""Train a single-class basketball ball detector (YOLOv8). flags.md F19.

Fine-tunes YOLOv8 on the Roboflow ``basketball-w2xcw`` data already on the
data Volume (``/data/roboflow_basketball/{train,valid,test}/`` with COCO
``_annotations.coco.json``). The COCO ``ball`` class (category_id 1) is the
only class we keep, remapped to YOLO class 0. The dataset is converted to
YOLO format in a container-local dir (ephemeral), then ``YOLO.train`` runs.
Best weights are copied to ``/artifacts/checkpoints/yolo_ball.pt``.

Usage::

    uv run modal run scripts/train_yolo_ball.py::train --epochs 80 --imgsz 1280
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from courtvision.logging import get_logger
from courtvision.modal_app import VOLUMES, app, image

_log = get_logger(__name__)

_DATA_ROOT = "/data/roboflow_basketball"
_BALL_CATEGORY_NAME = "ball"
# Roboflow exports both train/valid/test; YOLO uses train + val.
_SPLIT_MAP = {"train": "train", "valid": "val"}


def _coco_to_yolo(split_dir: Path, out_dir: Path) -> int:
    """Convert one COCO split to YOLO format under ``out_dir``.

    Writes symlinked images + ``<stem>.txt`` labels (YOLO normalized
    ``cls cx cy w h``) keeping only the ``ball`` category, remapped to class 0.
    Returns the number of images written.
    """
    coco_path = split_dir / "_annotations.coco.json"
    data = json.loads(coco_path.read_text())

    cats = {c["id"]: c["name"].lower() for c in data["categories"]}
    ball_ids = {cid for cid, name in cats.items() if name == _BALL_CATEGORY_NAME}
    if not ball_ids:
        raise ValueError(f"no '{_BALL_CATEGORY_NAME}' category in {coco_path}; cats={cats}")

    images = {img["id"]: img for img in data["images"]}
    anns_by_img: dict[int, list[dict[str, Any]]] = {}
    for ann in data["annotations"]:
        if ann["category_id"] in ball_ids:
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

    img_out = out_dir / "images"
    lbl_out = out_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    n = 0
    for img_id, img in images.items():
        fname = img["file_name"]
        iw, ih = float(img["width"]), float(img["height"])
        src = split_dir / fname
        if not src.exists():
            continue
        stem = Path(fname).stem
        # Symlink the image (avoids copying ~GBs); ultralytics reads it fine.
        link = img_out / fname
        if not link.exists():
            link.symlink_to(src)
        lines = []
        for ann in anns_by_img.get(img_id, []):
            x, y, w, h = ann["bbox"]  # COCO: top-left x,y + w,h (pixels)
            cx, cy = (x + w / 2.0) / iw, (y + h / 2.0) / ih
            lines.append(f"0 {cx:.6f} {cy:.6f} {w / iw:.6f} {h / ih:.6f}")
        # Empty label file = background image (valid YOLO negative sample).
        (lbl_out / f"{stem}.txt").write_text("\n".join(lines))
        n += 1
    return n


@app.function(image=image, gpu="A10G", volumes=VOLUMES, timeout=3600 * 6)
def train_yolo_ball_fn(config: dict[str, Any]) -> dict[str, Any]:
    """Build the YOLO dataset + train. Returns metrics + the saved ckpt relpath."""
    import shutil

    from ultralytics import YOLO  # type: ignore[attr-defined]

    from courtvision.modal_app import artifacts_volume, data_volume

    data_volume.reload()
    epochs = int(config.get("epochs", 80))
    imgsz = int(config.get("imgsz", 1280))
    batch = int(config.get("batch", 8))
    base = config.get("base_weights", "/artifacts/checkpoints/yolov8m.pt")
    if not Path(base).exists():
        base = "yolov8m.pt"  # ultralytics auto-download fallback

    ds_root = Path("/root/yolo_ball_ds")
    counts = {}
    for coco_split, yolo_split in _SPLIT_MAP.items():
        counts[yolo_split] = _coco_to_yolo(Path(_DATA_ROOT) / coco_split, ds_root / yolo_split)
    _log.info("yolo-ball dataset: %s", counts)

    data_yaml = ds_root / "data.yaml"
    data_yaml.write_text(
        f"path: {ds_root}\ntrain: train/images\nval: val/images\nnc: 1\nnames: ['ball']\n"
    )

    out_relpath = "checkpoints/yolo_ball.pt"
    out_ckpt = Path("/artifacts") / out_relpath  # best (for use)
    last_ckpt = out_ckpt.parent / "yolo_ball_last.pt"  # resume state, persisted each epoch
    metric_path = out_ckpt.parent / "yolo_ball.metric.json"  # best mAP50 across attempts
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    run_weights = Path("/root/yolo_runs/ball/weights")

    def _commit(trainer: Any) -> None:
        # Modal can preempt the worker mid-run (flags.md F20); the Function is
        # then restarted from scratch. So every epoch we (1) persist last.pt to
        # the Volume for resume, and (2) overwrite best.pt ONLY when this epoch's
        # mAP50 beats the best already on the Volume — so a restarted attempt's
        # early (worse) epochs can't clobber a better checkpoint from before.
        last = getattr(trainer, "last", None)
        if last and Path(last).exists():
            shutil.copy(last, last_ckpt)
        cur = float((getattr(trainer, "metrics", None) or {}).get("metrics/mAP50(B)", 0.0))
        prev = -1.0
        if metric_path.exists():
            try:
                prev = float(json.loads(metric_path.read_text()).get("map50", -1.0))
            except (ValueError, OSError):
                prev = -1.0
        best = getattr(trainer, "best", None)
        if best and Path(best).exists() and cur >= prev:
            shutil.copy(best, out_ckpt)
            metric_path.write_text(json.dumps({"map50": cur}))
        artifacts_volume.commit()

    model = None
    if last_ckpt.exists():
        # Resume from the Volume-persisted state (preemption recovery).
        try:
            run_weights.mkdir(parents=True, exist_ok=True)
            shutil.copy(last_ckpt, run_weights / "last.pt")
            model = YOLO(str(run_weights / "last.pt"))
            model.add_callback("on_fit_epoch_end", _commit)
            model.train(resume=True)
        except Exception as exc:
            _log.warning("resume from last.pt failed (%s); training fresh", exc)
            model = None
    if model is None:
        model = YOLO(base)
        model.add_callback("on_fit_epoch_end", _commit)
        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=int(config.get("patience", 20)),  # early-stop on plateau
            project="/root/yolo_runs",
            name="ball",
            exist_ok=True,
            verbose=True,
        )
    best = run_weights / "best.pt"
    if best.exists():
        shutil.copy(best, out_ckpt)
        artifacts_volume.commit()

    metrics = model.val(data=str(data_yaml), imgsz=imgsz)
    return {
        "checkpoint_relpath": out_relpath,
        "dataset_counts": counts,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }


@app.local_entrypoint()
def train(epochs: int = 60, imgsz: int = 1280, batch: int = 8, patience: int = 20) -> None:
    """Kick off YOLO ball-detector training on Modal (A10G).

    Run detached so the multi-hour job survives a local client disconnect::

        uv run modal run --detach scripts/train_yolo_ball.py::train
    """
    result = train_yolo_ball_fn.remote(
        {"epochs": epochs, "imgsz": imgsz, "batch": batch, "patience": patience}
    )
    print(json.dumps(result, indent=2))
