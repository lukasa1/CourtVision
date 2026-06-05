"""Stage 5 Modal infra — dedicated PoseConv3D image + validation. PRD §9.5 / Phase E.

Kept in its own module (not `modal_app.py`) so it doesn't collide with the
Stage 2 session editing the shared image, and so Stage 5's mm-stack is
decoupled from the shared-image F20 numpy/mmcv ABI fix.

Stack (all numpy-1 native): numpy<2 + torch 2.1.2 + mmcv 2.1.0 (via mim) +
mmaction2 1.2.0 **from source**. Unlike mmpose (F20), mmaction2 has no
xtcocotools dep, so numpy<2 is correct — numpy 2.x breaks mmaction2's pose
transforms (`np.Inf` removed) and torch 2.1.2's numpy ABI. mmaction2 is
git-cloned into ``/opt/mmaction2`` because the wheel omits the
`localizers.drn` submodule AND the configs/ dir that `build_poseconv3d` reads.
"""

from __future__ import annotations

from typing import Any

import modal

from courtvision.modal_app import (
    _MODAL_MAX_TIMEOUT_S,
    _WANDB_SECRETS,
    VOLUMES,
    app,
    artifacts_volume,
    data_volume,
    image,
)

_K400 = "/artifacts/checkpoints/poseconv3d_k400.pth"
_SA_DIR = "/data/multisubjects/data/MultiSubjects_SA"
_WORK_DIR = "/artifacts/stage5/work_dir_multisubjects"
_OUT_CKPT = "/artifacts/checkpoints/stage5/stage2_multisubjects.pth"
_CLIP_LEN = 32  # PRD §9.5.1 (T=32), consistent train + inference window

poseconv3d_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0", "git")
    # numpy<2: this is the mmaction2 stack (NOT mmpose), so F20's xtcocotools
    # numpy-2 issue doesn't apply. torch 2.1.2 + mmcv 2.1.0 + mmaction2 1.2.0 are
    # all numpy-1 native — numpy 2.x triggers `np.Inf` removal errors in
    # mmaction2's pose transforms and a torch `_ARRAY_API` ABI warning.
    .pip_install("numpy>=1.26,<2", "torch==2.1.2", "torchvision==0.16.2")
    # `importlib-metadata` explicitly: some openmim resolves omit it and the
    # `mim` entry script imports it at startup → ModuleNotFoundError otherwise.
    .pip_install("openmim", "importlib-metadata")
    .run_commands("mim install mmengine 'mmcv>=2.0.0,<2.2.0'")
    # Light runtime deps. numpy pinned <2 here too, and scipy<1.13 (its newer
    # wheels are numpy-2 built) — otherwise scipy/etc. silently upgrade numpy to
    # 2.x and break mmaction2's pose transforms (`np.Inf`). pandas/decord dropped
    # (not needed for keypoint-only posec3d training).
    .pip_install(
        "einops",
        "opencv-python-headless",
        "pydantic>=2",
        "rich",
        "omegaconf",
        "wandb",
        "scipy<1.13",
        "numpy>=1.26,<2",
    )
    # Install mmaction2 from SOURCE (the pip/mim 1.2.0 wheel is missing the
    # localizers.drn submodule → ModuleNotFoundError when the model registry
    # populates, which breaks both model build and training). Source install is
    # complete AND ships configs/ in the same place build_poseconv3d reads from.
    # --no-deps avoids re-resolving; the final `pip install numpy<2` is the
    # hard last word in case any earlier step slipped numpy 2.x back in.
    .run_commands(
        "git clone --depth 1 --branch v1.2.0 https://github.com/open-mmlab/mmaction2.git /opt/mmaction2",
        "cd /opt/mmaction2 && pip install --no-deps -e .",
        "pip install 'numpy>=1.26,<2'",
    )
    .add_local_python_source("courtvision")
)


@app.function(image=poseconv3d_image, volumes={"/artifacts": artifacts_volume}, timeout=900, cpu=2)
def verify_mmstack() -> dict:
    """Validate the dedicated image: import the mm-stack (the F20 risk) and
    sanity-load the K400 checkpoint. Reports versions + checkpoint structure.
    No GPU — pure import/load check."""
    import numpy as np
    import torch

    report: dict = {"numpy": np.__version__, "torch": torch.__version__}
    for mod in ("mmengine", "mmcv", "mmaction"):
        try:
            m = __import__(mod)
            report[mod] = getattr(m, "__version__", "imported")
        except Exception as exc:
            report[mod] = f"IMPORT FAILED: {type(exc).__name__}: {exc}"

    from pathlib import Path

    ckpt_path = Path(_K400)
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        report["k400_size_bytes"] = ckpt_path.stat().st_size
        if isinstance(ckpt, dict):
            report["k400_top_keys"] = sorted(ckpt.keys())[:8]
            state = ckpt.get("state_dict", ckpt)
            sample_keys = list(state.keys())
            report["k400_n_tensors"] = len(sample_keys)
            report["k400_sample_keys"] = sample_keys[:5]
            # Find the head/fc layer to see the pretrained class count.
            head = [k for k in sample_keys if "cls_head" in k or "fc_cls" in k]
            report["k400_head_keys"] = head[:4]
            for k in head:
                if state[k].ndim == 2:
                    report["k400_head_shape"] = list(state[k].shape)
                    break
    else:
        report["k400"] = "NOT FOUND on /artifacts"
    print(report)
    return report


@app.function(image=poseconv3d_image, volumes={"/artifacts": artifacts_volume}, timeout=900, cpu=4)
def verify_model_forward() -> dict:
    """E1 acceptance: build PoseConv3D (4-class, K400 backbone) and forward a
    synthetic ``[1, 17, 32, 56, 56]`` heatmap tensor → expect ``[1, 4]``, no NaN."""
    import torch

    from courtvision.stage5_classifier.model import build_poseconv3d

    model = build_poseconv3d(
        num_classes=4, pretrained_path="/artifacts/checkpoints/poseconv3d_k400.pth"
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        out = model(torch.zeros(1, 17, 32, 56, 56))
    report = {
        "output_shape": list(out.shape),
        "n_params": n_params,
        "has_nan": bool(torch.isnan(out).any()),
        "output_sample": out.flatten().tolist()[:4],
    }
    print(report)
    return report


@app.local_entrypoint()
def verify() -> None:
    """uv run modal run scripts/stage5_modal.py::verify"""
    import json

    print(json.dumps(verify_mmstack.remote(), indent=2))


@app.local_entrypoint()
def verify_model() -> None:
    """uv run modal run scripts/stage5_modal.py::verify_model"""
    import json

    print(json.dumps(verify_model_forward.remote(), indent=2))


@app.function(image=poseconv3d_image, volumes={"/artifacts": artifacts_volume}, timeout=300)
def clear_ms_workdir() -> str:
    """One-shot: wipe the MultiSubjects work_dir(s) on the Volume (clears the
    debug-contaminated state so the real run starts fresh)."""
    import shutil
    from pathlib import Path

    artifacts_volume.reload()
    removed = []
    for d in (_WORK_DIR, _WORK_DIR + "_debug"):
        if Path(d).exists():
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d)
    artifacts_volume.commit()
    return f"cleared: {removed}"


@app.local_entrypoint()
def clear_workdir() -> None:
    """uv run modal run scripts/stage5_modal.py::clear_workdir"""
    print(clear_ms_workdir.remote())


@app.function(image=poseconv3d_image, gpu="A10G", volumes=VOLUMES, timeout=900)
def infer_demo(min_frames: int = 60) -> dict:
    """E5 acceptance: load the trained classifier and run sliding-window shot
    inference on a held-out MultiSubjects 'shoot' clip → ShotEventSequence.
    Verifies the inference scaffold end-to-end on a clip with a known shot."""
    import glob as _glob
    import pickle
    import zipfile
    from pathlib import Path

    import torch

    from courtvision.io_utils import write_json
    from courtvision.stage5_classifier.infer import run_shot_inference
    from courtvision.stage5_classifier.model import build_poseconv3d

    data_volume.reload()
    artifacts_volume.reload()

    model = build_poseconv3d(num_classes=4)  # fresh; load the TRAINED weights below
    ck = torch.load(_OUT_CKPT, map_location="cpu")
    state = ck.get("state_dict", ck)
    incompat = model.load_state_dict(state, strict=False)
    model = model.cuda().eval()

    sa = "/root/sa_infer"
    with zipfile.ZipFile(f"{_SA_DIR}.zip") as zf:
        zf.extractall(sa)
    val = pickle.load(open(_glob.glob(f"{sa}/**/val.pkl", recursive=True)[0], "rb"))
    shoots = [x for x in val if x["label"] == 2 and x["total_frames"] >= min_frames]
    clip = max(shoots or val, key=lambda x: x["total_frames"])
    kp = clip["keypoint"][0]  # [T, 17, 2]
    ks = clip["keypoint_score"][0]  # [T, 17]

    seq = run_shot_inference(
        model,
        kp,
        ks,
        tuple(clip["img_shape"]),
        session_id=clip["frame_dir"],
        camera_id="msval",
    )
    out_rel = f"stage5/infer/shotevents_{clip['frame_dir']}.json"
    write_json(seq, Path("/artifacts") / out_rel)
    artifacts_volume.commit()
    return {
        "clip": clip["frame_dir"],
        "label_true": clip["label"],
        "total_frames": clip["total_frames"],
        "load_missing_keys": len(incompat.missing_keys),
        "load_unexpected_keys": len(incompat.unexpected_keys),
        "n_shot_events": len(seq.events),
        "events": [e.model_dump() for e in seq.events][:3],
        "out_relpath": out_rel,
    }


@app.local_entrypoint()
def infer() -> None:
    """uv run modal run scripts/stage5_modal.py::infer"""
    import json

    print(json.dumps(infer_demo.remote(), indent=2))


@app.function(image=image, volumes=VOLUMES, timeout=1200)
def probe_ms_videos() -> dict:
    """Feasibility probe: can we pull per-clip mp4s out of the MultiSubjects
    RAR on the Volume (for Stage 5 overlay rendering)? Reports the RAR(s),
    mp4-member count + sample paths, and times a single-member extraction."""
    import glob
    import time

    import rarfile

    data_volume.reload()
    rars = glob.glob("/data/multisubjects/data/*.rar")
    report: dict = {
        "rars": [{"path": r, "gb": round(__import__("os").path.getsize(r) / 1e9, 2)} for r in rars]
    }
    target = "/data/multisubjects/data/MultiSubjects_train_val_test.rar"
    if target not in rars:
        report["error"] = f"{target} not present"
        return report
    rf = rarfile.RarFile(target)
    names = rf.namelist()
    mp4s = [n for n in names if n.endswith(".mp4")]
    report["total_members"] = len(names)
    report["n_mp4"] = len(mp4s)
    report["sample_mp4_members"] = mp4s[:6]
    report["is_solid"] = rf.needs_password() is False and any(
        getattr(i, "compress_type", None) for i in rf.infolist()[:1]
    )
    if mp4s:
        m = mp4s[0]
        t0 = time.time()
        data = rf.read(m)
        report["extract_one"] = {
            "member": m,
            "bytes": len(data),
            "seconds": round(time.time() - t0, 1),
        }
    return report


@app.local_entrypoint()
def probe_videos() -> None:
    """uv run modal run scripts/stage5_modal.py::probe_videos"""
    import json

    print(json.dumps(probe_ms_videos.remote(), indent=2))


@app.function(image=image, volumes=VOLUMES, timeout=1800)
def extract_overlay_clips(min_frames: int = 50) -> list[dict]:
    """Pick one held-out VAL clip per action class (shoot/layup/dribble) with
    real keypoints, and extract its mp4 from the MultiSubjects RAR to the Volume
    (shared image has rarfile + unar). Returns the chosen clips."""
    import glob
    import pickle
    import zipfile
    from pathlib import Path

    import numpy as np
    import rarfile

    data_volume.reload()
    artifacts_volume.reload()
    sa = "/root/sa_ov"
    with zipfile.ZipFile(f"{_SA_DIR}.zip") as zf:
        zf.extractall(sa)
    val = pickle.load(open(glob.glob(f"{sa}/**/val.pkl", recursive=True)[0], "rb"))
    chosen = []
    for label in (2, 1, 0):  # shoot, layup, dribble
        cands = [
            x
            for x in val
            if x["label"] == label
            and x["total_frames"] >= min_frames
            and float(np.asarray(x["keypoint"]).max()) > 0
        ]
        if cands:
            chosen.append(max(cands, key=lambda x: x["total_frames"]))

    rf = rarfile.RarFile("/data/multisubjects/data/MultiSubjects_train_val_test.rar")
    names = rf.namelist()
    src_dir = Path("/artifacts/stage5_overlay_src")
    src_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for c in chosen:
        fd = c["frame_dir"]
        member = next((n for n in names if n.endswith(f"/{fd}.mp4")), None)
        if member is None:
            continue
        (src_dir / f"{fd}.mp4").write_bytes(rf.read(member))
        out.append({"frame_dir": fd, "label": int(c["label"]), "n_frames": int(c["total_frames"])})
    artifacts_volume.commit()
    return out


@app.function(image=poseconv3d_image, gpu="T4", volumes=VOLUMES, timeout=1800)
def render_overlay_clips(clips: list[dict]) -> list[dict]:
    """Render the shot-classifier overlay (skeleton + class-prob bars + shot
    markers) on each pre-extracted clip mp4 using the trained model."""
    import glob
    import pickle
    import zipfile
    from pathlib import Path

    import numpy as np
    import torch

    from courtvision.stage5_classifier.infer import _CLASS_ORDER, _nms_peaks, classify_windows
    from courtvision.stage5_classifier.model import build_poseconv3d
    from courtvision.stage5_classifier.overlay import make_shot_overlay_video

    data_volume.reload()
    artifacts_volume.reload()
    model = build_poseconv3d(num_classes=4)
    state = torch.load(_OUT_CKPT, map_location="cpu").get("state_dict")
    model.load_state_dict(state, strict=False)
    model = model.cuda().eval()

    sa = "/root/sa_ov2"
    with zipfile.ZipFile(f"{_SA_DIR}.zip") as zf:
        zf.extractall(sa)
    val = {
        x["frame_dir"]: x
        for x in pickle.load(open(glob.glob(f"{sa}/**/val.pkl", recursive=True)[0], "rb"))
    }

    out = []
    for c in clips:
        fd = c["frame_dir"]
        x = val[fd]
        kp = np.asarray(x["keypoint"])[0]
        ks = np.asarray(x["keypoint_score"])[0]
        centers, probs = classify_windows(model, kp, ks, tuple(x["img_shape"]))
        shot = [float(p[2] + p[1]) for p in probs]
        ev = [centers[i] for i in _nms_peaks(shot, centers, 0.5, 16)]
        op = f"stage5_overlays/{fd}_shot.mp4"
        make_shot_overlay_video(
            Path(f"/artifacts/stage5_overlay_src/{fd}.mp4"),
            kp,
            ks,
            centers,
            probs,
            Path("/artifacts") / op,
            class_order=_CLASS_ORDER,
            true_label=int(c["label"]),
            event_frames=ev,
        )
        out.append(
            {
                "frame_dir": fd,
                "true": _CLASS_ORDER[int(c["label"])],
                "overlay": op,
                "n_events": len(ev),
            }
        )
    artifacts_volume.commit()
    return out


@app.function(image=poseconv3d_image, gpu="A10G", volumes=VOLUMES, timeout=3600)
def diagnose_sampling(
    n_per_class: int = 40, clip_len: int = 32, stride: int = 0, ckpt: str = _OUT_CKPT
) -> dict:
    """Quantify the train/inference sampling mismatch (the E5 detector gap),
    parametrised by window length so we can sweep it.

    Training/val used ``UniformSampleFrames`` over the WHOLE trimmed clip;
    inference slides CONTIGUOUS ``clip_len``-frame windows. Reports a balanced
    PER-CLASS accuracy gap (uniform whole-clip top-1 vs contiguous-window
    top-1) at the given window length, plus the false-shoot rate on non-shot
    windows. ``stride=0`` -> ``clip_len // 8`` so the ~87.5% overlap is held
    constant across a length sweep (isolates window length from stride).

    NB the checkpoint was trained at clip_len=32, so longer windows are
    themselves slightly OOD; the *gap* (uniform vs contig, both fed the same
    length) controls for that and is the signal to read. The per-class split
    tests whether shots and layups want different window lengths.
    """
    import glob
    import json
    import pickle
    import zipfile
    from collections import Counter

    import numpy as np
    import torch
    from mmaction.datasets.transforms import (
        FormatShape,
        GeneratePoseTarget,
        PoseCompact,
        PoseDecode,
        Resize,
        UniformSampleFrames,
    )

    from courtvision.stage5_classifier.infer import _CLASS_ORDER, classify_windows
    from courtvision.stage5_classifier.model import build_poseconv3d

    if stride <= 0:
        stride = max(1, clip_len // 8)

    data_volume.reload()
    artifacts_volume.reload()
    model = build_poseconv3d(num_classes=4)
    state = torch.load(ckpt, map_location="cpu").get("state_dict")
    model.load_state_dict(state, strict=False)
    model = model.cuda().eval()
    device = next(model.parameters()).device

    sa = "/root/sa_diag"
    with zipfile.ZipFile(f"{_SA_DIR}.zip") as zf:
        zf.extractall(sa)
    val = pickle.load(open(glob.glob(f"{sa}/**/val.pkl", recursive=True)[0], "rb"))

    # Training-style sampling: UniformSampleFrames over the whole clip (the exact
    # _val_pipeline used during training, minus PackActionInputs so we grab the
    # raw heatmap tensor like classify_windows does).
    uniform_pipeline = [
        UniformSampleFrames(clip_len=clip_len, num_clips=1, test_mode=True),
        PoseDecode(),
        PoseCompact(hw_ratio=1.0, allow_imgpad=True),
        Resize(scale=(56, 56), keep_ratio=False),
        GeneratePoseTarget(sigma=0.6, use_score=True, with_kp=True, with_limb=False),
        FormatShape(input_format="NCTHW_Heatmap"),
    ]

    def uniform_prob(ann: dict):  # type: ignore[no-untyped-def]  # returns np.ndarray
        r: dict = {
            "keypoint": np.asarray(ann["keypoint"], dtype=np.float32).copy(),
            "keypoint_score": np.asarray(ann["keypoint_score"], dtype=np.float32).copy(),
            "img_shape": tuple(ann["img_shape"]),
            "modality": "Pose",
            "total_frames": int(ann["total_frames"]),
            "start_index": 0,
            "label": int(ann["label"]),
            "frame_dir": ann["frame_dir"],
        }
        for t in uniform_pipeline:
            r = t(r)
        x = torch.as_tensor(np.asarray(r["imgs"]), dtype=torch.float32, device=device)
        if x.ndim == 5 and x.shape[0] > 1:
            x = x[:1]
        with torch.inference_mode():
            return torch.softmax(model(x), dim=1)[0].cpu().numpy()

    # Balanced per-class sample (index 0/1/2 = dribble/layup/shoot).
    by_label: dict = {0: [], 1: [], 2: []}
    for x in val:
        by_label.setdefault(int(x["label"]), []).append(x)
    rng = np.random.default_rng(0)
    sample: list = []
    for items in by_label.values():
        idx = rng.permutation(len(items))[: min(n_per_class, len(items))]
        sample += [items[i] for i in idx]

    uni_ok = [0, 0, 0]
    uni_n = [0, 0, 0]
    win_ok = [0, 0, 0]
    win_n = [0, 0, 0]
    maj_ok = [0, 0, 0]
    conf = [[0, 0, 0, 0] for _ in range(3)]  # window-level confusion: true(3) x pred(4)
    nonshot_windows = 0
    nonshot_false_shoot = 0
    for ann in sample:
        lab = int(ann["label"])
        uni_n[lab] += 1
        uni_ok[lab] += int(np.argmax(uniform_prob(ann)) == lab)
        kp = np.asarray(ann["keypoint"])[0]
        ks = np.asarray(ann["keypoint_score"])[0]
        _, wp = classify_windows(
            model, kp, ks, tuple(ann["img_shape"]), clip_len=clip_len, stride=stride
        )
        args = [int(np.argmax(p)) for p in wp]
        win_n[lab] += len(args)
        win_ok[lab] += sum(int(a == lab) for a in args)
        maj_ok[lab] += int(Counter(args).most_common(1)[0][0] == lab)
        for a in args:
            conf[lab][a] += 1
        if lab != 2:
            nonshot_windows += len(args)
            nonshot_false_shoot += sum(int(a == 2) for a in args)

    def _acc(ok: list, n: list) -> dict:
        per: dict = {_CLASS_ORDER[c]: round(ok[c] / n[c], 3) if n[c] else None for c in range(3)}
        per["overall"] = round(sum(ok) / sum(n), 3) if sum(n) else None
        return per

    out = {
        "clip_len": clip_len,
        "clip_len_s": round(clip_len / 25.0, 2),
        "stride": stride,
        "n_clips": len(sample),
        "uniform_acc": _acc(uni_ok, uni_n),
        "contig_perwindow_acc": _acc(win_ok, win_n),
        "contig_majorityvote_acc": _acc(maj_ok, uni_n),
        "false_shoot_rate_on_nonshot_windows": (
            round(nonshot_false_shoot / nonshot_windows, 3) if nonshot_windows else None
        ),
        "contig_window_confusion": {
            _CLASS_ORDER[t]: {
                _CLASS_ORDER[p]: (round(conf[t][p] / sum(conf[t]), 3) if sum(conf[t]) else None)
                for p in range(4)
            }
            for t in range(3)
        },
    }
    print(json.dumps(out, indent=2))
    return out


@app.local_entrypoint()
def diag(n_per_class: int = 40, clip_len: int = 32, stride: int = 0, ckpt: str = "") -> None:
    """uv run modal run scripts/stage5_modal.py::diag — quantify the
    uniform-whole-clip vs contiguous-window sampling gap at one window length.
    Pass --ckpt to evaluate a specific checkpoint (default: the 32f model)."""
    import json

    kw = {"ckpt": ckpt} if ckpt else {}
    print(json.dumps(diagnose_sampling.remote(n_per_class, clip_len, stride, **kw), indent=2))


@app.local_entrypoint()
def sweep(n_per_class: int = 40, lengths: str = "64,80,96,128") -> None:
    """uv run modal run scripts/stage5_modal.py::sweep — sweep window length
    (constant ~87.5% overlap) to measure how a longer window affects the
    contiguous-window accuracy per class, plus the dribble->shoot leak (does
    dribble act as background?). Override e.g. --lengths 32,48,64."""
    import json

    cls = [int(x) for x in lengths.split(",")]
    results = list(diagnose_sampling.starmap([(n_per_class, cl, 0) for cl in cls]))
    print(json.dumps(results, indent=2))
    print(
        "\nwin    s   | shoot | layup | dribble-acc | dribble->shoot leak | overall contig"
    )
    for r in results:
        c = r["contig_perwindow_acc"]
        dr = r["contig_window_confusion"]["dribble"]
        print(
            f"{r['clip_len']:>3}f {r['clip_len_s']:>4.2f}s | "
            f"{c['shoot']:.2f}  | {c['layup']:.2f}  | "
            f"{dr['dribble']:.2f}        | {dr['shoot']:.2f}                | {c['overall']:.2f}"
        )


@app.local_entrypoint()
def overlay() -> None:
    """uv run modal run scripts/stage5_modal.py::overlay — make qualitative
    shot-classifier overlays on one held-out clip per class."""
    import json

    clips = extract_overlay_clips.remote()
    print("extracted clips:", json.dumps(clips))
    res = render_overlay_clips.remote(clips)
    print(json.dumps(res, indent=2))
    for r in res:
        print(
            f"pull: uv run modal volume get courtvision-artifacts {r['overlay']} outputs/stage5_overlays/"
        )


@app.local_entrypoint()
def rerender() -> None:
    """uv run modal run scripts/stage5_modal.py::rerender — re-render the three
    qualitative overlays on the clips already extracted to the artifact volume
    (skips the slow RAR extraction). Use after fixing the overlay/inference path."""
    import json

    clips = [
        {"frame_dir": "964_d_1", "label": 0},  # dribble
        {"frame_dir": "446_p_2", "label": 1},  # layup
        {"frame_dir": "442_s_3", "label": 2},  # shoot
    ]
    res = render_overlay_clips.remote(clips)
    print(json.dumps(res, indent=2))
    for r in res:
        print(
            f"pull: uv run modal volume get courtvision-artifacts {r['overlay']} outputs/stage5_overlays/"
        )


@app.function(image=poseconv3d_image, volumes={"/artifacts": artifacts_volume}, timeout=600, cpu=2)
def promote_ckpt(src_tag: str = "64f_bal") -> dict:
    """Promote a tagged checkpoint to the canonical _OUT_CKPT (so downstream
    inference/overlay use it by default), archiving the prior canonical weights
    once as ``*_32f.pth`` for provenance."""
    import shutil
    from pathlib import Path

    artifacts_volume.reload()
    src = _OUT_CKPT.replace(".pth", f"_{src_tag}.pth")
    if not Path(src).exists():
        raise FileNotFoundError(f"tagged checkpoint not found: {src}")
    out: dict = {"from": src, "promoted_to": _OUT_CKPT}
    archive = _OUT_CKPT.replace(".pth", "_32f.pth")
    if Path(_OUT_CKPT).exists() and not Path(archive).exists():
        shutil.copy(_OUT_CKPT, archive)
        out["archived_prev"] = archive
    shutil.copy(src, _OUT_CKPT)
    artifacts_volume.commit()
    return out


@app.local_entrypoint()
def promote(src_tag: str = "64f_bal") -> None:
    """uv run modal run scripts/stage5_modal.py::promote — make a tagged
    checkpoint the canonical model (archives the previous canonical once)."""
    import json

    print(json.dumps(promote_ckpt.remote(src_tag), indent=2))


@app.function(image=poseconv3d_image, gpu="A10G", volumes=VOLUMES, timeout=3600)
def action_overlay_session(
    session: str,
    camera_id: str,
    shoot_threshold: float = 0.75,
    min_track_frames: int = 32,
    nms_gap: int = 24,
    frame_interval: int = 1,
) -> dict:
    """Run the Stage 5 action classifier on one camera of a CAPTURE SESSION.

    Generalized version of ``action_overlay`` — takes session + camera args
    instead of being hardcoded to TrackID3x3 clip paths.

    Inputs (Modal volumes):
        /data/sessions/{session}/{A_trimmed.mp4 or B_trimmed.mp4}
        /artifacts/stage2/{session}/pose2d_{session}_{camera_id}.json

    Outputs (/artifacts):
        stage5/{session}/shotevents_{camera_id}.json
        stage5_overlays/{session}_{camera_id}_action.mp4
    """
    import json
    from collections import defaultdict
    from pathlib import Path

    import cv2
    import numpy as np
    import torch

    from courtvision.schemas import Pose2DSequence
    from courtvision.stage2_pose.halpe_to_coco17 import halpe26_to_coco17
    from courtvision.stage5_classifier.infer import _CLASS_ORDER, _segment_peaks, classify_windows
    from courtvision.stage5_classifier.model import build_poseconv3d
    from courtvision.stage5_classifier.overlay import make_action_overlay_video

    data_volume.reload()
    artifacts_volume.reload()
    model = build_poseconv3d(num_classes=4)
    state = torch.load(_OUT_CKPT, map_location="cpu").get("state_dict")
    model.load_state_dict(state, strict=False)
    model = model.cuda().eval()

    camera_suffix = {"cam_a": "A", "cam_b": "B"}.get(camera_id, camera_id.upper())
    pose_relpath = f"stage2/{session}/pose2d_{session}_{camera_id}.json"
    video_relpath = f"sessions/{session}/{camera_suffix}_trimmed.mp4"
    pose_path = Path("/artifacts") / pose_relpath
    video_path = Path("/data") / video_relpath
    seq = Pose2DSequence.model_validate_json(pose_path.read_text())

    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 60.0
    nframes = int(cap.get(7))
    img_shape = (int(cap.get(4)), int(cap.get(3)))  # (H, W)
    cap.release()

    by_track: dict[int, list] = defaultdict(list)
    for p in seq.poses:
        by_track[p.track_id].append(p)

    track_actions: dict[int, dict[int, tuple]] = {}
    track_events: dict[int, list[tuple[int, float]]] = {}
    track_summary: dict[int, dict] = {}
    flat_events: list[dict] = []

    for tid, tposes in by_track.items():
        tposes = sorted(tposes, key=lambda p: p.frame_idx)
        if len(tposes) < min_track_frames:
            continue
        fmin, fmax = tposes[0].frame_idx, tposes[-1].frame_idx
        present = {p.frame_idx: p for p in tposes}
        n = fmax - fmin + 1
        kp = np.zeros((n, 17, 2), np.float32)
        ks = np.zeros((n, 17), np.float32)
        last: Any = None
        for i in range(n):
            p = present.get(fmin + i)
            if p is not None:
                c = halpe26_to_coco17(p)
                kp[i] = [[k.x, k.y] for k in c.keypoints]
                ks[i] = [k.score for k in c.keypoints]
                last = (kp[i].copy(), ks[i].copy())
            elif last is not None:
                kp[i], ks[i] = last

        centers, probs = classify_windows(model, kp, ks, img_shape, frame_interval=frame_interval)
        centers_arr = np.asarray(centers)
        fa: dict[int, tuple] = {}
        for i in range(n):
            pr = probs[int(np.argmin(np.abs(centers_arr - i)))]
            top = int(np.argmax(pr))
            fa[fmin + i] = (_CLASS_ORDER[top], float(pr[top]))
        track_actions[tid] = fa

        shot_score = [float(p[2]) for p in probs]
        peaks = _segment_peaks(shot_score, centers, shoot_threshold, nms_gap)
        track_events[tid] = [(fmin + centers[i], round(shot_score[i], 3)) for i in peaks]
        for i in peaks:
            flat_events.append({
                "frame_idx": int(fmin + centers[i]),
                "track_id": int(tid),
                "p_shoot": round(float(shot_score[i]), 4),
                "p_layup": round(float(probs[i][1]), 4),
                "p_dribble": round(float(probs[i][0]), 4),
                "p_background": round(float(probs[i][3]), 4),
            })
        shoot_hi = sum(1 for s in shot_score if s >= shoot_threshold)
        track_summary[tid] = {
            "present": len(tposes), "windows": len(centers),
            "shoot>=thr": shoot_hi, "thr": shoot_threshold,
            "n_consolidated_events": len(peaks),
        }

    events_relpath = f"stage5/{session}/shotevents_{camera_id}.json"
    events_path = Path("/artifacts") / events_relpath
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text(json.dumps({
        "session": session, "camera_id": camera_id, "shoot_threshold": shoot_threshold,
        "frame_interval": frame_interval,
        "fps": fps, "n_frames": nframes,
        "n_events": len(flat_events),
        "events": sorted(flat_events, key=lambda e: e["frame_idx"]),
        "per_track_summary": track_summary,
    }, indent=2))

    overlay_relpath = f"stage5_overlays/{session}_{camera_id}_action.mp4"
    make_action_overlay_video(
        video_path, seq, track_actions, Path("/artifacts") / overlay_relpath,
        track_events=track_events,
    )
    artifacts_volume.commit()
    print(json.dumps({
        "session": session, "camera_id": camera_id,
        "events": events_relpath, "overlay": overlay_relpath,
        "n_events": len(flat_events), "tracks": track_summary,
    }, indent=2))
    return {
        "session": session, "camera_id": camera_id,
        "events_relpath": events_relpath, "overlay_relpath": overlay_relpath,
        "n_events": len(flat_events),
    }


@app.local_entrypoint()
def infer_session(
    session: str = "2026-06-02_1on1",
    shoot_threshold: float = 0.75,
    frame_interval: int = 1,
    tag: str = "",
) -> None:
    """Spawn Stage 5 inference on both cameras in parallel, then download outputs.

    Replaces ``scripts/run_stage5_session.py`` (broken: cross-module import of
    ``scripts.stage5_modal`` doesn't work because ``scripts/`` isn't on Modal's
    Python path). This entrypoint lives in the same module as the function it
    spawns, so no cross-module import is needed.

    ``frame_interval=2`` matches the 25 fps training window (2.56 s) at 60 fps
    inference (~2.13 s). See flags.md F27.
    """
    import subprocess
    from pathlib import Path

    print(f"=== Stage 5 — spawning both cameras (session={session}, "
          f"thr={shoot_threshold}, frame_interval={frame_interval}, tag={tag or '∅'}) ===")
    calls = {}
    for cam in ("cam_a", "cam_b"):
        calls[cam] = action_overlay_session.spawn(
            session, cam, shoot_threshold=shoot_threshold, frame_interval=frame_interval,
        )
        print(f"  spawned {cam}")
    results = {cam: call.get() for cam, call in calls.items()}
    for cam, r in results.items():
        print(f"  ✓ {cam}: n_events={r['n_events']}")

    out_dir = Path(f"outputs/sessions/{session}")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""

    pulls: list[tuple[str, Path]] = []
    for cam, r in results.items():
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


@app.function(image=poseconv3d_image, gpu="A10G", volumes=VOLUMES, timeout=1800)
def action_overlay(
    clip: str = "basket_S4T4_post", max_seconds: float = 20.0, min_track_frames: int = 32
) -> dict:
    """Run the Stage 5 action recognizer on one TrackID3x3 clip's Stage 2 pose
    tracks and render a multi-player rolling-action overlay. QUALITATIVE ONLY:
    no background class (labels forced every frame) and cross-domain from
    MultiSubjects → expect noisy labels; this is a sanity check, not validation.
    """
    import json
    from collections import defaultdict
    from pathlib import Path

    import cv2
    import numpy as np
    import torch

    from courtvision.schemas import Pose2DSequence
    from courtvision.stage2_pose.halpe_to_coco17 import halpe26_to_coco17
    from courtvision.stage5_classifier.infer import _CLASS_ORDER, _segment_peaks, classify_windows
    from courtvision.stage5_classifier.model import build_poseconv3d
    from courtvision.stage5_classifier.overlay import make_action_overlay_video

    data_volume.reload()
    artifacts_volume.reload()
    model = build_poseconv3d(num_classes=4)
    state = torch.load(_OUT_CKPT, map_location="cpu").get("state_dict")
    model.load_state_dict(state, strict=False)
    model = model.cuda().eval()

    seq = Pose2DSequence.model_validate_json(
        Path(f"/artifacts/stage2_batch/pose2d_{clip}.json").read_text()
    )
    vid = f"/data/trackid3x3/videos/Indoor/raw/{clip}.mp4"
    cap = cv2.VideoCapture(vid)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    nframes = int(cap.get(7))
    img_shape = (int(cap.get(4)), int(cap.get(3)))  # (H, W)
    cap.release()
    max_frames = min(nframes, int(max_seconds * fps))

    by_track: dict[int, list] = defaultdict(list)
    for p in seq.poses:
        if p.frame_idx < max_frames:
            by_track[p.track_id].append(p)

    track_actions: dict[int, dict[int, tuple]] = {}
    track_events: dict[int, list] = {}
    summary: dict[int, dict] = {}
    for tid, tposes in by_track.items():
        tposes = sorted(tposes, key=lambda p: p.frame_idx)
        if len(tposes) < min_track_frames:
            continue
        fmin, fmax = tposes[0].frame_idx, tposes[-1].frame_idx
        present = {p.frame_idx: p for p in tposes}
        n = fmax - fmin + 1
        kp = np.zeros((n, 17, 2), np.float32)
        ks = np.zeros((n, 17), np.float32)
        last: Any = None
        for i in range(n):
            p = present.get(fmin + i)
            if p is not None:
                c = halpe26_to_coco17(p)
                kp[i] = [[k.x, k.y] for k in c.keypoints]
                ks[i] = [k.score for k in c.keypoints]
                last = (kp[i].copy(), ks[i].copy())
            elif last is not None:
                kp[i], ks[i] = last
        centers, probs = classify_windows(model, kp, ks, img_shape)
        centers_arr = np.asarray(centers)
        fa: dict[int, tuple] = {}
        shoot_hi = 0
        for i in range(n):
            pr = probs[int(np.argmin(np.abs(centers_arr - i)))]
            top = int(np.argmax(pr))
            fa[fmin + i] = (_CLASS_ORDER[top], float(pr[top]))
            if top == 2 and float(pr[2]) >= 0.6:
                shoot_hi += 1
        track_actions[tid] = fa
        # Consolidated shot events (one per segment) → tight SHOOT flash in the
        # overlay. Captured at a low base (0.5); the overlay gates the flash at
        # shoot_thresh. Peak frames are mapped back to video coords (+ fmin).
        shot_score = [float(p[2]) for p in probs]
        peaks = _segment_peaks(shot_score, centers, 0.5, 24)
        track_events[tid] = [(fmin + centers[i], round(shot_score[i], 3)) for i in peaks]
        summary[tid] = {"present": len(tposes), "windows": len(centers), "shoot>=0.6": shoot_hi}

    out_rel = f"stage5_action_overlays/{clip}_action.mp4"
    make_action_overlay_video(
        Path(vid), seq, track_actions, Path("/artifacts") / out_rel,
        track_events=track_events, max_seconds=max_seconds,
    )
    artifacts_volume.commit()
    print(json.dumps({"clip": clip, "tracks": summary, "overlay": out_rel}, indent=2))
    return {"clip": clip, "tracks": summary, "overlay": out_rel}


@app.function(image=poseconv3d_image, gpu="A10G", volumes=VOLUMES, timeout=3600)
def evaluate_actions(min_track_frames: int = 32, base_threshold: float = 0.3) -> dict:
    """Run shot detection on EVERY TrackID3x3 indoor clip that has a Stage 2
    pose JSON; report shot-event counts at a confidence-threshold sweep + the
    global peak-confidence distribution, so we can pick an operating threshold.

    HONEST LIMITATION: TrackID3x3 has NO shot ground truth, so these are
    detection COUNTS + confidence distributions, NOT precision/recall. Use the
    confidence distribution to find where true shots separate from false fires;
    the per-clip top_events (track, video-frame, conf) let you spot-check in the
    overlay videos. True P/R needs a few hand-labelled clips.
    """
    import glob
    import json
    from collections import defaultdict
    from pathlib import Path

    import cv2
    import numpy as np
    import torch

    from courtvision.schemas import Pose2DSequence
    from courtvision.stage2_pose.halpe_to_coco17 import halpe26_to_coco17
    from courtvision.stage5_classifier.infer import run_shot_inference
    from courtvision.stage5_classifier.model import build_poseconv3d

    data_volume.reload()
    artifacts_volume.reload()
    model = build_poseconv3d(num_classes=4)
    state = torch.load(_OUT_CKPT, map_location="cpu").get("state_dict")
    model.load_state_dict(state, strict=False)
    model = model.cuda().eval()

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    per_clip: dict = {}
    all_confs: list = []
    for pj in sorted(glob.glob("/artifacts/stage2_batch/pose2d_*.json")):
        clip = Path(pj).stem.replace("pose2d_", "")
        vid = f"/data/trackid3x3/videos/Indoor/raw/{clip}.mp4"
        if not Path(vid).exists():
            continue
        cap = cv2.VideoCapture(vid)
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        img_shape = (int(cap.get(4)), int(cap.get(3)))
        nframes = int(cap.get(7))
        cap.release()
        seq = Pose2DSequence.model_validate_json(Path(pj).read_text())
        by_track: dict = defaultdict(list)
        for p in seq.poses:
            by_track[p.track_id].append(p)
        events: list = []  # (track_id, video_frame, p_shoot)
        n_tracks = 0
        for tid, tposes in by_track.items():
            tposes = sorted(tposes, key=lambda p: p.frame_idx)
            if len(tposes) < min_track_frames:
                continue
            n_tracks += 1
            fmin, fmax = tposes[0].frame_idx, tposes[-1].frame_idx
            present = {p.frame_idx: p for p in tposes}
            n = fmax - fmin + 1
            kp = np.zeros((n, 17, 2), np.float32)
            ks = np.zeros((n, 17), np.float32)
            last: Any = None
            for i in range(n):
                p = present.get(fmin + i)
                if p is not None:
                    c = halpe26_to_coco17(p)
                    kp[i] = [[k.x, k.y] for k in c.keypoints]
                    ks[i] = [k.score for k in c.keypoints]
                    last = (kp[i].copy(), ks[i].copy())
                elif last is not None:
                    kp[i], ks[i] = last
            ev = run_shot_inference(
                model, kp, ks, img_shape, session_id=clip, shooter_track_id=tid,
                score_threshold=base_threshold, merge_gap=24,
            )
            for e in ev.events:
                events.append((tid, fmin + e.peak_frame, round(e.p_shoot, 3)))
        confs = [c for _, _, c in events]
        all_confs += confs
        top = sorted(events, key=lambda t: t[2], reverse=True)[:5]
        per_clip[clip] = {
            "n_tracks": n_tracks,
            "dur_min": round(nframes / fps / 60.0, 2),
            "events_ge": {str(t): sum(1 for c in confs if c >= t) for t in thresholds},
            "top_events": [{"track": t, "frame": f, "p_shoot": c} for t, f, c in top],
        }

    arr = np.array(all_confs) if all_confs else np.array([0.0])
    summary = {
        "n_clips": len(per_clip),
        "total_events_ge": {str(t): int((arr >= t).sum()) for t in thresholds},
        "conf_percentiles": {str(q): round(float(np.percentile(arr, q)), 3) for q in (50, 75, 90, 95)},
        "conf_histogram": {
            f"{lo:.1f}-{lo + 0.1:.1f}": int(((arr >= lo) & (arr < lo + 0.1)).sum())
            for lo in np.arange(0.3, 1.0, 0.1)
        },
    }
    out = {"per_clip": per_clip, "summary": summary}
    print(json.dumps(out, indent=2))
    return out


@app.local_entrypoint()
def eval_actions(min_track_frames: int = 32) -> None:
    """uv run modal run scripts/stage5_modal.py::eval_actions — shot-detection
    counts + confidence distribution across all TrackID3x3 indoor clips."""
    import json

    res = evaluate_actions.remote(min_track_frames)
    s = res["summary"]
    print(json.dumps(s, indent=2))
    print("\nclip                  | tracks | min  | events>=  .5 .6 .7 .8 .9")
    for clip, r in sorted(res["per_clip"].items()):
        e = r["events_ge"]
        print(
            f"{clip:21} | {r['n_tracks']:>5}  | {r['dur_min']:>4} | "
            f"{e['0.5']:>3} {e['0.6']:>2} {e['0.7']:>2} {e['0.8']:>2} {e['0.9']:>2}"
        )


@app.function(image=poseconv3d_image, gpu="A10G", volumes=VOLUMES, timeout=3600)
def evaluate_actions_combined(
    min_track_frames: int = 32, window_half: int = 50
) -> dict:
    """Stage 5 (default thr 0.6) → Stage 6 forward-conf verification (flags F24).

    For every Stage-5-detected shot event on TrackID3x3 indoor clips, runs
    Stage 6 release detection in a ±``window_half``-frame window around the
    Stage 5 peak and records ``forward_conf``. Reports RAW (Stage 5 alone) vs
    VERIFIED (Stage 5 + Stage 6 gate) event counts so the F24 precision/recall
    claim can be re-tested on new datasets (e.g. iPhone footage with GT).
    """
    import glob
    import json
    from collections import defaultdict
    from pathlib import Path

    import cv2
    import numpy as np
    import torch

    from courtvision.schemas import Pose2DSequence
    from courtvision.stage2_pose.halpe_to_coco17 import halpe26_to_coco17
    from courtvision.stage5_classifier.infer import run_shot_inference
    from courtvision.stage5_classifier.model import build_poseconv3d
    from courtvision.stage6_release import analyze_release, is_verified_shot

    data_volume.reload()
    artifacts_volume.reload()
    model = build_poseconv3d(num_classes=4)
    state = torch.load(_OUT_CKPT, map_location="cpu").get("state_dict")
    model.load_state_dict(state, strict=False)
    model = model.cuda().eval()

    per_clip: dict = {}
    total_raw, total_verified = 0, 0
    for pj in sorted(glob.glob("/artifacts/stage2_batch/pose2d_*.json")):
        clip = Path(pj).stem.replace("pose2d_", "")
        vid = f"/data/trackid3x3/videos/Indoor/raw/{clip}.mp4"
        if not Path(vid).exists():
            continue
        cap = cv2.VideoCapture(vid)
        img_shape = (int(cap.get(4)), int(cap.get(3)))
        cap.release()
        seq = Pose2DSequence.model_validate_json(Path(pj).read_text())
        by_track: dict = defaultdict(list)
        for p in seq.poses:
            by_track[p.track_id].append(p)
        clip_events: list = []
        for tid, tposes in by_track.items():
            tposes = sorted(tposes, key=lambda p: p.frame_idx)
            if len(tposes) < min_track_frames:
                continue
            fmin, fmax = tposes[0].frame_idx, tposes[-1].frame_idx
            present = {p.frame_idx: p for p in tposes}
            n = fmax - fmin + 1
            kp = np.zeros((n, 17, 2), np.float32)
            ks = np.zeros((n, 17), np.float32)
            last: Any = None
            for i in range(n):
                p = present.get(fmin + i)
                if p is not None:
                    c = halpe26_to_coco17(p)
                    kp[i] = [[k.x, k.y] for k in c.keypoints]
                    ks[i] = [k.score for k in c.keypoints]
                    last = (kp[i].copy(), ks[i].copy())
                elif last is not None:
                    kp[i], ks[i] = last
            ev = run_shot_inference(
                model, kp, ks, img_shape, session_id=clip, shooter_track_id=tid
            )
            for e in ev.events:
                peak_local = e.peak_frame
                lo = max(0, peak_local - window_half)
                hi = min(n, peak_local + window_half)
                analysis = analyze_release(
                    kp, ks, f"{clip}_t{tid}_f{fmin + peak_local}", window=(lo, hi)
                )
                verified = is_verified_shot(analysis["forward"])
                clip_events.append(
                    {
                        "track": tid,
                        "peak_video_frame": fmin + peak_local,
                        "p_shoot": round(float(e.p_shoot), 3),
                        "fwd_frame": fmin + analysis["forward"].release_frame_idx,
                        "apex_frame": fmin + analysis["apex"].release_frame_idx,
                        "forward_conf": analysis["forward"].confidence_flag,
                        "verified": verified,
                    }
                )
        raw = len(clip_events)
        ver = sum(1 for e in clip_events if e["verified"])
        per_clip[clip] = {"raw": raw, "verified": ver, "events": clip_events}
        total_raw += raw
        total_verified += ver

    summary = {
        "n_clips": len(per_clip),
        "events_raw": total_raw,
        "events_verified": total_verified,
        "dropped_by_stage6": total_raw - total_verified,
        "stage5_score_threshold": 0.6,
        "stage6_gate": "forward_conf == 'high'",
    }
    out = {"summary": summary, "per_clip": per_clip}
    print(json.dumps(out, indent=2))
    return out


@app.local_entrypoint()
def eval_combined() -> None:
    """uv run modal run scripts/stage5_modal.py::eval_combined — Stage 5 (0.6) →
    Stage 6 verification across TrackID3x3 indoor clips (flags F24)."""
    import json

    res = evaluate_actions_combined.remote()
    print(json.dumps(res["summary"], indent=2))
    print("\nclip                  | raw | verified | dropped")
    for clip, r in sorted(res["per_clip"].items()):
        print(f"{clip:21} | {r['raw']:>3} | {r['verified']:>8} | {r['raw'] - r['verified']:>7}")


@app.function(image=poseconv3d_image, gpu="A10G", volumes=VOLUMES, timeout=3600)
def eval_on_test(ckpt: str, split_version: str = "v2", clip_len: int = 64) -> dict:
    """Evaluate a Stage 5 checkpoint on the held-out v2 test split (the
    before/after fine-tuning ablation). Single CENTERED contiguous window per
    clip → per-class accuracy + confusion matrix. Same `compute_subject_disjoint_split`
    function used during training, so train/val/test are guaranteed disjoint."""
    import glob
    import json
    import pickle
    import zipfile

    import numpy as np
    import torch
    from mmaction.datasets.transforms import (
        FormatShape,
        GeneratePoseTarget,
        PoseCompact,
        PoseDecode,
        Resize,
        SampleFrames,
    )

    from courtvision.stage5_classifier.infer import _CLASS_ORDER
    from courtvision.stage5_classifier.model import build_poseconv3d
    from courtvision.stage5_classifier.split import compute_subject_disjoint_split

    data_volume.reload()
    artifacts_volume.reload()
    model = build_poseconv3d(num_classes=4)
    state = torch.load(ckpt, map_location="cpu").get("state_dict")
    model.load_state_dict(state, strict=False)
    model = model.cuda().eval()
    device = next(model.parameters()).device

    sa = "/root/sa_test_eval"
    with zipfile.ZipFile(f"{_SA_DIR}.zip") as zf:
        zf.extractall(sa)
    all_items: list = []
    for fn in ("train.pkl", "val.pkl"):
        matches = glob.glob(f"{sa}/**/{fn}", recursive=True)
        all_items.extend(pickle.load(open(matches[0], "rb")))

    if split_version == "v2":
        test_fds = set(compute_subject_disjoint_split(all_items)["test"])
    else:
        matches = glob.glob(f"{sa}/**/val.pkl", recursive=True)
        test_fds = {x["frame_dir"] for x in pickle.load(open(matches[0], "rb"))}
    test_items = [x for x in all_items if x["frame_dir"] in test_fds]

    pipeline = [
        SampleFrames(clip_len=clip_len, frame_interval=1, num_clips=1, test_mode=True),
        PoseDecode(),
        PoseCompact(hw_ratio=1.0, allow_imgpad=True),
        Resize(scale=(56, 56), keep_ratio=False),
        GeneratePoseTarget(sigma=0.6, use_score=True, with_kp=True, with_limb=False),
        FormatShape(input_format="NCTHW_Heatmap"),
    ]

    conf = [[0, 0, 0, 0] for _ in range(3)]
    n_per_class = [0, 0, 0]
    correct_per_class = [0, 0, 0]
    for x in test_items:
        kp = np.asarray(x["keypoint"], dtype=np.float32).copy()
        ks = np.asarray(x["keypoint_score"], dtype=np.float32).copy()
        lab = int(x["label"])
        r: dict = {
            "keypoint": kp,
            "keypoint_score": ks,
            "img_shape": tuple(x["img_shape"]),
            "modality": "Pose",
            "total_frames": int(x["total_frames"]),
            "start_index": 0,
            "label": lab,
            "frame_dir": x["frame_dir"],
        }
        for t in pipeline:
            r = t(r)
        xt = torch.as_tensor(np.asarray(r["imgs"]), dtype=torch.float32, device=device)
        if xt.ndim == 5 and xt.shape[0] > 1:
            xt = xt[:1]
        with torch.inference_mode():
            probs = torch.softmax(model(xt), dim=1)[0].cpu().numpy()
        pred = int(np.argmax(probs))
        n_per_class[lab] += 1
        correct_per_class[lab] += int(pred == lab)
        conf[lab][pred] += 1

    per_class_acc = {
        _CLASS_ORDER[c]: round(correct_per_class[c] / n_per_class[c], 3) if n_per_class[c] else None
        for c in range(3)
    }
    overall = sum(correct_per_class) / sum(n_per_class) if sum(n_per_class) else 0.0
    mean_per_class = (
        sum(correct_per_class[c] / n_per_class[c] for c in range(3) if n_per_class[c]) / 3
    )
    confusion = {
        _CLASS_ORDER[t]: {
            _CLASS_ORDER[p]: (round(conf[t][p] / sum(conf[t]), 3) if sum(conf[t]) else None)
            for p in range(4)
        }
        for t in range(3)
    }
    out = {
        "ckpt": ckpt,
        "split_version": split_version,
        "clip_len": clip_len,
        "n_test": sum(n_per_class),
        "per_class_n": {_CLASS_ORDER[c]: n_per_class[c] for c in range(3)},
        "per_class_acc": per_class_acc,
        "overall_acc": round(overall, 3),
        "mean_per_class_acc": round(mean_per_class, 3),
        "confusion": confusion,
    }
    print(json.dumps(out, indent=2))
    return out


@app.local_entrypoint()
def test_eval(ckpt: str, split_version: str = "v2", clip_len: int = 64) -> None:
    """uv run modal run scripts/stage5_modal.py::test_eval --ckpt <path>"""
    import json

    print(json.dumps(eval_on_test.remote(ckpt, split_version, clip_len), indent=2))


@app.local_entrypoint()
def action_demo(clip: str = "basket_S4T4_post", max_seconds: float = 20.0) -> None:
    """uv run modal run scripts/stage5_modal.py::action_demo --clip basket_S4T4_post
    — action recognizer on a TrackID3x3 clip → multi-player overlay video."""
    import json

    res = action_overlay.remote(clip, max_seconds)
    print(json.dumps(res, indent=2))
    print(
        f"pull: uv run modal volume get courtvision-artifacts {res['overlay']} "
        "outputs/stage5_action_overlays/"
    )


@app.function(image=image, volumes=VOLUMES, timeout=180)
def list_action_clips() -> list[str]:
    """All TrackID3x3 indoor clips that have a Stage 2 pose JSON + a video."""
    import glob
    from pathlib import Path

    data_volume.reload()
    artifacts_volume.reload()
    clips = []
    for pj in sorted(glob.glob("/artifacts/stage2_batch/pose2d_*.json")):
        clip = Path(pj).stem.replace("pose2d_", "")
        if Path(f"/data/trackid3x3/videos/Indoor/raw/{clip}.mp4").exists():
            clips.append(clip)
    return clips


@app.local_entrypoint()
def action_demo_all(max_seconds: float = 60.0) -> None:
    """uv run modal run scripts/stage5_modal.py::action_demo_all — render the
    multi-player action overlay for EVERY TrackID3x3 indoor clip (parallel)."""
    clips = list_action_clips.remote()
    print(f"rendering {len(clips)} overlays: {clips}")
    results = list(action_overlay.starmap([(c, max_seconds, 32) for c in clips]))
    for r in results:
        print(f"  {r['overlay']}  ({len(r['tracks'])} tracks)")
    print(
        "\npull all: uv run modal volume get --force courtvision-artifacts "
        "stage5_action_overlays outputs/"
    )


# --- COCO-17 keypoint pose pipeline (posec3d) -------------------------------
# CONTIGUOUS window sampling (SampleFrames, frame_interval=1) — a random run of
# `clip_len` consecutive frames, exactly the sliding-window regime used at
# inference. This replaces UniformSampleFrames (which spread samples across the
# whole trimmed clip); that train/inference mismatch cratered contiguous-window
# accuracy from ~78% to ~57% (see diagnose_sampling / sweep).
def _train_pipeline(clip_len: int, crop_area_min: float = 0.56) -> list:
    return [
        dict(type="SampleFrames", clip_len=clip_len, frame_interval=1, num_clips=1),
        dict(type="PoseDecode"),
        dict(type="PoseCompact", hw_ratio=1.0, allow_imgpad=True),
        dict(type="Resize", scale=(-1, 64)),
        dict(type="RandomResizedCrop", area_range=(crop_area_min, 1.0)),
        dict(type="Resize", scale=(56, 56), keep_ratio=False),
        dict(
            type="Flip",
            flip_ratio=0.5,
            left_kp=[1, 3, 5, 7, 9, 11, 13, 15],
            right_kp=[2, 4, 6, 8, 10, 12, 14, 16],
        ),
        dict(type="GeneratePoseTarget", sigma=0.6, use_score=True, with_kp=True, with_limb=False),
        dict(type="FormatShape", input_format="NCTHW_Heatmap"),
        dict(type="PackActionInputs"),
    ]


def _val_pipeline(clip_len: int) -> list:
    return [
        dict(type="SampleFrames", clip_len=clip_len, frame_interval=1, num_clips=1, test_mode=True),
        dict(type="PoseDecode"),
        dict(type="PoseCompact", hw_ratio=1.0, allow_imgpad=True),
        dict(type="Resize", scale=(56, 56), keep_ratio=False),
        dict(type="GeneratePoseTarget", sigma=0.6, use_score=True, with_kp=True, with_limb=False),
        dict(type="FormatShape", input_format="NCTHW_Heatmap"),
        dict(type="PackActionInputs"),
    ]


@app.function(
    image=poseconv3d_image,
    gpu="A10G",
    volumes=VOLUMES,
    secrets=_WANDB_SECRETS,
    timeout=_MODAL_MAX_TIMEOUT_S,
)
def train_multisubjects(config: dict) -> dict:
    """Transfer-stage-2: fine-tune K400 PoseConv3D on MultiSubjects (3 action
    classes + an untrained 4th `background` logit). Preemption-resilient:
    checkpoints to the Volume each epoch and resumes from `last_checkpoint` on
    a Modal-preemption restart (flags.md F20 lessons). `debug=True` runs a tiny
    subset for 1 epoch as a smoke test.
    """
    import glob as _glob
    import os
    import pickle
    import shutil
    from pathlib import Path

    import mmaction  # noqa: F401 — populate the mmaction registry
    import torch
    from mmengine.config import Config
    from mmengine.hooks import Hook
    from mmengine.runner import Runner

    data_volume.reload()
    artifacts_volume.reload()
    debug = bool(config.get("debug", False))
    epochs = int(config.get("epochs", 1 if debug else 24))
    batch_size = int(config.get("batch_size", 8))
    clip_len = int(config.get("clip_len", _CLIP_LEN))
    tag = str(config.get("tag", "")).strip()
    split_version = str(config.get("split_version", "v1"))
    freeze_backbone = bool(config.get("freeze_backbone", False))
    # Regularisation knobs (defaults match the existing v2 recipe so old runs
    # are bit-for-bit unchanged; raise to combat overfitting).
    weight_decay = float(config.get("weight_decay", 3e-4))
    dropout_ratio = float(config.get("dropout", 0.5))
    label_smooth_eps = float(config.get("label_smooth_eps", 0.1))
    crop_area_min = float(config.get("crop_area_min", 0.56))
    # Isolate runs by tag (and debug) so a smoke run or a different recipe can't
    # contaminate another run's work_dir (which would trigger a bad resume).
    suffix = ("_" + tag if tag else "") + ("_debug" if debug else "")
    work_dir = _WORK_DIR + suffix
    out_ckpt = _OUT_CKPT.replace(".pth", f"_{tag}.pth") if tag else _OUT_CKPT
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    # 1. The SA annotations live in a zip on the Volume (not extracted). Unzip,
    #    then repackage the flat train/val lists → MMAction2 {split, annotations}.
    import zipfile

    sa_extract = "/root/sa"
    with zipfile.ZipFile(f"{_SA_DIR}.zip") as zf:
        zf.extractall(sa_extract)
    anns: list = []
    split: dict = {"train": [], "val": []}
    if split_version == "v2":
        # Subject-disjoint train/val/test (test is held out — kept out of the
        # training split dict; eval_on_test pulls it from the SAME deterministic
        # split function. See `stage5_classifier/split.py` and
        # `docs/stage5_training_method.md`.
        from courtvision.stage5_classifier.split import compute_subject_disjoint_split

        all_items: list = []
        for fn in ("train.pkl", "val.pkl"):
            matches = _glob.glob(f"{sa_extract}/**/{fn}", recursive=True)
            if not matches:
                raise FileNotFoundError(f"{fn} not found in {_SA_DIR}.zip")
            its = pickle.load(open(matches[0], "rb"))
            if debug:
                its = its[:12]
            all_items.extend(its)
        split_map = compute_subject_disjoint_split(all_items)
        split["train"] = split_map["train"]
        split["val"] = split_map["val"]
        anns.extend(all_items)
    else:
        for sp, fn in [("train", "train.pkl"), ("val", "val.pkl")]:
            matches = _glob.glob(f"{sa_extract}/**/{fn}", recursive=True)
            if not matches:
                raise FileNotFoundError(f"{fn} not found in {_SA_DIR}.zip")
            items = pickle.load(open(matches[0], "rb"))
            if debug:
                items = items[:24]
            for it in items:
                split[sp].append(it["frame_dir"])
                anns.append(it)
    # Class weights from THIS run's train set (works for v1 or v2).
    train_counts = [0, 0, 0, 0]
    train_fds = set(split["train"])
    for it in anns:
        if it["frame_dir"] in train_fds:
            train_counts[int(it["label"])] += 1
    # Inverse-frequency class weights to counter the layup minority (~13%);
    # background (4th logit) has no train samples so its weight is irrelevant.
    n_real = sum(train_counts[:3]) or 1
    class_weight = [
        (n_real / (3 * train_counts[c])) if train_counts[c] > 0 else 1.0 for c in range(3)
    ] + [1.0]
    combined = "/root/ms_combined.pkl"
    with open(combined, "wb") as f:
        pickle.dump({"split": split, "annotations": anns}, f)

    # 2. Backbone-only K400 checkpoint (drop the 400-class head so load_from
    #    doesn't shape-clash with our fresh 4-class head).
    k400 = torch.load(_K400, map_location="cpu")
    state = k400.get("state_dict", k400)
    backbone_ckpt = "/root/k400_backbone.pth"
    torch.save(
        {"state_dict": {k: v for k, v in state.items() if not k.startswith("cls_head.")}},
        backbone_ckpt,
    )

    # 3. Load the complete K400 config as base; patch only what differs.
    base_cfg = _glob.glob(
        "/opt/mmaction2/configs/skeleton/posec3d/slowonly_r50_*u48*k400-keypoint.py"
    )[0]
    cfg = Config.fromfile(base_cfg)
    cfg.model.cls_head.num_classes = 4
    cfg.model.cls_head.loss_cls = dict(type="CrossEntropyLoss", class_weight=class_weight)
    cfg.model.cls_head.label_smooth_eps = label_smooth_eps
    cfg.model.cls_head.dropout_ratio = dropout_ratio
    cfg.work_dir = work_dir
    # resume and load_from are MUTUALLY EXCLUSIVE in mmengine: if both are set,
    # it resumes from load_from — and our backbone-only ckpt has no 'meta' →
    # KeyError. So: a valid prior full-run checkpoint → resume (load_from=None);
    # otherwise → fresh start from the K400 backbone (and wipe any stale/partial
    # or debug-contaminated work_dir so it can't trigger a bad resume).
    resume_ok = False
    last_ptr = Path(work_dir) / "last_checkpoint"
    if last_ptr.exists():
        try:
            ck = torch.load(last_ptr.read_text().strip(), map_location="cpu")
            resume_ok = isinstance(ck, dict) and "meta" in ck
        except (OSError, RuntimeError, KeyError, EOFError):
            resume_ok = False
    if resume_ok:
        cfg.resume = True
        cfg.load_from = None
    else:
        cfg.resume = False
        cfg.load_from = backbone_ckpt
        shutil.rmtree(work_dir, ignore_errors=True)
        Path(work_dir).mkdir(parents=True, exist_ok=True)

    ds_train = dict(
        type="PoseDataset", ann_file=combined, split="train",
        pipeline=_train_pipeline(clip_len, crop_area_min=crop_area_min),
    )
    ds_val = dict(
        type="PoseDataset",
        ann_file=combined,
        split="val",
        pipeline=_val_pipeline(clip_len),
        test_mode=True,
    )
    cfg.train_dataloader.batch_size = batch_size
    cfg.train_dataloader.num_workers = 4
    cfg.train_dataloader.dataset = dict(type="RepeatDataset", times=1, dataset=ds_train)
    cfg.val_dataloader.batch_size = batch_size
    cfg.val_dataloader.num_workers = 4
    cfg.val_dataloader.dataset = ds_val
    cfg.test_dataloader = cfg.val_dataloader

    cfg.train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=epochs, val_begin=1, val_interval=1)
    cfg.param_scheduler = [
        dict(
            type="CosineAnnealingLR",
            eta_min=0,
            T_max=epochs,
            by_epoch=True,
            convert_to_iter_based=True,
        )
    ]
    cfg.optim_wrapper.optimizer.lr = 0.01  # PRD §9.5.2 stage-2 recipe
    cfg.optim_wrapper.optimizer.weight_decay = weight_decay  # L2 regularisation
    # Discriminative LR: backbone trains at 0.1x the fresh head's rate (LP-FT-
    # style); set to 0 for the LINEAR-PROBE baseline (head-only training).
    cfg.optim_wrapper.paramwise_cfg = dict(
        custom_keys={"backbone": dict(lr_mult=0.0 if freeze_backbone else 0.1)}
    )
    # Reproducibility for the v2 ablation runs.
    cfg.randomness = dict(seed=0, deterministic=False)
    cfg.default_hooks.checkpoint = dict(
        type="CheckpointHook", interval=1, max_keep_ckpts=2, save_best="acc/top1", rule="greater"
    )

    if os.environ.get("WANDB_API_KEY") and not debug:
        cfg.visualizer.vis_backends = [
            dict(type="LocalVisBackend"),
            dict(
                type="WandbVisBackend",
                init_kwargs=dict(
                    project="courtvision",
                    name=f"stage5-stage2-multisubjects{('-' + tag) if tag else ''}",
                ),
            ),
        ]

    runner = Runner.from_cfg(cfg)

    if freeze_backbone:
        # Linear-probe baseline: freeze all backbone params (head still trains).
        # BN running stats may still drift slightly under .train() — acceptable
        # for the report's before/after; the gain attributed to fine-tuning is
        # slightly conservative because of this.
        for p in runner.model.backbone.parameters():
            p.requires_grad = False

    class _VolumeCommit(Hook):
        """Persist the Volume after each epoch so a preemption can resume."""

        def after_train_epoch(self, runner: object) -> None:
            artifacts_volume.commit()

    runner.register_hook(_VolumeCommit(), priority="LOWEST")
    runner.train()

    # 4. Export the best checkpoint to the canonical (tagged) path.
    best = sorted(_glob.glob(f"{work_dir}/best_acc*.pth"))
    src = best[-1] if best else f"{work_dir}/last_checkpoint"
    Path(out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    if best:
        shutil.copy(src, out_ckpt)
        artifacts_volume.commit()
    return {
        "work_dir": work_dir,
        "out_ckpt": out_ckpt,
        "clip_len": clip_len,
        "split_version": split_version,
        "freeze_backbone": freeze_backbone,
        "class_weight": [round(w, 3) for w in class_weight],
        "resumed": bool(cfg.resume),
        "epochs": epochs,
        "best_ckpt": src,
    }


@app.local_entrypoint()
def train_ms(
    debug: bool = False,
    epochs: int = 24,
    batch_size: int = 4,
    clip_len: int = 64,
    tag: str = "64f_bal",
    split_version: str = "v1",
    freeze_backbone: bool = False,
    weight_decay: float = 3e-4,
    dropout: float = 0.5,
    label_smooth_eps: float = 0.1,
    crop_area_min: float = 0.56,
) -> None:
    """Step-1 retrain: 64f CONTIGUOUS windows + inverse-freq class weights +
    label smoothing + discriminative LR. Saves to a tagged checkpoint so the
    working 32f model is untouched.

    For the report's before/after fine-tuning ablation (docs/stage5_training_method.md):
      Full fine-tune: ``--tag 64f_bal_v2 --split-version v2 --batch-size 8``
      Linear probe:   ``--tag 64f_probe_v2 --split-version v2 --batch-size 8 --freeze-backbone``

    Smoke (blocks): ``uv run modal run scripts/stage5_modal.py::train_ms --debug``
    Full run (survives disconnect): ``uv run modal run --detach
    scripts/stage5_modal.py::train_ms``
    """
    import json

    cfg = {
        "debug": debug,
        "epochs": epochs,
        "batch_size": batch_size,
        "clip_len": clip_len,
        "tag": tag,
        "split_version": split_version,
        "freeze_backbone": freeze_backbone,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "label_smooth_eps": label_smooth_eps,
        "crop_area_min": crop_area_min,
    }
    if debug:
        print(json.dumps(train_multisubjects.remote(cfg), indent=2))
    else:
        # spawn (not remote) so training keeps running server-side independent
        # of this client; with `modal run --detach` it survives the session
        # ending. Monitor via W&B / the Volume checkpoint.
        call = train_multisubjects.spawn(cfg)
        print(f"spawned full MultiSubjects training: call_id={call.object_id}")


# ---------------------------------------------------------------------------
# Capture-session orchestrator (both cameras in parallel)
# ---------------------------------------------------------------------------
@app.function(image=image, volumes=VOLUMES, timeout=3600 * 2, cpu=1)
def orchestrate_stage5_session(session: str, shoot_threshold: float = 0.75) -> dict:
    """Server-side orchestrator — spawns action_overlay_session for both cameras in parallel."""
    print(f"=== Stage 5 — spawning both cameras at threshold {shoot_threshold} ===")
    calls = {
        cam: action_overlay_session.spawn(session, cam, shoot_threshold=shoot_threshold)
        for cam in ("cam_a", "cam_b")
    }
    for cam in calls:
        print(f"  spawned {cam}")
    results = {cam: call.get() for cam, call in calls.items()}
    for cam, r in results.items():
        print(f"  ✓ {cam}: events={r['events_relpath']}  overlay={r['overlay_relpath']}  "
              f"n_events={r['n_events']}")
    return {"session": session, "shoot_threshold": shoot_threshold, "results": results}


@app.local_entrypoint()
def run_session(session: str = "2026-06-02_1on1", shoot_threshold: float = 0.75) -> None:
    """Local entrypoint — one .remote() into the server-side orchestrator,
    then pull Stage 5 outputs (events JSON + overlay video) to outputs/."""
    import subprocess
    from pathlib import Path as _Path

    print(f"Kicking off Stage 5 orchestrator (session={session}, thr={shoot_threshold})...")
    result = orchestrate_stage5_session.remote(session, shoot_threshold=shoot_threshold)

    out_dir = _Path(f"outputs/sessions/{session}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pulls: list[tuple[str, _Path]] = []
    for cam, r in result["results"].items():
        pulls.append((r["events_relpath"], out_dir / f"shotevents_{cam}.json"))
        pulls.append((r["overlay_relpath"], out_dir / f"stage5_overlay_{cam}.mp4"))
    print(f"\nDownloading {len(pulls)} files to {out_dir}...")
    for rel, dest in pulls:
        cmd = ["uv", "run", "modal", "volume", "get",
               "courtvision-artifacts", rel, str(dest), "--force"]
        print(f"  pulling {rel}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"    ⚠ failed: {proc.stderr.strip()}")
    print(f"\n✓ Stage 5 outputs in {out_dir}")
