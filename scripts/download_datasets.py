"""Dataset downloaders. PRD §8.3.

Each function downloads one dataset directly onto the
``courtvision-data`` Modal Volume. Gated HF datasets need ``HF_TOKEN`` in
the submitting shell — exported via ``set -a; source .env; set +a``.

Usage::

    set -a; source .env; set +a
    uv run modal run scripts/download_datasets.py::download_multisubjects
"""

from __future__ import annotations

import os
from pathlib import Path

from courtvision.modal_app import (
    _HF_SECRETS,
    _KAGGLE_SECRETS,
    _ROBOFLOW_SECRETS,
    app,
    data_volume,
    image,
)

# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _hf_snapshot(repo_id: str, dest: Path) -> str:
    """Pull a full HF dataset snapshot to ``dest`` (creating parents)."""
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        token=token,
    )
    return local_path


# ---------------------------------------------------------------------------
# Downloaders
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=3600 * 2,
    cpu=4,
)
def download_trackid3x3() -> str:
    """TrackID3x3 (open-starlab). Git-clones the GitHub repo to /data/trackid3x3.

    What this gets us (~387 MB):
      - ``ground_truth/`` — MOT tracking, pose (10 keypoints/player), court
        keypoints, transformed coordinates. Indoor / Outdoor / Drone splits.
      - ``script/``, ``court_images/``, ``videos/gif/`` — code + sample previews.
      - Third-party tracker submodules in-tree (BoT-SORT, jersey-number-pipeline).

    What this does NOT get: the actual video files. Per the upstream README
    those live at https://drive.google.com/drive/folders/1aWqMwQKr5xKMjqms7-raYluSlxPsGvwX
    which gdown can't reliably fetch (file-level permission interstitials).
    When videos are needed for Stage 1/2 eval, download the folder manually
    in a browser and use ``extract_volume_zip`` to land the zip on the Volume.

    The function is idempotent: re-running wipes ``/data/trackid3x3/`` and
    re-clones. Hand-uploaded videos placed at a different relpath are
    unaffected.
    """
    import shutil
    import subprocess

    dest = Path("/data/trackid3x3")
    if dest.exists():
        shutil.rmtree(dest)
    subprocess.run(
        [
            "git",
            "clone",
            "--depth=1",
            "https://github.com/open-starlab/TrackID3x3.git",
            str(dest),
        ],
        check=True,
    )
    data_volume.commit()
    return str(dest)


@app.function(image=image, volumes={"/data": data_volume}, timeout=3600, cpu=4)
def extract_trackid3x3_uploads() -> str:
    """Extract every zip under ``/data/_uploads/trackid3x3/`` into ``/data/trackid3x3/``.

    The Drive bulk download nests everything under a PascalCase ``TrackID3x3/``
    prefix, but the existing GitHub clone lives at lowercase ``/data/trackid3x3``.
    To merge, this strips a leading ``TrackID3x3/`` from each zip entry so paths
    line up. Loose ``.mov`` files in ``_uploads/trackid3x3/`` get moved to
    ``/data/trackid3x3/_loose/`` with their original Drive names — the user can
    later rename/re-organize once provenance is figured out. Zips and loose
    files are removed from ``_uploads/`` after success to reclaim Volume space.
    """
    import shutil
    import zipfile

    data_volume.reload()
    src_dir = Path("/data/_uploads/trackid3x3")
    dst_root = Path("/data/trackid3x3")
    if not src_dir.exists():
        print(f"no uploads found at {src_dir}")
        return ""

    n_zips = 0
    n_files = 0
    n_loose = 0
    for src in sorted(src_dir.iterdir()):
        if src.is_file() and src.suffix.lower() == ".zip":
            print(f"extracting {src.name}...")
            with zipfile.ZipFile(src) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    parts = info.filename.split("/")
                    rel = "/".join(parts[1:]) if parts and parts[0] == "TrackID3x3" else info.filename
                    if not rel:
                        continue
                    target = dst_root / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as f_in, open(target, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    n_files += 1
            src.unlink()
            n_zips += 1
        elif src.is_file() and src.suffix.lower() == ".mov":
            loose_dst = dst_root / "_loose" / src.name
            loose_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(loose_dst))
            print(f"moved loose: {src.name} -> _loose/")
            n_loose += 1

    # Clean up empty _uploads dir.
    try:
        src_dir.rmdir()
        src_dir.parent.rmdir()
    except OSError:
        pass

    print(f"done: {n_zips} zips ({n_files} files extracted), {n_loose} loose .movs moved")
    data_volume.commit()
    return str(dst_root)


@app.function(image=image, volumes={"/data": data_volume}, timeout=3600 * 2, cpu=4)
def extract_volume_zip(src_relpath: str, dest_relpath: str = "") -> str:
    """Unzip an archive that was put on the Volume via ``modal volume put``.

    Used for datasets we can't fetch programmatically (e.g., TrackID3x3 from
    a brittle Google Drive folder). Source is at ``/data/<src_relpath>``;
    destination dir is ``/data/<dest_relpath>`` (defaults to the zip's stem).
    """
    import zipfile

    src = Path("/data") / src_relpath
    if not src.exists():
        raise FileNotFoundError(f"zip not found on volume: {src}")
    dest = Path("/data") / (dest_relpath or src.stem)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as zf:
        zf.extractall(dest)
    size_gb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e9
    print(f"extracted {src} → {dest} ({size_gb:.2f} GB)")
    src.unlink(missing_ok=True)  # reclaim space; comment out to keep the raw zip
    data_volume.commit()
    return str(dest)


@app.function(
    image=image,
    volumes={"/data": data_volume},
    secrets=_HF_SECRETS,
    timeout=3600 * 8,
    cpu=4,
)
def download_multisubjects() -> str:
    """MultiSubjects (Xu et al.). HF gated dataset; ~30-50 GB.

    Requires HF_TOKEN with access granted on the dataset page.
    """
    local_path = _hf_snapshot("Henu-Software/Henu-MultiSubjects", Path("/data/multisubjects"))
    data_volume.commit()
    return local_path


@app.function(
    image=image,
    volumes={"/data": data_volume},
    secrets=_HF_SECRETS,
    timeout=3600 * 12,
    cpu=4,
)
def download_finesports() -> str:
    """FineSports (Xu et al.). HF gated dataset; ~50-100 GB.

    Requires HF_TOKEN with access granted on the dataset page.
    """
    local_path = _hf_snapshot("zhixingsun/FineSports", Path("/data/finesports"))
    data_volume.commit()
    return local_path


def _print_zip_listing(zf, limit: int) -> None:
    names = zf.namelist()
    print(f"{len(names)} entries total. First {min(limit, len(names))}:")
    for n in names[:limit]:
        info = zf.getinfo(n)
        kind = "dir" if n.endswith("/") else "file"
        print(f"  {kind:4}  {info.file_size:>12}  {n}")
    prefixes = sorted({n.split("/", 1)[0] for n in names if n})
    print(f"\nunique top-level entries ({len(prefixes)}): {prefixes[:20]}")


@app.function(image=image, volumes={"/data": data_volume}, timeout=300, cpu=2)
def peek_archive(volume_relpath: str, inner_relpath: str = "", limit: int = 50) -> None:
    """List the top-level entries of a zip on the volume without extracting.

    With ``--inner-relpath``, stream-open a zip nested inside the outer zip
    and list its entries instead — no extraction to disk.

    Examples::
        uv run modal run scripts/download_datasets.py::peek_archive \\
            --volume-relpath=finesports/FineSports_CVPR2024.zip
        uv run modal run scripts/download_datasets.py::peek_archive \\
            --volume-relpath=finesports/FineSports_CVPR2024.zip \\
            --inner-relpath=FineSports_CVPR2024/FineSports_trimmed_video_frames.zip
    """
    import zipfile
    from pathlib import Path

    # Force a fresh view of the volume; without this, a peek invoked
    # shortly after a download may see a stale snapshot.
    data_volume.reload()

    path = Path("/data") / volume_relpath
    if not path.exists():
        print(f"not found on volume: {path}")
        parent = path.parent
        if parent.exists():
            print(f"\ncontents of {parent} (so you can find the right relpath):")
            for child in sorted(parent.iterdir()):
                size = child.stat().st_size if child.is_file() else "-"
                print(f"  {'dir' if child.is_dir() else 'file':4}  {size:>14}  {child.name}")
        else:
            print(f"parent dir also not found: {parent}")
        return
    print(f"{path}  ({path.stat().st_size / 1e9:.2f} GB)")
    if path.suffix.lower() != ".zip":
        print("only .zip is supported by this peek helper (rar needs unrar in the image)")
        return
    with zipfile.ZipFile(path) as zf:
        if not inner_relpath:
            _print_zip_listing(zf, limit)
            return
        try:
            inner_info = zf.getinfo(inner_relpath)
        except KeyError:
            print(f"inner entry not found in outer zip: {inner_relpath}")
            print("outer entries (for reference):")
            for n in zf.namelist()[:limit]:
                print(f"  {n}")
            return
        print(f"-> nested: {inner_relpath}  ({inner_info.file_size / 1e9:.2f} GB)")
        # ZipExtFile is seekable on Python 3.7+, so zipfile can read the
        # inner central directory without us extracting anything.
        with zf.open(inner_info) as inner_f, zipfile.ZipFile(inner_f) as inner_zf:
            _print_zip_listing(inner_zf, limit)


@app.function(image=image, volumes={"/data": data_volume}, timeout=120, cpu=2)
def list_dir(relpath: str = "") -> None:
    """List contents of a directory on the volume (non-recursive)."""
    from pathlib import Path

    data_volume.reload()
    path = Path("/data") / relpath
    if not path.exists():
        print(f"not found: {path}")
        return
    if path.is_file():
        print(f"file: {path}  ({path.stat().st_size:,} bytes)")
        return
    entries = sorted(path.iterdir())
    print(f"{path}  ({len(entries)} entries)")
    for child in entries:
        if child.is_dir():
            print(f"  dir   {'-':>14}  {child.name}/")
        else:
            print(f"  file  {child.stat().st_size:>14,}  {child.name}")


@app.function(image=image, volumes={"/data": data_volume}, timeout=600, cpu=2)
def count_finesports_classes(top_n: int = 80) -> None:
    """Tabulate clips and frames per action class in the FineSports nested zip."""
    import zipfile
    from collections import Counter
    from pathlib import Path

    data_volume.reload()
    outer = Path("/data/finesports/FineSports_CVPR2024.zip")
    inner_name = "FineSports_CVPR2024/FineSports_trimmed_video_frames.zip"

    clips_per_class: Counter[str] = Counter()
    frames_per_class: Counter[str] = Counter()
    seen_clips: set[tuple[str, str]] = set()
    with zipfile.ZipFile(outer) as zf, zf.open(inner_name) as inner_f, zipfile.ZipFile(
        inner_f
    ) as inner_zf:
        for name in inner_zf.namelist():
            # Expect: NewFrames/<class>/<clip_id>/<frame>.jpg
            parts = name.split("/")
            if len(parts) < 4 or parts[0] != "NewFrames" or not name.endswith(".jpg"):
                continue
            cls, clip = parts[1], parts[2]
            frames_per_class[cls] += 1
            key = (cls, clip)
            if key not in seen_clips:
                seen_clips.add(key)
                clips_per_class[cls] += 1
    print(f"{len(clips_per_class)} action classes total\n")
    print(f"{'class':<32} {'clips':>8} {'frames':>10}")
    for cls, n_clips in clips_per_class.most_common(top_n):
        print(f"{cls:<32} {n_clips:>8} {frames_per_class[cls]:>10}")
    print(
        f"\ntotals: {sum(clips_per_class.values()):,} clips, "
        f"{sum(frames_per_class.values()):,} frames"
    )


@app.function(
    image=image,
    volumes={"/data": data_volume},
    secrets=_KAGGLE_SECRETS,
    timeout=3600,
    cpu=4,
)
def download_deepsportradar() -> str:
    """DeepSportradar Basketball Instants Dataset. Kaggle; ~6 GB.

    Substitute for the unverifiable "U Arizona Roboflow basketball" dataset
    referenced in PRD §8.3 / decisions.md Decision 3. See flags.md F12.
    Academic benchmark (ACM MMSports 2022); literature-comparable mAP.
    Requires KAGGLE_USERNAME + KAGGLE_KEY and one-time terms acceptance at
    https://www.kaggle.com/datasets/deepsportradar/basketball-instants-dataset.
    """
    import subprocess

    dest = Path("/data/deepsportradar")
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "deepsportradar/basketball-instants-dataset",
            "-p",
            str(dest),
            "--unzip",
        ],
        check=True,
    )
    data_volume.commit()
    return str(dest)


@app.function(
    image=image,
    volumes={"/data": data_volume},
    secrets=_ROBOFLOW_SECRETS,
    timeout=3600 * 2,
    cpu=4,
)
def download_roboflow_basketball(
    workspace: str = "ownprojects",
    project: str = "basketball-w2xcw",
    version: int = 0,
    fmt: str = "coco",
) -> str:
    """Secondary Stage-3 mAP dataset: OwnProjects basketball-w2xcw on Roboflow.

    Parameterized so we can swap to another Roboflow project (e.g., if the
    U Arizona dataset resurfaces) without code changes. ``version=0`` selects
    the project's most recent version. See flags.md F12.
    """
    from roboflow import Roboflow

    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    proj = rf.workspace(workspace).project(project)
    if version:
        ver = proj.version(version)
    else:
        all_versions = proj.versions()
        if not all_versions:
            raise RuntimeError(f"No versions found for Roboflow project {workspace}/{project}")
        ver = all_versions[-1]
    ds = ver.download(fmt, location="/data/roboflow_basketball")
    data_volume.commit()
    return ds.location


@app.function(image=image, volumes={"/data": data_volume}, timeout=60, cpu=1)
def download_wasb_basketball() -> str:
    """DEFERRED — upstream link 404s. See flags.md F11."""
    print(
        "WASB basketball subset download is DEFERRED. Upstream link "
        "(ruiyan1995.github.io/SAM.html) returns 404. See flags.md F11. "
        "Stage 3 release-window-recall eval gate is paused until a working "
        "source is resolved."
    )
    return ""


@app.function(image=image, volumes={"/data": data_volume}, timeout=1800, cpu=4)
def build_examples_archive() -> str:
    """Curate ~50-100 MB of representative samples across every dataset.

    Writes to ``/data/_examples/<dataset>/`` on the Volume. After this runs,
    ``modal volume get courtvision-data _examples examples/`` syncs the whole
    thing to a local ``examples/`` directory at the repo root.

    Sampling is deterministic (alphabetical "first N" picks) so re-runs
    produce the same examples. Wipes ``/data/_examples/`` at the start so
    re-running is idempotent.
    """
    import shutil
    import zipfile
    from collections import defaultdict

    data_volume.reload()
    examples_root = Path("/data/_examples")
    if examples_root.exists():
        shutil.rmtree(examples_root)
    examples_root.mkdir()

    # --- TrackID3x3: a sliver of the GitHub clone --------------------------
    src = Path("/data/trackid3x3")
    dst = examples_root / "trackid3x3"
    dst.mkdir()
    if (src / "README.md").exists():
        shutil.copy(src / "README.md", dst)
    pose_src = src / "ground_truth/Indoor/pose"
    if pose_src.exists():
        pose_dst = dst / "ground_truth/Indoor/pose"
        pose_dst.mkdir(parents=True)
        for f in sorted(pose_src.glob("*.json"))[:2]:
            shutil.copy(f, pose_dst)
    mot_src = src / "ground_truth/Indoor/MOT"
    if mot_src.exists():
        mot_dst = dst / "ground_truth/Indoor/MOT"
        mot_dst.mkdir(parents=True)
        for f in sorted(mot_src.iterdir())[:2]:
            shutil.copy(f, mot_dst)
    gif_src = src / "videos/gif"
    if gif_src.exists():
        gifs = sorted(gif_src.glob("*.gif"))
        if gifs:
            gif_dst = dst / "videos/gif"
            gif_dst.mkdir(parents=True)
            shutil.copy(gifs[0], gif_dst)
    # 3 Indoor raw clips (paired with the pose annotations we already sampled).
    indoor_src = src / "videos/Indoor/raw"
    if indoor_src.exists():
        indoor_dst = dst / "videos/Indoor/raw"
        indoor_dst.mkdir(parents=True, exist_ok=True)
        # Prefer clips that pair with the pose JSONs in this sample.
        pose_stems = {p.stem for p in (dst / "ground_truth/Indoor/pose").glob("*.json")}
        clips = sorted(indoor_src.glob("*.mp4"))
        # Pick clips that match a pose JSON first (so user can run dataloader
        # sanity-checks); fill out to 3 with any remaining clips.
        paired = [c for c in clips if c.stem in pose_stems][:3]
        extras = [c for c in clips if c not in paired][: max(0, 3 - len(paired))]
        for clip in paired + extras:
            shutil.copy(clip, indoor_dst)

    # --- FineSports: 5 frames per coarse class, first-encountered clip ----
    # Walks the inner zip in central-directory order (== file order, == forward
    # seeks). Going alphabetical-by-class would force backward seeks through
    # the 16 GB stored stream and trigger ZipExtFile's read-and-discard restart,
    # blowing the timeout. The first-encountered clip per class is deterministic
    # because the inner zip's central directory is stable.
    outer = Path("/data/finesports/FineSports_CVPR2024.zip")
    inner_name = "FineSports_CVPR2024/FineSports_trimmed_video_frames.zip"
    fs_dst = examples_root / "finesports"
    fs_dst.mkdir()
    if outer.exists():
        with (
            zipfile.ZipFile(outer) as outer_zf,
            outer_zf.open(inner_name) as inner_f,
            zipfile.ZipFile(inner_f) as inner_zf,
        ):
            clip_per_class: dict[str, str] = {}
            frames_taken: dict[str, int] = defaultdict(int)
            for name in inner_zf.namelist():
                parts = name.split("/")
                if not (
                    len(parts) >= 4
                    and parts[0] == "NewFrames"
                    and name.endswith(".jpg")
                ):
                    continue
                cls, clip = parts[1], parts[2]
                clip_per_class.setdefault(cls, clip)
                if clip != clip_per_class[cls] or frames_taken[cls] >= 5:
                    continue
                out = fs_dst / Path(name)
                out.parent.mkdir(parents=True, exist_ok=True)
                with inner_zf.open(name) as f_in, open(out, "wb") as f_out:
                    f_out.write(f_in.read())
                frames_taken[cls] += 1

    # --- DeepSportradar: master JSON + 2 images x 2 arenas ----------------
    src = Path("/data/deepsportradar")
    dst = examples_root / "deepsportradar"
    dst.mkdir()
    if (src / "basketball-instants-dataset.json").exists():
        shutil.copy(src / "basketball-instants-dataset.json", dst)
    arenas = sorted([p for p in src.iterdir() if p.is_dir()])[:2]
    for arena in arenas:
        imgs = [
            p
            for p in arena.rglob("*")
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ][:2]
        for img in imgs:
            out = dst / img.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(img, out)

    # --- Roboflow basketball: 5 imgs/split + READMEs + annotation JSONs ---
    src = Path("/data/roboflow_basketball")
    dst = examples_root / "roboflow_basketball"
    dst.mkdir()
    for name in ("README.dataset.txt", "README.roboflow.txt"):
        if (src / name).exists():
            shutil.copy(src / name, dst)
    for split in ("train", "valid", "test"):
        split_src = src / split
        if not split_src.exists():
            continue
        split_dst = dst / split
        split_dst.mkdir()
        for f in split_src.glob("*.json"):
            shutil.copy(f, split_dst)
        for f in sorted(split_src.glob("*.jpg"))[:5]:
            shutil.copy(f, split_dst)

    # --- MultiSubjects: README + small SA.zip + note about the RARs -------
    src = Path("/data/multisubjects")
    dst = examples_root / "multisubjects"
    dst.mkdir()
    if (src / "README.md").exists():
        shutil.copy(src / "README.md", dst)
    sa_zip = src / "data/MultiSubjects_SA.zip"
    if sa_zip.exists():
        # Extract the small SA.zip (annotations: train.pkl + val.pkl) so the
        # examples folder shows actual files rather than an opaque archive.
        sa_dst = dst / "data" / "MultiSubjects_SA"
        sa_dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(sa_zip) as zf:
            zf.extractall(sa_dst)

    # Pull representative bits from the RAR archives. unrar + rarfile are in
    # the shared image; per-archive strategy:
    #   - MultiSubjects_person_GT.rar (25 MB)  → extract fully (small enough).
    #   - key_points_csv.rar          (372 MB) → extract the first 3 CSVs.
    #   - MultiSubjects_train_val_test.rar (20 GB) → extract the 1-2 smallest
    #     files (or first 1-2 if size sorting is unreliable) so we see at least
    #     one real clip/frame without blowing disk.
    #   - Subjects1000.rar            (20 GB) → skip; very likely same content
    #     type as train_val_test.
    import rarfile

    # The Modal image installs `unar` (not `unrar` — that's non-free in
    # Debian 12). Force rarfile to use it.
    rarfile.UNRAR_TOOL = "unar"

    def _extract_smallest(
        rar_path: Path,
        dest: Path,
        max_files: int = 2,
        size_cap_mb: int = 80,
        extensions: set[str] | None = None,
    ) -> None:
        with rarfile.RarFile(rar_path) as rf:
            infos = [i for i in rf.infolist() if not i.is_dir()]
            if extensions:
                infos = [i for i in infos if Path(i.filename).suffix.lower() in extensions]
            infos.sort(key=lambda i: i.file_size)
            taken = 0
            for info in infos:
                if taken >= max_files:
                    break
                if info.file_size > size_cap_mb * 1024 * 1024:
                    break
                try:
                    rf.extract(info, dest)
                    print(f"  extracted: {info.filename} ({info.file_size / 1e6:.1f} MB)")
                    taken += 1
                except Exception as exc:
                    print(f"  skip {info.filename}: {exc}")
            if taken == 0:
                print(f"  no matching files under {size_cap_mb} MB in {rar_path.name}")

    def _extract_one_per_class(
        rar_path: Path,
        dest: Path,
        class_letters: tuple[str, ...] = ("d", "p", "s"),
        size_cap_mb: int = 200,
    ) -> None:
        """For MultiSubjects bulk RAR — pick the smallest .mp4 for each class."""
        import re

        pat = re.compile(r"_([a-z])_\d+\.mp4$", re.IGNORECASE)
        with rarfile.RarFile(rar_path) as rf:
            infos = [i for i in rf.infolist() if not i.is_dir()]
            by_class: dict[str, list[rarfile.RarInfo]] = {c: [] for c in class_letters}
            for info in infos:
                m = pat.search(info.filename)
                if m and m.group(1).lower() in by_class:
                    by_class[m.group(1).lower()].append(info)
            for cls, candidates in by_class.items():
                candidates.sort(key=lambda i: i.file_size)
                picked = None
                for c in candidates:
                    if c.file_size <= size_cap_mb * 1024 * 1024:
                        picked = c
                        break
                if picked is None:
                    print(f"  class '{cls}': no .mp4 under {size_cap_mb} MB")
                    continue
                try:
                    rf.extract(picked, dest)
                    print(f"  class '{cls}': {picked.filename} ({picked.file_size / 1e6:.1f} MB)")
                except Exception as exc:
                    print(f"  class '{cls}' skip {picked.filename}: {exc}")

    person_gt = src / "data/MultiSubjects_person_GT.rar"
    if person_gt.exists():
        person_gt_dst = dst / "data" / "MultiSubjects_person_GT"
        person_gt_dst.mkdir(parents=True, exist_ok=True)
        try:
            with rarfile.RarFile(person_gt) as rf:
                rf.extractall(person_gt_dst)
        except Exception as exc:
            print(f"person_GT extract failed: {exc}")

    kp_csv = src / "data/key_points_csv.rar"
    if kp_csv.exists():
        kp_dst = dst / "data" / "key_points_csv_sample"
        kp_dst.mkdir(parents=True, exist_ok=True)
        _extract_smallest(kp_csv, kp_dst, max_files=3, size_cap_mb=80)

    tvt = src / "data/MultiSubjects_train_val_test.rar"
    if tvt.exists():
        tvt_dst = dst / "data" / "MultiSubjects_train_val_test_sample"
        tvt_dst.mkdir(parents=True, exist_ok=True)
        # Pull metadata first (manifests + class labels), then one .mp4 per
        # action class (d=dribble, p=layup, s=shoot) so the user sees all
        # three actions, not just one.
        _extract_smallest(
            tvt, tvt_dst, max_files=5, size_cap_mb=2, extensions={".txt"}
        )
        _extract_one_per_class(tvt, tvt_dst)

    (dst / "BULK_DATA_NOTE.md").write_text(
        "Full MultiSubjects data on the Modal Volume:\n"
        "  /data/multisubjects/data/MultiSubjects_train_val_test.rar  (~20 GB)\n"
        "  /data/multisubjects/data/Subjects1000.rar                  (~20 GB)\n"
        "  /data/multisubjects/data/MultiSubjects_person_GT.rar       (~25 MB)\n"
        "  /data/multisubjects/data/key_points_csv.rar                (~372 MB)\n"
        "  /data/multisubjects/data/MultiSubjects_SA.zip              (~36 MB)\n\n"
        "What's in this samples folder:\n"
        "  MultiSubjects_SA/             - full SA annotations (train.pkl + val.pkl)\n"
        "  MultiSubjects_person_GT/      - full person GT (small enough to include)\n"
        "  key_points_csv_sample/        - 3 smallest CSVs from the keypoints RAR\n"
        "  MultiSubjects_train_val_test_sample/ - 1-2 smallest files (one real clip)\n"
    )

    # --- summary ----------------------------------------------------------
    print(f"{'dataset':<25} {'files':>8} {'size MB':>10}")
    total_sz = 0.0
    total_files = 0
    for d in sorted(examples_root.iterdir()):
        if not d.is_dir():
            continue
        files = [f for f in d.rglob("*") if f.is_file()]
        sz = sum(f.stat().st_size for f in files) / 1e6
        total_sz += sz
        total_files += len(files)
        print(f"{d.name:<25} {len(files):>8} {sz:>10.1f}")
    print(f"{'TOTAL':<25} {total_files:>8} {total_sz:>10.1f}")

    data_volume.commit()
    return str(examples_root)
