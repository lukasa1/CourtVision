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

from courtvision.modal_app import _HF_SECRETS, app, data_volume, image

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


@app.function(image=image, volumes={"/data": data_volume}, timeout=3600 * 6, cpu=4)
def download_trackid3x3() -> str:
    # Source: TrackID3x3 (Yamada et al.) — Hugging Face TBD. Smallest dataset;
    # use as smoke test for the download pipeline.
    raise NotImplementedError("fill source URL during Phase A user-test step")


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


@app.function(image=image, volumes={"/data": data_volume}, timeout=3600 * 2, cpu=4)
def download_u_arizona() -> str:
    # Source: U Arizona Roboflow basketball — Roboflow public dataset. ~2 GB.
    raise NotImplementedError("fill source URL during Phase A user-test step")


@app.function(image=image, volumes={"/data": data_volume}, timeout=3600 * 4, cpu=4)
def download_wasb_basketball() -> str:
    # Source: WASB basketball subset (nttcom). ~5-10 GB.
    raise NotImplementedError("fill source URL during Phase A user-test step")
