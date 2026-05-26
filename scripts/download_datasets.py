"""Dataset downloaders. PRD §8.3.

Each function downloads one dataset directly onto the
``courtvision-data`` Modal Volume. The PRD doesn't specify source URLs —
they get filled during the Phase A user-test step (or weeks 3-4 for own
footage). Until then each function raises ``NotImplementedError`` with the
canonical source as a comment so the user knows where to look.

Usage::

    modal run scripts/download_datasets.py::download_trackid3x3
"""

from __future__ import annotations

from courtvision.modal_app import app, data_volume, image


@app.function(image=image, volumes={"/data": data_volume}, timeout=3600 * 6, cpu=4)
def download_trackid3x3() -> str:
    # Source: TrackID3x3 (Yamada et al.) — Hugging Face TBD. Smallest dataset;
    # use as smoke test for the download pipeline.
    raise NotImplementedError("fill source URL during Phase A user-test step")


@app.function(image=image, volumes={"/data": data_volume}, timeout=3600 * 8, cpu=4)
def download_multisubjects() -> str:
    # Source: MultiSubjects (Xu et al.) — Hugging Face. ~30-50 GB.
    raise NotImplementedError("fill source URL during Phase A user-test step")


@app.function(image=image, volumes={"/data": data_volume}, timeout=3600 * 12, cpu=4)
def download_finesports() -> str:
    # Source: FineSports (Xu et al.) — gated access; request from authors.
    # ~50-100 GB. If unavailable, Path A fallback (skip transfer stage 3).
    raise NotImplementedError("fill source URL during Phase A user-test step")


@app.function(image=image, volumes={"/data": data_volume}, timeout=3600 * 2, cpu=4)
def download_u_arizona() -> str:
    # Source: U Arizona Roboflow basketball — Roboflow public dataset. ~2 GB.
    raise NotImplementedError("fill source URL during Phase A user-test step")


@app.function(image=image, volumes={"/data": data_volume}, timeout=3600 * 4, cpu=4)
def download_wasb_basketball() -> str:
    # Source: WASB basketball subset (nttcom). ~5-10 GB.
    raise NotImplementedError("fill source URL during Phase A user-test step")
