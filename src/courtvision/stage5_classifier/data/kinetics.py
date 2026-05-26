"""Kinetics-400 — download-only (pretrained checkpoint, no dataloading needed).

Transfer stage 1 just initializes from MMAction2's published Kinetics-400
checkpoint; we never iterate Kinetics samples ourselves.
"""

from __future__ import annotations
