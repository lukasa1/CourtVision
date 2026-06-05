"""Stage 6 — 2D single-camera pose-only release-frame extraction.

No ball, no 3D: ``pose_apex`` (primary, wrist-height apex) and ``pose_forward``
(secondary, onset of overhead arm extension). The ball-based forward/backward
rules in the PRD await the future 3D variant. See README + flags F23.
"""

from .detect import analyze_release, is_verified_shot
from .rule_backward import detect_release_apex
from .rule_forward import detect_release_forward

__all__ = [
    "analyze_release",
    "detect_release_apex",
    "detect_release_forward",
    "is_verified_shot",
]
