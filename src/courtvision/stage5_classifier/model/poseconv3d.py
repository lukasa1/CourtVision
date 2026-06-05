"""PoseConv3D SlowOnly-R50 model factory. PRD §9.5.

``build_poseconv3d`` returns a thin ``nn.Module`` (backbone + cls head) that
forwards a raw keypoint-heatmap tensor ``[B, K=17, T, H, W] → [B, num_classes]``
— the interface the PRD E1 acceptance test and the Stage 5 inference scaffold
expect. Training (E2/E3) is driven separately by the MMAction2 ``Runner`` from
a config, so this factory is for the forward/acceptance/inference path.

Imports MMAction2, so it only runs inside the dedicated ``poseconv3d_image``
(``scripts/stage5_modal.py``). All heavy imports are deferred into the function.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

_MMACTION_ROOT = "/opt/mmaction2"
# SlowOnly-R50 PoseConv3D pretrained on Kinetics-400 keypoint heatmaps (PRD §9.5.1).
_CONFIG_GLOB = "configs/skeleton/posec3d/slowonly_r50_*u48*k400-keypoint.py"


def _find_config(mmaction_root: str) -> str:
    matches = sorted(glob.glob(str(Path(mmaction_root) / _CONFIG_GLOB)))
    if not matches:
        raise FileNotFoundError(
            f"no posec3d k400-keypoint config under {mmaction_root}/{_CONFIG_GLOB}"
        )
    return matches[0]


def build_poseconv3d(
    num_classes: int = 4,
    pretrained_path: Path | None = None,
    mmaction_root: str = _MMACTION_ROOT,
) -> Any:  # nn.Module — typed Any to avoid a top-level torch import
    """Build SlowOnly-R50 PoseConv3D with a fresh ``num_classes`` head.

    Builds the backbone + cls head from MMAction2's K400-keypoint config (so the
    architecture exactly matches the pretrained checkpoint), reinitialises the
    head to ``num_classes``, and — if ``pretrained_path`` is given — loads the
    K400 **backbone** weights (the 400-class head is intentionally dropped).
    """
    import mmaction  # noqa: F401 — registers MMAction2 modules in the registry
    import torch
    from mmaction.registry import MODELS
    from mmengine.config import Config
    from torch import nn

    cfg = Config.fromfile(_find_config(mmaction_root))
    backbone = MODELS.build(cfg.model.backbone)
    head_cfg = cfg.model.cls_head
    head_cfg.num_classes = num_classes
    cls_head = MODELS.build(head_cfg)

    if pretrained_path is not None:
        ckpt = torch.load(str(pretrained_path), map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        backbone_state = {
            k[len("backbone.") :]: v for k, v in state.items() if k.startswith("backbone.")
        }
        incompat = backbone.load_state_dict(backbone_state, strict=False)
        if incompat.missing_keys:
            raise RuntimeError(f"K400 backbone load missing keys: {incompat.missing_keys[:5]}")

    class PoseC3DClassifier(nn.Module):
        """backbone + cls head, callable on a raw ``[B, K, T, H, W]`` heatmap tensor."""

        def __init__(self, backbone: nn.Module, cls_head: nn.Module) -> None:
            super().__init__()
            self.backbone = backbone
            self.cls_head = cls_head

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feat = self.backbone(x)
            logits: torch.Tensor = self.cls_head(feat)
            return logits

    return PoseC3DClassifier(backbone, cls_head)


__all__ = ["build_poseconv3d"]
