from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MobileNetV3LargeMapping:
    """Default mapping for MobileNetV3-Large features to YOLO P3/P4/P5.

    timm `mobilenetv3_large_100` with `features_only=True` exposes reductions:
    [2, 4, 8, 16, 32]. We use (8, 16, 32) as (P3, P4, P5).
    """

    out_indices: tuple[int, int, int] = (2, 3, 4)  # reductions 8, 16, 32


class MobileNetV3LargeBackbone(nn.Module):
    """MobileNetV3-Large backbone wrapper (timm) that returns P3/P4/P5 features.

    Returns a list: [P3(stride=8), P4(stride=16), P5(stride=32)].
    """

    def __init__(
        self,
        pretrained: bool = False,
        out_indices: Iterable[int] = (2, 3, 4),
        model_name: str = "mobilenetv3_large_100",
    ) -> None:
        super().__init__()
        try:
            import timm  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Missing dependency 'timm'. Install with: pip install timm") from e

        out_idx = tuple(int(i) for i in out_indices)
        self.mapping = MobileNetV3LargeMapping(out_indices=out_idx)

        self.backbone = timm.create_model(
            str(model_name),
            pretrained=bool(pretrained),
            features_only=True,
            out_indices=out_idx,
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = self.backbone(x)
        return list(feats)

