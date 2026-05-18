from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class ECA(nn.Module):
    """ECA-Net: Efficient Channel Attention.

    Lazy-build implementation for Ultralytics YAML custom modules.

    Args:
        k: 1D conv kernel size for channel interaction. If set to 0, use an
           adaptive kernel computed from channel count (paper-style).
    """

    def __init__(self, k: int = 0):
        super().__init__()
        self.k = int(k)

        self._built: bool = False
        self.avg_pool: Optional[nn.Module] = None
        self.conv1d: Optional[nn.Module] = None

    @staticmethod
    def _adaptive_kernel(c: int, gamma: float = 2.0, b: float = 1.0) -> int:
        # Paper heuristic: k = | (log2(C)/gamma + b) |_odd
        t = int(abs((math.log2(max(1, int(c))) / gamma) + b))
        k = t if t % 2 == 1 else t + 1
        return max(3, int(k))

    def _build(self, c: int) -> None:
        k = self.k if self.k > 0 else self._adaptive_kernel(c)
        if k % 2 == 0:
            k += 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 1D conv over channel descriptor (C as length)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=int(k), padding=int(k) // 2, bias=False)
        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if not self._built:
            self._build(c)
        assert self.avg_pool is not None and self.conv1d is not None

        y = self.avg_pool(x)  # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = self.conv1d(y)  # (B, 1, C)
        y = torch.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y


def register_eca_for_ultralytics() -> None:
    """Register ECA into Ultralytics YAML parser globals()."""
    import ultralytics.nn.tasks as tasks

    tasks.__dict__["ECA"] = ECA

