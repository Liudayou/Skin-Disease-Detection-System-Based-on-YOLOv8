from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CBAM(nn.Module):
    """CBAM attention module (CBAM-style).

    This implementation is **lazy-build**: internal layers are instantiated on the first
    forward pass based on the input channel count, because Ultralytics YAML parser
    does not pass `c1/c2` to unknown custom modules.

    Args are kept simple so YAML can pass them as a list:
    - ratio: channel reduction ratio for the MLP
    - k: spatial attention kernel size (typically 7)
    """

    def __init__(self, ratio: int = 16, k: int = 7):
        super().__init__()
        self.ratio = int(ratio)
        self.k = int(k)

        self._built: bool = False
        self._c: Optional[int] = None

        # Channel attention
        self.avg_pool: Optional[nn.Module] = None
        self.max_pool: Optional[nn.Module] = None
        self.mlp: Optional[nn.Module] = None

        # Spatial attention
        self.spatial: Optional[nn.Module] = None

    def _build(self, c: int) -> None:
        hidden = max(1, int(c) // max(1, self.ratio))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP via 1x1 convs (works with NCHW tensors)
        self.mlp = nn.Sequential(
            nn.Conv2d(c, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c, kernel_size=1, bias=False),
        )

        k = int(self.k)
        if k not in (3, 5, 7):
            k = 7
        pad = k // 2
        self.spatial = nn.Conv2d(2, 1, kernel_size=k, padding=pad, bias=False)

        self._c = int(c)
        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if not self._built:
            self._build(c)

        assert self.avg_pool is not None and self.max_pool is not None and self.mlp is not None
        assert self.spatial is not None

        # Channel attention
        ca = torch.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        x = x * ca

        # Spatial attention
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.sigmoid(self.spatial(torch.cat([avg, mx], dim=1)))
        return x * sa


def register_cbam_for_ultralytics() -> None:
    """Register CBAM into Ultralytics YAML parser globals()."""
    import ultralytics.nn.tasks as tasks

    tasks.__dict__["CBAM"] = CBAM
    tasks.__dict__["CBAM"] = CBAM

