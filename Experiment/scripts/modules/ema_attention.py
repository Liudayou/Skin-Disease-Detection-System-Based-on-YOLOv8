from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class EMA(nn.Module):
    """Efficient Multi-scale Attention (EMA), implemented with lazy channel build.

    Notes
    - Ultralytics YAML parser does not automatically pass input channels to unknown custom modules.
      This module therefore builds its internal layers at first forward based on `x.shape[1]`.
    - Designed to be lightweight: group-wise computation + small convs.
    """

    def __init__(self, factor: int = 32):
        super().__init__()
        self.factor = int(factor)

        # Will be built on first forward
        self._built: bool = False
        self.groups: Optional[int] = None
        self.c_per_group: Optional[int] = None

        self.pool_h: Optional[nn.Module] = None
        self.pool_w: Optional[nn.Module] = None
        self.agp: Optional[nn.Module] = None
        self.conv1x1: Optional[nn.Module] = None
        self.conv3x3: Optional[nn.Module] = None
        self.gn: Optional[nn.Module] = None
        self.softmax: Optional[nn.Module] = None

    def _choose_groups(self, c: int) -> int:
        g = min(int(self.factor), int(c))
        while g > 1 and c % g != 0:
            g -= 1
        return max(1, g)

    def _build(self, c: int) -> None:
        g = self._choose_groups(c)
        cg = c // g

        # Pooling ops
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

        # Lightweight convs per group (operate on cg channels)
        self.conv1x1 = nn.Conv2d(cg, cg, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv3x3 = nn.Conv2d(cg, cg, kernel_size=3, stride=1, padding=1, bias=True)

        # Group-wise normalization (InstanceNorm-like within each group)
        self.gn = nn.GroupNorm(1, cg)
        self.softmax = nn.Softmax(dim=-1)

        self.groups = g
        self.c_per_group = cg
        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if not self._built:
            self._build(c)

        assert self.groups is not None and self.c_per_group is not None
        assert self.pool_h is not None and self.pool_w is not None and self.agp is not None
        assert self.conv1x1 is not None and self.conv3x3 is not None and self.gn is not None and self.softmax is not None

        g = self.groups
        cg = self.c_per_group

        # group reshape: (b*g, cg, h, w)
        xg = x.reshape(b * g, cg, h, w)

        # Coordinate attention-like gates
        x_h = self.pool_h(xg)  # (b*g, cg, h, 1)
        x_w = self.pool_w(xg).permute(0, 1, 3, 2)  # (b*g, cg, w, 1)
        y = torch.cat([x_h, x_w], dim=2)  # (b*g, cg, h+w, 1)
        y = self.conv1x1(y)

        y_h, y_w = torch.split(y, [h, w], dim=2)
        a_h = torch.sigmoid(y_h)
        a_w = torch.sigmoid(y_w.permute(0, 1, 3, 2))

        x1 = xg * a_h * a_w

        # Local+global interaction
        x2 = self.gn(x1)
        x3 = self.conv3x3(x2)

        # Attention weights from pooled channel descriptors
        # w1/w2: (b*g, 1, cg)
        w1 = self.softmax(self.agp(x2).reshape(b * g, 1, cg))
        w2 = self.softmax(self.agp(x3).reshape(b * g, 1, cg))

        # features: (b*g, cg, h*w)
        f1 = x3.reshape(b * g, cg, h * w)
        f2 = x2.reshape(b * g, cg, h * w)

        # (b*g, 1, h*w) -> (b*g, 1, h, w)
        att = (w1 @ f1 + w2 @ f2).reshape(b * g, 1, h, w)
        out = (x1 * torch.sigmoid(att)).reshape(b, c, h, w)
        return out


def register_ema_for_ultralytics() -> None:
    """Register EMA into Ultralytics YAML parser globals()."""
    import ultralytics.nn.tasks as tasks

    tasks.__dict__["EMA"] = EMA

