from __future__ import annotations

import torch
import torch.nn as nn


class PConv(nn.Module):
    """FasterNet PConv (Partial Convolution) as a drop-in Conv replacement."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        p: int | None = None,
        partial: int = 4,
        act: bool = True,
    ) -> None:
        super().__init__()
        c1 = int(c1)
        c2 = int(c2)
        partial = max(1, int(partial))

        c_partial = max(1, c1 // partial)
        c_bypass = c1 - c_partial
        if p is None:
            p = k // 2

        self.c1 = c1
        self.c2 = c2
        self.c_partial = c_partial
        self.c_bypass = c_bypass

        self.stride = int(s)
        self.conv_partial = nn.Conv2d(c_partial, c_partial, k, self.stride, p, bias=False)
        self.bn_partial = nn.BatchNorm2d(c_partial)

        self.conv_pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn_pw = nn.BatchNorm2d(c2)

        self.downsample = nn.AvgPool2d(kernel_size=self.stride, stride=self.stride) if self.stride > 1 else nn.Identity()
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xp, xb = torch.split(x, [self.c_partial, self.c_bypass], dim=1)
        xb = self.downsample(xb)
        yp = self.act(self.bn_partial(self.conv_partial(xp)))
        y = torch.cat([yp, xb], dim=1)
        y = self.act(self.bn_pw(self.conv_pw(y)))
        return y


def register_pconv_for_ultralytics() -> None:
    """Expose PConv symbol for Ultralytics YAML parsing."""
    import ultralytics.nn.tasks as tasks

    tasks.__dict__["PConv"] = PConv

