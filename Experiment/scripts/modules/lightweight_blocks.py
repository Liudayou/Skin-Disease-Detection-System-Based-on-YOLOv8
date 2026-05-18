"""Lightweight building blocks for GhostNet and MobileNetV3 backbones.

These classes MUST match the internal structure of the original .pt files.
Names like 'primary', 'cheap', 'ghost', 'conv', 'proj', 'use_res' are
exactly what the unpickled models contain.
"""

from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Ghost Module  (children: primary, cheap; attr: oup)
# ---------------------------------------------------------------------------

class GhostModule(nn.Module):
    """GhostV1 module with 'primary' and 'cheap' sub-sequential blocks."""

    def __init__(self, oup: int, kernel: int = 1, ratio: int = 2, stride: int = 1):
        super().__init__()
        self.oup = int(oup)
        init_ch = math.ceil(self.oup / max(1, int(ratio)))
        new_ch = init_ch * (max(1, int(ratio)) - 1)
        self.primary = nn.Sequential(
            nn.Conv2d(0, init_ch, int(kernel), int(stride), int(kernel) // 2, bias=False),
            nn.BatchNorm2d(init_ch),
            nn.SiLU(inplace=True),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(init_ch, new_ch, 3, 1, 1, groups=init_ch, bias=False),
            nn.BatchNorm2d(new_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary(x)
        x2 = self.cheap(x1)
        return torch.cat([x1, x2], dim=1)[:, :getattr(self, 'oup', x1.shape[1] + x2.shape[1])]


# ---------------------------------------------------------------------------
# GhostNet Bottleneck  (children: ghost)
# ---------------------------------------------------------------------------

class GhostNetBottleneck(nn.Module):
    """GhostV1 bottleneck with single 'ghost' module and internal DW + SE."""

    def __init__(self, c_mid: int, c_out: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        self.ghost = GhostModule(int(c_mid), kernel=1, ratio=2, stride=1)
        k = int(kernel)
        s = int(stride)
        c_mid = int(c_mid)
        c_out = int(c_out)
        if s > 1:
            self.conv_dw = nn.Sequential(
                nn.Conv2d(c_mid, c_mid, k, s, k // 2, groups=c_mid, bias=False),
                nn.BatchNorm2d(c_mid),
            )
        else:
            self.conv_dw = nn.Identity()
        self.ghost2 = GhostModule(c_out, kernel=1, ratio=2, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.ghost(x)
        if hasattr(self, 'conv_dw'):
            y = self.conv_dw(y)
        if hasattr(self, 'ghost2'):
            y = self.ghost2(y)
        # Handle shape mismatch (no explicit shortcut in unpickled model)
        if y.shape != residual.shape:
            residual = nn.functional.adaptive_avg_pool2d(residual, y.shape[2:])
            if residual.shape[1] != y.shape[1]:
                residual = nn.functional.pad(residual, [0, 0, 0, 0, 0, y.shape[1] - residual.shape[1]])
        return y + residual


GhostBottleneck = GhostNetBottleneck


# ---------------------------------------------------------------------------
# GhostNet Down  (children: down, blocks)
# ---------------------------------------------------------------------------

class GhostNetDown(nn.Module):
    def __init__(self, c_out: int, kernel: int = 3, ratio: int = 1, stride: int = 2):
        super().__init__()
        c_mid = int(c_out) * max(1, int(ratio))
        self.down = GhostModule(int(c_mid), kernel=1, ratio=2, stride=1)
        self.blocks = nn.ModuleList([
            GhostNetBottleneck(c_mid, int(c_out), kernel=int(kernel), stride=int(stride))
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        for blk in self.blocks:
            x = blk(x)
        return x


# ---------------------------------------------------------------------------
# GhostNet Stage  (children: proj, blocks)
# ---------------------------------------------------------------------------

class GhostNetStage(nn.Module):
    def __init__(self, c_out: int, kernel: int = 3, num_blocks: int = 2):
        super().__init__()
        c_out = int(c_out)
        kernel = int(kernel)
        num_blocks = max(1, int(num_blocks))
        self.proj = nn.Identity()  # placeholder, unpickled model may have real proj
        self.blocks = nn.ModuleList([
            GhostNetBottleneck(c_out, c_out, kernel=kernel, stride=1)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        for blk in self.blocks:
            x = blk(x)
        return x


# ---------------------------------------------------------------------------
# MobileNetV3 helpers
# ---------------------------------------------------------------------------

class SEModule(nn.Module):
    """Squeeze-and-Excitation.

    Supports both new-code and unpickled .pt structures:
    - New: avg_pool, fc1, act, fc2, gate
    - Unpickled: pool, fc (Sequential)
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(1, int(channels) // max(1, int(reduction)))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(int(channels), mid, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, int(channels), 1, 1, 0, bias=True),
            nn.Hardsigmoid(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Support both unpickled (pool+fc) and new code (pool+fc or avg_pool+fc1+...)
        pool = getattr(self, 'pool', None) or getattr(self, 'avg_pool', None)
        fc = getattr(self, 'fc', None)
        if fc is not None:
            y = fc(pool(x))
        else:
            # New code style
            y = self.fc2(self.act(self.fc1(pool(x))))
            y = self.gate(y)
        return x * y


class HSwish(nn.Module):
    """Hardswish with 'hs' child (Hsigmoid) for unpickled model compatibility."""
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.hs = Hsigmoid()
        self.inplace = inplace
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * nn.functional.relu6(x + 3, inplace=getattr(self, 'inplace', True)) / 6


class Hsigmoid(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.relu6(x + 3, inplace=True) / 6


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ---------------------------------------------------------------------------
# _MV3Block  (children: conv; attr: use_res)
# ---------------------------------------------------------------------------

class _MV3Block(nn.Module):
    """MobileNetV3 inverted residual with 'conv' child and 'use_res' flag."""

    def __init__(self, c_out: int, kernel: int = 3, stride: int = 1, expand: float = 1.0, use_se: bool = False):
        super().__init__()
        self.use_res = (int(stride) == 1)  # residual when stride=1
        c_out = int(c_out)
        k = int(kernel)
        s = int(stride)
        # Build conv as a single Sequential (matching original structure)
        mid = max(1, c_out * max(1, int(expand)))  # approximate
        layers = []
        # DW + PW structure
        layers.extend([
            nn.Conv2d(0, mid, k, s, k // 2, groups=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.Hardswish(inplace=True),
        ])
        if bool(use_se):
            layers.append(SEModule(mid))
        layers.extend([
            nn.Conv2d(mid, c_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_out),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if getattr(self, 'use_res', False) and out.shape == x.shape:
            return out + x
        return out


MV3InvertedResidual = _MV3Block


# ---------------------------------------------------------------------------
# MV3Down  (children: blocks)
# ---------------------------------------------------------------------------

class MV3Down(nn.Module):
    def __init__(self, c_out: int, kernel: int = 3, stride: int = 2, expand: float = 6.0, use_se: int = 0):
        super().__init__()
        self.blocks = nn.ModuleList([
            _MV3Block(int(c_out), kernel=int(kernel), stride=int(stride), expand=float(expand), use_se=bool(int(use_se)))
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


# ---------------------------------------------------------------------------
# MV3Stage  (children: proj, blocks)
# ---------------------------------------------------------------------------

class MV3Stage(nn.Module):
    def __init__(self, c_out: int, kernel: int = 3, num_blocks: int = 2, expand: float = 6.0, use_se: int = 1):
        super().__init__()
        c_out = int(c_out)
        kernel = int(kernel)
        num_blocks = max(1, int(num_blocks))
        expand = float(expand)
        use_se = bool(int(use_se))
        self.proj = nn.Identity()
        self.blocks = nn.ModuleList([
            _MV3Block(c_out, kernel=kernel, stride=1, expand=expand, use_se=use_se)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        for blk in self.blocks:
            x = blk(x)
        return x


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_lightweight_blocks_for_ultralytics() -> None:
    import ultralytics.nn.tasks as tasks
    for name, cls in [
        ("GhostModule", GhostModule),
        ("GhostNetBottleneck", GhostNetBottleneck),
        ("GhostBottleneck", GhostBottleneck),
        ("GhostNetDown", GhostNetDown),
        ("GhostNetStage", GhostNetStage),
        ("MV3Down", MV3Down),
        ("MV3Stage", MV3Stage),
        ("_MV3Block", _MV3Block),
        ("MV3InvertedResidual", MV3InvertedResidual),
        ("HSwish", HSwish),
        ("Hsigmoid", Hsigmoid),
        ("SEModule", SEModule),
        ("Identity", Identity),
    ]:
        tasks.__dict__[name] = cls
