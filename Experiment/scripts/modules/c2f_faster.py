from __future__ import annotations

import ast
import contextlib
from typing import Any

import torch
import torch.nn as nn


class PartialConv3(nn.Module):
    """FasterNet Partial 3x3 convolution (split-cat)."""

    def __init__(self, dim: int, n_div: int = 4):
        super().__init__()
        dim = int(dim)
        n_div = max(1, int(n_div))
        self.dim_conv3 = max(1, dim // n_div)
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        return torch.cat((x1, x2), dim=1)


class FasterBlock(nn.Module):
    """FasterNet block: partial spatial mixing + MLP with residual."""

    def __init__(self, dim: int, n_div: int = 4, mlp_ratio: float = 2.0, drop_path: float = 0.0):
        super().__init__()
        dim = int(dim)
        self.spatial_mixing = PartialConv3(dim, n_div=n_div)
        hidden = max(1, int(dim * float(mlp_ratio)))
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=False),
        )
        self.drop = nn.Dropout2d(float(drop_path)) if float(drop_path) > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop(self.mlp(x))
        return x


class C2f_Faster(nn.Module):
    """C2f variant where the internal blocks are FasterNet blocks."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        n_div: int = 4,
        mlp_ratio: float = 2.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv

        self.c = int(c2 * float(e))
        self.cv1 = Conv(int(c1), 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + int(n)) * self.c, int(c2), 1, 1)
        self.m = nn.ModuleList(
            FasterBlock(self.c, n_div=int(n_div), mlp_ratio=float(mlp_ratio), drop_path=float(drop_path)) for _ in range(int(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


def register_c2f_faster_for_ultralytics() -> None:
    """Register C2f_Faster and patch Ultralytics parse_model to treat it as a repeat/base module."""
    import ultralytics.nn.tasks as tasks

    from fpn_fusion import ASFFLite2, BiFPNFuse

    tasks.__dict__["C2f_Faster"] = C2f_Faster
    tasks.__dict__["BiFPNFuse"] = BiFPNFuse
    tasks.__dict__["ASFFLite2"] = ASFFLite2

    if getattr(tasks, "_c2f_faster_parse_model_patched", False) and getattr(tasks.parse_model, "__name__", "") == "parse_model_extended":
        return

    import torch as _torch

    LOGGER = tasks.LOGGER
    colorstr = tasks.colorstr
    make_divisible = tasks.make_divisible

    base_module_names = {
        "Classify",
        "Conv",
        "ConvTranspose",
        "GhostConv",
        "Bottleneck",
        "GhostBottleneck",
        "SPP",
        "SPPF",
        "C2fPSA",
        "C2PSA",
        "DWConv",
        "Focus",
        "BottleneckCSP",
        "C1",
        "C2",
        "C2f",
        "C3k2",
        "RepNCSPELAN4",
        "ELAN1",
        "ADown",
        "AConv",
        "SPPELAN",
        "C2fAttn",
        "C3",
        "C3TR",
        "C3Ghost",
        "DWConvTranspose2d",
        "C3x",
        "RepC3",
        "PSA",
        "SCDown",
        "C2fCIB",
        "A2C2f",
        "PConv",
        "C2f_Faster",
        "MV3Down",
        "MV3Stage",
        "GhostNetDown",
        "GhostNetStage",
        "EMBLiteDown",
        "EMBLiteStage",
    }

    repeat_module_names = {
        "BottleneckCSP",
        "C1",
        "C2",
        "C2f",
        "C3k2",
        "C2fAttn",
        "C3",
        "C3TR",
        "C3Ghost",
        "C3x",
        "RepC3",
        "C2fPSA",
        "C2fCIB",
        "C2PSA",
        "A2C2f",
        "C2f_Faster",
    }

    def parse_model_extended(d: dict, ch: int, verbose: bool = True):
        legacy = True
        max_channels = float("inf")
        nc, act, scales, end2end = (d.get(x) for x in ("nc", "activation", "scales", "end2end"))
        reg_max = d.get("reg_max", 16)
        depth, width = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple"))
        scale = d.get("scale")
        if scales:
            if not scale:
                scale = next(iter(scales.keys()))
                LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
            depth, width, max_channels = scales[scale]

        Conv = tasks.Conv
        if act:
            Conv.default_act = eval(act)
            if verbose:
                LOGGER.info(f"{colorstr('activation:')} {act}")

        if verbose:
            LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")

        ch = [ch]
        layers, save, c2 = [], [], ch[-1]

        base_modules = frozenset(tasks.__dict__[n] for n in base_module_names if n in tasks.__dict__)
        repeat_modules = frozenset(tasks.__dict__[n] for n in repeat_module_names if n in tasks.__dict__)

        for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
            m = getattr(_torch.nn, m[3:]) if "nn." in m else tasks.__dict__[m]
            for j, a in enumerate(args):
                if isinstance(a, str):
                    with contextlib.suppress(ValueError):
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

            n = n_ = max(round(n * depth), 1) if n > 1 else n

            if m in base_modules:
                c1, c2 = ch[f], args[0]
                if c2 != nc:
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
                if m is tasks.C2fAttn:
                    args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                    args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])
                args = [c1, c2, *args[1:]]
                if m in repeat_modules:
                    args.insert(2, n)
                    n = 1

            elif m is BiFPNFuse or m is ASFFLite2:
                c1a, c1b = ch[f[0]], ch[f[1]]
                c2 = make_divisible(min(int(args[0]), max_channels) * width, 8)
                args = [c1a, c1b, c2]
            elif m is tasks.Concat:
                c2 = sum(ch[x] for x in f)
            elif m in frozenset({tasks.Detect, tasks.Segment, tasks.Pose, tasks.OBB, tasks.WorldDetect, tasks.YOLOEDetect}):
                args.extend([reg_max, end2end, [ch[x] for x in f]])
                if m in {tasks.Detect, tasks.YOLOEDetect, tasks.Segment, tasks.Pose, tasks.OBB}:
                    m.legacy = legacy
            elif m in frozenset({tasks.TorchVision, tasks.Index}):
                c2 = args[0]
                c1 = ch[f]
                args = [*args[1:]]
            else:
                c2 = ch[f]

            m_ = _torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            t = str(m)[8:-2].replace("__main__.", "")
            m_.np = sum(x.numel() for x in m_.parameters())
            m_.i, m_.f, m_.type = i, f, t
            if verbose:
                LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)
        return _torch.nn.Sequential(*layers), sorted(save)

    tasks.parse_model = parse_model_extended  # type: ignore[assignment]
    tasks._c2f_faster_parse_model_patched = True

