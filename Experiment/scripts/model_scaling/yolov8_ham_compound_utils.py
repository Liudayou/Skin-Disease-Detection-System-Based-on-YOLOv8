"""生成 YOLOv8 HAM（nc=7）复合缩放 YAML 配置文件。

注意：Ultralytics 会从文件名猜测 scale，如果文件名不匹配 yolov8[nslmx]，
则回退到 scales 字典的第一个 key。因此 YAML 只写一个 scales entry（如 student），
文件名用 compound_ham_xxx.yaml 确保选中。

width 计算公式：w = 0.75 * sqrt(1 - r)，r 为目标参数量缩减比例。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

# Official YOLOv8 "m" compound constants (ultralytics yolov8.yaml).
M_DEPTH = 0.67
M_WIDTH = 0.75
M_MAX_CHANNELS = 768

# Backbone + head copied from ultralytics/cfg/models/v8/yolov8.yaml (structure only).
_YOLOV8_TEMPLATE: dict[str, Any] = {
    "nc": 7,
    "backbone": [
        [-1, 1, "Conv", [64, 3, 2]],
        [-1, 1, "Conv", [128, 3, 2]],
        [-1, 3, "C2f", [128, True]],
        [-1, 1, "Conv", [256, 3, 2]],
        [-1, 6, "C2f", [256, True]],
        [-1, 1, "Conv", [512, 3, 2]],
        [-1, 6, "C2f", [512, True]],
        [-1, 1, "Conv", [1024, 3, 2]],
        [-1, 3, "C2f", [1024, True]],
        [-1, 1, "SPPF", [1024, 5]],
    ],
    "head": [
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 6], 1, "Concat", [1]],
        [-1, 3, "C2f", [512]],
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 4], 1, "Concat", [1]],
        [-1, 3, "C2f", [256]],
        [-1, 1, "Conv", [256, 3, 2]],
        [[-1, 12], 1, "Concat", [1]],
        [-1, 3, "C2f", [512]],
        [-1, 1, "Conv", [512, 3, 2]],
        [[-1, 9], 1, "Concat", [1]],
        [-1, 3, "C2f", [1024]],
        [[15, 18, 21], 1, "Detect", ["nc"]],
    ],
}


def width_for_target_param_reduction(reduce_frac: float) -> float:
    """根据目标参数量缩减比例计算 width_multiple。

    r=0 返回官方 m 宽度，r>0 按 width² 近似参数量关系计算。
    """
    r = float(reduce_frac)
    if r < 0 or r >= 0.95:
        raise ValueError("reduce_frac must be in [0, 0.95)")
    if r == 0.0:
        return float(M_WIDTH)
    return float(M_WIDTH * math.sqrt(1.0 - r))


def compound_triplet(
    *,
    reduce_frac: float | None = None,
    depth: float | None = None,
    width: float | None = None,
    max_channels: int | None = None,
) -> tuple[float, float, int]:
    """返回学生模型的 (depth, width, max_channels) 三元组。"""
    d = float(M_DEPTH if depth is None else depth)
    mc = int(M_MAX_CHANNELS if max_channels is None else max_channels)
    if width is not None:
        w = float(width)
    elif reduce_frac is not None:
        w = width_for_target_param_reduction(reduce_frac)
    else:
        w = float(M_WIDTH)
    return d, w, mc


def build_student_yaml_dict(
    *,
    reduce_frac: float | None = None,
    depth: float | None = None,
    width: float | None = None,
    max_channels: int | None = None,
    nc: int = 7,
) -> dict[str, Any]:
    d, w, mc = compound_triplet(
        reduce_frac=reduce_frac, depth=depth, width=width, max_channels=max_channels
    )
    cfg = yaml.safe_load(yaml.safe_dump(_YOLOV8_TEMPLATE, sort_keys=False))
    cfg["nc"] = int(nc)
    # Single key so parse_model always picks it when guess_model_scale is "".
    cfg["scales"] = {"student": [d, w, mc]}
    return cfg


def write_student_yaml(path: Path, cfg: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def stem_for_preset(preset: str) -> str:
    """生成 YAML 文件名前缀，不能匹配 yolov8[nslmx]（否则 Ultralytics 会猜错 scale）。"""
    safe = preset.strip().replace("/", "_").replace(" ", "_")
    return f"compound_ham_{safe}"


PRESET_REDUCE_FRAC: dict[str, float] = {
    "m_ref": 0.0,  # same width as YOLOv8m (sanity / ablation)
    "lite20": 0.20,
    "lite30": 0.30,
    "lite40": 0.40,
}
