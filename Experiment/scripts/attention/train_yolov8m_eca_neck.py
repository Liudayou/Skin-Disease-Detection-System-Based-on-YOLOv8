#!/usr/bin/env python3
"""
注意力消融实验 — ECA 注意力模块插入 Neck (PAN-FPN)。
默认数据增强对齐 lowMixup015。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 向上查找 yolov8m_train_utils.py 所在目录
_SCRIPTS_ROOT = Path(__file__).resolve().parent
for _ in range(3):
    if (_SCRIPTS_ROOT / "yolov8m_train_utils.py").is_file():
        break
    _SCRIPTS_ROOT = _SCRIPTS_ROOT.parent
else:
    raise RuntimeError("找不到 yolov8m_train_utils.py 所在的 scripts 根目录")
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

# 注册 ECA 自定义算子到 Ultralytics
from modules.eca_attention import register_eca_for_ultralytics
from yolov8m_train_utils import (
    PROJECT_ROOT,
    add_model_name_cli_arg,
    add_mosaic_mixup_cli_args,
    run_training,
    train_aug_kwargs_from_ns,
)

def _default_weights() -> str:
    local = PROJECT_ROOT / "scripts" / "models" / "yolov8m.pt"
    return str(local) if local.is_file() else "yolov8m.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-root",
        type=str,
        default=str(PROJECT_ROOT / "dataset"),
        help="YOLO 数据集根目录（含 train/valid/test；已整理到 ExperimentCatalog_HAM_YOLOv8/dataset）",
    )
    p.add_argument(
        "--train-rel",
        type=str,
        default="train/images",
        help="训练集 images 相对 data-root 的路径",
    )
    p.add_argument(
        "--model-cfg",
        type=str,
        default=str(PROJECT_ROOT / "GraduationScripts" / "models" / "yolov8m_eca_neck.yaml"),
        help="ECA-Neck 模型 YAML",
    )
    p.add_argument("--weights", type=str, default=_default_weights(), help="用于迁移加载的预训练权重")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--no-amp", action="store_true", help="禁用 AMP")
    p.add_argument("--device", type=str, default="", help="留空则交给 Ultralytics 自动选择")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--run-name",
        type=str,
        default="yolov8m_eca_neck_ham_univ30_lowMixup015",
        help="实验名前缀；默认会自动加上 _e{epochs}",
    )
    p.add_argument("--no-epoch-tag", action="store_true", help="不对 run-name 追加 _e{epochs}")
    add_mosaic_mixup_cli_args(p, mosaic_default=1.0, mixup_default=0.15, close_mosaic_default=10)
    p.add_argument("--scale", type=float, default=0.5, help="几何 scale（lowMixup015 默认 0.5）")
    p.add_argument("--degrees", type=float, default=None)
    p.add_argument("--translate", type=float, default=None)
    add_model_name_cli_arg(p)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    tag = a.run_name if a.no_epoch_tag else f"{a.run_name}_e{a.epochs}"
    print(f"[run-name] 使用: {tag}" + ("（已禁用 _e 后缀）" if a.no_epoch_tag else ""))

    register_eca_for_ultralytics()

    run_training(
        run_name=tag,
        dataset_root=Path(a.data_root),
        train_rel=str(a.train_rel),
        model_cfg=Path(a.model_cfg),
        weights=str(a.weights),
        model_name=(a.model_name.strip() or None),
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        workers=a.workers,
        device=a.device or None,
        seed=a.seed,
        amp=not bool(a.no_amp),
        scale=float(a.scale) if a.scale is not None else None,
        degrees=float(a.degrees) if a.degrees is not None else None,
        translate=float(a.translate) if a.translate is not None else None,
        **train_aug_kwargs_from_ns(a),
    )


if __name__ == "__main__":
    main()
