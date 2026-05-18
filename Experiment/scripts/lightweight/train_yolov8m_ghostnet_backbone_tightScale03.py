#!/usr/bin/env python3
"""
轻量化消融实验 — GhostNet Backbone 替换实验。
增强参数对齐 tightScale03：mosaic=1.0、mixup=0.2、close_mosaic=10、scale=0.3。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS_ROOT = Path(__file__).resolve().parent
for _ in range(3):
    if (_SCRIPTS_ROOT / "yolov8m_train_utils.py").is_file():
        break
    _SCRIPTS_ROOT = _SCRIPTS_ROOT.parent
else:
    raise RuntimeError("找不到 yolov8m_train_utils.py 所在目录")
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

# 注册 GhostNet 等轻量化模块到 Ultralytics
from modules.lightweight_blocks import register_lightweight_blocks_for_ultralytics
from yolov8m_train_utils import PROJECT_ROOT, run_training

def _default_weights() -> str:
    local = PROJECT_ROOT / "scripts" / "models" / "yolov8m.pt"
    return str(local) if local.is_file() else "yolov8m.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "dataset"))
    p.add_argument("--train-rel", type=str, default="train/images")
    p.add_argument(
        "--model-cfg",
        type=str,
        default=str(PROJECT_ROOT / "GraduationScripts" / "models" / "yolov8m_ghostnet_backbone.yaml"),
    )
    p.add_argument("--weights", type=str, default=_default_weights())
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--run-name",
        type=str,
        default="yolov8m_ghostnet_backbone_ham_univ30_tightScale03",
    )
    p.add_argument("--no-epoch-tag", action="store_true")
    p.add_argument("--mosaic", type=float, default=None, help="默认 1.0")
    p.add_argument("--mixup", type=float, default=None, help="默认 0.2")
    p.add_argument("--close-mosaic", type=int, default=None, help="默认 10；0 表示不传")
    p.add_argument("--scale", type=float, default=None, help="默认 0.3；负数表示不传 scale")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    tag = a.run_name if a.no_epoch_tag else f"{a.run_name}_e{a.epochs}"
    print(f"[run-name] {tag}")

    register_lightweight_blocks_for_ultralytics()

    mosaic = 1.0 if a.mosaic is None else float(a.mosaic)
    mixup = 0.2 if a.mixup is None else float(a.mixup)
    cm = 10 if a.close_mosaic is None else int(a.close_mosaic)
    close_kw = None if cm <= 0 else cm
    if a.scale is not None and float(a.scale) < 0:
        scale_kw = None
    else:
        scale_kw = 0.3 if a.scale is None else float(a.scale)

    run_training(
        run_name=tag,
        dataset_root=Path(a.data_root),
        train_rel=str(a.train_rel),
        model_cfg=Path(a.model_cfg),
        weights=str(a.weights),
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        workers=a.workers,
        device=a.device or None,
        seed=a.seed,
        amp=not bool(a.no_amp),
        mosaic=mosaic,
        close_mosaic=close_kw,
        mixup=mixup,
        scale=scale_kw,
    )


if __name__ == "__main__":
    main()
