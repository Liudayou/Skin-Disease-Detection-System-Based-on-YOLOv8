#!/usr/bin/env python3
"""训练 YOLOv8l：Ultralytics 默认数据增强（本项目唯一数据增强基线）。

不向 ``run_training`` 传入 mosaic/hsv/… 等参数时，``model.train()`` 使用
``ultralytics/cfg/default.yaml`` 中的默认增强，主要包括：

- ``mosaic=1.0``，``close_mosaic=10``（最后 10 个 epoch 关闭 mosaic）
- HSV、平移、缩放、水平翻转等几何/颜色增强
- 检测任务下 ``mixup/cutmix/copy_paste`` 默认多为 0（见当前 ultralytics 版本配置）

若需更强 mixup 等，可用命令行覆盖（见下方可选参数）。

默认 ``--patience 0`` 关闭早停，会跑满 ``--epochs``。

输出目录
- ``model_scaling/yolov8l/train_results/<run_name>_e{epochs}``
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
    raise RuntimeError("Cannot locate scripts root (missing yolov8m_train_utils.py).")
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

from yolov8m_train_utils import PROJECT_ROOT, add_model_name_cli_arg, run_training


def _default_weights() -> str:
    local = PROJECT_ROOT / "scripts" / "models" / "yolov8l.pt"
    return str(local) if local.is_file() else "yolov8l.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "dataset"))
    p.add_argument("--train-rel", type=str, default="train/images")
    p.add_argument("--weights", type=str, default=_default_weights())
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--patience",
        type=int,
        default=0,
        help="早停：0=关闭（跑满 epochs）；>0 为无提升则停的 epoch 数（Ultralytics EarlyStopping）",
    )
    p.add_argument("--run-name", type=str, default="yolov8l_default_aug_ham_univ30")
    p.add_argument("--no-epoch-tag", action="store_true")
    p.add_argument(
        "--mixup",
        type=float,
        default=None,
        help="若设置（如 0.15），则覆盖默认 mixup；不设则完全使用 Ultralytics 默认",
    )
    p.add_argument(
        "--mosaic",
        type=float,
        default=None,
        help="若设置则覆盖默认 mosaic；不设则使用 Ultralytics 默认（通常为 1.0）",
    )
    p.add_argument(
        "--close-mosaic",
        type=int,
        default=None,
        help="若设置则覆盖 close_mosaic（默认一般为最后若干 epoch 关 mosaic）；不设则用包内默认",
    )
    add_model_name_cli_arg(p)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    tag = a.run_name if a.no_epoch_tag else f"{a.run_name}_e{a.epochs}"
    full_run_name = f"model_scaling/yolov8l/train_results/{tag}"
    print(f"[run-name] 使用: {full_run_name}")

    kw: dict = dict(
        run_name=full_run_name,
        dataset_root=Path(a.data_root),
        train_rel=str(a.train_rel),
        weights=str(a.weights),
        model_name=(a.model_name.strip() or None),
        epochs=int(a.epochs),
        imgsz=int(a.imgsz),
        batch=int(a.batch),
        workers=int(a.workers),
        device=(a.device or None),
        seed=int(a.seed),
        patience=int(a.patience),
    )
    if a.mixup is not None:
        kw["mixup"] = float(a.mixup)
    if a.mosaic is not None:
        kw["mosaic"] = float(a.mosaic)
    if a.close_mosaic is not None:
        cm = int(a.close_mosaic)
        kw["close_mosaic"] = cm if cm > 0 else None

    run_training(**kw)


if __name__ == "__main__":
    main()
