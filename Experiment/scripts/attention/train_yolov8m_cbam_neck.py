#!/usr/bin/env python3
"""
注意力消融实验 — CBAM 注意力模块插入 Neck (PAN-FPN)。

默认数据增强对齐 lowMixup015：mosaic=1.0、mixup=0.15、close_mosaic=10、scale=0.5。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# 向上查找 yolov8m_train_utils.py 所在目录并加入 sys.path
_SCRIPTS_ROOT = Path(__file__).resolve().parent
for _ in range(3):
    if (_SCRIPTS_ROOT / "yolov8m_train_utils.py").is_file():
        break
    _SCRIPTS_ROOT = _SCRIPTS_ROOT.parent
else:
    raise RuntimeError("找不到 yolov8m_train_utils.py 所在的 scripts 根目录")
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

# 注册 CBAM 自定义算子到 Ultralytics，否则加载 yaml 会报错
from modules.cbam_attention import register_cbam_for_ultralytics
from ultralytics import YOLO
from yolov8m_train_utils import (
    PROJECT_ROOT,
    add_model_name_cli_arg,
    add_mosaic_mixup_cli_args,
    metrics_from_detmetrics,
    run_training,
    train_aug_kwargs_from_ns,
)


def _default_weights() -> str:
    """优先用本地 yolov8m.pt，没有则让 Ultralytics 自动下载。"""
    local = PROJECT_ROOT / "scripts" / "models" / "yolov8m.pt"
    return str(local) if local.is_file() else "yolov8m.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "dataset"),
                    help="YOLO 数据集根目录（含 train/valid/test）")
    p.add_argument("--train-rel", type=str, default="train/images",
                    help="训练集 images 相对路径")
    p.add_argument("--model-cfg", type=str,
                    default=str(PROJECT_ROOT / "scripts" / "models" / "yolov8m_cbam_neck.yaml"),
                    help="CBAM-Neck 模型结构 YAML")
    p.add_argument("--weights", type=str, default=_default_weights(),
                    help="迁移学习预训练权重")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--no-amp", action="store_true", help="禁用混合精度训练")
    p.add_argument("--device", type=str, default="", help="留空自动选择 GPU")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", type=str,
                    default="yolov8m_cbam_neck_ham_univ30_lowMixup015",
                    help="实验名前缀，自动追加 _e{epochs}")
    p.add_argument("--no-epoch-tag", action="store_true", help="不追加 _e{epochs} 后缀")
    p.add_argument("--val-batch-cap", type=int, default=32,
                    help="验证 batch 上限，0 不限制")
    # 数据增强参数（对齐 lowMixup015 方案）
    add_mosaic_mixup_cli_args(p, mosaic_default=1.0, mixup_default=0.15, close_mosaic_default=10)
    p.add_argument("--scale", type=float, default=0.5, help="几何缩放范围")
    p.add_argument("--degrees", type=float, default=None)
    p.add_argument("--translate", type=float, default=None)
    p.add_argument("--cache", type=str, default="false",
                    choices=("false", "true", "ram", "disk"),
                    help="数据缓存策略：false 省内存，disk 用磁盘缓存")
    p.add_argument("--eval-test", action="store_true",
                    help="训练后在 test 集上评估")
    p.add_argument("--eval-test-only", action="store_true",
                    help="只做 test 评估（跳过训练）")
    add_model_name_cli_arg(p)
    return p.parse_args()


def _cache_kwarg(s: str) -> bool | str:
    return {"false": False, "true": True, "ram": "ram", "disk": "disk"}[s]


def _weights_from_run(run_name: str) -> Path:
    """从实验输出目录获取 best.pt（优先）或 last.pt。"""
    base = PROJECT_ROOT / "runs" / "detect" / run_name / "weights"
    best = base / "best.pt"
    last = base / "last.pt"
    if best.is_file():
        return best
    if last.is_file():
        return last
    raise FileNotFoundError(f"权重文件不存在: {base}")


def _eval_on_test(*, run_name: str, data_yaml: Path, imgsz: int, batch: int, device: str | None) -> Path:
    """在 test 集上评估并将指标写入 test_metrics.csv。"""
    weights = _weights_from_run(run_name)
    model = YOLO(str(weights))
    val_res = model.val(
        data=str(data_yaml), split="test",
        imgsz=int(imgsz), batch=int(batch),
        device=(device or None), verbose=False, plots=False,
    )
    fin = metrics_from_detmetrics(val_res)

    out_dir = PROJECT_ROOT / "results" / "attention" / "test_results" / "lowMixup015" / Path(run_name).name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "test_metrics.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "mAP50", "mAP50_95", "precision_mean", "recall_mean", "weights", "data_yaml"])
        w.writeheader()
        w.writerow({
            "split": "test",
            "mAP50": fin.get("mAP50"),
            "mAP50_95": fin.get("mAP50_95"),
            "precision_mean": fin.get("precision_mean"),
            "recall_mean": fin.get("recall_mean"),
            "weights": str(weights),
            "data_yaml": str(data_yaml),
        })
    print(f"[eval-test] 结果写入: {out_csv}")
    return out_csv


def main() -> None:
    a = parse_args()
    # 拼接完整实验名（含增强方案和 epoch 数）
    tag = a.run_name if a.no_epoch_tag else f"{a.run_name}_e{a.epochs}"
    full_run_name = f"attention/train_results/lowMixup015/{tag}"
    print(f"[run-name] {full_run_name}")

    # 必须在 YOLO 加载前注册 CBAM 自定义算子
    register_cbam_for_ultralytics()

    results_dir = PROJECT_ROOT / "results" / full_run_name
    data_yaml = results_dir / "data_runtime.yaml"

    # --eval-test-only 模式：跳过训练，直接评估
    if a.eval_test_only:
        if not data_yaml.is_file():
            raise FileNotFoundError(f"data_yaml 不存在（可能还没训练过）: {data_yaml}")
        _eval_on_test(run_name=full_run_name, data_yaml=data_yaml, imgsz=a.imgsz, batch=a.batch, device=(a.device or None))
        return

    # 正常训练流程
    run_training(
        run_name=full_run_name,
        dataset_root=Path(a.data_root),
        train_rel=str(a.train_rel),
        model_cfg=Path(a.model_cfg),  # 自定义 YAML → Ultralytics 按 yaml 构建模型
        weights=str(a.weights),
        model_name=(a.model_name.strip() or None),
        epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, workers=a.workers,
        device=a.device or None, seed=a.seed,
        amp=not bool(a.no_amp),
        val_batch_cap=None if int(a.val_batch_cap) <= 0 else int(a.val_batch_cap),
        scale=float(a.scale) if a.scale is not None else None,
        degrees=float(a.degrees) if a.degrees is not None else None,
        translate=float(a.translate) if a.translate is not None else None,
        cache=_cache_kwarg(a.cache),
        **train_aug_kwargs_from_ns(a),
    )

    # 训练后可选 test 评估
    if a.eval_test:
        if not data_yaml.is_file():
            raise FileNotFoundError(f"data_yaml 不存在: {data_yaml}")
        _eval_on_test(run_name=full_run_name, data_yaml=data_yaml, imgsz=a.imgsz, batch=a.batch, device=(a.device or None))


if __name__ == "__main__":
    main()
