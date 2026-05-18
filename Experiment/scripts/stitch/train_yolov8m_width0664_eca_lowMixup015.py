#!/usr/bin/env python3
"""Stitch方案：YOLOv8m width0664 + ECA（lowMixup015）。

模型：
- `scripts/models/yolov8m_eca_neck_width0664.yaml`（ECA 插入 Neck；scales.m width=0.664）

训练增强（默认 lowMixup015）：
- mosaic=1.0, mixup=0.15, close_mosaic=10, scale=0.5

输出：
- results/stitch/train_results/lowMixup015/<tag>/
- runs/detect/stitch/train_results/lowMixup015/<tag>/weights/{best,last}.pt

可选：
- `--eval-test` / `--eval-test-only` 写入 results/stitch/test_results/lowMixup015/<tag>/test_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_ROOT = _THIS_DIR.parent
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

from modules.eca_attention import register_eca_for_ultralytics
from ultralytics import YOLO
from yolov8m_train_utils import PROJECT_ROOT, metrics_from_detmetrics, run_training


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
        default=str(PROJECT_ROOT / "scripts" / "models" / "yolov8m_eca_neck_width0664.yaml"),
    )
    p.add_argument("--weights", type=str, default=_default_weights())
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument(
        "--run-name",
        type=str,
        default="yolov8m_w0664_eca_ham_univ30_lowMixup015",
        help="实验名前缀；默认会自动加上 _e{epochs}",
    )
    p.add_argument("--no-epoch-tag", action="store_true")
    p.add_argument("--val-batch-cap", type=int, default=32, help="0 表示不限制")
    # lowMixup015 defaults
    p.add_argument("--mosaic", type=float, default=1.0)
    p.add_argument("--mixup", type=float, default=0.15)
    p.add_argument("--close-mosaic", type=int, default=10)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--eval-test", action="store_true")
    p.add_argument("--eval-test-only", action="store_true")
    return p.parse_args()


def _weights_from_run(run_name: str) -> Path:
    base = PROJECT_ROOT / "runs" / "detect" / run_name / "weights"
    best = base / "best.pt"
    last = base / "last.pt"
    if best.is_file():
        return best
    if last.is_file():
        return last
    raise FileNotFoundError(f"missing weights under: {base}")


def _eval_on_test(*, run_name: str, data_yaml: Path, imgsz: int, batch: int, device: str | None) -> Path:
    weights = _weights_from_run(run_name)
    model = YOLO(str(weights))
    val_res = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=int(imgsz),
        batch=int(batch),
        device=(device or None),
        verbose=False,
        plots=False,
    )
    fin = metrics_from_detmetrics(val_res)

    out_dir = PROJECT_ROOT / "results" / "stitch" / "test_results" / "lowMixup015" / Path(run_name).name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "test_metrics.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["split", "mAP50", "mAP50_95", "precision_mean", "recall_mean", "weights", "data_yaml"],
        )
        w.writeheader()
        w.writerow(
            {
                "split": "test",
                "mAP50": fin.get("mAP50"),
                "mAP50_95": fin.get("mAP50_95"),
                "precision_mean": fin.get("precision_mean"),
                "recall_mean": fin.get("recall_mean"),
                "weights": str(weights),
                "data_yaml": str(data_yaml),
            }
        )
    print(f"[eval-test] wrote: {out_csv}")
    return out_csv


def main() -> None:
    a = parse_args()
    tag = a.run_name if a.no_epoch_tag else f"{a.run_name}_e{a.epochs}"
    full_run_name = f"stitch/train_results/lowMixup015/{tag}"
    print(f"[run-name] {full_run_name}")

    register_eca_for_ultralytics()

    results_dir = PROJECT_ROOT / "results" / full_run_name
    data_yaml = results_dir / "data_runtime.yaml"
    if a.eval_test_only:
        if not data_yaml.is_file():
            raise FileNotFoundError(f"missing data yaml (run not trained yet?): {data_yaml}")
        _eval_on_test(
            run_name=full_run_name,
            data_yaml=data_yaml,
            imgsz=int(a.imgsz),
            batch=int(a.batch),
            device=(a.device or None),
        )
        return

    cm = int(a.close_mosaic)
    close_kw = None if cm <= 0 else cm
    scale_kw = None if float(a.scale) < 0 else float(a.scale)

    run_training(
        run_name=full_run_name,
        dataset_root=Path(a.data_root),
        train_rel=str(a.train_rel),
        model_cfg=Path(a.model_cfg),
        weights=str(a.weights),
        epochs=int(a.epochs),
        imgsz=int(a.imgsz),
        batch=int(a.batch),
        workers=int(a.workers),
        device=(a.device or None),
        seed=int(a.seed),
        patience=int(a.patience),
        amp=not bool(a.no_amp),
        val_batch_cap=None if int(a.val_batch_cap) <= 0 else int(a.val_batch_cap),
        mosaic=float(a.mosaic),
        close_mosaic=close_kw,
        mixup=float(a.mixup),
        scale=scale_kw,
    )

    if a.eval_test:
        if not data_yaml.is_file():
            raise FileNotFoundError(f"missing data yaml after training: {data_yaml}")
        _eval_on_test(
            run_name=full_run_name,
            data_yaml=data_yaml,
            imgsz=int(a.imgsz),
            batch=int(a.batch),
            device=(a.device or None),
        )


if __name__ == "__main__":
    main()

