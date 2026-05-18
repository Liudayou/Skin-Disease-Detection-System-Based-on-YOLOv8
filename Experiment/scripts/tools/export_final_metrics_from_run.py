#!/usr/bin/env python3
"""从已有实验目录导出 final_metrics.csv，无需重新训练。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

from yolov8m_train_utils import (
    PROJECT_ROOT,
    RESULTS_SUBDIR,
    _avoid_overwrite,
    _file_size_bytes,
    _latency_fps_from_images,
    _list_images,
    apply_ultralytics_val_batch_cap,
    count_params,
    metrics_from_detmetrics,
    write_ultralytics_data_yaml,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--project", type=str, default="runs/detect")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--train-rel", type=str, default="train/images")
    p.add_argument("--val-rel", type=str, default="valid/images")
    p.add_argument("--test-rel", type=str, default="test/images")
    p.add_argument("--weights", type=str, default="", help="Only used for model_name default stem")
    p.add_argument("--model-name", type=str, default="")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", type=str, default="")
    args = p.parse_args()

    run_name = args.run_name.strip()
    dataset_root = Path(args.data_root).resolve()
    mname = (args.model_name or "").strip() or (Path(args.weights).stem if args.weights else "yolov8m")

    run_save_dir = (PROJECT_ROOT / args.project / run_name).resolve()
    best = run_save_dir / "weights" / "best.pt"
    if not best.is_file():
        best = run_save_dir / "weights" / "last.pt"
    if not best.is_file():
        raise FileNotFoundError(f"missing weights under: {run_save_dir / 'weights'}")

    apply_ultralytics_val_batch_cap(32)

    results_dir = PROJECT_ROOT / RESULTS_SUBDIR / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = write_ultralytics_data_yaml(
        dataset_root,
        results_dir / "data_runtime.yaml",
        train_rel=str(args.train_rel),
        val_rel=str(args.val_rel),
        test_rel=str(args.test_rel),
    )

    final_csv = _avoid_overwrite(results_dir / f"{mname}__{run_name}_final_metrics.csv")

    m2 = YOLO(str(best))
    params = count_params(m2.model)
    gfl = 0.0

    val_res = m2.val(
        data=str(data_yaml),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        split="val",
        plots=False,
        verbose=True,
        device=(args.device or None),
    )
    fin = metrics_from_detmetrics(val_res)

    model_size_bytes = _file_size_bytes(best)
    model_size_mb = model_size_bytes / (1024 * 1024) if model_size_bytes else None

    val_images = _list_images(dataset_root / str(args.val_rel))
    latency_ms, fps = _latency_fps_from_images(
        m2,
        val_images,
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=(args.device or None) or None,
    )

    with final_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "model_name",
                "weights_best",
                "dataset_root",
                "epochs",
                "imgsz",
                "batch",
                "accuracy_mAP50",
                "precision_mean",
                "recall_mean",
                "f1_mean",
                "mAP50",
                "mAP50_95",
                "params",
                "GFLOPs",
                "model_size_bytes",
                "model_size_mb",
                "latency_ms_per_img",
                "fps",
            ],
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerow(
            {
                "run_name": run_name,
                "model_name": mname,
                "weights_best": str(best),
                "dataset_root": str(dataset_root),
                "epochs": int(args.epochs),
                "imgsz": int(args.imgsz),
                "batch": int(args.batch),
                "accuracy_mAP50": fin.get("accuracy_mAP50"),
                "precision_mean": fin.get("precision_mean"),
                "recall_mean": fin.get("recall_mean"),
                "f1_mean": fin.get("f1_mean"),
                "mAP50": fin.get("mAP50"),
                "mAP50_95": fin.get("mAP50_95"),
                "params": params,
                "GFLOPs": gfl,
                "model_size_bytes": model_size_bytes,
                "model_size_mb": model_size_mb,
                "latency_ms_per_img": latency_ms,
                "fps": fps,
            }
        )

    print(f"[export] wrote: {final_csv}")


if __name__ == "__main__":
    main()
