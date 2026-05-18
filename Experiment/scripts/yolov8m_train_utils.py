"""
消融实验通用训练工具模块。

所有训练脚本（model_scaling / attention / lightweight / stitch）的公共依赖：
- 数据增强参数 CLI 注册与解析
- 训练流程封装（run_training）
- 每轮/最终指标 CSV 导出
- 模型参数量 / GFLOPs / 延迟 基准测试

所有产出物写入 Experiment/results/ 下，不污染 GraduationDesign/ 目录。
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import get_flops

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_SUBDIR = "results"
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Ultralytics 默认数据增强参数
DEFAULT_MOSAIC_MIXUP_MOSAIC = 1.0
DEFAULT_MOSAIC_MIXUP_CLOSE_MOSAIC = 0
DEFAULT_MOSAIC_MIXUP_MIXUP = 0.2

# 无头环境用 Agg 后端，避免 matplotlib GUI 报错
os.environ.setdefault("MPLBACKEND", "Agg")


def add_mosaic_mixup_cli_args(
    p: argparse.ArgumentParser,
    *,
    mosaic_default: float = DEFAULT_MOSAIC_MIXUP_MOSAIC,
    close_mosaic_default: int = DEFAULT_MOSAIC_MIXUP_CLOSE_MOSAIC,
    mixup_default: float = DEFAULT_MOSAIC_MIXUP_MIXUP,
    copy_paste_default: float = 0.0,
) -> None:
    """注册 mosaic / mixup 等数据增强 CLI 参数。"""
    p.add_argument("--mosaic", type=float, default=float(mosaic_default))
    p.add_argument("--close-mosaic", type=int, default=int(close_mosaic_default))
    p.add_argument("--mixup", type=float, default=float(mixup_default))
    p.add_argument("--copy-paste", type=float, default=float(copy_paste_default))


def add_model_name_cli_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model-name", type=str, default="")


def train_aug_kwargs_from_ns(ns: argparse.Namespace) -> dict[str, float | int | None]:
    """从 argparse 命名空间提取数据增强参数，转为 Ultralytics train() 可接受的字典。
    close_mosaic / mixup 为 0 时设为 None（Ultralytics 要求 None 表示不启用）。"""
    mosaic = float(ns.mosaic)
    close_mosaic = int(ns.close_mosaic)
    mixup = float(ns.mixup)
    copy_paste = float(ns.copy_paste)
    return {
        "mosaic": mosaic,
        "close_mosaic": int(close_mosaic) if int(close_mosaic) > 0 else None,
        "mixup": float(mixup) if float(mixup) > 0 else None,
        "copy_paste": float(copy_paste) if float(copy_paste) > 0 else None,
    }


# 增强参数 CSV 字段（记录实际生效的增强配置）
CSV_AUG_FIELDNAMES: list[str] = [
    "data_aug_runner_explicit",
    "data_aug_effective_summary",
    "effective_mosaic",
    "effective_mixup",
    "effective_close_mosaic",
    "effective_copy_paste",
    "effective_cutmix",
    "effective_degrees",
    "effective_translate",
    "effective_scale",
    "effective_shear",
    "effective_fliplr",
    "effective_flipud",
    "effective_hsv_h",
    "effective_hsv_s",
    "effective_hsv_v",
    "effective_erasing",
    "effective_auto_augment",
]


def _avoid_overwrite(path: Path) -> Path:
    """文件已存在时自动加 _v2、_v3 后缀，防止覆盖。"""
    if not path.exists():
        return path
    for i in range(2, 10_000):
        cand = path.with_name(f"{path.stem}_v{i}{path.suffix}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"cannot avoid overwrite: {path}")


def _file_size_bytes(p: Path) -> int:
    try:
        return int(p.stat().st_size)
    except FileNotFoundError:
        return 0


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _list_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted([p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in _IMG_EXTS])


def maybe_autosplit_from_test(dataset_root: Path, *, seed: int = 0) -> None:
    """如果 train/valid 目录已存在则跳过自动划分。"""
    dataset_root = dataset_root.resolve()
    if (dataset_root / "train" / "images").exists() and (dataset_root / "valid" / "images").exists():
        return


def write_ultralytics_data_yaml(
    dataset_root: Path,
    out_yaml: Path,
    *,
    train_rel: str = "train/images",
    val_rel: str = "valid/images",
    test_rel: str = "test/images",
) -> Path:
    """生成 Ultralytics 格式的 data.yaml，包含 7 类名称。"""
    dataset_root = dataset_root.resolve()
    data = {
        "path": str(dataset_root),
        "train": str(train_rel),
        "val": str(val_rel),
        "test": str(test_rel),
        "nc": 7,
        "names": CLASS_NAMES,
    }
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out_yaml


def count_params(model: Any) -> int:
    """模型总参数量。"""
    return int(sum(p.numel() for p in model.parameters()))


def gflops_at_imgsz(model: Any, imgsz: int = 640) -> float:
    return float(get_flops(model, imgsz=int(imgsz)))


def _scalar_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x.item()) if hasattr(x, "item") else float(x)
    except (TypeError, ValueError):
        return None


# 每轮训练指标 CSV 字段
EPOCH_METRICS_CSV_FIELDS: list[str] = [
    "model_name",
    "epoch",
    "mAP50",
    "mAP50_95",
    "precision_mean",
    "recall_mean",
    "f1_mean",
    "params",
    "GFLOPs",
    "data_aug_runner_explicit",
]


def metrics_from_detmetrics(metrics: Any) -> dict[str, float | None]:
    """从 Ultralytics DetMetrics 对象提取 P / R / mAP50 / mAP50-95 / F1。"""
    box = metrics.box
    p = float(box.mp) if hasattr(box, "mp") else None
    r = float(box.mr) if hasattr(box, "mr") else None
    map50 = float(box.map50) if hasattr(box, "map50") else None
    map5095 = float(box.map) if hasattr(box, "map") else None
    if p is not None and r is not None and (p + r) > 0:
        f1 = float(2 * p * r / (p + r + 1e-12))
    else:
        f1 = None
    return {
        "precision_mean": p,
        "recall_mean": r,
        "f1_mean": f1,
        "mAP50": map50,
        "mAP50_95": map5095,
        "accuracy_mAP50": map50,
    }


def _latency_fps_from_images(
    model: YOLO,
    images: list[Path],
    *,
    imgsz: int,
    batch: int,
    device: str | None,
    warmup_iters: int = 2,
    timed_iters: int = 5,
) -> tuple[float | None, float | None]:
    """用实际图片测量推理延迟和 FPS。预热 2 轮后计时 5 轮，CUDA 下自动同步。"""
    if not images:
        return None, None
    sample = [str(p) for p in images[: max(1, int(batch))]]

    def _sync() -> None:
        if torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.synchronize()

    for _ in range(max(0, int(warmup_iters))):
        _ = model.predict(source=sample, imgsz=int(imgsz), batch=int(batch), device=device, verbose=False, save=False)
    _sync()

    t0 = time.perf_counter()
    for _ in range(max(1, int(timed_iters))):
        _ = model.predict(source=sample, imgsz=int(imgsz), batch=int(batch), device=device, verbose=False, save=False)
    _sync()
    t1 = time.perf_counter()

    n_imgs = len(sample) * max(1, int(timed_iters))
    dt = max(1e-12, float(t1 - t0))
    fps = float(n_imgs / dt)
    latency_ms = float(1000.0 / fps)
    return latency_ms, fps


def apply_ultralytics_val_batch_cap(max_val_batch: int | None = 32) -> None:
    """兼容旧脚本的空操作，保留接口。"""
    _ = max_val_batch


def run_training(
    *,
    run_name: str,
    dataset_root: Path,
    weights: Path | str,
    model_name: str | None = None,
    model_cfg: Path | None = None,
    epochs: int = 50,
    patience: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    workers: int = 8,
    device: str | None = None,
    seed: int = 0,
    train_rel: str = "train/images",
    val_rel: str = "valid/images",
    test_rel: str = "test/images",
    mosaic: float | None = None,
    close_mosaic: int | None = None,
    mixup: float | None = None,
    cutmix: float | None = None,
    copy_paste: float | None = None,
    scale: float | None = None,
    degrees: float | None = None,
    translate: float | None = None,
    shear: float | None = None,
    perspective: float | None = None,
    flipud: float | None = None,
    fliplr: float | None = None,
    hsv_h: float | None = None,
    hsv_s: float | None = None,
    hsv_v: float | None = None,
    erasing: float | None = None,
    auto_augment: str | None = None,
    project: str = "runs/detect",
    plots: bool = False,
    amp: bool = True,
    val_batch_cap: int | None = 32,
    cache: bool | str = False,
    estimate_gflops: bool = False,
) -> None:
    """统一训练入口：训练 → best 权重 re-val → 导出 epoch + final 指标 CSV。

    weights 支持两种形式：
    - 本地 .pt 文件路径（必须存在）
    - Ultralytics 模型标识符（如 "yolov8m.pt"，自动下载）

    model_cfg 为自定义 YAML 时，weights 作为 pretrained 权重加载。
    """
    _ = (plots, seed, val_batch_cap, estimate_gflops)
    dataset_root = dataset_root.resolve()
    weights_path: Path | None = None
    weights_ref: str
    if isinstance(weights, Path):
        weights_path = weights.resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(f"weights not found: {weights_path}")
        weights_ref = str(weights_path)
    else:
        weights_ref = str(weights).strip()
        if not weights_ref:
            raise ValueError("weights is empty")
        p = Path(weights_ref)
        if p.is_file():
            weights_path = p.resolve()
            weights_ref = str(weights_path)

    maybe_autosplit_from_test(dataset_root, seed=0)

    results_dir = PROJECT_ROOT / RESULTS_SUBDIR / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_tag = str(run_name).replace("\\", "_").replace("/", "__")
    if weights_path is not None:
        default_mname = weights_path.stem
    else:
        default_mname = Path(weights_ref).stem if weights_ref else "yolov8m"
    mname = (model_name or "").strip() or default_mname

    data_yaml = write_ultralytics_data_yaml(
        dataset_root,
        results_dir / "data_runtime.yaml",
        train_rel=train_rel,
        val_rel=val_rel,
        test_rel=test_rel,
    )

    epoch_csv = _avoid_overwrite(results_dir / f"{mname}__{csv_tag}_epoch_metrics.csv")
    final_csv = _avoid_overwrite(results_dir / f"{mname}__{csv_tag}_final_metrics.csv")

    # 拼接增强参数摘要字符串，写入 CSV 便于溯源
    runner_explicit = "runner_kw explicit: " + "; ".join(
        [x for x in [
            f"mosaic={mosaic}" if mosaic is not None else "",
            f"close_mosaic={close_mosaic}" if close_mosaic is not None else "",
            f"mixup={mixup}" if mixup is not None else "",
            f"cutmix={cutmix}" if cutmix is not None else "",
            f"copy_paste={copy_paste}" if copy_paste is not None else "",
            f"scale={scale}" if scale is not None else "",
            f"degrees={degrees}" if degrees is not None else "",
            f"translate={translate}" if translate is not None else "",
            f"shear={shear}" if shear is not None else "",
            f"perspective={perspective}" if perspective is not None else "",
            f"flipud={flipud}" if flipud is not None else "",
            f"fliplr={fliplr}" if fliplr is not None else "",
            f"hsv_h={hsv_h}" if hsv_h is not None else "",
            f"hsv_s={hsv_s}" if hsv_s is not None else "",
            f"hsv_v={hsv_v}" if hsv_v is not None else "",
            f"erasing={erasing}" if erasing is not None else "",
            f"auto_augment={auto_augment}" if auto_augment is not None else "",
        ] if x]
    )

    model = YOLO(str(model_cfg)) if model_cfg is not None else YOLO(str(weights_ref))
    params = count_params(model.model)
    gfl = float(get_flops(model.model, imgsz=int(imgsz)))

    with epoch_csv.open("w", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=EPOCH_METRICS_CSV_FIELDS).writeheader()

    # 每轮结束回调：从 trainer.metrics 提取指标写入 CSV
    def _on_fit_epoch_end(trainer: Any) -> None:
        if RANK not in {-1, 0}:
            return
        md = getattr(trainer, "metrics", None)
        if not isinstance(md, dict) or "metrics/mAP50(B)" not in md:
            return
        p = _scalar_float(md.get("metrics/precision(B)"))
        r = _scalar_float(md.get("metrics/recall(B)"))
        map50 = _scalar_float(md.get("metrics/mAP50(B)"))
        map5095 = _scalar_float(md.get("metrics/mAP50-95(B)"))
        if p is not None and r is not None and (p + r) > 0:
            f1 = float(2 * p * r / (p + r + 1e-12))
        else:
            f1 = None
        row = {
            "model_name": mname,
            "epoch": int(getattr(trainer, "epoch", -1)) + 1,
            "mAP50": map50,
            "mAP50_95": map5095,
            "precision_mean": p,
            "recall_mean": r,
            "f1_mean": f1,
            "params": int(params),
            "GFLOPs": float(gfl),
            "data_aug_runner_explicit": runner_explicit,
        }
        with epoch_csv.open("a", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=EPOCH_METRICS_CSV_FIELDS).writerow(row)

    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)

    # 组装训练参数字典
    train_kw: dict[str, Any] = dict(
        data=str(data_yaml),
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch),
        workers=int(workers),
        cache=cache,
        seed=int(seed),
        patience=int(patience),
        project=str(PROJECT_ROOT / project),
        name=run_name,
        exist_ok=True,
        verbose=True,
        plots=False,
        val=True,
        amp=bool(amp),
    )
    if device is not None:
        train_kw["device"] = device
    # 自定义 YAML 模式：weights 作为 pretrained 权重
    if model_cfg is not None:
        train_kw["pretrained"] = str(weights_ref)
    if mosaic is not None:
        train_kw["mosaic"] = float(mosaic)
    if close_mosaic is not None:
        train_kw["close_mosaic"] = int(close_mosaic)
    if mixup is not None:
        train_kw["mixup"] = float(mixup)
    if cutmix is not None:
        train_kw["cutmix"] = float(cutmix)
    if copy_paste is not None:
        train_kw["copy_paste"] = float(copy_paste)
    if scale is not None:
        train_kw["scale"] = float(scale)
    if degrees is not None:
        train_kw["degrees"] = float(degrees)
    if translate is not None:
        train_kw["translate"] = float(translate)
    if shear is not None:
        train_kw["shear"] = float(shear)
    if perspective is not None:
        train_kw["perspective"] = float(perspective)
    if flipud is not None:
        train_kw["flipud"] = float(flipud)
    if fliplr is not None:
        train_kw["fliplr"] = float(fliplr)
    if hsv_h is not None:
        train_kw["hsv_h"] = float(hsv_h)
    if hsv_s is not None:
        train_kw["hsv_s"] = float(hsv_s)
    if hsv_v is not None:
        train_kw["hsv_v"] = float(hsv_v)
    if erasing is not None:
        train_kw["erasing"] = float(erasing)
    if auto_augment is not None:
        train_kw["auto_augment"] = str(auto_augment)

    model.train(**train_kw)

    # 用 best 权重重新 val，确保最终指标基于最优模型
    best = PROJECT_ROOT / project / run_name / "weights" / "best.pt"
    if not best.is_file():
        best = PROJECT_ROOT / project / run_name / "weights" / "last.pt"
    m2 = YOLO(str(best))
    val_res = m2.val(data=str(data_yaml), imgsz=int(imgsz), batch=int(batch), split="val", plots=False, verbose=True)
    fin = metrics_from_detmetrics(val_res)

    model_size_bytes = _file_size_bytes(best)
    model_size_mb = model_size_bytes / (1024 * 1024) if model_size_bytes else None

    # 测量推理延迟
    val_images = _list_images(dataset_root / val_rel)
    latency_ms, fps = _latency_fps_from_images(m2, val_images, imgsz=int(imgsz), batch=int(batch), device=device)

    # 写入最终指标 CSV
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
                "data_aug_runner_explicit",
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
                "epochs": int(epochs),
                "imgsz": int(imgsz),
                "batch": int(batch),
                "accuracy_mAP50": fin.get("accuracy_mAP50"),
                "precision_mean": fin.get("precision_mean"),
                "recall_mean": fin.get("recall_mean"),
                "f1_mean": fin.get("f1_mean"),
                "mAP50": fin.get("mAP50"),
                "mAP50_95": fin.get("mAP50_95"),
                "params": int(params),
                "GFLOPs": float(gfl),
                "model_size_bytes": int(model_size_bytes),
                "model_size_mb": model_size_mb,
                "latency_ms_per_img": latency_ms,
                "fps": fps,
                "data_aug_runner_explicit": runner_explicit,
            }
        )

    print(f"[{run_name}] final_metrics: {final_csv}")
