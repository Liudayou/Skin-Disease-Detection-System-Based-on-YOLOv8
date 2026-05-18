"""
批量 test 集评估工具。

读取 99_index/index.csv 中的实验列表，逐一在 test split 上评估，
输出 *_test_metrics.json 和 *_test_metrics.csv。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ExperimentRow:
    """index.csv 中一行实验记录。"""
    dest_dir: str
    run_name: str


def _read_index_rows(index_csv: Path) -> List[ExperimentRow]:
    """读取 index.csv 获取实验目录和 run_name 的映射。"""
    rows: List[ExperimentRow] = []
    with index_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            dest_dir = (r.get("dest_dir") or "").strip()
            run_name = (r.get("run_name") or "").strip()
            if not dest_dir or not run_name:
                continue
            rows.append(ExperimentRow(dest_dir=dest_dir, run_name=run_name))
    return rows


def _guess_dataset_root(project_root: Path) -> Path:
    """自动查找数据集根目录：环境变量 > dataset/ > autodl-tmp 兼容路径。"""
    env = os.environ.get("YOLO_DATASET_ROOT")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
    candidate = project_root / "dataset"
    if candidate.exists():
        return candidate
    autodl_tmp = project_root.parent
    for p in autodl_tmp.glob("*/test/images"):
        return p.parent.parent
    raise FileNotFoundError("找不到数据集根目录，设置 YOLO_DATASET_ROOT 环境变量指向你的 dataset 目录。")


def _weights_path(project_root: Path, run_name: str) -> Optional[Path]:
    """从 runs/detect/<run_name>/weights/ 获取 best.pt 或 last.pt。"""
    base = project_root / "runs" / "detect" / run_name / "weights"
    best = base / "best.pt"
    last = base / "last.pt"
    if best.exists():
        return best
    if last.exists():
        return last
    return None


def _read_names_and_nc(data_runtime_yaml: Path) -> Tuple[Optional[int], Optional[List[str]]]:
    """从 data_runtime.yaml 解析类别数和名称列表。"""
    if not data_runtime_yaml.exists():
        return None, None
    nc: Optional[int] = None
    names: List[str] = []
    in_names = False
    for line in data_runtime_yaml.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if s.startswith("nc:"):
            try:
                nc = int(s.split(":", 1)[1].strip())
            except Exception:
                nc = None
        if s.startswith("names:"):
            in_names = True
            continue
        if in_names:
            if s.startswith("- "):
                names.append(s[2:].strip())
            elif s and not s.startswith("-"):
                in_names = False
    return nc, names or None


def _write_test_data_yaml(out_path: Path, dataset_root: Path, nc: Optional[int], names: Optional[List[str]]) -> None:
    """生成仅含 test 路径的临时 data.yaml。"""
    lines: List[str] = []
    lines.append(f"path: {dataset_root.as_posix()}")
    lines.append("train: train/images")
    lines.append("val: valid/images")
    lines.append("test: test/images")
    if nc is not None:
        lines.append(f"nc: {nc}")
    if names:
        lines.append("names:")
        for n in names:
            lines.append(f"- {n}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metrics_to_dict(val_result: Any) -> Dict[str, Any]:
    """从 Ultralytics val 结果提取 mAP/speed 等指标。"""
    out: Dict[str, Any] = {}
    box = getattr(val_result, "box", None)
    if box is not None:
        for k in ("map", "map50", "map75"):
            v = getattr(box, k, None)
            if v is not None:
                out[k] = float(v)
    speed = getattr(val_result, "speed", None)
    if speed is not None and isinstance(speed, dict):
        out["speed"] = speed
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="批量在 test split 上评估 YOLO 实验，输出 *_test_metrics.json/csv。")
    ap.add_argument("--force", action="store_true", help="强制重新评估（即使已有结果文件）")
    ap.add_argument("--only-prefix", action="append", default=[], help="只评估 dest_dir 以此前缀开头的实验")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    index_csv = project_root / "99_index" / "index.csv"
    if not index_csv.exists():
        raise FileNotFoundError(f"缺少 {index_csv}")

    dataset_root = _guess_dataset_root(project_root)
    rows = _read_index_rows(index_csv)
    only_prefixes = [str(x).strip().strip("/") for x in (args.only_prefix or []) if str(x).strip()]
    if only_prefixes:
        rows = [r for r in rows if any(r.dest_dir.startswith(p) for p in only_prefixes)]

    from ultralytics import YOLO  # type: ignore

    summary: List[Dict[str, Any]] = []
    for r in rows:
        out_dir = project_root / "results" / r.dest_dir
        existing_json = out_dir / f"{r.run_name}_test_metrics.json"
        existing_csv = out_dir / f"{r.run_name}_test_metrics.csv"
        # 已有结果且未 --force 时跳过
        if (not args.force) and existing_json.exists() and existing_csv.exists():
            continue

        weights = _weights_path(project_root, r.run_name) or _weights_path(project_root, r.dest_dir)
        if weights is None:
            summary.append({"dest_dir": r.dest_dir, "run_name": r.run_name, "status": "skipped", "reason": "weights_not_found"})
            continue

        runtime_yaml = out_dir / "data_runtime.yaml"
        nc, names = _read_names_and_nc(runtime_yaml)
        test_data_yaml = out_dir / "data_runtime_test.yaml"
        _write_test_data_yaml(test_data_yaml, dataset_root, nc, names)

        model = YOLO(str(weights))
        val = model.val(data=str(test_data_yaml), split="test", verbose=False)
        metrics = _metrics_to_dict(val)
        metrics_row = {"dest_dir": r.dest_dir, "run_name": r.run_name, "weights": str(weights), "data_yaml": str(test_data_yaml), "status": "ok", **metrics}
        summary.append(metrics_row)

        out_dir.mkdir(parents=True, exist_ok=True)
        existing_json.write_text(json.dumps(metrics_row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with existing_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["run_name", "map50", "map", "map75", "weights", "data_yaml"])
            w.writeheader()
            w.writerow({"run_name": r.run_name, "map50": metrics.get("map50", ""), "map": metrics.get("map", ""), "map75": metrics.get("map75", ""), "weights": str(weights), "data_yaml": str(test_data_yaml)})

    # 写入汇总文件
    out_summary = project_root / "99_index" / "index_test_summary.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
