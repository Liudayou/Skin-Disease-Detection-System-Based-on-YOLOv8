#!/usr/bin/env python3
"""为 stitch 方案的两个模型生成评估图表，输出到 evaluation_plots/stitch/。"""
import sys
import types
from pathlib import Path

_SCRIPTS_ROOT = Path(__file__).resolve().parent / "scripts"
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

# 注册自定义模块
try:
    from modules.cbam_attention import register_cbam_for_ultralytics
    register_cbam_for_ultralytics()
    sys.modules["cbam_attention"] = sys.modules["modules.cbam_attention"]
    # 修复训练时拼写错误：部分旧权重文件中模块名为 cabm_attention / CABM 类
    sys.modules["cabm_attention"] = sys.modules["modules.cbam_attention"]
    import modules.cbam_attention as _cbam_mod
    _cbam_mod.CABM = _cbam_mod.CBAM
except Exception as e:
    print(f"[warn] CBAM: {e}")

try:
    from modules.eca_attention import register_eca_for_ultralytics
    register_eca_for_ultralytics()
    sys.modules["eca_attention"] = sys.modules["modules.eca_attention"]
except Exception as e:
    print(f"[warn] ECA: {e}")

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent

models = [
    {
        "tag": "yolov8m_w0664_cbam_ham_univ30_lowMixup015_e100",
        "weights": "Experiment/runs/detect/stitch/train_results/lowMixup015/yolov8m_w0664_cbam_ham_univ30_lowMixup015_e100/weights/best.pt",
    },
    {
        "tag": "yolov8m_w0664_eca_ham_univ30_lowMixup015_e100",
        "weights": "Experiment/runs/detect/stitch/train_results/lowMixup015/yolov8m_w0664_eca_ham_univ30_lowMixup015_e100/weights/best.pt",
    },
]

# 使用本地数据集配置
LOCAL_DATA_YAML = PROJECT_ROOT / "dataset" / "data.yaml"
assert LOCAL_DATA_YAML.exists(), f"Dataset not found: {LOCAL_DATA_YAML}"
print(f"Using dataset: {LOCAL_DATA_YAML}")

for m in models:
    w = Path(m["weights"])
    if not w.exists():
        print(f"[skip] weights not found: {w}")
        continue

    print(f"\n{'='*60}")
    print(f"[{m['tag']}] Generating evaluation plots...")

    model = YOLO(str(w))
    # 输出到 evaluation_plots/stitch/<tag>/
    out_dir = PROJECT_ROOT / "evaluation_plots" / "stitch" / m["tag"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # val 会在 <project>/<name>/ 下生成图片，然后我们复制到目标目录
    model.val(
        data=str(LOCAL_DATA_YAML),
        split="test",
        imgsz=640,
        batch=32,
        verbose=False,
        plots=True,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name=f"_tmp_stitch_eval_{m['tag']}",
        exist_ok=True,
    )

    # 把生成的图片复制到目标目录
    tmp_dir = PROJECT_ROOT / "runs" / "detect" / f"_tmp_stitch_eval_{m['tag']}"
    if tmp_dir.exists():
        import shutil
        for f in tmp_dir.iterdir():
            if f.suffix in (".png", ".jpg"):
                dest = out_dir / f.name
                shutil.copy2(f, dest)
                print(f"  -> {dest}")
        shutil.rmtree(tmp_dir)
        print(f"  cleaned tmp: {tmp_dir}")
    else:
        print("  [warn] tmp dir not found")

print("\nDone!")
