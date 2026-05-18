import os
import glob

files = glob.glob('Experiment/scripts/attention/ablations/train_yolov8m_*.py')

template = """#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

_SCRIPTS_ROOT = Path(__file__).resolve().parent
for _ in range(3):
    if (_SCRIPTS_ROOT / "yolov8m_train_utils.py").is_file():
        break
    _SCRIPTS_ROOT = _SCRIPTS_ROOT.parent
else:
    raise RuntimeError("找不到 yolov8m_train_utils.py")
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

# Register Custom modules
from modules.eca_attention import register_eca_for_ultralytics
from modules.ema_attention import register_ema_for_ultralytics
from modules.cbam import register_cbam_for_ultralytics

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

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "dataset"))
    p.add_argument("--train-rel", type=str, default="train/images")
    p.add_argument("--model-cfg", type=str, default=str(PROJECT_ROOT / "scripts" / "models" / "generated" / "attention_ablations" / "{YAML_NAME}"))
    p.add_argument("--weights", type=str, default=_default_weights())
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", type=str, default="attention_ablation__{ATTN}_{STAGE}")
    p.add_argument("--no-epoch-tag", action="store_true")
    add_mosaic_mixup_cli_args(p, mosaic_default=1.0, mixup_default=0.15, close_mosaic_default=10)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--degrees", type=float, default=None)
    p.add_argument("--translate", type=float, default=None)
    add_model_name_cli_arg(p)
    return p.parse_args()

def main():
    a = parse_args()
    tag = a.run_name if a.no_epoch_tag else f"{{a.run_name}}_e{{a.epochs}}"
    
    register_eca_for_ultralytics()
    register_ema_for_ultralytics()
    register_cbam_for_ultralytics()

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

if __name__ == '__main__':
    main()
"""

for f in files:
    filename = os.path.basename(f)
    attn = filename.split('_')[2]
    stage = filename.split('_')[3].split('.')[0]
    yaml_name = f"yolov8m_{attn}_{stage}.yaml"
    
    with open(f, 'w') as out:
        out.write(template.replace('{YAML_NAME}', yaml_name).replace('{ATTN}', attn).replace('{STAGE}', stage))

