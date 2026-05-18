#!/usr/bin/env python3
import argparse, sys
from pathlib import Path

_SCRIPTS_ROOT = Path(__file__).resolve().parent
for _ in range(3):
    if (_SCRIPTS_ROOT / "yolov8m_train_utils.py").is_file(): break
    _SCRIPTS_ROOT = _SCRIPTS_ROOT.parent
sys.path.insert(0, str(_SCRIPTS_ROOT))

try:
    from modules.eca_attention import register_eca_for_ultralytics
    register_eca_for_ultralytics()
except: pass
try:
    from modules.ema_attention import register_ema_for_ultralytics
    register_ema_for_ultralytics()
except: pass
try:
    from modules.cbam_attention import register_cbam_for_ultralytics
    register_cbam_for_ultralytics()
except: pass

from yolov8m_train_utils import PROJECT_ROOT, add_model_name_cli_arg, add_mosaic_mixup_cli_args, run_training, train_aug_kwargs_from_ns

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(PROJECT_ROOT/"dataset"))
    p.add_argument("--train-rel", default="train/images")
    p.add_argument("--model-cfg", default=str(PROJECT_ROOT/"scripts"/"models"/"generated"/"attention_ablations"/"yolov8m_eca_p3.yaml"))
    p.add_argument("--weights", default=str(PROJECT_ROOT/"scripts"/"models"/"yolov8m.pt"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default="attention_ablation__ECA_P3_e100")
    add_mosaic_mixup_cli_args(p, mosaic_default=1.0, mixup_default=0.15, close_mosaic_default=10)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--degrees", type=float, default=None)
    p.add_argument("--translate", type=float, default=None)
    add_model_name_cli_arg(p)
    a = p.parse_args()

    run_training(
        run_name=a.run_name,
        dataset_root=Path(a.data_root),
        train_rel=a.train_rel,
        model_cfg=Path(a.model_cfg),
        weights=a.weights,
        model_name=(a.model_name.strip() or None),
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        workers=a.workers,
        device=a.device or None,
        seed=a.seed,
        amp=not a.no_amp,
        scale=a.scale,
        degrees=a.degrees,
        translate=a.translate,
        **train_aug_kwargs_from_ns(a),
    )
if __name__ == '__main__': main()
