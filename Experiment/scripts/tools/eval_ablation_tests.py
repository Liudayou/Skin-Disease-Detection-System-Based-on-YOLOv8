import os
import sys
import glob
from ultralytics import YOLO
from pathlib import Path

# add modules to path
sys.path.insert(0, str(Path("Experiment/scripts").resolve()))

try:
    from modules.cbam_attention import register_cbam_for_ultralytics
    register_cbam_for_ultralytics()
except Exception as e: pass
try:
    from modules.eca_attention import register_eca_for_ultralytics
    register_eca_for_ultralytics()
except Exception as e: pass
try:
    from modules.ema_attention import register_ema_for_ultralytics
    register_ema_for_ultralytics()
except Exception as e: pass

runs_dir = Path("Experiment/runs/detect")
data_yaml = Path("Experiment/dataset/data.yaml")

ablations = [
    "attention_ablation__CBAM_P3_e100",
    "attention_ablation__CBAM_P4_e100",
    "attention_ablation__CBAM_P5_e100",
    "attention_ablation__ECA_P3_e100",
    "attention_ablation__ECA_P4_e100",
    "attention_ablation__ECA_P5_e100",
    "attention_ablation__EMA_P3_e100",
    "attention_ablation__EMA_P4_e100",
    "attention_ablation__EMA_P5_e100",
]

print(">>> Starting evaluation on TEST set for all 9 ablations...\n")

for run in ablations:
    weights_path = runs_dir / run / "weights" / "best.pt"
    if not weights_path.exists():
        print(f"[SKIP] Run {run} has no best.pt yet. Is training complete?")
        continue
    
    print(f"\n==================================================")
    print(f"Evaluating: {run}")
    print(f"==================================================")
    
    try:
        model = YOLO(str(weights_path))
        metrics = model.val(data=str(data_yaml), split="test", verbose=False, name=f"{run}_test_eval", project="Experiment/results/attention/test_results")
        
        map50 = getattr(metrics.box, 'map50', -1)
        map50_95 = getattr(metrics.box, 'map', -1)
        print(f"[SUCCESS] {run} | mAP@50: {map50:.4f} | mAP@50-95: {map50_95:.4f}")
    except Exception as e:
        print(f"[ERROR] failed to evaluate {run}: {e}")

print("\n>>> All evaluations complete! Results saved in Experiment/results/attention/test_results/")
