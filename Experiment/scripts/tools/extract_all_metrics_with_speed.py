import os
import csv
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

import sys
sys.path.insert(0, str(Path("Experiment/scripts").resolve()))

try:
    from modules.cbam_attention import register_cbam_for_ultralytics
    register_cbam_for_ultralytics()
except Exception: pass
try:
    from modules.eca_attention import register_eca_for_ultralytics
    register_eca_for_ultralytics()
except Exception: pass
try:
    from modules.ema_attention import register_ema_for_ultralytics
    register_ema_for_ultralytics()
except Exception: pass

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

out_csv = Path("Experiment/results/attention/test_results/ablations/test_results_all_with_speed.csv")
res_list = []

for run in ablations:
    weights_path = runs_dir / run / "weights" / "best.pt"
    if not weights_path.exists():
        continue
    
    print(f"Extracting full info for {run}...")
    model = YOLO(str(weights_path))
    
    # Get parameters limit
    params = sum(p.numel() for p in model.model.parameters())
    params_m = round(params / 1e6, 2)
    
    # Get GFLOPs
    gflops = round(get_flops(model.model, imgsz=640), 2)
    
    # Re-run val to get stable latency metrics (speed)
    metrics = model.val(data=str(data_yaml), split="test", verbose=False, save=False, plots=False)
    
    latency_ms = round(metrics.speed['inference'], 2)
    
    row = {
        "Model": run,
        "Params (M)": params_m,
        "GFLOPs": gflops,
        "Latency_Inference (ms)": latency_ms
    }
    
    for k, v in metrics.results_dict.items():
        row[k] = round(v, 4)
    res_list.append(row)

if res_list:
    headers = ["Model", "Params (M)", "GFLOPs", "Latency_Inference (ms)"] + list(metrics.results_dict.keys())
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(res_list)
    print(f"Saved all metrics including speed and params to {out_csv}")
