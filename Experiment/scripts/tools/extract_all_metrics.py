import os
import csv
from pathlib import Path
from ultralytics import YOLO

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

out_csv = Path("Experiment/results/attention/test_results/ablations/test_results_all.csv")
res_list = []

for run in ablations:
    weights_path = runs_dir / run / "weights" / "best.pt"
    if not weights_path.exists():
        continue
    
    print(f"Extracting metrics for {run}...")
    model = YOLO(str(weights_path))
    # just return metrics without writing images to disk again
    metrics = model.val(data=str(data_yaml), split="test", verbose=False, save=False, plots=False)
    
    row = {"Model": run}
    for k, v in metrics.results_dict.items():
        row[k] = round(v, 4)
    res_list.append(row)

if res_list:
    headers = ["Model"] + list(res_list[0].keys())[1:]
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(res_list)
    print(f"Saved all metrics to {out_csv}")
