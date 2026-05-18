import csv
import pandas as pd

# Read the comprehensive file
in_csv = "Experiment/results/attention/test_results/ablations/test_results_all_with_speed.csv"
out_csv = "Experiment/results/attention/test_results/ablations/test_results_merged.csv"

df = pd.read_csv(in_csv)

# Rename columns for better readability and to match previous names
rename_map = {
    'metrics/precision(B)': 'Precision',
    'metrics/recall(B)': 'Recall',
    'metrics/mAP50(B)': 'mAP@50',
    'metrics/mAP50-95(B)': 'mAP@50-95',
    'fitness': 'Fitness'
}

df.rename(columns=rename_map, inplace=True)

# Reorder columns slightly for better flow
desired_order = [
    'Model',
    'Precision',
    'Recall',
    'mAP@50',
    'mAP@50-95',
    'Fitness',
    'Params (M)',
    'GFLOPs',
    'Latency_Inference (ms)'
]

df = df[desired_order]

df.to_csv(out_csv, index=False)
print(f"Merged CSV created at: {out_csv}")
