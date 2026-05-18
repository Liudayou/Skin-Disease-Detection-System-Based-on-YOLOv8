import pandas as pd
import os

input_csv = "Experiment/results/attention/test_results/ablations/test_results_merged.csv"
output_dir = "Experiment/results/attention/test_results"

df = pd.read_csv(input_csv)

# 按照注意力机制分离：CBAM, ECA, EMA
for att in ['CBAM', 'ECA', 'EMA']:
    att_df = df[df['Model'].str.contains(att)]
    out_path = os.path.join(output_dir, f"{att}_ablation_test_results.csv")
    att_df.to_csv(out_path, index=False)
    print(f"Saved {att} results to {out_path}")

