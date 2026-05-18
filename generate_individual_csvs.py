import pandas as pd
import os
import glob

merged_csv = "Experiment/results/attention/test_results/ablations/test_results_merged.csv"
df = pd.read_csv(merged_csv)

# 针对每个模型，分离出它自己单独一行的数据并存为CSV，同时清理掉图片等非CSV文件
for index, row in df.iterrows():
    model_name = row['Model']
    out_dir = f"Experiment/results/attention/ablations/{model_name}/test_results"
    
    if os.path.exists(out_dir):
        # 根据您的要求，只保留CSV，删掉多余的图片(png/jpg)
        for f in glob.glob(f"{out_dir}/*"):
            if not f.endswith(".csv"):
                os.remove(f)
    else:
        os.makedirs(out_dir, exist_ok=True)
    
    # 把该模型的结果单独保存为独立的csv
    single_df = pd.DataFrame([row])
    out_path = os.path.join(out_dir, f"{model_name}_test_metrics.csv")
    single_df.to_csv(out_path, index=False)
    print(f"Saved solitary CSV for {model_name} to {out_path}")

