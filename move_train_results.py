import os
import shutil

src_base = "Experiment/results/attention/ablations"
dst_base = "Experiment/results/attention/train_results/ablations"

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

for run in ablations:
    src_dir = os.path.join(src_base, run)
    dst_dir = os.path.join(dst_base, run)
    
    os.makedirs(dst_dir, exist_ok=True)
    
    # 迁移训练结果文件
    for ext in ["epoch_metrics.csv", "final_metrics.csv", "data_runtime.yaml"]:
        for f in os.listdir(src_dir):
            if f.endswith(ext):
                src_file = os.path.join(src_dir, f)
                dst_file = os.path.join(dst_dir, ext) # 统一重命名为去掉前缀的标准名称，或者保留原名
                # 保留原名
                dst_file = os.path.join(dst_dir, f)
                shutil.copy2(src_file, dst_file)
                print(f"Copied {f} to {dst_dir}")

