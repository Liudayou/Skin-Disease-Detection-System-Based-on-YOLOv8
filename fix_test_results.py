import os
import shutil

src_base = "Experiment/results/attention/ablations"
dst_base = "Experiment/results/attention/test_results/ablations"

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
    src_dir = os.path.join(src_base, run, "test_results")
    dst_dir = os.path.join(dst_base, run)
    
    os.makedirs(dst_dir, exist_ok=True)
    
    if os.path.exists(src_dir):
        for f in os.listdir(src_dir):
            if f.endswith('.csv'):
                shutil.move(os.path.join(src_dir, f), os.path.join(dst_dir, f))
                print(f"Moved {f} to {dst_dir}")
        shutil.rmtree(src_dir)

