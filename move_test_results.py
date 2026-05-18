import os
import shutil

base_dir = "Experiment/results/attention/test_results/ablations"
target_base_dir = "Experiment/results/attention/ablations"

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
    # Use _test_eval2 as it corresponds to the latest evaluation
    src_dir = os.path.join(base_dir, f"{run}_test_eval2")
    dst_dir = os.path.join(target_base_dir, run, "test_results")
    
    if os.path.exists(src_dir):
        os.makedirs(dst_dir, exist_ok=True)
        # Move all files
        for f in os.listdir(src_dir):
            src_file = os.path.join(src_dir, f)
            dst_file = os.path.join(dst_dir, f)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)
        
        print(f"Moved {run} test results to {dst_dir}")

