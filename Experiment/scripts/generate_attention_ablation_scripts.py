import os
import yaml

ATTENTION_TYPES = {
    'ECA': [0],
    'EMA': [32],
    'CBAM': [16, 7]
}

def generate_yaml(attn_name, attn_args, stage):
    yaml_str = f"""# YOLOv8m + {attn_name} at {stage} only
nc: 80
scales:
  n: [0.33, 0.25, 1024]
  s: [0.33, 0.50, 1024]
  m: [0.67, 0.75, 768]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.25, 512]

backbone:
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 10
  - [[-1, 6], 1, Concat, [1]] # 11
  - [-1, 3, C2f, [512]] # 12 P4_up
"""
    curr_idx = 12
    p4_up_idx = 12
    if stage == 'P4':
        yaml_str += f"  - [-1, 1, {attn_name}, {attn_args}] # {curr_idx + 1}\n"
        curr_idx += 1
        p4_up_idx = curr_idx
    
    yaml_str += f"""
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # {curr_idx + 1}
  - [[-1, 4], 1, Concat, [1]] # {curr_idx + 2}
  - [-1, 3, C2f, [256]] # {curr_idx + 3} P3
"""
    curr_idx += 3
    p3_idx = curr_idx
    if stage == 'P3':
        yaml_str += f"  - [-1, 1, {attn_name}, {attn_args}] # {curr_idx + 1}\n"
        curr_idx += 1
        p3_idx = curr_idx

    yaml_str += f"""
  - [-1, 1, Conv, [256, 3, 2]] # {curr_idx + 1}
  - [[-1, {p4_up_idx}], 1, Concat, [1]] # {curr_idx + 2}
  - [-1, 3, C2f, [512]] # {curr_idx + 3} P4_down
"""
    curr_idx += 3
    p4_down_idx = curr_idx
    if stage == 'P4':
        yaml_str += f"  - [-1, 1, {attn_name}, {attn_args}] # {curr_idx + 1}\n"
        curr_idx += 1
        p4_down_idx = curr_idx

    yaml_str += f"""
  - [-1, 1, Conv, [512, 3, 2]] # {curr_idx + 1}
  - [[-1, 9], 1, Concat, [1]] # {curr_idx + 2}
  - [-1, 3, C2f, [1024]] # {curr_idx + 3} P5
"""
    curr_idx += 3
    p5_idx = curr_idx
    if stage == 'P5':
        yaml_str += f"  - [-1, 1, {attn_name}, {attn_args}] # {curr_idx + 1}\n"
        curr_idx += 1
        p5_idx = curr_idx
        
    yaml_str += f"\n  - [[{p3_idx}, {p4_down_idx}, {p5_idx}], 1, Detect, [nc]] # Detect(P3, P4, P5)\n"
    
    return yaml_str

os.makedirs('Experiment/scripts/models/generated/attention_ablations', exist_ok=True)
os.makedirs('Experiment/scripts/attention/ablations', exist_ok=True)

# Generate scripts
for attn, args in ATTENTION_TYPES.items():
    for stage in ['P3', 'P4', 'P5']:
        yaml_content = generate_yaml(attn, args, stage)
        yaml_path = f"Experiment/scripts/models/generated/attention_ablations/yolov8m_{attn.lower()}_{stage.lower()}.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)
        
        train_script = f"""import os
import sys

# Add the 'scripts' directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

from yolov8m_train_utils import TrainManager

def main():
    model_cfg = os.path.join("scripts", "models", "generated", "attention_ablations", "yolov8m_{attn.lower()}_{stage.lower()}.yaml")
    
    manager = TrainManager(
        run_name="attention__lowMixup015__yolov8m_{attn.lower()}_{stage.lower()}_ham_univ30_lowMixup015_e100",
        model_cfg=model_cfg,
        batch_size=32,
        augment_mode="low_mixup_015"
    )
    
    manager.train()

if __name__ == '__main__':
    main()
"""
        py_path = f"Experiment/scripts/attention/ablations/train_yolov8m_{attn.lower()}_{stage.lower()}.py"
        with open(py_path, "w") as f:
            f.write(train_script)

print("Generated all YAML and python scripts.")

