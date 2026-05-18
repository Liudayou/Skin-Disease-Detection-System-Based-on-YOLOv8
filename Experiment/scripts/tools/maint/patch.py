import re
from pathlib import Path

files = [
    Path(r'd:\SkinSystem\Experiment\scripts\attention\train_yolov8m_eca_neck.py'),
    Path(r'd:\SkinSystem\Experiment\scripts\attention\train_yolov8m_ema_neck.py')
]

for p in files:
    content = p.read_text('utf-8')
    
    # 1. change default run-name 
    content = re.sub(r'default="yolov8m_(eca|ema)_neck_ham_univ30_augMM"', r'default="yolov8m_\1_neck_ham_univ30_lowMixup015"', content)
    
    # 2. change add_mosaic_mixup_cli_args(p) and add extra args
    new_args = '''    add_mosaic_mixup_cli_args(p, mosaic_default=1.0, mixup_default=0.15, close_mosaic_default=10)
    p.add_argument("--scale", type=float, default=0.5, help="几何 scale（lowMixup015 默认 0.5）")
    p.add_argument("--degrees", type=float, default=None)
    p.add_argument("--translate", type=float, default=None)
'''
    content = re.sub(r'    add_mosaic_mixup_cli_args\(p\)\n', new_args, content)
    
    # 3. Add to run_training args
    new_kwargs = '''        amp=not bool(a.no_amp),
        scale=float(a.scale) if a.scale is not None else None,
        degrees=float(a.degrees) if a.degrees is not None else None,
        translate=float(a.translate) if a.translate is not None else None,'''
    content = re.sub(r'        amp=not bool\(a.no_amp\),', new_kwargs, content)
    
    p.write_text(content, 'utf-8')
    print(f"Patched {p.name}")

