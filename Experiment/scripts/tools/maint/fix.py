import re
from pathlib import Path

files = [
    Path(r'd:\SkinSystem\Experiment\scripts\attention\train_yolov8m_eca_neck.py'),
    Path(r'd:\SkinSystem\Experiment\scripts\attention\train_yolov8m_ema_neck.py')
]

for p in files:
    content = p.read_text('utf-8')
    
    # Clean up duplicated args
    content = re.sub(r'(\s+scale=float\(a\.scale\).*?\n\s+degrees=.*?\n\s+translate=.*?\n)+', r'\n        scale=float(a.scale) if a.scale is not None else None,\n        degrees=float(a.degrees) if a.degrees is not None else None,\n        translate=float(a.translate) if a.translate is not None else None,\n', content)
    
    p.write_text(content, 'utf-8')
    print(f"Fixed {p.name}")
