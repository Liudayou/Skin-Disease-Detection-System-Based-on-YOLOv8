import os
import shutil
from pathlib import Path

weights_dir = Path(r'd:\SkinSystem\Experiment\weights')

for item in weights_dir.iterdir():
    if not item.is_dir():
        continue
    
    name = item.name
    # Don't move directories that don't look like our run names
    if '__' not in name:
        continue
        
    parts = name.split('__')
    category = parts[0]  # will be 'attention', 'lightweight', 'noaug'
    
    if category == 'noaug':
        category = 'baseline'
        
    target_dir = weights_dir / category
    target_dir.mkdir(exist_ok=True)
    
    print(f"Moving '{name}' to '{category}'")
    shutil.move(str(item), str(target_dir / name))

print("Organization complete!")
