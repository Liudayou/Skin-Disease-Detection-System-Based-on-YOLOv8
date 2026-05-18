import os
import shutil
import re
from pathlib import Path

# Paths
scripts_dir = Path(r'd:\SkinSystem\Experiment\scripts')
modules_dir = scripts_dir / 'modules'
tools_dir = scripts_dir / 'tools'

# Create dirs
modules_dir.mkdir(exist_ok=True)
tools_dir.mkdir(exist_ok=True)

# Define groupings
modules_files = [
    'c2f_faster.py', 'cbam_attention.py', 'eca_attention.py', 
    'ema_attention.py', 'fpn_fusion.py', 'ham_local_backbones.py', 
    'lightweight_blocks.py', 'mobilenetv3_backbone.py', 'pconv_fasternet.py'
]

tools_files = [
    'eval_test_all.py', 'export_final_metrics_from_run.py'
]

# Move files
print('Moving files...')
for f in modules_files:
    src = scripts_dir / f
    if src.exists():
        shutil.move(str(src), str(modules_dir / f))
        print(f'  Moved {f} to modules/')

for f in tools_files:
    src = scripts_dir / f
    if src.exists():
        shutil.move(str(src), str(tools_dir / f))
        print(f'  Moved {f} to tools/')

# Patch all python scripts in scripts_dir
print('\nPatching imports in all scripts...')
module_names = [f.replace('.py', '') for f in modules_files]

patch_count = 0
for py_file in scripts_dir.rglob('*.py'):
    # Skip the tools directory as they don't usually import these, or if they do, we'll patch them too
    try:
        content = py_file.read_text(encoding='utf-8')
        original_content = content
        
        # Replace 'from module_name import ' with 'from modules.module_name import '
        for mod in module_names:
            # Match exact form: rom eca_attention import 
            content = re.sub(rf'^from\s+{mod}\s+import\s+', f'from modules.{mod} import ', content, flags=re.MULTILINE)
            # Match exact form: import eca_attention
            content = re.sub(rf'^import\s+{mod}(\s|$)', f'import modules.{mod}\\1', content, flags=re.MULTILINE)
            
        if content != original_content:
            py_file.write_text(content, encoding='utf-8')
            patch_count += 1
            print(f'  Patched imports in {py_file.relative_to(scripts_dir)}')
            
    except Exception as e:
        print(f'  Error processing {py_file}: {e}')

print(f'\nDone! Patched {patch_count} files.')
