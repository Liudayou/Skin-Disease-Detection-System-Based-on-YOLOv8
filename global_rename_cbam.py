import os
import glob
import re

def replace_in_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        return # Skip binary files

    new_content = content.replace('CBAM', 'CBAM').replace('cbam', 'cbam')
    
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated content in: {filepath}")

def rename_file(filepath):
    dir_name = os.path.dirname(filepath)
    base_name = os.path.basename(filepath)
    if 'cbam' in base_name.lower():
        new_base_name = base_name.replace('cbam', 'cbam').replace('CBAM', 'CBAM')
        new_filepath = os.path.join(dir_name, new_base_name)
        os.rename(filepath, new_filepath)
        print(f"Renamed: {filepath} -> {new_filepath}")
        return new_filepath
    return filepath

# Directories to search
targets = [
    'Experiment/scripts/**/*.py',
    'Experiment/scripts/**/*.yaml',
    'Experiment/skin_lesion_fullstack/backend/**/*.py',
    'Experiment/README.md',
    'Experiment/generate_notebook.py',
]

all_files = []
for target in targets:
    all_files.extend(glob.glob(target, recursive=True))

for filepath in all_files:
    if os.path.isfile(filepath):
        # 1. Replace content
        replace_in_file(filepath)
        # 2. Rename file
        rename_file(filepath)

# Let's also rename directories in runs and results if they contain cbam?
# To be safe, let's just rename weights if they exist and are referenced in app.py
for w in glob.glob('Experiment/weights/**/*cbam*', recursive=True):
    rename_file(w)

