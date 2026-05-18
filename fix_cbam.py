import os, glob

# 1. Update cbam_attention.py to also register 'CBAM'
mod_file = 'Experiment/scripts/modules/cbam_attention.py'
with open(mod_file, 'r') as f:
    code = f.read()
if 'tasks.__dict__["CBAM"] = CBAM' not in code:
    code = code.replace('tasks.__dict__["CBAM"] = CBAM', 'tasks.__dict__["CBAM"] = CBAM\n    tasks.__dict__["CBAM"] = CBAM')
    with open(mod_file, 'w') as f:
        f.write(code)

# 2. Rename generated YAMLs and update their content to say CBAM instead of CBAM
for yaml_file in glob.glob('Experiment/scripts/models/generated/attention_ablations/*cbam*.yaml'):
    new_yaml_file = yaml_file.replace('cbam', 'cbam')
    with open(yaml_file, 'r') as f:
        content = f.read()
    content = content.replace('CBAM', 'CBAM')
    with open(new_yaml_file, 'w') as f:
        f.write(content)
    os.remove(yaml_file)

# 3. Rename generated Python scripts and update contents
for py_file in glob.glob('Experiment/scripts/attention/ablations/*cbam*.py'):
    new_py_file = py_file.replace('cbam', 'cbam')
    with open(py_file, 'r') as f:
        content = f.read()
    content = content.replace('cbam_p', 'cbam_p')
    content = content.replace('CBAM_P', 'CBAM_P')
    with open(new_py_file, 'w') as f:
        f.write(content)
    os.remove(py_file)
