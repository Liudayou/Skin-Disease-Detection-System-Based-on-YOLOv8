import os, json

# 1. Fix Notebook
nb_path = 'Experiment/run_attention_ablations.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if 'source' in cell:
        new_source = []
        for line in cell['source']:
            line = line.replace('CBAM', 'CBAM').replace('cbam', 'cbam')
            new_source.append(line)
        cell['source'] = new_source

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

# 2. Fix README.md
readme_path = 'Experiment/README.md'
with open(readme_path, 'r', encoding='utf-8') as f:
    readme = f.read()

# Replace CBAM with CBAM only in the specific ablation table chapter for now (or globally if it won't break things? 
# globally it breaks the web app instructions `yolov8m_w0664_cbam_lowMixup015_best.pt` which is a real file.
# I will only replace CBAM with CBAM in the section I added)

lines = readme.split('\n')
new_lines = []
in_section = False
for line in lines:
    if '### 注意力机制单阶段消融实验' in line:
        in_section = True
    if '---' in line and in_section:
        in_section = False # ends at next rule or EOF

    if in_section:
        # replace cbam strings to cbam
        line = line.replace('CBAM', 'CBAM').replace('cbam', 'cbam')
    else:
        # User explicitly says "我用的是CBAM注意力模块" which implies I should rename the mention in the first part too.
        # But leave standard paths 'cbam_attention.py' intact? 
        # Actually I'd rather just replace CBAM with CBAM globally EXCEPT in paths and code blocks.
        pass
    new_lines.append(line)

with open(readme_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(new_lines))

