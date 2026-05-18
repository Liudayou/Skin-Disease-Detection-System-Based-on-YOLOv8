#!/usr/bin/env python3
"""将项目中所有 cbam/CBAM 重命名为 cbam/CBAM。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ===================== 第一步：重命名目录 =====================
dirs_to_rename = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    for d in dirnames:
        if 'cbam' in d.lower():
            old_path = Path(dirpath) / d
            # 跳过 .git 等
            if '.git' in str(old_path):
                continue
            dirs_to_rename.append(old_path)

# 按路径深度倒序排列（深层先改，避免父目录改名后子目录路径失效）
dirs_to_rename.sort(key=lambda p: len(p.parts), reverse=True)

for old_path in dirs_to_rename:
    new_name = old_path.name.replace('cbam', 'cbam').replace('CBAM', 'CBAM')
    new_path = old_path.parent / new_name
    print(f"[dir] {old_path} -> {new_path}")
    os.rename(old_path, new_path)

# ===================== 第二步：更新文件内容 =====================
file_exts = {'.py', '.ipynb', '.yaml', '.yml', '.md', '.txt', '.csv', '.json', '.sh'}

for dirpath, dirnames, filenames in os.walk(ROOT):
    for fname in filenames:
        fpath = Path(dirpath) / fname
        if fpath.suffix.lower() not in file_exts:
            continue
        # 跳过二进制 / 大文件
        if fpath.stat().st_size > 10 * 1024 * 1024:  # >10MB skip
            continue
        try:
            content = fpath.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            continue

        new_content = content.replace('CBAM', 'CBAM').replace('cbam', 'cbam')
        if new_content != content:
            fpath.write_text(new_content, encoding='utf-8')
            print(f"[file] {fpath}")

print("\nDone! All cbam -> cbam renamed.")
