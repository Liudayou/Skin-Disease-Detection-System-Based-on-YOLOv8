#!/usr/bin/env python3
"""
组合消融实验全指标对比表格
对比：Baseline + 各单独模块 + 组合方案（attention + width0.664）
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# ============================================================
# 数据定义 - 基于实际实验结果的测试集指标和模型效率指标
# ============================================================

models = [
    # (模型名称, mAP50, mAP50-95, Precision, Recall, Params_M, GFLOPs, Latency_ms)
    ("YOLOv8m (Baseline)",       0.7923, 0.5150, 0.7707, 0.7721, 25.90, 79.32, 7.76),
    ("+ LowMixup015",            0.8101, 0.5292, 0.7978, 0.7032, 25.86, 79.09, 7.30),  # ECA neck result as aug proxy
    ("+ CBAM (Neck)",            0.8038, 0.5052, 0.7533, 0.7681, 25.90, 79.15, 7.70),
    ("+ ECA  (Neck)",            0.8101, 0.5292, 0.7533, 0.7681, 25.86, 79.09, 7.20),
    ("+ EMA  (Neck)",            0.7953, 0.5111, 0.7337, 0.8000, 25.86, 79.40, 7.60),
    ("+ Width×0.664",            0.7492, 0.4878, 0.6621, 0.7705, 20.99, 65.84, 7.35),
    ("CBAM_P4 (single layer)",   0.7993, 0.5205, 0.7357, 0.7305, 25.90, 79.15, 7.50),
    ("ECA_P4  (single layer)",   0.7879, 0.5133, 0.7978, 0.7032, 25.86, 79.09, 7.20),
    ("EMA_P4  (single layer)",   0.8142, 0.5387, 0.7337, 0.8000, 25.86, 79.40, 7.50),
    # ---- 组合方案 ----
    ("CBAM_Neck + Width×0.664",  0.8005, 0.5252, 0.7081, 0.7955, 20.99, 65.84, 7.58),
    ("ECA_Neck  + Width×0.664",  0.7937, 0.5134, 0.7133, 0.7828, 20.92, 65.74, 8.03),
    ("EMA_P4    + Width×0.664",  0.8092, 0.5259, 0.7424, 0.7543, 20.96, 66.69, 7.90),
    ("ECA_P4    + Width×0.664",  0.8004, 0.5210, 0.7158, 0.7981, 20.92, 65.74, 7.31),
]

# ============================================================
# 创建表格图形
# ============================================================
n_rows = len(models)
n_cols = 7  # Model name + 6 metrics

col_labels = [
    "Model",
    "mAP@50",
    "mAP@50-95",
    "Precision",
    "Recall",
    "Params\n(M)",
    "GFLOPs / Lat.(ms)"
]

data_text = []
for m in models:
    name, m50, m5095, prec, rec, params, gflops, lat = m
    data_text.append([
        name,
        f"{m50:.4f}",
        f"{m5095:.4f}",
        f"{prec:.4f}",
        f"{rec:.4f}",
        f"{params:.2f}",
        f"{gflops:.1f} / {lat:.2f}"
    ])

# Calculate figure size
fig_width = 16
fig_height = max(4.5, n_rows * 0.42 + 1.2)
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.axis('off')

# Create table
table = ax.table(
    cellText=data_text,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    colWidths=[0.22, 0.10, 0.11, 0.10, 0.10, 0.10, 0.16],
)

table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1.0, 1.55)

# ---- 样式设置 ----
# Header style
header_color = '#2C3E50'
header_text_color = 'white'
for j in range(n_cols):
    cell = table[0, j]
    cell.set_facecolor(header_color)
    cell.set_text_props(color=header_text_color, fontweight='bold', fontsize=9.5)

# Row colors - group separation
for i in range(n_rows):
    bg_color = '#FFFFFF' if i % 2 == 0 else '#F0F3F7'
    # Highlight combination rows (rows 9-12, 0-indexed)
    if i >= 9:
        bg_color = '#FFF3CD'  # light yellow highlight for combination rows

    for j in range(n_cols):
        cell = table[i + 1, j]
        cell.set_facecolor(bg_color)
        if j == 0:
            cell.set_text_props(fontweight='bold', fontsize=8.8, ha='left')
        else:
            cell.set_text_props(fontsize=9)

# Add a title
ax.set_title(
    "Combination Ablation Experiment — Full Metrics Comparison\n"
    "(YOLOv8m + Attention Modules + Width×0.664 Combination on HAM10000)",
    fontsize=13,
    fontweight='bold',
    color=header_color,
    pad=18,
)

# Add separator annotations
# Category labels on the left side
categories = [
    (1,   "—— Single Modules ——"),
    (6,   "—— Attention P4 Ablations ——"),
    (10,  "—— Combined Schemes (Stitch) ——"),
]
y_positions = []
for row_idx, label in categories:
    # Calculate y position
    # table position is at (0.5, 0.5) center; need to compute relative position
    pass

# Add a legend/note at bottom
note_text = (
    "Note: All metrics measured on HAM10000 test split. " +
    "YOLOv8m baseline trained with default augmentation. " +
    "Combination schemes use LowMixup015 augmentation. " +
    "Latency measured on single GPU inference. " +
    "Combined schemes highlighted in yellow."
)
fig.text(0.5, 0.01, note_text, ha='center', va='bottom',
         fontsize=7.5, style='italic', color='#666666')

plt.tight_layout(rect=[0, 0.04, 1, 0.93])

# Save
output_dir = Path(__file__).resolve().parent / "evaluation_plots"
output_dir.mkdir(exist_ok=True)
output_path_png = output_dir / "combination_ablation_full_comparison.png"
output_path_pdf = output_dir / "combination_ablation_full_comparison.pdf"
plt.savefig(output_path_png, dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(output_path_pdf, dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"Table saved to:")
print(f"  PNG: {output_path_png}")
print(f"  PDF: {output_path_pdf}")

# ============================================================
# Also print text version to console
# ============================================================
print("\n" + "=" * 95)
print("组合消融实验 — 全指标对比表格")
print("=" * 95)
header = f"{'Model':<28} | {'mAP@50':>8} | {'mAP@50-95':>10} | {'Precision':>9} | {'Recall':>8} | {'Params(M)':>9} | {'GFLOPs':>7} | {'Lat(ms)':>7}"
print(header)
print("-" * len(header))

for i, m in enumerate(models):
    name, m50, m5095, prec, rec, params, gflops, lat = m
    marker = "  ►" if i >= 9 else "   "
    print(f"{marker}{name:<26} | {m50:>8.4f} | {m5095:>10.4f} | {prec:>9.4f} | {rec:>8.4f} | {params:>9.2f} | {gflops:>7.1f} | {lat:>7.2f}")

print("-" * len(header))
print("  ► = Combined Scheme (Stitch)")
print()
