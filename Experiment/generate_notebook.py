import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Attention Ablation Experiments\n",
                "\n",
                "本 Notebook 用于逐个运行 ECA, EMA, CBAM 在 P3, P4, P5 层的消融实验。执行时会将每轮（Epoch）的训练信息打印在下方输出框中。"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

experiments = [
    ("ECA", "P3"), ("ECA", "P4"), ("ECA", "P5"),
    ("EMA", "P3"), ("EMA", "P4"), ("EMA", "P5"),
    ("CBAM", "P3"), ("CBAM", "P4"), ("CBAM", "P5"),
]

for attn, stage in experiments:
    # Markdown cell for description
    md_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"### 运行 {attn} - {stage} 实验"
        ]
    }
    notebook["cells"].append(md_cell)
    
    # Code cell
    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            f"!python scripts/attention/ablations/train_yolov8m_{attn.lower()}_{stage.lower()}.py"
        ]
    }
    notebook["cells"].append(code_cell)

with open("Experiment/run_attention_ablations.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

