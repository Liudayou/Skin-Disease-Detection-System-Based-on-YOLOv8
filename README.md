# 基于改进 YOLOv8 的皮肤病变智能检测系统

本项目为毕业设计项目，包含两部分：**消融实验训练体系** 和 **皮肤病变 Web 检测系统**。

---

## 一、项目简介

针对皮肤镜图像中的多类别病变检测问题，基于 YOLOv8m 进行改进：

- **注意力机制**：在 Neck（PAN-FPN）中插入 CBAM / ECA / EMA 注意力模块
- **轻量化策略**：width 缩放（width_multiple=0.664）、GhostNet / MobileNetV3 / PConv Backbone 替换
- **数据增强**：lowMixup015（mosaic=1.0、mixup=0.15）、tightScale03（scale=0.3）
- **Stitch 组合方案**：width0664 + CBAM/ECA/EMA(P4) + lowMixup015 —— 最终采用 **方案四**（EMA P4 + width0664 + lowMixup015），mAP@50=0.8092

数据集：HAM10000 七分类（akiec / bcc / bkl / df / mel / nv / vasc），YOLO 格式标注。

---

## 二、目录结构

```
Experiment/
├── scripts/                    # 消融实验训练脚本
│   ├── yolov8m_train_utils.py  # 训练公共工具模块
│   ├── attention/              # 注意力消融实验
│   ├── augmentation/           # 数据增强消融实验
│   ├── lightweight/            # 轻量化消融实验
│   ├── model_scaling/          # 模型缩放实验
│   ├── stitch/                 # Stitch 组合方案
│   ├── models/                 # 模型结构 YAML + 预训练权重
│   ├── modules/                # 自定义算子（CBAM/ECA/EMA/PConv/GhostNet）
│   ├── backbones/              # 自定义 Backbone
│   └── tools/                  # 评估与可视化工具
├── weights/                    # 整理后的实验权重
│   ├── attention/
│   ├── augmentation/
│   ├── baseline/
│   ├── lightweight/
│   ├── model_scaling/
│   └── stitch/                 # 最终部署权重
├── results/                    # 训练指标 CSV（epoch + final + test）
├── evaluation_plots/           # 评估图表（混淆矩阵、PR 曲线等）
├── runs/detect/                # Ultralytics 训练产物
├── dataset/                    # HAM10000 YOLO 格式数据集
├── skin_lesion_fullstack/      # Web 全栈系统（正式版）
└── web_skin_detection/         # Web 简版演示（FastAPI + 纯 HTML）
```

---

## 三、环境配置

### 3.1 Python 环境

```bash
# 创建 conda 环境（GPU 版 PyTorch）
conda create -n skin_yolov8 python=3.10
conda activate skin_yolov8

# 安装 PyTorch（CUDA 12.x，按实际 CUDA 版本调整）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装后端依赖
pip install -r Experiment/skin_lesion_fullstack/backend/requirements.txt
```

> 如果需要训练实验，还需安装 `timm`：
> ```bash
> pip install timm
> ```

### 3.2 前端环境

```bash
cd Experiment/skin_lesion_fullstack/frontend
npm install
```

要求 Node.js >= 18。

---

## 四、启动 Web 检测系统

### 方式一：开发模式（推荐，前后端热更新）

需要**两个终端**：

**终端 1 — 后端：**
```powershell
conda activate skin_yolov8
cd d:\SkinSystem\Experiment\skin_lesion_fullstack\backend
python app.py
# 默认端口 5000
```

**终端 2 — 前端：**
```powershell
cd d:\SkinSystem\Experiment\skin_lesion_fullstack\frontend
npm run dev
# 默认端口 6006
```

浏览器访问：**http://localhost:6006**

### 方式二：单端口部署（构建后由 Flask 托管）

```powershell
cd d:\SkinSystem\Experiment\skin_lesion_fullstack\frontend
npm run build

conda activate skin_yolov8
cd d:\SkinSystem\Experiment\skin_lesion_fullstack\backend
python app.py
```

浏览器访问：**http://localhost:5000**

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `PORT` | 后端端口 | `5000` |
| `SKIN_DETECTION_MODEL` | 自定义权重路径 | 自动使用 stitch 权重 |

---

## 五、Web 系统功能

| 页面 | 功能 |
|------|------|
| **单张检测** | 上传图片 → YOLOv8 推理 → Canvas 可视化 → AI 初步诊断 |
| **批量检测** | 最多 32 张图片批量推理，缩略图选择查看 |
| **历史记录** | SQLite 持久化，分页查询、详情弹窗、删除 |
| **系统设置** | 模型状态、检测参数、AI 诊断 API 配置（MiMo / DeepSeek 等） |

### AI 诊断功能

系统支持通过 OpenAI 兼容的 Chat Completions API 调用 LLM，基于检测结果生成结构化医学参考信息。

**配置步骤：**
1. 进入「系统设置」页面
2. 填入 API Key（支持 MiMo、DeepSeek、通义千问等）
3. 保存后即可在检测页面点击「获取诊断」

---

## 六、消融实验

### 训练命令示例

```powershell
conda activate skin_yolov8
cd d:\SkinSystem\Experiment\scripts

# 注意力消融 — CBAM 注意力插入 Neck
python attention/train_yolov8m_cbam_neck.py --epochs 100 --batch 64

# 数据增强消融 — lowMixup015
python augmentation/train_yolov8m_lowmixup015.py --epochs 100 --batch 64

# 轻量化消融 — width 缩放
python lightweight/train_yolov8m_width0664_lowMixup015.py --epochs 100 --batch 64

# Stitch 组合方案
python stitch/train_yolov8m_width0664_cbam_lowMixup015.py --epochs 100 --batch 64

# 模型缩放基线
python model_scaling/train_yolov8m_default_aug.py --epochs 100 --batch 64
```

所有训练脚本支持的通用参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--epochs` | 训练轮数 | 100 |
| `--batch` | batch size | 64 |
| `--imgsz` | 推理输入尺寸 | 640 |
| `--device` | GPU | 自动 |
| `--no-amp` | 禁用混合精度 | 启用 AMP |
| `--mosaic` | Mosaic 增强概率 | 1.0 |
| `--mixup` | MixUp 增强概率 | 0.15 |
| `--close-mosaic` | 最后 N 轮关闭 Mosaic | 10 |
| `--scale` | 几何缩放范围 | 0.5 |
| `--eval-test` | 训练后在 test 集评估 | 否 |
| `--eval-test-only` | 只做 test 评估（跳过训练） | 否 |

### 评估工具

```powershell
cd d:\SkinSystem\Experiment\scripts

# 批量 test 评估
python tools/eval_test_all.py

# 生成评估图表（混淆矩阵、PR 曲线等）
python tools/generate_all_eval_plots.py --data /path/to/data.yaml

# 从已有实验目录导出指标（不重新训练）
python tools/export_final_metrics_from_run.py --run-name <实验名> --data-root <数据集路径>
```

### 消融实验分组

| 实验组 | 目录 | 变量 |
|--------|------|------|
| 基线 | `model_scaling/` | YOLOv8n/s/m/l 默认增强 |
| 数据增强 | `augmentation/` | lowMixup015 vs tightScale03 |
| 注意力机制（特定位置） | `attention/ablations/` | ECA、EMA、CBAM 在 P3/P4/P5 不同阶段的独立效果对比 |
| 注意力机制（全 Neck）| `attention/` | CBAM vs ECA vs EMA 插入整个 Neck |
| 轻量化 | `lightweight/` | GhostNet / MV3 / PConv / width0664 |
| Stitch 组合 | `stitch/` | width0664 + CBAM/ECA/EMA(P4) + lowMixup015 |

> 详细消融实验结果数据请查阅完整版 `README.md`。

### 最终方案

**🏆 方案四（EMA P4 + width0664 + lowMixup015）** 为本项目最终部署模型，在约 65.7 GFLOPs 的轻量计算量下取得 mAP@50=0.8092、mAP@50-95=0.5259，实现了精度-效率的最优平衡。

---

## 七、评估可视化

本项目将各个特定阶段下的实验验证结果以及数据走势可视化自动汇总至 `Experiment/evaluation_plots/` 目录下，按分类任务的不同阶段区分，核心包括：

1. **综合评价曲线**：PR 曲线、P/R/F1 曲线
2. **病变识别统计**：混淆矩阵（绝对数量 + 百分比分布）
3. **可视化前向对比矩阵**：人工真值标注与预测标注对比图

---

## 八、自定义模块说明

`scripts/modules/` 中包含以下自定义算子，训练和推理时自动注册到 Ultralytics：

| 模块 | 文件 | 说明 |
|------|------|------|
| CBAM | `cbam_attention.py` | Channel Attention + Batch Normalization 混合注意力 |
| ECA | `eca_attention.py` | Efficient Channel Attention |
| EMA | `ema_attention.py` | Efficient Multi-scale Attention |
| PConv | `pconv_fasternet.py` | Partial Convolution（FasterNet 轻量化卷积） |
| GhostNet | `lightweight_blocks.py` | GhostNet Backbone 模块 |
| MobileNetV3 | `mobilenetv3_backbone.py` | MobileNetV3-Large Backbone（基于 timm） |

---

## 九、权重文件

最终部署权重位于 `weights/stitch/` 及对应实验目录：

| 权重文件 | 参数量 | 说明 |
|----------|--------|------|
| `scheme4_ema_p4_width0664_lowMixup015_e100/best.pt` | ~42 MB | 🏆 **最终部署模型**：EMA P4 + width0664 + lowMixup015 |
| `yolov8m_w0664_cbam_lowMixup015_best.pt` | ~42 MB | width0664 + CBAM + lowMixup015（备选） |
| `yolov8m_w0664_eca_lowMixup015_best.pt` | ~42 MB | width0664 + ECA + lowMixup015（备选） |

---

## 十、数据集格式

数据集采用 YOLO 格式，目录结构：

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

类别（7 类，HAM10000）：

| ID | 英文 | 中文 |
|----|------|------|
| 0 | akiec | 光化性角化病 |
| 1 | bcc | 基底细胞癌 |
| 2 | bkl | 脂溢性角化病 |
| 3 | df | 皮肤纤维瘤 |
| 4 | mel | 黑色素瘤 |
| 5 | nv | 色素痣 |
| 6 | vasc | 血管病变 |

---

## 十一、常见问题

**Q: CUDA 不可用，回退到 CPU 推理？**
A: 检查 PyTorch 是否安装了 GPU 版本：`python -c "import torch; print(torch.cuda.is_available())"`。如果是 CPU 版，需要重新安装：`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

**Q: 加载 CBAM 权重报错？**
A: 确保 `scripts/` 目录在 Python 路径中，后端 `app.py` 会自动注册自定义算子。如果是独立脚本，需手动 `from modules.cbam_attention import register_cbam_for_ultralytics; register_cbam_for_ultralytics()`

**Q: AI 诊断 503 错误？**
A: 前往「系统设置」配置 API Key。模型名必须小写，如 `mimo-v2.5-pro`。

**Q: 前端页面打不开？**
A: 确保后端（5000）和前端（6006）都在运行。开发模式需要两个终端分别启动。
