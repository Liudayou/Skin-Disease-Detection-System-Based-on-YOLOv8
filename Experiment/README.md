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

### 注意力机制单阶段消融实验（阶段与机制横向对比）

下表为 ECA、EMA、CBAM 在 YOLOv8 Neck 特征金字塔的不同感受野（P3 / P4 / P5）独立插入的测试集全指标消融对比。各层级插入细节与关注点如下：
- **P3 (低层/长宽大/高分辨率)**：主要针对**微小病变（小目标）**。注意力插入该层级，旨在增强网络在早期浅层特征提取时，对细小血管特征或初期皮损边缘微小纹理通道权重的捕捉能力。
- **P4 (中层/中分辨率)**：主要针对**中等大小病灶**。此阶段为特征融合的中间态，模块能够增强不同通道在长短距离范围的上下文关联建模，抗击邻近的色斑混淆。
- **P5 (高层/长宽小/低分辨率)**：主要针对**大面积扩散病灶（大目标）**。深层含有丰富的强语义特征，在此处应用注意力机制主要为了忽略毛发/反光背景等强烈干扰信息，极速提取肿瘤的最核心、粗略的高阶边界轮廓进行多分类任务。

| 注意力机制 | 插入位置 | 实验网络 YAML 配置 | Precision | Recall | mAP@50 | mAP@50-95 | 参数量 (M) | GFLOPs | 推理延迟 (ms) |
| :-------: | :------: | :--------------: | :-------: | :----: | :----: | :-------: | :------: | :----: | :---------: |
|   CBAM    |    P3    | `yolov8m_cbam_p3.yaml` | 0.6962 | 0.7658 | 0.7692 | 0.4902 | 25.87 | 79.10 | 2.13 |
|   CBAM    |    P4    | `yolov8m_cbam_p4.yaml` | 0.7357 | 0.7305 | 0.7993 | 0.5205 | 25.90 | 79.15 | 1.88 |
|   CBAM    |    P5    | `yolov8m_cbam_p5.yaml` | 0.7166 | 0.7800 | 0.7856 | 0.5067 | 25.90 | 79.15 | 1.86 |
|   ECA     |    P3    | `yolov8m_eca_p3.yaml`  | 0.7022 | 0.7363 | 0.7723 | 0.4941 | 25.86 | 79.09 | 1.86 |
|   ECA     |    P4    | `yolov8m_eca_p4.yaml`  | 0.7978 | 0.7032 | 0.7879 | 0.5133 | 25.86 | 79.09 | 1.86 |
|   ECA     |    P5    | `yolov8m_eca_p5.yaml`  | 0.7029 | 0.7245 | 0.7536 | 0.4829 | 25.86 | 79.09 | 1.85 |
|   EMA     |    P3    | `yolov8m_ema_p3.yaml`  | 0.7813 | 0.7239 | 0.7928 | 0.5096 | 25.86 | 79.24 | 1.94 |
|   EMA     |    P4    | `yolov8m_ema_p4.yaml`  | 0.7337 | 0.8000 | 0.8142 | 0.5387 | 25.86 | 79.40 | 1.92 |
|   EMA     |    P5    | `yolov8m_ema_p5.yaml`  | 0.7921 | 0.7739 | 0.8152 | 0.5206 | 25.86 | 79.18 | 1.87 |

> **实验结论分析**：
> EMA（高效多尺度注意力机制）在保留高频空间信息与多尺度协同上表现突出，其中在 P4 中等分层和 P5 顶层插入获得了本组所有消融中最高的 `mAP@50 (0.8142, 0.8152)`；相比下，ECA 和 CBAM 的空间惩罚与权重重新分配在部分层级有不俗表现。三种消融操作几乎均没有增加核心计算负担，证明了此类轻量化即插即用模块对基线提点的泛用性。

### 注意力机制全 Neck 阶段插入消融实验

针对单一维度的注意力感受野有限的情况，本组实验将 CBAM、ECA 和 EMA 分别**全面插入到整个 Neck 网络（PAN-FPN的对应融合节点）**，以验证多尺度联合空间注意力机制对最终性能的提升。

| 注意力机制（全 Neck） | 实验网络 YAML 配置 | Precision | Recall | mAP@50 | mAP@50-95 | 参数量 (M) | GFLOPs | 推理延迟 (ms) |
| :-------: | :--------------: | :-------: | :----: | :----: | :-------: | :------: | :----: | :---------: |
|   CBAM    | `yolov8m_cbam_neck.yaml` | 0.7532 | 0.7680 | 0.8038 | 0.5051 | 25.99 | 79.50 | 7.36 |
|   ECA     | `yolov8m_eca_neck.yaml`  | 0.6766 | 0.7030 | 0.8100 | 0.5292 | 25.90 | 79.30 | 7.71 |
|   EMA     | `yolov8m_ema_neck.yaml`  | 0.6618 | 0.7064 | 0.7952 | 0.5110 | 25.91 | 79.90 | 7.74 |

> **实验结论分析（全位置插入）**：
> 相比单阶段的插入，**全面介入 Neck 层**并未带来预期的显著综合暴涨，其中 **ECA** 实现最高的全层泛化，取得了 `mAP@50` 高达 **0.8100** 的成绩！但同时可以观察到，在过度的多尺度融合网络中施加注意力堆叠，会一定程度引起梯度信息的冗杂和网络过度拟合。因此，针对轻量模型部署，仅在最佳单层（如 P4/P5 EMA 或 P4 ECA）进行插入可能更具投入产出比。

### 轻量化与网络通道缩放消融实验

为了满足最终医疗辅助诊断系统的低延迟及移动端部署需求，基于 YOLOv8m 基线，本项目分别探讨了 **缩减通道宽度（width0664）** 以及 **全主干网络（Backbone）替换** 两种轻量化思路。这部分的所有消融实验均统一采用 `lowMixup015` 数据组合方案以控制变量。

| 轻量化方案 | 实验网络 YAML 配置 | Precision | Recall | mAP@50 | mAP@50-95 | 参数量 (M) | GFLOPs | 推理延迟 (ms) |
| :-------: | :--------------: | :-------: | :----: | :----: | :-------: | :------: | :----: | :---------: |
|   Baseline (YOLOv8m) | `yolov8m.yaml`         | 0.7707 | 0.7721 | 0.7923 | 0.5149 | 25.90 | 79.32 | 7.76 |
|   width0664缩放       | `yolov8m_width0.664.yaml` | 0.6620 | 0.7704 | 0.7491 | 0.4878 | 20.88 | 65.51 | 7.67 |
|   MobileNetV3主干     | `yolov8m_mobilenetv3.yaml` | 0.6534 | 0.6104 | 0.6865 | 0.4540 | 43.22 | 98.17 | 8.12 |
|   GhostNet主干        | `yolov8m_ghostnet.yaml`    | 0.6378 | 0.6976 | 0.7947 | 0.5145 | 25.93 | 79.32*| 7.49 |
|   PConv(FasterNet)主干| `yolov8m_pconv.yaml`       | 0.6553 | 0.6390 | 0.7864 | 0.5154 | 25.72 | 79.32*| 8.86 |

> **轻量化分析**：直接通过调节通道广度因子（`width_multiple=0.664`），模型的参数量暴降约 **20%（从 25.9M 降至 20.88M）**，这是通过硬件直观加速的最优解（为最后的 Stitch 版打下基础结构）。相比之下，利用 `timm` 库桥接外部 Backbone（如 MobileNetV3）因为 YOLO 融合层的强行对齐，反而引起了参数量和计算量的激增和精度骤降。GhostNet 和 PConv 虽然做到了有效轻量级特征抓取，在精度上表现优异（接近基线测试水平 0.7947），但在实际特定设备上的 GFLOPs 表现未必全线占优。最终采取“宽幅截断（width因子缩小）”的直接工程化降参效益更好。

### 数据增强策略消融实验

针对医疗数据集极度受限（易于陷入严重过拟合）的问题，深入探索几何变换因子及图像混合组合对于此类色斑病变病理的干扰权重：

| 增强策略 | 核心增强变量配置 | Precision | Recall | mAP@50 | mAP@50-95 | 参数量 (M) | GFLOPs | 推理延迟 (ms) |
| :-------: | :---------------------------------- | :-------: | :----: | :----: | :-------: | :------: | :----: | :---------: |
| Default 基线   | `mosaic=1.0`, `mixup=0.0`, `scale=0.5` | 0.7707 | 0.7721 | 0.7923 | 0.5149 | 25.90 | 79.32 | 7.76 |
| lowMixup015   | `mosaic=1.0`, **`mixup=0.15`**, `scale=0.5` | 0.6836 | 0.6711 | **0.8027** | 0.5098 | 25.90 | 79.32 | 8.15 |
| tightScale03  | `mosaic=1.0`, `mixup=0.2`, **`scale=0.3`** | 0.7369 | 0.7756 | 0.7856 | 0.5053 | 25.90 | 79.32 | 7.76 |

> **数据增强分析**：YOLOv8 默认极强的多图混合数据增强若不做任何调教直接应用于单病灶皮肤图像会破坏生理病变结构的完整性。测试发现：引入微弱的小比例 `mixup=0.15` (`lowMixup015`) 对多分类的交叉容错率有巨大提振，其促成了 **0.8027** 的泛化 mAP，这能有效缓解由于 HAM10000 血管病变(`vasc`)及光化性角化病等极度长尾数据引起的单一崩盘。而收紧缩放变量 `scale`（`tightScale03`）在防止肿瘤特征畸变上也有稳定作用，总体验证 **微量混合比** 具备最佳病发特征留存率。

### 组合方案推荐（基于各消融实验结果综合分析）

基于上述各维度消融实验的独立结论，从**精度、效率和部署可行性**三个角度出发，给出以下最优组合策略推荐：

| 推荐方案 | 组合策略 | Precision | Recall | mAP@50 | mAP@50-95 | 参数量 (M) | GFLOPs | 推理延迟 (ms) | 适用场景 |
| :-------: | :------: | :-------: | :----: | :-----: | :-------: | :------: | :----: | :---------: | :------: |
| **方案一（已验证）** | width0664 + CBAM(全Neck) + lowMixup015 | 0.7081 | 0.7955 | 0.8005 | 0.5252 | 20.88 | 65.84 | 7.58 | 移动端/嵌入式部署（低功耗） |
| **方案二（已验证）** | width0664 + ECA(全Neck) + lowMixup015 | 0.7133 | 0.7828 | 0.7937 | 0.5134 | 20.88 | 65.74 | 8.03 | 移动端/嵌入式部署（备选） |
| **方案三（已验证）** | EMA P4 + lowMixup015（单阶段消融） | 0.7337 | 0.8000 | 0.8142 | 0.5387 | 25.91 | 79.40 | 1.92 | 服务器端部署（追求最高精度） |
| **🏆 方案四（最终采用）** | EMA P4 + width0664 + lowMixup015 | 0.7424 | 0.7543 | **0.8092** | **0.5259** | 20.88 | ~65.7 | ~7.5 | **精度与效率最优兼顾** |
| **方案五（已验证）** | ECA P4 + width0664 + lowMixup015 | 0.7158 | 0.7981 | **0.8004** | **0.5210** | 20.88 | ~65.5 | ~7.8 | 轻量化高精度组合 |

> 所有方案均已在 HAM10000 测试集上完成验证。其中方案三保持纯精度最高（mAP@50=0.8142），方案四在 width0664 轻量化压缩下以 mAP@50=0.8092 实现了**精度-效率最优平衡**，相比方案三仅损失 0.005 mAP@50 但 GFLOPs 降低约 17%。方案五以 ECA 轻量注意力 + width0664 组合，在 ~65.5 GFLOPs 下取得 mAP@50=0.8004，是轻量化部署的性价比之选。

**推荐理由分析：**

1. **方案四（EMA P4 + width0664 + lowMixup015）**：✨ **精度-效率帕累托最优选择**。实测 test 集 mAP@50=**0.8092**、mAP@50-95=**0.5259**，以仅 ~65.7 GFLOPs 的计算量逼近全精度 EMA P4 的表现。EMA 在多尺度空间保持和信息聚合上的优势，叠加 width0664 约 20% 参数量压缩，使其成为兼顾服务器端高精度需求与部署成本的**首选方案**。

2. **方案三（EMA P4 + lowMixup015）**：纯精度最高方案。EMA P4 单阶段消融取得了全部注意力实验中第二高的 mAP@50 (0.8142)，且 mAP@50-95 (0.5387) 位列全部实验第一，推理延迟仅 1.92ms。若部署环境对算力不敏感且追求极致精度，此方案为最优选择。

3. **方案一（width0664 + CBAM + lowMixup015）**：移动端/嵌入式部署首选。实测 mAP@50=0.8005、mAP@50-95=0.5252，参数量降至 20.88M，GFLOPs 降至 65.84。CBAM 的双重通道-空间注意力在轻量化约束下仍表现出色，Recall 达到 0.7955，对病灶漏检率控制最优。

4. **方案五（ECA P4 + width0664 + lowMixup015）**：轻量化高性价比方案。实测 mAP@50=**0.8004**、mAP@50-95=**0.5210**，以 ECA 作为最轻量的通道注意力模块，在 P4 单点插入 + width0664 压缩后仍保持超过 0.80 的 mAP@50，适合对模型体积和推理速度要求极致的场景。

5. **方案二（width0664 + ECA + lowMixup015）**：移动端备选方案。实测 mAP@50=0.7937、mAP@50-95=0.5134，轻量化程度相当但精度略逊于方案一和方案五。

> **🏆 最终方案选定：方案四**（EMA P4 + width0664 + lowMixup015）为本项目最终部署模型。该方案在 ~65.7 GFLOPs 的轻量计算量下取得 mAP@50=0.8092、mAP@50-95=0.5259，以极小精度代价（相比纯精度最优的方案三仅损失 0.005 mAP@50）换取了约 17% 的 GFLOPs 压缩，实现了精度-效率的帕累托最优平衡。EMA 高效多尺度注意力在 Neck P4 中等感受野层级的单点插入，叠加 width0664 宽度缩放和 lowMixup015 微量混合增强，形成了兼顾检测能力与推理速度的最优组合。

---

## 七、评估可视化与图表分析

本项目将各个特定阶段下的实验验证结果以及数据走势可视化自动汇总至 `Experiment/evaluation_plots/` 目录下，按分类任务的不同阶段区分，核心包括：

1. **综合评价曲线（`BoxPR_curve.png` 等）**：
   - `BoxPR_curve.png`（**PR 曲线**）：横轴 Recall，纵轴 Precision。它衡量了皮肤病检测系统在多个置信度阈值变动下精度和全集覆盖率的博弈平衡，也是直接计算 `mAP` 的积分投影；
   - `BoxP_curve.png` / `BoxR_curve.png` / `BoxF1_curve.png`：置信度不同变化与精准度（P）、召回率（R）和调和平均数（F1-score）的最优解折线图。借此可以在部署系统代码中设定最优判出阈值。

2. **病变识别统计（`confusion_matrix.png` 系列）**：
   - 包含基于绝对数量的 `confusion_matrix.png` 和显示百分比置信分布的 `confusion_matrix_normalized.png`。在诸如 HAM10000 这样存在细粒度形态交叠（例如黑色素瘤 `mel` 易于和良性正常色素痣 `nv` 混淆）的数据集上，这两个矩阵表最为清晰地反映了模型把什么特征认成了错误疾病的倾向频次。

3. **可视化前向对比矩阵（`val_batch_*.jpg` 系列）**：
   - 模型会在每次训练评估期选取一部分抽样结果（如 `val_batch0_labels.jpg` 人工真值标注对比 `val_batch0_pred.jpg` 预测标注出框）。这是通过可视对比去诊断模型的真实拟合感受（能否抓住发丝背景后的病变轮廓、识别框包围紧凑程度等）的核心参考图像。

### Stitch 组合方案评估图表（`evaluation_plots/stitch/`）

| 子目录 | 对应模型 |
|--------|----------|
| `scheme4_ema_p4_width0664_lowMixup015_e100/` | 方案四：EMA P4 + width0664 + lowMixup015 |
| `scheme5_eca_p4_width0664_lowMixup015_e100/` | 方案五：ECA P4 + width0664 + lowMixup015 |
| `yolov8m_w0664_cbam_ham_univ30_lowMixup015_e100/` | 方案一：CBAM 全 Neck + width0664 + lowMixup015 |
| `yolov8m_w0664_eca_ham_univ30_lowMixup015_e100/` | 方案二：ECA 全 Neck + width0664 + lowMixup015 |

---

## 八、自定义模块说明

`scripts/modules/` 中包含以下自定义算子，训练和推理时自动注册到 Ultralytics：

| 模块 | 文件 | 说明 |
|------|------|------|
| CBAM | `cbam_attention.py` | Channel Attention + Batch Normalization 混合注意力 |
| ECA | `eca_attention.py` | Efficient Channel Attention |
| EMA | `ema_attention.py` | Efficient Multi-scale Attention |
| PConv | `pconv_fasternet.py` | Partial Convolution（FasterNet 轻量化卷积） |
| C2f_Faster | `c2f_faster.py` | FasterNet 风格的 C2f 模块 |
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

评估图表（混淆矩阵、PR 曲线等）位于 `evaluation_plots/stitch/` 下对应子目录。

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
