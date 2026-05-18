"""
一键为所有实验权重生成评估图表（混淆矩阵、PR 曲线等）。

遍历 weights/ 下所有 best.pt，在指定 split 上调用 Ultralytics val 画图，
输出到 evaluation_plots/<类别>/<实验名>/ 目录。

用法：
    python generate_all_eval_plots.py --data /path/to/data.yaml
"""

import argparse
import multiprocessing
import sys
import types
from pathlib import Path

multiprocessing.freeze_support()

# scripts/ 加入 sys.path，导入自定义模块
_SCRIPTS_ROOT = Path(__file__).resolve().parent.parent  # Experiment/scripts
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

# 权重文件 (.pt) 中记录的模块路径是训练时注册的顶层名（如 cbam_attention.CBAM），
# 而实际代码在 modules.cbam_attention 下，需要创建 sys.modules 别名映射。
try:
    import modules.cbam_attention as _cbam
    sys.modules["cbam_attention"] = _cbam
except Exception:
    pass

try:
    import modules.eca_attention as _eca
    sys.modules["eca_attention"] = _eca
except Exception:
    pass

try:
    import modules.ema_attention as _ema
    sys.modules["ema_attention"] = _ema
except Exception:
    pass

try:
    import modules.pconv_fasternet as _pconv
    sys.modules["pconv_fasternet"] = _pconv
except Exception:
    pass

try:
    import modules.lightweight_blocks as _lw
    sys.modules["lightweight_blocks"] = _lw
except Exception:
    pass

try:
    import modules.mobilenetv3_backbone as _mv3
    sys.modules["mobilenetv3_backbone"] = _mv3
    # 训练脚本通过 ensure_local_backbones 将 backbones.mobilenetv3_backbone
    # 指向 modules/mobilenetv3_backbone.py，因此 .pt 中记录的模块路径是
    # "backbones.mobilenetv3_backbone"，需要创建 backbones 假包映射。
    if "backbones" not in sys.modules:
        _bb = types.ModuleType("backbones")
        _bb.__path__ = []  # type: ignore[attr-defined]
        sys.modules["backbones"] = _bb
    sys.modules["backbones.mobilenetv3_backbone"] = _mv3
except Exception:
    pass

# 注册所有自定义模块到 Ultralytics parse_model 全局命名空间，
# 否则加载包含自定义算子的 .pt 权重会报错。
try:
    from modules.cbam_attention import register_cbam_for_ultralytics
    register_cbam_for_ultralytics()
except Exception as e:
    print(f"[warn] 注册 CBAM 失败: {e}")

try:
    from modules.eca_attention import register_eca_for_ultralytics
    register_eca_for_ultralytics()
except Exception as e:
    print(f"[warn] 注册 ECA 失败: {e}")

try:
    from modules.ema_attention import register_ema_for_ultralytics
    register_ema_for_ultralytics()
except Exception as e:
    print(f"[warn] 注册 EMA 失败: {e}")

try:
    from modules.pconv_fasternet import register_pconv_for_ultralytics
    register_pconv_for_ultralytics()
except Exception as e:
    print(f"[warn] 注册 PConv 失败: {e}")

# C2f_Faster 依赖 fpn_fusion 模块，如果不存在则跳过
try:
    from modules.c2f_faster import register_c2f_faster_for_ultralytics
    register_c2f_faster_for_ultralytics()
except ImportError:
    print("[warn] C2f_Faster 的依赖缺失 (fpn_fusion)，跳过注册")
except Exception as e:
    print(f"[warn] 注册 C2f_Faster 失败: {e}")

# MobileNetV3 backbone（timm）
try:
    import ultralytics.nn.tasks as _tasks
    from modules.mobilenetv3_backbone import MobileNetV3LargeBackbone
    _tasks.__dict__["MobileNetV3LargeBackbone"] = MobileNetV3LargeBackbone
except Exception as e:
    print(f"[warn] 注册 MobileNetV3LargeBackbone 失败: {e}")

# lightweight_blocks (GhostNet 等)
try:
    from modules.lightweight_blocks import register_lightweight_blocks_for_ultralytics
    register_lightweight_blocks_for_ultralytics()
except ImportError:
    print("[warn] modules/lightweight_blocks.py 不存在，GhostNet backbone 模型将跳过或报错")
except Exception as e:
    print(f"[warn] 注册 lightweight_blocks 失败: {e}")

from ultralytics import YOLO

def _resolve_data_yaml(data_path: str) -> str:
    """将 data.yaml 中的相对路径转为绝对路径（Ultralytics val 对路径敏感）。"""
    import tempfile, yaml, os
    data_file = Path(data_path).resolve()
    with open(data_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_dir = data_file.parent
    changed = False
    for key in ("train", "val", "test"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str(data_dir / cfg[key])
            changed = True
    if not changed:
        return str(data_file)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.dump(cfg, tmp, default_flow_style=False)
    tmp.close()
    print(f"[info] data.yaml 中的相对路径已转为绝对路径: {tmp.name}")
    return tmp.name


def main():
    parser = argparse.ArgumentParser(description="一键为所有实验权重生成评估图表")
    parser.add_argument("--data", type=str, required=True, help="data.yaml 完整路径")
    parser.add_argument("--weights-dir", type=str, default=r"D:\SkinSystem\Experiment\weights", help="权重根目录")
    parser.add_argument("--out-dir", type=str, default=r"D:\SkinSystem\Experiment\evaluation_plots", help="图表输出目录")
    parser.add_argument("--split", type=str, default="test", help="数据集划分：test 或 val")
    parser.add_argument("--batch", type=int, default=16, help="评估 batch size")
    args = parser.parse_args()

    weights_root = Path(args.weights_dir).resolve()
    out_root = Path(args.out_dir).resolve()

    if not weights_root.exists():
        print(f"找不到权重目录: {weights_root}")
        return

    # 转换 data.yaml 中的相对路径为绝对路径
    data_yaml = _resolve_data_yaml(args.data)

    # 遍历所有的 best.pt
    best_weights = list(weights_root.rglob("best.pt"))
    print(f"共发现 {len(best_weights)} 个模型权重需要生成图谱。")

    for w_path in best_weights:
        # 解析分类目录，例如从 weights/attention/yolov8m_cbam.../best.pt 提取
        run_name = w_path.parent.name
        category = w_path.parent.parent.name

        print(f"\n" + "="*60)
        print(f"正在生成图表: [{category}] -> {run_name}")
        print("="*60)

        # 加载模型
        try:
            model = YOLO(str(w_path))
            
            # 使用 Ultralytics 原生的 project 和 name 参数来控制输出目录
            # 输出路径将会是: results/evaluation_plots/<category>/<run_name>
            project_dir = out_root / category
            
            model.val(
                data=data_yaml,
                split=args.split,
                batch=args.batch,
                workers=0,        # 避免 Windows 多进程问题
                plots=True,       # 核心：强制开启画图功能！
                project=str(project_dir),
                name=run_name,
                exist_ok=True,    # 如果文件夹已存在则直接覆盖写入
                verbose=False     # 关闭多余的控制台打印
            )
        except Exception as e:
            print(f"处理 {run_name} 时发生错误: {e}")

    print(f"\n所有图表生成完毕！请前往 {out_root} 查看。")

if __name__ == '__main__':
    main()
