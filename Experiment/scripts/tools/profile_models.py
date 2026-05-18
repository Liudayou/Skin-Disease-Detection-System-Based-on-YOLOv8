import sys
from pathlib import Path
sys.path.insert(0, str(Path("Experiment/scripts").resolve()))
try:
    from modules.cbam_attention import register_cbam_for_ultralytics
    register_cbam_for_ultralytics()
except: pass
try:
    from modules.eca_attention import register_eca_for_ultralytics
    register_eca_for_ultralytics()
except: pass
try:
    from modules.ema_attention import register_ema_for_ultralytics
    register_ema_for_ultralytics()
except: pass

from ultralytics import YOLO
import os

os.environ["OMP_NUM_THREADS"] = "1"
import torch

for m in ["yolov8m_cbam_neck.yaml", "yolov8m_eca_neck.yaml", "yolov8m_ema_neck.yaml"]:
    try:
        print(f"--- {m} ---")
        model = YOLO(f"Experiment/scripts/models/{m}")
        model.info(detailed=False)
        print("--------------")
    except Exception as e:
        print(f"Error {m}: {e}")
