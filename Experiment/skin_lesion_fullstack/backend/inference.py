"""YOLOv8 皮肤病变推理模块 — 支持 HAM10000 七分类检测。"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from threading import Lock
from typing import Any

import torch
from PIL import Image

# HAM10000 七类病变名称映射
CLASS_NAMES_EN = {
    0: "akiec",
    1: "bcc",
    2: "bkl",
    3: "df",
    4: "mel",
    5: "nv",
    6: "vasc",
}

CLASS_NAMES_ZH = {
    0: "光化性角化病 (akiec)",
    1: "基底细胞癌 (bcc)",
    2: "脂溢性角化病 (bkl)",
    3: "皮肤纤维瘤 (df)",
    4: "黑色素瘤 (mel)",
    5: "色素痣 (nv)",
    6: "血管病变 (vasc)",
}


def _pil_to_jpeg_b64(img: Image.Image, quality: int = 92) -> str:
    """PIL 图片转 JPEG base64 字符串。"""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


class SkinDetector:
    """线程安全的 YOLOv8 推理封装，支持 CUDA/CPU 自动切换。"""

    def __init__(self, weights_path: Path) -> None:
        self._weights = weights_path.resolve()
        self._model = None
        self._lock = Lock()
        self._device: str | None = None

    def load(self) -> None:
        from ultralytics import YOLO

        if not self._weights.is_file():
            raise FileNotFoundError(f"权重文件不存在: {self._weights}")
        with self._lock:
            self._model = YOLO(str(self._weights))
            # 优先 GPU 推理，无 CUDA 时回退 CPU
            if torch.cuda.is_available():
                self._device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                print(f"[推理] 使用 GPU: {gpu_name}")
            else:
                self._device = "cpu"
                print("[推理] CUDA 不可用，回退到 CPU 推理")

    def reload(self, weights_path: Path | None = None) -> None:
        if weights_path is not None:
            self._weights = weights_path.resolve()
        self.load()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def weights_path(self) -> Path:
        return self._weights

    @property
    def device(self) -> str:
        return self._device or "cpu"

    def predict(
        self,
        image_bytes: bytes,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        half: bool | None = None,
        return_server_plot: bool = False,
    ) -> dict[str, Any]:
        """单张图片推理，返回检测结果和 JPEG base64。

        Args:
            conf: 置信度阈值
            iou: NMS 的 IoU 阈值
            imgsz: 推理输入尺寸（与训练一致为 640）
            half: 半精度推理（CUDA 下默认开启，省显存）
            return_server_plot: 是否额外返回服务端绘制的 PNG（含标注框）
        """
        if self._model is None:
            self.load()

        raw = Image.open(io.BytesIO(image_bytes))
        # RGBA → RGB 白底合成
        if raw.mode == "RGBA":
            bg = Image.new("RGB", raw.size, (255, 255, 255))
            bg.paste(raw, mask=raw.split()[3])
            img = bg
        else:
            img = raw.convert("RGB")

        w0, h0 = img.size
        use_half = half if half is not None else (self._device == "cuda")
        kwargs: dict[str, Any] = {
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "verbose": False,
            "device": self._device,
        }
        if use_half and self._device == "cuda":
            kwargs["half"] = True

        with self._lock:
            results = self._model.predict(source=img, **kwargs)
        r = results[0]

        image_jpeg_b64 = _pil_to_jpeg_b64(img)

        names = r.names if isinstance(r.names, dict) else {i: n for i, n in enumerate(r.names)}
        dets: list[dict[str, Any]] = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(r.boxes)):
                ci = int(clss[i])
                dets.append(
                    {
                        "class_id": ci,
                        "name_en": str(names.get(ci, CLASS_NAMES_EN.get(ci, str(ci)))),
                        "name_zh": CLASS_NAMES_ZH.get(ci, CLASS_NAMES_EN.get(ci, str(ci))),
                        "confidence": float(confs[i]),
                        "xyxy": [float(x) for x in xyxy[i].tolist()],
                    }
                )

        out: dict[str, Any] = {
            "image_jpeg_base64": image_jpeg_b64,
            "image_width": w0,
            "image_height": h0,
            "detections": dets,
            "model_path": str(self._weights),
            "device": self._device,
            "half": bool(use_half and self._device == "cuda"),
        }

        # 服务端绘图（含检测框标注），用于 web_skin_detection 简版页面
        if return_server_plot:
            import cv2
            plotted_bgr = r.plot()
            plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)
            out_pil = Image.fromarray(plotted_rgb)
            buf = io.BytesIO()
            out_pil.save(buf, format="PNG")
            out["image_png_base64"] = base64.standard_b64encode(buf.getvalue()).decode("ascii")

        return out
