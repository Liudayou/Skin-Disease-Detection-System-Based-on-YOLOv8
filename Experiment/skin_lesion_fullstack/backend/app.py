"""
皮肤病变智能检测系统 — Flask API（配合 Vue 3 前端）

默认加载 stitch 实验的方案四权重（EMA P4 + width0664 + lowMixup015），可通过 SKIN_DETECTION_MODEL 环境变量覆盖。
启动: cd backend && python app.py
"""

from __future__ import annotations

import base64
import io
import os
import re
import sys
from pathlib import Path

from PIL import Image
from flask import Flask, abort, jsonify, request, Response, send_from_directory
from flask_cors import CORS

from history_store import add_record, delete_record, get_record, init_db, list_records
from inference import CLASS_NAMES_EN, CLASS_NAMES_ZH, SkinDetector
from llm_diagnose import diagnose as llm_diagnose, configure as llm_configure, get_config as llm_get_config, is_configured as llm_is_configured

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
# stitch 实验最优权重 — 方案四：EMA P4 + width0664 + lowMixup015
DEFAULT_WEIGHT = (
    PROJECT_ROOT
    / "weights/stitch/scheme4_ema_p4_width0664_lowMixup015_best.pt"
)
# 备选权重（CBAM 方案一）
CBAM_WEIGHT = (
    PROJECT_ROOT
    / "weights/stitch/yolov8m_w0664_cbam_lowMixup015_best.pt"
)
DATA_DIR = BASE_DIR / "data"
FULLSTACK_ROOT = BASE_DIR.parent
DIST_DIR = (FULLSTACK_ROOT / "frontend" / "dist").resolve()


def _jpeg_thumb_b64(image_bytes: bytes, max_edge: int = 240, quality: int = 82) -> str:
    """生成缩略图的 base64，控制 SQLite 单行体积。"""
    img = Image.open(io.BytesIO(image_bytes))
    # RGBA 转 RGB（白底合成）
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    else:
        img = img.convert("RGB")
    w, h = img.size
    m = max(w, h)
    if m > max_edge:
        s = max_edge / m
        img = img.resize((int(w * s), int(h * s)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


# baseline 权重作为兜底（无 CBAM 自定义算子的版本）
BASELINE_WEIGHT = PROJECT_ROOT / "runs/detect/yolov8m_baseline_ham_univ30_e100/weights/best.pt"


def _register_project_modules() -> None:
    """把仓库内的 EMA / CBAM / ECA 等自定义模块注册到 Ultralytics，否则加载自定义算子权重会报错。"""
    scripts = PROJECT_ROOT / "scripts"
    p = str(scripts)
    if p not in sys.path:
        sys.path.insert(0, p)

    # 注册 EMA（方案四默认模型使用）
    if (scripts / "ema_attention.py").is_file():
        try:
            from ema_attention import register_ema_for_ultralytics
            register_ema_for_ultralytics()
        except Exception:
            pass

    # 注册 CBAM（备选方案一 / 历史权重兼容）
    if (scripts / "cbam_attention.py").is_file():
        try:
            from cbam_attention import register_cbam_for_ultralytics
            register_cbam_for_ultralytics()
        except Exception:
            pass

    # 注册 ECA（备选方案二 / 方案五兼容）
    if (scripts / "eca_attention.py").is_file():
        try:
            from eca_attention import register_eca_for_ultralytics
            register_eca_for_ultralytics()
        except Exception:
            pass


def _weight_candidates() -> list[Path]:
    """按优先级收集候选权重路径：环境变量 > stitch > baseline。"""
    raw = os.environ.get("SKIN_DETECTION_MODEL", "").strip()
    out: list[Path] = []
    if raw:
        out.append(Path(raw).expanduser().resolve())
    out.append(DEFAULT_WEIGHT)
    out.append(BASELINE_WEIGHT)
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def _try_load_detector() -> tuple[SkinDetector, str | None]:
    """逐个尝试加载权重，方案四权重失败时自动降级到 baseline。"""
    msgs: list[str] = []
    fallback: SkinDetector | None = None
    for p in _weight_candidates():
        if not p.is_file():
            msgs.append(f"{p.name}: 文件不存在")
            continue
        d = SkinDetector(p)
        fallback = d
        try:
            d.load()
            return d, None
        except Exception as e:
            msgs.append(f"{p.name}: {e!s}"[:240])
    if fallback is not None:
        return fallback, " | ".join(msgs) if msgs else "模型加载失败"
    d0 = SkinDetector(_weight_candidates()[0])
    return d0, " | ".join(msgs) if msgs else "未找到可用权重文件"


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    _register_project_modules()
    init_db(DATA_DIR)
    detector, load_error = _try_load_detector()
    _retry_failed = bool(load_error)  # 仅首次加载失败时才重试，避免每次请求都尝试

    @app.before_request
    def _ensure_model() -> None:
        nonlocal _retry_failed
        if not detector.is_loaded and _retry_failed:
            try:
                detector.load()
                _retry_failed = False
            except Exception:
                pass

    def _api_only_help_html() -> str:
        port = int(os.environ.get("PORT", "5000"))
        return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/><title>皮肤病变检测 API</title>
<style>body{{font-family:sans-serif;max-width:42rem;margin:2rem auto;padding:0 1rem;line-height:1.6;color:#333}}
code{{background:#f4f4f5;padding:0 .2rem;border-radius:4px}}
a{{color:#409eff}}</style></head><body>
<h1>皮肤病变智能检测 · API 服务</h1>
<p>当前端口 <code>{port}</code> 上<strong>尚未构建前端</strong>（缺少 <code>frontend/dist/index.html</code>）。</p>
<p><strong>任选一种方式打开界面：</strong></p>
<ul>
  <li><strong>单端口（推荐云平台只映射一个端口）：</strong>在服务器执行
    <code>cd frontend && npm install && npm run build</code>，再重启本服务；然后直接访问本页同一端口根路径即可。</li>
  <li><strong>开发双端口：</strong>运行 <code>./start.sh</code> 或 <code>npm run dev</code>，浏览器打开 <code>http://localhost:6006/</code>。</li>
</ul>
<p>若使用 SSH / 端口转发，单端口模式只需映射当前端口。</p>
<ul>
  <li><a href="/api/health">/api/health</a> — 健康检查与模型信息</li>
</ul>
</body></html>"""

    @app.get("/")
    def index():
        """有 dist 就托管 Vue SPA，否则返回引导页。"""
        idx = DIST_DIR / "index.html"
        if idx.is_file():
            return send_from_directory(DIST_DIR, "index.html")
        return Response(_api_only_help_html(), mimetype="text/html; charset=utf-8")

    @app.get("/api/health")
    def health():
        ok = detector.is_loaded
        err = None
        if not ok:
            err = load_error or f"权重未加载: {detector.weights_path}"
        return jsonify(
            {
                "ok": ok,
                "weights": str(detector.weights_path),
                "weights_exists": detector.weights_path.is_file(),
                "classes_en": CLASS_NAMES_EN,
                "classes_zh": CLASS_NAMES_ZH,
                "device": detector.device if ok else None,
                "error": err,
            }
        )

    def _decode_image_payload() -> tuple[bytes, str | None]:
        """兼容 multipart 和 JSON base64 两种图片上传方式。"""
        if request.content_type and "multipart/form-data" in request.content_type:
            f = request.files.get("file")
            if not f or not f.filename:
                raise ValueError("缺少文件字段 file")
            body = f.read()
            if len(body) > 20 * 1024 * 1024:
                raise ValueError("图片过大（>20 MB）")
            return body, f.filename

        data = request.get_json(silent=True) or {}
        b64 = data.get("image_base64") or data.get("image")
        if not b64:
            raise ValueError("缺少 image_base64")
        # 兼容 data:image/xxx;base64,xxx 格式
        if isinstance(b64, str) and "," in b64 and "base64" in b64[:40]:
            b64 = b64.split(",", 1)[1]
        b64 = re.sub(r"\s+", "", str(b64))
        raw = base64.standard_b64decode(b64)
        if len(raw) > 20 * 1024 * 1024:
            raise ValueError("图片过大（>20 MB）")
        return raw, data.get("filename")

    @app.post("/api/predict")
    def predict():
        if not detector.is_loaded:
            return jsonify({"error": f"模型未加载，请检查权重: {detector.weights_path}"}), 503
        try:
            body, filename = _decode_image_payload()
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        conf = float(request.args.get("conf", request.form.get("conf", 0.25)))
        iou = float(request.args.get("iou", request.form.get("iou", 0.45)))
        save_history = request.args.get("save_history", "1") not in ("0", "false", "False")

        # 校验图片可解析
        try:
            _ = Image.open(io.BytesIO(body))
            _.load()
        except Exception:
            return jsonify({"error": "无法解析为有效图像"}), 400

        try:
            out = detector.predict(body, conf=conf, iou=iou, imgsz=640)
        except Exception as e:
            return jsonify({"error": f"推理失败: {e}"}), 500

        # 生成缩略图存入历史记录
        try:
            thumb = _jpeg_thumb_b64(body)
        except Exception:
            thumb = out["image_jpeg_base64"]
        if save_history:
            try:
                rid = add_record(
                    DATA_DIR,
                    filename=filename,
                    conf=conf,
                    iou=iou,
                    image_width=out["image_width"],
                    image_height=out["image_height"],
                    thumb_jpeg_base64=thumb,
                    detections=out["detections"],
                )
                out["history_id"] = rid
            except Exception:
                out["history_id"] = None

        return jsonify(out)

    @app.post("/api/predict/batch")
    def predict_batch():
        if not detector.is_loaded:
            return jsonify({"error": "模型未加载"}), 503
        files = request.files.getlist("files")
        if not files:
            return jsonify({"error": "缺少 files"}), 400
        conf = float(request.form.get("conf", 0.25))
        iou = float(request.form.get("iou", 0.45))
        results = []
        # 上限 32 张，防滥用
        for f in files[:32]:
            if not f.filename:
                continue
            body = f.read()
            if len(body) > 20 * 1024 * 1024:
                results.append({"filename": f.filename, "error": "文件过大"})
                continue
            try:
                one = detector.predict(body, conf=conf, iou=iou, imgsz=640)
                one["filename"] = f.filename
                results.append(one)
            except Exception as e:
                results.append({"filename": f.filename, "error": str(e)})
        return jsonify({"results": results})

    @app.get("/api/history")
    def history_list():
        """分页查询历史记录，支持 page/page_size 参数。"""
        try:
            page = max(1, int(request.args.get("page", 1)))
            page_size = min(100, max(1, int(request.args.get("page_size", 20))))
        except (ValueError, TypeError):
            page, page_size = 1, 20
        offset = (page - 1) * page_size
        result = list_records(DATA_DIR, limit=page_size, offset=offset)
        return jsonify(result)

    @app.get("/api/history/<int:rid>")
    def history_get(rid: int):
        row = get_record(DATA_DIR, rid)
        if not row:
            return jsonify({"error": "not found"}), 404
        return jsonify(row)

    @app.post("/api/diagnose")
    def diagnose_endpoint():
        """基于检测结果调用 LLM 生成初步诊断。"""
        data = request.get_json(silent=True) or {}
        detections = data.get("detections", [])
        image_info = data.get("image_info")
        if not isinstance(detections, list):
            return jsonify({"error": "detections 应为数组"}), 400
        result = llm_diagnose(detections, image_info)
        status = 200 if "error" not in result else 503
        return jsonify(result), status

    @app.get("/api/diagnose/config")
    def diagnose_config():
        """获取当前 LLM 配置（API Key 脱敏）。"""
        cfg = llm_get_config()
        cfg["configured"] = llm_is_configured()
        return jsonify(cfg)

    @app.post("/api/diagnose/config")
    def diagnose_config_update():
        """运行时更新 LLM 配置，无需重启服务。"""
        data = request.get_json(silent=True) or {}
        llm_configure(
            api_base=data.get("api_base") or None,
            api_key=data.get("api_key") or None,
            model=data.get("model") or None,
        )
        return jsonify({"ok": True, **llm_get_config(), "configured": llm_is_configured()})

    @app.delete("/api/history/<int:rid>")
    def history_del(rid: int):
        if delete_record(DATA_DIR, rid):
            return jsonify({"ok": True})
        return jsonify({"error": "not found"}), 404

    @app.get("/<path:subpath>")
    def spa_static(subpath: str):
        """Vue Router history 模式回退：非 /api 路径全部返回 index.html。"""
        if not DIST_DIR.is_dir():
            return jsonify(
                {"error": "frontend not built", "hint": "cd frontend && npm install && npm run build"}
            ), 503
        candidate = (DIST_DIR / subpath).resolve()
        # 防止路径穿越
        try:
            candidate.relative_to(DIST_DIR)
        except ValueError:
            abort(400)
        if candidate.is_file():
            return send_from_directory(DIST_DIR, subpath)
        idx = DIST_DIR / "index.html"
        if idx.is_file():
            return send_from_directory(DIST_DIR, "index.html")
        return jsonify({"error": "frontend not built"}), 503

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
