"""基于 LLM 的皮肤病变初步诊断模块。

调用 OpenAI 兼容的 Chat Completions API，基于 YOLOv8 检测结果生成结构化医学参考信息。
支持通过环境变量或 Settings 页面运行时配置 API 地址、Key、模型名。
"""

from __future__ import annotations

import json
import os
from typing import Any

import requests

# 默认配置 — MiMo v2.5 Pro（可通过环境变量或运行时 API 覆盖）
_API_BASE = os.environ.get("DIAGNOSIS_API_BASE", "https://token-plan-cn.xiaomimimo.com/v1")
_API_KEY = os.environ.get("DIAGNOSIS_API_KEY", "")
_MODEL = os.environ.get("DIAGNOSIS_MODEL", "mimo-v2.5-pro")
_TIMEOUT = 60

# 运行时覆盖（通过 Settings 页面设置）
_runtime_config: dict[str, str] = {}


def configure(*, api_base: str | None = None, api_key: str | None = None, model: str | None = None) -> None:
    """运行时更新配置，无需重启服务。"""
    if api_base is not None:
        _runtime_config["api_base"] = api_base.rstrip("/")
    if api_key is not None:
        _runtime_config["api_key"] = api_key
    if model is not None:
        _runtime_config["model"] = model


def get_config() -> dict[str, Any]:
    """返回当前配置（API Key 脱敏显示）。"""
    base = _runtime_config.get("api_base", _API_BASE)
    key = _runtime_config.get("api_key", _API_KEY)
    model = _runtime_config.get("model", _MODEL)
    masked_key = key[:8] + "..." + key[-4:] if len(key) > 12 else ("***" if key else "")
    return {
        "api_base": base,
        "api_key_set": bool(key),
        "api_key_preview": masked_key,
        "model": model,
    }


def _get_api_base() -> str:
    return _runtime_config.get("api_base", _API_BASE)

def _get_api_key() -> str:
    return _runtime_config.get("api_key", _API_KEY)

def _get_model() -> str:
    return _runtime_config.get("model", _MODEL)

def is_configured() -> bool:
    return bool(_get_api_key())


# 系统提示词 — 引导 LLM 输出结构化 JSON 格式的医学参考
SYSTEM_PROMPT = """你是一位专业的皮肤科医学助手。用户上传了一张皮肤镜/皮损照片，YOLOv8 目标检测模型已给出了初步检测结果。

你的任务是基于检测结果，提供一份**结构化的医学参考信息**。请注意：

1. 你不是执业医师，不能做出诊断结论
2. 你的输出是**辅助参考**，帮助用户了解可能的情况
3. 务必建议用户前往正规医疗机构就诊

请用以下 JSON 格式回复（直接返回 JSON，不要包裹在 markdown 代码块中）：

{
  "summary": "一句话概括检测结果",
  "risk_level": "低/中/高/需紧急关注",
  "lesion_analysis": [
    {
      "name_zh": "中文名",
      "name_en": "英文缩写",
      "description": "该病变的简要说明（50字以内）",
      "typical_features": "典型特征",
      "risk_factors": "风险因素",
      "urgency": "建议就诊紧迫程度"
    }
  ],
  "recommendations": ["建议1", "建议2", "..."],
  "suggested_department": "建议就诊科室",
  "disclaimer": "免责声明"
}"""


def diagnose(detections: list[dict[str, Any]], image_info: dict[str, Any] | None = None) -> dict[str, Any]:
    """基于检测结果调用 LLM 生成初步诊断。

    Args:
        detections: inference.py 返回的 detections 列表
        image_info: 可选，包含 image_width, image_height 等

    Returns:
        解析后的 JSON dict，或含 error 字段的错误 dict。
    """
    api_key = _get_api_key()
    if not api_key:
        return {"error": "未配置 API Key。请在系统设置中配置诊断 API。"}

    # 构造用户消息
    det_summary = []
    for d in detections:
        det_summary.append(
            f"- {d.get('name_zh', '')} ({d.get('name_en', '')}), "
            f"置信度: {d.get('confidence', 0):.1%}, "
            f"位置: [{', '.join(f'{x:.0f}' for x in d.get('xyxy', [0,0,0,0]))}]"
        )

    if not det_summary:
        user_msg = "检测结果：未检出任何病灶。请给出一般性皮肤护理建议。"
    else:
        user_msg = "检测结果：\n" + "\n".join(det_summary)
        if image_info:
            user_msg += f"\n\n图像尺寸: {image_info.get('image_width', '?')}×{image_info.get('image_height', '?')}"

    api_base = _get_api_base()
    model = _get_model()
    url = f"{api_base}/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.3,
        "max_tokens": 1500,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=_TIMEOUT)
        if resp.status_code != 200:
            # 返回 API 的详细错误信息，方便前端排查
            try:
                err_body = resp.json()
                err_msg = err_body.get("error", {}).get("message", "") or str(err_body)
            except Exception:
                err_msg = resp.text[:300]
            return {"error": f"API 返回 {resp.status_code}: {err_msg}"}
        data = resp.json()

        content = data["choices"][0]["message"]["content"].strip()

        # 解析 JSON — 有些模型会用 markdown 代码块包裹
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            if "```" in content:
                import re
                m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
                if m:
                    result = json.loads(m.group(1).strip())
                else:
                    result = {"summary": content, "risk_level": "未知", "disclaimer": "AI 返回格式异常"}
            else:
                result = {"summary": content, "risk_level": "未知", "disclaimer": "AI 返回格式异常"}

        result["model_used"] = model
        return result

    except requests.exceptions.Timeout:
        return {"error": "AI 诊断请求超时，请稍后重试。"}
    except requests.exceptions.RequestException as e:
        return {"error": f"AI 诊断请求失败: {str(e)[:200]}"}
    except (KeyError, IndexError) as e:
        return {"error": f"AI 返回格式异常: {str(e)[:200]}"}
