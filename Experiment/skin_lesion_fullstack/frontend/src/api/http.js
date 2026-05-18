import axios from "axios";

const http = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || "",
  timeout: 120000, // 推理可能耗时较长，给 2 分钟超时
});

// ── 健康检查 ──────────────────────────────────

export async function fetchHealth() {
  const { data } = await http.get("/api/health");
  return data;
}

// ── 检测接口 ──────────────────────────────────

/** JSON base64 方式上传图片（兼容旧接口） */
export async function predictImage({ base64, filename, conf, iou }) {
  const { data } = await http.post(
    "/api/predict",
    { image_base64: base64, filename },
    { params: { conf, iou, save_history: 1 } }
  );
  return data;
}

/** FormData 方式上传图片，比 base64 省约 33% 带宽 */
export async function predictImageFile({ file, conf, iou }) {
  const fd = new FormData();
  fd.append("file", file);
  const { data } = await http.post("/api/predict", fd, {
    params: { conf, iou, save_history: 1 },
  });
  return data;
}

/** 批量上传图片检测 */
export async function predictBatch({ files, conf, iou }) {
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f.raw || f));
  fd.append("conf", String(conf));
  fd.append("iou", String(iou));
  const { data } = await http.post("/api/predict/batch", fd);
  return data;
}

// ── 历史记录 ──────────────────────────────────

/** 分页查询历史记录，返回 { records, total } */
export async function fetchHistory({ page = 1, pageSize = 20 } = {}) {
  const { data } = await http.get("/api/history", {
    params: { page, page_size: pageSize },
  });
  return data;
}

export async function deleteHistory(id) {
  await http.delete(`/api/history/${id}`);
}

// ── AI 诊断 ───────────────────────────────────

/** 发送检测结果给 LLM，获取初步诊断 */
export async function diagnose({ detections, imageInfo }) {
  const { data } = await http.post("/api/diagnose", {
    detections,
    image_info: imageInfo,
  });
  return data;
}

/** 获取当前 LLM 配置（API Key 脱敏） */
export async function getDiagnoseConfig() {
  const { data } = await http.get("/api/diagnose/config");
  return data;
}

/** 更新 LLM 配置 */
export async function updateDiagnoseConfig({ apiKey, apiBase, model }) {
  const { data } = await http.post("/api/diagnose/config", {
    api_key: apiKey,
    api_base: apiBase,
    model,
  });
  return data;
}
