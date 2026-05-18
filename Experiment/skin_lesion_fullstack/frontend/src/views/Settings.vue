<template>
  <div>
    <el-row :gutter="16">
      <el-col :xs="24" :md="12">
        <el-card shadow="hover">
          <template #header>服务状态</template>
          <el-descriptions :column="1" border size="small">
            <el-descriptions-item label="API 可达">
              <el-tag :type="health ? 'success' : 'danger'">{{ health ? "是" : "否" }}</el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="模型已加载">
              <el-tag :type="health?.ok ? 'success' : 'danger'">{{ health?.ok ? "是" : "否" }}</el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="权重路径">
              <span class="path">{{ health?.weights || "—" }}</span>
            </el-descriptions-item>
            <el-descriptions-item label="推理设备">
              {{ health?.device || "—" }}
            </el-descriptions-item>
          </el-descriptions>
          <el-alert
            v-if="health && !health.ok"
            style="margin-top: 12px"
            :title="health.error || '请设置环境变量 SKIN_DETECTION_MODEL 指向 best.pt'"
            type="warning"
            show-icon
            :closable="false"
          />
          <el-button style="margin-top: 12px" text type="primary" @click="refreshHealth">刷新状态</el-button>
        </el-card>
      </el-col>
      <el-col :xs="24" :md="12">
        <el-card shadow="hover">
          <template #header>默认检测参数</template>
          <p class="hint">设置保存在浏览器本地（localStorage），下次打开自动加载。</p>
          <el-form label-width="100px">
            <el-form-item label="置信度 conf">
              <el-slider v-model="settings.conf" :min="0.05" :max="0.95" :step="0.05" show-input />
            </el-form-item>
            <el-form-item label="IoU 阈值">
              <el-slider v-model="settings.iou" :min="0.2" :max="0.9" :step="0.05" show-input />
            </el-form-item>
            <el-form-item label="输入尺寸">
              <el-select v-model="settings.imgsz">
                <el-option :value="320" label="320" />
                <el-option :value="640" label="640（推荐）" />
                <el-option :value="1280" label="1280" />
              </el-select>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" @click="saveSettings">保存设置</el-button>
              <el-button @click="resetSettings">恢复默认</el-button>
            </el-form-item>
          </el-form>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="16" style="margin-top: 16px">
      <el-col :span="24">
        <el-card shadow="hover">
          <template #header>
            <div class="card-header-row">
              <span>AI 诊断 API 配置</span>
              <el-tag v-if="llmConfig.configured" type="success" size="small">已配置</el-tag>
              <el-tag v-else type="info" size="small">未配置</el-tag>
            </div>
          </template>
          <p class="hint">
            配置 OpenAI 兼容的 Chat Completions API，用于在检测后提供 AI 初步诊断参考。
            支持 DeepSeek、通义千问、MiMo 等兼容 OpenAI 协议的服务。
          </p>
          <el-form label-width="120px">
            <el-form-item label="API Base URL">
              <el-input
                v-model="llmForm.apiBase"
                placeholder="https://token-plan-cn.xiaomimimo.com/v1"
                clearable
              />
            </el-form-item>
            <el-form-item label="API Key">
              <el-input
                v-model="llmForm.apiKey"
                type="password"
                show-password
                placeholder="sk-..."
                clearable
              />
              <span v-if="llmConfig.api_key_preview" class="key-preview">
                当前：{{ llmConfig.api_key_preview }}
              </span>
            </el-form-item>
            <el-form-item label="模型名">
              <el-input
                v-model="llmForm.model"
                placeholder="mimo-v2.5-pro"
                clearable
              />
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :loading="llmSaving" @click="saveLlmConfig">
                保存 API 配置
              </el-button>
              <el-button @click="loadLlmConfig">刷新状态</el-button>
            </el-form-item>
          </el-form>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ElMessage } from "element-plus";
import { onMounted, reactive, ref } from "vue";
import { fetchHealth, getDiagnoseConfig, updateDiagnoseConfig } from "../api/http";

const STORAGE_KEY = "skin_detect_settings";

const DEFAULTS = { conf: 0.25, iou: 0.45, imgsz: 640 };

const health = ref(null);
const settings = reactive({ ...DEFAULTS });

// LLM config state
const llmConfig = reactive({
  configured: false,
  api_base: "",
  api_key_set: false,
  api_key_preview: "",
  model: "",
});
const llmForm = reactive({
  apiBase: "",
  apiKey: "",
  model: "",
});
const llmSaving = ref(false);

function loadSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    if (saved.conf != null) settings.conf = saved.conf;
    if (saved.iou != null) settings.iou = saved.iou;
    if (saved.imgsz != null) settings.imgsz = saved.imgsz;
  } catch {
    /* ignore */
  }
}

function saveSettings() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  ElMessage.success("设置已保存");
}

function resetSettings() {
  Object.assign(settings, DEFAULTS);
  localStorage.removeItem(STORAGE_KEY);
  ElMessage.success("已恢复默认值");
}

async function refreshHealth() {
  try {
    health.value = await fetchHealth();
  } catch {
    health.value = null;
  }
}

async function loadLlmConfig() {
  try {
    const cfg = await getDiagnoseConfig();
    Object.assign(llmConfig, cfg);
    // Pre-fill form with current values
    if (cfg.api_base) llmForm.apiBase = cfg.api_base;
    if (cfg.model) llmForm.model = cfg.model;
    // Don't pre-fill API key (masked)
  } catch {
    /* ignore */
  }
}

async function saveLlmConfig() {
  if (!llmForm.apiKey && !llmConfig.api_key_set) {
    ElMessage.warning("请输入 API Key");
    return;
  }
  llmSaving.value = true;
  try {
    const result = await updateDiagnoseConfig({
      apiKey: llmForm.apiKey || undefined,
      apiBase: llmForm.apiBase || undefined,
      model: llmForm.model || undefined,
    });
    Object.assign(llmConfig, result);
    llmForm.apiKey = ""; // Clear for security
    ElMessage.success("AI 诊断 API 配置已保存");
  } catch (e) {
    ElMessage.error("保存失败：" + String(e.message || e));
  } finally {
    llmSaving.value = false;
  }
}

onMounted(() => {
  loadSettings();
  refreshHealth();
  loadLlmConfig();
});
</script>

<style scoped>
.path {
  word-break: break-all;
  font-size: 12px;
}
.hint {
  color: #909399;
  font-size: 13px;
  margin: 0 0 14px;
  line-height: 1.6;
}
.card-header-row {
  display: flex;
  align-items: center;
  gap: 10px;
}
.key-preview {
  font-size: 12px;
  color: #909399;
  margin-top: 4px;
  display: block;
}
:deep(.el-card) {
  animation: fadeInUp 0.4s ease-out;
}
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(16px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
