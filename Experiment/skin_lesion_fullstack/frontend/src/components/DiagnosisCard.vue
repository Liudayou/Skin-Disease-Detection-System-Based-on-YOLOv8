<template>
  <el-card shadow="hover" class="diagnosis-card">
    <template #header>
      <div class="card-header">
        <span>AI 初步诊断</span>
        <el-button
          v-if="!loading && !result"
          type="primary"
          size="small"
          :disabled="(!detections.length && !allowEmpty) || !apiConfigured"
          @click="$emit('diagnose')"
        >
          {{ apiConfigured ? '获取诊断' : '请先配置 API' }}
        </el-button>
      </div>
    </template>

    <!-- Not configured warning -->
    <el-alert
      v-if="!apiConfigured"
      title="未配置诊断 API"
      description="请前往「系统设置」页面配置 AI 诊断 API 的 Key 和地址。"
      type="warning"
      show-icon
      :closable="false"
    />

    <!-- Loading state -->
    <div v-else-if="loading" class="loading-wrap">
      <el-icon class="is-loading" :size="28"><i class="el-icon-loading" /></el-icon>
      <span>AI 正在分析检测结果，请稍候…</span>
    </div>

    <!-- Error state -->
    <el-alert
      v-else-if="error"
      :title="error"
      type="error"
      show-icon
      :closable="false"
    >
      <el-button size="small" @click="$emit('diagnose')" style="margin-top: 8px">重试</el-button>
    </el-alert>

    <!-- Result -->
    <div v-else-if="result" class="diagnosis-content">
      <!-- Summary -->
      <div class="summary-line">
        <el-tag :type="riskTagType" size="large" effect="dark">
          风险等级：{{ result.risk_level || '未知' }}
        </el-tag>
      </div>

      <p class="summary-text">{{ result.summary }}</p>

      <!-- Lesion analysis -->
      <div v-if="result.lesion_analysis?.length" class="section">
        <h4>病灶分析</h4>
        <div v-for="(lesion, i) in result.lesion_analysis" :key="i" class="lesion-item">
          <div class="lesion-header">
            <el-tag size="small" type="info">{{ lesion.name_en }}</el-tag>
            <span class="lesion-name">{{ lesion.name_zh }}</span>
          </div>
          <p class="lesion-desc">{{ lesion.description }}</p>
          <el-descriptions :column="1" size="small" border class="lesion-detail">
            <el-descriptions-item label="典型特征">{{ lesion.typical_features }}</el-descriptions-item>
            <el-descriptions-item label="风险因素">{{ lesion.risk_factors }}</el-descriptions-item>
            <el-descriptions-item label="就诊紧迫程度">
              <span :class="urgencyClass(lesion.urgency)">{{ lesion.urgency }}</span>
            </el-descriptions-item>
          </el-descriptions>
        </div>
      </div>

      <!-- Recommendations -->
      <div v-if="result.recommendations?.length" class="section">
        <h4>建议</h4>
        <ul class="rec-list">
          <li v-for="(rec, i) in result.recommendations" :key="i">{{ rec }}</li>
        </ul>
      </div>

      <!-- Department -->
      <div v-if="result.suggested_department" class="section">
        <el-alert
          :title="`建议就诊科室：${result.suggested_department}`"
          type="info"
          show-icon
          :closable="false"
        />
      </div>

      <!-- Disclaimer -->
      <div class="disclaimer">
        <el-alert
          :title="result.disclaimer || '本诊断仅供参考，不构成医疗建议。请以执业医师的诊断为准。'"
          type="warning"
          show-icon
          :closable="false"
        />
      </div>

      <p v-if="result.model_used" class="model-info">模型：{{ result.model_used }}</p>
    </div>
  </el-card>
</template>

<script setup>
/**
 * AI 初步诊断卡片
 * 接收检测结果 → 调用 LLM → 展示结构化医学参考（风险等级、病灶分析、建议等）
 */
import { computed } from "vue";

const props = defineProps({
  detections: { type: Array, default: () => [] },
  loading: { type: Boolean, default: false },
  result: { type: Object, default: null },    // LLM 返回的结构化诊断
  error: { type: String, default: "" },
  apiConfigured: { type: Boolean, default: false },  // 后端是否已配置 API Key
  allowEmpty: { type: Boolean, default: false },     // 无检测结果时是否允许诊断（给一般建议）
});

defineEmits(["diagnose"]);

// 风险等级 → 标签颜色映射
const riskTagType = computed(() => {
  const level = props.result?.risk_level || "";
  if (level.includes("紧急") || level.includes("高")) return "danger";
  if (level.includes("中")) return "warning";
  return "success";
});

// 就诊紧迫程度 → 文字颜色 class
function urgencyClass(urgency) {
  if (!urgency) return "";
  if (urgency.includes("紧急") || urgency.includes("立即")) return "urgency-high";
  if (urgency.includes("尽快") || urgency.includes("高")) return "urgency-medium";
  return "urgency-low";
}
</script>

<style scoped>
.diagnosis-card {
  margin-bottom: 20px;
  border: 1px solid rgba(64, 158, 255, 0.1) !important;
  background: linear-gradient(135deg, rgba(64, 158, 255, 0.02), rgba(255, 255, 255, 1));
}
.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.loading-wrap {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 24px 0;
  color: #909399;
  font-size: 14px;
}
.summary-line {
  margin-bottom: 12px;
}
.summary-text {
  font-size: 15px;
  color: #303133;
  margin: 0 0 16px;
  line-height: 1.6;
}
.section {
  margin-bottom: 16px;
}
.section h4 {
  font-size: 14px;
  font-weight: 600;
  color: #303133;
  margin: 0 0 8px;
  padding-bottom: 4px;
  border-bottom: 1px solid #ebeef5;
}
.lesion-item {
  background: linear-gradient(135deg, #f8faff, #f0f4ff);
  border-radius: 10px;
  padding: 14px;
  margin-bottom: 10px;
  border: 1px solid rgba(64, 158, 255, 0.08);
}
.lesion-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}
.lesion-name {
  font-weight: 600;
  font-size: 14px;
}
.lesion-desc {
  font-size: 13px;
  color: #606266;
  margin: 0 0 8px;
}
.lesion-detail {
  margin-top: 4px;
}
.rec-list {
  margin: 0;
  padding-left: 20px;
  font-size: 13px;
  color: #303133;
  line-height: 2;
}
.rec-list li::marker {
  color: #409eff;
}
.disclaimer {
  margin-top: 12px;
}
.model-info {
  font-size: 11px;
  color: #c0c4cc;
  text-align: right;
  margin: 8px 0 0;
}
.urgency-high {
  color: #f56c6c;
  font-weight: 600;
}
.urgency-medium {
  color: #e6a23c;
  font-weight: 600;
}
.urgency-low {
  color: #67c23a;
}
</style>
