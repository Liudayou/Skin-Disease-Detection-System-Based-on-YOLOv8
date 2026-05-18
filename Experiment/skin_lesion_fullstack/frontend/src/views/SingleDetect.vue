<template>
  <div>
    <p class="medical-disclaimer">
      声明：本系统仅供科研与工程验证，不能替代执业医师诊断。检测结果不构成医疗建议。
    </p>

    <!-- 图片上传区 -->
    <el-card shadow="hover" class="card-block">
      <template #header>图像上传</template>
      <el-upload
        class="upload"
        drag
        :auto-upload="false"
        :show-file-list="false"
        accept=".jpg,.jpeg,.png,.bmp,image/jpeg,image/png,image/bmp"
        @change="onFile"
      >
        <div class="el-icon--upload upload-ico">
          <svg width="52" height="52" viewBox="0 0 52 52" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect width="52" height="52" rx="12" fill="#409eff" fill-opacity="0.1"/>
            <path d="M26 14L18 24H22V36H30V24H34L26 14Z" fill="#409eff"/>
          </svg>
        </div>
        <div class="el-upload__text">将皮肤镜 / 皮损照片拖到此处，或 <em>点击上传</em></div>
        <template #tip>
          <div class="el-upload__tip">支持 JPG、PNG、BMP，单文件不超过 20 MB</div>
        </template>
      </el-upload>
    </el-card>

    <!-- 参数滑块 + 运行按钮 -->
    <el-card shadow="hover" class="card-block">
      <template #header>参数配置</template>
      <el-row :gutter="24" align="middle">
        <el-col :xs="24" :md="12">
          <div class="slider-row">
            <span>置信度阈值 conf</span>
            <el-slider v-model="conf" :min="0.05" :max="0.95" :step="0.05" show-input />
          </div>
        </el-col>
        <el-col :xs="24" :md="12">
          <div class="slider-row">
            <span>IoU 阈值（NMS）</span>
            <el-slider v-model="iou" :min="0.2" :max="0.9" :step="0.05" show-input />
          </div>
        </el-col>
      </el-row>
      <el-button type="primary" :loading="loading" :disabled="!rawFile" @click="run">
        运行检测
      </el-button>
    </el-card>

    <!-- 原图 vs 检测结果并排展示 -->
    <el-row v-if="originalSrc" :gutter="16" class="card-block">
      <el-col :xs="24" :md="12">
        <el-card shadow="never">
          <template #header>原始图像</template>
          <img :src="originalSrc" class="plain-img" alt="original" />
        </el-card>
      </el-col>
      <el-col :xs="24" :md="12">
        <el-card shadow="never">
          <template #header>检测结果（Canvas 可视化）</template>
          <ResultCanvas
            :src="originalSrc"
            :detections="detections"
            :box-space-width="imgW"
            :box-space-height="imgH"
          />
        </el-card>
      </el-col>
    </el-row>

    <!-- 检测结果表格 -->
    <el-card v-if="detections.length || ranOnce" shadow="hover" class="card-block">
      <template #header>结果详情</template>
      <el-table :data="tableRows" border stripe empty-text="未检出目标（可尝试降低置信度阈值）">
        <el-table-column prop="idx" label="序号" width="70" />
        <el-table-column prop="name" label="类别" min-width="160" />
        <el-table-column prop="conf" label="置信度" width="120" />
        <el-table-column prop="box" label="位置坐标 (x1,y1,x2,y2)" min-width="220" />
      </el-table>
    </el-card>

    <!-- AI 初步诊断卡片（检测完成后出现） -->
    <DiagnosisCard
      v-if="ranOnce"
      :detections="detections"
      :loading="diagnoseLoading"
      :result="diagnoseResult"
      :error="diagnoseError"
      :api-configured="apiConfigured"
      :allow-empty="true"
      @diagnose="runDiagnose"
    />
  </div>
</template>

<script setup>
import { ElMessage } from "element-plus";
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { predictImageFile, diagnose, getDiagnoseConfig } from "../api/http";
import ResultCanvas from "../components/ResultCanvas.vue";
import DiagnosisCard from "../components/DiagnosisCard.vue";

// ── 检测参数 ──────────────────────────────────
const conf = ref(0.25);
const iou = ref(0.45);
const loading = ref(false);
const rawFile = ref(null);       // 原始 File 对象，FormData 上传用
const originalSrc = ref("");     // 预览图 URL（blob 或 base64）
const imgW = ref(0);
const imgH = ref(0);
const detections = ref([]);
const ranOnce = ref(false);      // 标记是否至少运行过一次检测

// ── 诊断状态 ──────────────────────────────────
const apiConfigured = ref(false);
const diagnoseLoading = ref(false);
const diagnoseResult = ref(null);
const diagnoseError = ref("");

const tableRows = computed(() =>
  detections.value.map((d, i) => ({
    idx: i + 1,
    name: `${d.name_zh} (${d.name_en})`,
    conf: `${(d.confidence * 100).toFixed(1)}%`,
    box: d.xyxy.map((x) => x.toFixed(1)).join(", "),
  }))
);

/** 释放 blob URL，防止内存泄漏 */
function revokeBlobUrl() {
  if (originalSrc.value && originalSrc.value.startsWith("blob:")) {
    URL.revokeObjectURL(originalSrc.value);
  }
}

function onFile(uploadFile) {
  const file = uploadFile.raw;
  if (!file) return;
  const okType = ["image/jpeg", "image/png", "image/bmp"].includes(file.type);
  const okExt = /\.(jpe?g|png|bmp)$/i.test(file.name);
  if (!okType && !okExt) {
    ElMessage.error("仅支持 JPG、PNG、BMP 格式");
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    ElMessage.error("文件大小不能超过 20 MB");
    return;
  }
  revokeBlobUrl();
  rawFile.value = file;
  originalSrc.value = URL.createObjectURL(file);
  // 获取原图尺寸供检测框坐标映射
  const im = new Image();
  im.onload = () => {
    imgW.value = im.naturalWidth;
    imgH.value = im.naturalHeight;
  };
  im.src = originalSrc.value;
  detections.value = [];
  ranOnce.value = false;
  diagnoseResult.value = null;
  diagnoseError.value = "";
}

/** FormData 方式上传图片检测 */
async function run() {
  if (!rawFile.value) {
    ElMessage.warning("请先上传图像");
    return;
  }
  loading.value = true;
  ranOnce.value = true;
  diagnoseResult.value = null;
  diagnoseError.value = "";
  try {
    const data = await predictImageFile({
      file: rawFile.value,
      conf: conf.value,
      iou: iou.value,
    });
    if (data.error) throw new Error(data.error);
    detections.value = data.detections || [];
    // 后端返回的是服务端处理后的 JPEG，替换预览图
    if (data.image_jpeg_base64) {
      revokeBlobUrl();
      originalSrc.value = `data:image/jpeg;base64,${data.image_jpeg_base64}`;
    }
    imgW.value = data.image_width || imgW.value;
    imgH.value = data.image_height || imgH.value;
    ElMessage.success("检测完成");
  } catch (e) {
    ElMessage.error(String(e.message || e));
  } finally {
    loading.value = false;
  }
}

/** 调用 LLM 获取初步诊断 */
async function runDiagnose() {
  // 每次点击都重新检查配置（用户可能刚在设置页配好 Key 回来）
  if (!apiConfigured.value) {
    try {
      const cfg = await getDiagnoseConfig();
      apiConfigured.value = cfg.configured;
    } catch { /* ignore */ }
  }
  if (!apiConfigured.value) {
    diagnoseError.value = "未配置 API Key。请前往「系统设置」页面配置诊断 API。";
    return;
  }
  diagnoseLoading.value = true;
  diagnoseError.value = "";
  diagnoseResult.value = null;
  try {
    const result = await diagnose({
      detections: detections.value,
      imageInfo: { image_width: imgW.value, image_height: imgH.value },
    });
    if (result.error) {
      diagnoseError.value = result.error;
    } else {
      diagnoseResult.value = result;
    }
  } catch (e) {
    // axios 错误时提取后端返回的具体信息
    diagnoseError.value = e.response?.data?.error || e.message || String(e);
  } finally {
    diagnoseLoading.value = false;
  }
}

onMounted(async () => {
  try {
    const cfg = await getDiagnoseConfig();
    apiConfigured.value = cfg.configured;
  } catch {
    apiConfigured.value = false;
  }
});

onBeforeUnmount(() => {
  revokeBlobUrl();
});
</script>

<style scoped>
.card-block {
  margin-bottom: 20px;
  animation: fadeInUp 0.4s ease-out;
}
.card-block:nth-child(2) { animation-delay: 0.05s; }
.card-block:nth-child(3) { animation-delay: 0.1s; }
.card-block:nth-child(4) { animation-delay: 0.15s; }
.card-block:nth-child(5) { animation-delay: 0.2s; }

@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(16px); }
  to { opacity: 1; transform: translateY(0); }
}

.upload { width: 100%; }

.slider-row {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 12px;
}
.slider-row span {
  font-size: 13px;
  color: #606266;
  font-weight: 500;
}

.upload-ico {
  font-size: 52px;
  color: #409eff;
  margin-bottom: 12px;
  filter: drop-shadow(0 4px 8px rgba(64, 158, 255, 0.2));
}

.plain-img {
  max-width: 100%;
  border-radius: 10px;
  display: block;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
}
</style>
