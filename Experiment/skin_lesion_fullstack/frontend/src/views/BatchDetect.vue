<template>
  <div class="batch-page">
    <el-card shadow="hover" class="card-block">
      <template #header>批量上传</template>
      <el-upload
        multiple
        :auto-upload="false"
        :limit="32"
        :show-file-list="true"
        accept=".jpg,.jpeg,.png,.bmp"
        @change="onChange"
      >
        <el-button type="primary">选择多张图片</el-button>
        <template #tip>
          <span class="tip">最多 32 张，单张 ≤ 20 MB</span>
        </template>
      </el-upload>
      <el-row :gutter="16" style="margin-top: 16px">
        <el-col :span="12">
          <div class="slider-row">
            <span>置信度 conf</span>
            <el-slider v-model="conf" :min="0.05" :max="0.95" :step="0.05" show-input />
          </div>
        </el-col>
        <el-col :span="12">
          <div class="slider-row">
            <span>IoU</span>
            <el-slider v-model="iou" :min="0.2" :max="0.9" :step="0.05" show-input />
          </div>
        </el-col>
      </el-row>
      <el-button type="primary" :loading="loading" :disabled="!files.length" @click="runBatch">
        批量检测
      </el-button>
    </el-card>

    <el-row v-if="results.length" :gutter="16" class="row-flex">
      <el-col :span="7" class="thumb-col">
        <el-card shadow="never">
          <template #header>缩略图</template>
          <div class="thumb-list">
            <div
              v-for="(r, i) in results"
              :key="i"
              class="thumb-item"
              :class="{ active: i === activeIndex }"
              @click="activeIndex = i"
            >
              <img v-if="thumbOf(r)" :src="thumbOf(r)" alt="" />
              <div v-else class="thumb-err">{{ r.filename }}</div>
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="17">
        <el-card v-if="activeResult" shadow="never">
          <template #header>{{ activeResult.filename }}</template>
          <ResultCanvas
            v-if="!activeResult.error"
            :src="fullSrc(activeResult)"
            :detections="activeResult.detections || []"
            :box-space-width="activeResult.image_width"
            :box-space-height="activeResult.image_height"
          />
          <el-alert v-else :title="activeResult.error" type="error" show-icon :closable="false" />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ElMessage } from "element-plus";
import { computed, ref } from "vue";
import { predictBatch } from "../api/http";
import ResultCanvas from "../components/ResultCanvas.vue";

const conf = ref(0.25);
const iou = ref(0.45);
const files = ref([]);
const results = ref([]);
const activeIndex = ref(0);
const loading = ref(false);

const activeResult = computed(() => results.value[activeIndex.value]);

function onChange(_file, fileList) {
  const list = fileList.map((f) => f).filter((f) => f.raw);
  files.value = list.filter((f) => {
    const ok = /\.(jpe?g|png|bmp)$/i.test(f.name) || /image\/(jpeg|png|bmp)/.test(f.raw?.type || "");
    const big = (f.raw?.size || 0) <= 20 * 1024 * 1024;
    return ok && big;
  });
}

function thumbOf(r) {
  if (r.error || !r.image_jpeg_base64) return "";
  return `data:image/jpeg;base64,${r.image_jpeg_base64}`;
}

function fullSrc(r) {
  return thumbOf(r);
}

async function runBatch() {
  if (!files.value.length) return;
  loading.value = true;
  try {
    const data = await predictBatch({
      files: files.value.map((f) => f.raw),
      conf: conf.value,
      iou: iou.value,
    });
    results.value = data.results || [];
    activeIndex.value = 0;
    ElMessage.success("批量检测完成");
  } catch (e) {
    ElMessage.error(String(e.message || e));
  } finally {
    loading.value = false;
  }
}
</script>

<style scoped>
.card-block {
  margin-bottom: 20px;
  animation: fadeInUp 0.4s ease-out;
}
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(16px); }
  to { opacity: 1; transform: translateY(0); }
}
.tip {
  margin-left: 12px;
  color: #909399;
  font-size: 12px;
}
.slider-row {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.slider-row span {
  font-size: 13px;
  color: #606266;
  font-weight: 500;
}
.row-flex {
  align-items: flex-start;
}
.thumb-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 70vh;
  overflow: auto;
}
.thumb-item {
  border: 2px solid transparent;
  border-radius: 10px;
  cursor: pointer;
  overflow: hidden;
  background: #f5f7fa;
  transition: all 0.25s ease;
}
.thumb-item:hover {
  border-color: rgba(64, 158, 255, 0.3);
  transform: translateX(2px);
}
.thumb-item.active {
  border-color: #409eff;
  box-shadow: 0 2px 8px rgba(64, 158, 255, 0.2);
}
.thumb-item img {
  width: 100%;
  display: block;
  max-height: 120px;
  object-fit: cover;
}
.thumb-err {
  padding: 8px;
  font-size: 12px;
  color: #f56c6c;
}
</style>
