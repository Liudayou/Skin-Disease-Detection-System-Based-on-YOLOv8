<template>
  <div>
    <el-card shadow="hover">
      <template #header>
        <span>检测历史</span>
        <el-button style="float: right" text type="primary" @click="load">刷新</el-button>
      </template>
      <el-table v-loading="loading" :data="rows" border stripe>
        <el-table-column prop="id" label="ID" width="70" />
        <el-table-column label="时间" width="180">
          <template #default="{ row }">
            {{ formatTime(row.created_at) }}
          </template>
        </el-table-column>
        <el-table-column prop="filename" label="文件名" min-width="140" show-overflow-tooltip />
        <el-table-column prop="num_detections" label="检出数" width="90" />
        <el-table-column label="操作" width="200">
          <template #default="{ row }">
            <el-button link type="primary" @click="openDetail(row)">查看</el-button>
            <el-button link type="danger" @click="remove(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
      <div v-if="total > pageSize" class="pagination-wrap">
        <el-pagination
          v-model:current-page="page"
          :page-size="pageSize"
          :total="total"
          layout="prev, pager, next, total"
          @current-change="load"
        />
      </div>
    </el-card>

    <el-dialog v-model="visible" title="历史记录详情" width="900px" destroy-on-close>
      <div v-if="current">
        <p class="meta">
          conf={{ current.conf }}，iou={{ current.iou }}，尺寸 {{ current.image_width }}×{{
            current.image_height
          }}
        </p>
        <el-row :gutter="12">
          <el-col :span="14">
            <ResultCanvas
              v-if="thumbSrc"
              :src="thumbSrc"
              :detections="current.detections"
              :box-space-width="current.image_width"
              :box-space-height="current.image_height"
            />
          </el-col>
          <el-col :span="10">
            <el-table :data="detTable" size="small" border max-height="420">
              <el-table-column prop="idx" label="#" width="50" />
              <el-table-column prop="name" label="类别" />
              <el-table-column prop="conf" label="置信度" width="90" />
            </el-table>
          </el-col>
        </el-row>
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { ElMessage, ElMessageBox } from "element-plus";
import { computed, onMounted, ref } from "vue";
import { deleteHistory, fetchHistory } from "../api/http";
import ResultCanvas from "../components/ResultCanvas.vue";

const loading = ref(false);
const rows = ref([]);
const visible = ref(false);
const current = ref(null);
const page = ref(1);
const pageSize = ref(20);
const total = ref(0);

const thumbSrc = computed(() => {
  if (!current.value?.thumb_jpeg_base64) return "";
  return `data:image/jpeg;base64,${current.value.thumb_jpeg_base64}`;
});

const detTable = computed(() =>
  (current.value?.detections || []).map((d, i) => ({
    idx: i + 1,
    name: `${d.name_zh} (${d.name_en})`,
    conf: `${(d.confidence * 100).toFixed(1)}%`,
  }))
);

function formatTime(ts) {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  return d.toLocaleString("zh-CN");
}

async function load() {
  loading.value = true;
  try {
    const data = await fetchHistory({ page: page.value, pageSize: pageSize.value });
    rows.value = data.records;
    total.value = data.total;
  } catch (e) {
    ElMessage.error(String(e.message || e));
  } finally {
    loading.value = false;
  }
}

function openDetail(row) {
  current.value = row;
  visible.value = true;
}

async function remove(row) {
  await ElMessageBox.confirm("确定删除该条记录？", "提示", { type: "warning" });
  await deleteHistory(row.id);
  ElMessage.success("已删除");
  await load();
}

onMounted(load);
</script>

<style scoped>
.meta {
  font-size: 13px;
  color: #606266;
  margin-bottom: 12px;
  background: #f5f7fa;
  padding: 10px 14px;
  border-radius: 8px;
}
.pagination-wrap {
  display: flex;
  justify-content: flex-end;
  margin-top: 20px;
  padding-top: 16px;
  border-top: 1px solid #f0f2f5;
}
:deep(.el-table) {
  animation: fadeInUp 0.4s ease-out;
}
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(16px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
