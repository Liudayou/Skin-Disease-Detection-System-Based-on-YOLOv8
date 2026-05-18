<template>
  <el-container class="layout-root">
    <el-aside width="240px" class="side">
      <div class="brand">
        <div class="brand-icon">
          <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect width="32" height="32" rx="8" fill="url(#grad)"/>
            <path d="M16 8C14 8 12 10 12 12C12 14 14 16 14 18H18C18 16 20 14 20 12C20 10 18 8 16 8Z" fill="white" opacity="0.9"/>
            <rect x="13" y="19" width="6" height="2" rx="1" fill="white" opacity="0.7"/>
            <rect x="14.5" y="22" width="3" height="4" rx="1.5" fill="white" opacity="0.7"/>
            <defs>
              <linearGradient id="grad" x1="0" y1="0" x2="32" y2="32">
                <stop stop-color="#409eff"/>
                <stop offset="1" stop-color="#3a7bd5"/>
              </linearGradient>
            </defs>
          </svg>
        </div>
        <div class="brand-text">
          <div class="brand-title">皮肤病变检测</div>
          <div class="brand-sub">智能诊断系统</div>
        </div>
      </div>
      <el-menu
        :default-active="active"
        router
        class="side-menu"
        background-color="transparent"
        text-color="#606266"
        active-text-color="#409eff"
      >
        <el-menu-item index="/single">
          <span>单张检测</span>
        </el-menu-item>
        <el-menu-item index="/batch">
          <span>批量检测</span>
        </el-menu-item>
        <el-menu-item index="/history">
          <span>历史记录</span>
        </el-menu-item>
        <el-menu-item index="/settings">
          <span>系统设置</span>
        </el-menu-item>
      </el-menu>
      <div class="side-footer">
        <div class="status-dot" :class="health?.ok ? 'online' : 'offline'"></div>
        <span class="status-text">{{ health?.ok ? '模型就绪' : '模型未加载' }}</span>
      </div>
    </el-aside>
    <el-container>
      <el-header class="top-header">
        <div class="header-left">
          <h2 class="page-title">{{ pageTitle }}</h2>
        </div>
        <div class="header-right">
          <el-tag v-if="health?.ok" type="success" size="small" effect="dark" round>
            {{ health.device === 'cuda' ? 'GPU' : 'CPU' }} 推理
          </el-tag>
          <el-tag v-else type="danger" size="small" effect="dark" round>离线</el-tag>
          <div class="user-avatar">
            <span>{{ displayUser.charAt(0) }}</span>
          </div>
        </div>
      </el-header>
      <el-main class="main">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref } from "vue";
import { useRoute } from "vue-router";
import { fetchHealth } from "../api/http";

const route = useRoute();
const active = computed(() => route.path);
const health = ref(null);
const displayUser = ref("演示用户");

const PAGE_TITLES = {
  "/single": "单张检测",
  "/batch": "批量检测",
  "/history": "历史记录",
  "/settings": "系统设置",
};
const pageTitle = computed(() => PAGE_TITLES[route.path] || "皮肤病变检测");

let timer = null;

async function refreshHealth() {
  try {
    health.value = await fetchHealth();
  } catch {
    health.value = { ok: false };
  }
}

onMounted(() => {
  refreshHealth();
  timer = setInterval(refreshHealth, 30000);
});

onUnmounted(() => {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
});
</script>

<style scoped>
.layout-root {
  height: 100%;
  background: transparent;
}

/* ── Sidebar ───────────────────────────────── */
.side {
  background: #fff;
  border-right: 1px solid #f0f2f5;
  display: flex;
  flex-direction: column;
  box-shadow: 1px 0 8px rgba(0, 0, 0, 0.03);
  position: relative;
  z-index: 10;
}

.brand {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 24px 20px 20px;
  border-bottom: 1px solid #f0f2f5;
}
.brand-icon {
  font-size: 32px;
  line-height: 1;
}
.brand-title {
  font-size: 16px;
  font-weight: 700;
  color: #1a1a2e;
  letter-spacing: 1px;
}
.brand-sub {
  font-size: 11px;
  color: #909399;
  margin-top: 2px;
}

.side-menu {
  border-right: none;
  padding: 12px 8px;
  flex: 1;
  background: transparent !important;
}
.side-menu .el-menu-item {
  border-radius: 10px;
  margin-bottom: 4px;
  height: 46px;
  line-height: 46px;
  font-size: 14px;
  color: #606266;
  transition: all 0.25s ease;
}
.side-menu .el-menu-item:hover {
  background: rgba(64, 158, 255, 0.06) !important;
  color: #409eff !important;
}
.side-menu .el-menu-item.is-active {
  background: linear-gradient(135deg, rgba(64, 158, 255, 0.1), rgba(58, 123, 213, 0.06)) !important;
  color: #409eff !important;
  font-weight: 600;
}
.side-footer {
  padding: 16px 20px;
  border-top: 1px solid #f0f2f5;
  display: flex;
  align-items: center;
  gap: 8px;
}
.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}
.status-dot.online {
  background: #67c23a;
  box-shadow: 0 0 6px rgba(103, 194, 58, 0.4);
  animation: pulse 2s infinite;
}
.status-dot.offline {
  background: #f56c6c;
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}
.status-text {
  font-size: 12px;
  color: #909399;
}

/* ── Header ────────────────────────────────── */
.top-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: #fff;
  border-bottom: 1px solid #f0f2f5;
  padding: 0 28px;
  height: 64px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.03);
}
.page-title {
  font-size: 18px;
  font-weight: 700;
  color: #1a1a2e;
  margin: 0;
  letter-spacing: 0.5px;
}
.header-right {
  display: flex;
  align-items: center;
  gap: 14px;
}
.user-avatar {
  width: 36px;
  height: 36px;
  border-radius: 10px;
  background: linear-gradient(135deg, #409eff, #3a7bd5);
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  font-weight: 600;
  box-shadow: 0 2px 8px rgba(64, 158, 255, 0.3);
}

/* ── Main ──────────────────────────────────── */
.main {
  padding: 24px 28px;
  overflow: auto;
  background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
  min-height: 0;
}
</style>
