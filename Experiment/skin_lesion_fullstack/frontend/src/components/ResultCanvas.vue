<!--
  检测结果可视化组件：在原图上叠加 Canvas 绘制的检测框和标签。
  通过 ResizeObserver 自适应容器宽度，box-space 属性支持缩略图场景下的坐标映射。
-->
<template>
  <div ref="wrapRef" class="result-canvas-wrap">
    <img
      v-show="src"
      ref="imgRef"
      :src="src"
      class="base-img"
      alt="preview"
      @load="onImgLoad"
    />
    <canvas ref="cvRef" class="overlay-canvas" />
  </div>
</template>

<script setup>
import { nextTick, onMounted, onUnmounted, ref, watch, watchEffect } from "vue";
import { colorForClass } from "../utils/colors";

const props = defineProps({
  src: { type: String, default: "" },
  detections: { type: Array, default: () => [] },
  /** 检测框坐标所在空间的宽度（原图像素），缩略图展示时用于坐标换算 */
  boxSpaceWidth: { type: Number, default: 0 },
  boxSpaceHeight: { type: Number, default: 0 },
});

const wrapRef = ref(null);
const imgRef = ref(null);
const cvRef = ref(null);

/** 核心绘制：将检测框坐标从 box-space 缩放到当前图片显示尺寸 */
function draw() {
  const wrap = wrapRef.value;
  const img = imgRef.value;
  const cv = cvRef.value;
  if (!wrap || !img || !cv || !props.src) return;

  const iw = img.naturalWidth;
  const ih = img.naturalHeight;
  if (!iw || !ih) return;

  // 坐标空间可能和图片显示尺寸不同（比如历史详情用缩略图展示原图坐标）
  const bw = props.boxSpaceWidth || iw;
  const bh = props.boxSpaceHeight || ih;

  // 根据容器宽度自适应缩放
  const rect = wrap.getBoundingClientRect();
  const maxW = rect.width;
  const maxH = Math.min(640, window.innerHeight * 0.55);
  const scale = Math.min(maxW / iw, maxH / ih, 1);
  const dw = Math.round(iw * scale);
  const dh = Math.round(ih * scale);

  img.style.width = `${dw}px`;
  img.style.height = `${dh}px`;

  cv.width = dw;
  cv.height = dh;
  const ctx = cv.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, dw, dh);

  // 坐标缩放比
  const sx = dw / bw;
  const sy = dh / bh;

  for (const d of props.detections || []) {
    const [x1, y1, x2, y2] = d.xyxy;
    const c = colorForClass(d.name_en);
    ctx.strokeStyle = c;
    ctx.lineWidth = 2;
    ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);

    // 标签背景 + 文字
    const label = `${d.name_en} ${(d.confidence * 100).toFixed(1)}%`;
    ctx.font = "12px sans-serif";
    const tw = ctx.measureText(label).width;
    const pad = 4;
    const lx = x1 * sx;
    const ly = y1 * sy - 18;
    ctx.fillStyle = c + "99"; // 半透明背景
    ctx.fillRect(lx, ly < 0 ? y1 * sy : ly, tw + pad * 2, 16);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, lx + pad, (ly < 0 ? y1 * sy : ly) + 12);
  }
}

function onImgLoad() {
  nextTick(draw);
}

let ro = null;

onMounted(() => {
  window.addEventListener("resize", draw);
});

// ResizeObserver 监听容器尺寸变化，自动重绘
watchEffect((onCleanup) => {
  const el = wrapRef.value;
  if (!el) return;
  ro?.disconnect();
  ro = new ResizeObserver(() => draw());
  ro.observe(el);
  onCleanup(() => ro?.disconnect());
});

onUnmounted(() => {
  ro?.disconnect();
  ro = null;
  window.removeEventListener("resize", draw);
});

// props 变化时重绘
watch(
  () => [props.src, props.detections, props.boxSpaceWidth, props.boxSpaceHeight],
  () => nextTick(draw),
  { deep: true }
);
</script>

<style scoped>
.result-canvas-wrap {
  position: relative;
  display: inline-block;
  max-width: 100%;
  vertical-align: top;
}
.base-img {
  display: block;
  max-width: 100%;
  border-radius: 8px;
  border: 1px solid #ebeef5;
}
.overlay-canvas {
  position: absolute;
  left: 0;
  top: 0;
  pointer-events: none;
  border-radius: 8px;
}
</style>
