#!/usr/bin/env bash
#
# 一键启动皮肤病变检测系统
#
#   ./start.sh              双端口：Flask(API) + Vite(界面)，默认 5000 + 6006
#   ./start.sh --single     单端口：先 npm build，再只起 Flask，界面与 API 同在 PORT（默认 5000）
#
# 停止：Ctrl+C（双端口模式）；单端口模式 Ctrl+C 结束 Flask
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
用法:
  ./start.sh              开发模式：Flask + Vite（需同时访问 API 端口与前端端口）
  ./start.sh --single     生产/云平台推荐：只监听 PORT，构建前端后由 Flask 托管

环境变量:
  PORT                 后端端口，默认 5000（--single 时也是网站端口）
  VITE_DEV_PORT        仅双端口模式：前端端口，默认 6006
  VITE_API_PROXY       仅双端口：Vite 代理后端地址（默认 http://127.0.0.1:$PORT）
  NODE_BIN_DIR         可选，Node 的 bin 目录
  SKIN_DETECTION_MODEL 可选，best.pt 路径
EOF
  exit 0
fi

if [[ -n "${NODE_BIN_DIR:-}" && -d "$NODE_BIN_DIR" ]]; then
  export PATH="$NODE_BIN_DIR:$PATH"
fi

PY="python3"
command -v "$PY" &>/dev/null || PY="python"
command -v "$PY" &>/dev/null || { echo "[错误] 未找到 python3 或 python"; exit 1; }
command -v npm &>/dev/null || { echo "[错误] 未找到 npm，请安装 Node.js 18+，或设置 NODE_BIN_DIR"; exit 1; }

FLASK_PORT="${PORT:-5000}"

# ---------- 单端口：适合只映射了 5000 的 AutoDL / SSH ----------
if [[ "${1:-}" == "--single" || "${SINGLE_PORT:-}" == "1" ]]; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  单端口模式（Flask + 已构建 Vue）"
  echo "  端口: ${FLASK_PORT}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  if [[ ! -d "$ROOT/frontend/node_modules" ]]; then
    echo "==> npm install ..."
    (cd "$ROOT/frontend" && npm install)
  fi
  echo "==> npm run build （约需几十秒）..."
  (cd "$ROOT/frontend" && npm run build)
  echo ""
  echo "  请在浏览器打开: http://127.0.0.1:${FLASK_PORT}/"
  echo "  云平台: 把「自定义服务」指到本机 ${FLASK_PORT} 即可，无需再映射 6006"
  echo ""
  cd "$ROOT/backend"
  export PORT="$FLASK_PORT"
  exec "$PY" app.py
fi

# ---------- 双端口开发模式 ----------
VITE_PORT="${VITE_DEV_PORT:-6006}"
export VITE_API_PROXY="${VITE_API_PROXY:-http://127.0.0.1:${FLASK_PORT}}"

cleanup() {
  [[ -n "${VITE_PID:-}" ]] && {
    pkill -TERM -P "$VITE_PID" 2>/dev/null || true
    kill "$VITE_PID" 2>/dev/null || true
  }
  [[ -n "${FLASK_PID:-}" ]] && {
    pkill -TERM -P "$FLASK_PID" 2>/dev/null || true
    kill "$FLASK_PID" 2>/dev/null || true
  }
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  皮肤病变智能检测 · 双端口开发模式"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[后端] PORT=${FLASK_PORT}  ->  ${VITE_API_PROXY}"
echo "[前端] VITE_DEV_PORT=${VITE_PORT}"
echo "  若云平台未映射 ${VITE_PORT}，请改用:  ./start.sh --single"
echo ""

echo "==> 启动 Flask ..."
(
  cd "$ROOT/backend"
  export PORT="$FLASK_PORT"
  exec "$PY" app.py
) &
FLASK_PID=$!

echo "==> 等待 /api/health ..."
ok=0
for _ in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:${FLASK_PORT}/api/health" >/dev/null; then
    ok=1
    break
  fi
  sleep 0.5
  if ! kill -0 "$FLASK_PID" 2>/dev/null; then
    echo "[错误] 后端进程已退出，请检查依赖与权重路径。"
    exit 1
  fi
done
if [[ "$ok" != 1 ]]; then
  echo "[错误] 等待后端超时（${FLASK_PORT} 无响应）。"
  exit 1
fi
echo "==> 后端已就绪"

if [[ ! -d "$ROOT/frontend/node_modules" ]]; then
  echo "==> 首次运行：npm install ..."
  (cd "$ROOT/frontend" && npm install)
fi

echo "==> 启动 Vite ..."
echo ""
echo "  本机打开: http://127.0.0.1:${VITE_PORT}/"
echo "  若 ERR_EMPTY_RESPONSE: 请确认已映射 ${VITE_PORT}，或运行 ./start.sh --single"
echo ""
echo "  按 Ctrl+C 停止全部服务"
echo ""

(
  cd "$ROOT/frontend"
  export VITE_DEV_PORT="$VITE_PORT"
  export VITE_API_PROXY
  exec npm run dev
) &
VITE_PID=$!

wait "$VITE_PID"
