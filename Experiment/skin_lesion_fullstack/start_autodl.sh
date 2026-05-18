#!/usr/bin/env bash
# AutoDL / 恒源云：外网只映射实例内 6006、6008 时，用本脚本把「整站」绑在其中一个端口。
# 默认 6006（对应控制台「6006」那条 https://...:8443 链接）。
# 若你的链接对应 6008：  PORT=6008 ./start_autodl.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PORT="${PORT:-6006}"
echo "[AutoDL] 将监听实例内端口 ${PORT}，请用控制台里「${PORT}」对应的外网 HTTPS 打开（不要混用另一条）。"
exec "$ROOT/start.sh" --single
