#!/usr/bin/env bash
# AI 音乐工坊 一键启动：自动准备 numpy（缺则建 venv），起本地服务。
set -e
cd "$(dirname "$0")/.."

PY=python3
if ! $PY -c "import numpy" 2>/dev/null; then
  echo "[setup] 未检测到 numpy，正在创建虚拟环境并安装（一次性）…"
  [ -d .venv ] || $PY -m venv .venv
  ./.venv/bin/pip install -q --upgrade pip numpy
  PY=./.venv/bin/python3
fi

PORT="${AI_MUSIC_PORT:-8766}"
echo "启动 织歌 SongWeaver → http://127.0.0.1:${PORT}"
exec env AI_MUSIC_PORT="${PORT}" "$PY" backend/app/server.py
