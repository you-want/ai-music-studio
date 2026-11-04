# 织歌 SongWeaver 生产镜像（基础版：离线 Mock / LLM / Mureka 云引擎 + MV 导出）。
# 神经歌声引擎（DiffRhythm）依赖 2.8GB 权重与独立 venv，不打入镜像——
# 如需容器内神经渲染：挂载 DiffRhythm/pretrained 权重并在容器内跑 scripts/setup_neural.sh。
FROM python:3.12-slim

WORKDIR /app

# 先装依赖再拷代码，充分利用层缓存
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt \
    && apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY backend/ backend/
COPY frontend/ frontend/
COPY scripts/ scripts/
COPY config.example.json ./

# 容器内必须绑 0.0.0.0；服务端会拒绝"非本机绑定且无 Token"的启动（防裸奔），
# 因此部署时必须同时传 AI_MUSIC_TOKEN（见 docker-compose.yml）。
ENV AI_MUSIC_HOST=0.0.0.0 \
    AI_MUSIC_PORT=8766 \
    PYTHONUNBUFFERED=1
EXPOSE 8766

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('AI_MUSIC_PORT', '8766'), timeout=4)"

CMD ["python", "backend/app/server.py"]
