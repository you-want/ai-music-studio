# 更新日志 (Changelog)

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)；
版本号单一来源见 `backend/app/__init__.py` 的 `__version__`。

## [5.0.0] - 2026-11-05

v5 功能齐全后的**生产化加固专项**（三轮，8 个提交）。

### 修复（正确性）

- **MV 导出并发竞态**：per-song 锁 + ffmpeg 先写 `.part`（显式 `-f mp4`）+ `os.replace`
  原子落盘——并发导出不再产出损坏 mp4，失败/超时清掉残留半成品
- **全新 clone 样式 404**：`themes-scoped.css` / `utilities.css` 被误加入 .gitignore，
  而 `index.html → han-scoped.css → @import` 链运行时必需；已入库并留注释防回归
- 测试 stub 引用了未导入的 `AudioProviderError`（ruff F821，触发即 NameError）
- LLM 作曲接入 429/5xx 退避重试（此前只有写词路径有重试）；采样温度钳制到 [0, 2]

### 安全

- **会话 Cookie 不再保存原始访问 Token**：改为 `HMAC(token, 过期时间)` 派生凭证——
  12 小时自动作废、更换 `AI_MUSIC_TOKEN` 即全量吊销、无服务端状态、重启后依然有效
- 所有响应带 `X-Frame-Options: DENY` + `Referrer-Policy`；HTML 加基线 CSP
  （前端无内联脚本）；静态 404 不再回显请求路径；专辑 zip 目录名过滤 `..` 等穿越风险
- **请求字段统一限长**：description ≤500、title ≤60、歌词 ≤5000（保留换行结构）、
  重唱歌词 ≤2000——1MB 请求体不再原样流进 meta / LLM prompt / 参数指纹

### 健壮性 / 资源

- **同参数并发提交去重**：复用进行中的任务，双击或双端重复提交不再重复渲染、
  不再对云端引擎重复计费
- **本地合成渲染中途可取消**（渲染分组合间检查取消事件），半成品目录照常清理
- **磁盘残留 GC**：任务快照 24h、参考音频 7 天自动回收（进行中任务引用受保护）；
  神经 worker 日志 10MB 轮转
- **曲库指纹内存索引**：提交时 O(1) 命中，替代每次提交的全曲库扫描
- **日志系统**：访问日志 / 500 / 任务失败走 logging + 轮转文件
  （`backend/output/_logs/server.log`，5MB×3），`AI_MUSIC_LOG_LEVEL` 可控

### 工程化

- **GitHub Actions CI**：ruff + 全量测试（Python 3.9 / 3.12 矩阵）
- **ruff 静态检查**接入并清零全部 37 处存量告警
- **Docker 部署**：Dockerfile（内置 ffmpeg）+ docker-compose（Token 必填、
  曲库 volume 持久化）+ .dockerignore；容器内作曲→渲染→下载全链路已实测通过
- `__version__` 单一版本源，接入 Server 头与 `/api/health`
- 测试 107 → 124 个（新增字段限长、会话凭证、磁盘 GC、去重、渲染取消、
  指纹索引、安全头等回归用例）

### 前端

- 专辑等多任务进度聚合显示（N/M + 当前阶段）；双版本候选保留 A/B 分别展示
- 音频加载失败在歌词行给出可见提示；移除重复注册的 resize 监听

### 已知限制

- 取消 Mureka 云端任务仅停止本地等待，云端任务照常执行并计费（官方接口无取消能力），
  详见 `docs/CONFIG.md`
- 神经歌声引擎（DiffRhythm）不在 Docker 镜像内（2.8GB 权重 + 独立 venv），本机部署使用
