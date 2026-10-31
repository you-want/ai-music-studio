# 对标研究：成熟产品与开源模型（2026-10 调研）

> 本文回答两个问题：**生产级 AI 音乐产品长什么样**（商业标杆功能矩阵）、
> **本地引擎选谁**（开源模型对比与选型依据）。结论直接驱动本项目 v4 路线。

## 1. 开源全曲生成模型对比（能否在这台 Intel Mac 上跑）

| 模型 | 参数量 | 许可证 | 人声 | 长度 | CPU 可行性（6 核 i7-9750H / 32GB） | 结论 |
|---|---|---|---|---|---|---|
| **DiffRhythm v1.2**（ASLP-lab） | 1.07B (DiT) + VAE 768MB | **Apache-2.0** | ✅ 真声 | 95s / 4m45s(另下权重) | ✅ 官方 M1 Pro 10 步 ≈ 3 分钟/首；本机 fp32 估 10–40 分钟 | **✅ 选它**（权重已下载 2.8GB） |
| ACE-Step 1.5 (2026-01) | ~4B | MIT | ✅ | 至 4 分钟 | ❌ 官方明确"CPU 不可用速度"；MLX 版仅 Apple Silicon | GPU 机器备选 |
| YuE | 7B (LLaMA) | Apache-2.0 | ✅ | 5 分钟 | ❌ CPU 需数小时 | 排除 |
| Stable Audio Open 3.0 | 小/中档 | Community（<$1M 营收免费） | ❌ 纯音乐 | 至 6 分钟 | 未验证 | 许可与无人声不满足 |
| MusicGen (Meta) | 0.3–3.3B | CC-BY-NC 系 | ❌ | ≤30s | 可跑 | 只能做伴奏/采样 |

**选型依据**：唯一同时满足 ①真声演唱 ②宽松许可证（可商用） ③CPU 可行
④权重已在本机 的开源模型。官方 Mac 支持 2025-03 起，M1 Pro@10 步 3 分钟
（issue #23 维护者数据）；无官方量化版（int8/onnx 均无，维护者列为未来工作）。

关键工程事实（本仓库 `DiffRhythm/` 源码核实 + 本机实测）：
- 推理入口零 Gradio 依赖，`infer/infer.py` 是纯 argparse CLI，可 headless 子进程调用；
- 歌词必须带 `[mm:ss.xx]` 时间戳（`parse_lyrics` 静默丢弃纯文本行）→ 由本项目按
  小节时间自动生成 LRC；歌词文本需净化（emoji/♪ 等符号会让 LangSegment 判
  "other" 语言报错，runner 已做白名单过滤）；
- 2048 帧=95 秒、6144 帧=4m45s（需 full 版权重）；采样 steps=32/cfg=4.0；
- 上游强制 fp16（`load_checkpoint`），Intel CPU 转 fp32（算子兼容且更快）；
- **Intel Mac 的 MPS 会自动"可用"但 Metal 内存池仅 ~6.7GB，必然 OOM**（实测）——
  runner 默认 CPU，MPS 仅显式 `DIFFRHYTHM_DEVICE=mps`（Apple Silicon 用）；
- 依赖栈：torch 2.6 时代 —— 但 **Intel Mac 的 torch 止步 2.2.2（Python≤3.12）**，
  实测兼容（`selftest.py --verify-euler` 校验手动 Euler 采样与上游 odeint
  **位级一致（max|Δ|=0.00）**）；
- 两个安装坑（setup_neural.sh 已处理）：①新版 huggingface_hub 只认
  `model.safetensors`，MuQ 权重是 `pytorch_model.bin` 需转换；②numba/llvmlite
  新版无 macOS x86_64 轮子，需 pin 0.60.0/0.43.0；
- 内存峰值实测 ~9.4GB（cfm 2.2GB + muq 2.65GB + vae 0.6GB + fp32 工作区）；
- **本机（i7-9750H 6 核 CPU）实测计时**：模型加载 43s，采样约 45-120s/步
  （16 步默认），VAE 解码 2-3 分钟 —— 整曲约 **15-25 分钟/首**。

## 2. 商业产品功能矩阵（2026 现状）

| 能力 | Suno v5/5.5 | Udio | Mureka | AIVA | Soundraw | 本项目 v4 |
|---|---|---|---|---|---|---|
| 一句话生成完整歌曲 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅（四引擎） |
| 真人声演唱 | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ DiffRhythm 本地 |
| 简易/自定义双模式 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 自定义歌词+段落标记 | ✅ | ✅ | ✅ | ✅ | ❌ | ✅（按小节对齐） |
| AI 写词 | ✅ | ✅ | ✅ | ❌ | ❌ | ✅（LLM/模板兜底） |
| 参考音频风格迁移 | ✅（Covers） | ✅（remix） | ✅（URL 也行） | ❌ | ❌ | ✅（DiffRhythm ref-audio） |
| 声音克隆（10s 样本） | ✅ 5.5 | ❌ | ✅ | ❌ | ❌ | ❌ 路线图 |
| Personas/音色一致性 | ✅ | ✅ Sessions | ✅ | ❌ | ❌ | ❌ 路线图 |
| 双版本候选/再来一版 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ A/B + seed |
| 延长/续写 | ✅ | ✅ | ✅ | ✅ | ❌ | ✅（器乐引擎） |
| 只改一段（inpaint） | ✅（Studio） | ✅ | ✅ | ❌ | ❌ | ✅ 换副歌（器乐引擎） |
| 分轨 Stems | ✅ 至 12 轨 | ✅ | ✅ | ❌ | ✅ | ✅ 4 轨（器乐引擎） |
| MIDI 导出 | ❌ | ❌ | ❌ | ✅ 多轨 | ❌ | ✅（器乐引擎独有） |
| 时间线编辑器 DAW | ✅ Studio | ✅ Sessions | ❌ | ✅ | ✅ | ❌ 路线图 |
| 曲库/收藏/改名/搜索 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 封面/MV 导出 | ✅ | ✅ | ✅ | ❌ | ❌ | ✅（程序化封面+ffmpeg） |
| 本地运行 / 数据不出机器 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ **差异化** |
| 下载导出 | ✅ | ❌（2025-10 起仅流媒体） | ✅ API | ✅ | ✅ | ✅ WAV/MIDI/stems/MV/zip |
| 费用 | $8–30/月 | 订阅 | 金币制 | €11+/月 | $16.99/月 | **0 元**（LLM 可选免费 Key） |

**Udio 的教训**：版权和解后被"降级"为纯流媒体、用户作品下载被关 ——
**本地优先 + 完整导出 是真差异化**，这正是本项目的定位。

## 3. 差距路线图（按投入产出排序）

1. ~~真声演唱~~ → v4 已完成（DiffRhythm 本地引擎）
2. ~~参考音频风格迁移~~ → v4 已完成（ref-audio 上传 → MuQ-MuLan 风格嵌入）
3. ~~4m45s 全曲~~ → v4.1 已完成（full 权重 + audio_length 285 + UI 开关；本机 60-90 分钟/首）
4. ~~神经歌曲分轨~~ → v4.1 已完成（Demucs htdemucs，实测 114s/首）
5. ~~精确卡拉OK~~ → v4.1 已完成（faster-whisper 强制对齐，实测平均偏移 0.4s/置信度 0.72）
4. **音色一致性（Personas 平替）**：把某首歌的 style 嵌入向量缓存复用（`vocal.npy` 同机制）
5. **神经作品的分轨**：接 Demucs v4（本地源分离，CPU 约 1–3 分钟/首）→ 神经歌曲也有 stems
6. **声音克隆**：开源可商用方案有限（OpenVoice2 MIT / XTTS-v2 Coqui 许可），接位已留（AudioProvider 之外再加 VoiceProvider）
7. **时间线编辑器**：Web Audio 波形 + 段落重排（器乐引擎已具备音符级数据）
8. **GPU 服务器路线**：ACE-Step 1.5（MIT）或 DiffRhythm2（2025-10 发布，块级自回归流匹配，24kHz VAE，约 210s）；Dockerfile 已可扩展 CUDA 基础镜像
9. **声源更干净的翻唱 Covers**：DiffRhythm `--edit` 模式（保留旋律换段歌词/风格），本仓库 infer.py 已含 edit 分支，待产品化

## 4. 服务化与安全（生产加固清单，v4 已实施）

Python 官方文档明确 `http.server` 不建议直接生产暴露。本地工具的务实底线 + 本项目现状：

| 项 | 做法 | 状态 |
|---|---|---|
| 绑定 | 默认 127.0.0.1；`AI_MUSIC_HOST=0.0.0.0` 时**强制**要求 `AI_MUSIC_TOKEN`，否则拒启 | ✅ |
| 鉴权 | `/api/*` Bearer Token（`hmac.compare_digest`）；GET 媒体支持 `?token=`；静态页开放（JS 自带 token） | ✅ |
| 路径穿越 | 静态服务 `os.path.commonpath` 校验（替换可绕过的 `startswith`） | ✅ |
| 请求体 | JSON 1MB / 参考音频 25MB 上限，超出 413；上传文件名服务端生成 + 白名单净化 | ✅ |
| Range | 修复后缀 `bytes=-N` 语义、非法区间 416；文件分块流式发送（不再整读内存） | ✅ |
| 响应头 | `X-Content-Type-Options: nosniff`；无 CORS 头（同源使用） | ✅ |
| 出站 TLS | LLM 调用默认严格证书校验（certifi），仅 `AI_MUSIC_INSECURE_SSL=1` 显式降级 | ✅ |
| 长任务 | 队列 + 信号量（渲染 2 并发/神经 1 并发）+ 进度轮询 + 可取消 | ✅ |
| 测试隔离 | 测试显式传配置路径时不再回落开发者本机 config（修掉"测试偷偷调用真 LLM"缺陷） | ✅ |
| 更进一步（LAN/公网生产） | 反代（Caddy/nginx）+ TLS + 速率限制；或迁 FastAPI/uvicorn | 路线图 |

## 5. 参考

- DiffRhythm: https://github.com/ASLP-lab/DiffRhythm （Apache-2.0；issue #23 为 Mac/CPU 数据点）
- DiffRhythm 2: https://github.com/ASLP-lab/DiffRhythm2
- ACE-Step: https://github.com/ace-step/ACE-Step · ACE-Step 1.5: https://github.com/ace-step/ACE-Step-1.5
- YuE: https://github.com/multimodal-art-projection/YuE
- Stable Audio 3.0: https://huggingface.co/stabilityai/stable-audio-3-optimized
- PyTorch 平台支持（Intel Mac 止步 2.2.2）: https://pypi.org/pypi/torch/json
- Suno: https://suno.com · Udio: https://www.udio.com · Mureka: https://platform.mureka.ai
- AIVA: https://www.aiva.ai · Soundraw: https://soundraw.io
- http.server 安全须知: https://docs.python.org/3/library/http.server.html#security-considerations
