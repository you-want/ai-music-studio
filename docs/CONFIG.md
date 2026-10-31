# AI 音乐工坊 · 配置指南

> 回答两个问题：**后期需要配置什么？用什么模型？**
> 按"优先级"从上往下配，每一级都独立有效——什么都不配也能用，配得越多越强。

## 0. 一句话结论（TL;DR）

| 你想要 | 要做什么 | 花钱吗 |
|---|---|---|
| 现在就能用 | **什么都不用配** | ¥0 |
| 更聪明的 AI 写词/作曲（强烈建议） | 注册智谱 → 配一个免费 Key（2 分钟） | ¥0 |
| 演唱音色更好听 | 下载 macOS 增强语音 | ¥0 |
| 服务常驻后台/开机自启 | 配一个 launchd | ¥0 |
| **以假乱真的真声演唱** | 接入 Mureka 或 Suno 的 API | 💰 付费 |

---

## 1. 配置全景图

| # | 配置项 | 作用 | 必须？ | 费用 |
|---|---|---|---|---|
| 1 | [智谱 GLM Key（免费大模型）](#31-首选智谱-glm-45-flash免费推荐) | AI 写词、作曲决策（和声/曲式更贴合描述） | 建议 | **免费** |
| 2 | [macOS 增强语音](#5-演唱音色优化免费本地) | 演唱音色更清晰 | 可选 | 免费 |
| 3 | [常驻运行 / 开机自启](#2-运行与环境配置) | 不用手动启动 | 可选 | 免费 |
| 4 | [付费 LLM（DeepSeek 等）](#33-付费-llm可选) | 歌词文学性更强 | 可选 | 极低 |
| 5 | [真声演唱 API（Mureka/Suno）](#4-真声演唱ai-歌声唯一需要花钱的效果) | 真人级歌声唱歌词 | 可选 | 付费 |

---

## 2. 运行与环境配置

### 2.1 基础运行（已就绪）

- 依赖：Python 3.9+ 与 `numpy`（`scripts/run.sh` 会自动检测并建 venv 安装）；
- 启动：`cd ~/you-want/ai-music-studio && bash scripts/run.sh` → http://127.0.0.1:8766

### 2.2 环境变量（都是可选的）

| 变量 | 默认 | 说明 |
|---|---|---|
| `AI_MUSIC_PORT` | `8766` | 服务端口（8765 被你的 ai-cover-studio 占用，所以默认 8766） |
| `AI_MUSIC_HOST` | `127.0.0.1` | 监听地址（局域网访问改 `0.0.0.0`，注意安全） |
| `AI_MUSIC_CONFIG` | 项目根 `config.local.json` | 配置文件路径 |
| `AI_MUSIC_LLM_BASE_URL` | `https://open.bigmodel.cn/api/paas/v4` | OpenAI 兼容端点 |
| `AI_MUSIC_LLM_API_KEY` | 空 | 大模型 Key（设置了即启用 LLM） |
| `AI_MUSIC_LLM_MODEL` | `glm-4-flash` | 模型名 |

### 2.3 常驻运行（开机自启，可选）

创建 `~/Library/Launchers/aimusic.plist`：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>ai-music-studio</string>
  <key>ProgramArguments</key><array>
    <string>/bin/bash</string>
    <string>-c</string>
    <string>cd /Users/rain9/you-want/ai-music-studio &amp;&amp; bash scripts/run.sh</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/tmp/aimusic.log</string>
  <key>StandardErrorPath</key><string>/tmp/aimusic.log</string>
</dict></plist>
```

加载：`launchctl load ~/Library/Launchers/aimusic.plist`（卸载把 `load` 换成 `unload`）。

### 2.4 数据管理

| 目录 | 内容 | 说明 |
|---|---|---|
| `backend/output/<id>/` | 每首歌：整曲 WAV、MIDI、4 条分轨、meta/spec/score JSON | **作品库**，删作品=删目录 |
| `backend/output/.tts_cache/` | 演唱用单字 TTS 缓存 | 可整个删掉，只是下次演唱要重新合成 |
| `config.local.json` | 你的 LLM Key 等配置 | **已加入 .gitignore，不要提交/外传** |

备份作品 = 拷贝 `backend/output/`；想彻底重置 = 删掉整个 `output/`。

---

## 3. 大模型配置（免费 → 付费）

> LLM 在本产品里负责两件事：**✍️ AI 写词** 和 **作曲决策**（结构/和弦/调式更贴合你的描述）。
> 演唱、编曲、渲染始终是本地完成，LLM 只影响"创意层"。配好 Key 后引擎选"自动"即可，
> LLM 失败会自动回落离线引擎，**永远不会因为网络问题生成失败**。

### 3.1 首选：智谱 GLM-4.5-Flash（免费，推荐）

为什么：**官方完全免费**、中文创作能力强、注册就送 2000 万 tokens 体验额度；
比 GLM-4-Flash 更新更强，比 GLM-4.7-Flash 限流更少（社区实测）。

**配置步骤（2 分钟）：**

1. 打开 [open.bigmodel.cn](https://open.bigmodel.cn) → 注册/登录（手机号即可）；
2. 右上角头像 → **API Keys** → 新建并复制（形如 `xxxxxxxx.yyyyyyyy`）；
3. 三种配置方式**任选其一**：
   - **网页里配（最简单）**：打开音乐工坊 → 右上角 ⚙️ 引擎设置 → 粘贴 Key → 保存
     （Base URL 填 `https://open.bigmodel.cn/api/paas/v4`，模型填 `glm-4.5-flash`）；
   - **配置文件**：复制 `config.example.json` 为项目根的 `config.local.json`，改成：
     ```json
     { "llm": {
         "enabled": true,
         "base_url": "https://open.bigmodel.cn/api/paas/v4",
         "api_key": "你的Key",
         "model": "glm-4.5-flash"
     } }
     ```
     然后重启服务（`bash scripts/run.sh`）；
   - **环境变量**：`AI_MUSIC_LLM_API_KEY=你的Key AI_MUSIC_LLM_MODEL=glm-4.5-flash bash scripts/run.sh`
4. **验证**：网页 🎛 进阶模式 → 点 **✍️ AI 写词** → 按钮显示"✍️ 已写入(LLM)"即成功
   （没配 Key 时显示"已写入(模板)"）。

### 3.2 其他免费选项

| 方案 | 配置 | 说明 |
|---|---|---|
| **本地 Ollama** | `base_url=http://127.0.0.1:11434/v1`，`model=qwen2.5:7b` | 完全离线；`brew install ollama && ollama pull qwen2.5:7b`；推荐 `qwen2.5:7b/14b`、`glm4:9b`、`deepseek-r1:7b`；需要 8GB+ 内存 |
| 智谱 GLM-4-Flash | `model=glm-4-flash` | 老款免费模型，作为 4.5-Flash 的备用 |
| 硅基流动/OpenRouter 免费档 | 对应 base_url + free 模型名 | 有免费额度的聚合平台，按平台文档填 |

### 3.3 付费 LLM（可选）

写一首歌的歌词/作曲决策只消耗几千 tokens，**成本不到一分钱**，付费主要换来更好的文学性和稳定性：

| 平台 | base_url | 模型名 | 参考价 |
|---|---|---|---|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` | 输入约 ¥1–4 / 百万 tokens |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` | $0.15 / 百万输入 |
| 智谱旗舰 | `https://open.bigmodel.cn/api/paas/v4` | `glm-4.5` / `glm-4.6` | 见[官方定价页](https://docs.bigmodel.cn/cn/guide/start/pricing) |

### 3.4 模型选型速查表

| 用途 | 首选 | 备选 | 说明 |
|---|---|---|---|
| AI 写词（免费） | **GLM-4.5-Flash** | GLM-4-Flash / 本地 Qwen2.5 | 中文歌词质量最好的免费选择 |
| 作曲决策（免费） | **GLM-4.5-Flash** | GLM-4-Flash | 输出 SongSpec JSON，有校验护栏兜底 |
| 歌词文学性（付费） | DeepSeek-V3 | GPT-4o / GLM-4.6 | 单次成本忽略不计 |
| 完全离线 | Ollama + Qwen2.5:7b | llama3.1:8b | 无网络环境用 |

---

## 4. 真声演唱（AI 歌声）——v4 已有本地免费方案

**v4 起，真声演唱不再必须花钱**：内置 **DiffRhythm 本地神经歌声引擎**
（Apache-2.0 开源模型，1.07B 参数，本仓库 `DiffRhythm/` 已含全部权重）。

### 方案对比（2026-10 更新）

| 方案 | 价格 | 真实度 | 速度 | 建议 |
|---|---|---|---|---|
| **A. 本地 DiffRhythm（v4 内置）** | **0 元** | ⭐⭐⭐⭐（开源模型里第一梯队） | CPU 15-30 分钟/首（实测） | **默认推荐**，`bash scripts/setup_neural.sh` |
| B. Mureka 官方 API | 生成约 $0.01/分钟 级别 | ⭐⭐⭐⭐⭐ 中文歌声极好 | 秒级-分钟级 | 追求极致音质/批量生产时接入 |
| C. Suno（第三方 API） | 官方 $10/月；第三方 $0.04-0.08/首 | ⭐⭐⭐⭐⭐ | 快 | 第三方稳定性/合规风险自担 |
| D. macOS TTS 逐字演唱（内置"🎤 AI 演唱"） | 0 元 | ⭐⭐（机器人 Vocaloid 感） | 快 | 无 venv 环境的兜底 |

### 安装本地 DiffRhythm（一次性）

```bash
bash scripts/setup_neural.sh     # uv 装 Python 3.11 + torch 2.2.2 CPU + 依赖 + HF 缓存
DiffRhythm/.venv-neural/bin/python backend/neural/selftest.py --mini   # 自检（应 PASS）
```

- 权重（`DiffRhythm/pretrained/cfm_model.pt` 等 2.8GB）需已在本机；
- 依赖 `espeak-ng`（`brew install espeak-ng`）；
- 网页「引擎」下拉选 **🧠 神经歌声** 即可；固定 95 秒/首；
- 生成后自动两步后处理（可在 config 关闭）：
  **歌词对齐**（faster-whisper，每句歌词真实起止时间，卡拉OK精确跟唱，约 20s）
  与**分轨分离**（demucs htdemucs，人声/鼓/贝斯/伴奏，约 2 分钟，约 1.5GB 模型自动下载）；
- **全曲版 4m45s**：`DiffRhythm/pretrained/cfm_model_full.pt`（2.2GB，
  `huggingface-cli download ASLP-lab/DiffRhythm-1_2-full cfm_model.pt --local-dir .`）
  就绪后网页会出现"🕐 生成完整版"开关，耗时约为 95s 版 3 倍（本机 60-90 分钟）；
- 调优（可选）：`config.local.json` 加
  `"neural": {"steps": 12, "timeout_min": 120, "device": "cpu",
  "audio_length": 95, "whisper_model": "small", "align": true, "separate": true}`
  （steps 越小越快，8 以下音质下降）；
- **设备说明**：默认 CPU（fp32）。CUDA 机器自动用 GPU。**MPS 不自动启用**——
  Intel Mac 的 Metal 内存池仅 ~6.7GB 会 OOM（实测），Apple Silicon 用户可显式
  `"device": "mps"` 提速；
- 常见问题：首次生成多花 1-2 分钟加载模型属正常；报"缺少权重"请核对
  `DiffRhythm/pretrained/` 下三个文件；CPU 占满风扇狂转是正常现象。

### 云端 API 接入（可选，架构已预留）

`backend/app/providers/audio_base.py` 定义了音频成品 Provider 契约
（`available()` + `generate()`）。接入 Mureka/Suno 只需：

1. 新增 `providers/mureka.py` 实现 `AudioProvider`：把风格/歌词发出去，轮询拿回音频；
2. 在 `jobs.py` 的 `AUDIO_PROVIDERS` 集合加入 `"mureka"`；
3. 前端 `index.html` 引擎下拉加一项。

`jobs._run_audio`、进度、取消、曲库展示全部自动复用，预计 1-2 小时工作量。

### 参考音频风格迁移（v4）

进阶模式选神经歌声后可上传 **≥10 秒** wav/mp3/flac（`POST /api/ref-audio`），
生成会向参考音频的曲风/编曲/演唱风格靠拢（MuQ-MuLan 嵌入，同 Mureka 的
"参考歌曲"能力）。参考音频存于 `backend/output/_refs/`。

---

## 4b. 局域网部署（Token 鉴权，v4）

```bash
AI_MUSIC_HOST=0.0.0.0 AI_MUSIC_TOKEN=你的口令 bash scripts/run.sh
```

- **不设 Token 会被拒绝启动**（防裸奔暴露局域网）；
- 局域网设备用 `http://<你的IP>:8766/?token=你的口令` 打开，页面会自动携带鉴权；
- API 直连方需 `Authorization: Bearer 你的口令`（GET 下载可用 `?token=`）；
- 明确边界：这是轻量口令而非用户体系；若要公网生产，请加反代
  （Caddy/nginx 做 TLS + 速率限制），见 `docs/BENCHMARK.md` §4。

---

## 5. 演唱音色优化（免费，本地）

本产品的演唱用的是 macOS 系统语音（当前：**婷婷 Tingting**）。可以免费升级：

1. **下载增强版语音**：系统设置 → 辅助功能 → 朗读内容 → 系统声音 → 管理声音…
   → 找到中文（中国大陆）→ 下载 **"婷婷（增强版）"**（音质明显更好）；
2. 查看本机可用语音：终端执行 `say -v '?' | grep zh_CN`；
3. **切换默认音色**：改 `backend/app/music/singer.py` 里 `available_voice()` 的优先级
   （把 `"Tingting"` 改成 `"Tingting (Enhanced)"` 或列表里任意名字），重启服务生效。

---

## 6. 进阶：代码级可调参数（想深玩再看）

| 想改什么 | 文件 | 位置 |
|---|---|---|
| 和弦进行、速度区间、歌词模板 | `backend/app/music/mock_composer.py` | `PROFILES` / `LYRICS` |
| 旋律节奏型、贝斯模式、鼓组模板 | `backend/app/music/realizer.py` | `RHYTHM_POOL` / `BASS_PATTERNS` / `DRUM_PATTERNS` |
| 音色（谐波/包络/颤音）、人声共振峰 | `backend/app/music/synth.py` | `WAVES["vocal_hum"]` 等 |
| 混响/母带 | `backend/app/music/synth.py` | `_master()` |
| 演唱的移调极限/延音算法 | `backend/app/music/singer.py` | `render_sing_mono()` |

改完跑测试：`cd backend && python3 -m unittest discover -s tests`（当前 47 个用例全绿）。

---

## 7. 常见问题

| 现象 | 原因/解决 |
|---|---|
| 端口被占用 `Address already in use` | `AI_MUSIC_PORT=9000 bash scripts/run.sh` |
| 写词显示"(模板)"而不是"(LLM)" | Key 未配置/未生效：检查 ⚙️ 设置或 `config.local.json`，重启服务 |
| 生成报"LLM 作曲失败" | 引擎选了"大模型 LLM"但 Key 无效；改"自动"即可回落离线引擎 |
| 演唱没声音 | 非中文语音环境：`say -v '?' | grep zh_CN` 确认；没有则去系统设置下载 |
| 第一次生成很慢 | 演唱的逐字 TTS 首次合成（约 10–20s）；之后有缓存会很快 |
| 找不到旧作品的分轨 | v1 时代作品没有分轨；重新生成即有 |

---

## 8. 你的行动清单（按优先级）

- [ ] **1.（2 分钟，免费，强烈建议）** 注册 [open.bigmodel.cn](https://open.bigmodel.cn) 拿 Key
      → 网页 ⚙️ 引擎设置里粘贴保存（模型 `glm-4.5-flash`）→ 用 AI 写词验证
- [ ] **2.（可选，免费）** 下载 macOS "婷婷（增强版）"语音，演唱音质提升一档
- [ ] **3.（可选）** 按 §2.3 配置 launchd 常驻运行
- [ ] **4.（可选，付费）** 想要真人级歌声：选 **Mureka**（推荐）或 Suno，注册拿 Key 给我 → 我来接
- [ ] **5.（随时）** 把试听感受告诉我（音色/唱法/编曲口味），我来调参
