# 🎵 织歌 SongWeaver（AI Music Studio）

**一句话，织一首歌。** 完全本地运行的 AI 音乐创作程序：
AI 决定调性、和声、曲式、歌词，本地编曲引擎展开成旋律/贝斯/和声/鼓组，
再用内置合成器渲染成 **44.1kHz 立体声 WAV**，同时导出 **标准 MIDI** 和 **4 条分轨**。

> ✅ **零费用即可完整使用**：默认离线 Mock 作曲引擎，不联网、不要 API Key。
> 想要更强的"作曲大脑"和 AI 写词？网页里免费开启 **GLM-4-Flash**（智谱免费模型）即可，
> 也兼容 DeepSeek / OpenAI / 本地 Ollama 等任何 OpenAI 兼容端点。
>
> 🧠 **v4 新增：神经歌声引擎（DiffRhythm）**——开源 Apache-2.0 模型本地推理，
> 输入歌词与风格描述（或上传 ≥10s 参考音频做风格迁移），直接产出**带真人声演唱的
> 95 秒完整歌曲**。CPU 渲染约 15-30 分钟/首（本机实测 28.5 分钟/16 步），依旧零费用。
> 对标调研见 [docs/BENCHMARK.md](docs/BENCHMARK.md)。

## 快速开始

```bash
cd ~/you-want/ai-music-studio
bash scripts/run.sh              # 启动（自动检查/安装 numpy）
# 可选：安装神经歌声引擎（一次性，约 5 分钟 + 已下载的 2.8GB 权重）
# bash scripts/setup_neural.sh && DiffRhythm/.venv-neural/bin/python backend/neural/selftest.py
# 打开 http://127.0.0.1:8766
```

唯一依赖：Python 3.9+ 和 `numpy`（脚本会自动建 venv 安装）。
端口被占用时：`AI_MUSIC_PORT=9000 bash scripts/run.sh`；
局域网共享：`AI_MUSIC_HOST=0.0.0.0 AI_MUSIC_TOKEN=你的口令 bash scripts/run.sh`
（不设 Token 会被拒绝启动，防裸奔）。

## 功能（v2，对标 Mureka）

**两种创作模式**（对齐 Mureka 的 简易/自定义）：
- 💬 **简易模式**：一句话描述 → 自动识别风格情绪（9 风格×6 情绪，含说唱：boom-bap/trap 双律动）、时长 15–120s、BPM；
- 🎛 **进阶模式**：自定义歌名、**歌词编辑器**（支持 `[Verse]/[Chorus]/[Bridge]` 中英文段落
  标记，按小节对齐旋律）、**✍️ AI 写词**（LLM 优先、离线模板兜底，可编辑后再生成）、
  **主奏音色**（合成主音/方波电子/钢琴/拨弦古筝/电钢/**人声哼唱**）、**调性锁定**（12 调）、
  **双版本候选**（一次生成 A/B 两版对比挑选）。

**播放与作品**：
- 播放器 + **实时频谱可视化**（Web Audio）+ 卡拉OK式当前歌词；
- **钢琴卷帘**：主旋律/和声/琶音/贝斯/鼓组分色可视化，段落底色与和弦符号，
  点击跳转、播放头自动跟随；
- **分轨导出 Stems**：主唱(旋律)/和声/贝斯/鼓 4 条独立 WAV，可导入 DAW 混音；
- **AI 封面**：每首歌按 seed 生成专属封面（作品卡缩略图 + 下载 PNG）；
- **🎬 MV 导出**：封面 + 音频 → mp4（本地 ffmpeg，一键下载发朋友圈）；
- **📦 分享包**：整曲 + MIDI + 4 分轨 + 封面 + 歌词 + 曲目信息打包 zip；
- **🎼 换副歌**：不满意副歌？只重写副歌旋律（主歌/和声/鼓与原曲完全一致）；
- **🔁 再来一版**（同参数换 seed）/ **⏱ 延长 30s**（同 seed 续写更长结构）；
- **作品目录**：收藏置顶、重命名、搜索过滤、删除；所有作品本地持久保存；
- 下载：整曲 WAV / MIDI / 封面 PNG / 4 条分轨 / MV / 分享包。

**音乐品质（v3 母带链）**：段落动态（intro 渐入/主歌减配/副歌全奏/桥段抽鼓转调/
outro 渐出）、人声侧链避让（唱时伴奏自动让位）、总线 EQ + 3:1 压缩 + 软限幅、
琶音 ping-pong 立体声、半终止-终止式和声、旋律问答乐句；
演唱带滑音、字头起音随机化、句尾气声、长音延音等人性化细节。

**性能**：整曲+分轨一次渲染（float32）、同时渲染上限 2 首排队、
同参数结果缓存秒回、演唱单字 TTS 磁盘缓存 + 启动后台预热。

**双引擎作曲 + 神经演唱（v4 四引擎）**：
- 🧩 **离线 Mock**：算法作曲，确定性可复现（同 seed 同曲），永远可用；
- 🤖 **大模型 LLM**：免费 GLM-4-Flash 做作曲决策与写词（更贴合描述），
  输出经严格校验修复，失败自动回落 Mock，永不中断；
- 🧠 **神经歌声 DiffRhythm**（v4）：`bash scripts/setup_neural.sh` 一键安装。
  作曲决策仍由 LLM/Mock 完成歌词与结构，再交给开源 DiffRhythm（Apache-2.0，1.07B）
  本地推理出**真人声演唱**；支持上传 ≥10s 参考音频做风格迁移；95 秒/首
  （装 full 权重可选 4m45s 完整版），CPU 约 15-30 分钟，进度可见、可取消、排队渲染。
  生成后自动两步后处理：**Whisper 强制对齐**（每句歌词真实起止时间，卡拉OK跟嘴型）
  + **Demucs 源分离**（人声/鼓/贝斯/伴奏 4 分轨，神经歌曲也能进 DAW）。
  神经作品可下载 WAV/分轨、导出 MV 与分享包（无 MIDI/换副歌——模型不产出音符级曲谱）；
- ☁️ **Mureka 云 API（v5 已接入）**：设置里填 Key 即可用官方云端引擎
  （约 $0.05/首、含商用授权、1-2 分钟出歌），产物照常走本地对齐与分轨；
  `providers/audio_base.py` 抽象保持开放（Suno 第三方代理因合规风险不接入）。

**人声说明**：五种人声形态，按需选择——
1. **人声哼唱**：共振峰合成 aah/ooh 哼唱旋律（默认音色之一）；
2. **🎤 AI 演唱（实验性，默认开启）**：生成的歌只要有歌词（模板/自定义/AI 写词）
   就会用 macOS 自带 TTS（婷婷）把歌词**逐字唱到旋律的每个音符上**
   （基频检测 + 音高重采样 + 长音延音），配合卡拉OK歌词，完全本地零费用，
   首次渲染因逐字合成稍慢（约 10–20s，常用字有磁盘缓存，越用越快）；
   效果定位"机器人 Vocaloid/合唱感"；进阶模式可取消勾选改为纯器乐；
3. **🧠 神经歌声（v4，推荐）**：开源 DiffRhythm 本地推理，**真声级演唱**整首歌词，
   效果对标商业产品的"人声演唱"档（见上方快速开始的安装步骤）；
4. ☁️ **Mureka 云端**（v5）：官方 API 真声，速度快、按首付费；
5. 🎨 **音色 Persona**（v5）：从任意作品提取"同一副嗓子"反复出歌（做专辑）——
   进阶模式选 Persona，或在播放器点「提取音色」。

## 与 Mureka 的能力对照

| 能力 | Mureka（云端付费） | 本项目（本地） |
|---|---|---|
| 一句话生成歌曲 | ✅ 含真声演唱 | ✅ 器乐+人声哼唱 |
| **真声演唱** | ✅ | ✅ **v4：DiffRhythm 本地神经网络** |
| **参考音频风格迁移** | ✅ | ✅ **v4：≥10s 参考音频 → 演唱风格靠拢** |
| **分轨（含神经歌曲）** | ✅ | ✅ v4.1：器乐原生 4 轨 + 神经歌曲 Demucs 分离（Mureka 云端产物同样处理） |
| 简易/自定义双模式 | ✅ | ✅ |
| 自定义歌词 + 段落标记 | ✅ | ✅（按小节对齐旋律） |
| AI 写词 | ✅ | ✅（免费 GLM 或离线模板） |
| 音色/人声选择 | ✅ 多种歌声 | ✅ 5 种器乐 + 哼唱声部（神经引擎由模型决定） |
| 双版本候选 | ✅ | ✅ A/B |
| 再生成 / 延长 | ✅ | ✅（再来一版 / 延长30s） |
| **重唱某一段** | ✅ Replace Section | ✅ **v5：分段重唱**（区间外保留原曲） |
| **音色一致性** | ✅ Personas | ✅ **v5：Persona**（从作品/参考音频提取音色） |
| **专辑批量** | ✅ | ✅ **v5：同一 Persona 连出多首 + 打包** |
| 分轨 Stems | ✅ | ✅ 4 轨 WAV（器乐引擎） |
| 曲库 / 封面 | ✅ | ✅（AI 程序化封面） |
| MIDI 导出 | ❌ | ✅（器乐引擎） |
| 费用 | 金币/会员制 | **0 元**（LLM 可选免费 Key） |

## 免费开启大模型作曲/写词（可选）

1. 到 [open.bigmodel.cn](https://open.bigmodel.cn) 注册并创建 API Key（`glm-4-flash` 为免费模型）。
2. 网页右上角 **⚙️ 引擎设置** → 填入 Key → 保存 → 引擎选"自动"。
   （或复制 `config.example.json` 为 `config.local.json` 填好 Key，
   或用环境变量 `AI_MUSIC_LLM_API_KEY` / `AI_MUSIC_LLM_BASE_URL` / `AI_MUSIC_LLM_MODEL`。）
3. 其他端点示例：DeepSeek `https://api.deepseek.com/v1`（收费）、
   本地 Ollama `http://127.0.0.1:11434/v1`（免费，model 填本地模型名）。
4. **完整的后期配置文档（模型选型 / 真声演唱 API / 常驻运行 / 常见问题）
   见 [docs/CONFIG.md](docs/CONFIG.md)。**

## 项目结构

```
ai-music-studio/
├── brand/                   # 品牌资产（Logo SVG / favicon / 1024px PNG）
├── scripts/run.sh            # 一键启动
├── scripts/setup_neural.sh   # 神经歌声引擎安装（v4，可选）
├── config.example.json       # LLM 配置样例（复制为 config.local.json）
├── docs/DESIGN.md            # 设计文档（架构 / SongSpec 契约 / 路线图）
├── docs/BENCHMARK.md         # 对标研究：商业产品矩阵 + 开源模型选型（v4）
├── DiffRhythm/               # 开源模型（Apache-2.0）+ 预训练权重 + 独立 venv
├── frontend/                 # 单页应用（无框架）：index.html + style.css + app.js
└── backend/
    ├── requirements.txt      # 仅 numpy
    ├── output/               # 作品目录（每首歌一个目录：wav/mid/meta/spec/score）
    ├── neural/               # v4 神经引擎
    │   ├── diffrhythm_runner.py  # 模型推理子进程（进度 JSON 协议）
    │   └── selftest.py           # 安装自检（--mini 微型生成 / --verify-euler）
    ├── tests/                # 79 个单元 + 端到端测试
    └── app/
        ├── server.py         # HTTP API + 静态资源 + Range 流媒体 + Token 鉴权（纯标准库）
        ├── jobs.py           # 异步任务：作曲 → 编曲/神经渲染，前端轮询进度，可取消
        ├── store.py          # 曲库持久化
        ├── schemas.py        # SongSpec 校验/修复（LLM 输出的护栏）
        ├── providers/
        │   ├── mock.py       # 离线算法作曲
        │   ├── llm.py        # OpenAI 兼容 LLM 适配（GLM-4-Flash 免费）
        │   ├── audio_base.py # v4：音频成品 Provider 抽象（云 API 接位）
        │   └── diffrhythm.py # v4：本地神经歌声引擎编排（子进程/进度/超时/取消）
        └── music/
            ├── theory.py     # 音阶/和弦解析（MIDI 编号体系）
            ├── mock_composer.py  # 关键词理解 → 调式 → 和声进行 → 曲式 → 歌词
            ├── realizer.py   # 编曲：旋律动机/贝斯模式/和声垫/鼓组模板
            ├── synth.py      # numpy 加法合成器 → WAV（含简易混响）
            └── midi.py       # 标准 MIDI（SMF format 1）写入
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/compose` | `{description, genre, mood, duration_sec, bpm, provider, lyrics_req?, ref_audio?, full_song?}` → `{job_id}` |
| POST | `/api/lyrics` | AI 写词：`{description, genre, mood, lyrics_req?}` → `{lyrics, source}` |
| GET | `/api/jobs/{id}` | 任务进度（composing/arranging/rendering/singing_neural %） |
| DELETE | `/api/jobs/{id}` | 取消任务（终止神经渲染子进程） |
| POST | `/api/ref-audio` | 上传参考音频（wav/mp3/flac ≤25MB）→ `{ref_audio}` |
| GET | `/api/songs` · `/api/songs/{id}` | 曲库列表 / 曲谱详情 |
| GET | `/api/songs/{id}/audio.wav` · `/song.mid` | 下载（支持 Range 拖动） |
| DELETE | `/api/songs/{id}` | 删除 |
| GET/POST | `/api/config` | 查看/设置 LLM 配置（Key 脱敏） |
| GET | `/api/meta` · `/api/health` | 风格/情绪元数据（含引擎可用性）· 健康检查 |

设置 `AI_MUSIC_TOKEN` 后所有 `/api/*` 需携带 `Authorization: Bearer <token>`
（GET 下载类额外接受 `?token=`）。

## 许可证与致谢

- 本项目代码以 [Apache-2.0](LICENSE) 发布
- 内置的神经歌声引擎基于 [ASLP-lab/DiffRhythm](https://github.com/ASLP-lab/DiffRhythm)（Apache-2.0），
  vendored 副本与修改声明见 [DiffRhythm/NOTICE.md](DiffRhythm/NOTICE.md)
- 对齐使用 [faster-whisper](https://github.com/SYSTRAN/faster-whisper)，分轨使用 [Demucs](https://github.com/adefossez/demucs)

## 测试

```bash
cd backend && python3 -m unittest discover -s tests   # 103 个用例
```

## 关于收费模型（现阶段的取舍）

- **现在**：Mock 引擎 + 免费大模型已覆盖完整闭环，不花一分钱。
- **想要"带人声演唱的成品歌"**（Suno 级别）属于付费 API
  （Suno/Udio 约 $8–30/月，按量约 $0.05–0.5/首）。
  架构已预留：`providers/` 里加一个 `suno.py`，`jobs.py` 渲染阶段改为下载
  外部音频即可，前端零改动。详见 `docs/DESIGN.md` §8。
