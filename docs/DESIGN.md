# 织歌 SongWeaver（AI Music Studio）— 设计文档

> 一个完全本地运行的 AI 音乐创作程序：输入一句描述，产出可播放、可下载（WAV / MIDI）的完整编曲。
> 默认使用 **离线 Mock 作曲引擎**（零配置、零费用），可一键切换到 **免费大模型**（智谱 GLM-4-Flash），
> 并为付费音乐生成 API（Suno / Udio 等）预留了 Provider 扩展位。

## 1. 目标与约束

| 目标 | 做法 |
|---|---|
| 零费用跑通全流程 | Mock 算法作曲 + 本地合成器渲染 WAV，不依赖任何付费 API |
| 接入"AI 大脑" | OpenAI 兼容协议适配器，默认对接免费的 GLM-4-Flash（智谱 bigmodel） |
| 产物真实可用 | 标准 MIDI 文件 + 44.1kHz 立体声 WAV，可导入 DAW |
| 可视化 | 网页端钢琴卷帘（Piano Roll）+ 卡拉OK式歌词 + 分区结构图 |
| 确定性可复现 | 同一 seed 生成完全相同的曲子（Mock 引擎） |

**不做什么（v1）**：人声合成/演唱（需要 Suno 级别的歌声模型，见 §8 路线图）、
云端分布式、用户系统。

## 2. 总体架构

```
┌──────────────────────────── 浏览器 (frontend/) ───────────────────────────┐
│  表单(风格/情绪/时长/速度)  │  播放器  │  钢琴卷帘 Canvas  │  歌词  │  曲库   │
└───────────────┬───────────────────────────────────────────────────────────┘
                │ HTTP (JSON / WAV / MIDI)
┌───────────────▼──────────── 后端 backend/app (Python 标准库 + numpy) ──────┐
│ server.py  HTTP 路由 + 静态资源 + Range 流媒体                             │
│ jobs.py    异步任务队列（生成是慢操作，前端轮询进度）                        │
│ store.py   曲库持久化  output/<id>/{meta,song,score}.json + song.wav/.mid  │
│                                                                            │
│ providers/  ── 作曲 Provider 抽象（核心扩展点）                              │
│   ├─ mock.py     离线算法作曲：关键词理解→和声→结构→歌词                     │
│   └─ llm.py      OpenAI 兼容 Chat API（GLM-4-Flash 免费额度/本地 Ollama…）  │
│                  输出 SongSpec JSON，校验失败自动回落 Mock                   │
│                                                                            │
│ music/                                                                     │
│   ├─ theory.py   音名/音阶/和弦解析（MIDI 编号体系 C4=60）                   │
│   ├─ schemas.py  SongSpec 校验与修复（LLM 输出的"护栏"）                     │
│   ├─ mock_composer.py  离线作曲：情绪→调式→进行→结构→歌词                   │
│   ├─ realizer.py 编曲器：把高层规格展开为音符事件（旋律/贝斯/和声垫/鼓）       │
│   ├─ synth.py    加法合成器（numpy 向量化）→ 44.1kHz 立体声 WAV              │
│   └─ midi.py     标准 MIDI 文件写入（SMF format 1）                         │
└────────────────────────────────────────────────────────────────────────────┘
```

### 生成流水线

```
用户描述 ──► Provider(mock/llm) ──► SongSpec（高层乐谱）
                                     │  调性/速度/曲式/和弦进行/歌词
                                     ▼
                                realizer.py 编曲展开
                                     │  旋律(动机重复+轮廓控制)、贝斯、
                                     │  和声垫/琶音/律动、鼓组(16步+填充+摇摆)
                                     ▼
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
              midi.py → song.mid               synth.py → song.wav
```

**关键设计：LLM 只负责"作曲决策"（和声、结构、歌词、速度、调式），
音符级展开交给确定性编曲器。** 这样即使 LLM 输出不合规，也有 schemas 校验修复 +
Mock 回落双保险，永远不会产出"坏歌"。

## 3. SongSpec（Provider 输出契约）

```jsonc
{
  "version": 1,
  "title": "晚风来信",
  "description": "一首适合夏天傍晚的轻快小曲",
  "genre": "pop",            // pop|rock|electronic|ballad|folk|guofeng|lofi|jazz
  "mood": "happy",           // happy|sad|chill|energetic|romantic|epic
  "key": "C",                // 调性主音 C..B
  "mode": "major",           // major|minor|dorian|chinese(五声音阶)
  "bpm": 104,
  "beats_per_bar": 4,
  "structure": [             // 曲式：每个 section 一行和弦(每小节一个)
    { "name": "intro",  "bars": 2, "chords": ["C","C"],       "lyrics": ["",""] },
    { "name": "verse",  "bars": 8, "chords": ["C","G","Am","F","C","G","Am","F"],
      "lyrics": ["清晨的光落在窗台","背包里装满了期待","风把云朵吹得散开","脚步轻快像在比赛",
                 "清晨的光落在窗台","背包里装满了期待","风把云朵吹得散开","脚步轻快像在比赛"] },
    { "name": "chorus", "bars": 8, "chords": ["F","G","C","Am","F","G","C","C"],
      "lyrics": ["…"] }
  ],
  "provider": "mock",        // 谁产出的：mock | llm
  "seed": 1712345678
}
```

`schemas.validate_and_repair_spec()` 是所有 Provider 输出的必经之路：
类型收敛、BPM 钳制(60–184)、和弦符号合法性校验、按目标时长增删小节、
歌词按小节对齐。无法修复 → 抛 `SpecError` → 上层回落 Mock。

## 4. 编曲器（realizer.py）规则摘要

- **旋律**：按 section 家族（主歌/副歌…）维护"动机"（节奏型 + 轮廓种子），
  4 小节乐句正弦弧线控制音高走向；强拍吸附和弦音，弱拍走音阶级进；
  段落结尾强制落回主音。副歌整体比主歌高 2 个级度、力度 +8。
- **贝斯**：每风格一种模式（流行根音切分 / 摇滚八分推进 / 电子 octave 跳跃 /
  民谣根-五度 / 古风长音 / lofi 慵懒切分 / 爵士行走贝斯）。
- **和声垫**：每小节全音符和弦（三和弦/七和弦按声位堆叠在 48–59 音区）；
  电子/民谣/古风附加八分琶音，lofi/爵士改为反拍键琴 stab。
- **鼓组**：16 步/小节风格模板 + 爵士/lofi swing 偏移 + 段落结尾军鼓填充；
  引入段落规则（流行从主歌进鼓，电子从头进鼓）。
- **古风**：chinese 五声音阶旋律 + 古筝式 pluck 音色 + 弦乐垫。

## 5. 合成器（synth.py）

numpy 加法合成：每个音符 = 谐波叠置（每谐波独立衰减率）× ADSR/指数包络
× 颤音(延迟启动) × 声像；鼓组为参数化合成（kick 频率扫描、snare 噪声+腔体音、
hat 微分高通噪声、clap 三连脉冲）。总线：软削波 tanh + 三抽头简易混响 + 峰值归一。

音色预设：synth_lead / square_lead / piano / keys(Rhodes) / pluck(古筝·吉他) /
warm_pad / strings / sub_bass，按风格映射。

## 6. Provider 抽象

```
provide(req) -> (SongSpec | None, info)
```

| Provider | 说明 | 配置 |
|---|---|---|
| `mock` | 离线算法作曲，永远可用，确定性 | 无 |
| `llm` | OpenAI 兼容 `/chat/completions`；默认智谱 `glm-4-flash`（**免费**），也支持 DeepSeek/OpenAI/本地 Ollama | 环境变量 `AI_MUSIC_LLM_BASE_URL/API_KEY/MODEL` 或 `config.local.json` |
| 付费（预留） | Suno / Udio / ACE Studio：返回成品音频而非 SongSpec，接入点是 `jobs.py` 渲染阶段旁路 | 未来 |

## 7. API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/compose` | 创建生成任务 → `{job_id}` |
| GET | `/api/jobs/{id}` | 轮询任务（stage: composing/arranging/rendering x%） |
| GET | `/api/songs` | 曲库列表 |
| GET | `/api/songs/{id}` | 曲谱（spec + 展开后的音符事件，供钢琴卷帘） |
| GET | `/api/songs/{id}/audio.wav` | WAV（支持 Range，可拖动进度条） |
| GET | `/api/songs/{id}/song.mid` | MIDI 下载 |
| DELETE | `/api/songs/{id}` | 删除 |
| GET/POST | `/api/config` | 查看/设置 LLM 配置（key 脱敏返回） |
| GET | `/api/config/meta` | 风格/情绪/默认值元数据 |

## 8. 收费大模型说明与路线图

**本项目默认配置即可完整使用，不花一分钱：**
1. Mock 引擎：纯离线，无任何网络请求。
2. GLM-4-Flash（智谱 open.bigmodel.cn）：免费模型，注册即有 API Key，
   填入配置即可让"AI 作曲决策"更强（更贴合描述的和声/歌词/结构）。

**若要达到 Suno 级"带人声成品歌"**，属于付费 API（Suno 约 $8–30/月起，
按量约 $0.05–0.5/首）。架构已预留：在 `providers/` 增加 `suno.py`
实现 `provide()` 返回外部音频 URL，再在 `jobs.py` 把下载代替本地渲染即可，
前端无需改动。

## 9. v2 增补（对标 Mureka 后的迭代）

参考 Mureka（mureka.ai/cn）的产品形态补齐的能力，全部保持"本地可跑、零费用优先"：

| v2 能力 | 实现要点 |
|---|---|
| 简易/进阶双模式 | 进阶 = 歌名 + 歌词编辑器 + 音色/调性 + 双版本 |
| 自定义歌词 | `schemas.parse_lyrics_text()` 解析 `[Verse]/[Chorus]/[Bridge]`（中英文标记）→ `apply_lyrics()` 按段落对齐到每小节；结构缺失的段落歌词并入自动池不丢失 |
| AI 写词 | `POST /api/lyrics`：LLM（GLM 免费模型）优先，`mock_composer.mock_lyrics_text()` 模板兜底（自动把用户描述揉成副歌钩子句） |
| 人声哼唱声部 | `synth.WAVES["vocal_hum"]`：共振峰(formant)合成，aah/ooh 每 2 拍交替 + 气息噪声 + 5.4Hz 颤音；MIDI 对应 GM 53 (Voice Oohs) |
| 音色/调性选择 | `spec.lead_voice` 覆盖风格默认音色（synth + MIDI 程序号联动）；调性 12 锁定 |
| 双版本候选 | 前端对同一参数用两个相邻 seed 各发一单，A/B 对比 |
| 分轨 Stems | `synth.render_stems()`：声部分组（vocal/harmony/bass/drums）各自渲染归一化成干声 WAV；任务流水线在整曲渲染后自动导出，`GET /api/songs/{id}/stems/{name}.wav` |
| 再来一版/延长 | 前端基于 `lastParams`：换 seed 重生成 / 加 30 秒同 seed 续构 |
| AI 封面 | 前端 canvas 按 song id 哈希确定性绘制（渐变+圆环+均衡条+标题），作品缩略图与 PNG 下载共用 |
| AI 演唱（实验） | `music/singer.py`：歌词逐字 → macOS `say`(婷婷) 合成单字（内存+磁盘双缓存 + 8 并发预热）→ FFT 测基频 → 重采样到旋律音高 → 按音符时长截断/尾音延音 → 替代器乐主旋律混入主唱声部；**`sing` 缺省视为开启**（"生成的歌要唱出歌词"是默认行为），显式传 false 或取消勾选则纯器乐；非 macOS 自动跳过 |

**仍然明确的边界**：真声演唱、参考音频风格迁移、声音克隆需要
Mureka/Suno 级付费模型，未在本项目实现；`providers/` 抽象已为其留位。

## 10. v3 增补（音乐质量 / 性能 / 效率三线优化）

| 类别 | 能力 | 实现要点 |
|---|---|---|
| 音乐 | 母带链 | `_master()`：混响 → FFT 静态 EQ(去28Hz隆隆声+高频存在感) → 100Hz 块级总线压缩(3:1 快攻慢放) → tanh 软限幅 → 峰值归一 |
| 音乐 | 人声侧链避让 | `_mix_master()`：人声 100Hz 包络 → 和声/贝斯最多避让 2.5dB，人声永远清晰 |
| 音乐 | 段落动态 | intro 渐入(0.55→1)/bridge 收(0.85)/outro 渐出(1→0.5) 增益曲线作用于人声+和声；intro/outro/bridge 抽掉琶音与键琴；ballad/folk/pop 桥段抽鼓 |
| 音乐 | 和声音乐性 | 主歌末小节半终止(属和弦)推向副歌；outro 落主和弦；桥段整体上移两度制造调性对比 |
| 音乐 | 旋律问答 | 乐句弧线 + 偶数小节上扬/奇数小节回应；旋律随机源与编曲随机源分离 |
| 音乐 | 演唱人性化 | 滑音(衔接紧的相邻音头 90ms 从上音高滑入)、字头起音随机化、力度/微时值抖动(±9ms)、句尾气声、超长音节自然压缩 |
| 音乐 | 立体声 | 琶音/键琴 ping-pong 左右交替，镲片微摆动 |
| 性能 | 渲染一次化 | `render_all()`：母带与 4 分轨共用同批 float32 组 buffer（原先整曲渲染两遍，现耗时/内存约减半，长曲峰值 ~650MB→~250MB） |
| 性能 | 渲染并发上限 | `JobManager` 信号量最多同时渲染 2 首，其余排队提示 |
| 性能 | 结果缓存 | 请求指纹（参数+seed 的 md5）命中即复用已有作品，秒回 |
| 性能 | TTS 预热 | 服务启动后台预合成歌词模板常用字（磁盘缓存跨歌曲复用） |
| 性能 | LLM 缓存 | 写词同参数不再重复调用；429/5xx 退避重试一次 |
| 功能 | MV 导出 | `GET /api/songs/{id}/video.mp4`：服务端 numpy 手绘 PNG 封面（coverpng.py，无 PIL 依赖）+ ffmpeg 合成 mp4，结果缓存 |
| 功能 | 分享包 | `GET /api/songs/{id}/bundle.zip`：整曲+MIDI+4分轨+封面+歌词+meta 打包 |
| 功能 | 换副歌 | `POST /api/songs/{id}/regen_chorus`：realizer 按"段落家族"独立播种旋律（family_seeds），只换副歌种子 → 主歌/和声/鼓完全一致 |
| 功能 | 作品管理 | PATCH 改名（同步 meta/spec/score）、收藏置顶、前端搜索过滤 |
| 功能 | 频谱可视化 | 前端 Web Audio AnalyserNode + canvas 实时频谱 |

## 11. v4 增补（神经歌声引擎 + 生产加固）

对标调研见 [BENCHMARK.md](BENCHMARK.md)。两条主线：

### 11.1 音频成品 Provider 旁路 + DiffRhythm 本地引擎

```
_req (provider=diffrhythm)
  └─► jobs._compose          LLM/Mock 产出歌词+结构（沿用原有作曲决策）
      └─► jobs._run_audio    【旁路：不走 realizer/synth/midi】
            provider.generate(req, spec, out_wav, progress_cb, cancel_event)
              ├─ build_lyric_lines(spec)   小节时间( beats_per_bar*60/bpm ) → LRC 时间戳
              ├─ build_style_prompt(spec)  genre/mood 英文关键词 + 用户描述
              ├─ ref_audio?                ≥10s 参考音频 → MuQ-MuLan 风格嵌入（风格迁移）
              └─ subprocess: backend/neural/diffrhythm_runner.py（DiffRhythm venv, torch CPU）
                   stdout JSON 行协议: progress{frac,note} / done{wav,duration_sec} / error
            lite_score(spec, duration)   轻量曲谱：段落/歌词/和弦齐全、音符数组为空
            store.save(...)              meta.audio_only=true；卡拉OK/封面/MV/zip 全兼容
```

| 设计点 | 决策与理由 |
|---|---|
| 通用抽象 | `providers/audio_base.py`：`available()` + `generate()` 契约；Mureka/Suno 云 API 实现同一接口即可接入，`jobs._run_audio` 零改动 |
| 子进程隔离 | 主应用保持"仅 numpy"轻依赖秒启动；模型栈（torch 等约 4GB）在独立 venv，崩溃不影响服务 |
| fp32-on-CPU | 上游 `load_checkpoint` 强制 fp16；Intel CPU 无高效 fp16 kernel → runner 转 fp32（GPU 路径不动） |
| 手动 Euler | 复刻 `CFM.sample` 的 odeint-euler 展开步进：①真实逐步进度 ②只保留当前步张量省 ~100MB ③`selftest.py --verify-euler` 数值等价校验 |
| LRC 时间戳 | 模型静默丢弃无时间戳歌词行；末尾加哨兵行抵消 2048 帧下"丢最后一行"的模型行为 |
| 双信号量 | 本地渲染 2 并发（`_slots`）、神经渲染 1 并发（`_neural_slot`，模型占 6-8GB 内存）互不挤占 |
| 取消 | `DELETE /api/jobs/{id}` → `cancel_event` → watchdog kill 子进程 → 任务置 canceled；各阶段边界都有检查点 |
| 前端降级 | `score.audio_only` → 钢琴卷帘画"段落+歌词时间轴"，隐藏 MIDI/分轨/换副歌/延长；卡拉OK、封面、MV、bundle 正常 |

### 11.3 v4.1 后处理链 + 写词要求

| 能力 | 实现 |
|---|---|
| 歌词强制对齐 | 生成后 faster-whisper 转写（int8 CPU，词级时间戳）→ `align_runner.py` 内 Needleman-Wunsch 单调字符对齐（时间带约束防漂移）→ 每句歌词真实起止时间写入 `score.lyric_events`，置信度 <0.35 的行回落小节估算；前端卡拉OK按播放秒数跟唱 |
| Demucs 源分离 | `demucs_runner.py`（htdemucs）→ vocals/drums/bass/other 映射为 stem_vocal/drums/bass/harmony（神经作品的"和声"轨即伴奏）；失败不影响成品 |
| 全曲 285s | full 权重独立文件（`cfm_model_full.pt`），vendored `prepare_model` 按帧数选载（防错载）；超时自动 ×3；UI 开关仅在权重就绪时出现 |
| 引擎探测 | faster_whisper/demucs 可用性后台探测（`/api/meta` 永不阻塞），流水线内同步等待（wait=True） |
| 说唱类型 | rap 档案（小调 loop / BPM 自动分 boom-bap 与 trap 双鼓组 / 808 贝斯 / 稀疏暗色动机 / 默认纯 beat） |
| 写词要求 | `lyrics_req` 字段贯通：AI 写词与 LLM 作曲双路遵循，计入结果指纹；文本亦参与 mock 的风格/情绪识别 |

### 11.2 生产加固（局域网部署形态）

| 项 | 实现 |
|---|---|
| 强制 Token | `AI_MUSIC_HOST` 非本机回环时未设 `AI_MUSIC_TOKEN` → 拒绝启动（exit 2） |
| 鉴权 | `/api/*` Bearer（`hmac.compare_digest`）；GET/HEAD 媒体兼容 `?token=`；静态页开放（前端从 URL 取 token 自动附加） |
| 传输 | 文件分块流式（1MB chunks）；Range 后缀语义修复 + 416 |
| 输入 | JSON 1MB / 音频 25MB 上限 → 413；上传文件名白名单净化 + 服务端生成；magic bytes 校验格式 |
| 静态防护 | `os.path.commonpath` 替换 `startswith` 前缀检查 |
| 出站安全 | LLM TLS 默认严格校验（certifi），显式 `AI_MUSIC_INSECURE_SSL=1` 才降级 |
| 并发 | `POST /api/config` 加锁；响应统一 `nosniff` |
| 测试 | 配置隔离修复（显式 config path 不回落项目根），测试不再触碰开发者真实 LLM 配置 |


## 12. v5 增补（本地极致 + 云 API）

| 能力 | 实现 |
|---|---|
| 常驻神经 worker | `diffrhythm_runner.py --serve`：模型常驻（~10GB），stdin 任务协议，空闲 30min/50 任务自杀；`NeuralWorkerHandle` 懒启动/崩溃自动重启/超时取消即杀（代价=下次重载）；连续生成每首省 1-2 分钟 |
| 音色 Persona | MuQ-MuLan 嵌入（L2 归一 [1,512]）缓存为 .npy：从作品人声分轨 / 参考音频 / 文本描述三种来源提取（`persona_runner.py` 只加载 MuQ）；生成任务 `style_prompt_npy` 直接替换文本风格向量 |
| 分段重唱 | DiffRhythm edit 模式：参考曲编码为 latent（`vae_sample` 消耗 RNG → 任务开头先播种），`fixed_span_mask` 圈区间，`step_cond=where(mask,0,cond)` + 采样后 `where(mask,sampled,cond)` 缝合；整首 LRC 只改目标段文本；产出为新作品（不覆盖原曲）；未编辑段为 VAE 重建（音质细微差异，已文档化） |
| 专辑 | `POST /api/album`：同一 Persona 顺序排队 N 首（神经槽=1 串行）；`GET /api/album.zip?ids=` 打包 WAV+封面+歌词册 |
| Mureka 云 API | `providers/mureka.py` 实现 AudioProvider：generate 提交→轮询 query→MP3 下载→ffmpeg 转 WAV，之后复用本地对齐+分轨；`/v1/files/upload` 参考音频→reference_id；响应字段容错解析 |
| 配置 | `mureka` 节（enabled/api_key/model/vocal_gender）+ env `AI_MUSIC_MUREKA_KEY`；设置弹窗云 API 区（key 脱敏） |

