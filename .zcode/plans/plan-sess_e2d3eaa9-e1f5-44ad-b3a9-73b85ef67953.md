# v5：本地做到极致 + 云 API 接入（对标 Suno/Mureka）

## Context

产品四引擎已闭环（含说唱、对齐、分轨、全曲），但对标 Suno/Mureka 仍有四大差距：**每次生成重载模型 1-2 分钟**、**无法复用音色**（每首都是新嗓子，做不了专辑）、**无法只重唱某一段**（Suno 的 Replace Section）、**云 API 只有抽象没实现**。调研确认：Mureka 官方 REST 是最佳云接入（~$0.05/首含商用授权）；Suno 第三方代理有合规风险，规避；本地引擎继续用 DiffRhythm 1.2（YuE2-GGUF 列入观察）。

用户已确认优先级：**先本地极致，再云 API**。本计划两阶段一起做，第三阶段（分享页/新引擎接入）只进路线图。

## Phase 1 — 本地极致

### 1.1 常驻神经工作进程（一切的地基）
- `backend/neural/diffrhythm_runner.py` 加 `--serve` 模式：模型加载后吐 `{"type":"ready"}`，循环读 stdin 的 job 文件路径逐个执行，stdout JSON 行协议加 `"id"` 回显；每任务开始时重置 seed（torch+random，编辑模式必须在参考编码前）；stderr 改写日志文件（保留崩溃线索）；空闲 30 分钟/50 任务后自杀
- `backend/app/providers/diffrhythm.py`：`NeuralWorkerHandle` —— 懒启动、单飞、EOF/超时标记死亡并在下次调用自动重启（重启代价=1 次模型加载）；watchdog 从"杀进程"改为"标记放弃当前任务"；`_run_subprocess_job` 保留给 align/demucs（后续同样常驻化，本次先不动）
- 收益：连续生成/批量时每首省 1-2 分钟

### 1.2 音色一致性（Persona，对标 Suno Personas）
- 新 `backend/neural/persona_runner.py`：只加载 MuQ（~2.4GB，远快于全家桶），输入 wav（优先 demucs 人声分轨，兜底整曲）或文本描述 → 单位范数 [1,512] fp32 嵌入 → `np.save`
- 存储：`backend/output/personas/<id>/` = `embedding.npy` + `meta.json`（名称、来源歌曲/音频、创建时间）
- runner 生成任务新增 `style_prompt_npy` 字段：加载后 `.to(dt)` 直接替换文本风格向量（调研确认 drop-in，向量已归一，负向 vocal.npy 保持不变）
- API：`GET/POST/DELETE /api/personas`（POST 支持三种来源：song_id=从作品提取 / ref_audio=从上传音频 / text=文字描述）
- 前端：进阶模式新增「音色 Persona」选择行（无/已有列表）；播放器面板加「🎨 从这首歌提取音色」按钮；生成时 persona 嵌入随任务下发

### 1.3 分段重唱（对标 Suno Replace Section）
- `diffrhythm_runner.py` 支持 edit 任务：`get_reference_latent(edit=True, ref_song, segments)`（**先播种再编码**——`vae_sample` 消耗 RNG）、`fixed_span_mask` 由区间构建、`step_cond = where(mask, 0, cond)`、采样后 `sampled = where(mask, sampled, cond)` 缝合（缺这两步分别导致"无视原曲"与"丢失未编辑段"）；歌词传整首 LRC、只改目标段文本
- provider 前置校验：参考曲时长 ≥ audio_length（否则 cfm.py:176 深处报广播错误，必须提前拦截）
- 管道：新端点 `POST /api/songs/{id}/reseg` `{section_index, new_lyrics, persona_id?}` → 新任务走 edit 渲染 + 对齐 + 分轨 → 产出为**新作品**（标题加"· 重唱版"，不覆盖原曲）
- 前端：播放器面板「🎤 重唱片段」→ 弹层选段落（lite score 的 sections 带秒距）+ 该段新歌词 + 可选音色

### 1.4 批量/专辑（薄版）
- `POST /api/album` `{persona_id, count, theme}` → 顺序排队 N 个神经任务（复用 _neural_slot=1），全部带同一 persona 嵌入
- 前端：Persona 行加「以该音色生成专辑」入口；完成后一键打包 album zip（全部 WAV+封面+歌词册）
- 注意内存：常驻 worker ~10GB + demucs 瞬时 ~2GB，串行执行无压力

## Phase 2 — Mureka 云 API（AudioProvider 首个实现）

- 新 `backend/app/providers/mureka.py`：实现 `AudioProvider`——`POST /v1/song/generate`（lyrics/prompt/reference_id/vocal_gender/instrumental/model）→ 轮询 `GET /v1/song/query/{task_id}` → 下载 MP3 → 转码 WAV（ffmpeg，现成）落 `store.wav_path`；参考音频走 `POST /v1/files/upload` 得 `reference_id`（复用现有 /api/ref-audio 上传）；纯 stdlib urllib，带 429/5xx 退避（对齐 llm.py 的 `_chat` 模式）
- `jobs.py`：`mureka` 加入 `AUDIO_PROVIDERS` 集合，自动复用 `_run_audio` 全链路（进度/取消/对齐/分轨——Mureka MP3 下载后照样跑 Whisper 对齐与 Demucs 分轨，补齐它自身不给的部分）；meta 记录 `provider="mureka"` 与任务 id
- 配置：`config.local.json` 增 `mureka: {enabled, api_key, model}`；env `AI_MUSIC_MUREKA_KEY`；设置弹窗加「云 API」区（key 脱敏、余额提示留待后续）
- 前端：引擎下拉加 `☁ Mureka（云 · ~$0.05/首）`（未配 key 时禁用+提示）；无 key 时一切照旧
- 成本透明：meta 记录每次云端任务（供日后账单页）

## 不做（进 BENCHMARK 路线图）
分享页导出、YuE2-GGUF/HeartMuLa 引擎接入（CC BY-NC 许可待评估）、ACE-Step 服务器模式、封面 AI 绘图。

## 关键文件
- 改：`backend/neural/diffrhythm_runner.py`（serve/edit/npy/seed）、`backend/app/providers/diffrhythm.py`（worker 管理/persona/edit/校验）、`backend/app/jobs.py`（persona/edit/album/mureka 路由）、`backend/app/server.py`（personas/album/reseg/settings 路由）、`backend/app/store.py`（personas 目录）、`frontend/index.html`+`app.js`（persona 行/重唱弹层/专辑入口/设置云区/引擎项）、`config.example.json`
- 新：`backend/neural/persona_runner.py`、`backend/app/providers/mureka.py`、`backend/app/personas.py`（存储层小模块）
- 复用：`_run_subprocess_job`（persona_runner 走它）、`_run_audio`（mureka 全链路复用）、lite_score/对齐/分轨（edit 产物复用全套后处理）、`/api/ref-audio`（persona 音频来源）

## 测试
- 桩测试沿用现有模式：worker 协议（假 python 循环进程：ready→任务→done）、persona 存储 CRUD、reseg 参数校验（曲长不足 400 拦截）、mureka 客户端（本地 http.server 假 Mureka：submit→poll→下载）
- 真机验证：① 连续两首生成，第二首无模型加载日志（常驻生效，计时对比）② 从《夜班不低头》提取 persona → 用该音色生成新歌 ③ 对《夜班不低头》副歌做重唱（新歌词）→ 验证区间外音频不变、对齐分轨自动完成 ④ Mureka 无真实 key，桩测试为准，真机留待用户配 key
- 全量回归 ≥95 用例保持绿

## 风险
- 常驻进程内存 ~10GB 长驻：空闲自杀 + 任务计数回收兜底；32GB 机器无压力
- edit 模式 VAE 往返会轻微改变未编辑段音质（采样性重建，非比特精确）——文档明示
- Mureka 契约来自官方文档抓取，实际字段以联调为准（客户端层做字段容错）
