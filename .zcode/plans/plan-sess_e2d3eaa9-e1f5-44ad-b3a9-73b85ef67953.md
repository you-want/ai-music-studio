# AI 音乐工坊 → 生产级产品（v4：真声演唱引擎 + 生产加固）

## Context

项目已有完整 v3 闭环（Mock/LLM 作曲 → numpy 合成渲染 → TTS 演唱/stems/MV），但对标 Suno/Mureka 仍缺"真人声演唱"。调研结论：开源 **DiffRhythm**（Apache-2.0，1.07B）是最优本地引擎，**2.8GB 权重已下载**但零集成。用户决策：**① 接入本地 DiffRhythm + 通用音频 Provider 抽象（为云 API 预留）；② 部署形态 = 局域网 + 简单 Token**。参考对标（Suno/Udio/Mureka/AIVA/Soundraw、ACE-Step/YuE）沉淀为 BENCHMARK 文档。

关键事实：本机只有 Python 3.14（无法装 torch）；Intel Mac torch 上限 2.2.2（需 Python ≤3.12，用已装的 `uv` 装 3.11）；espeak-ng ✓、ffmpeg ✓；DiffRhythm 权重为 2048 帧（95 秒）变体；CPU 推理每首约 10–40 分钟（进度可见 + 排队）。

## Workstream 1 — 音频成品 Provider 抽象 + DiffRhythm 引擎（核心）

**新文件 `backend/app/providers/audio_base.py`**：`AudioProvider` 抽象 —— `available() -> (bool, reason)`、`generate(req, spec, out_wav_path, progress_cb) -> None`（产出成品 WAV，而非 SongSpec 渲染）。云端付费 API（Mureka/Suno）日后实现同一接口 HTTP 轮询即可，`docs/CONFIG.md` §4 已预留此形态，直接对齐。

**新文件 `backend/neural/diffrhythm_runner.py`**（子进程入口，跑在神经 venv 里）：
- argv 收 JSON：歌词行（含小节秒距）、风格 prompt（genre/mood 中文标签 + description）、seed、steps、输出路径、可选参考音频路径
- 由小节时间（`beats_per_bar*60/bpm`）生成带 `[mm:ss.xx]` 时间戳的 LRC（DiffRhythm `parse_lyrics` 只认时间戳行，纯文本会被静默丢弃 —— 这是必须自己做 LRC 的原因）；歌词超 95s 按小节裁剪
- cwd=DiffRhythm 根目录复用 `infer/infer_utils.py` 的 `prepare_model/get_lrc_token/get_style_prompt/get_negative_style_prompt` 与 `infer/inference()`；对 vendored 代码做**最小标注补丁**：CPU 下 fp32（fp16 在 Intel CPU 慢且有算子风险），由环境变量 `DIFFRHYTHM_CPU_FLOAT=1` 控制，默认不动 GPU 路径
- stdout 按行吐 JSON 进度（load/muq/cfm 32步×CFG/vae 各阶段 frac），结束写 `{"type":"done","wav":...}`
- `HF_HOME=DiffRhythm/pretrained/hf`（MuQ 首次文本编码要从 HF 拉 xlm-roberta tokenizer，setup 脚本预下载）
- steps 默认 16（config 可调 8–32）；时长固定 95s（4m45s full 变体需另下权重，列为路线图）

**新文件 `backend/app/providers/diffrhythm.py`**：可用性探测（venv python + 权重文件存在）；`generate()` 起 `subprocess.Popen` 跑 runner，逐行解析进度 → progress_cb，超时（默认 75 分钟，config 可调）与崩溃处理；输出 WAV 是 44.1kHz（与现有管线一致，直接落盘）。

**改 `backend/app/jobs.py`**：
- `STAGES_ZH` 增 `"singing_neural"`（提示首次加载 1–2 分钟、CPU 渲染 10–40 分钟）；新增 `self._neural_slot = BoundedSemaphore(1)`（神经任务单独限 1 并发，不占普通渲染槽）
- `_run()` 分支：`provider` 属音频类 → 仍走 `_compose()`（LLM/mock 产出歌词与结构），但跳过 realizer/synth/midi 块，改为 `provider.generate()` → WAV 拷入 `store.wav_path`；spec 标 `provider="diffrhythm"`、`audio_only=True`
- 新增 `DELETE /api/jobs/{id}` 支持：杀神经子进程、任务置 canceled

**改 `backend/app/store.py`**：`save()` 加 `score=None` 分支 —— 跳过 score.json 或写"lite score"（sections + 每段歌词秒距、`melody:[]` 等空音符数组），`duration_sec` 用 `wave` 模块从 WAV 读；`meta` 增 `audio_only`。这样卡拉OK歌词、段落高亮、封面、MV、bundle 全部照常工作，仅钢琴卷帘不画音符。

**改 `backend/app/schemas.py`**：`parse_request`（schemas.py:75）provider 白名单 += `"diffrhythm"`。

**改 `backend/app/server.py` `/api/meta`**：返回 `engines: {diffrhythm: {available, reason}}`（探测结果缓存 60s，探测 3s 超时防卡接口）。

**配置**：`config.local.json` 增可选 `neural: {enabled, steps, timeout_min, python}`；env `AI_MUSIC_NEURAL_PYTHON` 指定 venv 解释器；`config.example.json` 同步。

## Workstream 2 — 前端适配（frontend/）

- `index.html`（provider 下拉 79-84 行）：增 `🧠 神经歌声 DiffRhythm` 选项；`/api/meta` 报不可用时禁用 + tooltip 说明安装方法
- `app.js`：
  - 徽章三态：`app.js:273`（播放器）与 `app.js:674`（曲库卡）provider→mock🧩/llm🤖/diffrhythm🧠
  - `drawRoll()`（app.js:319-429）与 `state.bpm = data.score.bpm`（app.js:263）判空：audio_only 时只画段落底色+歌词时间轴，隐藏音符图例与 MIDI 链接（stems 的 HEAD 探测已有，自动禁用）
  - 神经任务显示已耗时 mm:ss + 取消按钮（调新 DELETE /api/jobs）
  - 进阶模式：神经引擎选中时显示"可选：上传 ≥10s 参考音频做风格迁移"（见 W3 上传端点）
- 设置弹窗增"神经引擎"区：可用性状态、steps 说明

## Workstream 3 — 生产加固（局域网 + Token）

**改 `backend/app/server.py`**：
- **Token 鉴权**：`AI_MUSIC_TOKEN` 设置后所有 `/api/*` 需 `Authorization: Bearer`；GET 媒体类（wav/mp4/png/zip/mid）额外接受 `?token=`（`<audio>` 标签带不了 header），`hmac.compare_digest` 比较；启动时若 `AI_MUSIC_HOST=0.0.0.0` 且无 token → 拒绝启动并打印提示（防裸奔）
- 静态文件路径守卫改 `os.path.commonpath`（修 `startswith` 前缀穿透，server.py:441-444）
- `_json_body` 加 1MB 上限 → 413（server.py:94-101）
- Range 修复：后缀 `bytes=-N` 语义、start≥end → 416（server.py:415-417）；`_send_file` 改 1MB 分块流式发送（server.py:103-111）
- 响应头 `X-Content-Type-Options: nosniff`；`STATE.config` 变更加锁（server.py:176）

**改 `backend/app/providers/llm.py`**：SSL 默认开启校验（certifi 存在则用之），仅显式 `AI_MUSIC_INSECURE_SSL=1` 才关闭（llm.py:55-64 现在默认不安全）。

**新端点 `POST /api/ref-audio`**：上传参考音频（raw body，≤25MB，wav/mp3/flac，magic bytes 校验）→ 存临时目录 → 神经引擎走 DiffRhythm `get_style_prompt` 音频路径（≥10s 自动取中段）。这把 README 能力表里"参考音频风格迁移 ❌"变成 ✅。上传文件固定放库目录内、文件名服务端生成（无路径注入面）。

**可选（最后做）**：`Dockerfile`（linux/amd64，torch CPU，供以后上服务器/GPU 机器）。

## Workstream 4 — 脚本与文档

- **新 `scripts/setup_neural.sh`**：`uv python install 3.11` → `uv venv DiffRhythm/.venv-neural` → 装 `torch==2.2.2 torchaudio==2.2.2` + 精简依赖（`numpy<2, transformers==4.49.0, x-transformers==2.1.2, torchdiffeq==0.2.5, ema-pytorch, librosa, muq==0.1.0, jieba, pypinyin, cn2an, pykakasi, phonemizer, Unidecode, inflect, py3langid, onnxruntime, tokenizers, mutagen, accelerate`；去掉 bitsandbytes/pylance/pyarrow）→ 预下载 HF tokenizer → 幂等可重复执行
- **新 `backend/neural/selftest.py`**：venv 内跑 —— 加载三模型（打印耗时/内存）、LRC 编码、可选 `--mini` 单步小样本出声，验证 torch 2.2.2 兼容性（x-transformers/SDPA 若有不兼容在此暴露并记 CONFIG.md）
- **新 `docs/BENCHMARK.md`**：完整对标研究 —— 商业产品功能矩阵（Suno/Udio/Mureka/AIVA/Soundraw vs 本项目逐项）、开源模型选型依据（含 CPU 可行性数据与来源链接）、差距路线图（声音克隆、翻唱 covers、demucs 12-stem 分离、时间线编辑器、ACE-Step GPU 路线）
- **更新**：README（能力表三行翻绿：真声演唱✅/参考风格迁移✅/双引擎→四引擎、快速开始加 `--neural`）、DESIGN.md §11 v4 架构补记、CONFIG.md（神经引擎安装/排障、Token 部署、云 API 接位说明）

## 测试（backend/tests/，沿用 create_server + 端口 0 模式）

- 新 `test_neural_provider.py`：stub runner（假 python 脚本吐进度 JSON + 写小 WAV）→ 走通 jobs 分支端到端：任务完成、meta `provider="diffrhythm"`/`audio_only=true`、lite score 结构、/audio.wav 可 Range 播放；可用性探测失败时选项禁用；LRC 生成器单测（小节→时间戳数学、裁剪）
- 新 `test_server_hardening.py`：Range 后缀修复、413、路径穿越 403、token 401/200（header 与 ?token= 两路）
- 33 个既有用例必须全绿；`/api/meta` 增字段不破坏既有断言

## 验证（端到端）

1. `cd backend && python3 -m unittest discover -s tests` 全绿
2. `bash scripts/setup_neural.sh && .venv-neural python backend/neural/selftest.py` —— 模型加载成功并打印耗时（预计 1–2 分钟、~6-8GB RAM）
3. 真机神经生成：起服务 → `provider=diffrhythm` 提交（短歌词）→ 轮询看到 singing_neural 阶段进度 → 得 95s/44.1kHz WAV；浏览器播放、卡拉OK、封面、MV 导出、bundle.zip 均正常；钢琴卷帘无报错（仅段落视图）
4. Token 模式：`AI_MUSIC_HOST=0.0.0.0 AI_MUSIC_TOKEN=test` → 无 token 401、带 token 200、`<audio src="...?token=test">` 可播；本机 127.0.0.1 无 token 照常
5. 一次 CPU 实测后把真实耗时写回 README/BENCHMARK

## 风险与对策

- **torch 2.2.2 兼容**：DiffRhythm 按 torch 2.6 写 —— selftest 先行暴露；已知差异点（fp16→fp32、torchaudio 2.2 API 一致）都有对位处理；实在不行回退 pinned 老版 x-transformers
- **CPU 耗时不达预期**（>40min）：文档明示预期；steps 降到 8–10；进度/取消保证体验
- **MuQ tokenizer 首次联网**：setup 脚本预下载到本地 HF 缓存，运行时零网络
