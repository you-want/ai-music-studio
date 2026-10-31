#!/usr/bin/env python
"""DiffRhythm 歌词→完整歌曲 生成测试。

用法：
    # 在 DiffRhythm 根目录下运行：
    /Users/rain9/.local/share/mamba/envs/drr/bin/python test_diffrhythm.py

生成 95 秒中文歌曲（folk 风格，成都歌词示例），验证：
  1. 模型下载（cfm_model.pt + vae_model.pt + muq）
  2. MPS 设备推理
  3. 输出 44.1kHz WAV
"""
import os
import sys
import time

# 必须从 DiffRhythm 仓库根目录运行
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "infer"))

import torch
import torchaudio

print(f"cwd: {os.getcwd()}")
print(f"torch: {torch.__version__}, mps: {torch.backends.mps.is_available()}")

# 设置 HuggingFace cache（避免下到奇怪位置）
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "pretrained", "hf"))

from infer_utils import (
    prepare_model,
    get_lrc_token,
    get_style_prompt,
    get_negative_style_prompt,
    get_reference_latent,
    decode_audio,
)
from infer import inference
import random

# ---------- 配置 ----------
LRC_PATH = os.path.join(ROOT, "infer", "example", "eg_cn.lrc")
REF_PROMPT = "folk, acoustic guitar, emotional, Chinese, female vocal"
AUDIO_LENGTH = 95          # 秒（95 → max_frames=2048，用 DiffRhythm-1_2）
CHUNKED = True             # 分块解码省内存
OUTPUT_DIR = os.path.join(ROOT, "test_output")
OUTPUT_WAV = os.path.join(OUTPUT_DIR, "output.wav")

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

print(f"\n=== 加载模型 (device={device}) ===")
s_t = time.time()
cfm, tokenizer, muq, vae = prepare_model(2048, device)
print(f"模型加载完成，耗时 {time.time()-s_t:.1f}s")

# 准备歌词
print("\n=== 解析歌词 ===")
with open(LRC_PATH, "r", encoding="utf-8") as f:
    lrc = f.read()
lrc_prompt, start_time, end_frame, song_duration = get_lrc_token(
    2048, lrc, tokenizer, AUDIO_LENGTH, device
)
print(f"end_frame={end_frame}, song_duration={song_duration}")

# 风格 prompt
print("\n=== 生成风格 prompt ===")
style_prompt = get_style_prompt(muq, prompt=REF_PROMPT)
negative_style_prompt = get_negative_style_prompt(device)

# latent prompt（非编辑模式，全零）
latent_prompt, pred_frames = get_reference_latent(
    device, 2048, False, None, None, vae
)

# 推理
print(f"\n=== 开始推理 (steps=32, cfg=4.0, chunked={CHUNKED}) ===")
s_t = time.time()
generated_songs = inference(
    cfm_model=cfm,
    vae_model=vae,
    cond=latent_prompt,
    text=lrc_prompt,
    duration=end_frame,
    style_prompt=style_prompt,
    negative_style_prompt=negative_style_prompt,
    start_time=start_time,
    pred_frames=pred_frames,
    chunked=CHUNKED,
    batch_infer_num=1,
    song_duration=song_duration,
)
e_t = time.time() - s_t
print(f"推理完成，耗时 {e_t:.1f}s")

# 保存
os.makedirs(OUTPUT_DIR, exist_ok=True)
generated_song = random.sample(generated_songs, 1)[0]
torchaudio.save(OUTPUT_WAV, generated_song.cpu(), sample_rate=44100)
print(f"\n✅ 输出: {OUTPUT_WAV}")
print(f"   文件: {os.path.getsize(OUTPUT_WAV) / 1024 / 1024:.1f} MB")
