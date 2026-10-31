"""DiffRhythm 推理子进程（跑在 DiffRhythm/.venv-neural 里，cwd 必须是 DiffRhythm 根目录）。

用法： python diffrhythm_runner.py <job.json>
job.json 字段：
    lyric_lines : [[秒, 文本], ...]      # 由后端按小节时间算好（纯文本会被模型静默丢弃，必须带时间戳）
    style_prompt: str | null             # 文本风格 prompt（与 ref_audio 二选一）
    ref_audio   : str | null             # 参考音频绝对路径（>=10s，取中段 10s 做风格）
    seed        : int | null
    steps       : int                    # CFM 采样步数（CPU 建议 8-16，越大越慢越细）
    out_wav     : str                    # 输出 WAV 绝对路径（44.1kHz 立体声 int16）

stdout 只输出 JSON 行（父进程逐行解析）：
    {"type":"progress","frac":0.37,"note":"..."}
    {"type":"done","wav":"...","duration_sec":95.0}
    {"type":"error","error":"..."}
其余诊断信息全部走 stderr。

设计说明：
- 本 runner 是 CPU 路径：模型强制 fp32（上游 load_checkpoint 会 .half()，Intel CPU
  上 fp16 慢且有算子风险）；GPU 用户请直接用 DiffRhythm/infer/infer.py。
- 采样采用与 CFM.sample() 完全一致的 Euler 展开步进（odeint euler 等价），
  以获得真实的逐步进度，且只保留当前步张量（比上游 32 步轨迹省 ~100MB 内存）。
  传 --legacy 可退回上游 cfm.sample() 路径（无逐步进度）。
"""
import json
import math
import os
import random
import sys
import time
import traceback
import wave

# ---- 在 import torch 前定好环境 ----
os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), "pretrained", "hf"))
sys.path.insert(0, os.getcwd())          # model/ g2p/ thirdparty/
sys.path.insert(0, os.path.join(os.getcwd(), "infer"))  # infer_utils

import torch  # noqa: E402


def emit(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def log(msg):
    sys.stderr.write("[runner] %s\n" % msg)
    sys.stderr.flush()


def wav_duration(path):
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate() or 1)
    except Exception:
        return 0.0


def rss_mb():
    """进程峰值内存（macOS ru_maxrss 单位是字节，Linux 是 KB）。"""
    import resource
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / 1048576.0 if sys.platform == "darwin" else v / 1024.0


def main():
    job = json.load(open(sys.argv[1], "r", encoding="utf-8"))
    legacy = "--legacy" in sys.argv
    steps = int(job.get("steps") or 16)
    steps = max(2, min(64, steps))
    cfg_strength = 4.0
    seed = job.get("seed")
    out_wav = job["out_wav"]
    lyric_lines = job.get("lyric_lines") or []
    audio_length = int(job.get("audio_length") or 95)
    max_frames = 2048 if audio_length == 95 else 6144  # 1_2 标准版 / 1_2-full

    t0 = time.time()

    def prog(frac, note):
        emit({"type": "progress", "frac": round(max(0.0, min(1.0, frac)), 4),
              "note": note})

    # ---------------- 设备 ----------------
    # 默认 CPU（fp32，本 runner 的基准路径）。CUDA 机器自动用 GPU；
    # MPS 不参与自动选择：Intel Mac 的 Metal 内存池只有 ~6.7GB 会 OOM，
    # Apple Silicon 用户可显式 DIFFRHYTHM_DEVICE=mps 提速。
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.environ.get("DIFFRHYTHM_DEVICE"):
        device = os.environ["DIFFRHYTHM_DEVICE"]
    log("device=%s steps=%d seed=%s" % (device, steps, seed))

    # ---------------- 加载模型 ----------------
    from infer_utils import (decode_audio, get_lrc_token, get_negative_style_prompt,
                             get_style_prompt, prepare_model)
    prog(0.02, "加载 DiffRhythm 模型（首次 1-2 分钟）…")
    cfm, tokenizer, muq, vae = prepare_model(max_frames, device)
    if device == "cpu":
        cfm = cfm.float()  # 上游 load_checkpoint 强制 fp16；CPU 转 fp32
    dt = next(cfm.parameters()).dtype
    log("model loaded in %.1fs dtype=%s" % (time.time() - t0, dt))
    prog(0.12, "模型就绪")

    # ---------------- 歌词 -> LRC token ----------------
    # 格式 [mm:ss.xx]文本（parse_lyrics 读 line[1:9] 为时间戳）
    def fmt_ts(sec):
        sec = max(0.0, min(sec, audio_length - 0.1))
        m = int(sec // 60)
        s = sec - m * 60
        return "[%02d:%05.2f]" % (m, s)

    def sanitize(text):
        """剔除会让 g2p/LangSegment 崩溃的符号（emoji、音符等→'other'语言），
        保留中英文、数字与常用标点。"""
        import re
        return re.sub(r"[^\u4e00-\u9fff\u3400-\u4dbfA-Za-z0-9 ,.，。！？：；、'\"-]+",
                      " ", str(text)).strip() or "啊"

    lrc_text = "\n".join(
        fmt_ts(t) + sanitize(txt) for t, txt in lyric_lines if sanitize(txt))
    # 哨兵行：get_lrc_token 在 2048 帧时会丢弃最后一行，丢这句而不是真歌词
    # （用 g2p 安全字符，不能用 ♪ 等符号——LangSegment 会判成 other 语言报错）
    lrc_text += "\n%s啊" % fmt_ts(audio_length - 0.1)

    lrc_prompt, start_time, end_frame, song_duration = get_lrc_token(
        max_frames, lrc_text, tokenizer, audio_length, device)
    start_time = start_time.to(dt)
    song_duration = song_duration.to(dt)
    log("lyrics lines=%d end_frame=%d" % (len(lyric_lines), end_frame))
    prog(0.14, "歌词编码完成")

    # ---------------- 风格 prompt ----------------
    if job.get("ref_audio"):
        prog(0.15, "分析参考音频风格…")
        style_prompt = get_style_prompt(muq, wav_path=job["ref_audio"])
    else:
        prog(0.15, "编码风格描述…")
        style_prompt = get_style_prompt(muq, prompt=job["style_prompt"])
    negative_style_prompt = get_negative_style_prompt(device)
    style_prompt = style_prompt.to(dt)
    negative_style_prompt = negative_style_prompt.to(dt)
    prog(0.16, "风格就绪")

    # ---------------- 条件（非 edit：全零 latent + 全段可预测） ----------------
    # 尺寸取 end_frame（sample() 内部会把 cond 截到 duration，等价）
    cond = torch.zeros(1, end_frame, 64, device=device, dtype=dt)
    step_cond = torch.zeros_like(cond)
    fixed_span_mask = torch.ones(1, end_frame, 1, dtype=torch.bool, device=device)
    duration = torch.full((1,), end_frame, device=device, dtype=torch.long)

    if seed is not None:
        torch.manual_seed(int(seed))
        random.seed(int(seed))

    def transformer_call(x, t, style, drop_audio, drop_text):
        return cfm.transformer(
            x=x, cond=step_cond, text=lrc_prompt, time=t,
            drop_audio_cond=drop_audio, drop_text=drop_text, drop_prompt=False,
            style_prompt=style, start_time=start_time, duration=song_duration)

    def fn(t, x):
        pred = transformer_call(x, t, style_prompt, False, False)
        if cfg_strength < 1e-5:
            return pred
        null_pred = transformer_call(x, t, negative_style_prompt, True, True)
        return pred + (pred - null_pred) * cfg_strength

    s_t = time.time()
    if legacy:
        # 上游路径（无逐步进度）
        latents, _ = cfm.sample(
            cond=cond, text=lrc_prompt, duration=end_frame,
            style_prompt=style_prompt, max_duration=end_frame,
            song_duration=song_duration, negative_style_prompt=negative_style_prompt,
            steps=steps, cfg_strength=cfg_strength, start_time=start_time,
            latent_pred_segments=[(0, end_frame)], batch_infer_num=1)
        sampled = latents[0]
        prog(0.90, "采样完成，正在解码音频…")
    else:
        # Euler 逐步（与 odeint(method=euler) 数学等价），真实进度 + 省内存
        with torch.inference_mode():
            y0 = torch.randn(end_frame, cfm.num_channels, device=device,
                             dtype=step_cond.dtype).unsqueeze(0)
            t_grid = torch.linspace(0, 1, steps, device=device, dtype=step_cond.dtype)
            x = y0
            span = 0.90 - 0.16
            for i in range(steps - 1):
                x = x + (t_grid[i + 1] - t_grid[i]) * fn(t_grid[i], x)
                done = (i + 1) / float(steps - 1)
                remain = (time.time() - s_t) / done * (1 - done) if done > 0 else 0
                prog(0.16 + span * done,
                     "神经网络采样 %d/%d 步（预计还需 %d 分钟）"
                     % (i + 1, steps - 1, int(math.ceil(remain / 60))))
            sampled = x
        prog(0.90, "采样完成，正在解码音频…")
    log("sampling cost %.1fs" % (time.time() - s_t))

    # ---------------- VAE 解码 -> WAV ----------------
    v_t = time.time()
    with torch.inference_mode():
        latent = sampled.to(torch.float32).transpose(1, 2)  # [b d t]
        output = decode_audio(latent, vae, chunked=True)
        output = output.reshape(output.shape[1], -1)  # [d n]（batch=1）
        output = (output / torch.max(torch.abs(output))
                  .clamp(min=1e-8)).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    arr = output.numpy().T.astype("<i2")  # [n, 2]

    os.makedirs(os.path.dirname(out_wav) or ".", exist_ok=True)
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(arr.shape[1])
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(arr.tobytes())
    log("vae decode + save cost %.1fs" % (time.time() - v_t))

    prog(1.0, "完成")
    emit({"type": "done", "wav": out_wav,
          "duration_sec": round(wav_duration(out_wav), 2)})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        try:
            emit({"type": "error", "error": "%s: %s" % (type(e).__name__, e)})
        except Exception:
            pass
        traceback.print_exc()
        sys.exit(1)
