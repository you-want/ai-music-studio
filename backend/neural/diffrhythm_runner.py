"""DiffRhythm 推理子进程（跑在 DiffRhythm/.venv-neural 里，cwd 必须是 DiffRhythm 根目录）。

两种运行方式：
    python diffrhythm_runner.py <job.json>            # 一次性（调试/兼容）
    python diffrhythm_runner.py --serve               # 常驻模式（生产默认）
常驻模式：加载模型后吐 {"type":"ready"}，循环从 stdin 读 job.json 路径逐个执行，
    每个 job 的 done/error 带 "id" 回显；空闲 30 分钟或 50 个任务后自动退出
    （父进程下次调用会重新拉起并重载模型）。

job.json 字段：
    id          : str                          # 任务 id（回显用）
    lyric_lines : [[秒, 文本], ...]            # 整首歌词（纯文本会被模型静默丢弃，必须带时间戳）
    style_prompt: str | null                   # 文本风格 prompt
    ref_audio   : str | null                   # 参考音频绝对路径（>=10s，取中段 10s 做风格）
    style_prompt_npy : str | null              # persona 嵌入 .npy（[1,512]，优先级最高）
    edit        : {"ref_song": wav路径, "segments": [[起秒, 止秒], ...]} | null
                                                 # 分段重唱：区间外写回原曲 latent，区间内按新歌词重采样
    seed        : int | null
    steps       : int                          # CFM 采样步数（CPU 建议 8-16）
    audio_length: int                          # 95 或 285
    out_wav     : str                          # 输出 WAV 绝对路径

stdout 只输出 JSON 行（父进程逐行解析）：
    {"type":"ready"}                                  # 仅 serve 模式
    {"type":"progress","id":"..","frac":0.37,"note":"..."}
    {"type":"done","id":"..","wav":"..","duration_sec":95.0}
    {"type":"error","id":"..","error":"..."}
其余诊断信息全部走 stderr。

设计说明：
- CPU 路径强制 fp32（上游 load_checkpoint 会 .half()，Intel CPU 上 fp16 慢且有算子风险）。
- 采样为与 CFM.sample() 等价的 Euler 展开步进，真实逐步进度 + 省内存。
- edit 任务的两条生命线（缺一即错）：step_cond = where(mask, 0, cond)（区间外用原曲
  latent 作条件）与采样后 sampled = where(mask, sampled, cond)（区间外写回原曲）。
- edit 的 vae_sample 在编码参考 latent 时消耗 RNG —— seed 必须在每个任务最开头设置。
"""
import json
import math
import os
import numpy as np
import platform
import random
import sys
import time
import traceback
import wave

# ---- 在 import torch 前定好环境 ----
os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), "pretrained", "hf"))
sys.path.insert(0, os.getcwd())          # model/ g2p/ thirdparty/
sys.path.insert(0, os.path.join(os.getcwd(), "infer"))  # infer_utils


try:
    # espeak-ng 动态库/数据路径（espeakng-loader 自带多平台二进制）
    import espeakng_loader
    os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", espeakng_loader.get_library_path())
    os.environ.setdefault("ESPEAK_DATA_PATH",
                            os.path.dirname(espeakng_loader.get_data_path()))
    os.environ.setdefault("PHONEMIZER_ESPEAK_PATH", os.path.join(
        os.path.dirname(espeakng_loader.get_library_path()), "espeak-ng"))
except ImportError:
    pass

import torch  # noqa: E402

SERVE_IDLE_SEC = 120 * 60
SERVE_MAX_JOBS = 50
CONSECUTIVE_ERROR_EXIT = 3


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
    import resource
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / 1048576.0 if sys.platform == "darwin" else v / 1024.0


def pick_device():
    # 优先级：CUDA（NVIDIA）> Apple Silicon 的 MPS（GPU 加速）> CPU。
    # Intel Mac 的 Metal 内存池过小会 OOM，故仅 arm64 自动启用 MPS；
    # DIFFRHYTHM_DEVICE 环境变量可显式覆盖一切。
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and platform.machine() == "arm64" and torch.backends.mps.is_available():
        device = "mps"
    if os.environ.get("DIFFRHYTHM_DEVICE"):
        device = os.environ["DIFFRHYTHM_DEVICE"]
    return device


def load_models(device, max_frames):
    from infer_utils import prepare_model
    cfm, tokenizer, muq, vae = prepare_model(max_frames, device)
    if device in ("cpu", "mps"):
        cfm = cfm.float()  # 上游强制 fp16；CPU 转 fp32
    dt = next(cfm.parameters()).dtype
    return cfm, tokenizer, muq, vae, dt


def sanitize(text):
    """剔除会让 g2p/LangSegment 崩溃的符号（emoji、音符等→'other'语言）。"""
    import re
    return re.sub(r"[^\u4e00-\u9fff\u3400-\u4dbfA-Za-z0-9 ,.，。！？：；、'\"-]+",
                  " ", str(text)).strip() or "啊"


def run_job(job, ctx, prog):
    """执行单个生成任务。ctx: {cfm, tokenizer, muq, vae, device, dt}"""
    jid = str(job.get("id") or "")
    legacy = bool(job.get("legacy"))
    steps = max(2, min(64, int(job.get("steps") or 16)))
    cfg_strength = 4.0
    seed = job.get("seed")
    out_wav = job["out_wav"]
    lyric_lines = job.get("lyric_lines") or []
    audio_length = int(job.get("audio_length") or 95)
    max_frames = 2048 if audio_length == 95 else 6144
    edit = job.get("edit") or None

    # ---- 任务级 seed：必须最先设置（edit 的 vae_sample 在编码时就消耗 RNG）----
    if seed is not None:
        torch.manual_seed(int(seed))
        random.seed(int(seed))

    cfm, tokenizer, muq, vae, dt = ctx["cfm"], ctx["tokenizer"], ctx["muq"], ctx["vae"], ctx["dt"]
    device = ctx["device"]
    s_t = time.time()

    # ---- 歌词 -> LRC token ----
    from infer_utils import (decode_audio, get_lrc_token, get_negative_style_prompt,
                             get_reference_latent, get_style_prompt)

    def fmt_ts(sec):
        sec = max(0.0, min(sec, audio_length - 0.1))
        return "[%02d:%05.2f]" % (int(sec // 60), sec - int(sec // 60) * 60)

    lrc_text = "\n".join(
        fmt_ts(t) + sanitize(txt) for t, txt in lyric_lines if sanitize(txt))
    # 哨兵行：get_lrc_token 在 2048 帧时丢弃最后一行，丢这句而不是真歌词
    lrc_text += "\n%s啊" % fmt_ts(audio_length - 0.1)

    lrc_prompt, start_time, end_frame, song_duration = get_lrc_token(
        max_frames, lrc_text, tokenizer, audio_length, device)
    start_time = start_time.to(dt)
    song_duration = song_duration.to(dt)
    log("id=%s lyrics=%d end_frame=%d" % (jid, len(lyric_lines), end_frame))
    prog(0.14, "歌词编码完成")

    # ---- 风格 prompt（persona 嵌入 + 文本风格融合 > persona 单独 > 参考音频 > 文本） ----
    if job.get("style_prompt_npy"):
        prog(0.15, "加载音色 Persona…")
        import torch.nn.functional as F
        persona = torch.from_numpy(
            np.load(job["style_prompt_npy"]).astype("float32")).to(device)
        if job.get("style_prompt"):
            # 融合：人声身份（persona）+ 曲风语境（文本）各半，再归一化
            # —— 只用人声嵌入会丢失编曲/器乐风格，生成质量显著下降
            text_emb = get_style_prompt(muq, prompt=job["style_prompt"]).float()
            blended = F.normalize(persona + text_emb.to(device), dim=-1)
            style_prompt = blended
            log("style = persona + text blended")
        else:
            style_prompt = persona
            log("style = persona only")
    elif job.get("ref_audio"):
        prog(0.15, "分析参考音频风格…")
        style_prompt = get_style_prompt(muq, wav_path=job["ref_audio"])
    else:
        prog(0.15, "编码风格描述…")
        style_prompt = get_style_prompt(muq, prompt=job.get("style_prompt"))
    negative_style_prompt = get_negative_style_prompt(device)
    style_prompt = style_prompt.to(dt)
    negative_style_prompt = negative_style_prompt.to(dt)
    prog(0.16, "风格就绪")

    # ---- 条件构造 ----
    if edit:
        # 分段重唱：参考曲编码为 latent（内部 vae_sample 消耗 RNG，seed 已在任务开头设置）
        prog(0.16, "编码原曲 latent…")
        seg_json = json.dumps(edit.get("segments") or [])
        cond, pred_frames = get_reference_latent(
            device, max_frames, True, seg_json, edit.get("ref_song"), vae)
        cond = cond[:, :end_frame, :].to(dt)  # 与 sample() 的截断语义一致
        mask_1d = torch.zeros(end_frame, dtype=torch.bool, device=device)
        for sf, ef in pred_frames:
            mask_1d[sf:min(ef, end_frame)] = True
        fixed_span_mask = mask_1d.view(1, end_frame, 1)
        step_cond = torch.where(fixed_span_mask, torch.zeros_like(cond), cond)
        log("edit segments=%s pred_frames=%s" % (edit.get("segments"), pred_frames))
    else:
        cond = torch.zeros(1, end_frame, 64, device=device, dtype=dt)
        fixed_span_mask = torch.ones(1, end_frame, 1, dtype=torch.bool, device=device)
        step_cond = torch.zeros_like(cond)

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

    # ---- Euler 逐步采样 ----
    if legacy:
        latents, _ = cfm.sample(
            cond=cond, text=lrc_prompt, duration=end_frame,
            style_prompt=style_prompt, max_duration=end_frame,
            song_duration=song_duration, negative_style_prompt=negative_style_prompt,
            steps=steps, cfg_strength=cfg_strength, start_time=start_time,
            latent_pred_segments=pred_frames if edit else [(0, end_frame)],
            batch_infer_num=1)
        sampled = latents[0]
        prog(0.90, "采样完成，正在解码音频…")
    else:
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
    log("id=%s sampling cost %.1fs" % (jid, time.time() - s_t))
    log("id=%s sampled stats: absmax=%.4f nan=%s" % (
        jid, sampled.abs().max().item(), torch.isnan(sampled).any().item()))

    # edit：区间外写回原曲 latent（缺这步会丢失未编辑段！）
    if edit:
        sampled = torch.where(fixed_span_mask, sampled, cond)

    # ---- VAE 解码 -> WAV ----
    v_t = time.time()
    # VAE 解码固定走 CPU：常驻 worker 里 MPS 解码实测会产生全零输出（状态污染），
    # CPU 解码 95s 曲约 20-40s，可接受；采样（大头算力）仍在 GPU。
    with torch.inference_mode():
        latent = sampled.to(torch.float32).cpu().transpose(1, 2)  # [b d t]
        vae_cpu = vae.to("cpu")
        output = decode_audio(latent, vae_cpu, chunked=True)
        output = output.reshape(output.shape[1], -1)  # [d n]（batch=1）
        output = (output / torch.max(torch.abs(output))
                  .clamp(min=1e-8)).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    arr = output.cpu().numpy().T.astype("<i2")  # [n, 2]（MPS 上必须先拷回主机内存）

    os.makedirs(os.path.dirname(out_wav) or ".", exist_ok=True)
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(arr.shape[1])
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(arr.tobytes())
    log("id=%s int16 stats: absmax=%d rms=%.1f" % (
        jid, int(np.abs(arr).max()), float(np.sqrt((arr.astype(np.float32) ** 2).mean()))))
    log("id=%s vae decode + save cost %.1fs" % (jid, time.time() - v_t))

    prog(1.0, "完成")
    emit({"type": "done", "id": jid, "wav": out_wav,
          "duration_sec": round(wav_duration(out_wav), 2)})


def _max_frames_of(audio_length):
    return 2048 if int(audio_length or 95) == 95 else 6144


def main():
    argv = sys.argv[1:]
    serve = "--serve" in argv
    paths = [a for a in argv if not a.startswith("--")]

    device = pick_device()
    ctx = None

    def prog(frac, note):
        emit({"type": "progress", "frac": round(max(0.0, min(1.0, frac)), 4), "note": note})

    if serve:
        # ---------------- 常驻模式 ----------------
        max_frames = 2048  # 首任务前无法知道；2048 权重默认。全曲任务出现时按需重载
        t0 = time.time()
        prog(0.02, "加载 DiffRhythm 模型（首次 1-2 分钟）…")
        cfm, tokenizer, muq, vae, dt = load_models(device, max_frames)
        ctx = {"cfm": cfm, "tokenizer": tokenizer, "muq": muq, "vae": vae,
               "device": device, "dt": dt, "max_frames": max_frames}
        log("worker ready in %.1fs dtype=%s rss=%.0fMB" % (time.time() - t0, dt, rss_mb()))
        emit({"type": "ready"})
        jobs_done, err_streak, last_active = 0, 0, time.time()
        for line in sys.stdin:  # EOF（父进程关闭管道）-> 退出
            line = line.strip()
            if not line:
                continue
            if time.time() - last_active > SERVE_IDLE_SEC:
                log("idle timeout, exiting")
                break
            path = line
            try:
                job = json.load(open(path, "r", encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                emit({"type": "error", "id": path, "error": "job 文件不可读: %s" % e})
                continue
            # 全曲任务（6144 帧）与默认 2048 帧模型不同 -> 透明重载
            need_frames = _max_frames_of(job.get("audio_length"))
            if need_frames != ctx["max_frames"]:
                log("reloading models for max_frames=%d" % need_frames)
                cfm, tokenizer, muq, vae, dt = load_models(device, need_frames)
                ctx = {"cfm": cfm, "tokenizer": tokenizer, "muq": muq, "vae": vae,
                       "device": device, "dt": dt, "max_frames": need_frames}
            jid = str(job.get("id") or path)
            last_active = time.time()

            def prog2(frac, note, _jid=jid):
                emit({"type": "progress", "id": _jid,
                      "frac": round(max(0.0, min(1.0, frac)), 4), "note": note})
            try:
                run_job(job, ctx, prog2)
                jobs_done += 1
                err_streak = 0
            except Exception as e:  # noqa: BLE001
                err_streak += 1
                emit({"type": "error", "id": jid, "error": "%s: %s" % (type(e).__name__, e)})
                traceback.print_exc()
                if err_streak >= CONSECUTIVE_ERROR_EXIT:
                    log("too many consecutive errors, exiting for respawn")
                    break
            if jobs_done >= SERVE_MAX_JOBS:
                log("job cap reached, exiting for recycle")
                break
        log("worker exit (jobs=%d)" % jobs_done)
        return

    # ---------------- 一次性模式（调试/兼容） ----------------
    job = json.load(open(paths[0], "r", encoding="utf-8")) if paths else {}
    prog(0.02, "加载 DiffRhythm 模型…")
    cfm, tokenizer, muq, vae, dt = load_models(device, _max_frames_of(job.get("audio_length")))
    ctx = {"cfm": cfm, "tokenizer": tokenizer, "muq": muq, "vae": vae,
           "device": device, "dt": dt}
    run_job(job, ctx, prog)


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
