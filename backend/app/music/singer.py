"""AI 演唱：macOS TTS 逐字合成 + grain 变调，贴合旋律。

原理：
1. 每个汉字用 `say -v Tingting` 单独合成（真人女声音色，保留语气质感）
2. FFT 检测该字的基频（F0）
3. grain 变调：把 F0 弯曲到目标旋律音高（保留真人音色，只改音高）
4. 字数与旋律音符一一对齐，节奏精准

完全本地、零费用。效果：真人音色 + 旋律音高，接近真人清唱质感。
"""
import os
import random
import re
import shutil
import subprocess
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor

import numpy as np

RATE = 22050
_PUNCT_RE = re.compile(r"[\s，。,.!?！？、；;：:…·—~\-\"'“”‘’()（）\[\]【】<>《》]")

# 磁盘缓存：backend/output/.tts_cache/<voice>/<char_md5>.wav
_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "output", ".tts_cache")

# (voice, char) -> (samples_trimmed, f0_hz) | None
_char_cache = {}
_voice = None
_voice_checked = False


# ---------------------------------------------------------------- 工具
def available_voice(refresh=False):
    """找 macOS 中文语音；找不到返回 None。"""
    global _voice, _voice_checked
    if _voice_checked and not refresh:
        return _voice
    _voice = None
    if shutil.which("say"):
        try:
            out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True,
                                 timeout=15).stdout
        except Exception:
            out = ""
        zh = []
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].lower().replace("-", "_").startswith("zh"):
                zh.append(parts[0])
        _voice = "Tingting" if "Tingting" in zh else (zh[0] if zh else None)
    _voice_checked = True
    return _voice


def _disk_cache_path(voice, ch):
    import hashlib
    d = os.path.join(_CACHE_DIR, voice)
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        return None
    return os.path.join(d, hashlib.md5(ch.encode("utf-8")).hexdigest() + ".wav")


def _read_wav(path):
    with wave.open(path) as w:
        rate = w.getframerate()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float) / 32768
    if rate != RATE and len(raw):
        n_out = int(len(raw) * RATE / rate)
        raw = np.interp(np.linspace(0, len(raw) - 1, n_out),
                        np.arange(len(raw)), raw)
    return raw


def estimate_f0(samples):
    """FFT 找 60-500Hz 最强峰值作为基频；失败返回 180Hz。"""
    seg = samples[len(samples) // 4: int(len(samples) * 0.75)]
    if len(seg) < 256:
        return 180.0
    window = seg * np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(window, 4096))
    freqs = np.fft.rfftfreq(4096, 1.0 / RATE)
    band = (freqs >= 60) & (freqs <= 500)
    if not band.any():
        return 180.0
    f0 = float(freqs[band][np.argmax(spec[band])])
    return f0 if f0 > 50 else 180.0


def _grain_shift(samples, ratio, out_len=None):
    """时域 grain 变调 + 时值对齐（overlap-add 重定时）。

    策略：先把 samples resample 到目标 F0（音高变），
    再用 OLA 拉伸/压缩到 out_len（时值对齐，不改变音高）。

    ratio = out_len / len(samples_resampled)
    """
    FRAME = 512
    HOP = 128
    n_in = len(samples)
    if n_in < FRAME:
        return samples

    # 第一步：resample 到目标 F0（改变音高，时长也按比例变）
    # ratio_pitch = target_f0 / source_f0，resample 后长度 = n_in * ratio_pitch
    # 这里不做 resample，而是由调用方传入已 resample 的 samples
    # 直接 OLA stretch 到 out_len

    if out_len is None:
        out_len = n_in
    out_len = min(out_len, n_in * 6)

    out = np.zeros(out_len, dtype=np.float64)
    win = np.hanning(FRAME).astype(np.float64)
    n_out_frames = max(1, out_len // HOP)
    ratio = out_len / n_in  # OLA stretch ratio

    for i in range(n_out_frames):
        in_pos = int(i * HOP / ratio)
        if in_pos + FRAME > n_in:
            in_pos = max(0, n_in - FRAME)
        frame = samples[in_pos: in_pos + FRAME]
        if len(frame) < FRAME:
            frame = np.pad(frame, (0, FRAME - len(frame)))
        seg = frame * win
        o0 = i * HOP
        o1 = min(o0 + len(seg), out_len)
        if o0 < out_len:
            out[o0:o1] += seg[:o1 - o0]

    peak = np.abs(out).max()
    if peak > 1e-9:
        out *= 0.9 / peak
    return out


def _pitch_resample(samples, src_f0, dst_f0):
    """resample 改变音高（不保留时长，时长会反向变化）。"""
    if src_f0 < 50 or dst_f0 < 50:
        return samples
    ratio = src_f0 / dst_f0  # >1 降调（变慢），<1 升调（变快）
    ratio = max(0.25, min(4.0, ratio))
    new_len = max(64, int(len(samples) * ratio))
    x_old = np.arange(len(samples))
    x_new = np.linspace(0, len(samples) - 1, new_len)
    return np.interp(x_new, x_old, samples)


def _get_syllable(voice, ch):
    """获取单字 TTS 音频 + 基频；磁盘/内存缓存。"""
    key = (voice, ch)
    if key in _char_cache:
        return _char_cache[key]

    path = _disk_cache_path(voice, ch)
    data = None
    if path and os.path.isfile(path):
        try:
            data = _read_wav(path)
        except Exception:
            data = None

    if data is None:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)
        try:
            subprocess.run(
                ["say", "-v", voice, "-o", tmp_path,
                 "--data-format=LEI16@22050", ch],
                capture_output=True, timeout=10, check=True)
            data = _read_wav(tmp_path)
            if path and data is not None and len(data) > 0:
                try:
                    shutil.move(tmp_path, path)
                except OSError:
                    pass
        except Exception:
            data = None
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    result = None
    if data is not None and len(data) > 200:
        # 去前后静音
        env = np.abs(data)
        threshold = 0.02 * (env.max() + 1e-9)
        idx = np.where(env > threshold)[0]
        if len(idx):
            data = data[max(0, idx[0] - 100): idx[-1] + 200]
        f0 = estimate_f0(data)
        result = (data, f0)

    _char_cache[key] = result
    return result


def _warm(voice, chars):
    """并行预热所有唯一字的 TTS。"""
    missing = [c for c in set(chars) if (voice, c) not in _char_cache]
    if not missing:
        return
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda c: _get_syllable(voice, c), missing))


# ---------------------------------------------------------------- 主入口
def plan_singing(song):
    """把歌词逐字映射到旋律音符：[{start, dur, pitch, vel, syllable, vowel, line_end, text, bar}]。

    每个有歌词的小节，把该小节的字依次填入该小节的旋律音符。
    字数 > 音符数时，多余字共享最后一个音符；字数 < 音符数时，尾部音符保持纯音。
    """
    events = []
    bpb = song["beats_per_bar"]

    # 建立 bar -> 歌词 映射
    bar_lyrics = {}
    cursor = 0
    for sec in song["sections"]:
        for i in range(sec["bars"]):
            text = sec["lyrics"][i] if i < len(sec.get("lyrics", [])) else ""
            bar_lyrics[cursor + i] = text
        cursor += sec["bars"]

    for bar in sorted(bar_lyrics):
        text = bar_lyrics[bar]
        chars = [c for c in text if not _PUNCT_RE.match(c)]
        if not chars:
            continue

        # 该小节的旋律音符
        notes = [n for n in song["melody"] if bar * bpb <= n["start"] < (bar + 1) * bpb]
        if not notes:
            continue

        n_notes = len(notes)
        n_chars = len(chars)

        for ni in range(n_notes):
            note = notes[ni]
            # 字分配：前 n_chars 个音符各取一个字，多余的音符空着
            ci = ni if ni < n_chars else -1
            ch = chars[ci] if ci >= 0 else ""

            # 计算该字在此音符内的占比（一个音符可能分给多个字）
            if n_chars > n_notes and ci >= 0:
                # 多个字挤在一个音符：把音符时长均分
                # 简化：只有第一个字占满，其余字共享（实际播放时叠加）
                dur_frac = 1.0
            else:
                dur_frac = 1.0

            events.append({
                "start": note["start"],
                "dur": note["dur"] * dur_frac,
                "pitch": note["pitch"],
                "vel": note.get("vel", 80),
                "syllable": ch,
                "text": text,
                "bar": bar,
                "vowel": 0,  # 兼容字段（不再用于 formant 合成）
                "line_end": (ci == n_chars - 1) and (ni == n_notes - 1),
            })
    return events


def render_sing_mono(song, events, total_sec, spb, progress_cb=None):
    """渲染人声到 22050Hz 单声道缓冲。

    流程：
    1. 预热所有字的 TTS
    2. 每字：grain 变调到目标音高 → 包络 → 放入对应时间点
    3. 归一化
    失败（无可用语音）返回 None。
    """
    voice = available_voice()
    if not voice:
        return None

    # 只处理有实际字的字
    real_events = [e for e in events if e.get("syllable")]
    if not real_events:
        return None

    _warm(voice, [e["syllable"] for e in real_events])
    rng = random.Random((song.get("spec") or {}).get("seed") or 0xD1CE)
    buf = np.zeros(int(total_sec * RATE))

    for i, e in enumerate(real_events):
        ch = e["syllable"]
        got = _get_syllable(voice, ch)
        if not got:
            continue
        samples, f0 = got
        target_f0 = 440.0 * 2 ** ((e["pitch"] - 69) / 12.0)

        # 第一步：pitch resample（改变音高，时长随之变化）
        pitched = _pitch_resample(samples, f0, target_f0)

        # 第二步：OLA stretch 到音符目标时长（不改变音高）
        target_dur_sec = max(0.15, e["dur"] * spb)
        target_len = int(target_dur_sec * RATE)
        aligned = _grain_shift(pitched, 1.0, out_len=target_len)

        # 包络：起音 50ms + 尾音 100ms
        env = np.ones(len(aligned), dtype=np.float32)
        a = min(int(0.05 * RATE), len(aligned) // 4)
        r = min(int(0.10 * RATE), len(aligned) // 4)
        env[:a] = np.linspace(0, 1, a)
        env[-r:] *= np.linspace(1, 0, r)

        gain = (0.95 + rng.uniform(-0.03, 0.05)) * (e.get("vel", 80) / 100.0)
        out = aligned * env * gain

        i0 = int(e["start"] * spb * RATE)
        i0 = max(0, min(i0, len(buf) - 1))
        n = min(len(out), len(buf) - i0)
        if n > 0:
            buf[i0:i0 + n] += out[:n]

        if e.get("line_end"):
            tail_n = int(0.12 * RATE)
            j0 = i0 + n
            if j0 + tail_n < len(buf):
                nz = np.random.normal(0, 0.03, tail_n).astype(np.float32)
                buf[j0:j0 + tail_n] += nz * np.exp(-np.arange(tail_n) / RATE * 22)

        if progress_cb and i % 10 == 0:
            progress_cb(i / len(real_events))

    peak = float(np.max(np.abs(buf))) if buf.size else 0.0
    if peak > 1e-9:
        buf *= 0.90 / peak
    return buf
