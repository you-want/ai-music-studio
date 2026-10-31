"""本地合成器：音符事件 -> 44.1kHz 立体声 WAV。

numpy 加法合成：每个音符 = 谐波叠置(每谐波独立衰减) × 包络 × 颤音(延迟启动)
× 声像；鼓组参数化合成。总线：软削波 + 三抽头简易混响 + 峰值归一。
时间单位换算：事件里的 start/dur 均为"拍"，乘 60/bpm 变秒。
"""
import math
import wave

import numpy as np

RATE = 44100

# 谐波表: (倍频, 振幅, 衰减率)；decay>0 时该谐波按 exp(-decay*t) 衰减
WAVES = {
    "synth_lead": {"harm": [(1, 1, 0), (2, .45, 0), (3, .3, 0), (4, .18, 0), (5, .12, 0), (6, .08, 0)],
                   "attack": .012, "release": .12, "sustain": .85, "decay": 0,
                   "vib": (5.2, .012, .25), "detune": 0},
    "square_lead": {"harm": [(1, 1, 0), (3, .33, 0), (5, .2, 0), (7, .14, 0), (9, .1, 0)],
                    "attack": .008, "release": .1, "sustain": .8, "decay": 0,
                    "vib": (5.5, .01, .3), "detune": 0},
    "piano": {"harm": [(1, 1, 0), (2, .5, .7), (3, .28, 1.2), (4, .14, 2.0), (5, .08, 3.0)],
              "attack": .004, "release": .18, "sustain": .3, "decay": 1.6,
              "vib": None, "detune": 0},
    "keys": {"harm": [(1, 1, 0), (2, .35, 1.0), (3, .12, 2.0), (4, .06, 3.0)],
             "attack": .005, "release": .12, "sustain": .35, "decay": 1.2,
             "vib": None, "detune": 0},
    "pluck": {"harm": [(1, 1, 0), (2, .55, 1.8), (3, .35, 2.6), (4, .2, 3.6), (5, .12, 5), (6, .08, 6.5)],
              "attack": .003, "release": .06, "sustain": .0, "decay": 3.2,
              "vib": None, "detune": 0},
    "warm_pad": {"harm": [(1, 1, 0), (2, .4, 0), (3, .18, 0), (4, .08, 0), (5, .04, 0)],
                 "attack": .55, "release": .5, "sustain": .8, "decay": 0,
                 "vib": None, "detune": 4.0},
    "strings": {"harm": [(1, 1, 0), (2, .3, 0), (3, .12, 0)],
                "attack": .35, "release": .4, "sustain": .85, "decay": 0,
                "vib": (5.0, .008, .4), "detune": 0},
    "sub_bass": {"harm": [(1, 1, 0), (2, .18, 0)],
                 "attack": .008, "release": .08, "sustain": .9, "decay": 0,
                 "vib": None, "detune": 0},
    "synth_bass": {"harm": [(1, 1, 0), (2, .25, 0), (3, .12, 0)],
                   "attack": .006, "release": .07, "sustain": .85, "decay": 0,
                   "vib": None, "detune": 0},
    # 人声哼唱：共振峰(formant)合成，aah/ooh 交替，模仿合唱团"啦/呜"哼唱
    "vocal_hum": {
        "formants": [
            [(780, 80, 1.0), (1150, 90, 0.50), (2800, 180, 0.22)],  # aah
            [(330, 70, 1.0), (760, 90, 0.42), (2400, 200, 0.10)],   # ooh
        ],
        "attack": .045, "release": .18, "sustain": .9, "decay": .06,
        "vib": (5.4, .022, .15), "detune": 0, "breath": .05, "n_harm": 30,
    },
    # 人声演唱：5 元音共振峰(a/e/i/o/u)按汉字尾音轮转，模拟真人唱歌元音变化
    # F1 F2 F3 (Hz, 带宽, 幅度)  — 参考经典 formant 表
    "vocal_sing": {
        "formants": [
            # a (啊) — 开口大
            [(800,  90, 1.00), (1200, 110, 0.50), (2800, 180, 0.25)],
            # e (诶)
            [(450,  80, 1.00), (2000, 120, 0.40), (2900, 180, 0.15)],
            # i (衣) — 窄口
            [(300,  70, 1.00), (2300, 150, 0.55), (3100, 200, 0.30)],
            # o (噢)
            [(450,  80, 1.00), (800,  100, 0.55), (2400, 200, 0.10)],
            # u (乌)
            [(330,  70, 1.00), (760,  90, 0.45), (2400, 200, 0.10)],
        ],
        "attack": .055, "release": .22, "sustain": .92, "decay": .04,
        "vib": (5.5, .018, .20), "detune": 1.2, "breath": .04, "n_harm": 40,
        # 每个汉字映射到 5 种元音之一（按拼音韵母近似）
        "vowel_map": "aeiou",
    },
}

# genre -> 每个声部用哪个音色
VOICE_FOR = {
    "pop": {"melody": "synth_lead", "pads": "warm_pad", "arps": "pluck", "stabs": "keys", "bass": "sub_bass"},
    "rock": {"melody": "square_lead", "pads": "strings", "arps": "pluck", "stabs": "keys", "bass": "synth_bass"},
    "electronic": {"melody": "square_lead", "pads": "warm_pad", "arps": "pluck", "stabs": "keys", "bass": "synth_bass"},
    "ballad": {"melody": "piano", "pads": "strings", "arps": "piano", "stabs": "keys", "bass": "sub_bass"},
    "folk": {"melody": "pluck", "pads": "warm_pad", "arps": "pluck", "stabs": "keys", "bass": "sub_bass"},
    "guofeng": {"melody": "pluck", "pads": "strings", "arps": "pluck", "stabs": "keys", "bass": "sub_bass"},
    "lofi": {"melody": "keys", "pads": "warm_pad", "arps": "keys", "stabs": "keys", "bass": "sub_bass"},
    "jazz": {"melody": "keys", "pads": "warm_pad", "arps": "keys", "stabs": "keys", "bass": "sub_bass"},
    "rap": {"melody": "pluck", "pads": "warm_pad", "arps": "pluck", "stabs": "keys", "bass": "sub_bass"},
}

ROLE_GAIN = {"pads": .20, "arps": .13, "stabs": .22, "bass": .32, "melody": .30}
ROLE_PAN = {"pads": 0.0, "arps": -0.22, "stabs": 0.1, "bass": 0.0, "melody": 0.15}
DRUM_GAIN = {"kick": .95, "snare": .55, "hat": .22, "ohat": .26, "clap": .5, "rim": .4}
# 分轨分组：主唱(主旋律) / 和声(垫+琶音+键琴) / 贝斯 / 鼓
GROUP_ROLES = [("vocal", ["melody"]), ("harmony", ["pads", "arps", "stabs"]),
               ("bass", ["bass"]), ("drums", ["drums"])]

# 人声演唱时主唱轨增益（比器乐高，让唱词清晰）
SING_VOCAL_GAIN = 0.85


def _env(t, dur, cfg):
    a = cfg["attack"]
    env = np.clip(t / a, 0, 1) if a > 0 else np.ones_like(t)
    level = np.exp(-cfg["decay"] * t) if cfg["decay"] > 0 else np.full_like(t, cfg["sustain"])
    rel = np.clip((dur - t) / max(cfg["release"], 1e-4), 0, 1)
    return env * level * rel


def _formant_amps(f0, fset, n_harm):
    """共振峰驱动的谐波振幅表（已归一化）。"""
    amps = []
    for k in range(1, n_harm + 1):
        fk = k * f0
        if fk > 15000:
            break
        a = sum(A * math.exp(-((fk - F) / Bw) ** 2) for (F, Bw, A) in fset)
        amps.append(a)
    mx = max(amps) or 1.0
    return [a / mx for a in amps]


def _render_tone(pitch, dur_sec, vel, cfg, phase0=0.0, formant_variant=0):
    n = max(64, int(dur_sec * RATE))
    t = np.arange(n) / RATE
    freq = 440.0 * 2 ** ((pitch - 69) / 12.0)
    vib = cfg.get("vib")
    if vib:
        rate, depth, delay = vib
        mask = (t > delay).astype(float)
        freq = freq * (1.0 + depth * np.sin(2 * math.pi * rate * np.maximum(t - delay, 0)) * mask)
    phase = 2 * math.pi * np.cumsum(freq) / RATE + phase0
    sig = np.zeros(n)
    if "formants" in cfg:
        fsets = cfg["formants"]
        fset = fsets[formant_variant % len(fsets)]
        f0 = 440.0 * 2 ** ((pitch - 69) / 12.0)
        for k, a in enumerate(_formant_amps(f0, fset, int(cfg.get("n_harm", 24))), start=1):
            sig += a * np.sin(phase * k)
    else:
        for mult, amp, hdec in cfg["harm"]:
            a = amp * (np.exp(-hdec * t) if hdec else 1.0)
            sig += a * np.sin(phase * mult)
    env = _env(t, dur_sec, cfg)
    sig *= env * (vel / 127.0)
    breath = cfg.get("breath") or 0
    if breath:
        nz = np.diff(np.random.uniform(-1, 1, n + 1))
        sig += nz * breath * env * (vel / 127.0)
    detune = cfg.get("detune") or 0
    if detune:
        for sign in (-1, 1):
            f2 = freq * 2 ** (sign * detune / 1200.0)
            ph2 = 2 * math.pi * np.cumsum(f2) / RATE
            d2 = np.zeros(n)
            for mult, amp, hdec in cfg["harm"]:
                a = amp * (np.exp(-hdec * t) if hdec else 1.0)
                d2 += a * np.sin(ph2 * mult)
            sig += d2 * env * (vel / 127.0)
        sig *= 0.5
    return sig


def _add(bufL, bufR, sig, start_sec, pan):
    i0 = int(start_sec * RATE)
    if i0 >= len(bufL) or i0 < 0:
        return
    n = min(len(sig), len(bufL) - i0)
    gL = math.sqrt((1 - pan) / 2 + 0.5)  # 等效功率声像
    gR = math.sqrt(pan / 2 + 0.5)
    bufL[i0:i0 + n] += sig[:n] * gL
    bufR[i0:i0 + n] += sig[:n] * gR


# ---------------------------------------------------------------- 鼓组合成
def _kick():
    n = int(0.30 * RATE)
    t = np.arange(n) / RATE
    f = 42 + 115 * np.exp(-t * 32)
    ph = 2 * math.pi * np.cumsum(f) / RATE
    return np.sin(ph) * np.exp(-t * 7) + np.random.uniform(-1, 1, n) * np.exp(-t * 180) * 0.4


def _snare():
    n = int(0.25 * RATE)
    t = np.arange(n) / RATE
    noise = np.diff(np.random.uniform(-1, 1, n + 1))
    tone = np.sin(2 * math.pi * 185 * t) * np.exp(-t * 22) * 0.6
    return (noise * 0.8 + tone) * np.exp(-t * 16)


def _hat(open_):
    n = int((0.3 if open_ else 0.06) * RATE)
    t = np.arange(n) / RATE
    noise = np.diff(np.random.uniform(-1, 1, n + 1))
    return noise * np.exp(-t * (9 if open_ else 45)) * 0.9


def _clap():
    n = int(0.35 * RATE)
    t = np.arange(n) / RATE
    sig = np.zeros(n)
    for off in (0.0, 0.012, 0.024):
        mask = (t >= off).astype(float)
        sig += np.random.uniform(-1, 1, n) * np.exp(-np.maximum(t - off, 0) * 38) * mask
    return sig * 0.8 + np.random.uniform(-1, 1, n) * np.exp(-t * 8) * 0.2


def _rim():
    n = int(0.09 * RATE)
    t = np.arange(n) / RATE
    return np.sin(2 * math.pi * 820 * t) * np.exp(-t * 60) \
        + np.random.uniform(-1, 1, n) * np.exp(-t * 120) * 0.3


_DRUM_FN = {"kick": _kick, "snare": _snare, "hat": lambda: _hat(False),
            "ohat": lambda: _hat(True), "clap": _clap, "rim": _rim}


def render_wav(song, path, progress_cb=None):
    """song(realize 产物) -> WAV 文件。progress_cb(fraction 0..1) 汇报渲染进度。"""
    groups = _render_groups(song, progress_cb)
    out = _mix_master(song, groups)
    _write_wav(out, path)
    if progress_cb:
        progress_cb(1.0)
    return path


def render_stems(song, out_dir, progress_cb=None):
    """分轨导出：每个声部独立归一化成 WAV（干声，无总线混响）。

    stem_vocal.wav(主唱/主旋律) stem_harmony.wav(和声) stem_bass.wav stem_drums.wav
    """
    groups = _render_groups(song, progress_cb)
    return _write_stems(groups, out_dir, progress_cb)


def render_all(song, wav_path, stems_dir, progress_cb=None):
    """一次渲染出整曲母带 + 4 条分轨（共用同一批声部 buffer，性能翻倍）。"""
    groups = _render_groups(song, progress_cb)
    if progress_cb:
        progress_cb(0.85)
    out = _mix_master(song, groups)
    _write_wav(out, wav_path)
    if progress_cb:
        progress_cb(0.92)
    paths = _write_stems(groups, stems_dir, None)
    if progress_cb:
        progress_cb(1.0)
    return wav_path, paths


def _render_groups(song, progress_cb=None):
    """渲染全部分组（float32，只渲染一次；母带与分轨共用）。"""
    import os
    n_total, spb, total_sec, voices, _ = _prepare(song)
    groups = {}
    for gi, (group, roles) in enumerate(GROUP_ROLES):
        L = np.zeros(n_total, dtype=np.float32)
        R = np.zeros(n_total, dtype=np.float32)
        for role in roles:
            _render_role(L, R, song, role, voices, spb, total_sec)
        groups[group] = (L, R)
        if progress_cb:
            progress_cb(min(0.8, (gi + 1) / len(GROUP_ROLES) * 0.8))
    return groups


def _write_stems(groups, out_dir, progress_cb=None):
    import os
    paths = []
    for gi, (group, _roles) in enumerate(GROUP_ROLES):
        L, R = groups[group]
        peak = max(float(np.max(np.abs(L))) if L.size else 0.0,
                   float(np.max(np.abs(R))) if R.size else 0.0, 1e-9)
        g = np.float32(0.9 / peak)
        path = os.path.join(out_dir, "stem_%s.wav" % group)
        _write_wav(np.stack([L * g, R * g], axis=1), path)
        paths.append(path)
        if progress_cb:
            progress_cb((gi + 1) / len(GROUP_ROLES))
    return paths


def _render_sing(bufL, bufR, song, spb, total_sec):
    """AI 演唱：共振峰假声逐字合成，音高跟随旋律，替代器乐主旋律。"""
    from . import singer  # 局部导入，避免模块初始化顺序问题
    mono = singer.render_sing_mono(song, song.get("sing_events") or [], total_sec, spb)
    if mono is None:
        return
    # 22050 -> 44100 升采样
    up = np.interp(np.linspace(0, len(mono) - 1, len(mono) * 2),
                   np.arange(len(mono)), mono)
    n = min(len(up), len(bufL))
    bufL[:n] += up[:n] * SING_VOCAL_GAIN
    bufR[:n] += up[:n] * SING_VOCAL_GAIN


def _prepare(song):
    spb = 60.0 / song["bpm"]
    total_sec = song["duration_sec"] + 0.6
    n_total = int(total_sec * RATE)
    voices = dict(VOICE_FOR.get(song["spec"]["genre"], VOICE_FOR["pop"]))
    lv = (song.get("spec") or {}).get("lead_voice")
    if lv in WAVES:
        voices["melody"] = lv  # 用户指定的主奏/人声音色覆盖
    return n_total, spb, total_sec, voices, None


def _pan_of(role, e, base):
    """立体声摆动：琶音/键琴左右交替（ping-pong），其他声部用固定声像。"""
    if role == "arps":
        return base + (0.5 if int(e["start"] * 2) % 2 else -0.5) * 0.45
    if role == "stabs":
        return base + (0.35 if int(e["start"]) % 2 else -0.35)
    return base


def _render_role(bufL, bufR, song, role, voices, spb, total_sec):
    if role == "melody" and song.get("sing_events"):
        _render_sing(bufL, bufR, song, spb, total_sec)  # AI 演唱替代器乐主旋律
        return
    if role == "drums":
        cache = {}
        for e in song["drums"]:
            dtype = e["type"] if e["type"] in _DRUM_FN else "hat"
            if dtype not in cache:
                cache[dtype] = _DRUM_FN[dtype]()
            gain = DRUM_GAIN.get(dtype, .3) * (e["vel"] / 127.0)
            # 镲片微立体声，底鼓/军鼓居中
            pan = (0.12 if int(e["start"] * 4) % 2 else -0.12) if dtype in ("hat", "ohat") else 0.0
            _add(bufL, bufR, cache[dtype] * gain, e["start"] * spb, pan)
        return
    cfg_name = voices.get(role)
    cfg = WAVES.get(cfg_name, WAVES["synth_lead"])
    gain = ROLE_GAIN.get(role, .2)
    if cfg_name == "vocal_hum":
        gain = ROLE_GAIN.get(role, .2) * 1.15
    base_pan = ROLE_PAN.get(role, 0.0)
    for e in song.get(role, []):
        start_sec = e["start"] * spb
        if start_sec >= total_sec:
            continue
        vel = e.get("vel", 90)
        pan = _pan_of(role, e, base_pan)
        if "pitches" in e:  # 和弦事件
            for k, p in enumerate(e["pitches"]):
                sig = _render_tone(p, max(0.05, e["dur"] * spb), vel, cfg, phase0=k * 0.7)
                _add(bufL, bufR, sig * gain, start_sec, pan)
        else:
            variant = int(e["start"] / 2) % 2 if cfg_name == "vocal_hum" else 0
            sig = _render_tone(e["pitch"], max(0.05, e["dur"] * spb), vel, cfg,
                               formant_variant=variant)
            _add(bufL, bufR, sig * gain, start_sec, pan)


def _mix_master(song, groups):
    """混音：段落动态曲线 + 人声侧链避让 → 交给总线 _master。"""
    n_total = len(groups["vocal"][0])
    curve = _dynamics_curve(song, n_total)

    vL, vR = groups["vocal"]
    hL, hR = groups["harmony"]
    bL, bR = groups["bass"]
    dL, dR = groups["drums"]

    # 人声包络 → 和声/贝斯避让（最多约 2.5dB），人声永远听得清
    vmono = (np.abs(vL) + np.abs(vR)) * 0.5
    env = _upsample_env(_block_env(vmono), n_total)
    peak_env = float(np.max(env)) if env.size else 0.0
    if peak_env > 1e-9:
        env = env / peak_env
    duck = (1.0 - 0.25 * env).astype(np.float32)

    L = vL * curve + (hL * curve) * duck + bL * duck + dL
    R = vR * curve + (hR * curve) * duck + bR * duck + dR
    return _master(L, R)


def _dynamics_curve(song, n_total):
    """段落动态：intro 渐入(0.55→1) / bridge 收(0.85) / outro 渐出(1→0.5)。"""
    curve = np.ones(n_total, dtype=np.float32)
    spb = 60.0 / song["bpm"]
    for sec in song["sections"]:
        i0 = min(max(0, int(sec["start_beat"] * spb * RATE)), max(0, n_total - 2))
        i1 = min(n_total, max(i0 + 2, int((sec["start_beat"] + sec["beats"]) * spb * RATE)))
        if sec["name"] == "intro":
            curve[i0:i1] = np.linspace(0.55, 1.0, i1 - i0, dtype=np.float32)
        elif sec["name"] == "outro":
            curve[i0:i1] = np.linspace(1.0, 0.5, i1 - i0, dtype=np.float32)
        elif sec["name"] == "bridge":
            curve[i0:i1] = np.float32(0.85)
    return curve


def _block_env(x, blocks_per_sec=100):
    """低速率（默认100Hz）分块 RMS 包络，供侧链/压缩使用。"""
    n_blk = max(1, len(x) // blocks_per_sec)
    usable = (len(x) // n_blk) * n_blk
    if usable <= 0:
        return np.zeros(1, dtype=np.float32)
    blocks = np.asarray(x[:usable], dtype=np.float32).reshape(-1, n_blk)
    return np.sqrt((blocks ** 2).mean(axis=1) + 1e-12)


def _upsample_env(env, n_total):
    if len(env) >= n_total:
        return env[:n_total].astype(np.float32)
    idx = np.linspace(0, len(env) - 1, n_total)
    return np.interp(idx, np.arange(len(env)), env).astype(np.float32)


def _fft_eq(mix, highpass_hz=28.0, shelf_lo=2500.0, shelf_hi=9000.0, shelf_gain=1.26):
    """静态 EQ（FFT 域）：去次声隆隆声 + 高频存在感提升。"""
    n = mix.shape[1]
    spec = np.fft.rfft(mix, axis=1)
    freqs = np.fft.rfftfreq(n, 1.0 / RATE).astype(np.float32)
    gain = np.ones_like(freqs)
    low = freqs < highpass_hz
    gain[low] = (freqs[low] / highpass_hz) ** 2
    hi = freqs >= shelf_hi
    gain[hi] = shelf_gain
    band = (freqs >= shelf_lo) & (freqs < shelf_hi)
    rise = (freqs[band] - shelf_lo) / max(shelf_hi - shelf_lo, 1.0)
    gain[band] = 1.0 + (shelf_gain - 1.0) * np.minimum(1.0, rise * 2.0)
    return np.fft.irfft(spec * gain[None, :], n=n, axis=1).astype(np.float32)


def _compress(mix, thresh=10 ** (-14 / 20.0), ratio=3.0, makeup=1.26):
    """缓慢的总线压缩（3:1，-14dB 起，快攻慢放）。"""
    L, R = mix[0], mix[1]
    mono = (np.abs(L) + np.abs(R)) * 0.5
    env = _block_env(mono)
    level = np.maximum(env, 1e-6)
    gain = np.ones_like(level)
    over = level > thresh
    gain[over] = (thresh / level[over]) ** (1.0 - 1.0 / ratio)
    gain = np.clip(gain, 0.4, 1.0)
    outg = np.empty_like(gain)
    acc = 1.0
    for i, g in enumerate(gain):  # 块级(100Hz)循环，量级千次以内
        acc += (g - acc) * (0.25 if g < acc else 0.06)
        outg[i] = acc
    g_up = _upsample_env(outg * makeup, mix.shape[1])
    return (mix * g_up[None, :]).astype(np.float32)


def _master(L, R):
    """总线：三抽头混响 → EQ → 压缩 → 软限幅 → 峰值归一。"""
    mix = np.stack([L, R]).astype(np.float32)
    for delay_s, g in ((0.089, .22), (0.031, .12), (0.147, .09)):
        d = int(delay_s * RATE)
        mix[:, d:] += np.float32(g) * mix[:, :-d]
    mix = _fft_eq(mix)
    mix = _compress(mix)
    out = mix.T
    out = np.tanh(out * np.float32(1.15))
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1e-9:
        out = (out * np.float32(0.90 / peak)).astype(np.float32)
    return out


def _write_wav(out, path):
    pcm = (out * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())
