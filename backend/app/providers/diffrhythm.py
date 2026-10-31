"""DiffRhythm 本地神经歌声 Provider（Apache-2.0 模型，权重在 DiffRhythm/pretrained/）。

歌词(带小节时间戳) + 风格描述/参考音频 -> 95s/285s 真人声演唱成品 WAV。
模型推理在独立子进程（backend/neural/diffrhythm_runner.py，跑在 DiffRhythm/.venv-neural，
torch CPU）里进行，本模块只负责编排：探测可用性、拼装任务、解析进度、超时/取消。
生成后可选两步后处理（同样子进程隔离，失败不影响成品）：
    align()    faster-whisper 强制对齐 -> 每句歌词真实起止时间（精确卡拉OK）
    separate() demucs 源分离 -> 人声/鼓/贝斯/伴奏 4 分轨

配置（config.local.json 可选 "neural" 节，或环境变量）：
    neural.python        AI_MUSIC_NEURAL_PYTHON   venv 解释器（默认 DiffRhythm/.venv-neural）
    neural.steps         AI_MUSIC_NEURAL_STEPS    CFM 采样步数 2-64（默认 16）
    neural.timeout_min   AI_MUSIC_NEURAL_TIMEOUT  95s 单曲超时分钟（默认 90，285s 自动 x3）
    neural.device        AI_MUSIC_NEURAL_DEVICE   cpu / mps / cuda（默认 cpu，cuda 自动）
    neural.audio_length  AI_MUSIC_NEURAL_LENGTH   95 或 285（285 需 full 权重）
    neural.whisper_model AI_MUSIC_NEURAL_WHISPER  对齐用 whisper 模型（默认 small）
    neural.align / neural.separate                true/false 开关后处理
"""
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid

from .audio_base import AudioProvider, AudioProviderError, CancelledError

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))  # 项目根
DIFFRHYTHM_DIR = os.path.join(_ROOT, "DiffRhythm")
RUNNER = os.path.join(_ROOT, "backend", "neural", "diffrhythm_runner.py")
ALIGN_RUNNER = os.path.join(_ROOT, "backend", "neural", "align_runner.py")
DEMUCS_RUNNER = os.path.join(_ROOT, "backend", "neural", "demucs_runner.py")
PERSONA_RUNNER = os.path.join(_ROOT, "backend", "neural", "persona_runner.py")
FULL_CFM = os.path.join(DIFFRHYTHM_DIR, "pretrained", "cfm_model_full.pt")

# 风格英文关键词（MuQ-MuLan 文本编码器，参考官方示例 prompt 形态）
_GENRE_TAGS = {
    "pop": "pop, catchy, modern, radio-friendly",
    "rock": "rock, electric guitar, powerful drums",
    "electronic": "electronic, EDM, synthesizer, dance",
    "ballad": "ballad, piano, emotional, slow",
    "folk": "folk, acoustic guitar, warm, storytelling",
    "guofeng": "chinese traditional, guzheng, oriental, ancient china style",
    "lofi": "lo-fi hip hop, chill, mellow beats, vinyl crackle",
    "jazz": "jazz, swing, saxophone, smooth",
    "rap": "hip hop, rap, boom bap, drum machine, rhythmic rap flow, rap vocal",
    "rap_trap": "trap, hip hop, 808 bass, hi-hat rolls, rap vocal, dark mood",
}
_MOOD_TAGS = {
    "happy": "uplifting, bright", "sad": "melancholic, sorrowful",
    "chill": "relaxed, mellow", "energetic": "energetic, powerful",
    "romantic": "romantic, tender", "epic": "epic, cinematic, orchestral",
}

_avail_cache = {"ts": 0.0, "result": (False, "未探测")}
_mod_cache = {}     # module名 -> (ts, bool|None)：venv 可选依赖缓存（None=检测中）
_probe_inflight = set()


def full_weights_ready():
    """完整版（285s）权重是否已下载。"""
    return os.path.isfile(FULL_CFM)


def load_neural_config(config_path=None):
    """读 neural 配置节：环境变量 > config.local.json > 默认值。"""
    cfg = {"enabled": True, "steps": 16, "timeout_min": 150, "python": None,
           "device": None, "audio_length": 95, "whisper_model": "small",
           "align": True, "separate": True}
    path = config_path or os.environ.get("AI_MUSIC_CONFIG") or os.path.join(_ROOT, "config.local.json")
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k in cfg:
                if k in (data.get("neural") or {}):
                    cfg[k] = data["neural"][k]
        except (OSError, ValueError):
            pass
    env_map = {"AI_MUSIC_NEURAL_PYTHON": "python", "AI_MUSIC_NEURAL_STEPS": "steps",
               "AI_MUSIC_NEURAL_TIMEOUT": "timeout_min", "AI_MUSIC_NEURAL_DEVICE": "device",
               "AI_MUSIC_NEURAL_LENGTH": "audio_length",
               "AI_MUSIC_NEURAL_WHISPER": "whisper_model"}
    for env_k, cfg_k in env_map.items():
        if os.environ.get(env_k):
            try:
                cfg[cfg_k] = int(os.environ[env_k]) if cfg_k not in ("python", "device", "whisper_model") \
                    else os.environ[env_k]
            except ValueError:
                pass
    cfg["steps"] = max(2, min(64, int(cfg["steps"] or 16)))
    cfg["timeout_min"] = max(1, int(cfg["timeout_min"] or 150))
    length = int(cfg["audio_length"] or 95)
    cfg["audio_length"] = length if length == 95 else min(285, max(96, length))
    return cfg


def venv_has_module(module, config_path=None, wait=False):
    """venv 里是否有某可选依赖（faster_whisper / demucs）。
    默认后台探测（/api/meta 永不阻塞，未探测完返回 None）；
    wait=True 用于生成流水线内：同步探测并返回确定的 bool。"""
    now = time.time()
    ts, state = _mod_cache.get(module, (0.0, None))
    if state is not None and now - ts < 600:
        return state
    if module in _probe_inflight and not wait:
        return state
    cfg = load_neural_config(config_path)
    py = cfg.get("python") or default_python()

    def _probe_sync():
        ok = False
        if os.path.isfile(py):
            try:
                r = subprocess.run([py, "-c", "import %s" % module],
                                   capture_output=True, timeout=120)
                ok = (r.returncode == 0)
            except (OSError, subprocess.SubprocessError):
                ok = False
        _mod_cache[module] = (time.time(), ok)
        return ok

    if wait:
        return _probe_sync()
    _probe_inflight.add(module)

    def _job():
        try:
            _probe_sync()
        finally:
            _probe_inflight.discard(module)
    threading.Thread(target=_job, daemon=True).start()
    return state


def resolve_audio_length(req=None, cfg=None):
    """95（标准）或 285（完整版，需 full 权重）。req.full_song 可请求 285。"""
    cfg = cfg or load_neural_config()
    length = int(cfg.get("audio_length") or 95)
    if req and req.get("full_song"):
        length = 285
    if length != 95 and not full_weights_ready():
        raise AudioProviderError(
            "完整版（285s）需要 DiffRhythm-full 权重：DiffRhythm/pretrained/cfm_model_full.pt "
            "（见 docs/CONFIG.md「全曲版」）")
    return length


def build_lyric_lines(spec, max_sec=93.0):
    """SongSpec -> [[起始秒, 歌词], ...]（按小节时间排布，空行跳过，按时长截断）。"""
    bpm = spec.get("bpm") or 100
    bpb = spec.get("beats_per_bar") or 4
    bar_sec = bpb * 60.0 / max(30, bpm)
    lines, bar = [], 0
    for sec in spec.get("structure", []):
        for i, text in enumerate(sec.get("lyrics", [])):
            t = bar * bar_sec
            if text and text.strip() and t < max_sec:
                lines.append([round(t, 2), str(text).strip()])
            bar += 1
    return lines


def default_python():
    """venv 解释器默认路径（含 Windows 变体）。"""
    p = os.path.join(DIFFRHYTHM_DIR, ".venv-neural", "Scripts", "python.exe" if os.name == "nt" else "bin", "python")
    return p if os.name == "nt" else os.path.join(DIFFRHYTHM_DIR, ".venv-neural", "bin", "python")


class DiffRhythmProvider(AudioProvider):
    name = "diffrhythm"

    def __init__(self, config_path=None):
        self.config_path = config_path

    # ------------------------------------------------------------ 可用性
    def available(self):
        now = time.time()
        if now - _avail_cache["ts"] < 60:
            return _avail_cache["result"]
        result = self._probe()
        _avail_cache.update(ts=now, result=result)
        return result

    def _probe(self):
        cfg = load_neural_config(self.config_path)
        if not cfg.get("enabled", True):
            return False, "神经引擎已在配置中禁用（neural.enabled=false）"
        py = cfg.get("python") or default_python()
        if not (os.path.isfile(py) and os.access(py, os.X_OK)):
            return False, "未找到神经引擎 venv（%s）——运行 bash scripts/setup_neural.sh 安装" % py
        if not os.path.isfile(RUNNER):
            return False, "缺少 runner：%s" % RUNNER
        for f in ("pretrained/cfm_model.pt", "pretrained/vae_model.pt"):
            if not os.path.isfile(os.path.join(DIFFRHYTHM_DIR, f)):
                return False, "缺少模型权重 %s" % f
        muq_dir = os.path.join(DIFFRHYTHM_DIR, "pretrained", "muq")
        if not any(os.path.isfile(os.path.join(muq_dir, f))
                   for f in ("model.safetensors", "pytorch_model.bin")):
            return False, "缺少模型权重 pretrained/muq/*"
        if not shutil.which("espeak-ng"):
            return False, "缺少 espeak-ng（brew install espeak-ng）"
        return True, "就绪（CPU 渲染约 15-30 分钟/首）"

    def engine_info(self):
        cfg = load_neural_config(self.config_path)
        ok, reason = self.available()
        return {"available": ok, "reason": reason, "steps": cfg["steps"],
                "timeout_min": cfg["timeout_min"],
                "audio_length_sec": cfg["audio_length"],
                "full_available": ok and full_weights_ready(),
                "align_available": ok and cfg.get("align", True)
                and venv_has_module("faster_whisper", self.config_path),
                "separate_available": ok and cfg.get("separate", True)
                and venv_has_module("demucs", self.config_path)}

    # ------------------------------------------------------------ 歌词/风格
    @staticmethod
    def build_style_prompt(spec, req=None):
        req = req or {}
        genre = spec.get("genre")
        if genre == "rap" and (spec.get("bpm") or 0) >= 108:
            genre = "rap_trap"  # 高速说唱用 trap 标签
        genre = genre if genre in _GENRE_TAGS else "pop"
        mood = spec.get("mood") if spec.get("mood") in _MOOD_TAGS else "chill"
        parts = [_GENRE_TAGS[genre], _MOOD_TAGS[mood], "vocal, singing, mandarin song"]
        desc = (req.get("description") or spec.get("description") or "").strip()
        if desc:
            parts.append(desc[:120])
        return ", ".join(parts)

    # ------------------------------------------------------------ 生成
    def generate(self, req, spec, out_wav, progress_cb=None, cancel_event=None):
        progress_cb = progress_cb or (lambda frac, note: None)
        cfg = load_neural_config(self.config_path)
        py = cfg.get("python") or default_python()
        if not os.path.isfile(py):
            raise AudioProviderError("神经引擎 venv 不存在（%s）——先运行 scripts/setup_neural.sh" % py)
        audio_length = resolve_audio_length(req, cfg)

        job = {
            "lyric_lines": build_lyric_lines(spec, max_sec=audio_length - 2),
            "style_prompt": None if req.get("ref_audio_path") else self.build_style_prompt(spec, req),
            "ref_audio": req.get("ref_audio_path"),
            "seed": spec.get("seed"),
            "steps": cfg["steps"],
            "audio_length": audio_length,
            "out_wav": os.path.abspath(out_wav),
        }
        if not job["lyric_lines"]:
            raise AudioProviderError("没有可演唱的歌词（神经引擎需要歌词）")
        job_path = os.path.join(os.path.dirname(os.path.abspath(out_wav)), "neural_job.json")
        job["_job_path"] = job_path

        job["id"] = uuid.uuid4().hex[:12]
        if req.get("style_prompt_npy"):
            job["style_prompt_npy"] = os.path.abspath(req["style_prompt_npy"])
        edit = req.get("edit")
        if edit:
            ref = os.path.abspath(edit.get("ref_song") or "")
            dur = _wav_duration(ref)
            if dur < audio_length - 1.0:
                raise AudioProviderError(
                    "原曲时长 %.0fs 不足以做 %.0fs 的分段重唱" % (dur, audio_length))
            job["edit"] = {"ref_song": ref,
                           "segments": edit.get("segments") or [[0, audio_length]]}
        with open(job_path, "w", encoding="utf-8") as f:
            json.dump(job, f, ensure_ascii=False)

        # 常驻 worker：模型只加载一次，超时/取消杀掉 worker（下次任务自动重启重载）
        timeout_min = cfg["timeout_min"] * (1 if audio_length == 95 else 3)
        done = _WORKER.run_job(job, timeout_min, progress_cb, cancel_event)
        if not os.path.isfile(out_wav):
            raise AudioProviderError("DiffRhythm 未产出音频文件")
        return done


def _wav_duration(path):
    import wave
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate() or 1)
    except Exception:
        return 0.0


class NeuralWorker:
    """常驻 diffrhythm_runner --serve 的管理器：懒启动、单飞、死亡自动重启。

    协议：stdin 发 job.json 路径行；stdout JSON 行（ready/done/error/progress，
    done/error 带 id）。超时或取消时杀掉 worker 进程（代价=下次任务重载模型 1-2 分钟）。
    """

    def __init__(self):
        self._proc = None
        self._log_fh = None
        self._lock = threading.Lock()

    def _log_path(self):
        return os.path.join(DIFFRHYTHM_DIR, ".worker.log")

    def _spawn(self):
        cfg = load_neural_config()
        py = cfg.get("python") or default_python()
        if not os.path.isfile(py):
            raise AudioProviderError("神经引擎 venv 不存在——先运行 scripts/setup_neural.sh")
        env = dict(os.environ)
        env.setdefault("HF_HOME", os.path.join(DIFFRHYTHM_DIR, "pretrained", "hf"))
        env.setdefault("OMP_NUM_THREADS", str(min(8, os.cpu_count() or 4)))
        if cfg.get("device"):
            env["DIFFRHYTHM_DEVICE"] = str(cfg["device"])
        env.pop("PYTHONPATH", None)
        env["PYTHONUNBUFFERED"] = "1"
        if self._log_fh:
            try:
                self._log_fh.close()
            except OSError:
                pass
        self._log_fh = open(self._log_path(), "ab")
        proc = subprocess.Popen(
            [py, RUNNER, "--serve"], cwd=DIFFRHYTHM_DIR, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log_fh,
            text=True, encoding="utf-8", errors="replace")
        self._proc = proc
        # 等待 ready（首次含模型加载 1-2 分钟）
        deadline = time.time() + 420
        for line in proc.stdout:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("type") == "ready":
                return
            if msg.get("type") == "error":
                self._proc = None
                raise AudioProviderError("神经 worker 启动失败: %s" % msg.get("error"))
            if time.time() > deadline:
                break
        if proc.poll() is None:
            proc.kill()
        self._proc = None
        raise AudioProviderError("神经 worker 启动超时（详情见 %s）" % self._log_path())

    def _ensure(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        self._spawn()

    def _kill(self):
        if self._proc is not None:
            try:
                self._proc.kill()
            except OSError:
                pass
        self._proc = None

    def run_job(self, job, timeout_min, progress_cb=None, cancel_event=None):
        progress_cb = progress_cb or (lambda frac, note: None)
        with self._lock:
            self._ensure()
            proc = self._proc
            jid = str(job.get("id") or uuid.uuid4().hex[:12])
            path = job.get("_job_path")
            if not path:
                raise AudioProviderError("worker 任务缺少 _job_path")
            try:
                proc.stdin.write(path.strip() + "\n")
                proc.stdin.flush()
            except (OSError, ValueError):
                self._kill()
                raise AudioProviderError("神经 worker 管道已断开，请重试（将自动重启）")

            q = queue.Queue()

            def _pump():
                try:
                    for line in proc.stdout:
                        q.put(line)
                    q.put(None)  # EOF
                except (OSError, ValueError):
                    q.put(None)
            threading.Thread(target=_pump, daemon=True).start()

            deadline = time.time() + timeout_min * 60
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    self._kill()
                    raise CancelledError("已取消")
                if time.time() > deadline:
                    self._kill()
                    raise AudioProviderError("渲染超时（%d 分钟），worker 已重启" % timeout_min)
                try:
                    raw = q.get(timeout=0.5)
                except queue.Empty:
                    if proc.poll() is not None:
                        self._proc = None
                        raise AudioProviderError(
                            "神经 worker 意外退出（详情见 %s）" % self._log_path())
                    continue
                if raw is None:
                    self._proc = None
                    raise AudioProviderError(
                        "神经 worker 管道关闭（详情见 %s）" % self._log_path())
                raw = raw.strip()
                if not raw.startswith("{"):
                    continue
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                mtype = msg.get("type")
                if mtype == "progress":
                    progress_cb(0.06 + 0.88 * float(msg.get("frac") or 0),
                                str(msg.get("note") or ""))
                elif mtype == "done":
                    return msg
                elif mtype == "error":
                    raise AudioProviderError("DiffRhythm 渲染失败: %s" % msg.get("error"))


# 模块级单例：全进程共用一个常驻 worker（模型 ~10GB 只加载一次）
_WORKER = NeuralWorker()

# ------------------------------------------------------------ 后处理（模块级，jobs 直接调用）
def _run_subprocess_job(script, job, timeout_min, config_path=None, progress_cb=None,
                        cancel_event=None):
    """通用子进程执行：JSON 进度协议，返回最后一个 done 消息 dict。"""
    cfg = load_neural_config(config_path)
    py = cfg.get("python") or default_python()
    job_path = job.get("_job_path")
    if not job_path:
        raise AudioProviderError("缺少 job 文件路径")
    with open(job_path, "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False)
    env = dict(os.environ)
    env.setdefault("HF_HOME", os.path.join(DIFFRHYTHM_DIR, "pretrained", "hf"))
    env.setdefault("OMP_NUM_THREADS", str(min(8, os.cpu_count() or 4)))
    env.pop("PYTHONPATH", None)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [py, script, job_path], cwd=DIFFRHYTHM_DIR, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        encoding="utf-8", errors="replace")
    deadline = time.time() + timeout_min * 60
    stop = threading.Event()

    def watchdog():
        while not stop.wait(timeout=1.0):
            if cancel_event is not None and cancel_event.is_set():
                proc.kill()
                return
            if time.time() > deadline:
                proc.kill()
                return
    threading.Thread(target=watchdog, daemon=True).start()

    error, done = None, None
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("type") == "progress":
                if progress_cb:
                    progress_cb(float(msg.get("frac") or 0), str(msg.get("note") or ""))
            elif msg.get("type") == "done":
                done = msg
                break
            elif msg.get("type") == "error":
                error = str(msg.get("error") or "未知错误")
        code = proc.wait()
    finally:
        stop.set()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("已取消")
    if error:
        raise AudioProviderError("%s 失败: %s" % (os.path.basename(script), error))
    if code != 0:
        raise AudioProviderError("%s 进程异常退出（code=%d）" % (os.path.basename(script), code))
    return done

def align(wav, lines, config_path=None, progress_cb=None, cancel_event=None):
    """Whisper 强制对齐：-> [[start,end,text,conf],...] 或 None（不可用/失败不致命）。
    lines 格式 [[est_sec, text], ...]。"""
    cfg = load_neural_config(config_path)
    if not cfg.get("align", True) or venv_has_module("faster_whisper", config_path, wait=True) is not True:
        return None
    import tempfile
    fd, job_path = tempfile.mkstemp(prefix="align-", suffix=".json",
                                    dir=os.path.dirname(os.path.abspath(wav)))
    os.close(fd)
    out_path = job_path + ".out"
    job = {"wav": os.path.abspath(wav), "lines": lines,
           "model": cfg.get("whisper_model") or "small", "out": out_path,
           "_job_path": job_path}
    try:
        done = _run_subprocess_job(ALIGN_RUNNER, job, 15, config_path, progress_cb, cancel_event)
        if not done or not done.get("events"):
            return None
        with open(out_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (AudioProviderError, CancelledError, OSError, ValueError):
        return None  # 对齐是增强功能，任何失败都退回小节时间轴
    finally:
        for p in (job_path, out_path):
            try:
                os.remove(p)
            except OSError:
                pass


def separate(wav, out_dir, config_path=None, progress_cb=None, cancel_event=None):
    """Demucs 源分离 -> {vocals/drums/bass/other: 文件名}；不可用时返回 None。"""
    cfg = load_neural_config(config_path)
    if not cfg.get("separate", True) or venv_has_module("demucs", config_path, wait=True) is not True:
        return None
    out_dir = os.path.abspath(out_dir)  # 子进程 cwd 不同，全部路径必须绝对
    job = {"wav": os.path.abspath(wav), "out_dir": out_dir,
           "model": "htdemucs", "_job_path": os.path.join(out_dir, "demucs_job.json")}
    done = _run_subprocess_job(DEMUCS_RUNNER, job, 30, config_path, progress_cb, cancel_event)
    return (done or {}).get("stems") or {}


def lite_score(spec, duration_sec, lyric_events=None):
    """audio_only 作品的轻量曲谱：段落/歌词/和弦齐全、音符数组为空。
    lyric_events: Whisper 对齐后的 [[start,end,text,conf],...]（无则空数组）。"""
    bpm = spec.get("bpm") or 100
    bpb = spec.get("beats_per_bar") or 4
    sections, start_bar = [], 0
    for sec in spec.get("structure", []):
        bars = int(sec.get("bars") or 0)
        sections.append({"name": sec.get("name"), "label": sec.get("name"),
                         "bars": bars, "start_bar": start_bar,
                         "start_beat": start_bar * bpb, "beats": bars * bpb,
                         "lyrics": list(sec.get("lyrics", [])),
                         "chords": list(sec.get("chords", []))})
        start_bar += bars
    return {"duration_sec": duration_sec, "bpm": bpm, "beats_per_bar": bpb,
            "total_beats": start_bar * bpb, "audio_only": True,
            "lyric_events": [list(e) for e in (lyric_events or [])],
            "sections": sections, "melody": [], "bass": [], "arps": [],
            "pads": [], "stabs": [], "drums": [], "chords": []}


# 模块级单例（server / jobs 共用）
provider = DiffRhythmProvider()
