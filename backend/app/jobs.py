"""异步任务队列：生成一首歌 = 作曲 -> 编曲 -> 渲染，前端轮询进度。

v4：音频成品类 Provider（DiffRhythm 本地神经歌声）走独立旁路——
作曲仍由 LLM/Mock 产出歌词与结构，但渲染交给神经模型子进程，
产出 95s 真人声演唱 WAV；支持任务取消（DELETE /api/jobs/{id}）。
v5：多音频 Provider（本地 DiffRhythm 常驻 worker / 云端 Mureka）、
音色 Persona（embedding 下发）、分段重唱（edit 旁路）、专辑批量。
"""
import os
import json
import shutil
import threading
import time
import uuid
import wave

from . import personas, schemas, store as store_mod
from .music import midi, mock_composer, realizer, singer, synth
from .providers import llm as llm_provider
from .providers import mureka as mureka_provider
from .providers.audio_base import CancelledError
from .providers.diffrhythm import align as neural_align
from .providers.diffrhythm import build_lyric_lines as neural_lyric_lines
from .providers.diffrhythm import lite_score as neural_lite_score
from .providers.diffrhythm import provider as diffrhythm_provider
from .providers.diffrhythm import resolve_audio_length as neural_resolve_length
from .providers.diffrhythm import separate as neural_separate
from .providers.diffrhythm import venv_has_module as neural_has_module

STAGES_ZH = {
    "queued": "排队中…",
    "composing": "AI 正在构思和声与结构…",
    "arranging": "正在编曲与生成旋律…",
    "rendering": "正在渲染音频与分轨…",
    "stems": "正在导出分轨…",
    "singing_neural": "🧠 歌声渲染中…",
    "done": "完成",
    "error": "出错",
    "canceled": "已取消",
}

MAX_RENDER_SLOTS = 2      # 本地渲染同时上限（防内存挤爆），其余排队
MAX_NEURAL_SLOTS = 1      # 神经模型同时上限（模型占 ~10GB 内存）
MAX_PENDING_JOBS = 32     # 防止无限创建后台线程耗尽内存
AUDIO_PROVIDERS = {"diffrhythm", "mureka"}


def _wav_duration(path):
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate() or 1)
    except Exception:
        return 0.0


class JobManager:
    def __init__(self, store, config_getter):
        self.store = store
        self._get_config = config_getter
        self._jobs = {}
        self._cancel_events = {}
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(MAX_RENDER_SLOTS)
        self._neural_slot = threading.BoundedSemaphore(MAX_NEURAL_SLOTS)
        self._job_dir = (os.path.join(self.store.lib_dir, "_jobs")
                         if self.store is not None else None)
        if self._job_dir:
            os.makedirs(self._job_dir, exist_ok=True)
            self._load_interrupted_jobs()

    def _job_path(self, job_id):
        if not self._job_dir:
            return None
        safe = "".join(c for c in str(job_id) if c.isalnum() or c in "-_")
        return os.path.join(self._job_dir, safe + ".json")

    def _persist(self, job):
        """Persist lifecycle snapshots so restart behavior is explicit."""
        if not self._job_dir:
            return
        try:
            store_mod.Library._write_json(self._job_path(job["id"]), dict(job))
        except (OSError, TypeError, ValueError):
            # A status snapshot must never break audio generation.
            pass

    def _load_interrupted_jobs(self):
        try:
            names = os.listdir(self._job_dir)
        except OSError:
            return
        for name in names:
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(self._job_dir, name), "r", encoding="utf-8") as f:
                    job = json.load(f)
            except (OSError, ValueError):
                continue
            if not isinstance(job, dict) or not job.get("id"):
                continue
            if job.get("status") in ("queued", "running"):
                job.update(status="error", stage="error",
                           stage_note="服务重启，中断任务",
                           error="服务在任务完成前重启，任务未继续，请重新提交",
                           interrupted=True)
                self._persist(job)
            self._jobs[job["id"]] = job

    def submit(self, req):
        job_id = uuid.uuid4().hex[:12]
        job = {"id": job_id, "status": "queued", "stage": "queued",
               "stage_note": STAGES_ZH["queued"], "progress": 0.0,
               "created_at": time.time(), "error": None, "result": None}
        with self._lock:
            pending = sum(1 for item in self._jobs.values()
                          if item["status"] in ("queued", "running"))
            if pending >= MAX_PENDING_JOBS:
                raise RuntimeError("任务队列已满（最多同时保留 %d 个未完成任务），请稍后重试"
                                   % MAX_PENDING_JOBS)
            self._jobs[job_id] = job
            self._cancel_events[job_id] = threading.Event()
            self._persist(job)
            if len(self._jobs) > 100:  # 只清理终态任务，不能丢运行中任务
                terminal = [item for item in self._jobs.values()
                            if item["status"] in ("done", "error", "canceled")]
                for old in sorted(terminal, key=lambda j: j["created_at"])[:len(self._jobs) - 100]:
                    self._jobs.pop(old["id"], None)
                    self._cancel_events.pop(old["id"], None)
        # 结果复用缓存：同参数(+seed)直接返回已有作品
        fp = store_mod.fingerprint_of(req)
        hit = self.store.find_fingerprint(fp)
        if hit:
            job.update(status="done", stage="done", progress=1.0,
                       stage_note="♻️ 与已有作品相同，直接复用（换个 seed 或「再来一版」可得新曲）",
                       result={"song_id": hit["id"], "title": hit["title"],
                               "duration_sec": hit["duration_sec"],
                               "provider": hit.get("provider", "mock"), "cached": True})
            self._persist(job)
            return job_id
        threading.Thread(target=self._run, args=(job, dict(req), fp), daemon=True).start()
        return job_id

    def status(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def cancel(self, job_id):
        """请求取消任务；返回最新任务状态（不存在返回 None）。"""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job["status"] in ("done", "error", "canceled"):
                return dict(job)
            job["cancel_requested"] = True
            self._persist(job)
        ev = self._cancel_events.get(job_id)
        if ev:
            ev.set()
        return self.status(job_id)

    def _finished(self, job_id):
        with self._lock:
            self._cancel_events.pop(job_id, None)

    # ---------------------------------------------------------------- 流水线
    def _run(self, job, req, fp=None):
        ev = self._cancel_events.get(job["id"])
        try:
            # 分段重唱：服务端已构建完整 spec，不走作曲
            if req.get("edit_of"):
                self._run_edit(job, req, fp, ev)
                return

            job.update(status="running", stage="composing", stage_note=STAGES_ZH["composing"])
            prebuilt = req.pop("fixed_spec", None)  # 整首重制：服务端已构建完整 spec（跳过作曲）
            if req.get("provider") in AUDIO_PROVIDERS:
                # 音频类引擎固定产出 95s（或全曲 285s）：作曲按此时长铺排歌词结构
                req["duration_sec"] = neural_resolve_length(req)
            if prebuilt:
                spec, info = prebuilt, {"provider": "prebuilt"}
            else:
                spec, info = self._compose(req)
            if spec is None:
                raise RuntimeError("作曲失败: %s" % (info or {}).get("reason", "未知"))
            if req.get("lyrics"):
                spec = schemas.apply_lyrics(spec, req["lyrics"])  # 进阶模式自定义歌词

            if req.get("provider") in AUDIO_PROVIDERS:
                source = (info or {}).get("provider", "mock")
                spec["composer_provider"] = "mock" if source.startswith("mock") else source
                if (info or {}).get("provider", "").startswith("mock"):
                    job.update(stage_note="离线作曲已完成，继续生成音频")
                self._run_audio(job, req, spec, info, fp, ev)
                return

            if req.get("lead_voice"):
                spec["lead_voice"] = req["lead_voice"]
            spec["provider"] = "local"
            source = (info or {}).get("provider", "mock")
            spec["composer_provider"] = "mock" if source.startswith("mock") else source

            # 渲染槽位：最多同时 2 首，其余排队（防止并发渲染挤爆内存）
            job.update(stage_note="等待渲染槽位…（最多同时渲染 %d 首）" % MAX_RENDER_SLOTS)
            with self._slots:
                self._check_cancel(ev)
                job.update(stage="arranging", stage_note=STAGES_ZH["arranging"], progress=0.05)
                song = realizer.realize(spec)
                # 默认行为：有歌词就唱出来（显式传 sing=false 可关闭）；
                # 说唱默认纯 beat——TTS 逐字演唱撑不起 rap flow，人声交给神经引擎
                sing = req.get("sing")
                if sing is None:
                    sing = spec.get("genre") != "rap"
                if sing:
                    if singer.available_voice():
                        spec["sing"] = True
                        song["sing_events"] = singer.plan_singing(song)
                song_id = self.store.new_song_id()
                self.store.prepare(song_id)
                job.update(song_id=song_id)

                job.update(stage="rendering", stage_note=STAGES_ZH["rendering"], progress=0.08)
                synth.render_all(song, self.store.wav_path(song_id),
                                 self.store.song_dir(song_id),
                                 progress_cb=lambda f: job.update(progress=0.08 + 0.82 * f))
                midi.write_midi_file(song, self.store.midi_path(song_id))
            meta = self.store.save(song_id, spec, song, fingerprint=fp, request=req)
            self._done(job, meta, info.get("provider", "mock"))
        except CancelledError:
            job.update(status="canceled", stage="canceled",
                       stage_note=STAGES_ZH["canceled"], progress=0.0)
            self._persist(job)
        except Exception as e:  # 任何阶段失败都要反映到任务状态
            job.update(status="error", stage="error", stage_note=STAGES_ZH["error"],
                       error=str(e))
            self._persist(job)
        finally:
            self._finished(job["id"])
            self._cleanup_incomplete(job)

    def _cleanup_incomplete(self, job):
        """失败/取消时清掉半成品目录（还没有 meta.json 的 song 目录）。"""
        sid = job.get("song_id")
        if sid and not self.store.exists(sid):
            shutil.rmtree(self.store.song_dir(sid), ignore_errors=True)

    def _audio_provider(self, name):
        if name == "mureka":
            return mureka_provider.provider
        return diffrhythm_provider

    def _resolve_persona(self, req):
        """persona_id -> 嵌入 .npy 绝对路径（注入 req.style_prompt_npy）。"""
        if req.get("persona_id"):
            path = personas.embedding_path(self.store, req["persona_id"])
            if not path:
                raise RuntimeError("音色 Persona 不存在或已删除")
            req["style_prompt_npy"] = path

    def _run_audio(self, job, req, spec, info, fp, ev):
        """音频成品 Provider 旁路：作曲结果(歌词/结构) -> 模型/云端演唱。"""
        spec["provider"] = req.get("provider")
        source = (info or {}).get("provider", "mock")
        spec["composer_provider"] = "mock" if source.startswith("mock") else source
        spec["audio_only"] = True
        spec.pop("lead_voice", None)  # 音色由模型决定
        self._resolve_persona(req)
        provider = self._audio_provider(req.get("provider"))
        ok, reason = provider.available()
        if not ok:
            raise RuntimeError("引擎不可用: %s" % reason)
        job.update(stage_note="等待渲染槽位…（同时只跑 %d 首）" % MAX_NEURAL_SLOTS)
        with self._neural_slot:
            self._check_cancel(ev)
            job.update(stage="singing_neural", stage_note=STAGES_ZH["singing_neural"],
                       progress=0.04)
            song_id = self.store.new_song_id()
            self.store.prepare(song_id)
            job.update(song_id=song_id)
            if req.get("ref_audio"):
                req["ref_audio_path"] = self._resolve_ref_audio(req["ref_audio"])
            provider.generate(
                req, spec, self.store.wav_path(song_id),
                progress_cb=lambda f, note="": job.update(
                    progress=0.04 + 0.86 * f,
                    stage_note=note or STAGES_ZH["singing_neural"]),
                cancel_event=ev)
            self._check_cancel(ev)
            score = self._postprocess_audio(job, req, spec, song_id, ev)
        meta = self.store.save(song_id, spec, score, fingerprint=fp, request=req)
        self._done(job, meta, req.get("provider", "diffrhythm"))

    def _run_edit(self, job, req, fp, ev):
        """分段重唱：原曲为参考 latent，指定区间按新歌词重采样（其余原样保留）。"""
        spec = req.get("edit_spec")
        if not isinstance(spec, dict) or not spec.get("structure"):
            raise RuntimeError("重唱任务缺少原曲信息")
        spec["provider"] = "diffrhythm"
        spec["audio_only"] = True
        self._resolve_persona(req)
        job.update(stage_note="等待神经渲染槽位…（同时只跑 %d 首）" % MAX_NEURAL_SLOTS)
        with self._neural_slot:
            self._check_cancel(ev)
            job.update(stage="singing_neural", stage_note=STAGES_ZH["singing_neural"],
                       progress=0.04)
            song_id = self.store.new_song_id()
            self.store.prepare(song_id)
            job.update(song_id=song_id)
            diffrhythm_provider.generate(
                req, spec, self.store.wav_path(song_id),
                progress_cb=lambda f, note="": job.update(
                    progress=0.04 + 0.86 * f,
                    stage_note=note or STAGES_ZH["singing_neural"]),
                cancel_event=ev)
            self._check_cancel(ev)
            score = self._postprocess_audio(job, req, spec, song_id, ev)
        meta = self.store.save(song_id, spec, score, fingerprint=fp, request=req)
        self._done(job, meta, "diffrhythm-edit")

    def _postprocess_audio(self, job, req, spec, song_id, ev):
        """音频作品统一后处理：Whisper 对齐 + Demucs 分轨 + lite 曲谱。"""
        wav = self.store.wav_path(song_id)
        duration = _wav_duration(wav)
        lines = neural_lyric_lines(spec, max_sec=neural_resolve_length(req) - 2)
        score = None
        # ① Whisper 强制对齐：每句歌词的真实起止时间（失败退回小节时间轴）
        if neural_has_module("faster_whisper", wait=True) is True:
            job.update(stage_note="🎧 对齐歌词时间轴（Whisper）…", progress=0.92)
            events = neural_align(
                wav, lines,
                progress_cb=lambda f, note="": job.update(progress=0.92 + 0.03 * f))
            score = neural_lite_score(spec, duration, lyric_events=events)
        # ② Demucs 源分离：人声/鼓/贝斯/伴奏 4 分轨（失败不影响成品）
        if neural_has_module("demucs", wait=True) is True:
            job.update(stage_note="🎚 分离分轨（Demucs）…", progress=0.96)
            try:
                neural_separate(
                    wav, self.store.song_dir(song_id),
                    progress_cb=lambda f, note="": job.update(progress=0.96 + 0.03 * f))
            except Exception as e:  # noqa: BLE001
                job.update(stage_note="分轨失败（不影响成品）: %s" % str(e)[:80])
        return score or neural_lite_score(spec, duration)

    def _resolve_ref_audio(self, name):
        """参考音频名（服务端生成存于 output/_refs/）-> 绝对路径。"""
        safe = "".join(c for c in str(name) if c.isalnum() or c in "-_.")
        path = self.store.ref_path(safe)
        if not safe or not os.path.isfile(path):
            raise RuntimeError("参考音频不存在或已失效，请重新上传")
        return path

    @staticmethod
    def _check_cancel(ev):
        if ev is not None and ev.is_set():
            raise CancelledError("已取消")

    def _done(self, job, meta, provider_note):
        job.update(status="done", stage="done", stage_note=STAGES_ZH["done"],
                   progress=1.0,
                   result={"song_id": meta["id"], "title": meta["title"],
                           "duration_sec": meta["duration_sec"],
                           "provider": meta["provider"],
                           "provider_note": provider_note})
        self._persist(job)

    def _compose(self, req):
        """按 provider 选择：auto = LLM(启用时) 优先、失败回落 mock；
        显式选 llm 但没配置 / 失败则直接报错（不静默降级）。
        音频类引擎（diffrhythm/mureka）复用 auto 逻辑做作曲，渲染阶段另行旁路。"""
        mode = req.get("provider") or "local"
        explicit_composer = req.get("composer_provider")
        cfg = self._get_config()
        llm_cfg = cfg.get("llm", {})
        if explicit_composer == "mock":
            spec = mock_composer.compose(req)
            spec["provider"] = "mock"
            spec["composer_provider"] = "mock"
            return spec, {"provider": "mock"}
        if explicit_composer == "llm":
            if not llm_cfg.get("enabled"):
                raise RuntimeError("LLM 未启用：请在右上角「引擎设置」里配置 API Key，或选择离线作曲")
            spec, info = llm_provider.provide(req, cfg)
            if spec is None:
                raise RuntimeError("LLM 作曲失败: %s" % info.get("reason", "未知"))
            spec["composer_provider"] = "llm"
            return spec, info
        if mode == "local":
            spec = mock_composer.compose(req)
            spec["provider"] = "mock"
            spec["composer_provider"] = "mock"
            return spec, {"provider": "mock"}
        if mode == "llm":
            if not llm_cfg.get("enabled"):
                raise RuntimeError("LLM 未启用：请在右上角「引擎设置」里配置 API Key，"
                                   "或把引擎切换为 自动 / 离线 Mock")
            spec, info = llm_provider.provide(req, cfg)
            if spec is None:
                raise RuntimeError("LLM 作曲失败: %s" % info.get("reason", "未知"))
            return spec, info
        info = {}
        if llm_cfg.get("enabled"):
            if mode in AUDIO_PROVIDERS:
                # 音频流程有整体时间预算：LLM 挂起（慢滴响应绕过 socket 超时）
                # 会拖死整个任务 —— 3 分钟硬预算，超时回落 Mock
                spec, info = self._llm_with_budget(req, cfg, budget_sec=180)
            else:
                spec, info = llm_provider.provide(req, cfg)
            if spec is not None:
                return spec, info
            info = {"provider": "mock(回落)", "reason": info.get("reason", "LLM 失败")}
        spec = mock_composer.compose(req)
        spec["provider"] = "mock"
        return spec, info

    def _llm_with_budget(self, req, cfg, budget_sec=180):
        """带硬预算的 LLM 作曲：后台线程执行，超时放弃（线程自行消亡）回落。"""
        box = {}

        def worker():
            try:
                box["result"] = llm_provider.provide(req, cfg)
            except Exception as e:  # noqa: BLE001
                box["result"] = (None, {"reason": str(e)})
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        t.join(budget_sec)
        if "result" not in box:
            return None, {"reason": "LLM 作曲超过 %d 秒未返回" % budget_sec}
        return box["result"]
