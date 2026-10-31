"""神经歌声引擎（DiffRhythm Provider）测试：单元 + 桩子进程端到端。"""
import json
import os
import re
import shutil
import sys
import tempfile
import textwrap
import threading
import time
import unittest
import urllib.request
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app import server as server_mod  # noqa: E402
from app.jobs import JobManager  # noqa: E402
from app.music import mock_composer  # noqa: E402
from app.providers import diffrhythm as dr  # noqa: E402
from app.providers.audio_base import AudioProvider, CancelledError  # noqa: E402
from app.providers.diffrhythm import DiffRhythmProvider  # noqa: E402


def _make_wav(path, sec=1.0, sr=44100):
    t = np.linspace(0, sec, int(sr * sec), endpoint=False)
    data = (np.sin(2 * np.pi * 440 * t) * 20000).astype("<i2")
    stereo = np.column_stack([data, data])
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(stereo.tobytes())


class TestLrcAndStyle(unittest.TestCase):
    def _spec(self, bpm=120, bars=(2, 4)):
        spec = mock_composer.compose({"description": "轻快流行", "duration_sec": 15, "seed": 3})
        spec["bpm"] = bpm
        return spec

    def test_lyric_lines_bar_timing(self):
        spec = self._spec(bpm=120)  # 每小节 2 秒
        lines = dr.build_lyric_lines(spec)
        self.assertTrue(lines, "有歌词的 spec 应产出 LRC 行")
        for t, text in lines:
            self.assertIsInstance(t, float)
            self.assertTrue(text.strip())
            self.assertLess(t, 93.0)
        # 时间戳应随小节递增
        times = [t for t, _ in lines]
        self.assertEqual(times, sorted(times))

    def test_lyric_lines_cap_at_93s(self):
        spec = self._spec(bpm=60)  # 每小节 4 秒，结构拉长超过 93s
        for sec in spec["structure"]:
            sec["lyrics"] = ["歌词%d" % i for i in range(sec["bars"])]
        spec["structure"] *= 6  # 大量重复段落
        lines = dr.build_lyric_lines(spec)
        self.assertTrue(all(t < 93.0 for t, _ in lines))

    def test_style_prompt_contains_genre_mood(self):
        spec = {"genre": "guofeng", "mood": "epic", "description": "仗剑天涯"}
        p = DiffRhythmProvider.build_style_prompt(spec)
        self.assertIn("chinese traditional", p)
        self.assertIn("epic", p)
        self.assertIn("仗剑天涯", p)

    def test_lite_score_shape(self):
        spec = mock_composer.compose({"description": "欢快流行", "duration_sec": 20, "seed": 5})
        spec["audio_only"] = True
        score = dr.lite_score(spec, 95.0)
        self.assertEqual(score["duration_sec"], 95.0)
        self.assertTrue(score["audio_only"])
        self.assertEqual(score["melody"], [])
        self.assertEqual(score["total_beats"],
                         sum(s["bars"] for s in score["sections"]) * score["beats_per_bar"])
        self.assertTrue(score["sections"])
        self.assertIn("lyrics", score["sections"][0])


class TestStubSubprocess(unittest.TestCase):
    """用桩 venv python 走通 子进程 -> 进度解析 -> WAV 落盘 全链路。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="aimusic-dr-")
        self.wav_out = os.path.join(self.tmp, "out.wav")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_stub_python(self, delay=0.0):
        stub = os.path.join(self.tmp, "stub_python.py")
        with open(stub, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent("""
                import json, sys, time, wave
                import numpy as np
                delay = %f
                job = json.load(open(sys.argv[2], encoding="utf-8"))
                for i in range(1, 4):
                    print(json.dumps({"type": "progress", "frac": i / 4.0,
                                      "note": "stub step %%d" %% i}), flush=True)
                    if delay:
                        time.sleep(delay)
                sr, sec = 44100, 0.5
                t = np.linspace(0, sec, int(sr * sec), endpoint=False)
                data = (np.sin(2 * np.pi * 440 * t) * 20000).astype("<i2")
                stereo = np.column_stack([data, data])
                with wave.open(job["out_wav"], "wb") as w:
                    w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
                    w.writeframes(stereo.tobytes())
                print(json.dumps({"type": "done", "wav": job["out_wav"],
                                  "duration_sec": sec}), flush=True)
            """ % delay))
        launcher = os.path.join(self.tmp, "stub-python")
        with open(launcher, "w", encoding="utf-8") as f:
            f.write("#!/bin/bash\nexec /usr/bin/env python3 %s \"$@\"\n" % stub)
        os.chmod(launcher, 0o755)
        return launcher

    def _provider(self, stub_py):
        p = DiffRhythmProvider()
        p._forced_python = stub_py
        orig = dr.load_neural_config

        def patched(config_path=None):
            cfg = orig(config_path)
            cfg["python"] = stub_py
            cfg["timeout_min"] = 1
            return cfg
        dr.load_neural_config = patched
        self.addCleanup(setattr, dr, "load_neural_config", orig)
        return p

    def test_generate_via_subprocess(self):
        stub = self._make_stub_python()
        p = self._provider(stub)
        spec = mock_composer.compose({"description": "欢快流行", "duration_sec": 20, "seed": 9})
        fracs = []
        p.generate({"description": "欢快流行"}, spec, self.wav_out,
                   progress_cb=lambda f, note="": fracs.append(f))
        self.assertTrue(os.path.isfile(self.wav_out))
        with wave.open(self.wav_out, "rb") as w:
            self.assertEqual(w.getframerate(), 44100)
            self.assertEqual(w.getnchannels(), 2)
        self.assertGreaterEqual(len(fracs), 3)
        self.assertTrue(all(0.06 <= f <= 0.95 for f in fracs))  # 映射到任务区间

    def test_generate_error_propagates(self):
        stub_py = os.path.join(self.tmp, "bad-python")
        with open(stub_py, "w", encoding="utf-8") as f:
            f.write('#!/bin/bash\necho \'{"type":"error","error":"boom"}\'\nexit 1\n')
        os.chmod(stub_py, 0o755)
        p = self._provider(stub_py)
        spec = mock_composer.compose({"description": "欢快流行", "duration_sec": 15, "seed": 9})
        from app.providers.audio_base import AudioProviderError
        with self.assertRaises(AudioProviderError):
            p.generate({"description": "x"}, spec, self.wav_out)

    def test_cancel_aborts(self):
        stub = self._make_stub_python(delay=10)
        p = self._provider(stub)
        spec = mock_composer.compose({"description": "欢快流行", "duration_sec": 15, "seed": 9})
        import threading
        ev = threading.Event()
        threading.Timer(0.5, ev.set).start()
        t0 = time.time()
        with self.assertRaises(CancelledError):
            p.generate({"description": "x"}, spec, self.wav_out, cancel_event=ev)
        self.assertLess(time.time() - t0, 8)  # 未等 10s 的 sleep 走完


class TestNeuralJobEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="aimusic-neural-")
        cls.httpd = server_mod.create_server(
            "127.0.0.1", 0,
            lib_dir=os.path.join(cls.tmp, "lib"),
            config_path=os.path.join(cls.tmp, "config.local.json"),
        )
        cls.base = "http://127.0.0.1:%d" % cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

        # 桩 provider：不依赖真实 venv/权重
        class StubProvider(AudioProvider):
            name = "diffrhythm"

            def available(self):
                return True, "stub"

            def generate(self, req, spec, out_wav, progress_cb=None, cancel_event=None):
                sr, sec = 44100, 0.3
                t = np.linspace(0, sec, int(sr * sec), endpoint=False)
                data = (np.sin(2 * np.pi * 523 * t) * 20000).astype("<i2")
                stereo = np.column_stack([data, data])
                with wave.open(out_wav, "wb") as w:
                    w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
                    w.writeframes(stereo.tobytes())
                if progress_cb:
                    progress_cb(1.0, "stub done")

        import app.jobs as jobs_mod
        cls._orig = jobs_mod.diffrhythm_provider
        cls._orig_has = jobs_mod.neural_has_module
        jobs_mod.diffrhythm_provider = StubProvider()
        jobs_mod.neural_has_module = lambda module, config_path=None, wait=False: False  # 跳过对齐/分轨

    @classmethod
    def tearDownClass(cls):
        import app.jobs as jobs_mod
        jobs_mod.diffrhythm_provider = cls._orig
        jobs_mod.neural_has_module = cls._orig_has
        cls.httpd.shutdown()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _req(self, path, method="GET", body=None, timeout=120):
        data = json.dumps(body).encode() if isinstance(body, dict) else body
        r = urllib.request.Request(self.base + path, data=data, method=method,
                                   headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_01_meta_reports_engine(self):
        _, b = self._req("/api/meta")
        meta = json.loads(b)
        self.assertIn("engines", meta)
        self.assertIn("diffrhythm", meta["engines"])
        self.assertIn("available", meta["engines"]["diffrhythm"])

    def test_02_compose_neural_end_to_end(self):
        _, b = self._req("/api/compose", "POST", {
            "description": "夜晚路灯下的告别", "provider": "diffrhythm",
            "duration_sec": 30, "seed": 11,
        })
        job_id = json.loads(b)["job_id"]
        job = None
        for _ in range(300):
            _, b = self._req("/api/jobs/" + job_id)
            job = json.loads(b)
            if job["status"] in ("done", "error", "canceled"):
                break
            time.sleep(0.2)
        self.assertEqual(job["status"], "done", job.get("error"))
        sid = job["result"]["song_id"]
        self.assertEqual(job["result"]["provider"], "diffrhythm")

        _, b = self._req("/api/songs/" + sid)
        full = json.loads(b)
        self.assertEqual(full["meta"]["provider"], "diffrhythm")
        self.assertTrue(full["meta"]["audio_only"])
        self.assertTrue(full["score"]["audio_only"])
        self.assertEqual(full["score"]["melody"], [])
        self.assertTrue(full["score"]["sections"], "lite score 应有段落")
        self.assertGreater(full["meta"]["duration_sec"], 0)

        # WAV 可 Range 播放
        st, raw = self._req("/api/songs/%s/audio.wav" % sid)
        self.assertEqual(st, 200)
        self.assertEqual(raw[:4], b"RIFF")
        req = urllib.request.Request(self.base + "/api/songs/%s/audio.wav" % sid,
                                     headers={"Range": "bytes=0-99"})
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 206)
            self.assertEqual(len(resp.read()), 100)

        # MIDI 不应存在（神经作品无曲谱）
        st, _ = self._req("/api/songs/%s/song.mid" % sid)
        self.assertEqual(st, 404)

        # 分享包/封面照常可用
        st, raw = self._req("/api/songs/%s/bundle.zip" % sid)
        self.assertEqual(st, 200)

    def test_03_ref_audio_upload_and_compose(self):
        import io
        sr = 8000
        t = np.linspace(0, 1, sr, endpoint=False)
        data = (np.sin(2 * np.pi * 300 * t) * 20000).astype("<i2")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes(data.tobytes())
        st, b = self._req("/api/ref-audio", "POST", buf.getvalue())
        d = json.loads(b)
        self.assertEqual(st, 200)
        self.assertTrue(re.match(r"^ref-[\w]+\.wav$", d["ref_audio"]))

        # 非法格式被拒
        st, b = self._req("/api/ref-audio", "POST", b"not audio at all")
        self.assertEqual(st, 400)

        # 带参考音频提交（桩 provider 忽略，但验证字段透传）
        _, b = self._req("/api/compose", "POST", {
            "description": "参考风格测试", "provider": "diffrhythm",
            "duration_sec": 20, "seed": 12, "ref_audio": d["ref_audio"],
        })
        job_id = json.loads(b)["job_id"]
        for _ in range(300):
            _, b = self._req("/api/jobs/" + job_id)
            job = json.loads(b)
            if job["status"] in ("done", "error", "canceled"):
                break
            time.sleep(0.2)
        self.assertEqual(job["status"], "done", job.get("error"))

    def test_04_job_cancel(self):
        import app.jobs as jobs_mod
        from app.providers.audio_base import AudioProvider

        class BlockingProvider(AudioProvider):
            name = "diffrhythm"

            def available(self):
                return True, "stub"

            def generate(self, req, spec, out_wav, progress_cb=None, cancel_event=None):
                assert cancel_event is not None
                for _ in range(300):
                    if cancel_event.is_set():
                        raise CancelledError("已取消")
                    time.sleep(0.1)
                raise AudioProviderError("stub timeout")

        orig = jobs_mod.diffrhythm_provider
        jobs_mod.diffrhythm_provider = BlockingProvider()
        try:
            _, b = self._req("/api/compose", "POST", {
                "description": "取消测试", "provider": "diffrhythm",
                "duration_sec": 15, "seed": 13,
            })
            job_id = json.loads(b)["job_id"]
            time.sleep(0.5)
            st, b = self._req("/api/jobs/" + job_id, method="DELETE")
            d = json.loads(b)
            self.assertTrue(d["canceled"])
            job = None
            for _ in range(50):  # 取消是异步生效的，轮询到终态
                _, b = self._req("/api/jobs/" + job_id)
                job = json.loads(b)
                if job["status"] in ("canceled", "done", "error"):
                    break
                time.sleep(0.1)
            self.assertEqual(job["status"], "canceled")
        finally:
            jobs_mod.diffrhythm_provider = orig


class TestJobManagerUnits(unittest.TestCase):
    def test_cancel_unknown_job(self):
        jm = JobManager(store=None, config_getter=lambda: {"llm": {}})
        self.assertIsNone(jm.cancel("nope"))


class TestAligner(unittest.TestCase):
    """Whisper 对齐纯算法单测：伪造字符时间轴（含漏字/插字/时间抖动）。"""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "neural", "align_runner.py")
        spec = importlib.util.spec_from_file_location("align_runner", path)
        cls.al = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.al)

    def _timeline_from_truth(self, line_times, char_sec=0.28, drop=0.15, noise=0.08):
        """按"真实演唱"造字符时间轴：随机丢字（模型没唱清）、插噪声字（气声）。"""
        import random
        rng = random.Random(7)
        timeline = []
        for est, text in line_times:
            t = est + rng.uniform(-0.1, 0.1)
            for ch in text:
                if rng.random() < drop:
                    t += char_sec  # 漏唱
                    continue
                timeline.append([ch, t, t + char_sec])
                t += char_sec
                if rng.random() < noise:
                    timeline.append(["呣", t, t + char_sec * 0.5])
                    t += char_sec * 0.5
        return timeline

    def test_alignment_recovers_times(self):
        lines = [[5.0, "凌晨的街只剩几盏灯"], [12.0, "我推开便利店的门"],
                 [19.0, "深夜便利店亮着一盏灯"]]
        timeline = self._timeline_from_truth(lines)
        events = self.al.align_chars_to_lines(lines, timeline)
        self.assertEqual(len(events), 3)
        for (est, text), (start, end, got, conf) in zip(lines, events):
            self.assertEqual(got, text)
            self.assertLess(abs(start - est), 2.0, "对齐时间应接近真实演唱位置")
            self.assertGreater(end, start)
            self.assertGreater(conf, 0.4)
        # 单调性
        for a, b in zip(events, events[1:]):
            self.assertLessEqual(a[0], b[0])

    def test_empty_timeline_falls_back(self):
        lines = [[3.0, "第一句"], [8.0, "第二句"]]
        events = self.al.align_chars_to_lines(lines, [])
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0][3], 0.0)  # conf=0 → 回落 LRC 估计

    def test_low_confidence_uses_estimate(self):
        # 时间轴完全无关内容 → 置信度极低 → 回落估计值
        lines = [[3.0, "完全对不上的歌词"]]
        timeline = [["x", 50.0, 50.2], ["y", 50.2, 50.4]]
        events = self.al.align_chars_to_lines(lines, timeline)
        self.assertEqual(events[0][0], 3.0)  # 回落 LRC 估计

    def test_lite_score_lyric_events(self):
        spec = mock_composer.compose({"description": "欢快流行", "duration_sec": 20, "seed": 5})
        score = dr.lite_score(spec, 95.0, lyric_events=[[1.0, 4.0, "词", 0.9]])
        self.assertEqual(score["lyric_events"], [[1.0, 4.0, "词", 0.9]])
        score2 = dr.lite_score(spec, 95.0)
        self.assertEqual(score2["lyric_events"], [])


class TestFullLength(unittest.TestCase):
    def test_resolve_length_full_requires_weights(self):
        p = DiffRhythmProvider()
        orig = dr.FULL_CFM
        dr.FULL_CFM = "/nonexistent/cfm_model_full.pt"
        try:
            self.assertEqual(dr.resolve_audio_length({}), 95)
            with self.assertRaises(Exception):
                dr.resolve_audio_length({"full_song": True})
        finally:
            dr.FULL_CFM = orig

    def test_resolve_length_full_ok_when_weights_exist(self):
        p = DiffRhythmProvider()
        orig = dr.FULL_CFM
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            dr.FULL_CFM = f.name
            try:
                self.assertEqual(dr.resolve_audio_length({"full_song": True}), 285)
            finally:
                dr.FULL_CFM = orig


class TestDemucsStub(unittest.TestCase):
    """桩 demucs 子进程：验证分轨产出与命名映射。"""

    def test_separate_moves_stems(self):
        tmp = tempfile.mkdtemp(prefix="aimusic-demucs-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        stub = os.path.join(tmp, "stub_demucs.py")
        with open(stub, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent("""
                import json, os, sys, wave
                import numpy as np
                job = json.load(open(sys.argv[2], encoding="utf-8"))
                os.makedirs(job["out_dir"], exist_ok=True)
                t = np.linspace(0, 0.2, 8820, endpoint=False)
                tone = (np.sin(2*np.pi*440*t)*9000).astype("<i2")
                stems = {}
                for name, target in (("vocals","stem_vocal.wav"),("drums","stem_drums.wav"),
                                     ("bass","stem_bass.wav"),("other","stem_harmony.wav")):
                    p = os.path.join(job["out_dir"], target)
                    with wave.open(p, "wb") as w:
                        w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100)
                        w.writeframes(tone.tobytes())
                    stems[name] = target
                print(json.dumps({"type":"done","stems":stems}), flush=True)
            """))
        launcher = os.path.join(tmp, "stub-python")
        with open(launcher, "w", encoding="utf-8") as f:
            f.write("#!/bin/bash\nexec /usr/bin/env python3 %s \"$@\"\n" % stub)
        os.chmod(launcher, 0o755)

        orig_cfg = dr.load_neural_config
        orig_has = dr.venv_has_module
        dr.load_neural_config = lambda config_path=None: {
            "enabled": True, "steps": 16, "timeout_min": 5, "python": launcher,
            "device": None, "audio_length": 95, "whisper_model": "small",
            "align": True, "separate": True}
        dr.venv_has_module = lambda module, config_path=None, wait=False: True
        try:
            wav = os.path.join(tmp, "song.wav")
            _make_wav(wav, sec=0.2, sr=44100)
            out_dir = os.path.join(tmp, "songdir")
            os.makedirs(out_dir, exist_ok=True)
            stems = dr.separate(wav, out_dir)
            self.assertIn("vocals", stems)
            for f in ("stem_vocal.wav", "stem_drums.wav", "stem_bass.wav", "stem_harmony.wav"):
                self.assertTrue(os.path.isfile(os.path.join(out_dir, f)), f)
        finally:
            dr.load_neural_config = orig_cfg
            dr.venv_has_module = orig_has

    def test_align_disabled_returns_none(self):
        orig_has = dr.venv_has_module
        dr.venv_has_module = lambda module, config_path=None, wait=False: False
        try:
            self.assertIsNone(dr.align("/tmp/x.wav", [[1.0, "词"]]))
        finally:
            dr.venv_has_module = orig_has


if __name__ == "__main__":
    unittest.main()
