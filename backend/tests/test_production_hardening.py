"""生产加固回归测试：请求字段限长、派生会话凭证、磁盘残留 GC。

对应生产化审计修复（P0-3 会话 Cookie 派生 / P1-1 字段限长 / P1-2~4 泄漏清理）。
"""
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

_BACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACK not in sys.path:
    sys.path.insert(0, _BACK)

from app import schemas  # noqa: E402
from app import server as server_mod  # noqa: E402
from app.jobs import JobManager, JOB_SNAPSHOT_RETENTION_SEC, REF_AUDIO_RETENTION_SEC  # noqa: E402
from app.store import Library  # noqa: E402


class TestSecurityHeaders(unittest.TestCase):
    """安全响应头：API 带 X-Frame-Options/Referrer-Policy，HTML 加 CSP，404 不回显路径。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="aimusic-sec3-")
        cls.httpd = server_mod.create_server(
            "127.0.0.1", 0,
            lib_dir=os.path.join(cls.tmp, "lib"),
            config_path=os.path.join(cls.tmp, "config.local.json"),
        )
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _get(self, path):
        req = urllib.request.Request("http://127.0.0.1:%d%s" % (self.port, path))
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()

    def test_api_security_headers(self):
        st, h, _ = self._get("/api/health")
        self.assertEqual(st, 200)
        self.assertEqual(h.get("X-Frame-Options"), "DENY")
        self.assertEqual(h.get("Referrer-Policy"), "no-referrer")

    def test_html_page_has_csp(self):
        st, h, _ = self._get("/")
        self.assertEqual(st, 200)
        csp = h.get("Content-Security-Policy", "")
        self.assertIn("default-src 'self'", csp)
        self.assertIn("script-src 'self'", csp)

    def test_static_404_does_not_echo_path(self):
        st, _, body = self._get("/no-such-file-xyz.js")
        self.assertEqual(st, 404)
        self.assertNotIn(b"no-such-file-xyz.js", body)


class TestRequestFieldLimits(unittest.TestCase):
    def test_free_text_fields_are_clipped(self):
        req = schemas.parse_request({
            "description": "描" * 5000,
            "title": "题" * 500,
            "lyrics_req": "求" * 1000,
        })
        self.assertEqual(len(req["description"]), 500)
        self.assertEqual(len(req["title"]), 60)
        self.assertEqual(len(req["lyrics_req"]), 200)

    def test_lyrics_keeps_line_structure_and_is_clipped(self):
        long_line = "副歌一句" * 30          # 120 字
        lyrics = "[Verse]\n第一句\n\n[Chorus]\n" + "\n".join([long_line] * 80)
        req = schemas.parse_request({"lyrics": lyrics})
        self.assertLessEqual(len(req["lyrics"]), 5000)
        self.assertIn("[Verse]", req["lyrics"])
        self.assertIn("[Chorus]", req["lyrics"])   # 段落标记靠换行识别，必须保留
        self.assertIn("\n\n", req["lyrics"])

    def test_empty_text_fields_become_none(self):
        req = schemas.parse_request({"description": "  \n ", "lyrics": "", "title": ""})
        self.assertIsNone(req["description"])
        self.assertIsNone(req["lyrics"])
        self.assertIsNone(req["title"])

    def test_validate_repair_clips_llm_description_and_lines(self):
        raw = {
            "title": "测试",
            "structure": [{"name": "verse", "bars": 4,
                           "chords": ["C"], "lyrics": ["超" * 500]}],
        }
        spec, _ = schemas.validate_and_repair_spec(raw, {"duration_sec": 30})
        self.assertLessEqual(len(spec["description"]), 500)
        for sec in spec["structure"]:
            for line in sec["lyrics"]:
                self.assertLessEqual(len(line), 200)


class TestDerivedSessionValue(unittest.TestCase):
    TOKEN = "unit-test-token"

    def test_roundtrip_and_rejections(self):
        h = server_mod.Handler
        exp = int(time.time()) + 100
        value = h._session_value(self.TOKEN, exp)
        self.assertNotIn(self.TOKEN, value)          # 凭证不含原始 Token
        self.assertTrue(h._validate_session(self.TOKEN, value))
        # 换 Token 立即吊销（旧凭证失效）
        self.assertFalse(h._validate_session("rotated", value))
        # 篡改 MAC / 过期 / 乱格式均拒绝
        bad_mac = ("0" * 64) + (".%d" % exp)
        self.assertFalse(h._validate_session(self.TOKEN, bad_mac))
        expired = h._session_value(self.TOKEN, int(time.time()) - 10)
        self.assertFalse(h._validate_session(self.TOKEN, expired))
        self.assertFalse(h._validate_session(self.TOKEN, "not-a-session"))
        self.assertFalse(h._validate_session(self.TOKEN, ""))

    def test_far_future_expiry_rejected(self):
        h = server_mod.Handler
        far = int(time.time()) + h.SESSION_TTL_SEC * 10  # 伪造超长有效期
        value = h._session_value(self.TOKEN, far)
        self.assertFalse(h._validate_session(self.TOKEN, value))


class TestDiskGc(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="aimusic-gc-")
        self.lib = Library(os.path.join(self.tmp, "lib"))
        self.jm = JobManager(store=self.lib, config_getter=lambda: {"llm": {}})
        self.old_ts = time.time() - JOB_SNAPSHOT_RETENTION_SEC - 3600

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def _age_file(path, ts):
        os.utime(path, (ts, ts))

    def test_old_terminal_job_snapshot_removed(self):
        d = self.jm._job_dir
        stale = os.path.join(d, "oldjob.json")
        with open(stale, "w") as f:
            f.write("{}")
        self._age_file(stale, self.old_ts)
        self.jm._gc_disk()
        self.assertFalse(os.path.exists(stale))

    def test_active_job_snapshot_kept_even_if_old(self):
        d = self.jm._job_dir
        self.jm._jobs["livejob"] = {"id": "livejob", "status": "running",
                                    "created_at": time.time()}
        stale = os.path.join(d, "livejob.json")
        with open(stale, "w") as f:
            f.write("{}")
        self._age_file(stale, self.old_ts)
        self.jm._gc_disk()
        self.assertTrue(os.path.exists(stale))  # 进行中任务的快照不清理

    def test_old_ref_audio_removed_protected_kept(self):
        refs = self.lib.refs_dir()
        old_ref = os.path.join(refs, "ref-old.wav")
        protected = os.path.join(refs, "ref-used.wav")
        for p in (old_ref, protected):
            with open(p, "wb") as f:
                f.write(b"RIFF")
            self._age_file(p, time.time() - REF_AUDIO_RETENTION_SEC - 3600)
        self.jm._job_refs["job-1"] = "ref-used.wav"  # 模拟进行中任务引用
        self.jm._gc_disk()
        self.assertFalse(os.path.exists(old_ref))
        self.assertTrue(os.path.exists(protected))


class TestSubmitDedup(unittest.TestCase):
    """同参数并发提交去重：第二次提交应复用进行中的任务，不重复渲染/计费。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="aimusic-dedup-")
        self.lib = Library(os.path.join(self.tmp, "lib"))
        self.jm = JobManager(store=self.lib, config_getter=lambda: {"llm": {}})

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_concurrent_same_params_share_one_job(self):
        from app import schemas
        req = schemas.parse_request(
            {"description": "去重测试小曲", "duration_sec": 15, "seed": 4242})
        id1 = self.jm.submit(dict(req))
        id2 = self.jm.submit(dict(req))  # 任务 1 仍在渲染中（毫秒级内提交）
        self.assertEqual(id1, id2)
        self.assertEqual(len([j for j in self.jm._jobs.values()
                              if j["status"] in ("queued", "running")]), 1)
        # 等任务收尾，避免 teardown 与后台线程竞争临时目录
        # （预算放宽到 ~180s：慢机器/CI 高负载下 mock 渲染也可能很慢）
        for _ in range(900):
            st = self.jm.status(id1)
            if st["status"] in ("done", "error", "canceled"):
                break
            time.sleep(0.2)
        self.assertEqual(st["status"], "done", st.get("error"))
        # 任务结束后同参数再提交 → 命中结果缓存（也只此一份）
        id3 = self.jm.submit(dict(req))
        st3 = self.jm.status(id3)
        self.assertTrue(st3["result"]["cached"])


class TestLocalRenderCancel(unittest.TestCase):
    """本地合成渲染进行中的取消：progress_cb 检查后应立即中止并清掉半成品。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="aimusic-cancel-")
        self.lib = Library(os.path.join(self.tmp, "lib"))
        self.jm = JobManager(store=self.lib, config_getter=lambda: {"llm": {}})

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cancel_during_render(self):
        import app.jobs as jobs_mod
        from app import schemas

        def fake_render(song, wav_path, stems_dir, progress_cb=None):
            for i in range(100):  # 模拟 10 秒渲染，逐组上报进度
                if progress_cb:
                    progress_cb(i / 100.0)
                time.sleep(0.1)
            return wav_path, []

        orig = (jobs_mod.realizer.realize, jobs_mod.synth.render_all,
                jobs_mod.singer.available_voice)
        jobs_mod.realizer.realize = lambda spec: {"genre": spec.get("genre")}
        jobs_mod.synth.render_all = fake_render
        jobs_mod.singer.available_voice = lambda: None
        try:
            req = schemas.parse_request(
                {"description": "取消测试", "duration_sec": 15, "seed": 7})
            jid = self.jm.submit(dict(req))
            song_id = None
            for _ in range(600):
                st = self.jm.status(jid)
                song_id = st.get("song_id") or song_id
                if st["stage"] == "rendering":
                    break
                time.sleep(0.1)
            self.assertEqual(st["stage"], "rendering")
            self.assertTrue(self.lib.exists(song_id) or
                            os.path.isdir(self.lib.song_dir(song_id)))
            self.jm.cancel(jid)
            for _ in range(600):
                st = self.jm.status(jid)
                if st["status"] in ("canceled", "done", "error"):
                    break
                time.sleep(0.2)
            self.assertEqual(st["status"], "canceled", st)
            self.assertFalse(os.path.isdir(self.lib.song_dir(song_id)),
                             "取消后半成品目录应被清理")
        finally:
            (jobs_mod.realizer.realize, jobs_mod.synth.render_all,
             jobs_mod.singer.available_voice) = orig


class TestFingerprintIndex(unittest.TestCase):
    """曲库指纹内存索引：O(1) 查询、写时维护、外部删除后惰性重建。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="aimusic-fp-")
        self.lib = Library(os.path.join(self.tmp, "lib"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def _spec(title):
        return {"title": title, "description": "", "genre": "pop", "mood": "happy",
                "key": "C", "mode": "major", "bpm": 100,
                "structure": [{"name": "verse", "bars": 4, "chords": ["C"], "lyrics": []}]}

    def _save(self, title, fp):
        sid = self.lib.new_song_id()
        self.lib.prepare(sid)
        self.lib.save(sid, self._spec(title), {"duration_sec": 10.0}, fingerprint=fp)
        return sid

    def test_save_find_delete_maintain_index(self):
        sid1 = self._save("歌一", "fp-1")
        sid2 = self._save("歌二", "fp-2")
        self.assertEqual(self.lib.find_fingerprint("fp-1")["id"], sid1)
        self.assertEqual(self.lib.find_fingerprint("fp-2")["id"], sid2)
        self.assertIsNone(self.lib.find_fingerprint("fp-none"))
        self.lib.delete(sid1)
        self.assertIsNone(self.lib.find_fingerprint("fp-1"))
        self.assertEqual(self.lib.find_fingerprint("fp-2")["id"], sid2)

    def test_stale_index_rebuilt_lazily(self):
        sid = self._save("歌", "fp-x")
        self.assertEqual(self.lib.find_fingerprint("fp-x")["id"], sid)
        # 绕过 Library.delete 模拟外部删除，并把索引指向已消失的作品
        shutil.rmtree(self.lib.song_dir(sid))
        self.lib._fp_index["fp-x"] = sid
        self.assertIsNone(self.lib.find_fingerprint("fp-x"))
        # 重建后的索引与磁盘一致：重新保存可再命中
        sid2 = self._save("歌二", "fp-y")
        self.assertEqual(self.lib.find_fingerprint("fp-y")["id"], sid2)


if __name__ == "__main__":
    unittest.main()

