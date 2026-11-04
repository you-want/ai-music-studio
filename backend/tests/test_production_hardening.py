"""生产加固回归测试：请求字段限长、派生会话凭证、磁盘残留 GC。

对应生产化审计修复（P0-3 会话 Cookie 派生 / P1-1 字段限长 / P1-2~4 泄漏清理）。
"""
import os
import shutil
import sys
import tempfile
import time
import unittest

_BACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACK not in sys.path:
    sys.path.insert(0, _BACK)

from app import schemas  # noqa: E402
from app import server as server_mod  # noqa: E402
from app.jobs import JobManager, JOB_SNAPSHOT_RETENTION_SEC, REF_AUDIO_RETENTION_SEC  # noqa: E402
from app.store import Library  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
