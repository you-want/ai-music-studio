"""端到端 API 测试：起真实服务器线程，走完 生成→轮询→曲库→下载→删除。"""
import json
import os
import shutil
import struct
import sys
import tempfile
import threading
import time
import unittest
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app import server as server_mod  # noqa: E402


def _request(url, method="GET", body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, resp.read(), dict(resp.headers)


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="aimusic-test-")
        cls.httpd = server_mod.create_server(
            "127.0.0.1", 0,
            lib_dir=os.path.join(cls.tmp, "lib"),
            config_path=os.path.join(cls.tmp, "config.local.json"),
        )
        cls.base = "http://127.0.0.1:%d" % cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_01_health_and_meta(self):
        st, body, _ = _request(self.base + "/api/health")
        self.assertEqual(st, 200)
        self.assertTrue(json.loads(body)["ok"])
        st, body, _ = _request(self.base + "/api/meta")
        meta = json.loads(body)
        self.assertTrue(len(meta["genres"]) >= 8)
        self.assertTrue(len(meta["inspirations"]) >= 10)

    def test_02_static_frontend(self):
        st, body, headers = _request(self.base + "/")
        self.assertEqual(st, 200)
        self.assertIn("织歌", body.decode("utf-8"))
        st, body, _ = _request(self.base + "/app.js")
        self.assertEqual(st, 200)

    def test_03_compose_flow(self):
        st, body, _ = _request(self.base + "/api/compose", "POST",
                               {"description": "欢快的夏天流行歌", "duration_sec": 12})
        self.assertEqual(st, 200)
        job_id = json.loads(body)["job_id"]

        deadline = time.time() + 180
        job = None
        while time.time() < deadline:
            _, jbody, _ = _request(self.base + "/api/jobs/" + job_id)
            job = json.loads(jbody)
            if job["status"] in ("done", "error"):
                break
            time.sleep(0.4)
        self.assertEqual(job["status"], "done", "任务应成功: %s" % job)
        song_id = job["result"]["song_id"]

        # 曲库列表
        _, body, _ = _request(self.base + "/api/songs")
        songs = json.loads(body)["songs"]
        self.assertEqual(len(songs), 1)
        self.assertEqual(songs[0]["id"], song_id)

        # 曲谱
        _, body, _ = _request(self.base + "/api/songs/" + song_id)
        full = json.loads(body)
        self.assertIn("score", full)
        self.assertGreater(len(full["score"]["melody"]), 0)

        # WAV
        st, wav, headers = _request(self.base + "/api/songs/%s/audio.wav" % song_id)
        self.assertEqual(st, 200)
        self.assertEqual(wav[:4], b"RIFF")
        # Range 请求
        st, part, headers = _request(self.base + "/api/songs/%s/audio.wav" % song_id,
                                     headers={"Range": "bytes=0-1023"})
        self.assertEqual(st, 206)
        self.assertEqual(len(part), 1024)
        self.assertTrue(headers.get("Content-Range", "").startswith("bytes 0-1023/"))

        # MIDI
        st, mid, _ = _request(self.base + "/api/songs/%s/song.mid" % song_id)
        self.assertEqual(mid[:4], b"MThd")
        self.assertEqual(struct.unpack(">H", mid[10:12])[0], 5)

        # 删除
        st, body, _ = _request(self.base + "/api/songs/" + song_id, "DELETE")
        self.assertTrue(json.loads(body)["deleted"])
        _, body, _ = _request(self.base + "/api/songs")
        self.assertEqual(len(json.loads(body)["songs"]), 0)

    def test_04_config_roundtrip(self):
        st, body, _ = _request(self.base + "/api/config", "POST",
                               {"llm": {"enabled": True, "base_url": "https://x/v4",
                                        "model": "glm-4-flash", "api_key": "abcd1234efgh"}})
        cfg = json.loads(body)
        self.assertEqual(cfg["llm"]["model"], "glm-4-flash")
        self.assertEqual(cfg["llm"]["api_key_masked"], "abcd****gh")
        # 不带 key 再保存，key 应保留
        st, body, _ = _request(self.base + "/api/config", "POST",
                               {"llm": {"model": "glm-4-flash"}})
        cfg = json.loads(body)
        self.assertTrue(cfg["llm"]["api_key_set"])
        self.assertEqual(cfg["llm"]["base_url"], "https://x/v4")


if __name__ == "__main__":
    unittest.main()
