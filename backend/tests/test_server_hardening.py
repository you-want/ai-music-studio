"""生产加固测试：Token 鉴权、Range 语义、请求体上限、路径穿越防护。"""
import http.client
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.request
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TOKEN = "test-token-12345"


def _make_wav(path, sec=1.0, sr=8000):
    t = np.linspace(0, sec, int(sr * sec), endpoint=False)
    data = (np.sin(2 * np.pi * 440 * t) * 20000).astype("<i2")
    stereo = np.column_stack([data, data])
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(stereo.tobytes())


class _ServerBase(unittest.TestCase):
    """共享：起服务 + 手工放一首最小作品进曲库。"""

    SONG_ID = "20260101-000000-deadbeef"

    @classmethod
    def seed_song(cls, lib_dir):
        d = os.path.join(lib_dir, cls.SONG_ID)
        os.makedirs(d, exist_ok=True)
        meta = {"id": cls.SONG_ID, "title": "加固测试曲", "provider": "mock",
                "duration_sec": 1.0, "bpm": 100, "genre": "pop", "mood": "happy",
                "created_at": 1, "favorite": False, "sections": []}
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        with open(os.path.join(d, "song.json"), "w", encoding="utf-8") as f:
            json.dump({"title": meta["title"], "structure": []}, f)
        _make_wav(os.path.join(d, "song.wav"))

    @classmethod
    def raw_request(cls, method, path, body=None, headers=None):
        """发原始路径（不做客户端 URL 规范化），返回 (status, headers, body)。"""
        conn = http.client.HTTPConnection("127.0.0.1", cls.port, timeout=30)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            resp = conn.getresponse()
            data = resp.read()
            return resp.status, dict(resp.getheaders()), data
        finally:
            conn.close()


class TestTokenAuth(_ServerBase):
    @classmethod
    def setUpClass(cls):
        from app import server as server_mod
        cls.tmp = tempfile.mkdtemp(prefix="aimusic-sec-")
        os.environ["AI_MUSIC_TOKEN"] = TOKEN
        try:
            cls.httpd = server_mod.create_server(
                "127.0.0.1", 0,
                lib_dir=os.path.join(cls.tmp, "lib"),
                config_path=os.path.join(cls.tmp, "config.local.json"),
            )
        finally:
            os.environ.pop("AI_MUSIC_TOKEN", None)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.seed_song(os.path.join(cls.tmp, "lib"))

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_health_is_open(self):
        st, _, body = self.raw_request("GET", "/api/health")
        self.assertEqual(st, 200)

    def test_api_requires_token(self):
        st, headers, _ = self.raw_request("GET", "/api/meta")
        self.assertEqual(st, 401)
        self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")

    def test_bearer_header_ok(self):
        st, _, body = self.raw_request(
            "GET", "/api/meta", headers={"Authorization": "Bearer " + TOKEN})
        self.assertEqual(st, 200)
        self.assertIn("genres", json.loads(body))

    def test_session_cookie_authenticates_media_and_api(self):
        st, headers, body = self.raw_request(
            "POST", "/api/session", headers={"Authorization": "Bearer " + TOKEN})
        self.assertEqual(st, 200)
        cookie = headers["Set-Cookie"].split(";", 1)[0]
        self.assertIn("HttpOnly", headers["Set-Cookie"])
        self.assertIn("SameSite=Strict", headers["Set-Cookie"])
        st, _, body = self.raw_request(
            "GET", "/api/songs/%s/audio.wav" % self.SONG_ID,
            headers={"Cookie": cookie})
        self.assertEqual(st, 200)
        self.assertEqual(body[:4], b"RIFF")
        st, _, _ = self.raw_request("POST", "/api/config",
                                     body=json.dumps({}).encode(),
                                     headers={"Cookie": cookie})
        self.assertEqual(st, 200)

    def test_query_token_is_rejected(self):
        st, _, _ = self.raw_request(
            "GET", "/api/songs/%s/audio.wav?token=%s" % (self.SONG_ID, TOKEN))
        self.assertEqual(st, 401)

    def test_wrong_token_401(self):
        st, _, _ = self.raw_request("GET", "/api/meta?token=wrong")
        self.assertEqual(st, 401)
        st, _, _ = self.raw_request(
            "GET", "/api/meta", headers={"Authorization": "Bearer wrong"})
        self.assertEqual(st, 401)

    def test_static_page_open_but_api_guarded(self):
        st, _, body = self.raw_request("GET", "/")
        self.assertEqual(st, 200)  # 静态页面本身不拦（前端 JS 会带 token 调 API）


class TestRangeAndLimits(_ServerBase):
    @classmethod
    def setUpClass(cls):
        from app import server as server_mod
        cls.tmp = tempfile.mkdtemp(prefix="aimusic-sec2-")
        cls.httpd = server_mod.create_server(
            "127.0.0.1", 0,
            lib_dir=os.path.join(cls.tmp, "lib"),
            config_path=os.path.join(cls.tmp, "config.local.json"),
        )
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.seed_song(os.path.join(cls.tmp, "lib"))

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _wav_size(self):
        return os.path.getsize(os.path.join(
            self.tmp, "lib", self.SONG_ID, "song.wav"))

    def test_range_suffix_semantics(self):
        """bytes=-N 应返回最后 N 字节（修复前被误解析为 0..N）。"""
        size = self._wav_size()
        st, headers, body = self.raw_request(
            "GET", "/api/songs/%s/audio.wav" % self.SONG_ID,
            headers={"Range": "bytes=-7"})
        self.assertEqual(st, 206)
        self.assertEqual(headers.get("Content-Range"),
                         "bytes %d-%d/%d" % (size - 7, size - 1, size))
        self.assertEqual(len(body), 7)
        with open(os.path.join(self.tmp, "lib", self.SONG_ID, "song.wav"), "rb") as f:
            f.seek(size - 7)
            self.assertEqual(body, f.read())

    def test_range_prefix(self):
        st, headers, body = self.raw_request(
            "GET", "/api/songs/%s/audio.wav" % self.SONG_ID,
            headers={"Range": "bytes=0-99"})
        self.assertEqual(st, 206)
        self.assertEqual(len(body), 100)
        self.assertTrue(headers.get("Content-Range", "").startswith("bytes 0-99/"))

    def test_range_out_of_bounds_416(self):
        size = self._wav_size()
        st, headers, _ = self.raw_request(
            "GET", "/api/songs/%s/audio.wav" % self.SONG_ID,
            headers={"Range": "bytes=%d-" % (size + 10)})
        self.assertEqual(st, 416)
        self.assertEqual(headers.get("Content-Range"), "bytes */%d" % size)

    def test_body_size_cap_413(self):
        big = json.dumps({"description": "x" * (1 * 1024 * 1024 + 100)}).encode()
        try:
            st, _, _ = self.raw_request("POST", "/api/compose", body=big,
                                        headers={"Content-Type": "application/json"})
            self.assertEqual(st, 413)
        except (BrokenPipeError, ConnectionResetError):
            pass  # 服务端提前断开（发 413 后关连接）同样属于正确的拒绝行为

    def test_traversal_blocked(self):
        for path in ("/../README.md", "/../../config.local.json",
                     "/..%2f..%2fconfig.local.json"):
            st, _, _ = self.raw_request("GET", path)
            self.assertIn(st, (403, 404), "%s 应被拒（实际 %d）" % (path, st))

    def test_nosniff_header(self):
        st, headers, _ = self.raw_request("GET", "/api/health")
        self.assertEqual(headers.get("X-Content-Type-Options"), "nosniff")


class TestNoTokenOpenByDefault(_ServerBase):
    """不设 Token 时一切照旧（本机模式）。"""

    @classmethod
    def setUpClass(cls):
        from app import server as server_mod
        cls.tmp = tempfile.mkdtemp(prefix="aimusic-sec3-")
        os.environ.pop("AI_MUSIC_TOKEN", None)
        cls.httpd = server_mod.create_server(
            "127.0.0.1", 0,
            lib_dir=os.path.join(cls.tmp, "lib"),
            config_path=os.path.join(cls.tmp, "config.local.json"),
        )
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.seed_song(os.path.join(cls.tmp, "lib"))

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_meta_open_without_token(self):
        st, _, _ = self.raw_request("GET", "/api/meta")
        self.assertEqual(st, 200)


if __name__ == "__main__":
    unittest.main()
