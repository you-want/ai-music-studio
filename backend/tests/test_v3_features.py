"""v3 优化测试：母带链 / 渲染一次化 / 编曲动态 / 换副歌 / 结果缓存 / 新导出。"""
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
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app import schemas  # noqa: E402
from app.music import coverpng, midi, mock_composer, realizer, synth  # noqa: E402


class TestMasterChain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = mock_composer.compose({"description": "欢快流行", "duration_sec": 15, "seed": 3})
        cls.song = realizer.realize(spec)
        cls.tmp = tempfile.mkdtemp()
        t0 = time.time()
        cls.wav, cls.stems = synth.render_all(cls.song, os.path.join(cls.tmp, "m.wav"), cls.tmp)
        cls.render_sec = time.time() - t0

    def test_render_all_outputs(self):
        self.assertTrue(os.path.isfile(self.wav))
        names = sorted(os.path.basename(p) for p in self.stems)
        self.assertEqual(names, ["stem_bass.wav", "stem_drums.wav",
                                 "stem_harmony.wav", "stem_vocal.wav"])
        with wave.open(self.wav) as w:
            d = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float) / 32767
        rms = float(np.sqrt((d ** 2).mean()))
        self.assertGreater(rms, 0.03, "母带几乎无声")
        self.assertFalse(np.isnan(d).any())
        self.assertLessEqual(float(np.abs(d).max()), 0.95, "母带削波")
        # 分轨也正常
        for p in self.stems:
            with wave.open(p) as w:
                dd = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float) / 32767
            self.assertGreater(float(np.sqrt((dd ** 2).mean())), 0.005, p)

    def test_render_faster_than_double(self):
        """渲染一次化：整曲+分轨应明显快于两遍全渲染（宽松断言 < 6s/15s 曲）。"""
        self.assertLess(self.render_sec, 6.0, "渲染过慢: %.1fs" % self.render_sec)


class TestArrangementDynamics(unittest.TestCase):
    def test_intro_outro_bridge_no_arps(self):
        spec = mock_composer.compose({"description": "电子舞曲 蹦迪", "duration_sec": 30, "seed": 5})
        song = realizer.realize(spec)
        bpb = song["beats_per_bar"]
        name_of = {s["start_bar"]: s["name"] for s in song["sections"] for _ in [0]}
        def sec_name(bar):
            for s in song["sections"]:
                if s["start_bar"] <= bar < s["start_bar"] + s["bars"]:
                    return s["name"]
            return "?"
        for e in song["arps"]:
            self.assertNotIn(sec_name(int(e["start"] / bpb)), ("intro", "outro", "bridge"))

    def test_half_cadence_and_bridge_shift(self):
        spec = mock_composer.compose({"description": "欢快流行 一首小情歌", "duration_sec": 60, "seed": 9})
        key_pc = theory_note_pc(spec["key"])
        scale = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10],
                 "chinese": [0, 2, 4, 7, 9], "dorian": [0, 2, 3, 5, 7, 9, 10],
                 "minor_pentatonic": [0, 3, 5, 7, 10]}[spec["mode"]]
        dominant_root = (key_pc + scale[4]) % 12
        names = [s["name"] for s in spec["structure"]]
        for i, sec in enumerate(spec["structure"]):
            if sec["name"] == "verse" and i + 1 < len(names) and names[i + 1] == "chorus":
                from app.music import theory
                root, _q = theory.parse_chord(sec["chords"][-1])
                self.assertEqual(root, dominant_root, "主歌末小节应为属和弦(半终止)")
            if sec["name"] == "bridge":
                for c in sec["chords"]:
                    theory.parse_chord(c)  # 桥段和弦全部合法


def theory_note_pc(name):
    from app.music import theory
    return theory.note_to_pc(name)


class TestRegenChorus(unittest.TestCase):
    def test_family_seed_only_changes_chorus(self):
        spec = mock_composer.compose({"description": "欢快流行", "duration_sec": 40, "seed": 42})
        a = realizer.realize(spec)
        spec2 = dict(spec)
        spec2["family_seeds"] = {"chorus": 987654321}
        b = realizer.realize(spec2)
        va = [n for n in a["melody"] if _fam_of(a, n["start"]) == "verse"]
        vb = [n for n in b["melody"] if _fam_of(b, n["start"]) == "verse"]
        ca = [n for n in a["melody"] if _fam_of(a, n["start"]) == "chorus"]
        cb = [n for n in b["melody"] if _fam_of(b, n["start"]) == "chorus"]
        self.assertEqual(len(va), len(vb))
        self.assertEqual([n["pitch"] for n in va], [n["pitch"] for n in vb], "主歌应完全一致")
        self.assertNotEqual([n["pitch"] for n in ca], [n["pitch"] for n in cb], "副歌应被重写")
        # 贝斯/鼓不变
        self.assertEqual(a["bass"], b["bass"])
        self.assertEqual(a["drums"], b["drums"])


def _fam_of(song, start):
    bpb = song["beats_per_bar"]
    bar = int(start // bpb)
    for s in song["sections"]:
        if s["start_bar"] <= bar < s["start_bar"] + s["bars"]:
            return s["name"]
    return "?"


class TestCoverPng(unittest.TestCase):
    def test_png_valid(self):
        p = os.path.join(tempfile.mkdtemp(), "c.png")
        coverpng.make_cover("test-id", size=256, path=p)
        raw = open(p, "rb").read()
        self.assertEqual(raw[:8], b"\x89PNG\r\n\x1a\n")
        w = struct.unpack(">I", raw[16:20])[0]
        self.assertEqual(w, 256)


class TestV3API(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import server as server_mod
        cls.tmp = tempfile.mkdtemp(prefix="aimusic-v3-")
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

    def _req(self, path, method="GET", body=None, raw=False):
        data = json.dumps(body).encode() if isinstance(body, dict) else body
        r = urllib.request.Request(self.base + path, data=data, method=method,
                                   headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=120) as resp:
                payload = resp.read()
                return resp.status, (payload if raw else json.loads(payload))
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def _compose(self, params, timeout=240):
        _, b = self._req("/api/compose", "POST", params)
        jid = b["job_id"]
        t0 = time.time()
        while time.time() - t0 < timeout:
            _, b = self._req("/api/jobs/" + jid)
            j = b
            if j["status"] in ("done", "error"):
                return j
            time.sleep(0.3)
        raise AssertionError("timeout")

    def test_full_v3_flow(self):
        j = self._compose({"description": "清新民谣弹唱", "duration_sec": 15, "seed": 555,
                           "title": "v3测试曲"})
        self.assertEqual(j["status"], "done", j.get("error"))
        sid = j["result"]["song_id"]

        # 结果复用缓存：同参数再次提交 → 直接复用
        j2 = self._compose({"description": "清新民谣弹唱", "duration_sec": 15, "seed": 555,
                            "title": "v3测试曲"})
        self.assertEqual(j2["status"], "done")
        self.assertTrue(j2["result"].get("cached"))
        self.assertEqual(j2["result"]["song_id"], sid)

        # 换副歌 → 新作品，主歌一致
        st, b = self._req("/api/songs/%s/regen_chorus" % sid, "POST")
        self.assertEqual(st, 200, b)
        new_id = b["song_id"]
        self.assertNotEqual(new_id, sid)
        _, a = self._req("/api/songs/" + sid)
        _, c = self._req("/api/songs/" + new_id)
        self.assertEqual(a["spec"]["structure"], c["spec"]["structure"], "曲式/和弦应一致")

        # 改名
        st, b = self._req("/api/songs/" + sid, "PATCH", {"title": "改名后的歌"})
        self.assertEqual(st, 200)
        self.assertEqual(b["title"], "改名后的歌")

        # 收藏
        st, b = self._req("/api/songs/%s/favorite" % sid, "POST", {"favorite": True})
        self.assertTrue(b["favorite"])
        _, b = self._req("/api/songs")
        self.assertTrue(b["songs"][0]["favorite"], "收藏应排在最前")

        # 分享包
        st, raw = self._req("/api/songs/%s/bundle.zip" % sid, raw=True)
        self.assertEqual(st, 200)
        self.assertEqual(raw[:2], b"PK")

        # MV（本机有 ffmpeg）
        st, raw = self._req("/api/songs/%s/video.mp4" % sid, raw=True)
        self.assertEqual(st, 200, raw[:200])
        self.assertIn(b"ftyp", raw[:32])


if __name__ == "__main__":
    unittest.main()
