"""v2 功能测试：自定义歌词、AI 写词兜底、人声哼唱、分轨导出、进阶 API。"""
import json
import os
import shutil
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
from app.music import midi, mock_composer, realizer, synth  # noqa: E402


class TestLyricsText(unittest.TestCase):
    def test_parse_markers(self):
        text = "[Verse]\n甲\n乙\n[Chorus]\n丙\n[主歌]\n丁"
        b = schemas.parse_lyrics_text(text)
        self.assertEqual(b["verse"], ["甲", "乙", "丁"])  # 中文标记并入 verse
        self.assertEqual(b["chorus"], ["丙"])

    def test_parse_no_markers(self):
        b = schemas.parse_lyrics_text("一行\n两行\n\n三行")
        self.assertEqual(b["auto"], ["一行", "两行", "三行"])

    def test_apply_lyrics(self):
        spec = mock_composer.compose({"description": "欢快流行", "duration_sec": 30, "seed": 1})
        schemas.apply_lyrics(spec, "[Verse]\n主词一\n主词二\n[Chorus]\n副词一\n副词二")
        self.assertEqual(spec["lyrics_source"], "custom")
        verse = next(s for s in spec["structure"] if s["name"] == "verse")
        chorus = next(s for s in spec["structure"] if s["name"] == "chorus")
        self.assertEqual(verse["lyrics"][0], "主词一")
        self.assertEqual(chorus["lyrics"][0], "副词一")
        for sec in spec["structure"]:
            if sec["name"] in ("intro", "outro"):
                self.assertTrue(all(x == "" for x in sec["lyrics"]))

    def test_apply_lyrics_spill_missing_section(self):
        # 15s 短曲可能没有主歌：[Verse] 歌词应并入自动分配而不是丢掉
        spec = mock_composer.compose({"description": "欢快", "duration_sec": 15, "seed": 5})
        schemas.apply_lyrics(spec, "[Verse]\nAAA\n[Chorus]\nBBB")
        used = [l for s in spec["structure"] for l in s["lyrics"]]
        self.assertIn("AAA", used)
        self.assertIn("BBB", used)

    def test_mock_lyrics_text(self):
        t = mock_composer.mock_lyrics_text({"description": "夏天的傍晚骑车回家"})
        for marker in ("[Verse]", "[Chorus]", "[Bridge]"):
            self.assertIn(marker, t)
        self.assertGreater(len(t), 100)


class TestVocalHum(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = mock_composer.compose({"description": "抒情钢琴", "duration_sec": 12, "seed": 2})
        spec["lead_voice"] = "vocal_hum"
        cls.song = realizer.realize(spec)
        cls.tmp = tempfile.mkdtemp()

    def test_render_and_stems(self):
        wav = os.path.join(self.tmp, "main.wav")
        synth.render_wav(self.song, wav)
        with wave.open(wav) as w:
            data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float) / 32767
        self.assertGreater(float(np.sqrt((data ** 2).mean())), 0.02)
        stems = synth.render_stems(self.song, self.tmp)
        self.assertEqual(len(stems), 4)
        names = sorted(os.path.basename(p) for p in stems)
        self.assertEqual(names, ["stem_bass.wav", "stem_drums.wav",
                                 "stem_harmony.wav", "stem_vocal.wav"])
        for p in stems:
            with wave.open(p) as w:
                d = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float) / 32767
                self.assertGreater(float(np.sqrt((d ** 2).mean())), 0.005, p)

    def test_midi_voice_program(self):
        from app.music.midi import VOICE_GM
        path = os.path.join(self.tmp, "a.mid")
        data = midi.write_midi_file(self.song, path)
        self.assertIn(bytes([0xC0, VOICE_GM["vocal_hum"]]), data)


class TestSinging(unittest.TestCase):
    """AI 演唱（macOS TTS）：无 say/中文语音的机器自动跳过。"""

    @classmethod
    def setUpClass(cls):
        from app.music import singer
        cls.voice = singer.available_voice()
        spec = mock_composer.compose({"description": "欢快流行", "duration_sec": 15, "seed": 11})
        cls.song = realizer.realize(spec)
        cls.events = singer.plan_singing(cls.song) if cls.voice else []

    @unittest.skipUnless(__import__("app.music.singer", fromlist=["available_voice"]).available_voice(),
                         "需要 macOS say + 中文语音")
    def test_plan_singing(self):
        self.assertGreater(len(self.events), 10)
        self.assertTrue(all("\u4e00" <= e["syllable"] <= "\u9fff" for e in self.events[:5]))
        melody = {n["pitch"] for n in self.song["melody"]}
        self.assertTrue(all(e["pitch"] in melody for e in self.events))

    @unittest.skipUnless(__import__("app.music.singer", fromlist=["available_voice"]).available_voice(),
                         "需要 macOS say + 中文语音")
    def test_render_with_singing(self):
        import wave as wavemod
        self.song["sing_events"] = self.events
        tmp = tempfile.mkdtemp()
        wav = os.path.join(tmp, "s.wav")
        synth.render_wav(self.song, wav)
        with wavemod.open(wav) as w:
            d = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float) / 32767
        self.assertGreater(float(np.sqrt((d ** 2).mean())), 0.03)
        stems = synth.render_stems(self.song, tmp)
        self.assertTrue(any(p.endswith("stem_vocal.wav") for p in stems))


class TestAdvancedAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import server as server_mod
        cls.tmp = tempfile.mkdtemp(prefix="aimusic-v2-")
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

    def _req(self, path, method="GET", body=None, headers=None):
        data = json.dumps(body).encode() if isinstance(body, dict) else body
        r = urllib.request.Request(self.base + path, data=data, method=method,
                                   headers={"Content-Type": "application/json", **(headers or {})})
        with urllib.request.urlopen(r, timeout=60) as resp:
            return resp.status, resp.read()

    def _wait_done(self, job_id, timeout=180):
        t0 = time.time()
        while time.time() - t0 < timeout:
            _, b = self._req("/api/jobs/" + job_id)
            j = json.loads(b)
            if j["status"] in ("done", "error"):
                return j
            time.sleep(0.3)
        raise AssertionError("job timeout")

    def test_meta_has_voices_keys(self):
        _, b = self._req("/api/meta")
        meta = json.loads(b)
        ids = [v["id"] for v in meta["voices"]]
        self.assertIn("vocal_hum", ids)
        self.assertIn("auto", ids)
        self.assertGreaterEqual(len(meta["keys"]), 13)
        self.assertIn("sing_supported", meta)

    def test_lyrics_endpoint(self):
        _, b = self._req("/api/lyrics", "POST", {"description": "夏夜星空下的约定"})
        d = json.loads(b)
        self.assertIn(d["source"], ("mock", "llm"))
        self.assertIn("[Chorus]", d["lyrics"])

    def test_advanced_compose_full(self):
        lyrics = "[Verse]\n夏夜的风吹过屋顶\n星星掉进了你的眼睛\n\n[Chorus]\n约好了要看流星\n别迟到了别分心"
        _, b = self._req("/api/compose", "POST", {
            "description": "清新民谣", "duration_sec": 20,
            "title": "夏夜约定", "lyrics": lyrics,
            "lead_voice": "vocal_hum", "key": "G", "seed": 42,
        })
        job = self._wait_done(json.loads(b)["job_id"])
        self.assertEqual(job["status"], "done", job.get("error"))
        sid = job["result"]["song_id"]
        _, b = self._req("/api/songs/" + sid)
        full = json.loads(b)
        self.assertEqual(full["meta"]["title"], "夏夜约定")
        self.assertEqual(full["spec"]["key"], "G")
        self.assertEqual(full["spec"]["lyrics_source"], "custom")
        used = [l for s in full["spec"]["structure"] for l in s["lyrics"]]
        self.assertIn("夏夜的风吹过屋顶", used)
        # 分轨可下载
        for name in ("vocal", "harmony", "bass", "drums"):
            st, raw = self._req("/api/songs/%s/stems/%s.wav" % (sid, name))
            self.assertEqual(st, 200)
            self.assertEqual(raw[:4], b"RIFF")

    def test_default_sing_on_and_off(self):
        # 不传 sing -> 默认演唱；显式 sing=false -> 纯器乐
        _, b = self._req("/api/compose", "POST",
                         {"description": "轻快流行", "duration_sec": 12, "seed": 77})
        job = self._wait_done(json.loads(b)["job_id"])
        self.assertEqual(job["status"], "done", job.get("error"))
        _, b = self._req("/api/songs/" + job["result"]["song_id"])
        self.assertEqual(json.loads(b)["spec"].get("sing"), True)

        _, b = self._req("/api/compose", "POST",
                         {"description": "轻快流行", "duration_sec": 12, "seed": 78, "sing": False})
        job = self._wait_done(json.loads(b)["job_id"])
        self.assertEqual(job["status"], "done", job.get("error"))
        _, b = self._req("/api/songs/" + job["result"]["song_id"])
        self.assertNotIn("sing", json.loads(b)["spec"])

    def test_sing_compose(self):
        _, b = self._req("/api/compose", "POST", {
            "description": "欢快流行", "duration_sec": 15,
            "lyrics": "[Chorus]\n啦啦啦 大声唱出来\n把烦恼都甩在身后\n今天的天空特别蓝\n快乐就是这么简单",
            "seed": 21, "sing": True,
        })
        job = self._wait_done(json.loads(b)["job_id"])
        self.assertEqual(job["status"], "done", job.get("error"))
        sid = job["result"]["song_id"]
        _, b = self._req("/api/songs/" + sid)
        full = json.loads(b)
        self.assertEqual(full["spec"].get("sing"), True)
        st, raw = self._req("/api/songs/%s/stems/vocal.wav" % sid)
        self.assertEqual(st, 200)
        self.assertEqual(raw[:4], b"RIFF")


if __name__ == "__main__":
    unittest.main()
