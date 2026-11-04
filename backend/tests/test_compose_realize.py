"""Mock 作曲 + 编曲器测试。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.music import mock_composer, realizer
from app import schemas
from app.schemas import parse_request


class TestMockComposer(unittest.TestCase):
    def test_basic_spec(self):
        spec = mock_composer.compose(parse_request({"description": "欢快的流行歌", "duration_sec": 30}))
        self.assertTrue(spec["title"])
        self.assertIn(spec["genre"], mock_composer.PROFILES)
        self.assertEqual(spec["beats_per_bar"], 4)
        total = sum(s["bars"] for s in spec["structure"])
        self.assertGreaterEqual(total, 4)
        for sec in spec["structure"]:
            self.assertEqual(len(sec["chords"]), sec["bars"])
            self.assertEqual(len(sec["lyrics"]), sec["bars"])

    def test_deterministic(self):
        req = {"description": "雨夜 lo-fi", "duration_sec": 40, "seed": 12345}
        a = mock_composer.compose(req)
        b = mock_composer.compose(req)
        self.assertEqual(a, b)

    def test_keyword_detect(self):
        spec = mock_composer.compose({"description": "古风江湖仙侠水墨", "duration_sec": 30})
        self.assertEqual(spec["genre"], "guofeng")
        self.assertEqual(spec["mode"], "chinese")
        spec2 = mock_composer.compose({"description": "悲伤的夜曲", "duration_sec": 30})
        self.assertEqual(spec2["mood"], "sad")


class TestRealizer(unittest.TestCase):
    def setUp(self):
        self.spec = mock_composer.compose({"description": "轻快的夏日流行", "duration_sec": 25, "seed": 7})
        self.song = realizer.realize(self.spec)

    def test_song_fields(self):
        song = self.song
        self.assertEqual(song["total_beats"], sum(s["bars"] for s in self.spec["structure"]) * 4)
        self.assertGreater(song["duration_sec"], 10)
        self.assertEqual(len(song["chords"]), song["total_beats"] // 4)
        self.assertEqual(song["sections"][0]["start_beat"], 0)
        # sections 首尾相接
        for a, b in zip(song["sections"], song["sections"][1:]):
            self.assertEqual(a["start_beat"] + a["beats"], b["start_beat"])

    def test_melody_sane(self):
        self.assertGreater(len(self.song["melody"]), 8)
        for n in self.song["melody"]:
            self.assertTrue(55 <= n["pitch"] <= 92, "melody pitch %s out of range" % n["pitch"])
            self.assertTrue(55 <= n["vel"] <= 115)
            self.assertGreater(n["dur"], 0)
        # 按时间排序
        starts = [n["start"] for n in self.song["melody"]]
        self.assertEqual(starts, sorted(starts))

    def test_bass_sane(self):
        for n in self.song["bass"]:
            self.assertTrue(33 <= n["pitch"] <= 52, "bass pitch %s out of range" % n["pitch"])

    def test_drums_present(self):
        types = {d["type"] for d in self.song["drums"]}
        self.assertIn("kick", types)
        for d in self.song["drums"]:
            self.assertTrue(20 <= d["vel"] <= 127)
            self.assertGreaterEqual(d["start"], 0)

    def test_genres_all_render(self):
        for genre in ("pop", "rock", "electronic", "ballad", "folk", "guofeng", "lofi", "jazz", "rap"):
            song = realizer.realize(mock_composer.compose(
                {"description": genre, "duration_sec": 15, "seed": 42}))
            self.assertGreater(len(song["melody"]), 0, genre)
            self.assertGreater(len(song["drums"]), 0, genre)


class TestRapGenre(unittest.TestCase):
    """说唱类型：作曲档案、boom-bap/trap 双鼓组、说唱 flow 旋律、风格标签。"""

    def test_detect_and_compose(self):
        spec = mock_composer.compose({"description": "一首燃炸的说唱", "duration_sec": 40, "seed": 7})
        self.assertEqual(spec["genre"], "rap")
        self.assertEqual(spec["mode"], "minor")
        self.assertIn(82, range(82, 97) if spec["bpm"] in range(82, 97) else [spec["bpm"]])
        self.assertTrue(any("麦在手里" in line or "verse里" in line for s in spec["structure"]
                            for line in s["lyrics"]), "应使用说唱歌词模板")

    def test_boombap_vs_trap_drums(self):
        base = mock_composer.compose({"description": "深夜说唱", "duration_sec": 40, "seed": 7})
        lo = realizer.realize(dict(base, bpm=88))
        hi = realizer.realize(dict(base, bpm=130))
        lo_beats = sorted({round(d["start"] % 4, 2) for d in lo["drums"] if d["type"] == "kick"})
        hi_claps = [d for d in hi["drums"] if d["type"] == "clap"]
        # boom-bap：kick 落在 swing 位移的切分（0.07 偏移存在）
        self.assertTrue(any(abs(b - round(b, 0)) > 0.01 for b in lo_beats), "boom-bap 应有 swing")
        # trap：clap 半拍（第 3 拍 = beat 2），无 swing
        self.assertTrue(hi_claps)
        self.assertTrue(all(abs(d["start"] % 4 - 2.0) < 0.01 for d in hi_claps))
        hi_kick = sorted({round(d["start"] % 4, 2) for d in hi["drums"] if d["type"] == "kick"})
        # trap：kick 在 0/1.5/2.5 的八分切分上，但无 0.07 swing 偏移
        self.assertEqual(hi_kick, [0.0, 1.5, 2.5])

    def test_trap_hat_roll(self):
        base = mock_composer.compose({"description": "深夜说唱", "duration_sec": 40, "seed": 7})
        hi = realizer.realize(dict(base, bpm=130))
        hats = [d for d in hi["drums"] if d["type"] == "hat"]
        self.assertGreater(len(hats), 8 * (hi["total_beats"] / 4) * 0.9, "trap hat 应更密(≥8/小节)")

    def test_rap_flow_narrow_melody(self):
        base = mock_composer.compose({"description": "深夜说唱", "duration_sec": 40, "seed": 7})
        song = realizer.realize(base)
        pitches = [m["pitch"] for m in song["melody"]]
        self.assertLessEqual(max(pitches) - min(pitches), 17, "说唱旋律应为窄带 talk-sing")

    def test_style_prompt_rap(self):
        from app.providers.diffrhythm import DiffRhythmProvider as P
        lo = P.build_style_prompt({"genre": "rap", "mood": "energetic", "bpm": 88})
        hi = P.build_style_prompt({"genre": "rap", "mood": "energetic", "bpm": 140})
        self.assertIn("rap", lo)
        self.assertIn("trap", hi)


class TestLyricsReq(unittest.TestCase):
    """写词要求：解析净化、指纹区分、风格/情绪识别参与。"""

    def test_audio_provider_selection_is_preserved(self):
        for provider in ("diffrhythm", "mureka"):
            with self.subTest(provider=provider):
                self.assertEqual(schemas.parse_request({"provider": provider})["provider"],
                                 provider)

    def test_local_first_defaults_and_invalid_provider_rejection(self):
        req = schemas.parse_request({})
        self.assertEqual(req["provider"], "local")
        self.assertEqual(req["composer_provider"], "mock")
        with self.assertRaises(schemas.RequestError):
            schemas.parse_request({"provider": "typo-cloud"})

    def test_parse_sanitize(self):
        req = schemas.parse_request({"lyrics_req": "  写一个\n\n深夜便利店的故事，押 an 韵  "})
        self.assertEqual(req["lyrics_req"], "写一个 深夜便利店的故事，押 an 韵")
        self.assertIsNone(schemas.parse_request({"lyrics_req": "   "})["lyrics_req"])
        self.assertIsNone(schemas.parse_request({})["lyrics_req"])
        self.assertEqual(len(schemas.parse_request({"lyrics_req": "字" * 300})["lyrics_req"]), 200)

    def test_fingerprint_distinguishes(self):
        from app.store import fingerprint_of
        a = fingerprint_of({"description": "x", "provider": "mock"})
        b = fingerprint_of({"description": "x", "provider": "mock", "lyrics_req": "押 an 韵"})
        self.assertNotEqual(a, b)

    def test_detect_uses_requirements(self):
        spec = mock_composer.compose({"description": "随手写一首",
                                      "lyrics_req": "深夜便利店 押韵 说唱",
                                      "duration_sec": 30, "seed": 5})
        self.assertEqual(spec["genre"], "rap")


if __name__ == "__main__":
    unittest.main()
