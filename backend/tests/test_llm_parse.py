"""LLM Provider 的解析/修复/兜底逻辑测试（不发真实网络请求）。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app import schemas
from app.providers import llm


class TestExtractJson(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(llm._extract_json('{"a": 1}'), {"a": 1})

    def test_fenced(self):
        text = '好的，以下是结果：\n```json\n{"title": "x"}\n```\n希望你喜欢'
        self.assertEqual(llm._extract_json(text), {"title": "x"})

    def test_wrapped(self):
        text = '模型输出如下 {"a": {"b": 2}} 以上'
        self.assertEqual(llm._extract_json(text), {"a": {"b": 2}})

    def test_garbage(self):
        with self.assertRaises((ValueError, Exception)):
            llm._extract_json("完全不是 JSON")


class TestProvideGuard(unittest.TestCase):
    def test_disabled_config(self):
        spec, info = llm.provide({"description": "test"}, {"llm": {"enabled": False}})
        self.assertIsNone(spec)
        self.assertIn("reason", info)

    def test_endpoint_normalize(self):
        self.assertEqual(llm._endpoint("https://x/v4"), "https://x/v4/chat/completions")
        self.assertEqual(llm._endpoint("https://x/v1/"), "https://x/v1/chat/completions")
        self.assertEqual(llm._endpoint("https://x/v1/chat/completions"),
                         "https://x/v1/chat/completions")

    def test_temperature_clamped(self):
        # 手填配置的容错：越界/非法值都收敛到 [0, 2]
        self.assertEqual(llm._temperature({"temperature": 5}), 2.0)
        self.assertEqual(llm._temperature({"temperature": -1}), 0.0)
        self.assertEqual(llm._temperature({"temperature": "abc"}), 0.9)
        self.assertEqual(llm._temperature({}), 0.9)
        self.assertEqual(llm._temperature({"temperature": 1.3}), 1.3)


class TestSpecRepair(unittest.TestCase):
    def test_repair_llm_spec(self):
        raw = {
            "title": "测试",
            "genre": "pop", "mood": "happy",
            "key": "C", "mode": "major", "bpm": 300,  # bpm 越界 -> 钳到 184
            "structure": [
                {"name": "verse", "bars": 4, "chords": ["C", "G", "Am", "F"], "lyrics": ["a", "b", "c", "d"]},
                {"name": "chorus", "bars": 8, "chords": ["F", "G", "C", "Am", "F", "G", "C", "C"]},
            ],
        }
        spec, _ = schemas.validate_and_repair_spec(raw, schemas.parse_request({"duration_sec": 60}))
        self.assertEqual(spec["bpm"], 184)
        total = sum(s["bars"] for s in spec["structure"])
        target = schemas.bars_for_duration(60, 184)
        self.assertEqual(total, target)
        for sec in spec["structure"]:
            self.assertEqual(len(sec["chords"]), sec["bars"])
            self.assertEqual(len(sec["lyrics"]), sec["bars"])

    def test_bad_chords_repaired(self):
        raw = {
            "title": "x", "genre": "pop", "mood": "happy",
            "key": "C", "mode": "major", "bpm": 100,
            "structure": [{"name": "chorus", "bars": 4, "chords": ["C13", "垃圾和弦", "G", "Am"]}],
        }
        spec, warnings = schemas.validate_and_repair_spec(raw, schemas.parse_request({"duration_sec": 30}))
        self.assertTrue(spec["structure"][0]["chords"])
        self.assertTrue(any("忽略非法和弦" in w for w in warnings))

    def test_empty_structure_raises(self):
        with self.assertRaises(schemas.SpecError):
            schemas.validate_and_repair_spec(
                {"title": "x", "structure": []},
                schemas.parse_request({"duration_sec": 30}))


if __name__ == "__main__":
    unittest.main()
