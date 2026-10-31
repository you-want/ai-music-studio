"""乐理库测试。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.music import theory


class TestTheory(unittest.TestCase):
    def test_note_to_pc(self):
        self.assertEqual(theory.note_to_pc("C"), 0)
        self.assertEqual(theory.note_to_pc("C#"), 1)
        self.assertEqual(theory.note_to_pc("Bb"), 10)
        self.assertEqual(theory.note_to_pc("E"), 4)
        with self.assertRaises(ValueError):
            theory.note_to_pc("H")

    def test_parse_chord(self):
        self.assertEqual(theory.parse_chord("C"), (0, "maj"))
        self.assertEqual(theory.parse_chord("Am7"), (9, "m7"))
        self.assertEqual(theory.parse_chord("Fmaj7"), (5, "maj7"))
        self.assertEqual(theory.parse_chord("Gsus4"), (7, "sus4"))
        with self.assertRaises(ValueError):
            theory.parse_chord("X13")

    def test_chord_symbol_roundtrip(self):
        for root in range(12):
            for q in ("maj", "min", "m7", "maj7", "7", "sus4"):
                sym = theory.chord_symbol(root, q)
                self.assertEqual(theory.parse_chord(sym), (root, q))

    def test_chord_pitches(self):
        self.assertEqual(theory.chord_pitches(0, "maj", 48), [48, 52, 55])
        self.assertEqual(theory.chord_pitches(9, "min", 48), [57, 60, 64])

    def test_degree_pitch(self):
        # C 大调 degree 0 -> C3(48), degree 7 -> C4(60)
        self.assertEqual(theory.degree_pitch(0, theory.SCALES["major"], 0), 48)
        self.assertEqual(theory.degree_pitch(0, theory.SCALES["major"], 7), 60)
        # 负级度
        self.assertEqual(theory.degree_pitch(0, theory.SCALES["major"], -7), 36)

    def test_chord_tone_degrees(self):
        degs = theory.chord_tone_degrees(0, theory.SCALES["major"], 0, "maj")
        self.assertIn(0, degs)   # C
        self.assertIn(2, degs)   # E
        self.assertIn(4, degs)   # G
        self.assertNotIn(1, degs)

    def test_snap_range(self):
        d = theory.snap_degree_to_pitch_range(0, 0, theory.SCALES["major"], 60, 88)
        self.assertGreaterEqual(theory.degree_pitch(0, theory.SCALES["major"], d), 60)


if __name__ == "__main__":
    unittest.main()
