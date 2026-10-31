"""MIDI 写入与合成器渲染测试。"""
import os
import struct
import sys
import tempfile
import unittest
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.music import midi, mock_composer, realizer, synth


class TestMidi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = mock_composer.compose({"description": "流行 demo", "duration_sec": 15, "seed": 3})
        cls.song = realizer.realize(spec)
        cls.path = os.path.join(tempfile.mkdtemp(), "t.mid")
        cls.data = midi.write_midi_file(cls.song, cls.path)

    def test_header(self):
        d = self.data
        self.assertEqual(d[:4], b"MThd")
        fmt, ntrk, ppq = struct.unpack(">HHH", d[8:14])
        self.assertEqual(fmt, 1)
        self.assertEqual(ntrk, 5)
        self.assertEqual(ppq, 480)
        # chunk 长度自洽
        self.assertEqual(struct.unpack(">I", d[4:8])[0], 6)
        pos = 14
        while pos < len(d):
            tag = d[pos:pos + 4]
            ln = struct.unpack(">I", d[pos + 4:pos + 8])[0]
            self.assertEqual(tag, b"MTrk")
            pos += 8 + ln
        self.assertEqual(pos, len(d))

    def test_file_written(self):
        self.assertGreater(os.path.getsize(self.path), 200)


class TestSynth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = mock_composer.compose({"description": "流行 demo", "duration_sec": 12, "seed": 3})
        cls.song = realizer.realize(spec)
        cls.path = os.path.join(tempfile.mkdtemp(), "t.wav")
        synth.render_wav(cls.song, cls.path)

    def test_wav_format(self):
        with wave.open(self.path) as w:
            self.assertEqual(w.getnchannels(), 2)
            self.assertEqual(w.getsampwidth(), 2)
            self.assertEqual(w.getframerate(), 44100)
            dur = w.getnframes() / w.getframerate()
            self.assertAlmostEqual(dur, self.song["duration_sec"] + 0.6, delta=0.2)

    def test_not_silent(self):
        with wave.open(self.path) as w:
            data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float) / 32767
        rms = float(np.sqrt((data ** 2).mean()))
        self.assertGreater(rms, 0.02, "音频几乎无声")
        self.assertLess(float(np.abs(data).max()), 1.0, "削波")

    def test_drums_synths(self):
        fns = (synth._kick, synth._snare, lambda: synth._hat(False),
               lambda: synth._hat(True), synth._rim, synth._clap)
        for fn in fns:
            sig = fn()
            self.assertEqual(len(sig), len(np.asarray(sig)))
            self.assertGreater(float(np.abs(sig).max()), 0.05)


if __name__ == "__main__":
    unittest.main()
