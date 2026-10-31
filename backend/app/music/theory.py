"""乐理工具：音名、音阶、和弦解析。

所有 pitch 均为 MIDI 编号（C4 = 60）。degree（级度）指音阶中的序数，
支持越界（负数/超过一个八度），由本模块负责换算成 pitch。
"""
import re

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

NOTE_TO_PC = {name: i for i, name in enumerate(NOTE_NAMES)}
NOTE_TO_PC.update({"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10, "E#": 5, "B#": 0})

# 音阶（半音偏移）。chinese = 宫调式五声音阶。
SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "chinese": [0, 2, 4, 7, 9],
    "minor_pentatonic": [0, 3, 5, 7, 10],
}

# 和弦品质 -> 半音堆叠。maj 即大三和弦（符号里省略）。
CHORD_INTERVALS = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "6": [0, 4, 7, 9],
    "m6": [0, 3, 7, 9],
    "7": [0, 4, 7, 10],
    "m7": [0, 3, 7, 10],
    "maj7": [0, 4, 7, 11],
    "add9": [0, 4, 7, 14],
}

_QUALITY_ALIASES = {
    "": "maj", "M": "maj", "major": "maj",
    "m": "min", "min": "min", "minor": "min",
    "dom7": "7",
}

_SUFFIX = {"maj": "", "min": "m"}

_CHORD_RE = re.compile(r"^([A-Ga-g][#b]?)(.*)$")


def note_to_pc(name):
    """音名 -> 音高类(0..11)。非法输入抛 ValueError；整数(0-11)直接取模。"""
    if isinstance(name, bool):
        raise ValueError("无法识别的音名: %r" % (name,))
    if isinstance(name, int):
        return name % 12
    s = str(name).strip()
    if s in NOTE_TO_PC:
        return NOTE_TO_PC[s]
    if s in NOTE_TO_PC or (len(s) >= 1 and s[0].upper() + s[1:] in NOTE_TO_PC):
        return NOTE_TO_PC[s[0].upper() + s[1:]]
    m = re.match(r"^([A-Ga-g])([#b]+)$", s)
    if m:
        base = NOTE_TO_PC[m.group(1).upper()]
        for ch in m.group(2):
            base += 1 if ch == "#" else -1
        return base % 12
    raise ValueError("无法识别的音名: %r" % (name,))


def parse_chord(symbol):
    """'Am7' -> (root_pc=9, quality='m7')。非法抛 ValueError。"""
    s = str(symbol).strip()
    m = _CHORD_RE.match(s)
    if not m:
        raise ValueError("无法识别的和弦: %r" % (symbol,))
    root = note_to_pc(m.group(1))
    q = m.group(2).strip()
    q = _QUALITY_ALIASES.get(q, q)
    if q not in CHORD_INTERVALS:
        raise ValueError("不支持的和弦品质: %r (来自 %r)" % (q, symbol))
    return root, q


def chord_symbol(root_pc, quality):
    """(9, 'min') -> 'Am'。quality 用规范名。"""
    return NOTE_NAMES[root_pc % 12] + _SUFFIX.get(quality, quality if quality != "maj" else "")


def degree_root_pc(key_pc, scale, degree):
    """音阶级度 -> 音高类(0..11)。degree 支持负数/超过音阶长度。"""
    n = len(scale)
    return (key_pc + scale[degree % n] + 12 * (degree // n)) % 12


def degree_pitch(key_pc, scale, degree, base=48):
    """音阶级度 -> MIDI 编号。degree 0 对应 base(默认48, 即C3) + key_pc。"""
    n = len(scale)
    return base + key_pc + 12 * (degree // n) + scale[degree % n]


def chord_pitches(root_pc, quality, base=48):
    """和弦声位：root 落在 base..base+11 音区（默认 48..59, C3..B3）。"""
    root_midi = base + (root_pc % 12)
    return [root_midi + iv for iv in CHORD_INTERVALS[quality]]


def chord_tone_degrees(key_pc, scale, root_pc, quality, lo=-8, hi=22):
    """找出落在和弦内的音阶级度（供旋律强拍吸附）。"""
    pcs = {(root_pc + iv) % 12 for iv in CHORD_INTERVALS[quality]}
    return [d for d in range(lo, hi) if degree_pitch(key_pc, scale, d) % 12 in pcs]


def snap_degree_to_pitch_range(degree, key_pc, scale, lo=60, hi=88):
    """平移八度使 pitch 落在 [lo, hi]。返回调整后的 degree。"""
    n = len(scale)
    p = degree_pitch(key_pc, scale, degree)
    while p < lo:
        degree += n
        p += 12
    while p > hi:
        degree -= n
        p -= 12
    return degree
