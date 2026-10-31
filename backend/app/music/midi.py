"""标准 MIDI 文件写入（SMF format 1, PPQ=480）。

轨道：0 速度/拍号/段落标记；1 旋律；2 和声(垫/琶音/键琴)；3 贝斯；4 鼓组(通道9)。
"""
import struct

PPQ = 480

# genre -> (旋律音色, 和声音色, 贝斯音色)  GM 程序号
GM_PROGRAMS = {
    "pop": (81, 89, 33),
    "rock": (30, 48, 33),
    "electronic": (81, 89, 38),
    "ballad": (0, 48, 33),
    "folk": (24, 89, 33),
    "guofeng": (107, 48, 32),
    "lofi": (4, 89, 33),
    "jazz": (4, 4, 32),
}
DRUM_NOTE = {"kick": 36, "snare": 38, "hat": 42, "ohat": 46, "clap": 39, "rim": 37}


# 用户指定主奏音色 -> GM 程序号（vocal_hum 对应 Voice Oohs）
VOICE_GM = {"synth_lead": 81, "square_lead": 81, "piano": 0,
            "pluck": 107, "keys": 4, "vocal_hum": 53}


def _varlen(n):
    """MIDI 变长数量编码。"""
    n = max(0, int(n))
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def _chunk(tag, payload):
    return tag + struct.pack(">I", len(payload)) + payload


def _meta(tick, last, meta_type, data):
    delta = max(0, tick - last)
    return delta, _varlen(delta) + bytes([0xFF, meta_type]) + _varlen(len(data)) + data, tick


def _note_track(notes, prog=None, channel=0):
    """notes: [(start_beat, dur_beat, pitch, vel)] -> MTrk bytes。"""
    evs = []
    for start, dur, pitch, vel in notes:
        t0 = int(round(start * PPQ))
        t1 = max(t0 + 10, int(round((start + dur) * PPQ)))
        evs.append((t0, 1, bytes([0x90 | channel, int(pitch) & 0x7F, int(vel) & 0x7F])))
        evs.append((t1, 0, bytes([0x80 | channel, int(pitch) & 0x7F, 0])))
    evs.sort(key=lambda e: (e[0], e[1]))
    data = bytearray()
    last = 0
    if prog is not None:
        data += bytes([0xC0 | channel, prog])
    for tick, _order, payload in evs:
        data += _varlen(max(0, tick - last)) + payload
        last = tick
    return _chunk(b"MTrk", bytes(data) + b"\x00\xff\x2f\x00")


def write_midi_file(song, path):
    bpb = song["beats_per_bar"]
    bpm = song["bpm"]
    prog_m, prog_h, prog_b = GM_PROGRAMS.get(song["spec"]["genre"], GM_PROGRAMS["pop"])
    lead_voice = (song.get("spec") or {}).get("lead_voice")
    if lead_voice in VOICE_GM:
        prog_m = VOICE_GM[lead_voice]

    # 轨道 0：速度 / 拍号 / 段落标记
    tempo_us = int(round(60_000_000 / bpm))
    ev0 = [(0, 0, b"\xff\x51\x03" + struct.pack(">I", tempo_us)[1:]),
           (0, 0, b"\xff\x58\x04" + bytes([bpb, 2, 24, 8]))]
    for sec in song["sections"]:
        text = sec.get("label") or sec["name"]
        ev0.append((int(round(sec["start_beat"] * PPQ)), 0,
                    b"\xff\x06" + _varlen(len(text.encode("utf-8"))) + text.encode("utf-8")))
    ev0.sort(key=lambda e: e[0])
    data = bytearray()
    last = 0
    for tick, _o, payload in ev0:
        data += _varlen(max(0, tick - last)) + payload
        last = tick
    track0 = _chunk(b"MTrk", bytes(data) + b"\x00\xff\x2f\x00")

    melody = [(e["start"], e["dur"], e["pitch"], e["vel"]) for e in song["melody"]]
    harmony = []
    for e in song["pads"]:
        for p in e["pitches"]:
            harmony.append((e["start"], e["dur"], p, e["vel"]))
    for e in song["arps"]:
        harmony.append((e["start"], e["dur"], e["pitch"], e["vel"]))
    for e in song["stabs"]:
        for p in e["pitches"]:
            harmony.append((e["start"], e["dur"], p, e["vel"]))
    bass = [(e["start"], e["dur"], e["pitch"], e["vel"]) for e in song["bass"]]
    drums = [(e["start"], min(0.25, 60.0 / bpm), DRUM_NOTE[e["type"]], e["vel"])
             for e in song["drums"]]

    smf = (b"MThd" + struct.pack(">IHHH", 6, 1, 5, PPQ)
           + track0
           + _note_track(melody, prog_m, 0)
           + _note_track(harmony, prog_h, 1)
           + _note_track(bass, prog_b, 2)
           + _note_track(drums, None, 9))
    with open(path, "wb") as f:
        f.write(smf)
    return smf
