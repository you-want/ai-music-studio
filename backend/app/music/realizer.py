"""编曲器：把高层 SongSpec 展开为可演奏的音符事件。

输入  spec（调性/速度/曲式/和弦/歌词）
输出  song（melody/bass/pads/arps/stabs/drums 事件，单位均为"拍"）

设计要点：
- 旋律按 section 家族（主歌/副歌…）维护动机（节奏型+轮廓），4 小节乐句
  用正弦弧线控制走向；强拍吸附和弦音，段落结尾落回主音。
- 贝斯/鼓组按风格套模板，爵士/lofi 有 swing，段落尾有军鼓填充。
- 全部随机走 spec.seed 派生的 rng，同一 seed 完全可复现。
"""
import math
import random
import zlib

from .. import schemas
from . import theory

BASS_LO, BASS_HI = 34, 50
MELODY_LO, MELODY_HI = 60, 88

# ---------------------------------------------------------------- 节奏模板
# 每项为 (起始拍, 时值拍)，一 小节 = 4 拍
RHYTHM_POOL = {
    "verse": [
        [(0, 1), (1, .5), (1.5, .5), (2, 1), (3, 1)],
        [(0, .5), (.5, .5), (1, 1), (2, 1.5), (3.5, .5)],
        [(0, 1.5), (2, .5), (2.5, .5), (3, 1)],
    ],
    "pre": [
        [(0, .5), (.5, .5), (1, .5), (1.5, .5), (2, 1), (3, 1)],
        [(0, 1), (1, .5), (1.5, .5), (2, .5), (2.5, .5), (3, .5), (3.5, .5)],
    ],
    "chorus": [
        [(0, .5), (.5, .5), (1, 1), (2, .5), (2.5, .5), (3, 1)],
        [(0, 1), (1, .5), (1.5, .5), (2, 1), (3, .5), (3.5, .5)],
        [(0, .5), (.5, .5), (1, .5), (1.5, .5), (2, 1), (3, 1)],
    ],
    "bridge": [
        [(0, 1), (1, 1), (2, 1.5), (3.5, .5)],
        [(0, .5), (.5, .5), (1, 1), (2, 1), (3, 1)],
    ],
}
FAMILY_OF = {"verse": "verse", "prechorus": "pre", "chorus": "chorus", "bridge": "bridge"}
FAMILY_VEL = {"verse": 0, "pre": 4, "chorus": 8, "bridge": 2}

# 说唱专用：稀疏暗色动机（trap beat 的主角是鼓组和 808，旋律只做点缀）
RAP_RHYTHM = {
    "verse": [
        [(0, .75)],
        [(0, .5), (1.5, .75)],
        [(.5, .75), (2.5, .75)],
        [(0, .75), (2, .5), (3, .75)],
    ],
    "pre": [[(0, .75), (2, .5)], [(.5, .5), (2, .75)]],
    "chorus": [
        [(0, .75), (1.5, .5), (2.5, .75)],
        [(0, .5), (1, .5), (2, .75), (3.5, .5)],
        [(0, .75), (2, .75), (3.25, .5)],
    ],
    "bridge": [[(0, .75), (2, .75)], [(.5, .5), (2.5, .5)]],
}

# ---------------------------------------------------------------- 鼓组模板
# step: 0..15（十六分音符）；type: kick/snare/hat/ohat/clap/rim
_DRUM_DEFAULT_VEL = {"kick": 96, "snare": 90, "hat": 55, "ohat": 55, "clap": 85, "rim": 55}

DRUM_PATTERNS = {
    "pop":        {"kick": [0, 8, 10], "snare": [4, 12], "hat": [0, 2, 4, 6, 8, 10, 12], "ohat": [14]},
    "rock":       {"kick": [0, 6, 8], "snare": [4, 12], "hat": [0, 2, 4, 6, 8, 10, 12, 14]},
    "electronic": {"kick": [0, 4, 8, 12], "clap": [4, 12], "hat": [2, 6, 10, 14]},
    "ballad":     {"kick": [0, 8], "snare": [4, 12], "hat": [0, 4, 8, 12]},
    "folk":       {"kick": [0, 8], "snare": [4, 12], "hat": [0, 2, 4, 6, 8, 10, 12, 14]},
    "guofeng":    {"kick": [0, 8], "rim": [4, 12]},
    "lofi":       {"kick": [0, 7, 10], "snare": [4, 12], "hat": [0, 2, 4, 6, 8, 10, 12, 14]},
    "jazz":       {"kick": [0], "snare": [4, 12], "hat": [0, 4, 8, 12, 2, 6, 10, 14]},
    # 说唱两套：BPM<108 走 boom-bap（摇摆 hat + 底鼓 0/7/10），>=108 走 trap
    # （半拍军鼓在第 3 拍 + 结尾 hi-hat 小滚奏）
    "rap":        {"kick": [0, 7, 10], "snare": [4, 12], "hat": [0, 2, 4, 6, 8, 10, 12, 14], "ohat": [15]},
    "rap_trap":   {"kick": [0, 6, 10], "clap": [8],
                   "hat": [0, 2, 4, 6, 8, 10, 12, 12.66, 13.33, 14, 14.66, 15]},
}
DRUM_VEL_ADJ = {
    "ballad": {"snare": -30, "kick": -10, "hat": -12},
    "guofeng": {"kick": -35, "rim": -5},
    "lofi": {"hat": -12, "snare": -30, "kick": -12},
    "jazz": {"kick": -50, "snare": -40, "hat": 5},
    "folk": {"hat": -8},
    "rap": {"kick": 6, "snare": 4, "hat": -4},
    "rap_trap": {"kick": 8, "clap": 6, "hat": -2},
}
SWING_GENRES = {"jazz", "lofi"}
SWING_BEATS = 0.07
FILL_GENRES = {"pop", "rock", "electronic", "folk", "lofi", "jazz", "ballad", "rap"}

# ---------------------------------------------------------------- 贝斯模板
# (起始拍, 时值, 种类 root/third/fifth/oct/approach)
BASS_PATTERNS = {
    "pop": [(0, 1.5, "root"), (1.5, .5, "root"), (2, 1, "root"), (3, 1, "fifth")],
    "rock": [(i * .5, .45, "root") for i in range(8)],
    "electronic": [(0, .45, "root"), (.5, .45, "root"), (1, .45, "root"), (1.5, .45, "oct"),
                   (2, .45, "root"), (2.5, .45, "root"), (3, .45, "fifth"), (3.5, .45, "oct")],
    "ballad": [(0, 2, "root"), (2, 1.5, "fifth"), (3.5, .5, "root")],
    "folk": [(0, 1.5, "root"), (2, 1, "fifth"), (3, 1, "root")],
    "guofeng": [(0, 3.5, "root")],
    "lofi": [(0, 1.4, "root"), (2, 1, "root"), (3.25, .6, "fifth")],
    "jazz": "walking",
    # 说唱 808：稀疏长音贴底鼓走，副歌前半拍加八度点缀
    "rap": [(0, 1.75, "root"), (2, 1.25, "root"), (3.5, .5, "oct")],
}
ARP_GENRES = {"electronic", "folk", "guofeng"}
STAB_GENRES = {"lofi", "jazz"}


def realize(spec):
    bpb = spec.get("beats_per_bar", 4)
    bpm = spec["bpm"]
    seed = spec.get("seed") or 0
    rng = random.Random(seed ^ 0xB455)  # 贝斯/鼓等编曲随机源（与旋律独立，便于只重生成某段落）

    key_pc = theory.note_to_pc(spec["key"])
    scale = theory.SCALES[spec["mode"]]
    n = len(scale)

    # ---- 展开结构 ----
    sections, chords_by_bar = [], []
    bar_cursor = 0
    for sec in spec["structure"]:
        start = bar_cursor
        for i in range(sec["bars"]):
            symbol = sec["chords"][i % len(sec["chords"])]
            try:
                root, quality = theory.parse_chord(symbol)
            except ValueError:
                root, quality = key_pc, "maj"
            chords_by_bar.append({
                "bar": start + i, "symbol": symbol, "root": root, "quality": quality,
                "pitches": theory.chord_pitches(root, quality, 48),
            })
        sections.append({
            "name": sec["name"], "label": schemas.SECTION_LABELS_ZH.get(sec["name"], sec["name"]),
            "start_bar": start, "bars": sec["bars"],
            "start_beat": start * bpb, "beats": sec["bars"] * bpb,
            "lyrics": list(sec.get("lyrics", [])),
        })
        bar_cursor += sec["bars"]
    total_bars = bar_cursor
    total_beats = total_bars * bpb

    melody = _melody(spec, sections, chords_by_bar, key_pc, scale, n, bpb)
    if spec["genre"] == "rap":
        melody = _rap_flow(spec, melody, key_pc, scale, n)
    bass = _bass(spec, sections, chords_by_bar, total_bars, bpb, rng)
    pads = _pads(chords_by_bar, bpb, spec["genre"])
    arps = _arps(spec, chords_by_bar, bpb) if spec["genre"] in ARP_GENRES else []
    stabs = _stabs(spec, chords_by_bar, bpb) if spec["genre"] in STAB_GENRES else []
    drums = _drums(spec, sections, total_bars, bpb, rng)

    # ---- 编曲动态：intro/outro/bridge 减配，副歌前段落留白 ----
    def _quiet(bar):
        return _section_of(sections, int(bar))["name"] in ("intro", "outro", "bridge")

    arps = [e for e in arps if not _quiet(e["start"] / bpb)]
    stabs = [e for e in stabs if not _quiet(e["start"] / bpb)]
    if spec["genre"] in ("ballad", "folk", "pop"):  # 桥段抽掉鼓，制造段落对比
        drums = [d for d in drums
                 if _section_of(sections, int(d["start"] / bpb))["name"] != "bridge"]

    return {
        "spec": {k: spec[k] for k in ("version", "title", "description", "genre", "mood",
                                      "key", "mode", "bpm", "beats_per_bar", "provider",
                                      "lyrics_source", "lead_voice", "sing",
                                      "seed", "family_seeds") if k in spec},
        "bpm": bpm, "beats_per_bar": bpb, "total_beats": total_beats,
        "duration_sec": round(total_beats * 60.0 / bpm + 1.2, 2),
        "key_pc": key_pc, "scale": scale, "sections": sections,
        "chords": chords_by_bar, "melody": melody, "bass": bass,
        "pads": pads, "arps": arps, "stabs": stabs, "drums": drums,
    }


# ---------------------------------------------------------------- 旋律
def _melody(spec, sections, chords_by_bar, key_pc, scale, n, bpb):
    """旋律引擎。随机源按"段落家族"独立播种：
    family_seeds = {"chorus": x} 改变副歌种子即可只重写副歌（换副歌功能）。"""
    seed = spec.get("seed") or 0
    fam_seeds = spec.get("family_seeds") or {}
    rng = random.Random(seed ^ 0x9E10)
    family_state = {}

    def state_for(fam, pool):
        if fam not in family_state:
            frng = random.Random((seed ^ 0x9E10 ^ zlib.crc32(fam.encode())
                                  ^ (fam_seeds.get(fam, 0) & 0xFFFFFFFF)) & 0xFFFFFFFF)
            family_state[fam] = {"pattern": frng.choice(pool),
                                 "arc_off": frng.randint(-1, 1)}
        return family_state[fam]

    is_rap = spec["genre"] == "rap"

    def pool_for(fam):
        return RAP_RHYTHM[fam] if is_rap else RHYTHM_POOL[fam]

    events = []
    prev_deg = None
    tonic_pitch = theory.degree_pitch(key_pc, scale, n)

    for sec in sections:
        name = sec["name"]
        if name == "intro":
            continue
        if name == "outro":
            for i in range(sec["bars"]):
                deg = n if i % 2 == 0 else n - 1
                events.append({"pitch": theory.degree_pitch(key_pc, scale, deg),
                               "start": sec["start_beat"] + i * bpb, "dur": bpb * 0.92, "vel": 68})
                prev_deg = deg
            continue

        fam = FAMILY_OF[name]
        pool = pool_for(fam)
        state = state_for(fam, pool)
        base_deg = n + (0 if fam == "verse" else 1 if fam == "pre" else 2 if fam == "chorus" else 0)

        for bi in range(sec["bars"]):
            bar = sec["start_bar"] + bi
            chord = chords_by_bar[bar]
            tone_degs = set(theory.chord_tone_degrees(key_pc, scale, chord["root"], chord["quality"]))
            pattern = state["pattern"] if bi % 2 == 0 or rng.random() < 0.5 else rng.choice(pool)
            pos4 = bi % 4
            # 轮廓：乐句正弦弧线 + 段落基础位 + 问答起伏（偶数小节上扬、奇数小节回应）
            target = base_deg + round(2.5 * math.sin(math.pi * (pos4 + 0.5) / 4)) + state["arc_off"]
            target += 1 if bi % 2 == 0 else -1

            slots = list(pattern)
            is_last_bar = bi == sec["bars"] - 1
            if is_last_bar:
                slots = [s for s in slots if s[0] < bpb - 1]
                slots.append((bpb - 1, 1.0))

            for off, dur in slots:
                strong = off % 2 == 0
                if not strong and rng.random() < 0.12:
                    continue  # 弱拍偶尔休止，给旋律留气口
                lo = (prev_deg - 3) if prev_deg is not None else base_deg - 1
                hi = (prev_deg + 3) if prev_deg is not None else base_deg + 3
                cands = [d for d in range(lo, hi + 1)]
                if strong:
                    strong_cands = [d for d in cands if d in tone_degs]
                    if strong_cands:
                        cands = strong_cands
                cands = [theory.snap_degree_to_pitch_range(d, key_pc, scale, MELODY_LO, MELODY_HI)
                         for d in cands]
                pick = min(cands, key=lambda d: (abs(d - target), rng.random()))
                prev_deg = pick
                vel = (70 if is_rap else 82) + FAMILY_VEL[fam] + (6 if off == 0 else 0) + rng.randint(-3, 3)
                if is_last_bar and off == bpb - 1:
                    pick_deg_pitch = tonic_pitch
                    vel += 4
                else:
                    pick_deg_pitch = theory.degree_pitch(key_pc, scale, pick)
                events.append({"pitch": int(pick_deg_pitch),
                               "start": round(sec["start_beat"] + bi * bpb + off, 3),
                               "dur": round(dur, 3), "vel": int(_clamp(vel, 60, 112))})
    return sorted(events, key=lambda e: (e["start"], e["pitch"]))


def _rap_flow(spec, melody, key_pc, scale, n):
    """说唱 flow：主歌念唱化——音高压进主音附近窄带（talk-sing），
    力度收平、减少大跳；副歌保留多一点旋律性（带宽翻倍）。"""
    tonic_pitch = theory.degree_pitch(key_pc, scale, n)
    sections = spec["structure"]
    start_of = {}
    bar_cursor = 0
    for sec in sections:
        for i in range(sec["bars"]):
            start_of[bar_cursor + i] = sec["name"]
        bar_cursor += sec["bars"]

    def band_for(pitch, sec_name):
        half = 5 if sec_name in ("chorus", "prechorus") else 3  # 半音半宽
        center = tonic_pitch + (2 if sec_name in ("chorus", "prechorus") else 0)
        while pitch > center + half:
            pitch -= 12
        while pitch < center - half:
            pitch += 12
        return int(_clamp(pitch, tonic_pitch - 7, tonic_pitch + 10))

    out = []
    for e in melody:
        sec_name = start_of.get(int(e["start"] // spec.get("beats_per_bar", 4)), "verse")
        pitch = band_for(e["pitch"], sec_name)
        vel = e["vel"] - (10 if sec_name not in ("chorus", "prechorus") else 4)
        out.append(dict(e, pitch=pitch, vel=int(_clamp(vel, 52, 104))))
    return sorted(out, key=lambda e: (e["start"], e["pitch"]))


# ---------------------------------------------------------------- 贝斯
def _bass(spec, sections, chords_by_bar, total_bars, bpb, rng):
    genre = spec["genre"]
    events = []
    drums_start = _first_drums_bar(spec, sections)
    for bar in range(total_bars):
        if bar < drums_start:
            continue
        chord = chords_by_bar[bar]
        pitches = chord["pitches"]

        def _fold(p):
            """折回贝斯音区：>52 再降八度，<34 升回来。"""
            p -= 12
            if p > BASS_HI + 2:
                p -= 12
            return max(BASS_LO, p)

        tones = {
            "root": _fold(pitches[0]),
            "third": _fold(pitches[1] if len(pitches) > 1 else pitches[0] + 4),
            "fifth": _fold(pitches[2] if len(pitches) > 2 else pitches[0] + 7),
            "oct": min(BASS_HI, pitches[0]),
            "approach": None,
        }
        next_chord = chords_by_bar[bar + 1] if bar + 1 < total_bars else None
        boost = 4 if _section_of(sections, bar)["name"] in ("chorus", "prechorus") else 0

        if genre == "jazz":
            slots = [(b, 0.9, k) for b, k in
                     zip((0, 1, 2, 3), ("root", "third", "fifth", "approach"))]
        else:
            slots = BASS_PATTERNS[genre]
        for off, dur, kind in slots:
            if kind == "approach":
                if next_chord is None:
                    pitch = tones["root"]
                else:
                    pitch = _clamp(next_chord["pitches"][0] - 13, BASS_LO, BASS_HI)
            else:
                pitch = tones[kind]
            vel = 76 + boost + (8 if off == 0 else 0) + rng.randint(-3, 3)
            events.append({"pitch": int(pitch),
                           "start": round(bar * bpb + off, 3), "dur": round(dur, 3),
                           "vel": int(_clamp(vel, 50, 110))})
    return sorted(events, key=lambda e: e["start"])


# ---------------------------------------------------------------- 和声垫 / 琶音 / 键琴
def _pads(chords_by_bar, bpb, genre):
    if genre == "rap":
        # 说唱：和声垫隔小节铺、更暗更轻，给 808 和鼓让位
        return [{"pitches": list(c["pitches"]), "start": c["bar"] * bpb,
                 "dur": bpb * 1.9, "vel": 40}
                for c in chords_by_bar if c["bar"] % 2 == 0]
    vel = 58 if genre in ("ballad", "guofeng") else 50
    return [{"pitches": list(c["pitches"]), "start": c["bar"] * bpb, "dur": bpb * 0.98, "vel": vel}
            for c in chords_by_bar]


def _arps(spec, chords_by_bar, bpb):
    steps = 8
    seq3 = [0, 1, 2, 1, 0, 1, 2, 1]
    seq4 = [0, 1, 2, 3, 2, 1, 0, 1]
    events = []
    for c in chords_by_bar:
        tones = c["pitches"]
        seq = seq4 if len(tones) >= 4 else seq3
        for i in range(steps):
            pitch = tones[seq[i] % len(tones)] + 12
            events.append({"pitch": int(pitch), "start": round(c["bar"] * bpb + i * (bpb / steps), 3),
                           "dur": round(bpb / steps * 0.85, 3), "vel": 52 + (6 if i % 4 == 0 else 0)})
    return events


def _stabs(spec, chords_by_bar, bpb):
    hits = [(0.5, 0.5), (1.5, 0.5), (2.5, 0.5), (3.5, 0.5)] if spec["genre"] == "lofi" \
        else [(0.5, 0.7), (2.25, 0.5)]
    vel = 48 if spec["genre"] == "lofi" else 44
    events = []
    for c in chords_by_bar:
        for off, dur in hits:
            events.append({"pitches": list(c["pitches"]), "start": round(c["bar"] * bpb + off, 3),
                           "dur": dur, "vel": vel})
    return events


# ---------------------------------------------------------------- 鼓组
def _first_drums_bar(spec, sections):
    if spec["genre"] in ("electronic", "lofi", "rap"):
        return 0  # 说唱的 beat 从头进
    for sec in sections:
        if sec["name"] in ("verse", "chorus"):
            return sec["start_bar"]
    return 0


def _section_of(sections, bar):
    for sec in sections:
        if sec["start_bar"] <= bar < sec["start_bar"] + sec["bars"]:
            return sec
    return sections[-1]


def _drums(spec, sections, total_bars, bpb, rng):
    genre = spec["genre"]
    pattern_key = genre
    if genre == "rap" and spec["bpm"] >= 108:
        pattern_key = "rap_trap"  # 高速走 trap 半拍律动
    pattern = DRUM_PATTERNS.get(pattern_key, DRUM_PATTERNS["pop"])
    vel_adj = DRUM_VEL_ADJ.get(pattern_key, {})
    swing = genre in SWING_GENRES or (genre == "rap" and spec["bpm"] < 108)
    start_bar = _first_drums_bar(spec, sections)
    events = []

    for bar in range(start_bar, total_bars):
        sec = _section_of(sections, bar)
        sec_level = 0.72 if sec["name"] == "intro" else 1.08 if sec["name"] == "chorus" \
            else 0.9 if sec["name"] == "outro" else 1.0
        for dtype, steps in pattern.items():
            base = _DRUM_DEFAULT_VEL[dtype] + vel_adj.get(dtype, 0)
            for step in steps:
                off = step / 4.0
                if swing and step % 4 == 2:
                    off += SWING_BEATS
                vel = base * sec_level * (1.12 if step % 8 == 0 else 1.0)
                vel += rng.randint(-4, 4)
                events.append({"type": dtype, "start": round(bar * bpb + off, 3),
                               "vel": int(_clamp(vel, 20, 127))})
        # 段落末尾军鼓填充
        is_section_end = bar == sec["start_bar"] + sec["bars"] - 1
        next_sec = _section_of(sections, bar + 1) if bar + 1 < total_bars else None
        if is_section_end and next_sec is not None and sec["name"] in ("verse", "prechorus", "bridge") \
                and genre in FILL_GENRES and sec["bars"] >= 4:
            fill_type = "rim" if genre == "guofeng" else "snare"
            for k, step in enumerate((12, 13, 14, 15)):
                vel = (55 + k * 15) * (0.55 if genre == "jazz" else 1.0)
                events.append({"type": fill_type, "start": round(bar * bpb + step / 4.0, 3),
                               "vel": int(_clamp(vel, 20, 120))})
    return sorted(events, key=lambda e: e["start"])


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))
