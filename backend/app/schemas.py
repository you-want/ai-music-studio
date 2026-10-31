"""Compose 请求与 SongSpec 的规范化、校验与修复。

所有 Provider（mock / llm / 未来的付费 API）产出的高层乐谱都必须经过
validate_and_repair_spec()：这是 LLM 输出的"护栏"，坏数据在这里被
钳制、修复，修不了就抛 SpecError 由上层回落 Mock。
"""
import re

from .music import theory

GENRES = ["pop", "rock", "electronic", "ballad", "folk", "guofeng", "lofi", "jazz", "rap"]
MOODS = ["happy", "sad", "chill", "energetic", "romantic", "epic"]
SECTION_NAMES = ["intro", "verse", "prechorus", "chorus", "bridge", "outro"]
PROVIDERS = ["auto", "mock", "llm", "diffrhythm"]  # diffrhythm = 神经歌声（音频成品类）
MODES = list(theory.SCALES.keys())

LEAD_VOICES = ["synth_lead", "square_lead", "piano", "pluck", "keys", "vocal_hum"]
LEAD_VOICE_LABELS_ZH = {
    "synth_lead": "合成主音", "square_lead": "方波电子", "piano": "钢琴",
    "pluck": "拨弦(古筝)", "keys": "电钢", "vocal_hum": "人声哼唱",
}

GENRE_LABELS_ZH = {
    "pop": "流行", "rock": "摇滚", "electronic": "电子", "ballad": "抒情钢琴",
    "folk": "民谣", "guofeng": "国风", "lofi": "Lo-Fi", "jazz": "爵士", "rap": "说唱",
}
MOOD_LABELS_ZH = {
    "happy": "快乐", "sad": "悲伤", "chill": "放松", "energetic": "燃",
    "romantic": "浪漫", "epic": "史诗",
}
SECTION_LABELS_ZH = {
    "intro": "前奏", "verse": "主歌", "prechorus": "预副歌", "chorus": "副歌",
    "bridge": "桥段", "outro": "尾声",
}

DEFAULTS = {
    "duration_sec": 30.0,
    "bpm": None,
    "beats_per_bar": 4,
}


class SpecError(ValueError):
    """SongSpec 无法修复时抛出。"""


def _as_int(v, default=None):
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return default


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def bars_for_duration(duration_sec, bpm, beats_per_bar=4):
    """目标时长 -> 目标小节数。"""
    if bpm <= 0:
        bpm = 100
    return max(4, int(round(duration_sec * bpm / 60.0 / beats_per_bar)))


def parse_request(raw):
    """把 /api/compose 请求体规范化；非法值尽量纠正而不是报错。"""
    raw = raw or {}
    req = {
        "description": str(raw.get("description") or "").strip(),
        "genre": raw.get("genre") if raw.get("genre") in GENRES else "auto",
        "mood": raw.get("mood") if raw.get("mood") in MOODS else "auto",
        "bpm": _clamp(_as_int(raw.get("bpm"), 0) or 0, 0, 240) or None,
        "duration_sec": _clamp(float(_as_int(raw.get("duration_sec"), 0) or 0) or DEFAULTS["duration_sec"], 10.0, 180.0),
        "key": None,
        "mode": raw.get("mode") if raw.get("mode") in MODES else None,
        "provider": raw.get("provider") if raw.get("provider") in PROVIDERS else "auto",
        "title": str(raw.get("title") or "").strip() or None,
        "seed": _as_int(raw.get("seed")),
        "lyrics": str(raw.get("lyrics") or "").strip() or None,
        "lead_voice": raw.get("lead_voice") if raw.get("lead_voice") in LEAD_VOICES else None,
        # tri-state: None=用户没表态 -> 默认演唱；True/False=显式开关
        "sing": None if raw.get("sing") is None else bool(raw.get("sing")),
        # 神经引擎参考音频：服务端生成的文件名（只允许安全字符）
        "ref_audio": sanitize_filename(raw.get("ref_audio")),
        # 神经引擎完整版 4m45s（需 full 权重）
        "full_song": bool(raw.get("full_song")),
        # 写词要求（主题/故事/必须出现的词/押韵），AI 写词与作曲共同遵循
        "lyrics_req": _clip_text(raw.get("lyrics_req"), 200),
    }
    try:
        req["key"] = theory.note_to_pc(raw["key"]) if raw.get("key") else None
    except ValueError:
        req["key"] = None
    return req


def sanitize_filename(name):
    r"""参考音频等上传文件名的白名单净化：非 [\w.-] 字符全部剔除。"""
    if not name:
        return None
    safe = "".join(c for c in str(name) if c.isalnum() or c in "-_.")
    return safe[:80] or None


def _clip_text(v, limit):
    """自由文本字段：去空白、限长，空串归一为 None。"""
    if not v:
        return None
    text = re.sub(r"\s+", " ", str(v)).strip()
    return text[:limit] or None


def validate_and_repair_spec(raw, req):
    """校验/修复 SongSpec。返回 (spec, warnings)。修不了抛 SpecError。"""
    if not isinstance(raw, dict):
        raise SpecError("spec 不是对象")
    warnings = []

    title = str(raw.get("title") or "").strip() or (req.get("title") or "未命名小曲")
    if len(title) > 40:
        title = title[:40]
        warnings.append("标题过长已截断")

    # 调性 / 调式
    try:
        key_pc = theory.note_to_pc(raw.get("key") or "C")
    except ValueError:
        key_pc = 0
        warnings.append("调性非法，回落 C")
    mode = raw.get("mode") if raw.get("mode") in MODES else None
    if mode is None:
        mode = "chinese" if raw.get("genre") == "guofeng" else "major"
        warnings.append("调式缺失，按 %s 处理" % mode)

    genre = raw.get("genre") if raw.get("genre") in GENRES else "pop"
    mood = raw.get("mood") if raw.get("mood") in MOODS else "chill"
    bpm = _clamp(_as_int(raw.get("bpm"), 0) or 0, 60, 184)
    if bpm == 0:
        bpm = 100
    beats_per_bar = 4

    # 结构
    structure = raw.get("structure")
    if not isinstance(structure, list) or not structure:
        raise SpecError("structure 缺失或为空")
    fixed = []
    for i, sec in enumerate(structure):
        if not isinstance(sec, dict):
            raise SpecError("structure[%d] 不是对象" % i)
        name = sec.get("name") if sec.get("name") in SECTION_NAMES else "verse"
        bars = _clamp(_as_int(sec.get("bars"), 0) or 0, 1, 16)
        chords_raw = sec.get("chords") if isinstance(sec.get("chords"), list) else []
        chords = []
        for c in chords_raw:
            try:
                root, q = theory.parse_chord(c)
                chords.append(theory.chord_symbol(root, q))
            except ValueError:
                warnings.append("忽略非法和弦 %r" % (c,))
        if not chords:
            tonic = theory.chord_symbol(key_pc, "min" if mode in ("minor", "minor_pentatonic") else "maj")
            chords = [tonic]
            warnings.append("section %s 无有效和弦，用主和弦填充" % name)
        lyrics = [str(x) for x in sec.get("lyrics", []) if isinstance(x, (str, int))] \
            if isinstance(sec.get("lyrics"), list) else []
        fixed.append({"name": name, "bars": bars, "chords": chords, "lyrics": lyrics})

    target = bars_for_duration(req.get("duration_sec", 30), bpm, beats_per_bar)
    fixed = _fit_structure(fixed, target)
    for sec in fixed:
        while len(sec["chords"]) < sec["bars"]:
            sec["chords"].append(sec["chords"][-1])
        sec["chords"] = sec["chords"][: sec["bars"]]
        while len(sec["lyrics"]) < sec["bars"]:
            sec["lyrics"].append("")
        sec["lyrics"] = sec["lyrics"][: sec["bars"]]

    spec = {
        "version": 1,
        "title": title,
        "description": str(raw.get("description") or req.get("description") or ""),
        "genre": genre,
        "mood": mood,
        "key": theory.NOTE_NAMES[key_pc],
        "mode": mode,
        "bpm": bpm,
        "beats_per_bar": beats_per_bar,
        "structure": fixed,
        "provider": raw.get("provider") or "mock",
        "lyrics_source": raw.get("lyrics_source") or "unknown",
        "seed": _as_int(raw.get("seed")),
        "warnings": warnings,
    }
    return spec, warnings


def _fit_structure(sections, target_bars):
    """把结构增删到目标小节数（优先砍/加尾声与副歌）。"""
    total = sum(s["bars"] for s in sections)
    while total > target_bars:
        victim = None
        for s in reversed(sections):
            if s["bars"] > 2:
                victim = s
                break
        if victim is None:
            break
        victim["bars"] -= 1
        total -= 1
    while total < target_bars:
        grow = None
        for s in reversed(sections):
            if s["name"] in ("chorus", "outro", "verse") and s["bars"] < 16:
                grow = s
                break
        if grow is not None:
            grow["bars"] += 1
            total += 1
        else:
            # 所有 section 都到上限：复制一段副歌
            donor = None
            for s in reversed(sections):
                if s["name"] in ("chorus", "verse"):
                    donor = s
                    break
            donor = donor or sections[-1]
            add = min(8, target_bars - total)
            sections.append({"name": "chorus", "bars": add,
                             "chords": list(donor["chords"]), "lyrics": list(donor["lyrics"])})
            total += add
    return sections


# ---------------------------------------------------------------- 自定义歌词
# 进阶模式：用户粘贴带段落标记的歌词，如
#   [Verse]\n歌词...\n[Chorus]\n歌词...   （支持中英文标记）
SECTION_MARKERS = {
    "verse": ("verse", "主歌"),
    "chorus": ("chorus", "副歌"),
    "prechorus": ("prechorus", "预副歌", "导歌"),
    "bridge": ("bridge", "桥段", "桥"),
    "intro": ("intro", "前奏"),
    "outro": ("outro", "尾声"),
}
_MARKER_RE = re.compile(r"^[\[【]\s*([^\]】]{1,16})\s*[\]】]$")


def parse_lyrics_text(text):
    """歌词文本 -> {section_name: [行, ...]}；无标记的行进 'auto' 桶。"""
    buckets = {}
    current = "auto"
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _MARKER_RE.match(line)
        if m:
            word = m.group(1).strip().lower()
            current = "auto"
            for key, aliases in SECTION_MARKERS.items():
                if any(a in word for a in aliases):
                    current = key
                    break
            continue
        buckets.setdefault(current, []).append(line)
    return buckets


def apply_lyrics(spec, text):
    """把用户歌词按段落覆盖到 spec.structure 的 lyrics 上。

    有标记 -> 对应段落循环填入；无标记 -> 自动分配到演唱段落；
    前奏/尾声始终为空；某段落没有用户歌词时保留原有（模板/LLM）歌词。
    """
    buckets = parse_lyrics_text(text)
    singing = ("verse", "prechorus", "chorus", "bridge")
    has_any = any(buckets.get(k) for k in list(SECTION_MARKERS) + ["auto"])
    if not has_any:
        return spec
    # 结构里不存在的段落（如短曲没有主歌），其歌词并入自动池，保证不丢失
    present = {sec["name"] for sec in spec["structure"]}
    spilled = []
    for name in ("verse", "prechorus", "bridge", "chorus"):
        if buckets.get(name) and name not in present:
            spilled.extend(buckets[name])
    auto = buckets.get("auto") or []
    for sec in spec["structure"]:
        name = sec["name"]
        if name not in singing:
            sec["lyrics"] = [""] * sec["bars"]
            continue
        lines = (buckets.get(name) or []) + spilled
        if not lines:
            lines = auto
        if not lines:
            continue  # 保留引擎生成的歌词
        sec["lyrics"] = [lines[i % len(lines)] for i in range(sec["bars"])]
    spec["lyrics_source"] = "custom"
    return spec
