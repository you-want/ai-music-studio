"""离线 Mock 作曲引擎：不联网、零费用、确定性。

理解用户描述里的风格/情绪关键词 -> 选调式与速度 -> 套和声进行 ->
按目标时长拼曲式 -> 填示例歌词 -> 起标题。同一 seed 永远产出同一首。
"""
import hashlib
import random
import re

from .. import schemas
from . import theory

GENRE_KEYWORDS = {
    "pop": ["流行", "pop", "抖音", "神曲", "口水歌", "金曲"],
    "rock": ["摇滚", "rock", "乐队", "电吉他", "朋克", "punk", "金属"],
    "electronic": ["电子", "电音", "edm", "techno", "house", "蹦迪", "夜店", "合成器", "赛博"],
    "ballad": ["抒情", "ballad", "钢琴曲", "慢歌", "情歌", "夜曲", "哄睡"],
    "folk": ["民谣", "folk", "木吉他", "城市", "南方", "姑娘", "远方"],
    "guofeng": ["古风", "国风", "中国风", "水墨", "江湖", "仙侠", "戏曲", "古筝", "长安"],
    "lofi": ["lofi", "lo-fi", "学习", "专注", "熬夜", "chill", "雨声", "咖啡", "耳机"],
    "jazz": ["爵士", "jazz", "蓝调", "blues", "摇摆", "swing", "酒吧", "萨克斯"],
    "rap": ["说唱", "rap", "嘻哈", "hiphop", "hip hop", "freestyle", "punchline",
            " verse", "verse", "陷阱说唱", "trap", "battle", "flow", "mc"],
}
MOOD_KEYWORDS = {
    "happy": ["快乐", "开心", "欢快", "阳光", "愉快", "高兴", "幸福", "happy", "joy", "夏"],
    "sad": ["悲伤", "难过", "伤心", "忧郁", "伤感", "流泪", "失恋", "孤独", "sad", "雨", "夜", "回忆"],
    "chill": ["轻松", "放松", "惬意", "治愈", "温柔", "安静", "chill", "午後", "午后", "慢"],
    "energetic": ["燃", "热血", "活力", "动感", "冲刺", "奋斗", "energetic", "跑", "运动", "战斗"],
    "romantic": ["浪漫", "爱情", "甜蜜", "心动", "喜欢", "表白", "romantic", "爱", "恋"],
    "epic": ["史诗", "大气", "磅礴", "恢弘", "宏大", "英雄", "epic", "战争", "山海"],
}

# 每个风格：速度区间 + 大调/小调可用和声进行（级度, 品质）
PROFILES = {
    "pop": {
        "bpm": (96, 118),
        "major": [[(0, "maj"), (4, "maj"), (5, "min"), (3, "maj")],
                  [(5, "min"), (3, "maj"), (0, "maj"), (4, "maj")]],
        "minor": [[(0, "min"), (5, "maj"), (2, "maj"), (6, "maj")],
                  [(0, "min"), (6, "maj"), (5, "maj"), (4, "min")]],
    },
    "rock": {
        "bpm": (120, 150),
        "major": [[(0, "maj"), (3, "maj"), (4, "maj"), (3, "maj")],
                  [(0, "maj"), (6, "maj"), (3, "maj"), (4, "maj")]],
        "minor": [[(0, "min"), (5, "maj"), (6, "maj"), (0, "min")],
                  [(0, "min"), (3, "maj"), (6, "maj"), (4, "min")]],
    },
    "electronic": {
        "bpm": (120, 132),
        "major": [[(0, "maj"), (5, "min"), (3, "maj"), (4, "maj")]],
        "minor": [[(0, "min"), (5, "maj"), (2, "maj"), (6, "maj")],
                  [(0, "min"), (2, "maj"), (5, "maj"), (6, "maj")]],
    },
    "ballad": {
        "bpm": (68, 84),
        "major": [[(0, "maj"), (4, "maj"), (5, "min"), (3, "maj")],
                  [(5, "maj"), (3, "maj"), (0, "maj"), (4, "maj")]],
        "minor": [[(0, "min"), (3, "maj"), (5, "maj"), (4, "min")],
                  [(0, "min"), (5, "maj"), (2, "maj"), (6, "maj")]],
    },
    "folk": {
        "bpm": (88, 108),
        "major": [[(0, "maj"), (4, "maj"), (5, "min"), (3, "maj")],
                  [(0, "maj"), (3, "maj"), (5, "min"), (4, "maj")]],
        "minor": [[(0, "min"), (5, "maj"), (2, "maj"), (6, "maj")]],
    },
    "guofeng": {
        "bpm": (76, 96),
        "major": [[(0, "maj"), (3, "maj"), (4, "maj"), (0, "maj")],
                  [(0, "maj"), (5, "min"), (4, "maj"), (3, "maj")],
                  [(0, "maj"), (4, "maj"), (5, "min"), (3, "maj")]],
        "minor": [[(0, "min"), (5, "maj"), (3, "maj"), (4, "min")]],
    },
    "lofi": {
        "bpm": (72, 88),
        "major": [[(0, "maj7"), (5, "m7"), (1, "m7"), (4, "maj7")],
                  [(0, "maj7"), (3, "maj7"), (1, "m7"), (4, "7")]],
        "minor": [[(0, "m7"), (2, "maj7"), (5, "maj7"), (1, "m7")]],
    },
    "jazz": {
        "bpm": (104, 132),
        "major": [[(1, "m7"), (4, "maj7"), (0, "maj7"), (0, "maj7")],
                  [(0, "maj7"), (5, "m7"), (1, "m7"), (4, "7")]],
        "minor": [[(0, "m7"), (1, "m7"), (4, "7"), (0, "m7")],
                  [(0, "m7"), (5, "maj7"), (2, "maj7"), (6, "m7")]],
    },
    # 说唱：小调 2-4 和弦循环（loop），hook 段才有和声变化；BPM 82-96 为 boom-bap，
    # >=108 走 trap 半拍律动（编曲器按 BPM 自动切鼓组）
    "rap": {
        "bpm": (82, 96),
        "major": [[(0, "maj"), (1, "min"), (4, "maj"), (3, "maj")]],
        "minor": [[(0, "min"), (5, "maj"), (2, "maj"), (6, "maj")],
                  [(0, "min"), (3, "min"), (0, "min"), (6, "maj")],
                  [(0, "min"), (6, "maj"), (5, "maj"), (6, "maj")]],
    },
}

MOOD_KEY_POOL = {
    "happy": (["C", "G", "D", "F"], "major"),
    "sad": (["A", "D", "E", "B"], "minor"),
    "chill": (["C", "F", "G", "D"], "major"),
    "energetic": (["G", "E", "A", "D"], "major"),
    "romantic": (["F", "C", "G", "Bb"], "major"),
    "epic": (["A", "D", "E", "G"], "minor"),
}

LYRICS = {
    "happy": {
        "verse": ["清晨的光落在窗台", "背包里装满了期待", "风把云朵吹得散开", "脚步轻快像在比赛"],
        "chorus": ["啦啦啦 大声唱出来", "把烦恼都甩在身后", "今天的天空特别蓝", "快乐就是这么简单"],
    },
    "sad": {
        "verse": ["路灯把影子拉得很长", "对话框停在没有发送", "你说的话还在回响", "房间安静得有点凉"],
        "chorus": ["我把想念折成纸船", "漂向你离开的方向", "如果眼泪也会说话", "它一定说了晚安"],
    },
    "chill": {
        "verse": ["午后三点 阳光刚好", "咖啡冒着小小的泡", "猫在沙发角落睡着", "时间跟着慢了下来"],
        "chorus": ["就这样吧 挺好的", "不用急着去哪里", "风吹过来的时候", "整个世界都温柔"],
    },
    "energetic": {
        "verse": ["心跳是擂动的鼓", "血液里烧着一团火", "跌倒了就当热身", "终点线在前面闪"],
        "chorus": ["冲吧 别回头看", "世界会为热血让路", "这一刻我们发光", "让全世界都听见"],
    },
    "romantic": {
        "verse": ["你笑起来像春天", "奶茶半糖刚刚好", "电影散场的人潮里", "我只想牵你的手"],
        "chorus": ["月亮替我说爱你", "晚风也在偷偷练习", "如果心跳有声音", "你一定早就听见"],
    },
    "epic": {
        "verse": ["风从远古山谷吹来", "星辰在大地上排列", "先祖的火种未熄灭", "少年踏上了长路"],
        "chorus": ["让号角响彻云霄", "让旗帜迎风燃烧", "千山万水都走过", "名字刻进了传说"],
    },
}

TITLE_NOUNS = ["晚风", "星河", "夏夜", "候鸟", "纸飞机", "灯塔", "橘子海", "野草",
               "银河", "六月", "雨后", "长街", "孤岛", "初雪", "蝉鸣", "南风"]
TITLE_SUFFIX = ["来信", "练习曲", "物语", "之后", "电台", "备忘录", "漫游记", "小夜曲", "碎片"]

# 说唱歌词：双句押韵、口语化带态度（genre==rap 时优先于情绪模板）
RAP_LYRICS = {
    "energetic": {
        "verse": ["麦在手里 节奏在胸口", "质疑声再多我不回头", "一步一个脚印往前走", "终点之前绝不停手"],
        "chorus": ["把音量推到最大", "让怀疑的人都听见", "这条路是我自己选的", "跪着也要走完它"],
    },
    "sad": {
        "verse": ["城市把梦想磨出了茧", "口袋里装着没寄的信", "霓虹照亮每个人的脸", "没人看见我的夜晚"],
        "chorus": ["把心事写进verse里", "节奏替我说不出的委屈", "如果生活是一场雨", "我就在雨里继续"],
    },
    "chill": {
        "verse": ["下午四点 阳光斜进窗", "节拍慢得像猫的呼吸", "口袋里没有大事发生", "今天的计划就是没有计划"],
        "chorus": ["慢慢来 不着急", "生活总会给答案", "耳机里放着beat", "世界只剩半拍懒"],
    },
    "happy": {
        "verse": ["早八的闹钟也叫不醒", "我哼着beat出门的兴致", "红绿灯为我打拍子", "今天运气踩在鼓点上"],
        "chorus": ["嘿 把烦恼调成静音", "快乐今天免费供应", "脚步踩着轻快的鼓点", "好心情循环播放不停"],
    },
}


def _seed_of(req):
    raw = "%s|%s|%s|%s|%s|%s" % (
        req.get("description", ""), req.get("genre"), req.get("mood"),
        req.get("bpm"), req.get("duration_sec"), req.get("seed"),
    )
    return int(hashlib.md5(raw.encode("utf-8")).hexdigest()[:12], 16)


def _detect(text, table, default):
    """按关键词计数选出最匹配项；无命中返回 (default, False)。"""
    scores = {}
    low = text.lower()
    for key, words in table.items():
        s = sum(low.count(w.lower()) for w in words)
        if s:
            scores[key] = s
    if not scores:
        return default, False
    return max(scores.items(), key=lambda kv: kv[1])[0], True


def _degree_to_symbol(key_pc, scale, degree, quality):
    root_pc = theory.degree_root_pc(key_pc, scale, degree)
    return theory.chord_symbol(root_pc, quality)


def _build_structure(target_bars, genre, mood, rng):
    """按目标小节数拼曲式：intro + (主歌/副歌交替) + outro，凑满目标。"""
    secs = []
    remaining = target_bars
    intro = 2 if target_bars <= 30 else 4
    secs.append(["intro", intro])
    remaining -= intro
    outro = 2 if remaining >= 10 else 0
    remaining -= outro

    i = 0
    if target_bars <= 8:  # ≈20秒以内：前奏+副歌即可
        pattern = ["chorus"]
    else:
        pattern = ["verse", "chorus"]
    while remaining >= 4:
        name = pattern[i % len(pattern)]
        cap = 8
        if i < len(pattern) - 1:
            cap = min(6, remaining - 4)  # 给后面的段落至少留 4 小节
        bars = max(1, min(cap, remaining))
        secs.append([name, bars])
        remaining -= bars
        i += 1
    if remaining > 0:  # 零头给最后一个副歌
        tail = None
        for s in reversed(secs):
            if s[0] == "chorus" and s[1] < 16:
                tail = s
                break
        (tail or secs[-1])[1] += remaining
    if outro:
        secs.append(["outro", outro])
    if not any(n == "chorus" for n, _ in secs):
        secs.insert(1 if len(secs) > 1 else 0, ["chorus", max(4, min(8, target_bars // 2))])
    return [(name, bars) for name, bars in secs]


def _title_for(req, mood, rng):
    if req.get("title"):
        return req["title"]
    desc = (req.get("description") or "").strip()
    clean = re.sub(r"[\s，。,.!?！？、]", "", desc)
    if 2 <= len(clean) <= 10:
        return clean
    if desc:
        for noun in TITLE_NOUNS:
            if noun in desc:
                return noun + rng.choice(TITLE_SUFFIX)
    return rng.choice(TITLE_NOUNS) + rng.choice(TITLE_SUFFIX)


def compose(req):
    """Compose 请求 -> SongSpec（provider='mock'）。"""
    req = schemas.parse_request(req)
    seed = req["seed"] if req["seed"] is not None else _seed_of(req)
    rng = random.Random(seed)

    desc = req["description"]
    # 写词要求也参与风格/情绪识别（例：要求里写"失恋" -> sad）
    detect_text = (desc + " " + (req.get("lyrics_req") or "")).strip()
    # 显式选择的风格/情绪必须生效；仅 auto 时按描述关键词识别
    # （修复：此前关键词识别总是覆盖用户在 UI 里点的 genre/mood）
    if req["genre"] != "auto":
        genre = req["genre"]
    else:
        genre, _ = _detect(detect_text, GENRE_KEYWORDS, rng.choice(list(PROFILES.keys())))
    if req["mood"] != "auto":
        mood = req["mood"]
    else:
        mood, _ = _detect(detect_text, MOOD_KEYWORDS, rng.choice(list(schemas.MOODS)))

    profile = PROFILES[genre]
    if genre == "guofeng":
        mode = "chinese"
    elif req["mode"]:
        mode = req["mode"]
    elif genre == "rap" and mood in ("happy", "romantic"):
        mode = "major"
    elif mood in ("sad", "epic") or genre == "rap":
        mode = "minor"  # 说唱默认小调，除明快情绪
    else:
        mode = "major"
    scale = theory.SCALES[mode]

    keys_pool, default_mode = MOOD_KEY_POOL[mood]
    if req["key"] is not None:
        key_pc = req["key"]
    elif mode == default_mode:
        key_pc = theory.note_to_pc(rng.choice(keys_pool))
    else:
        key_pc = rng.randrange(12)

    bpm = req["bpm"] or rng.randint(*profile["bpm"])
    target_bars = schemas.bars_for_duration(req["duration_sec"], bpm)
    structure = _build_structure(target_bars, genre, mood, rng)

    progs = profile["minor" if mode in ("minor", "minor_pentatonic") else "major"]
    prog = rng.choice(progs)
    tonic_quality = "min" if mode in ("minor", "minor_pentatonic") else "maj"
    tonic = theory.chord_symbol(key_pc, tonic_quality)

    lines = (RAP_LYRICS.get(mood) if genre == "rap" else None) or LYRICS[mood]
    sections = []
    for si, (name, bars) in enumerate(structure):
        next_name = structure[si + 1][0] if si + 1 < len(structure) else None
        chords = []
        for i in range(bars):
            if name == "outro" and i == bars - 1:
                chords.append(tonic)  # 终止式：落主和弦
            elif next_name in ("chorus", "prechorus") and i == bars - 1 \
                    and name in ("verse", "prechorus", "bridge"):
                chords.append(_degree_to_symbol(key_pc, scale, 4, "maj"))  # 半终止→属和弦，推动进副歌
            else:
                deg, quality = prog[(i) % len(prog)]
                if name == "bridge":
                    deg = deg + 2  # 桥段整体上移两度（调性对比），回副歌时豁然开朗
                chords.append(_degree_to_symbol(key_pc, scale, deg, quality))
        lyrics = []
        if name == "verse":
            src = lines["verse"]
            lyrics = [src[i % len(src)] for i in range(bars)]
        elif name == "chorus":
            src = lines["chorus"]
            lyrics = [src[i % len(src)] for i in range(bars)]
        elif name == "prechorus":
            src = lines["verse"][2:] + lines["chorus"][:1]
            lyrics = [src[i % len(src)] for i in range(bars)]
        elif name == "bridge":
            src = list(reversed(lines["chorus"]))
            lyrics = [src[i % len(src)] for i in range(bars)]
        else:
            lyrics = [""] * bars
        sections.append({"name": name, "bars": bars, "chords": chords, "lyrics": lyrics})

    spec = {
        "version": 1,
        "title": _title_for(req, mood, rng),
        "description": desc,
        "genre": genre,
        "mood": mood,
        "key": theory.NOTE_NAMES[key_pc],
        "mode": mode,
        "bpm": bpm,
        "beats_per_bar": 4,
        "structure": sections,
        "provider": "mock",
        "lyrics_source": "mock-template",
        "seed": seed,
        "warnings": [],
    }
    if req.get("lead_voice"):
        spec["lead_voice"] = req["lead_voice"]
    return spec


def mock_lyrics_text(req):
    """离线 AI 写词兜底：按情绪模板 + 描述钩子行拼一段带标记的歌词。"""
    req = schemas.parse_request(req)
    seed = req["seed"] if req["seed"] is not None else _seed_of(req)
    rng = random.Random(seed ^ 0xA11CE)
    detect_text = ((req["description"] or "") + " " + (req.get("lyrics_req") or "")).strip()
    if req["mood"] != "auto":
        mood = req["mood"]
    else:
        mood, _ = _detect(detect_text, MOOD_KEYWORDS, rng.choice(list(schemas.MOODS)))
    if req["genre"] != "auto":
        genre = req["genre"]
    else:
        genre, _ = _detect(detect_text, GENRE_KEYWORDS, "pop")
    tpl = (RAP_LYRICS.get(mood) if genre == "rap" else None) or LYRICS[mood]

    def shuffled(key):
        lines = list(tpl[key])
        rng.shuffle(lines)
        return lines

    verse1, verse2 = shuffled("verse"), shuffled("verse")
    chorus = shuffled("chorus")
    hook = re.sub(r"[\s，。,.!?！？、]", "", req["description"] or "")
    if 2 <= len(hook) <= 8:
        chorus[0] = hook  # 用用户描述做副歌钩子句
    bridge = [chorus[2], chorus[3]]
    return "[Verse]\n%s\n\n[Chorus]\n%s\n\n[Verse]\n%s\n\n[Chorus]\n%s\n\n[Bridge]\n%s\n\n[Chorus]\n%s" % (
        "\n".join(verse1), "\n".join(chorus), "\n".join(verse2[:2] + verse1[:2]),
        "\n".join(chorus), "\n".join(bridge), "\n".join(chorus))
