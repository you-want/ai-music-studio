"""《深夜便利店》概念专辑批量生成 + 自动质检门。

10 首曲目（同一「夜班嗓」Persona，10 种风格视角），每首流程：
  LLM 作曲(歌词要求) → 神经演唱(32步) → 母带 → 对齐 → 分轨 → 入库 → 质检门
质检不合格（时长/响度/对齐/分轨/歌词源/人声能量任一不过）→ 换 seed 重生一次。
"""
import sys, time, json, wave, glob, os
sys.path.insert(0, ".")
import numpy as np

from app.jobs import JobManager
from app.providers import llm
from app.providers.diffrhythm import (provider as dr_provider, build_lyric_lines,
                                      resolve_audio_length, align as neural_align,
                                      separate as neural_separate, master_wav,
                                      lite_score as neural_lite_score,
                                      venv_has_module as neural_has_module)
from app import personas, store as store_mod

PERSONA = "20261031-213737-814770"  # 夜班嗓
jm = JobManager(store=store_mod.Library("output"), config_getter=lambda: llm.load_config())
lib = jm.store

TRACKS = [
    ("卷帘门升起", "pop", "chill", "专辑开场：凌晨的便利店刚开张，卷帘门升起像舞台开幕，货架上灯光次第亮起。副歌钩子「欢迎光临」重复。押 eng/ing 韵。", 20261121),
    ("热咖啡", "ballad", "romantic", "深夜常客与一杯热咖啡的默契，含蓄的温暖。意象：咖啡机蒸汽、杯沿的温度、玻璃窗雾气。副歌钩子「多加一分甜」重复。押 an 韵。", 20261122),
    ("关东煮之夜", "lofi", "chill", "关东煮锅前的治愈系夜晚，慢节奏。意象：萝卜和鱼丸在汤里浮动、夜班店员的哈欠、外卖小哥的眼神。副歌「关东煮之夜」重复。押 i 韵。", 20261123),
    ("夜班说唱", "rap", "energetic", "夜班店员的自我宣言，态度要足但不下流。意象：扫码枪节奏、货架补货、凌晨四点的困。副歌「我不打烊」重复三遍。押 ang 韵。", 20261124),
    ("货架上的星空", "pop", "dreamy", "把货架想象成星空的浪漫视角。意象：矿泉水瓶反光像星星、顶灯是月亮、条形码是星座。副歌「货架上的星空」重复。押 ing/eng 韵。", 20261125),
    ("雨夜配送", "folk", "sad", "雨夜给独居老人送外卖的瞬间的共情。意象：雨点打在保温箱、楼道声控灯、一句谢谢。副歌钩子「雨再大也要送到」重复。押 ao 韵。", 20261126),
    ("收银台圆舞曲", "jazz", "happy", "收银台前的轻快小确幸，三拍子感觉。意象：零钱叮当、扫码声像打击乐、塑料袋响。副歌「叮一声晚安」重复。押 an 韵。", 20261127),
    ("打烊之前", "ballad", "sad", "打烊前的告别与不舍，全专情感最重的一首。意象：关掉的灯一排排熄灭、最后一位客人、 doors 锁上的声音。副歌钩子「打烊之前 再陪我一会」重复。押 ei/ui 韵。", 20261128),
    ("便利店之光", "rock", "epic", "把便利店升华为城市灯塔的燃曲。意象：它是黑夜里唯一的白光、候鸟归巢般的夜归人。副歌「你永远不会熄灭」重复。押 ie/ue 韵。", 20261129),
    ("天亮了", "folk", "happy", "专辑尾声：清晨交班，走出店门迎接日出，与开场呼应收束全篇。意象：晨光洗掉疲惫、豆浆油条、回家路上哼着歌。副歌「天亮了 回家了」重复。押 ao/ia 韵。", 20261130),
]


def wav_rms(path):
    try:
        with wave.open(path, "rb") as w:
            data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)
        return float(np.sqrt((data ** 2).mean())), w.getnframes() / float(w.getframerate() or 1)
    except Exception:
        return 0.0, 0.0


def quality_gate(sid, spec):
    """-> (ok: bool, report: dict)。任一硬指标不过即不合格。"""
    d = lib.song_dir(sid)
    rep = {}
    wav = lib.wav_path(sid)
    rep["rms"], rep["dur"] = wav_rms(wav)
    rep["dur_ok"] = 92 <= rep["dur"] <= 100
    rep["rms_ok"] = 2200 <= rep["rms"] <= 16000
    sp = os.path.join(d, "score.json")
    ev = 0
    if os.path.isfile(sp):
        try:
            ev = len(json.load(open(sp, encoding="utf-8")).get("lyric_events", []))
        except ValueError:
            pass
    rep["align"] = ev
    rep["align_ok"] = ev >= 10
    rep["stems_ok"] = all(os.path.isfile(os.path.join(d, "stem_%s.wav" % n))
                          for n in ("vocal", "harmony", "bass", "drums"))
    vr = 0.0
    vp = os.path.join(d, "stem_vocal.wav")
    if os.path.isfile(vp):
        vr, _ = wav_rms(vp)
    rep["vocal_rms"] = vr
    rep["vocal_ok"] = vr >= 1200
    rep["lyrics_llm"] = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8")).get("lyrics_source") == "llm"
    ok = all([rep["dur_ok"], rep["rms_ok"], rep["align_ok"], rep["stems_ok"],
              rep["vocal_ok"], rep["lyrics_llm"]])
    return ok, rep


def generate_one(track, seed):
    name, genre, mood, req_text, base_seed = track
    req = {"description": "《深夜便利店》概念专辑曲目：%s" % name,
           "genre": genre, "mood": mood, "provider": "diffrhythm",
           "duration_sec": 95, "persona_id": PERSONA,
           "lyrics_req": req_text + " 店名统一叫「深夜便利店」。", "seed": seed}
    req["duration_sec"] = resolve_audio_length(req)
    spec, info = jm._compose(req)
    if spec is None:
        raise RuntimeError("作曲失败: %s" % (info or {}).get("reason"))
    if (info or {}).get("provider", "").startswith("mock"):
        raise RuntimeError("LLM 超时回落 Mock（歌词质量不合格）")
    spec["composer_provider"] = "llm"
    sid = lib.new_song_id()
    lib.prepare(sid)
    dr_provider.generate(req, spec, lib.wav_path(sid),
                         progress_cb=lambda f, n="": print("  %.0f%% %s" % (f * 100, n), flush=True))
    master_wav(lib.wav_path(sid))
    events = None
    if neural_has_module("faster_whisper", wait=True) is True:
        events = neural_align(lib.wav_path(sid),
                              build_lyric_lines(spec, max_sec=resolve_audio_length(req) - 2))
    if neural_has_module("demucs", wait=True) is True:
        neural_separate(lib.wav_path(sid), lib.song_dir(sid))
    import wave as _w
    with _w.open(lib.wav_path(sid), "rb") as w:
        dur = w.getnframes() / float(w.getframerate())
    score = neural_lite_score(spec, dur, lyric_events=events)
    meta = lib.save(sid, spec, score)
    return sid, meta


print("=== 《深夜便利店》概念专辑：10 首，质检门+自动重生成 ===", flush=True)
report = []
for idx, track in enumerate(TRACKS, 1):
    name, genre, mood, _, seed = track
    print("\n=== [%d/10] 《%s》(%s/%s) ===" % (idx, name, genre, mood), flush=True)
    ok, sid, rep, attempts = False, None, {}, 0
    while attempts < 2 and not ok:
        attempts += 1
        try:
            sid, meta = generate_one(track, seed + attempts)
            ok, rep = quality_gate(sid, None)
            rep["title"] = meta["title"]
            print("  质检: %s | %s" % ("PASS" if ok else "FAIL", json.dumps(
                {k: v for k, v in rep.items() if k != "title"}, ensure_ascii=False)), flush=True)
        except Exception as e:  # noqa: BLE001
            print("  异常: %s" % str(e)[:120], flush=True)
            rep = {"error": str(e)[:120]}
            ok = False
        if not ok and attempts < 2:
            print("  ↻ 不合格，换 seed 重新生成…", flush=True)
    report.append({"track": idx, "name": name, "song_id": sid,
                   "ok": ok, "attempts": attempts,
                   **{k: v for k, v in rep.items() if k != "title"}})
    if sid and not ok:
        print("  （两次仍不合格，保留人工复听；继续下一首）", flush=True)

with open("output/album_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=1)
passed = sum(1 for r in report if r["ok"])
print("\n=== 专辑完成：%d/%10s 通过质检 ===" % (passed, len(report)), flush=True)
for r in report:
    print(" [%d] %s 《%s》 %s" % (r["track"], "PASS" if r["ok"] else "FAIL",
                                 r.get("title", r["name"]), r.get("error", "")), flush=True)
