"""歌词时间轴强制对齐子进程（跑在 DiffRhythm/.venv-neural，faster-whisper）。

用途：DiffRhythm 是"软对齐"生成，唱出来的词与 Karaoke 高亮有偏差。
本脚本用本地 Whisper 转写生成的歌曲（词级时间戳），再把已知歌词
单调对齐到转写字符时间轴上，反写出每句歌词的真实起止时间。

用法： python align_runner.py <job.json>
job.json: {"wav": "...", "lines": [[12.6, "歌词"], ...], "model": "small", "out": "..."}
stdout:   {"type":"done","events":[[start,end,text,confidence],...]} / {"type":"error",...}

纯算法部分 align_chars_to_lines() 与 Whisper 解耦，可独立单测。
"""
import json
import sys

# ---------------- 纯算法：单调字符对齐 ----------------

GAP = -0.6          # 跳过歌词字/转写字的罚分
MATCH = 1.0
MISMATCH = -0.4


def align_chars_to_lines(lines, char_timeline):
    """把歌词行对齐到字符时间轴。

    lines: [[est_sec, text], ...]（按时间排序）
    char_timeline: [(char, start_sec, end_sec), ...]（Whisper 转写，按时间排序）
    返回: [[start, end, text, confidence], ...]，confidence<0.35 的行 start/end
    回落到 LRC 估计值。
    """
    lyric_chars = []  # (line_idx, char)
    for li, (_t, text) in enumerate(lines):
        for ch in str(text):
            if not ch.isspace():
                lyric_chars.append((li, ch))

    m, n = len(lyric_chars), len(char_timeline)
    if m == 0 or n == 0:
        return _fallback(lines, char_timeline)

    # Needleman-Wunsch（全局单调对齐），带时间带约束防止远距离误配
    est = [t for t, _ in lines]
    total = est[-1] if est and est[-1] > 0 else 1.0
    NEG = float("-inf")
    dp = [[NEG] * (n + 1) for _ in range(m + 1)]
    bt = [[0] * (n + 1) for _ in range(m + 1)]  # 1=diag 2=up(跳过歌词字) 3=left(跳过转写字)
    dp[0][0] = 0.0
    for i in range(m + 1):
        for j in range(n + 1):
            if dp[i][j] == NEG:
                continue
            # diag: 对齐 lyric_chars[i] 与 char_timeline[j]
            if i < m and j < n:
                li, ch = lyric_chars[i]
                # 时间带：该行估计时间 ±40% 全曲，超带宽只允许 gap
                band_ok = True
                if est:
                    expect = est[min(li, len(est) - 1)]
                    tmid = (char_timeline[j][1] + char_timeline[j][2]) / 2
                    span = max(total, 1.0)
                    if abs(tmid - expect) > 0.45 * span:
                        band_ok = False
                s = MATCH if (band_ok and ch == char_timeline[j][0]) else MISMATCH
                v = dp[i][j] + s
                if v > dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = v
                    bt[i + 1][j + 1] = 1
            if i < m:  # up: 歌词字没有对应转写（模型没唱/唱糊了）
                if dp[i][j] + GAP > dp[i + 1][j]:
                    dp[i + 1][j] = dp[i][j] + GAP
                    bt[i + 1][j] = 2
            if j < n:  # left: 转写字没有对应歌词（气声/和声/改词）
                if dp[i][j] + GAP > dp[i][j + 1]:
                    dp[i][j + 1] = dp[i][j] + GAP
                    bt[i][j + 1] = 3

    # 回溯：记录每个歌词字匹配到的转写字索引（-1 = 未匹配）
    char_hit = [-1] * m
    i, j = m, n
    while i > 0 or j > 0:
        b = bt[i][j]
        if b == 1:
            char_hit[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif b == 2:
            i -= 1
        elif b == 3:
            j -= 1
        else:
            break

    # 聚合到行：行首 = 该行第一个命中字的时间（无命中则用邻近命中/估计值插值）
    events, prev_end = [], 0.0
    for li, (est_t, text) in enumerate(lines):
        idxs = [k for k, (lc, _) in enumerate(lyric_chars) if lc == li and char_hit[k] >= 0]
        if idxs:
            starts = [char_timeline[char_hit[k]][1] for k in idxs]
            ends = [char_timeline[char_hit[k]][2] for k in idxs]
            start, end = min(starts), max(ends)
            n_chars = sum(1 for (lc, _) in lyric_chars if lc == li)
            conf = len(idxs) / float(n_chars)
        else:
            start, end, conf = est_t, est_t + 3.0, 0.0
        if conf < 0.35:  # 低置信 → LRC 估计值兜底
            start, end = est_t, est_t + 3.0
        start = max(start, prev_end)  # 保证单调
        prev_end = end = max(end, start + 0.5)
        events.append([round(start, 2), round(end, 2), str(text), round(conf, 2)])
    return events


def _fallback(lines, char_timeline):
    out, prev = [], 0.0
    for est_t, text in lines:
        start, end = max(est_t, prev), est_t + 3.0
        out.append([round(start, 2), round(end, 2), str(text), 0.0])
        prev = end
    return out


# ---------------- Whisper 转写 + 主流程 ----------------

def transcribe_chars(wav, model_size):
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(
        wav, language="zh", task="transcribe", word_timestamps=True,
        vad_filter=True, vad_parameters={"min_silence_duration_ms": 400})
    timeline = []
    for seg in segments:
        for w in (seg.words or []):
            chars = [c for c in w.word if not c.isspace()]
            if not chars:
                continue
            dur = max(0.01, (w.end - w.start) / len(chars))
            for k, c in enumerate(chars):
                timeline.append((c, w.start + k * dur, w.start + (k + 1) * dur))
    return timeline


def main():
    job = json.load(open(sys.argv[1], "r", encoding="utf-8"))
    lines = [[float(t), str(x)] for t, x in (job.get("lines") or [])]

    def emit(p):
        sys.stdout.write(json.dumps(p, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    timeline = transcribe_chars(job["wav"], job.get("model") or "small")
    events = align_chars_to_lines(lines, timeline)
    if job.get("out"):
        with open(job["out"], "w", encoding="utf-8") as f:
            json.dump(events, f, ensure_ascii=False)
    emit({"type": "done", "events": events, "timeline_chars": len(timeline)})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        try:
            sys.stdout.write(json.dumps({"type": "error", "error": "%s: %s" % (type(e).__name__, e)}) + "\n")
        except Exception:
            pass
        sys.exit(1)
