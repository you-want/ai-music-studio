"""LLM Provider：OpenAI 兼容 Chat API 作曲决策。

默认对接智谱 GLM-4-Flash（免费模型），也支持 DeepSeek / OpenAI / 本地 Ollama
等任何 OpenAI 兼容端点。LLM 只产出高层 SongSpec（结构/和弦/歌词/速度/调式），
音符级展开仍由本地编曲器完成 —— 输出不合规时返回可读错误，由上层决定是否切回本地作曲。

配置优先级：环境变量 AI_MUSIC_LLM_* > config.local.json > 默认值。
"""
import json
import os
import re
import ssl
import tempfile
import time
import http.client
import socket
import urllib.error
import urllib.request

from ..schemas import (GENRES, MOODS, SECTION_NAMES, GENRE_LABELS_ZH, MOOD_LABELS_ZH,
                       parse_request, validate_and_repair_spec)

DEFAULT_CONFIG = {
    "enabled": False,
    "base_url": "https://open.bigmodel.cn/api/paas/v4",
    "api_key": "",
    "model": "glm-4-flash",   # 智谱免费模型
    "temperature": 0.9,
    "timeout": 90,
}

SYSTEM_PROMPT = """你是一位专业的音乐制作人。根据用户的描述，输出一份"歌曲高层设计"JSON。

只输出 JSON 本身，不要任何解释、注释或 markdown 代码块。格式：
{
  "title": "歌名(中文, 2-12字, 诗意)",
  "genre": "pop|rock|electronic|ballad|folk|guofeng|lofi|jazz|rap 之一",
  "mood": "happy|sad|chill|energetic|romantic|epic 之一",
  "key": "C|C#|D|D#|E|F|F#|G|G#|A|A#|B 之一",
  "mode": "major|minor|chinese 之一 (中国风用 chinese)",
  "bpm": 60-184 的整数,
  "structure": [
    {"name": "intro|verse|prechorus|chorus|bridge|outro 之一",
     "bars": 2-16 的整数,
     "chords": ["每小节一个和弦符号, 如 C / Am / F / G / Em / Dm / Am7 / Fmaj7 / Csus4"],
     "lyrics": ["每小节一句中文歌词; intro/outro 用空字符串"]}
  ]
}

规则：
1. structure 总小节数必须等于 %d 小节（会给你目标值）。
2. chords 数组长度必须等于 bars；lyrics 数组长度必须等于 bars。
3. 副歌和弦进行要有推动力（如卡农进行、4536、1645）；主歌平稳。
4. 歌词要口语、有画面感、贴合用户描述；主歌叙事、副歌点题，副歌两段歌词一致。
5. 和弦符号必须是 根音音名 + 可选品质(m/m7/7/maj7/sus4/sus2/add9/dim/6/m6)。
6. 若给出"歌词要求"（主题/故事/必须出现的意象/押韵），歌词必须严格遵循，
   要求里的关键词要自然融进歌词，不能原样照抄一整句。"""


def _ssl_context():
    """创建 SSL context：优先 certifi；默认严格校验证书。
    仅当显式设置 AI_MUSIC_INSECURE_SSL=1 时才关闭校验（兼容自签证书服务）。"""
    if os.environ.get("AI_MUSIC_INSECURE_SSL") == "1":
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    try:
        import certifi  # type: ignore
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _endpoint(base_url):
    base = (base_url or "").strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _temperature(llm):
    """钳制采样温度到 [0, 2]（配置手填时的容错）。"""
    try:
        return max(0.0, min(2.0, float(llm.get("temperature", 0.9))))
    except (TypeError, ValueError):
        return 0.9


def _extract_json(text):
    """从模型回复里抠出 JSON（剥掉代码块围栏、前后废话）。"""
    if not text:
        raise ValueError("空回复")
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("回复中没有 JSON 对象")
    return json.loads(text[start:end + 1])


def load_config(config_path=None):
    cfg = dict(DEFAULT_CONFIG)
    source = "default"
    path = config_path or os.environ.get("AI_MUSIC_CONFIG")
    # __file__ = .../backend/app/providers/llm.py
    # 4 次 dirname → 项目根目录
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if path:
        # 显式指定配置文件（服务器/测试传入）→ 只认这一个，不回落项目根的
        # config.local.json / config.json，保证测试环境与开发者本机配置隔离
        candidates = [path]
    else:
        candidates = [os.path.join(here, "config.local.json"), os.path.join(here, "config.json")]
    for c in candidates:
        if c and os.path.isfile(c):
            try:
                with open(c, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cfg.update({k: v for k, v in (data.get("llm") or {}).items() if k in DEFAULT_CONFIG})
                source = os.path.basename(c)
                break
            except (OSError, ValueError):
                pass
    env_map = {"AI_MUSIC_LLM_BASE_URL": "base_url", "AI_MUSIC_LLM_API_KEY": "api_key",
               "AI_MUSIC_LLM_MODEL": "model"}
    for env_k, cfg_k in env_map.items():
        v = os.environ.get(env_k)
        if v:
            cfg[cfg_k] = v
            source = "env:" + env_k
    cfg["enabled"] = bool(cfg["enabled"] and cfg["api_key"] and cfg["model"])
    return {"llm": cfg, "source": source}


def save_config(config_path, llm_cfg):
    """运行时修改的配置持久化到 config.local.json。"""
    clean = {k: llm_cfg.get(k, DEFAULT_CONFIG[k]) for k in DEFAULT_CONFIG}
    existing = {}
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (OSError, ValueError):
            existing = {}
    existing["llm"] = clean
    directory = os.path.dirname(config_path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".config-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, config_path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
    clean["enabled"] = bool(clean["enabled"] and clean["api_key"] and clean["model"])
    return {"llm": clean, "source": "config.local.json"}


def provide(req, config):
    """req -> (SongSpec|None, info)。任何失败返回 (None, {'reason': ...})。"""
    llm = config["llm"]
    if not (llm.get("enabled") and llm.get("api_key") and llm.get("model")):
        return None, {"reason": "LLM 未启用或缺少 api_key/model"}
    req = parse_request(req)
    target_bars = None  # 由 prompt 里给目标值
    try:
        from ..schemas import bars_for_duration
        target_bars = bars_for_duration(req["duration_sec"], req["bpm"] or 100)
    except Exception:
        target_bars = 16

    user_prompt = (
        "用户描述：%s\n"
        "风格：%s；情绪：%s；目标 BPM：%s；目标总小节数：%d（4拍/小节）。\n"
        "%s"
        "风格只能取：%s。情绪只能取：%s。section name 只能取：%s。\n"
        "请输出 SongSpec JSON。" % (
            req["description"] or "（用户没有具体描述，自由发挥一首好听的曲子）",
            req["genre"], req["mood"], req["bpm"] or "由你根据风格决定",
            target_bars,
            ("歌词要求：%s\n" % req["lyrics_req"]) if req.get("lyrics_req") else "",
            "/".join(GENRES), "/".join(MOODS), "/".join(SECTION_NAMES))
    )
    payload = {
        "model": llm["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT % target_bars},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": _temperature(llm),
        "max_tokens": 3000,
    }
    t0 = time.time()
    try:
        data = _chat(payload, llm)  # 429/5xx 自动退避重试
        content = data["choices"][0]["message"]["content"]
    except Exception as e:  # 网络/鉴权/限流等
        return None, {"reason": "LLM 请求失败: %s" % e}
    try:
        raw = _extract_json(content)
    except (ValueError, json.JSONDecodeError) as e:
        return None, {"reason": "LLM 回复无法解析: %s" % e}
    try:
        raw["description"] = req["description"]
        raw["provider"] = "llm"
        raw["lyrics_source"] = "llm"
        spec, warnings = validate_and_repair_spec(raw, req)
        if not spec.get("seed"):
            spec["seed"] = int(time.time())
        spec["warnings"] = warnings + ["LLM 用时 %dms" % int((time.time() - t0) * 1000)]
        return spec, {"provider": "llm", "model": llm["model"],
                      "elapsed_ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        return None, {"reason": "SongSpec 校验失败: %s" % e}


LYRICS_SYSTEM = """你是一位中文流行歌词创作人。根据主题与情绪写一首歌的歌词。
只输出纯文本歌词，不要解释、不要 markdown。格式（标记必须原样保留）：
[Verse]
四行
[Chorus]
四行
[Verse]
四行
[Chorus]
四行
[Bridge]
两行
[Chorus]
四行
要求：口语化、有画面感、贴合主题；副歌要有一句重复的记忆点(钩子)。"""

LYRICS_RAP_ADDENDUM = """
说唱（rap）附加要求：每两句押同一个韵（句尾押韵要明显），
每行 8-14 字、节奏感强、口语化有态度；副歌可以是简短有力的 hook（可重复两遍）；
允许写入 "yeah / uh / check" 之类的即兴语气词但每段不超过一处。"""


_lyrics_cache = {}  # (主题,风格,情绪) -> 歌词文本；同参数不重复调用


def _chat(payload, llm, attempts=3):
    """带退避重试的 Chat 调用：429/5xx 及连接级故障（代理闪断/慢滴）
    均退避重试，鉴权类错误（401/403）直接抛出。"""
    http_req = urllib.request.Request(
        _endpoint(llm["base_url"]),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer %s" % llm["api_key"]},
        method="POST",
    )
    last_err = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(http_req, timeout=int(llm.get("timeout", 90)),
                                        context=_ssl_context()) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503, 504) and attempt < attempts - 1:
                time.sleep(2)
                continue
            raise
        except (http.client.RemoteDisconnected, ConnectionResetError,
                TimeoutError, socket.timeout) as e:
            last_err = e
            if attempt < attempts - 1:
                time.sleep(3 * (attempt + 1))
                continue
            raise
    raise last_err


def provide_lyrics(req, config):
    """写词：req -> (歌词文本|None, info)。失败返回 (None, reason)。带缓存。"""
    llm = config["llm"]
    if not (llm.get("enabled") and llm.get("api_key") and llm.get("model")):
        return None, {"reason": "LLM 未启用"}
    req = parse_request(req)
    system = LYRICS_SYSTEM + (LYRICS_RAP_ADDENDUM if req["genre"] == "rap" else "")
    user = ("主题/描述：%s\n风格：%s；情绪：%s。请按格式输出歌词。" % (
        req["description"] or "自由发挥，写一首好听的歌",
        GENRE_LABELS_ZH.get(req["genre"], "不限"),
        MOOD_LABELS_ZH.get(req["mood"], "不限"),
    ))
    if req.get("lyrics_req"):
        user += "\n写词要求（必须严格遵循）：%s" % req["lyrics_req"]
    cache_key = (user, req["mood"], req["genre"], system[-40:])
    if cache_key in _lyrics_cache:
        return _lyrics_cache[cache_key], {"model": llm["model"], "cached": True}
    payload = {
        "model": llm["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": _temperature(llm),
        "max_tokens": 1200,
    }
    try:
        data = _chat(payload, llm)
        content = data["choices"][0]["message"]["content"].strip()
        if "[Chorus]" not in content and "副歌" not in content:
            return None, {"reason": "LLM 歌词缺少段落标记"}
        if len(_lyrics_cache) > 200:
            _lyrics_cache.clear()
        _lyrics_cache[cache_key] = content
        return content, {"model": llm["model"]}
    except Exception as e:  # noqa: BLE001
        return None, {"reason": "LLM 请求失败: %s" % e}
