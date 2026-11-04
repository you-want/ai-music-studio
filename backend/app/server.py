"""HTTP 服务器：API + 前端静态资源（仅用 Python 标准库）。

启动： python3 backend/app/server.py   （或 bash scripts/run.sh）
默认 http://127.0.0.1:8766（可用环境变量 AI_MUSIC_PORT 覆盖）
"""
import hmac
from http.cookies import SimpleCookie
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_BACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACK not in sys.path:
    sys.path.insert(0, _BACK)

import copy  # noqa: E402

from app import personas as persona_store  # noqa: E402
from app import schemas  # noqa: E402
from app import __version__  # noqa: E402
from app.providers import mureka as mureka_provider  # noqa: E402
from app.jobs import JobManager  # noqa: E402
from app.music import coverpng, midi, mock_composer, realizer, singer, synth, theory  # noqa: E402
from app.providers import diffrhythm as diffrhythm_provider  # noqa: E402
from app.providers import llm as llm_provider  # noqa: E402
from app.store import Library  # noqa: E402

ROOT = os.path.dirname(_BACK)
FRONTEND_DIR = os.path.join(ROOT, "frontend")
CONFIG_PATH = os.environ.get("AI_MUSIC_CONFIG") or os.path.join(ROOT, "config.local.json")

MAX_JSON_BODY = 1 * 1024 * 1024       # JSON 请求体上限 1MB
MAX_REF_AUDIO = 25 * 1024 * 1024      # 参考音频上限 25MB
SEND_CHUNK = 1 * 1024 * 1024          # 文件流式发送块大小

MIME = {
    ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml",
    ".png": "image/png", ".ico": "image/x-icon", ".json": "application/json; charset=utf-8",
    ".wav": "audio/wav", ".mid": "audio/midi", ".mp3": "audio/mpeg",
    ".mp4": "video/mp4", ".zip": "application/zip",
}

INSPIRATIONS = [
    "夏天的傍晚，骑车载着西瓜和晚霞回家",
    "深夜一个人戴耳机赶稿，窗外有雨",
    "古风江湖：仗剑天涯，月下独酌",
    "清晨跑步，心跳和脚步一起加速",
    "毕业那天，教室的粉笔灰在光里飞",
    "和喜欢的人逛超市，推车歪歪扭扭",
    "雪山、号角与远征的军队",
    "午后猫咪踩键盘，阳光慵懒得像黄油",
    "宇宙飞船离开地球的那一刻",
    "外婆家的老院子，蝉鸣和蒲扇",
    "霓虹雨夜的城市漫游，赛博感",
    "春天第一场野餐，风筝飞得很高",
]


class StudioState:
    """进程内共享：曲库 / 任务队列 / 配置。"""

    def __init__(self, lib_dir=None, config_path=None, frontend_dir=None):
        self.lib = Library(lib_dir or os.path.join(_BACK, "output"))
        self.config_path = config_path or CONFIG_PATH
        self.frontend_dir = frontend_dir or FRONTEND_DIR
        self.config = llm_provider.load_config(self.config_path)
        self.config_lock = threading.Lock()  # POST /api/config 的并发写保护
        self.jobs = JobManager(self.lib, lambda: self.config)
        self.token = os.environ.get("AI_MUSIC_TOKEN") or ""  # 局域网部署的简单鉴权


STATE = StudioState()


class _BodyTooLarge(Exception):
    pass


class Handler(BaseHTTPRequestHandler):
    server_version = "SongWeaver/" + __version__
    protocol_version = "HTTP/1.1"
    _video_locks = {}  # song_id -> Lock：MV 导出防并发双写（见 _video）

    # ------------------------------------------------------------ 基础
    def log_message(self, fmt, *args):
        message = fmt % args
        # Query-string bearer tokens must never land in access logs.
        message = re.sub(r"([?&]token=)[^& ]+", r"\1<redacted>", message)
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), message))

    def _send(self, code, body, content_type="application/json; charset=utf-8", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _check_auth(self, path, query):
        """Protect API requests with a bearer header or a derived session cookie."""
        if not STATE.token or path == "/api/health" or not path.startswith("/api/"):
            return None
        auth = self.headers.get("Authorization") or ""
        if auth.startswith("Bearer "):
            supplied = auth[7:].strip()
            if supplied and hmac.compare_digest(supplied, STATE.token):
                return None
        else:
            cookie = SimpleCookie()
            try:
                cookie.load(self.headers.get("Cookie", ""))
                morsel = cookie.get("aimusic_session")
            except Exception:
                morsel = None
            if morsel and self._validate_session(STATE.token, morsel.value):
                return None
        return self._send(401, {"error": "未授权：缺少或错误的访问 Token"},
                          extra={"WWW-Authenticate": "Bearer"})

    SESSION_TTL_SEC = 43200  # 会话 Cookie 有效期（12 小时）

    @staticmethod
    def _session_value(token, exp):
        """访问 Token 的派生会话凭证：HMAC(token, "v1:过期时间") + 过期时间。

        Cookie 里不再保存原始 Token：泄漏 Cookie 只得到一个 12 小时后自动
        作废的凭证；更换 AI_MUSIC_TOKEN 即可吊销全部会话；无服务端状态，
        服务重启后依然有效。
        """
        msg = ("aimusic-session-v1:%d" % exp).encode("utf-8")
        mac = hmac.new(token.encode("utf-8"), msg, "sha256").hexdigest()
        return "%s.%d" % (mac, exp)

    @classmethod
    def _validate_session(cls, token, value):
        try:
            mac, exp_s = value.rsplit(".", 1)
            exp = int(exp_s)
        except (ValueError, AttributeError):
            return False
        if exp <= time.time() or exp > int(time.time()) + cls.SESSION_TTL_SEC * 2:
            return False
        return hmac.compare_digest(mac + "." + exp_s, cls._session_value(token, exp))

    def _json_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_JSON_BODY:
            raise _BodyTooLarge()
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    def _send_file(self, path, extra_headers=None):
        try:
            size = os.path.getsize(path)
            f = open(path, "rb")
        except OSError:
            return self._send(404, {"error": "not found"})
        ext = os.path.splitext(path)[1].lower()
        self.send_response(200)
        self.send_header("Content-Type", MIME.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(size))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-cache")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            if self.command != "HEAD":
                while True:
                    chunk = f.read(SEND_CHUNK)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            f.close()

    # ------------------------------------------------------------ 路由
    def do_GET(self):
        self._route("GET")

    def do_HEAD(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def do_DELETE(self):
        self._route("DELETE")

    def do_PATCH(self):
        self._route("PATCH")

    def _route(self, method):
        parts = urllib.parse.urlparse(self.path)
        path, query = parts.path, urllib.parse.parse_qs(parts.query)
        try:
            denied = self._check_auth(path, query)
            if denied is not None:
                return denied
            if path == "/api/health":
                return self._send(200, {"ok": True, "name": "SongWeaver (织歌)",
                                        "version": __version__})
            if path == "/api/session" and method == "POST":
                # 用一次性 Bearer 换取派生会话 Cookie（Cookie 不落原始 Token）
                if not STATE.token:
                    return self._send(200, {"ok": True})
                secure = "; Secure" if self.headers.get("X-Forwarded-Proto", "").lower() == "https" else ""
                exp = int(time.time()) + Handler.SESSION_TTL_SEC
                return self._send(200, {"ok": True}, extra={
                    "Set-Cookie": "aimusic_session=%s; Path=/; HttpOnly; SameSite=Strict; Max-Age=%d%s"
                    % (Handler._session_value(STATE.token, exp), Handler.SESSION_TTL_SEC, secure)})
            if path == "/api/meta" and method == "GET":
                return self._send(200, {
                    "genres": [{"id": g, "label": schemas.GENRE_LABELS_ZH[g]} for g in schemas.GENRES],
                    "moods": [{"id": m, "label": schemas.MOOD_LABELS_ZH[m]} for m in schemas.MOODS],
                    "voices": [{"id": "auto", "label": "自动（按风格）"}]
                              + [{"id": v, "label": schemas.LEAD_VOICE_LABELS_ZH[v]}
                                 for v in schemas.LEAD_VOICES],
                    "keys": [{"id": "auto", "label": "自动"}]
                            + [{"id": n, "label": n} for n in theory.NOTE_NAMES],
                    "lyrics_hint": "支持段落标记：[Verse] 主歌 / [Chorus] 副歌 / [Bridge] 桥段，"
                                   "无标记时自动分配",
                    "sing_supported": singer.available_voice() is not None,
                    "inspirations": INSPIRATIONS,
                    "engines": {"diffrhythm": diffrhythm_provider.provider.engine_info(),
                                "mureka": mureka_provider.provider.engine_info()},
                    "defaults": {"duration_sec": 30, "provider": "auto"},
                })
            if path == "/api/ref-audio" and method == "POST":
                return self._ref_audio_upload()
            if path == "/api/personas" and method == "GET":
                return self._send(200, {"personas": persona_store.list_personas(STATE.lib)})
            if path == "/api/personas" and method == "POST":
                return self._persona_create()
            m = re.match(r"^/api/personas/([\w-]+)$", path)
            if m and method == "DELETE":
                return self._send(200, {"deleted": persona_store.delete(STATE.lib, m.group(1))})
            if path == "/api/album" and method == "POST":
                return self._album_create()
            if path == "/api/album.zip" and method == "GET":
                return self._album_zip(query)
            m = re.match(r"^/api/songs/([\w-]+)/reseg$", path)
            if m and method == "POST":
                return self._reseg(m.group(1))
            if path == "/api/lyrics" and method == "POST":
                req = schemas.parse_request(self._json_body())
                text, source = None, "mock"
                if STATE.config["llm"].get("enabled"):
                    text, _info = llm_provider.provide_lyrics(req, STATE.config)
                    if text:
                        source = "llm"
                if not text:
                    from app.music import mock_composer
                    text = mock_composer.mock_lyrics_text(req)
                return self._send(200, {"lyrics": text, "source": source})
            if path == "/api/config" and method == "GET":
                return self._api_config_get()
            if path == "/api/config" and method == "POST":
                body = self._json_body()
                llm_now = dict(STATE.config["llm"])
                for k in ("enabled", "base_url", "api_key", "model", "temperature"):
                    if k in (body.get("llm") or {}):
                        v = body["llm"][k]
                        if k == "api_key" and str(v).strip().endswith("*"):
                            continue  # 脱敏值回传时忽略
                        llm_now[k] = v
                with STATE.config_lock:
                    STATE.config = llm_provider.save_config(STATE.config_path, llm_now)
                    if body.get("mureka"):
                        self._save_mureka(STATE.config_path, body["mureka"])
                    if body.get("neural"):
                        self._save_neural(STATE.config_path, body["neural"])
                return self._api_config_get()
            if path == "/api/config/test" and method == "POST":
                # 测试 LLM 连接：发一条最小 chat 请求，验证 base_url + api_key + model 是否可用
                llm_now = dict(STATE.config["llm"])
                body = self._json_body()
                if body.get("llm"):
                    for k in ("enabled", "base_url", "api_key", "model", "temperature"):
                        if k in body["llm"] and body["llm"][k] not in (None, ""):
                            llm_now[k] = body["llm"][k]
                # 允许用 body 里临时传入的 api_key 覆盖（不写回 config.local.json）
                if body.get("llm", {}).get("api_key"):
                    llm_now["api_key"] = body["llm"]["api_key"]
                if not (llm_now.get("api_key") and llm_now.get("model") and llm_now.get("base_url")):
                    return self._send(200, {"ok": False, "reason": "缺少 base_url / api_key / model，请先填写"})
                if not llm_now.get("enabled"):
                    return self._send(200, {"ok": False, "reason": "LLM 未勾选启用（checkbox）"})
                t0 = time.time()
                try:
                    payload = {
                        "model": llm_now["model"],
                        "messages": [{"role": "user", "content": "Hi, reply with just 'OK'."}],
                        "max_tokens": 10,
                        "temperature": 0.1,
                    }
                    url = llm_provider._endpoint(llm_now["base_url"])
                    req = urllib.request.Request(
                        url, data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json",
                                 "Authorization": "Bearer %s" % llm_now["api_key"]},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=int(llm_now.get("timeout", 30)),
                                                context=llm_provider._ssl_context()) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    elapsed_ms = int((time.time() - t0) * 1000)
                    content = ""
                    try:
                        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
                    except Exception:
                        pass
                    return self._send(200, {
                        "ok": True,
                        "elapsed_ms": elapsed_ms,
                        "model": llm_now["model"],
                        "base_url": llm_now["base_url"],
                        "reply_preview": (content or "").strip()[:80],
                    })
                except Exception as e:
                    elapsed_ms = int((time.time() - t0) * 1000)
                    return self._send(200, {
                        "ok": False,
                        "reason": str(e),
                        "elapsed_ms": elapsed_ms,
                        "base_url": llm_now.get("base_url", ""),
                        "model": llm_now.get("model", ""),
                    })
            if path == "/api/compose" and method == "POST":
                req = schemas.parse_request(self._json_body())
                if req.get("ref_audio"):
                    ref = STATE.lib.ref_path(req["ref_audio"])
                    if not os.path.isfile(ref):
                        return self._send(400, {"error": "参考音频不存在或已失效，请重新上传"})
                    req["ref_audio_path"] = ref
                if not req["description"] and req["genre"] == "auto":
                    req["description"] = "一段随机的心情小曲"
                try:
                    job_id = STATE.jobs.submit(req)
                except RuntimeError as e:
                    if "任务队列已满" in str(e):
                        return self._send(429, {"error": str(e), "retry_after_sec": 10},
                                          extra={"Retry-After": "10"})
                    raise
                return self._send(200, {"job_id": job_id})
            m = re.match(r"^/api/jobs/([\w-]+)$", path)
            if m and method == "GET":
                job = STATE.jobs.status(m.group(1))
                return self._send(200, job) if job else self._send(404, {"error": "job not found"})
            if m and method == "DELETE":
                job = STATE.jobs.cancel(m.group(1))
                return self._send(200, {"canceled": True, "job": job}) if job \
                    else self._send(404, {"error": "job not found"})
            if path == "/api/songs" and method == "GET":
                return self._send(200, {"songs": STATE.lib.list_songs()})
            m = re.match(r"^/api/songs/([\w-]+)/regen_chorus$", path)
            if m and method == "POST":
                return self._regen_chorus(m.group(1))
            m = re.match(r"^/api/songs/([\w-]+)/favorite$", path)
            if m and method == "POST":
                return self._favorite(m.group(1))
            m = re.match(r"^/api/songs/([\w-]+)/(bundle\.zip|video\.mp4)$", path)
            if m and method == "GET":
                return self._export_route(m.group(1), m.group(2))
            m = re.match(r"^/api/songs/([\w-]+)$", path)
            if m:
                return self._song_route(method, m.group(1))
            m = re.match(r"^/api/songs/([\w-]+)/(audio\.wav|song\.mid)$", path)
            if m and method == "GET":
                return self._audio_route(m.group(1), m.group(2))
            m = re.match(r"^/api/songs/([\w-]+)/stems/(vocal|harmony|bass|drums)\.wav$", path)
            if m and method == "GET":
                return self._stem_route(m.group(1), m.group(2))
            if path.startswith("/api/"):
                return self._send(404, {"error": "unknown api"})
            return self._static(path)
        except schemas.RequestError as e:
            return self._send(400, {"error": str(e)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except _BodyTooLarge:
            return self._send(413, {"error": "请求体过大（上限 1MB）"})
        except Exception as e:  # noqa: BLE001
            return self._send(500, {"error": str(e)})

    def _api_config_get(self):
        llm = dict(STATE.config["llm"])
        key = llm.get("api_key") or ""
        llm["api_key_set"] = bool(key)
        llm["api_key_masked"] = (key[:4] + "****" + key[-2:]) if len(key) > 8 else ("****" if key else "")
        llm.pop("api_key", None)
        mk = mureka_provider.load_mureka_config(STATE.config_path)
        mkey = mk.get("api_key") or ""
        nd = diffrhythm_provider.load_neural_config(STATE.config_path)
        return self._send(200, {
            "llm": llm, "source": STATE.config["source"],
            "mureka": {"enabled": bool(mk["enabled"] and mkey), "model": mk["model"],
                       "api_key_set": bool(mkey),
                       "api_key_masked": (mkey[:4] + "****" + mkey[-2:]) if len(mkey) > 8 else ("****" if mkey else "")},
            "neural": {"steps": nd["steps"]},
        })

    @staticmethod
    def _atomic_json_write(path, data):
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".config-", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass

    @staticmethod
    def _save_neural(config_path, neural_body):
        """设置弹窗提交的 neural 节（采样步数）合并写回 config.local.json。"""
        try:
            steps = max(8, min(32, int(neural_body.get("steps") or 16)))
        except (TypeError, ValueError):
            return
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = {}
        merged = dict(data.get("neural") or {})
        merged["steps"] = steps
        data["neural"] = merged
        Handler._atomic_json_write(config_path, data)

    @staticmethod
    def _save_mureka(config_path, mureka_body):
        """把设置弹窗提交的 mureka 节合并写回 config.local.json（脱敏值忽略）。"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = {}
        section = {}
        for k in ("enabled", "api_key", "model", "vocal_gender"):
            if k in mureka_body:
                val = mureka_body[k]
                if k == "api_key" and str(val).strip().endswith("*"):
                    continue
                section[k] = val
        if section:
            merged = dict(data.get("mureka") or {})
            merged.update(section)
            data["mureka"] = merged
            Handler._atomic_json_write(config_path, data)

    # ------------------------------------------------------------ 曲库
    def _song_route(self, method, song_id):
        lib = STATE.lib
        if method == "DELETE":
            return self._send(200, {"deleted": lib.delete(song_id)})
        if method == "PATCH":  # 重命名
            if not lib.exists(song_id):
                return self._send(404, {"error": "song not found"})
            title = (self._json_body() or {}).get("title")
            try:
                return self._send(200, lib.rename(song_id, title))
            except (ValueError, OSError) as e:
                return self._send(400, {"error": str(e)})
        if method == "GET":
            if not lib.exists(song_id):
                return self._send(404, {"error": "song not found"})
            return self._send(200, {
                "meta": lib.get_meta(song_id),
                "spec": lib.get_spec(song_id),
                "score": lib.get_score(song_id),
            })
        return self._send(405, {"error": "method not allowed"})

    def _favorite(self, song_id):
        if not STATE.lib.exists(song_id):
            return self._send(404, {"error": "song not found"})
        fav = bool((self._json_body() or {}).get("favorite"))
        return self._send(200, STATE.lib.set_favorite(song_id, fav))

    def _persona_create(self):
        """创建音色 Persona（同步，MuQ 嵌入提取约 1 分钟）。"""
        body = self._json_body()
        try:
            meta = persona_store.create(
                STATE.lib,
                name=body.get("name") or "",
                source=body.get("source") or "",
                song_id=body.get("song_id"),
                ref_audio_name=body.get("ref_audio"),
                text=body.get("text"),
            )
        except (ValueError, RuntimeError) as e:
            return self._send(400, {"error": str(e)})
        return self._send(200, {"persona": meta})

    def _reseg(self, song_id):
        """分段重唱：原曲为参考，指定段落按新歌词重采样（产出新作品）。"""
        lib = STATE.lib
        if not lib.exists(song_id):
            return self._send(404, {"error": "song not found"})
        meta = lib.get_meta(song_id)
        if not meta.get("audio_only") or meta.get("provider") != "diffrhythm":
            return self._send(400, {"error": "仅本地神经歌声作品支持分段重唱"})
        body = self._json_body()
        try:
            sec_idx = int(body.get("section_index", -1))
        except (TypeError, ValueError):
            return self._send(400, {"error": "section_index 非法"})
        new_lyrics = schemas.clip_multiline_text(body.get("new_lyrics"), 2000) or ""
        if not new_lyrics:
            return self._send(400, {"error": "新歌词不能为空"})
        try:
            spec = lib.get_spec(song_id)
            score = lib.get_score(song_id)
        except (OSError, ValueError):
            return self._send(400, {"error": "作品缺少曲谱，无法定位段落"})
        sections = score.get("sections") or []
        if not (0 <= sec_idx < len(sections)):
            return self._send(400, {"error": "段落序号越界（0-%d）" % (len(sections) - 1)})
        sec_meta = sections[sec_idx]

        bpm = score.get("bpm") or meta.get("bpm") or 100
        spb = 60.0 / max(30, bpm)
        sec_start = sec_meta["start_beat"] * spb
        sec_end = min((sec_meta["start_beat"] + sec_meta["beats"]) * spb,
                      float(meta.get("duration_sec") or 95))
        lines = [ln.strip() for ln in new_lyrics.splitlines() if ln.strip()]

        new_spec = copy.deepcopy(spec)
        target = new_spec["structure"][sec_idx]
        target["lyrics"] = [lines[i % len(lines)] for i in range(target["bars"])]
        new_spec["title"] = (spec.get("title") or "未命名") + " · 重唱版"
        new_spec["provider"] = "diffrhythm"
        new_spec["audio_only"] = True

        audio_length = 95 if float(meta.get("duration_sec") or 95) <= 120 else 285
        req = {
            "provider": "diffrhythm", "edit_of": song_id, "edit_spec": new_spec,
            "duration_sec": audio_length,
            "seed": random.randint(1, 2 ** 30),
        }
        if audio_length == 285:
            req["full_song"] = True
        persona_id = schemas.sanitize_filename(body.get("persona_id"))
        if persona_id:
            req["persona_id"] = persona_id
        if body.get("full_remake"):
            # 整首重制：不用 edit 模式（无 VAE 往返损失），全程重新演唱——质量最佳
            req["fixed_spec"] = new_spec
        else:
            req["edit"] = {"ref_song": os.path.abspath(lib.wav_path(song_id)),
                           "segments": [[round(sec_start, 2), round(sec_end, 2)]]}
        job_id = STATE.jobs.submit(req)
        return self._send(200, {"job_id": job_id})

    def _album_create(self):
        body = self._json_body()
        persona_id = schemas.sanitize_filename(body.get("persona_id"))
        if not persona_id or not persona_store.get(STATE.lib, persona_id):
            return self._send(400, {"error": "需要有效的音色 Persona"})
        try:
            count = max(1, min(5, int(body.get("count") or 2)))
        except (TypeError, ValueError):
            count = 2
        theme = str(body.get("theme") or "").strip()[:200] or None
        genre = body.get("genre") if body.get("genre") in schemas.GENRES else "auto"
        job_ids = []
        for i in range(count):
            job_ids.append(STATE.jobs.submit({
                "provider": "diffrhythm",
                "description": (theme or "一张概念专辑的第 %d 首") % (i + 1) if theme
                else "一张概念专辑的第 %d 首，与前面曲目同一副嗓子" % (i + 1),
                "genre": genre, "duration_sec": 95,
                "persona_id": persona_id,
                "seed": random.randint(1, 2 ** 30),
            }))
        return self._send(200, {"job_ids": job_ids, "count": count})

    def _album_zip(self, query):
        """按 ids 打包多首作品（wav+封面+歌词+meta），用于专辑下载。"""
        import io
        raw = (query.get("ids") or [""])[0]
        ids = [i.strip() for i in raw.split(",") if i.strip()][:8]
        if not ids:
            return self._send(400, {"error": "缺少 ids"})
        lib = STATE.lib
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for sid in ids:
                if not lib.exists(sid):
                    continue
                d = lib.song_dir(sid)
                folder = "album/" + re.sub(r'[\\/:*?"<>|]', "_",
                                           lib.get_meta(sid).get("title", sid))
                z.write(lib.wav_path(sid), folder + "/song.wav")
                z.write(self._ensure_cover(sid, d), folder + "/cover.png")
                z.writestr(folder + "/lyrics.txt", self._lyrics_text(sid))
                z.write(os.path.join(d, "meta.json"), folder + "/meta.json")
        fname = urllib.parse.quote("织歌-专辑.zip")
        return self._send(200, buf.getvalue(), "application/zip",
                          {"Content-Disposition": "attachment; filename*=UTF-8''%s" % fname})

    def _ref_audio_upload(self):
        """上传参考音频（神经引擎风格迁移用）：raw body，服务端生成文件名。"""
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return self._send(400, {"error": "缺少音频数据"})
        if length > MAX_REF_AUDIO:
            return self._send(413, {"error": "音频过大（上限 25MB）"})
        data = self.rfile.read(length)
        if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            ext = "wav"
        elif data[:4] == b"fLaC":
            ext = "flac"
        elif data[:3] == b"ID3" or (len(data) > 1 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
            ext = "mp3"
        else:
            return self._send(400, {"error": "仅支持 wav / mp3 / flac 格式"})
        name = "ref-%s.%s" % (uuid.uuid4().hex[:10], ext)
        with open(STATE.lib.ref_path(name), "wb") as f:
            f.write(data)
        return self._send(200, {"ref_audio": name, "bytes": length,
                                "hint": "≥10 秒效果最佳；生成时在进阶模式传入 ref_audio 字段"})

    def _regen_chorus(self, song_id):
        """只重写副歌旋律：其余（主歌/和声/鼓）与原曲完全一致。同步渲染。"""
        lib = STATE.lib
        if not lib.exists(song_id):
            return self._send(404, {"error": "song not found"})
        try:
            spec = lib.get_spec(song_id)
            fam = spec.setdefault("family_seeds", {})
            fam["chorus"] = random.randint(1, 2 ** 30)
            spec["title"] = (spec.get("title") or "未命名") + " · 副歌B"
            song = realizer.realize(spec)
            if spec.get("sing") and singer.available_voice():
                song["sing_events"] = singer.plan_singing(song)
            new_id = lib.new_song_id()
            lib.prepare(new_id)
            synth.render_all(song, lib.wav_path(new_id), lib.song_dir(new_id))
            midi.write_midi_file(song, lib.midi_path(new_id))
            lib.save(new_id, spec, song)
            return self._send(200, {"song_id": new_id, "title": spec["title"]})
        except Exception as e:  # noqa: BLE001
            return self._send(500, {"error": str(e)})

    def _export_route(self, song_id, filename):
        lib = STATE.lib
        if not lib.exists(song_id):
            return self._send(404, {"error": "song not found"})
        song_dir = lib.song_dir(song_id)
        if filename == "bundle.zip":
            return self._bundle(song_id, song_dir)
        return self._video(song_id, song_dir)

    def _ensure_cover(self, song_id, song_dir):
        cover = os.path.join(song_dir, "cover.png")
        if not os.path.isfile(cover):
            coverpng.make_cover(song_id, size=1024, path=cover)
        return cover

    def _lyrics_text(self, song_id):
        spec = STATE.lib.get_spec(song_id)
        lines = [spec.get("title", "")]
        for sec in spec.get("structure", []):
            label = schemas.SECTION_LABELS_ZH.get(sec["name"], sec["name"])
            lines.append("")
            lines.append("[%s]" % label)
            lines.extend(line or "" for line in sec.get("lyrics", []))
        return "\n".join(lines)

    def _bundle(self, song_id, song_dir):
        """分享包 zip：整曲 + MIDI + 分轨 + 封面 + 歌词 + 曲目信息。"""
        import io
        lib = STATE.lib
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for name, path in [("song.wav", lib.wav_path(song_id)),
                               ("song.mid", lib.midi_path(song_id)),
                               ("meta.json", os.path.join(song_dir, "meta.json"))]:
                if os.path.isfile(path):
                    z.write(path, name)
            for stem in sorted(os.listdir(song_dir)) if os.path.isdir(song_dir) else []:
                if stem.startswith("stem_") and stem.endswith(".wav"):
                    z.write(os.path.join(song_dir, stem), stem)
            z.writestr("lyrics.txt", self._lyrics_text(song_id))
            with open(self._ensure_cover(song_id, song_dir), "rb") as f:
                z.writestr("cover.png", f.read())
        title = lib.get_meta(song_id).get("title", song_id)
        fname = urllib.parse.quote("%s-分享包.zip" % title)
        return self._send(200, buf.getvalue(), "application/zip",
                          {"Content-Disposition": "attachment; filename*=UTF-8''%s" % fname})

    def _video(self, song_id, song_dir):
        """MV 导出：封面图 + 音频 -> mp4（ffmpeg，首次生成后缓存）。

        并发安全：per-song 锁防双跑；ffmpeg 先写 .part 再 os.replace 原子落盘，
        避免两个请求同时写 video.mp4 产出损坏文件。
        """
        if not shutil.which("ffmpeg"):
            return self._send(501, {"error": "未安装 ffmpeg，无法导出 MV（brew install ffmpeg）"})
        mp4 = os.path.join(song_dir, "video.mp4")
        # setdefault 原子取/建锁：同一首歌并发导出时串行，不同歌互不影响
        with Handler._video_locks.setdefault(song_id, threading.Lock()):
            if not os.path.isfile(mp4):
                cover = self._ensure_cover(song_id, song_dir)
                part = mp4 + ".part"
                try:
                    subprocess.run(
                        ["ffmpeg", "-y", "-loop", "1", "-i", cover,
                         "-i", STATE.lib.wav_path(song_id),
                         "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
                         "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
                         # 输出是 .part 临时名，ffmpeg 无法从扩展名推断封装格式，须显式指定
                         "-f", "mp4", part],
                        capture_output=True, timeout=300, check=True)
                    os.replace(part, mp4)
                except subprocess.CalledProcessError as e:
                    return self._send(500, {"error": "ffmpeg 失败: %s"
                                            % e.stderr.decode("utf-8", "ignore")[-300:]})
                except subprocess.TimeoutExpired:
                    return self._send(500, {"error": "ffmpeg 超时"})
                finally:
                    # 成功时 .part 已被 replace 走（remove 抛 FileNotFoundError 被吞）；
                    # 失败/超时清掉残留半成品
                    try:
                        os.remove(part)
                    except OSError:
                        pass
        size = os.path.getsize(mp4)
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition", "attachment; filename=video.mp4")
        self.end_headers()
        if self.command != "HEAD":
            with open(mp4, "rb") as f:
                shutil.copyfileobj(f, self.wfile)

    def _audio_route(self, song_id, filename):
        lib = STATE.lib
        path = lib.wav_path(song_id) if filename == "audio.wav" else lib.midi_path(song_id)
        if not os.path.isfile(path):
            return self._send(404, {"error": "file not found"})
        size = os.path.getsize(path)
        ctype = "audio/wav" if filename.endswith(".wav") else "audio/midi"
        rng = self.headers.get("Range")
        if rng:
            m = re.match(r"^bytes=(\d*)-(\d*)$", rng.strip())
            if m and (m.group(1) or m.group(2)):
                start_s, end_s = m.group(1), m.group(2)
                if not start_s:  # 后缀语义：bytes=-N 表示最后 N 字节
                    n = int(end_s)
                    if n <= 0:
                        return self._send(416, {"error": "range not satisfiable"},
                                          extra={"Content-Range": "bytes */%d" % size})
                    start, end = max(0, size - n), size - 1
                else:
                    start = int(start_s)
                    end = min(int(end_s), size - 1) if end_s else size - 1
                    if start >= size or start > end:
                        return self._send(416, {"error": "range not satisfiable"},
                                          extra={"Content-Range": "bytes */%d" % size})
                length = end - start + 1
                self.send_response(206)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(length))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                if self.command == "HEAD":
                    return None
                with open(path, "rb") as f:
                    f.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = f.read(min(SEND_CHUNK, remaining))
                        if not chunk:
                            break
                        try:
                            self.wfile.write(chunk)
                        except (BrokenPipeError, ConnectionResetError):
                            return None
                        remaining -= len(chunk)
                return None
        return self._send_file(path, {"Accept-Ranges": "bytes"})

    def _stem_route(self, song_id, name):
        path = os.path.join(STATE.lib.song_dir(song_id), "stem_%s.wav" % name)
        if not os.path.isfile(path):
            return self._send(404, {"error": "stem not found (旧作品无分轨)"})
        return self._send_file(path, {"Accept-Ranges": "bytes"})

    # ------------------------------------------------------------ 静态
    def _static(self, path):
        if path in ("/", "/index.html"):
            return self._send_file(os.path.join(STATE.frontend_dir, "index.html"))
        rel = path.lstrip("/")
        target = os.path.normpath(os.path.join(STATE.frontend_dir, rel))
        try:
            inside = os.path.commonpath([STATE.frontend_dir, target]) == STATE.frontend_dir
        except ValueError:  # 不同盘符等
            inside = False
        if not inside:
            return self._send(403, {"error": "forbidden"})
        if os.path.isfile(target):
            return self._send_file(target)
        return self._send(404, {"error": "not found", "path": path})


def create_server(host="127.0.0.1", port=8766, lib_dir=None, config_path=None, frontend_dir=None):
    state = StudioState(lib_dir, config_path, frontend_dir)
    global STATE
    STATE = state
    return ThreadingHTTPServer((host, port), Handler)


def main():
    host = os.environ.get("AI_MUSIC_HOST", "127.0.0.1")
    port = int(os.environ.get("AI_MUSIC_PORT", "8766"))
    token = os.environ.get("AI_MUSIC_TOKEN") or ""
    # 局域网/公网裸奔保护：非本机绑定时强制要求设置访问 Token
    if host not in ("127.0.0.1", "localhost", "::1") and not token:
        print("  ⛔ 拒绝启动：AI_MUSIC_HOST=%s 会把服务暴露给局域网，"
              "必须同时设置 AI_MUSIC_TOKEN（访问口令）" % host, flush=True)
        print("     例如：AI_MUSIC_HOST=0.0.0.0 AI_MUSIC_TOKEN=你的口令 bash scripts/run.sh",
              flush=True)
        sys.exit(2)
    # 后台预热：演唱音库 + 神经引擎可用性探测（faster_whisper/demucs import 一次）
    def _preheat():
        try:
            chars = set()
            for mood in schemas.MOODS:
                for v in mock_composer.LYRICS.get(mood, {}).values():
                    chars.update("".join(v))
            if singer.available_voice():
                singer._warm(singer.available_voice(), chars)
        except Exception:
            pass
        try:
            diffrhythm_provider.provider.engine_info()
        except Exception:
            pass
    threading.Thread(target=_preheat, daemon=True).start()
    httpd = ThreadingHTTPServer((host, port), Handler)
    print()
    print("  🎵 织歌 SongWeaver 已启动", flush=True)
    print("  ➜  本地访问:  http://%s:%d" % (host, port), flush=True)
    print("  ➜  引擎状态:  Mock(离线) 就绪；LLM %s；神经歌声(DiffRhythm) %s"
          % ("已启用(%s)" % STATE.config["llm"].get("model")
             if STATE.config["llm"].get("enabled") else "未配置(可在网页设置里免费开启 GLM-4-Flash)",
             "就绪" if diffrhythm_provider.provider.available()[0]
             else "未安装(bash scripts/setup_neural.sh)"),
          flush=True)
    if token:
        print("  ➜  访问鉴权:  已开启（局域网设备访问需携带 Token）", flush=True)
    print("  ➜  作品目录:  %s" % STATE.lib.lib_dir, flush=True)
    print("  Ctrl+C 退出", flush=True)
    print(flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n再见 🎧")


if __name__ == "__main__":
    main()
