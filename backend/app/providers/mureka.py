"""Mureka 云端歌曲生成 Provider（昆仑万维官方 REST，~$0.05/首含商用授权）。

契约（platform.mureka.ai/docs，2026-10 核对）：
    POST /v1/song/generate      {model, lyrics?, prompt?, reference_id?, vocal_gender?,
                                 instrumental?, duration?} -> 异步任务 id
    GET  /v1/song/query/{id}    轮询（succeeded / failed / running）
    POST /v1/files/upload       参考音频 -> reference_id
    产出：MP3 下载 URL（本 Provider 落地后用 ffmpeg 转成 44.1kHz WAV 交给主管线，
          之后照常走 Whisper 对齐 + Demucs 分轨）
响应字段做了容错解析（官方文档部分字段折叠，以联调为准）。

配置（config.local.json "mureka" 节，或环境变量）：
    mureka.enabled  AI_MUSIC_MUREKA_ENABLED  默认 true（有 key 即启用）
    mureka.api_key  AI_MUSIC_MUREKA_KEY      必填
    mureka.model    AI_MUSIC_MUREKA_MODEL    默认 mureka-6
"""
import base64
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request

from .audio_base import AudioProvider, AudioProviderError, CancelledError

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
DEFAULT_BASE = "https://api.mureka.ai"
POLL_INTERVAL = 5
POLL_TIMEOUT_MIN = 15


def load_mureka_config(config_path=None):
    cfg = {"enabled": True, "api_key": "", "model": "mureka-6",
           "base": DEFAULT_BASE, "vocal_gender": ""}
    path = config_path or os.environ.get("AI_MUSIC_CONFIG") or os.path.join(_ROOT, "config.local.json")
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k in cfg:
                if k in (data.get("mureka") or {}):
                    cfg[k] = data["mureka"][k]
        except (OSError, ValueError):
            pass
    if os.environ.get("AI_MUSIC_MUREKA_KEY"):
        cfg["api_key"] = os.environ["AI_MUSIC_MUREKA_KEY"]
    if os.environ.get("AI_MUSIC_MUREKA_MODEL"):
        cfg["model"] = os.environ["AI_MUSIC_MUREKA_MODEL"]
    cfg["enabled"] = bool(cfg["enabled"] and cfg["api_key"])
    return cfg


def _find_first_url(obj, suffix=".mp3"):
    """在任意嵌套的响应 JSON 里找第一个指定后缀的 URL（官方字段可能有变）。"""
    if isinstance(obj, str):
        return obj if obj.startswith("http") and obj.endswith(suffix) else None
    if isinstance(obj, dict):
        for v in obj.values():
            hit = _find_first_url(v, suffix)
            if hit:
                return hit
    if isinstance(obj, list):
        for v in obj:
            hit = _find_first_url(v, suffix)
            if hit:
                return hit
    return None


def _task_state(obj):
    """容忍字段名差异地取任务状态与 id。"""
    for k in ("state", "status"):
        v = obj.get(k) if isinstance(obj, dict) else None
        if isinstance(v, str):
            return v
        if isinstance(obj, dict):
            inner = obj.get("data") or {}
            if isinstance(inner, dict) and isinstance(inner.get(k), str):
                return inner[k]
    if isinstance(obj, dict):
        inner = obj.get("data")
        if isinstance(inner, dict):
            for k in ("state", "status"):
                if isinstance(inner.get(k), str):
                    return inner[k]
    return ""


def _task_id(obj):
    if isinstance(obj, dict):
        for k in ("id", "task_id"):
            if obj.get(k):
                return str(obj[k])
        inner = obj.get("data")
        if isinstance(inner, dict):
            for k in ("id", "task_id"):
                if inner.get(k):
                    return str(inner[k])
    return ""


class MurekaProvider(AudioProvider):
    name = "mureka"

    def __init__(self, config_path=None):
        self.config_path = config_path

    # ------------------------------------------------------------ 可用性
    def available(self):
        cfg = load_mureka_config(self.config_path)
        if not cfg["api_key"]:
            return False, "未配置 Mureka API Key（设置 → 云 API；约 $0.05/首）"
        if not cfg["enabled"]:
            return False, "Mureka 已在配置中禁用"
        return True, "就绪（云端生成约 1-2 分钟，~$0.05/首，含商用授权）"

    def engine_info(self):
        ok, reason = self.available()
        cfg = load_mureka_config(self.config_path)
        return {"available": ok, "reason": reason, "model": cfg["model"],
                "cloud": True}

    # ------------------------------------------------------------ HTTP
    def _request(self, method, url, cfg, body=None, raw=None, content_type=None,
                 timeout=60):
        headers = {"Authorization": "Bearer %s" % cfg["api_key"]}
        data = None
        if raw is not None:
            data = raw
            headers["Content-Type"] = content_type or "application/octet-stream"
        elif body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        last = None
        for attempt in range(3):
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "ignore")[:200]
                if e.code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    last = AudioProviderError("Mureka %s: %s" % (e.code, detail))
                    continue
                raise AudioProviderError("Mureka %s: %s" % (e.code, detail))
            except (urllib.error.URLError, OSError, ValueError) as e:
                last = AudioProviderError("Mureka 网络错误: %s" % e)
                if attempt < 2:
                    time.sleep(2)
                    continue
                raise last
        raise last or AudioProviderError("Mureka 请求失败")

    def _upload_reference(self, cfg, wav_path):
        """参考音频 -> reference_id（multipart 上传，格式以官方文档为准）。"""
        boundary = "----SongWeaverBoundary%s" % int(time.time())
        fname = os.path.basename(wav_path)
        with open(wav_path, "rb") as f:
            payload = f.read()
        body = b""
        body += ("--%s\r\nContent-Disposition: form-data; name=\"file\"; "
                 "filename=\"%s\"\r\nContent-Type: audio/wav\r\n\r\n"
                 % (boundary, fname)).encode("utf-8")
        body += payload + b"\r\n"
        body += ("--%s--\r\n" % boundary).encode("utf-8")
        resp = self._request(
            "POST", cfg["base"] + "/v1/files/upload", cfg, raw=body,
            content_type="multipart/form-data; boundary=%s" % boundary, timeout=120)
        rid = _task_id(resp) or ""
        if not rid:
            rid = (resp.get("reference_id") or "") if isinstance(resp, dict) else ""
        if not rid:
            raise AudioProviderError("Mureka 参考音频上传失败: %s" % str(resp)[:120])
        return rid

    # ------------------------------------------------------------ 生成
    def generate(self, req, spec, out_wav, progress_cb=None, cancel_event=None):
        progress_cb = progress_cb or (lambda frac, note: None)
        cfg = load_mureka_config(self.config_path)
        if not cfg["api_key"]:
            raise AudioProviderError("未配置 Mureka API Key（设置 → 云 API）")

        payload = {"model": cfg["model"]}
        desc = (req.get("description") or spec.get("description") or "").strip()
        if desc:
            payload["prompt"] = desc[:300]
        # 歌词：带段落标记的纯文本（Mureka 支持结构化歌词）
        lines = []
        for sec in spec.get("structure", []):
            label = {"verse": "Verse", "chorus": "Chorus", "bridge": "Bridge",
                     "prechorus": "Prechorus", "intro": "Intro", "outro": "Outro"}.get(
                         sec.get("name"), sec.get("name") or "")
            if label:
                lines.append("[%s]" % label)
            lines.extend(l for l in sec.get("lyrics", []) if l and l.strip())
        if lines:
            payload["lyrics"] = "\n".join(lines)
        gender = (cfg.get("vocal_gender") or "").strip()
        if gender in ("female", "male"):
            payload["vocal_gender"] = gender

        if req.get("ref_audio_path"):
            progress_cb(0.05, "上传参考音频到 Mureka…")
            payload["reference_id"] = self._upload_reference(cfg, req["ref_audio_path"])

        progress_cb(0.10, "提交 Mureka 生成任务…")
        resp = self._request("POST", cfg["base"] + "/v1/song/generate", cfg,
                             body=payload, timeout=120)
        tid = _task_id(resp)
        if not tid:
            raise AudioProviderError("Mureka 未返回任务 id: %s" % str(resp)[:160])

        # 轮询（约 1-2 分钟）
        deadline = time.time() + POLL_TIMEOUT_MIN * 60
        t0 = time.time()
        audio_url = None
        while time.time() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError("已取消")
            time.sleep(POLL_INTERVAL)
            state_obj = self._request("GET", "%s/v1/song/query/%s" % (cfg["base"], tid),
                                      cfg, timeout=60)
            state = _task_state(state_obj).lower()
            elapsed = time.time() - t0
            progress_cb(min(0.85, 0.2 + elapsed / 150.0),
                        "Mureka 云端生成中…（已等待 %d 秒）" % int(elapsed))
            if state in ("succeeded", "success", "done", "complete", "completed"):
                audio_url = _find_first_url(state_obj, ".mp3")
                break
            if state in ("failed", "error", "cancelled", "canceled"):
                raise AudioProviderError("Mureka 生成失败: %s" % str(state_obj)[:160])
        if not audio_url:
            raise AudioProviderError("Mureka 轮询超时（%d 分钟）" % POLL_TIMEOUT_MIN)

        progress_cb(0.90, "下载云端音频…")
        mp3 = out_wav + ".tmp.mp3"
        urllib.request.urlretrieve(audio_url, mp3)

        progress_cb(0.96, "转码 WAV…")
        if not shutil.which("ffmpeg"):
            raise AudioProviderError("未安装 ffmpeg，无法转换 Mureka 音频（brew install ffmpeg）")
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3, "-ar", "44100", "-ac", "2",
             "-sample_fmt", "s16", out_wav],
            capture_output=True, timeout=300, check=True)
        os.remove(mp3)
        if not os.path.isfile(out_wav):
            raise AudioProviderError("Mureka 音频转码失败")
        progress_cb(1.0, "完成")


provider = MurekaProvider()
