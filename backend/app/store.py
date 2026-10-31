"""曲库持久化：output/<song_id>/ 下保存 meta/spec/score 与音频文件。"""
import hashlib
import json
import os
import shutil
import time
import uuid

_FP_KEYS = ("description", "genre", "mood", "bpm", "duration_sec", "key", "mode",
            "provider", "seed", "lyrics", "lead_voice", "sing", "ref_audio",
            "full_song", "lyrics_req")


def fingerprint_of(req):
    """生成参数指纹：同参数(+seed)视为同一首（用于结果复用缓存）。"""
    payload = {k: req.get(k) for k in _FP_KEYS}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


class Library:
    def __init__(self, lib_dir):
        self.lib_dir = lib_dir
        os.makedirs(lib_dir, exist_ok=True)

    def new_song_id(self):
        return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    def song_dir(self, song_id):
        # 防路径穿越
        safe = "".join(c for c in song_id if c.isalnum() or c in "-_")
        return os.path.join(self.lib_dir, safe)

    def wav_path(self, song_id):
        return os.path.join(self.song_dir(song_id), "song.wav")

    def midi_path(self, song_id):
        return os.path.join(self.song_dir(song_id), "song.mid")

    def refs_dir(self):
        """参考音频库（神经引擎风格迁移用，服务端生成文件名）。"""
        d = os.path.join(self.lib_dir, "_refs")
        os.makedirs(d, exist_ok=True)
        return d

    def ref_path(self, name):
        return os.path.join(self.refs_dir(), name)

    def prepare(self, song_id):
        os.makedirs(self.song_dir(song_id), exist_ok=True)

    def save(self, song_id, spec, score, fingerprint=None):
        d = self.song_dir(song_id)
        os.makedirs(d, exist_ok=True)
        meta = {
            "id": song_id,
            "title": spec["title"],
            "description": spec.get("description", ""),
            "genre": spec["genre"], "mood": spec["mood"],
            "key": spec["key"], "mode": spec["mode"], "bpm": spec["bpm"],
            "duration_sec": score["duration_sec"],
            "total_bars": sum(s["bars"] for s in spec["structure"]),
            "provider": spec.get("provider", "mock"),
            "audio_only": bool(spec.get("audio_only")),
            "lyrics_source": spec.get("lyrics_source", ""),
            "sing": bool(spec.get("sing")),
            "favorite": False,
            "fingerprint": fingerprint,
            "created_at": int(time.time()),
            "sections": [{"name": s["name"], "bars": s["bars"]} for s in spec["structure"]],
        }
        self._write_json(os.path.join(d, "meta.json"), meta)
        self._write_json(os.path.join(d, "song.json"), spec)
        self._write_json(os.path.join(d, "score.json"), score)
        return meta

    @staticmethod
    def _write_json(path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @staticmethod
    def _read_json(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_songs(self):
        out = []
        try:
            names = sorted(os.listdir(self.lib_dir), reverse=True)
        except OSError:
            return out
        for name in names:
            meta_path = os.path.join(self.lib_dir, name, "meta.json")
            if os.path.isfile(meta_path):
                try:
                    out.append(self._read_json(meta_path))
                except (OSError, ValueError):
                    continue
        # 收藏优先，其余按创建时间倒序
        out.sort(key=lambda m: (not m.get("favorite", False), -m.get("created_at", 0)))
        return out

    def find_fingerprint(self, fp):
        """按参数指纹找已有作品（结果复用缓存）。"""
        if not fp:
            return None
        for meta in self.list_songs():
            if meta.get("fingerprint") == fp:
                return meta
        return None

    def rename(self, song_id, title):
        """改名：同步 meta / spec / score 三处标题。"""
        d = self.song_dir(song_id)
        title = str(title).strip()[:60]
        if not title:
            raise ValueError("标题不能为空")
        meta = self._read_json(os.path.join(d, "meta.json"))
        meta["title"] = title
        self._write_json(os.path.join(d, "meta.json"), meta)
        spec_path = os.path.join(d, "song.json")
        if os.path.isfile(spec_path):
            spec = self._read_json(spec_path)
            spec["title"] = title
            self._write_json(spec_path, spec)
        score_path = os.path.join(d, "score.json")
        if os.path.isfile(score_path):
            score = self._read_json(score_path)
            if isinstance(score.get("spec"), dict):
                score["spec"]["title"] = title
            self._write_json(score_path, score)
        return meta

    def set_favorite(self, song_id, favorite):
        meta_path = os.path.join(self.song_dir(song_id), "meta.json")
        meta = self._read_json(meta_path)
        meta["favorite"] = bool(favorite)
        self._write_json(meta_path, meta)
        return meta

    def exists(self, song_id):
        return os.path.isfile(os.path.join(self.song_dir(song_id), "meta.json"))

    def get_meta(self, song_id):
        return self._read_json(os.path.join(self.song_dir(song_id), "meta.json"))

    def get_spec(self, song_id):
        return self._read_json(os.path.join(self.song_dir(song_id), "song.json"))

    def get_score(self, song_id):
        return self._read_json(os.path.join(self.song_dir(song_id), "score.json"))

    def delete(self, song_id):
        d = self.song_dir(song_id)
        if os.path.isdir(d):
            shutil.rmtree(d)
            return True
        return False
