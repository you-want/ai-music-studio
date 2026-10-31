"""音色 Persona 存储与提取编排。

一个 Persona = 某首歌/某段音频的"嗓音+曲风"嵌入（MuQ-MuLan [1,512] 向量），
存储于 <曲库>/personas/<id>/：embedding.npy + meta.json。
生成神经歌曲时把 embedding.npy 作为 style_prompt 下发，实现"同一副嗓子反复出歌"。
"""
import json
import os
import time
import uuid

from .providers import diffrhythm as dr


def personas_dir(lib):
    d = os.path.join(lib.lib_dir, "personas")
    os.makedirs(d, exist_ok=True)
    return d


def _meta_path(lib, persona_id):
    return os.path.join(personas_dir(lib), persona_id, "meta.json")


def _emb_path(lib, persona_id):
    return os.path.join(personas_dir(lib), persona_id, "embedding.npy")


def list_personas(lib):
    out = []
    root = personas_dir(lib)
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return out
    for name in names:
        mp = os.path.join(root, name, "meta.json")
        if os.path.isfile(mp):
            try:
                with open(mp, "r", encoding="utf-8") as f:
                    out.append(json.load(f))
            except (OSError, ValueError):
                continue
    out.sort(key=lambda m: -m.get("created_at", 0))
    return out


def get(lib, persona_id):
    safe = "".join(c for c in str(persona_id) if c.isalnum() or c in "-_")
    mp = _meta_path(lib, safe)
    if not os.path.isfile(mp):
        return None
    with open(mp, "r", encoding="utf-8") as f:
        return json.load(f)


def delete(lib, persona_id):
    import shutil
    safe = "".join(c for c in str(persona_id) if c.isalnum() or c in "-_")
    d = os.path.join(personas_dir(lib), safe)
    if os.path.isdir(d):
        shutil.rmtree(d)
        return True
    return False


def _extract_embedding(lib, wav_path=None, text=None, timeout_min=10):
    """跑 persona_runner 子进程，返回嵌入 npy 绝对路径。失败抛 RuntimeError。"""
    import tempfile
    cfg = dr.load_neural_config()
    py = cfg.get("python") or dr.default_python()
    if not os.path.isfile(py):
        raise RuntimeError("神经引擎 venv 不存在——先运行 scripts/setup_neural.sh")
    pid = uuid.uuid4().hex[:10]
    tmp_json = os.path.join(tempfile.gettempdir(), "persona-%s.json" % pid)
    out_npy = os.path.join(tempfile.gettempdir(), "persona-%s.npy" % pid)
    job = {"out": out_npy, "_job_path": tmp_json}
    if wav_path:
        job["wav"] = os.path.abspath(wav_path)
    else:
        job["text"] = text
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False)
    try:
        done = dr._run_subprocess_job(dr.PERSONA_RUNNER, job, timeout_min, None)
        if not done or not os.path.isfile(out_npy):
            raise RuntimeError("persona 嵌入未产出")
        return out_npy
    finally:
        try:
            os.remove(tmp_json)
        except OSError:
            pass


def _pick_song_vocal(lib, song_id):
    """优先人声分轨（demucs 已分离），否则整曲。"""
    d = lib.song_dir(song_id)
    vocal = os.path.join(d, "stem_vocal.wav")
    full = lib.wav_path(song_id)
    if os.path.isfile(vocal):
        return vocal
    if os.path.isfile(full):
        return full
    raise RuntimeError("作品不存在或缺少音频")


def create(lib, name, source, config_path=None,
           song_id=None, ref_audio_name=None, text=None):
    """source: "song" | "audio" | "text"。返回 meta。"""
    name = (str(name) or "").strip()[:24] or "未命名音色"
    if source == "song":
        if not song_id:
            raise ValueError("缺少 song_id")
        wav = _pick_song_vocal(lib, song_id)
        npy = _extract_embedding(lib, wav_path=wav)
        src = {"type": "song", "song_id": song_id}
    elif source == "audio":
        if not ref_audio_name:
            raise ValueError("缺少 ref_audio")
        safe = "".join(c for c in str(ref_audio_name) if c.isalnum() or c in "-_.")
        path = lib.ref_path(safe)
        if not safe or not os.path.isfile(path):
            raise RuntimeError("参考音频不存在，请重新上传")
        npy = _extract_embedding(lib, wav_path=path)
        src = {"type": "audio", "ref_audio": safe}
    elif source == "text":
        if not (text or "").strip():
            raise ValueError("缺少文字描述")
        npy = _extract_embedding(lib, text=text.strip())
        src = {"type": "text"}
    else:
        raise ValueError("未知来源: %s" % source)

    persona_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    d = os.path.join(personas_dir(lib), persona_id)
    os.makedirs(d, exist_ok=True)
    os.replace(npy, _emb_path(lib, persona_id))
    meta = {"id": persona_id, "name": name, "source": src,
            "created_at": int(time.time())}
    with open(_meta_path(lib, persona_id), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    return meta


def embedding_path(lib, persona_id):
    p = get(lib, persona_id)
    if not p:
        return None
    path = _emb_path(lib, persona_id)
    return path if os.path.isfile(path) else None
