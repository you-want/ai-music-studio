"""Demucs 源分离子进程（跑在 DiffRhythm/.venv-neural）：整曲 WAV -> 4 分轨。

用法： python demucs_runner.py <job.json>
job.json: {"wav": "...", "out_dir": "...", "model": "htdemucs"}
输出: out_dir/vocals.wav drums.wav bass.wav other.wav（重采样为原曲格式不变）
stdout: {"type":"done","stems":{...}} / {"type":"error",...}
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile


def main():
    job = json.load(open(sys.argv[1], "r", encoding="utf-8"))
    wav = job["wav"]
    out_dir = job["out_dir"]
    model = job.get("model") or "htdemucs"
    emit = lambda p: (sys.stdout.write(json.dumps(p, ensure_ascii=False) + "\n"),
                      sys.stdout.flush())

    tmp = tempfile.mkdtemp(prefix="demucs-out-")
    try:
        # demucs CLI: <tmp>/<model>/<track名>/vocals.wav ...
        cmd = [sys.executable, "-m", "demucs", "-n", model, "--device", "cpu",
               "-o", tmp, wav]
        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", str(min(8, os.cpu_count() or 4)))
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError("demucs 失败: %s" % proc.stderr[-400:])
        track_dir = os.path.join(tmp, model, os.path.splitext(os.path.basename(wav))[0])
        os.makedirs(out_dir, exist_ok=True)
        stems = {}
        for name, target in (("vocals", "stem_vocal.wav"), ("drums", "stem_drums.wav"),
                             ("bass", "stem_bass.wav"), ("other", "stem_harmony.wav")):
            src = os.path.join(track_dir, name + ".wav")
            if os.path.isfile(src):
                shutil.move(src, os.path.join(out_dir, target))
                stems[name] = target
        if "vocals" not in stems:
            raise RuntimeError("demucs 未产出 vocals 分轨")
        emit({"type": "done", "stems": stems})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        try:
            sys.stdout.write(json.dumps({"type": "error", "error": "%s: %s" % (type(e).__name__, e)}) + "\n")
        except Exception:
            pass
        sys.exit(1)
