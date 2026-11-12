"""音色 Persona 嵌入提取子进程（跑在 DiffRhythm/.venv-neural，只加载 MuQ ~2.4GB）。

用法： python persona_runner.py <job.json>
job.json: {"wav": "...", "text": "...", "out": "/abs/persona.npy"}
    wav 与 text 二选一：wav 取中段 10s（24kHz 单声道）走 MuQ 音频编码；
    text 直接走 MuQ 文本编码（同一对比学习空间）。
输出嵌入为 L2 归一化的 [1,512] float32 .npy，可经 style_prompt_npy 直接替换
生成任务的文本风格向量。

stdout: {"type":"done","out":"...","dim":512} / {"type":"error","error":"..."}
"""
import json
import os
import sys

os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), "pretrained", "hf"))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "infer"))

import numpy as np  # noqa: E402

try:
    # espeak-ng 动态库/数据路径（espeakng-loader 自带多平台二进制）
    import espeakng_loader
    os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", espeakng_loader.get_library_path())
    os.environ.setdefault("ESPEAK_DATA_PATH",
                            os.path.dirname(espeakng_loader.get_data_path()))
    os.environ.setdefault("PHONEMIZER_ESPEAK_PATH", os.path.join(
        os.path.dirname(espeakng_loader.get_library_path()), "espeak-ng"))
except ImportError:
    pass

import torch  # noqa: E402


def main():
    job = json.load(open(sys.argv[1], "r", encoding="utf-8"))

    def emit(p):
        sys.stdout.write(json.dumps(p, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    from muq import MuQMuLan
    mulan = MuQMuLan.from_pretrained("./pretrained/muq").eval()

    if job.get("wav"):
        import librosa
        wav_path = job["wav"]
        dur = librosa.get_duration(path=wav_path)
        if dur < 10:
            raise ValueError("音频太短（%.1fs），需要 >=10 秒" % dur)
        mid = dur // 2
        wav, _ = librosa.load(wav_path, sr=24000, offset=mid - 5, duration=10)
        with torch.no_grad():
            emb = mulan(wavs=torch.tensor(wav).unsqueeze(0))
    else:
        text = (job.get("text") or "").strip()
        if not text:
            raise ValueError("需要 wav 或 text 之一")
        with torch.no_grad():
            emb = mulan(texts=[text])

    emb = emb.detach().cpu().float().numpy()  # [1, 512]，内部已 L2 归一
    np.save(job["out"], emb)
    emit({"type": "done", "out": job["out"], "dim": int(emb.shape[-1])})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        try:
            sys.stdout.write(json.dumps({"type": "error", "error": "%s: %s" % (type(e).__name__, e)}) + "\n")
        except Exception:
            pass
        sys.exit(1)
