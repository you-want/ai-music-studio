"""音频成品 Provider 抽象：产出"成品 WAV"而非 SongSpec 渲染。

普通 Provider（mock/llm）返回高层乐谱，由本地 realizer + synth 渲染；
音频类 Provider（DiffRhythm 本地模型、未来的 Mureka/Suno 云 API）直接
产出带人声演唱的成品音频，jobs.py 渲染阶段整体旁路。

接口约定：
    available() -> (bool, reason)         # 引擎是否可用（前端据此禁用选项）
    generate(req, spec, out_wav, progress_cb, cancel_event)
                                          # 阻塞式生成，把 44.1kHz WAV 写到 out_wav
                                          # progress_cb(frac: 0..1, note: str)
                                          # cancel_event.set() 后应尽快中止并抛出
"""
import threading


class AudioProviderError(RuntimeError):
    """音频生成失败（网络/模型/超时/被取消）。"""


class CancelledError(AudioProviderError):
    """任务被用户取消。"""


class AudioProvider:
    name = "audio"

    def available(self):
        """-> (available: bool, reason: str)。reason 在不可用时给人看。"""
        raise NotImplementedError

    def generate(self, req, spec, out_wav, progress_cb=None,
                 cancel_event=None):
        """生成成品音频。失败抛 AudioProviderError（上层转任务错误）。"""
        raise NotImplementedError


def make_cancel_event():
    return threading.Event()
