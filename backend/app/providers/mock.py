"""Mock Provider：离线算法作曲，永远可用、零费用。"""
import time

from ..music import mock_composer


def provide(req):
    t0 = time.time()
    spec = mock_composer.compose(req)
    return spec, {"provider": "mock", "elapsed_ms": int((time.time() - t0) * 1000)}
