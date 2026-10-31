"""服务端封面生成：numpy 绘制 → 手写 PNG 编码（不依赖 PIL）。

与前端 canvas 封面同一套确定性算法（按 song_id 哈希），用于 MV 导出与分享包。
"""
import hashlib
import struct
import zlib

import numpy as np


def _hsv_to_rgb(h, s, v):
    """标量 HSV -> RGB(0..1)。"""
    h = (h % 360) / 60.0
    i = int(h)
    f = h - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    table = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    r, g, b = table[i % 6]
    return np.array([r, g, b], dtype=np.float32)


def _png_write(path, arr):
    """arr: HxWx3 uint8 -> PNG 文件（filter 0 行压缩）。"""
    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[i].tobytes() for i in range(h))

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)
    return path


def make_cover(song_id, size=1024, path=None):
    """按 song_id 确定性绘制封面，写入 path 并返回路径。"""
    h = int(hashlib.md5(str(song_id).encode("utf-8")).hexdigest()[:12], 16)
    ys, xs = np.mgrid[0:size, 0:size]
    x = (xs / size).astype(np.float32)
    y = (ys / size).astype(np.float32)
    hue1 = h % 360
    hue2 = (hue1 + 60 + (h >> 4) % 50) % 360
    hue3 = (hue1 + 190) % 360
    c1, c2 = _hsv_to_rgb(hue1, .72, .52), _hsv_to_rgb(hue2, .62, .30)
    t = ((x + y) / 2)[..., None]
    img = c1 * (1 - t) + c2 * t

    # 圆环装饰
    for i in range(4):
        cx = 0.22 + ((h >> i) & 7) / 11.0
        cy = 0.24 + ((h >> (i + 2)) & 7) / 11.0
        r = 0.10 + ((h >> (i + 3)) & 3) * 0.055
        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        ring = np.exp(-((d - r) / 0.012) ** 2)
        a = ((0.5 - i * 0.11) * ring)[..., None]
        img = img * (1 - a) + _hsv_to_rgb(hue3, .85, .72) * a

    # 均衡器条
    white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    for i in range(5):
        bx = 0.14 + i * 0.12
        bh = 0.08 + ((h >> (i * 3)) & 7) / 16.0
        m = ((x >= bx) & (x <= bx + 0.06) & (y >= 0.8 - bh) & (y <= 0.8))[..., None]
        img = img * (1 - m) + white * 0.55 * m

    out = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return _png_write(path, out)
