#!/bin/bash
# 神经歌声引擎（DiffRhythm）一键安装：Python 3.11 + torch 2.2.2 CPU + 依赖 + HF 缓存预热
# 幂等：可重复执行。安装后运行 backend/neural/selftest.py 验证。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIFFRHYTHM="$ROOT/DiffRhythm"
VENV="$DIFFRHYTHM/.venv-neural"
PY="$VENV/bin/python"

echo "==> 0/5 检查 uv（Python 版本管理器）"
if command -v uv >/dev/null 2>&1 && [ "$(uname -m)" = "arm64" ] \
   && [ "$(uv --version 2>/dev/null)" != "$(uv --version 2>/dev/null | grep -o arm64)" ] \
   && uv --version 2>/dev/null | grep -q x86_64; then
  echo "    uv 为 x86_64（Rosetta 运行），换装 arm64 原生版…"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
  if [ -x "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
  else
    echo "    未找到 uv，正在安装…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi
echo "    uv: $(uv --version)"

echo "==> 1/5 安装 Python 3.11（Intel Mac 上 torch 最高 2.2.2，需要 Python ≤3.12）"
uv python install 3.11

echo "==> 2/5 创建 venv: $VENV"
MACH=$(uname -m)
VARCH=""
if [ -x "$PY" ]; then
  VARCH=$(file -L "$PY" 2>/dev/null | grep -oE "arm64|x86_64" | head -1 || true)
fi
if [ ! -x "$PY" ]; then
  uv venv --python 3.11 "$VENV"
elif [ -n "$VARCH" ] && [ "$VARCH" != "$MACH" ]; then
  echo "    现有 venv 为 ${VARCH}，与机器 ($MACH) 不符 —— 重建（换机迁移）"
  rm -rf "$VENV"
  uv python uninstall 3.11 --all-arches 2>/dev/null || true
  uv python install 3.11
  uv venv --python 3.11 "$VENV"
else
  echo "    已存在，跳过"
fi

echo "==> 3/5 安装 PyTorch 与依赖（首次约 3-6 分钟）"
# 国内网络可设 AI_MUSIC_UV_INDEX 加速（如清华镜像）
if [ -n "${AI_MUSIC_UV_INDEX:-}" ]; then
  export UV_DEFAULT_INDEX="$AI_MUSIC_UV_INDEX"
  echo "    使用镜像: $UV_DEFAULT_INDEX"
fi
COMMON_PKGS="py3langid==0.3.0 faster-whisper>=1.0 demucs>=4.0 transformers==4.49.0 tokenizers x-transformers==2.1.2 torchdiffeq==0.2.5 ema-pytorch einops accelerate muq==0.1.0 mutagen jieba pypinyin cn2an pykakasi phonemizer Unidecode inflect py3langid onnxruntime prefigure huggingface_hub safetensors"
if [ "$(uname -m)" = "arm64" ]; then
  # Apple Silicon：现代 torch + MPS GPU 加速；现代 numba/llvmlite 有 arm64 轮子
  uv pip install -q -p "$PY" \
    "torch>=2.6,<3" "torchaudio" "numpy" \
    "librosa==0.10.2.post1" \
    $COMMON_PKGS
else
  # Intel Mac：torch 上限 2.2.2；numba/llvmlite 必须用 0.60/0.43
  # （更新版不再发布 macOS x86_64 轮子，会触发源码编译失败）
  uv pip install -q -p "$PY" \
    "torch==2.2.2" "torchaudio==2.2.2" "numpy<2" \
    "numba==0.60.0" "llvmlite==0.43.0" "librosa==0.10.2.post1" \
    $COMMON_PKGS
fi

echo "==> 3c/5 补齐 g2p 大文件（多音字模型 80MB + 词典，仓库不入库、按需拉取）"
G2P_DIR="$DIFFRHYTHM/g2p/sources/g2p_chinese_model"
if [ ! -f "$G2P_DIR/poly_bert_model.onnx" ] || [ ! -f "$DIFFRHYTHM/g2p/sources/chinese_lexicon.txt" ]; then
  TMPG2P="$(mktemp -d)"
  if git clone --depth 1 https://github.com/ASLP-lab/DiffRhythm "$TMPG2P" 2>/dev/null; then
    mkdir -p "$G2P_DIR"
    cp -f "$TMPG2P/g2p/sources/g2p_chinese_model/"* "$G2P_DIR/" 2>/dev/null || true
    cp -f "$TMPG2P/g2p/sources/chinese_lexicon.txt" "$DIFFRHYTHM/g2p/sources/" 2>/dev/null || true
  fi
  rm -rf "$TMPG2P"
fi
if [ ! -f "$G2P_DIR/poly_bert_model.onnx" ]; then
  echo "    ⚠️ 缺少 $G2P_DIR/poly_bert_model.onnx"
  echo "       请从 https://github.com/ASLP-lab/DiffRhythm 的 g2p/sources 手动补齐"
fi

echo "==> 4/5 检查模型权重与 espeak-ng"
MISS=0
for f in pretrained/cfm_model.pt pretrained/vae_model.pt pretrained/muq/pytorch_model.bin; do
  if [ ! -f "$DIFFRHYTHM/$f" ]; then
    echo "    ⚠️ 缺少权重 $f"
    MISS=1
  fi
done
if [ "$MISS" = "1" ]; then
  echo "    请从 HuggingFace 下载（ASLP-lab/DiffRhythm-1_2 与 ASLP-lab/DiffRhythm-vae、OpenMuQ/MuQ-MuLan-large）"
  exit 1
fi
if ! command -v espeak-ng >/dev/null 2>&1; then
  echo "    ⚠️ 缺少 espeak-ng：brew install espeak-ng"
  exit 1
fi
echo "    权重与 espeak-ng 就绪"

echo "==> 3b/5 转换 MuQ 权重为 safetensors（新版 huggingface_hub 只认 model.safetensors）"
if [ ! -f "$DIFFRHYTHM/pretrained/muq/model.safetensors" ]; then
  "$PY" - "$DIFFRHYTHM" << 'EOF'
import os, sys, torch
from safetensors.torch import save_file
d = sys.argv[1]
src = os.path.join(d, "pretrained", "muq", "pytorch_model.bin")
dst = os.path.join(d, "pretrained", "muq", "model.safetensors")
sd = torch.load(src, map_location="cpu", weights_only=True)
save_file({k: v.contiguous() for k, v in sd.items()}, dst, metadata={"format": "pt"})
print("    转换完成:", dst)
EOF
else
  echo "    已存在，跳过"
fi

echo "==> 5/5 预热 HF 缓存（xlm-roberta tokenizer，避免首次生成时联网）"
HF_HOME="$DIFFRHYTHM/pretrained/hf" "$PY" - << 'EOF'
from huggingface_hub import snapshot_download
try:
    p = snapshot_download("xlm-roberta-base", allow_patterns=["tokenizer*", "sentencepiece*"])
    print("    tokenizer 缓存于:", p)
except Exception as e:
    print("    ⚠️ 预热失败（不影响安装，首次生成会自动重试）:", e)
EOF

echo
echo "✅ 神经引擎安装完成。验证："
echo "   $PY $ROOT/backend/neural/selftest.py          # 加载模型自检（约 1-2 分钟）"
echo "   $PY $ROOT/backend/neural/selftest.py --mini   # 再加一次 2 步微型生成"
echo "然后重启 织歌 SongWeaver，网页引擎选择「🧠 神经歌声」。"
