# DiffRhythm vendoring 说明（NOTICE）

本目录（`DiffRhythm/`）包含 [ASLP-lab/DiffRhythm](https://github.com/ASLP-lab/DiffRhythm)
的 vendored 副本（Apache License 2.0），用于「织歌 SongWeaver」的本地神经歌声引擎。

- 上游版权：Copyright (c) 2025 ASLP-Lab，许可证见各源文件头部与本仓库根目录 `LICENSE`
- 本目录不含模型权重（`pretrained/` 已被 .gitignore 排除），权重请按 `scripts/setup_neural.sh` 与 `docs/CONFIG.md` 获取

## 相对上游的修改（Apache-2.0 要求声明）

1. `infer/infer_utils.py::prepare_model`
   按 max_frames 选择本地权重文件（2048 → `cfm_model.pt`；6144 → `cfm_model_full.pt`），
   避免两个模型变体互相错载。
2. 其余推理代码未改动。CPU fp32、逐步采样进度、LRC 生成、对齐与分轨等能力
   均在本仓库 `backend/neural/` 与 `backend/app/providers/` 中实现，不修改上游文件。
