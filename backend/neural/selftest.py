"""DiffRhythm 神经引擎自检（在 DiffRhythm/.venv-neural 的 python 里运行，cwd=DiffRhythm 根目录）。

    python backend/neural/selftest.py           # 模型加载 + 歌词编码 + 风格编码（约 1-2 分钟）
    python backend/neural/selftest.py --mini    # 再跑一次 2 步微型采样+解码（出真音频）
    python backend/neural/selftest.py --verify-euler  # 校验 runner 的手动 Euler 与上游 odeint 一致

全部通过打印 PASS 并退出码 0；任何一步失败打印 FAIL 与原因，退出码 1。
"""
import os
import resource
import sys
import time
import wave

os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), "pretrained", "hf"))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "infer"))

RESULTS = []


def step(name, fn):
    t0 = time.time()
    try:
        out = fn()
        RESULTS.append((name, True, "%.1fs" % (time.time() - t0), out))
        print("  ✅ %s (%.1fs) %s" % (name, time.time() - t0, out or ""))
        return True
    except Exception as e:  # noqa: BLE001
        RESULTS.append((name, False, "%.1fs" % (time.time() - t0), str(e)))
        print("  ❌ %s: %s" % (name, e))
        import traceback
        traceback.print_exc()
        return False


def rss_mb():
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / 1048576.0 if sys.platform == "darwin" else v / 1024.0  # macOS: bytes


def main():
    mini = "--mini" in sys.argv
    verify_euler = "--verify-euler" in sys.argv
    print("=== DiffRhythm 神经引擎自检 ===")
    print("cwd: %s" % os.getcwd())
    for f in ("pretrained/cfm_model.pt", "pretrained/vae_model.pt",
              "pretrained/muq/pytorch_model.bin"):
        assert os.path.isfile(f), "缺少权重 %s（cwd 必须是 DiffRhythm 根目录）" % f

    import torch
    print("torch %s | threads=%d | rss=%.0fMB" % (torch.__version__,
                                                  torch.get_num_threads(), rss_mb()))

    ctx = {}

    def load_models():
        from diffrhythm_runner import pick_device
        device = pick_device()
        from infer_utils import prepare_model
        t0 = time.time()
        cfm, tokenizer, muq, vae = prepare_model(2048, device)
        if device == "cpu":
            cfm = cfm.float()
        dt = next(cfm.parameters()).dtype
        ctx.update(cfm=cfm, tokenizer=tokenizer, muq=muq, vae=vae, device=device, dt=dt)
        return "device=%s 加载 %.1fs, rss=%.0fMB" % (device, time.time() - t0, rss_mb())

    def tokenize():
        from infer_utils import get_lrc_token
        lrc = "[00:00.50]测试第一句\n[00:05.00]测试第二句\n[01:34.90]啊"
        lrc_emb, start_time, end_frame, song_dur = get_lrc_token(
            2048, lrc, ctx["tokenizer"], 95, ctx.get("device", "cpu"))
        assert lrc_emb.shape == (1, 2048), lrc_emb.shape
        assert end_frame == 2048
        ctx["lrc_emb"] = lrc_emb
        ctx["start_time"] = start_time.to(ctx["dt"])
        ctx["song_dur"] = song_dur.to(ctx["dt"])
        return "tokens=%d end_frame=%d" % (int((lrc_emb > 0).sum()), end_frame)

    def style_text():
        from infer_utils import get_style_prompt, get_negative_style_prompt
        sp = get_style_prompt(ctx["muq"], prompt="pop, uplifting, mandarin song, female vocal")
        neg = get_negative_style_prompt(ctx.get("device", "cpu"))
        ctx["style"] = sp.to(ctx["dt"])
        ctx["neg"] = neg.to(ctx["dt"])
        return "style=%s neg=%s" % (tuple(sp.shape), tuple(neg.shape))

    # ---- Euler 等价性校验（可选）：手动逐步 vs 上游 odeint ----
    def euler_check():
        from infer_utils import get_reference_latent
        cfm = ctx["cfm"]
        cond, pred_frames = get_reference_latent(ctx.get("device", "cpu"), 2048, False, None, None, ctx["vae"])
        cond = cond.to(ctx["dt"])
        step_cond = torch.zeros_like(cond)
        end = 2048
        text = ctx["lrc_emb"]
        sp, neg = ctx["style"], ctx["neg"]
        st, sd = ctx["start_time"], ctx["song_dur"]

        def fn(t, x):
            pred = cfm.transformer(x=x, cond=step_cond, text=text, time=t,
                                   drop_audio_cond=False, drop_text=False, drop_prompt=False,
                                   style_prompt=sp, start_time=st, duration=sd)
            null = cfm.transformer(x=x, cond=step_cond, text=text, time=t,
                                   drop_audio_cond=True, drop_text=True, drop_prompt=False,
                                   style_prompt=neg, start_time=st, duration=sd)
            return pred + (pred - null) * 4.0

        torch.manual_seed(7)
        # 上游路径
        latents, _ = cfm.sample(cond=cond, text=text, duration=end,
                                style_prompt=sp, max_duration=end, song_duration=sd,
                                negative_style_prompt=neg, steps=4, cfg_strength=4.0,
                                start_time=st, latent_pred_segments=[(0, end)],
                                batch_infer_num=1, seed=7)
        ref = latents[0]
        # 手动 Euler（runner 路径；seed 在 randn 前设置，与上游 sample 内部一致）
        torch.manual_seed(7)
        with torch.inference_mode():
            x = torch.randn(end, cfm.num_channels, device=cond.device, dtype=step_cond.dtype).unsqueeze(0)
            t_grid = torch.linspace(0, 1, 4, device=cond.device, dtype=step_cond.dtype)
            for i in range(3):
                x = x + (t_grid[i + 1] - t_grid[i]) * fn(t_grid[i], x)
        diff = (x - ref).abs().max().item()
        assert diff < 1e-3, "手动 Euler 与 odeint 偏差过大: %g" % diff
        return "max|Δ|=%.2e（等价）" % diff

    def mini_gen():
        from infer_utils import decode_audio, get_reference_latent
        cfm, vae = ctx["cfm"], ctx["vae"]
        cond, _ = get_reference_latent(ctx.get("device", "cpu"), 2048, False, None, None, vae)
        cond = cond.to(ctx["dt"])
        end = 2048
        step_cond = torch.zeros_like(cond)
        sp, neg = ctx["style"], ctx["neg"]
        st, sd = ctx["start_time"], ctx["song_dur"]

        def fn(t, x):
            pred = cfm.transformer(x=x, cond=step_cond, text=ctx["lrc_emb"], time=t,
                                   drop_audio_cond=False, drop_text=False, drop_prompt=False,
                                   style_prompt=sp, start_time=st, duration=sd)
            null = cfm.transformer(x=x, cond=step_cond, text=ctx["lrc_emb"], time=t,
                                   drop_audio_cond=True, drop_text=True, drop_prompt=False,
                                   style_prompt=neg, start_time=st, duration=sd)
            return pred + (pred - null) * 4.0

        torch.manual_seed(42)
        with torch.inference_mode():
            x = torch.randn(end, cfm.num_channels, device=cond.device, dtype=step_cond.dtype).unsqueeze(0)
            t_grid = torch.linspace(0, 1, 3, device=cond.device, dtype=step_cond.dtype)
            t0 = time.time()
            for i in range(2):
                x = x + (t_grid[i + 1] - t_grid[i]) * fn(t_grid[i], x)
            sample_sec = time.time() - t0
            latent = x.to(torch.float32).transpose(1, 2)
            out = decode_audio(latent, vae, chunked=True).reshape(2, -1)
            out = (out / out.abs().max().clamp(min=1e-8)).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        arr = out.numpy().T.astype("<i2")
        path = os.path.join(os.getcwd(), "test_output", "selftest.wav")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with wave.open(path, "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(44100)
            w.writeframes(arr.tobytes())
        return "2 步采样 %.1fs → %s（%.0fMB rss；正式 16 步约为其 8 倍时长）" % (
            sample_sec, path, rss_mb())

    ok = True
    ok &= step("加载模型 (cfm/vae/muq)", load_models)
    ok &= step("歌词编码 g2p + LRC token", tokenize)
    ok &= step("风格编码 (MuQ-MuLan 文本)", style_text)
    if verify_euler:
        ok &= step("Euler 等价性校验", euler_check)
    if mini:
        ok &= step("微型生成 (2 步采样+VAE 解码)", mini_gen)

    print()
    if ok:
        print("PASS ✅ 神经引擎可用。真实生成约 15-30 分钟/首（16 步，取决于本机负载）。")
        sys.exit(0)
    print("FAIL ❌ 存在失败步骤，请把上方错误信息发给维护者或查阅 docs/CONFIG.md")
    sys.exit(1)


if __name__ == "__main__":
    main()
