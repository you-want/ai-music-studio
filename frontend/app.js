/* 织歌 SongWeaver 前端逻辑（无框架）
 * v2 新增：简易/进阶双模式、自定义歌词、AI 写词、音色/调性选择、
 * 双版本候选、分轨下载、再来一版/延长、AI 封面生成 */
"use strict";

const $ = (id) => document.getElementById(id);

/* 局域网 Token 模式：首次打开 /?token=xxx 时换取 HttpOnly 会话 Cookie。 */
let URL_TOKEN = new URLSearchParams(location.search).get("token") || "";
function api(path, opts) {
  opts = opts || {};
  const headers = Object.assign({}, opts.headers || {});
  if (URL_TOKEN) headers["Authorization"] = "Bearer " + URL_TOKEN;
  return fetch(path, Object.assign({}, opts, { headers: headers }));
}
function mediaUrl(path) {
  return path;
}

async function establishSession() {
  if (!URL_TOKEN) return;
  const response = await fetch("/api/session", {
    method: "POST",
    headers: { Authorization: "Bearer " + URL_TOKEN },
  });
  if (!response.ok) throw new Error("访问 Token 无效");
  history.replaceState(null, "", location.pathname + location.hash);
  URL_TOKEN = "";
}

const state = {
  meta: null,
  selectedGenre: "auto",
  selectedMood: "auto",
  mode: "simple",
  currentSongId: null,
  currentParams: null,
  score: null,
  pxb: 12,
  bpm: 100,
  pollTimer: null,
  lastParams: null,   // 上次生成的参数（供 再来一版/延长 使用）
  engines: {},        // 引擎可用性（/api/meta.engines）
  personas: [],       // 音色 Persona 列表
  refAudio: null,     // 已上传参考音频文件名（神经引擎风格迁移）
  activeJobIds: [],   // 进行中的任务（供取消）
  jobStartedAt: 0,
  candidateSongIds: [],
  pollFailures: 0,
};

const ACTIVE_JOBS_KEY = "songweaver.activeJobs";

/* ================= 初始化 ================= */
document.addEventListener("DOMContentLoaded", async () => {
  bindEvents();
  try { await establishSession(); } catch (e) { console.error(e); }
  await loadMeta();
  await loadPersonas();
  await refreshConfigBadge();
  await refreshLibrary();
  // 深链：/?song=<id> 直接打开某首作品（&autoplay=1 自动播放；也便于自动化验证）
  const deepLink = new URLSearchParams(location.search);
  if (deepLink.get("song")) {
    try {
      await loadSong(deepLink.get("song"), deepLink.get("autoplay") === "1");
    } catch (e) { console.error("深链作品加载失败", e); }
  }
});

async function loadMeta() {
  try {
    const resp = await api("/api/meta");
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    state.meta = await resp.json();
  } catch (e) {
    console.error("meta 加载失败", e);
    return;
  }
  buildChips($("genre-chips"), [{ id: "auto", label: "自动" }]
    .concat(state.meta.genres), state.selectedGenre, (id) => { state.selectedGenre = id; });
  buildChips($("mood-chips"), [{ id: "auto", label: "自动" }]
    .concat(state.meta.moods), state.selectedMood, (id) => { state.selectedMood = id; });
  fillSelect($("voice"), state.meta.voices || []);
  fillSelect($("key-sel"), state.meta.keys || []);
  if ($("sing-row") && state.meta.sing_supported === false) {
    $("sing-row").classList.add("hidden");
  }
  // 神经歌声引擎可用性：不可用时禁用选项并提示安装方法
  state.engines = (state.meta && state.meta.engines) || {};
  const neural = state.engines.diffrhythm || {};
  const mopt = $("opt-mureka");
  if (mopt) {
    const mk = state.engines.mureka || {};
    if (mk.available === false) {
      mopt.disabled = true;
      mopt.title = mk.reason || "在设置里配置 Mureka API Key";
    } else if (mk.available === true) {
      mopt.title = mk.reason || "";
    }
  }
  const opt = $("opt-diffrhythm");
  if (opt) {
    if (neural.available === false) {
      opt.disabled = true;
      opt.textContent = "🧠 神经歌声（未安装）";
      opt.title = neural.reason || "运行 bash scripts/setup_neural.sh 安装";
    } else if (neural.available === true) {
      opt.title = neural.reason || "";
    }
  }
  // 完整版 4m45s 开关：仅当 full 权重就绪时显示（onProviderChange 会再按引擎联动）
  if ($("full-song-row")) {
    $("full-song-row").classList.toggle("hidden", neural.full_available !== true);
  }
}

const PROVIDER_BADGE = {
  local: "🎹 本地合成器",
  mock: "🎹 本地合成器",
  diffrhythm: "🧠 神经歌声·真人演唱",
  mureka: "☁️ Mureka 云端演唱",
};
function providerBadge(provider) {
  return PROVIDER_BADGE[provider] || "🧩 离线引擎";
}
function providerIcon(provider) {
  return { local: "🎹", mock: "🎹", diffrhythm: "🧠", mureka: "☁️" }[provider] || "🎹";
}

function composerLabel(provider) {
  return provider === "llm" ? "LLM 作曲" : "本地作曲";
}

function fillSelect(sel, items) {
  sel.innerHTML = "";
  items.forEach((it) => {
    const o = document.createElement("option");
    o.value = it.id;
    o.textContent = it.label;
    sel.appendChild(o);
  });
}

function buildChips(container, items, activeId, onPick) {
  container.innerHTML = "";
  items.forEach((it) => {
    const el = document.createElement("button");
    el.type = "button";
    el.className = "chip" + (it.id === activeId ? " active" : "");
    el.textContent = it.label;
    el.onclick = () => {
      container.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
      el.classList.add("active");
      onPick(it.id);
    };
    container.appendChild(el);
  });
}

/* ================= 创作参数收集 ================= */
function collectParams() {
  const p = {
    description: $("prompt").value.trim(),
    genre: state.selectedGenre,
    mood: state.selectedMood,
    duration_sec: parseInt($("duration").value, 10),
    bpm: parseInt($("bpm").value, 10) || null,
    provider: $("provider").value,
    composer_provider: $("composer-provider").value,
  };
  if (state.mode === "advanced") {
    p.title = $("title").value.trim() || null;
    p.lyrics = $("lyrics").value.trim() || null;
    p.lyrics_req = $("lyrics-req").value.trim() || null;
    p.lead_voice = $("voice").value !== "auto" ? $("voice").value : null;
    p.key = $("key-sel").value !== "auto" ? $("key-sel").value : null;
    p.sing = !!$("sing").checked;
  }
  if (p.provider === "diffrhythm") {
    if (state.refAudio) p.ref_audio = state.refAudio;
    if ($("full-song") && $("full-song").checked) p.full_song = true;
    const pid = $("persona-sel") ? $("persona-sel").value : "";
    if (pid && pid !== "none") p.persona_id = pid;
  }
  return p;
}

/* ================= 生成流程 ================= */
async function generate(overrides, sourceParams) {
  const params = Object.assign({}, sourceParams || collectParams(), overrides || {});
  if (!params.description && !params.lyrics && params.genre === "auto") {
    params.description = "一段随机的心情小曲";
  }
  state.lastParams = params;
  if (params.provider === "mureka") {
    const tracks = state.mode === "advanced" && $("dual").checked && !overrides ? 2 : 1;
    const composerNote = params.composer_provider === "llm"
      ? "，并发送到你配置的 LLM 服务商生成作曲设计" : "";
    if (!confirm("将把创作描述和歌词发送给 Mureka 云端生成" + (tracks > 1 ? " 2 首作品" : "作品")
      + composerNote + "。服务商可能按次收费，费用以其实际计费为准。继续吗？")) return;
  } else if (params.composer_provider === "llm") {
    if (!confirm("将把创作描述和歌词发送到你配置的 LLM 服务商用于作曲。继续吗？")) return;
  }
  $("candidate-switch").classList.add("hidden");
  state.candidateSongIds = [];
  const btn = $("btn-generate");
  btn.disabled = true;
  showJob(true, "提交任务…", 0);

  const dual = state.mode === "advanced" && $("dual").checked && !overrides;
  try {
    const ids = [];
    if (dual) {
      const seed = Math.floor(Math.random() * 1e9);
      for (const k of [0, 1]) {
        const resp = await api("/api/compose", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(Object.assign({}, params, { seed: seed + k * 7 })),
        });
        const data = await resp.json();
        if (!data.job_id) throw new Error(data.error || "提交失败");
        ids.push(data.job_id);
      }
    } else {
      const resp = await api("/api/compose", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
      });
      const data = await resp.json();
      if (!data.job_id) throw new Error(data.error || "提交失败");
      ids.push(data.job_id);
    }
    state.activeJobIds = ids;
    sessionStorage.setItem(ACTIVE_JOBS_KEY, JSON.stringify(ids));
    state.jobStartedAt = Date.now();
    state.pollFailures = 0;
    $("btn-job-cancel").classList.remove("hidden");
    pollJobs(ids, dual ? ["版本 A", "版本 B"] : null);
  } catch (e) {
    alert("出错了：" + e.message);
    btn.disabled = false;
    showJob(false);
  }
}

async function cancelJob() {
  const ids = state.activeJobIds || [];
  for (const id of ids) {
    try { await api("/api/jobs/" + id, { method: "DELETE" }); } catch (e) { /* 忽略 */ }
  }
}

function pollJobs(jobIds, labels) {
  clearTimeout(state.pollTimer);
  const poll = async () => {
    try {
      const jobs = [];
      for (const id of jobIds) {
        jobs.push(await (await api("/api/jobs/" + id)).json());
      }
      const done = jobs.filter((j) => j.status === "done");
      const failed = jobs.filter((j) => j.status === "error");
      const canceled = jobs.filter((j) => j.status === "canceled");
      const running = jobs.filter((j) => !["done", "error", "canceled"].includes(j.status));
      const avg = Math.round(jobs.reduce((s, j) => s + (j.progress || 0), 0) / jobs.length * 100);
      let stage;
      if (jobs.length > 1 && labels && labels.length === jobs.length) {
        // 双版本候选：按 A/B 分别展示
        stage = labels.map((lb, i) => lb + "：" + (jobs[i].stage_note || jobs[i].stage)).join(" ／ ");
      } else if (jobs.length > 1) {
        // 专辑等多任务：聚合展示（已完成 N/M + 当前阶段的第一个任务）
        const cur = running[0] || failed[0] || canceled[0] || jobs[0];
        stage = "共 " + jobs.length + " 个任务（已完成 " + done.length + "）· "
          + (cur.stage_note || cur.stage || "…");
      } else {
        stage = jobs[0].stage_note || jobs[0].stage;
      }
      // 神经渲染耗时展示（CPU 慢，给用户时间预期）
      const neural = jobs.some((j) => j.stage === "singing_neural" && j.status === "running");
      if (neural && state.jobStartedAt) {
        const s = Math.floor((Date.now() - state.jobStartedAt) / 1000);
        stage += "（已 " + Math.floor(s / 60) + " 分 " + (s % 60) + " 秒）";
      }
      showJob(true, stage, avg);

      if (canceled.length === jobs.length) {
        clearTimeout(state.pollTimer);
        state.activeJobIds = [];
        $("btn-job-cancel").classList.add("hidden");
        $("btn-generate").disabled = false;
        showJob(false);
        sessionStorage.removeItem(ACTIVE_JOBS_KEY);
      } else if (done.length === jobs.length) {
        clearTimeout(state.pollTimer);
        state.activeJobIds = [];
        $("btn-job-cancel").classList.add("hidden");
        $("btn-generate").disabled = false;
        setTimeout(() => showJob(false), 1200);
        state.candidateSongIds = done.map((j) => j.result.song_id);
        if (done.length > 1) {
          $("candidate-switch").classList.remove("hidden");
          const selectCandidate = async (index) => {
            [$("candidate-a"), $("candidate-b")].forEach((button, i) => {
              button.classList.toggle("active", i === index);
              button.setAttribute("aria-pressed", i === index ? "true" : "false");
            });
            try { await loadSong(state.candidateSongIds[index], false); }
            catch (e) { alert("候选作品加载失败：" + e.message); }
          };
          $("candidate-a").onclick = () => selectCandidate(0);
          $("candidate-b").onclick = () => selectCandidate(1);
          $("candidate-a").classList.add("active");
          $("candidate-a").setAttribute("aria-pressed", "true");
          $("candidate-b").classList.remove("active");
          $("candidate-b").setAttribute("aria-pressed", "false");
        }
        try { await loadSong(done[0].result.song_id, true); }
        catch (e) { showJob(true, "作品已生成，但加载失败：" + e.message, 100); }
        await refreshLibrary();
        sessionStorage.removeItem(ACTIVE_JOBS_KEY);
      } else if ((failed.length || canceled.length) && !running.length) {
        clearTimeout(state.pollTimer);
        state.activeJobIds = [];
        $("btn-job-cancel").classList.add("hidden");
        $("btn-generate").disabled = false;
        if (failed.length) showJob(true, "❌ " + (failed[0].error || "生成失败"), 100);
        if (done.length) {
          try { await loadSong(done[0].result.song_id, true); }
          catch (e) { showJob(true, "作品已生成，但加载失败：" + e.message, 100); }
        }
        setTimeout(() => { showJob(false); $("btn-generate").disabled = false; }, 4200);
        sessionStorage.removeItem(ACTIVE_JOBS_KEY);
      }
      state.pollFailures = 0;
      if (state.activeJobIds.length) state.pollTimer = setTimeout(poll, 1400);
    } catch (e) {
      state.pollFailures += 1;
      const delay = Math.min(10000, 1000 * (2 ** Math.min(state.pollFailures, 3)));
      showJob(true, "连接中断，正在重连（" + e.message + "）", 0);
      state.pollTimer = setTimeout(poll, delay);
    }
  };
  state.activeJobIds = jobIds;
  poll();
}

function showJob(show, text, pct) {
  $("job-status").classList.toggle("hidden", !show);
  if (text !== undefined) $("job-stage").textContent = text;
  if (pct !== undefined) $("job-bar").style.width = pct + "%";
}

/* ================= AI 写词 ================= */
async function aiLyrics() {
  const btn = $("btn-ai-lyrics");
  btn.disabled = true;
  const old = btn.textContent;
  btn.textContent = "✍️ 写词中…";
  try {
    const resp = await api("/api/lyrics", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        description: $("prompt").value.trim() || $("title").value.trim(),
        genre: state.selectedGenre,
        mood: state.selectedMood,
        lyrics_req: ($("lyrics-req") && $("lyrics-req").value.trim()) || null,
      }),
    });
    const data = await resp.json();
    if (data.lyrics) {
      $("lyrics").value = data.lyrics;
      btn.textContent = data.source === "llm" ? "✍️ 已写入(LLM)" : "✍️ 已写入(模板)";
    } else {
      btn.textContent = "❌ 失败";
    }
  } catch (e) {
    btn.textContent = "❌ 失败";
  }
  setTimeout(() => { btn.textContent = old; btn.disabled = false; }, 1800);
}

/* ================= AI 封面 ================= */
function hashCode(s) {
  let h = 0;
  for (let i = 0; i < String(s).length; i++) h = (h * 31 + String(s).charCodeAt(i)) | 0;
  return Math.abs(h);
}

function makeCover(id, title, size) {
  size = size || 120;
  const c = document.createElement("canvas");
  c.width = c.height = size * 2;
  const x = c.getContext("2d");
  x.scale(2, 2);
  const h = hashCode(id);
  const hue1 = h % 360;
  const hue2 = (hue1 + 60 + (h >> 4) % 50) % 360;
  const hue3 = (hue1 + 190) % 360;
  const g = x.createLinearGradient(0, 0, size, size);
  g.addColorStop(0, "hsl(" + hue1 + ",72%,52%)");
  g.addColorStop(1, "hsl(" + hue2 + ",62%,30%)");
  x.fillStyle = g;
  x.fillRect(0, 0, size, size);
  for (let i = 0; i < 4; i++) {
    x.beginPath();
    x.strokeStyle = "hsla(" + hue3 + ",85%,72%," + (0.5 - i * 0.11) + ")";
    x.lineWidth = 1.6;
    x.arc(size * (0.22 + ((h >> i) & 7) / 11), size * (0.24 + ((h >> (i + 2)) & 7) / 11),
          size * 0.1 + ((h >> (i + 3)) & 3) * size * 0.055, 0, 7);
    x.stroke();
  }
  for (let i = 0; i < 5; i++) {
    const bh = size * (0.08 + ((h >> (i * 3)) & 7) / 16);
    x.fillStyle = "rgba(255,255,255,0.5)";
    x.fillRect(size * 0.14 + i * size * 0.12, size * 0.8 - bh, size * 0.06, bh);
  }
  if (title) {
    x.fillStyle = "rgba(255,255,255,0.94)";
    x.shadowColor = "rgba(0,0,0,0.45)";
    x.shadowBlur = 4;
    x.font = "600 " + Math.max(11, Math.round(size * 0.1)) + "px system-ui, 'PingFang SC'";
    x.fillText(String(title).slice(0, size > 140 ? 10 : 6), size * 0.08, size * 0.93);
  }
  return c.toDataURL("image/png");
}

/* ================= 加载歌曲 ================= */
async function loadSong(songId, autoplay) {
  const resp = await api("/api/songs/" + songId);
  if (!resp.ok) throw new Error("作品加载失败（HTTP " + resp.status + "）");
  const data = await resp.json();
  if (!data.meta) return;
  state.currentSongId = songId;
  state.currentParams = data.meta.request || null;
  if (state.currentParams) state.lastParams = Object.assign({}, state.currentParams);
  state.score = data.score;
  state.bpm = data.score.bpm || 100;
  const audioOnly = !!data.meta.audio_only
    || !(data.score && data.score.melody && data.score.melody.length);

  $("player-panel").classList.remove("hidden");
  const main = document.querySelector("main");
  if (main) main.classList.add("has-player");
  $("song-title").textContent = data.meta.title;
  $("song-tags").innerHTML = [
    labelOf("genres", data.meta.genre),
    labelOf("moods", data.meta.mood),
    keyLabel(data.spec),
    data.meta.bpm + " BPM",
    fmtTime(data.meta.duration_sec),
    providerBadge(data.meta.provider),
    composerLabel(data.meta.composer_provider),
    data.spec.lyrics_source === "custom" ? "📝 自定义歌词" : "",
    audioOnly ? "🎤 真声演唱" : (data.spec.sing ? "🎤 AI 演唱" : ""),
  ].filter(Boolean).map((t, i) => `<span class="tag ${["", "cyan", "gold", "", "", "cyan", "", "gold"][i]}">${t}</span>`).join("");

  const coverURL = makeCover(songId, data.meta.title, 150);
  $("cover").src = coverURL;
  $("dl-cover").href = coverURL;

  state.currentFav = !!data.meta.favorite;
  updateFavBtn();

  const audio = $("audio");
  audio.src = mediaUrl("/api/songs/" + songId + "/audio.wav");
  $("dl-wav").href = mediaUrl("/api/songs/" + songId + "/audio.wav");
  $("dl-wav").download = data.meta.title + ".wav";
  // 神经演唱作品没有音符级曲谱/MIDI：隐藏不适用的入口
  $("dl-midi").classList.toggle("hidden", audioOnly);
  $("btn-regen").classList.toggle("hidden", audioOnly);
  $("btn-extend").classList.toggle("hidden", audioOnly);
  // 神经作品的"和声"分轨实为伴奏（demucs other 轨）
  $("stem-harmony").textContent = audioOnly ? "伴奏" : "和声";
  $("dl-midi").href = mediaUrl("/api/songs/" + songId + "/song.mid");
  $("dl-midi").download = data.meta.title + ".mid";
  $("dl-bundle").href = mediaUrl("/api/songs/" + songId + "/bundle.zip");
  $("dl-bundle").download = data.meta.title + "-分享包.zip";
  ["vocal", "harmony", "bass", "drums"].forEach((n) => {
    const a = $("stem-" + n);
    a.href = mediaUrl("/api/songs/" + songId + "/stems/" + n + ".wav");
    a.download = data.meta.title + "-" + n + ".wav";
  });
  api("/api/songs/" + songId + "/stems/vocal.wav", { method: "HEAD" }).then((r) => {
    document.querySelectorAll(".stem-link").forEach((a) => a.classList.toggle("disabled", !r.ok));
  }).catch(() => {});

  drawRoll();
  rebuildLyrics();   // 切歌必须重建歌词索引（否则显示上一首的歌词）
  updateLyric(0, true);
  if (autoplay) { try { await audio.play(); } catch (e) { /* 自动播放被拦时忽略 */ }
  }
}

function labelOf(kind, id) {
  const list = (state.meta && state.meta[kind]) || [];
  const found = list.find((x) => x.id === id);
  return found ? found.label : id;
}

const MODE_ZH = { major: "大调", minor: "小调", dorian: "多利亚", chinese: "五声·宫", minor_pentatonic: "小调五声" };
function keyLabel(spec) {
  return spec.key + " " + (MODE_ZH[spec.mode] || spec.mode);
}

/* ================= 钢琴卷帘 ================= */
function drawRoll() {
  const score = state.score;
  if (!score) return;
  if (score.audio_only || !(score.melody && score.melody.length)) {
    return drawRollLite();
  }
  $("roll-hint").textContent = "🎹 钢琴卷帘：紫=主旋律/人声 · 青=和声垫/琶音 · 金=贝斯 · 底部=鼓组。点击卷帘任意位置可跳转播放。";
  const canvas = $("roll");
  const dpr = window.devicePixelRatio || 1;
  const pxb = Math.max(9, Math.min(26, ($("roll-wrap").clientWidth - 2) / score.total_beats));
  state.pxb = pxb;
  const W = Math.max($("roll-wrap").clientWidth - 2, score.total_beats * pxb);

  const pitches = [];
  score.melody.forEach((n) => pitches.push(n.pitch));
  score.bass.forEach((n) => pitches.push(n.pitch));
  score.arps.forEach((n) => pitches.push(n.pitch));
  score.pads.forEach((n) => n.pitches.forEach((p) => pitches.push(p)));
  const pMin = Math.max(28, Math.min(...pitches) - 2);
  const pMax = Math.min(100, Math.max(...pitches) + 2);
  const rowH = 7;
  const rollH = (pMax - pMin + 1) * rowH;
  const drumH = 58;
  const labelH = 20;
  const H = labelH + rollH + drumH + 6;

  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width = W + "px";
  canvas.style.height = H + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const secColor = { intro: "#4a5578", verse: "#3d4a75", prechorus: "#54428a", chorus: "#6b3fa0", bridge: "#2f6b66", outro: "#3c4460" };
  const bpb = score.beats_per_bar;

  score.sections.forEach((sec) => {
    const x = sec.start_beat * pxb;
    const w = sec.beats * pxb;
    ctx.fillStyle = secColor[sec.name] || "#3d4a75";
    ctx.globalAlpha = 0.35;
    ctx.fillRect(x, 0, w, H);
    ctx.globalAlpha = 1;
    ctx.fillStyle = "rgba(255,255,255,.75)";
    ctx.font = "11px system-ui";
    ctx.fillText(sec.label + " " + sec.bars + "小节", x + 5, 13);
    ctx.strokeStyle = "rgba(255,255,255,.22)";
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  });

  const totalBars = score.total_beats / bpb;
  const chordStep = pxb < 13 ? 2 : 1;
  for (let bar = 0; bar < totalBars; bar++) {
    const x = bar * bpb * pxb;
    ctx.strokeStyle = "rgba(255,255,255,.10)";
    ctx.beginPath(); ctx.moveTo(x, labelH); ctx.lineTo(x, H); ctx.stroke();
    if (bar % chordStep === 0 && score.chords[bar]) {
      ctx.fillStyle = "rgba(160,235,255,.8)";
      ctx.font = "10px system-ui";
      ctx.fillText(score.chords[bar].symbol, x + 3, labelH + 10);
    }
  }

  const yOf = (p) => labelH + (pMax - p) * rowH;

  score.pads.forEach((n) => {
    n.pitches.forEach((p) => {
      ctx.fillStyle = "rgba(0,212,255,.16)";
      ctx.fillRect(n.start * pxb, yOf(p) - rowH / 2 + 1, n.dur * pxb - 1, rowH - 2);
    });
  });
  score.arps.forEach((n) => {
    ctx.fillStyle = "rgba(0,212,255,.4)";
    ctx.fillRect(n.start * pxb, yOf(n.pitch) - 2, Math.max(3, n.dur * pxb), 3.5);
  });
  score.stabs.forEach((n) => {
    n.pitches.forEach((p) => {
      ctx.fillStyle = "rgba(0,212,255,.34)";
      ctx.fillRect(n.start * pxb, yOf(p) - 2.5, Math.max(3, n.dur * pxb), 4.5);
    });
  });
  score.bass.forEach((n) => {
    ctx.fillStyle = "rgba(255,200,87,.55)";
    roundRect(ctx, n.start * pxb, yOf(n.pitch) - rowH / 2 + 1.5, Math.max(3, n.dur * pxb - 1), rowH - 3, 2);
    ctx.fill();
  });
  score.melody.forEach((n) => {
    const grad = ctx.createLinearGradient(0, yOf(n.pitch) - 4, 0, yOf(n.pitch) + 4);
    grad.addColorStop(0, "#b39bff");
    grad.addColorStop(1, "#7c5cff");
    ctx.fillStyle = grad;
    roundRect(ctx, n.start * pxb, yOf(n.pitch) - rowH / 2, Math.max(3.5, n.dur * pxb - 1.5), rowH, 2.5);
    ctx.fill();
  });

  const drumTop = labelH + rollH + 4;
  const drumRows = { kick: 0, snare: 1, clap: 1, rim: 1, hat: 2, ohat: 2 };
  const drumColors = { kick: "#ff8f66", snare: "#ffd166", clap: "#ffd166", rim: "#e0e6ff", hat: "#8fd3ff", ohat: "#8fd3ff" };
  ctx.strokeStyle = "rgba(255,255,255,.15)";
  ctx.beginPath(); ctx.moveTo(0, drumTop - 2); ctx.lineTo(W, drumTop - 2); ctx.stroke();
  score.drums.forEach((n) => {
    const row = drumRows[n.type] !== undefined ? drumRows[n.type] : 3;
    const x = n.start * pxb;
    const y = drumTop + row * 16;
    ctx.fillStyle = drumColors[n.type] || "#aaa";
    ctx.globalAlpha = 0.35 + 0.65 * (n.vel / 127);
    if (n.type === "kick") { ctx.beginPath(); ctx.arc(x + 2, y + 5, 3.4, 0, 7); ctx.fill(); }
    else { ctx.fillRect(x, y + 1, 3.5, 8); }
    ctx.globalAlpha = 1;
  });
  ctx.fillStyle = "rgba(255,255,255,.35)";
  ctx.font = "10px system-ui";
  ctx.fillText("🥁 鼓组", 5, drumTop + 44);
}

function drawRollLite() {
  /* 神经演唱作品的轻量时间轴：段落色带 + 每段首句歌词（无音符级曲谱） */
  const score = state.score;
  const canvas = $("roll");
  const dpr = window.devicePixelRatio || 1;
  const W = Math.max($("roll-wrap").clientWidth - 2, 600);
  const H = 104;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width = W + "px";
  canvas.style.height = H + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const bpb = score.beats_per_bar || 4;
  const pxb = Math.max(9, Math.min(26, W / Math.max(1, score.total_beats)));
  state.pxb = pxb;
  const secColor = { intro: "#4a5578", verse: "#3d4a75", prechorus: "#54428a", chorus: "#6b3fa0", bridge: "#2f6b66", outro: "#3c4460" };

  score.sections.forEach((sec) => {
    const x = sec.start_beat * pxb;
    const w = sec.beats * pxb;
    ctx.fillStyle = secColor[sec.name] || "#3d4a75";
    ctx.globalAlpha = 0.45;
    ctx.fillRect(x, 0, w, H);
    ctx.globalAlpha = 1;
    ctx.fillStyle = "rgba(255,255,255,.85)";
    ctx.font = "11px system-ui";
    ctx.fillText(sec.label + " " + sec.bars + "小节", x + 5, 14);
    const firstLyric = (sec.lyrics || []).find((t) => t && t.trim());
    if (firstLyric) {
      ctx.fillStyle = "rgba(255,255,255,.66)";
      ctx.font = "10px system-ui";
      ctx.fillText(String(firstLyric).slice(0, Math.max(4, Math.floor(w / 11))), x + 5, 30);
    }
    ctx.strokeStyle = "rgba(255,255,255,.22)";
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  });

  ctx.fillStyle = "rgba(255,255,255,.4)";
  ctx.font = "10px system-ui";
  ctx.fillText("🎧 神经歌声演唱 · 段落与歌词时间轴（无音符级曲谱）", 5, H - 8);
  $("roll-hint").textContent = "🎧 神经歌声作品：展示段落结构与歌词时间轴，点击任意位置跳转播放；MIDI/分轨/换副歌不适用于此类作品。";
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/* ================= 播放器 ================= */
function bindPlayer() {
  const audio = $("audio");
  const playBtn = $("btn-play");
  playBtn.onclick = () => { audio.paused ? audio.play() : audio.pause(); };
  audio.onplay = () => {
    playBtn.textContent = "⏸";
    ensureGraph();
    if (actx && actx.state === "suspended") actx.resume();
    raf();
  };
  audio.onpause = () => { playBtn.textContent = "▶"; drawSpectrum(); };
  audio.ontimeupdate = () => {
    const dur = audio.duration || (state.score ? state.score.duration_sec : 1);
    $("seek-fill").style.width = (audio.currentTime / dur * 100) + "%";
    $("seek").setAttribute("aria-valuenow", String(Math.round(audio.currentTime / dur * 100)));
    $("time-label").textContent = fmtTime(audio.currentTime) + " / " + fmtTime(dur);
    const beat = audio.currentTime * state.bpm / 60;
    $("playhead").style.transform = "translateX(" + (beat * state.pxb) + "px)";
    updateLyric(beat);
  };
  audio.onended = () => { playBtn.textContent = "▶"; drawSpectrum(); };
  audio.onerror = () => {
    if (audio.src) $("lyric-now").textContent = "⚠️ 音频加载失败（文件缺失或服务不可达）";
  };

  $("seek").onclick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = (e.clientX - rect.left) / rect.width;
    const dur = audio.duration || (state.score ? state.score.duration_sec : 0);
    if (dur) audio.currentTime = ratio * dur;
  };
  $("seek").onkeydown = (e) => {
    const dur = audio.duration || (state.score ? state.score.duration_sec : 0);
    if (!dur) return;
    const steps = { ArrowLeft: -5, ArrowDown: -5, ArrowRight: 5, ArrowUp: 5,
      PageDown: -15, PageUp: 15, Home: -dur, End: dur };
    if (!(e.key in steps)) return;
    e.preventDefault();
    audio.currentTime = Math.max(0, Math.min(dur, audio.currentTime + steps[e.key]));
  };
  $("roll-wrap").onclick = (e) => {
    if (e.target.id !== "roll") return;
    const rect = $("roll").getBoundingClientRect();
    const beat = (e.clientX - rect.left) / state.pxb;
    audio.currentTime = Math.max(0, beat * 60 / state.bpm - 0.01);
  };
}

let lyricBars = [];
let lyricEvents = [];  // Whisper 对齐后的歌词真实时间轴（神经作品）
function rebuildLyrics() {
  lyricBars = [];
  lyricEvents = [];
  if (!state.score) return;
  if (state.score.lyric_events && state.score.lyric_events.length) {
    lyricEvents = state.score.lyric_events
      .slice().sort((a, b) => a[0] - b[0])
      .map((e) => ({ start: e[0], end: e[1], text: e[2] }));
    return;
  }
  state.score.sections.forEach((sec) => {
    (sec.lyrics || []).forEach((text, i) => {
      if (text) lyricBars.push({ bar: sec.start_bar + i, text });
    });
  });
}
function updateLyric(beat, force) {
  if (!lyricBars.length && !lyricEvents.length && !force) rebuildLyrics();
  let text = "";
  if (lyricEvents.length) {
    // Whisper 精确时间轴：按播放秒数对齐（时间轴有 0.15s 提前量，体验更像跟唱）
    const now = ($("audio") && $("audio").currentTime) || 0;
    for (const it of lyricEvents) { if (it.start <= now + 0.15) text = it.text; else break; }
  } else {
    const bar = Math.floor(beat / (state.score ? state.score.beats_per_bar : 4));
    for (const it of lyricBars) { if (it.bar <= bar) text = it.text; else break; }
  }
  if (force || $("lyric-now").textContent !== (text || "♪")) {
    $("lyric-now").textContent = text || "♪";
  }
}

/* ================= 频谱可视化 ================= */
let actx = null, analyser = null, freqData = null;
function ensureGraph() {
  if (actx) return;
  try {
    actx = new (window.AudioContext || window.webkitAudioContext)();
    const src = actx.createMediaElementSource($("audio"));
    analyser = actx.createAnalyser();
    analyser.fftSize = 256;
    analyser.smoothingTimeConstant = 0.78;
    src.connect(analyser);
    analyser.connect(actx.destination);
    freqData = new Uint8Array(analyser.frequencyBinCount);
  } catch (e) { console.warn("频谱初始化失败", e); }
}

function drawSpectrum() {
  const c = $("spectrum");
  if (!c) return;
  const dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth || 600, H = c.clientHeight || 46;
  if (c.width !== Math.round(W * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const playing = !$("audio").paused && analyser;
  if (playing) analyser.getByteFrequencyData(freqData);
  const n = 56;
  for (let i = 0; i < n; i++) {
    let v;
    if (playing) v = freqData[Math.floor(i * freqData.length / n * 0.72)] / 255;
    else v = 0.05 + 0.03 * Math.sin(Date.now() / 500 + i * 0.6);
    const h = Math.max(2, v * H);
    const grad = ctx.createLinearGradient(0, H - h, 0, H);
    grad.addColorStop(0, "#61e8ff");
    grad.addColorStop(1, "#7c5cff");
    ctx.fillStyle = grad;
    ctx.fillRect(i * W / n + W / n * 0.18, H - h, W / n * 0.6, h);
  }
}

function raf() {
  const audio = $("audio");
  if (audio.paused) { drawSpectrum(); return; }
  const wrap = $("roll-wrap");
  const x = audio.currentTime * state.bpm / 60 * state.pxb;
  if (x < wrap.scrollLeft + 40 || x > wrap.scrollLeft + wrap.clientWidth - 80) {
    wrap.scrollLeft = x - wrap.clientWidth * 0.4;
  }
  drawSpectrum();
  requestAnimationFrame(raf);
}

function fmtTime(s) {
  s = Math.max(0, Math.round(s || 0));
  return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
}

/* ================= 收藏 / 改名 / 换副歌 / MV 导出 ================= */
function updateFavBtn() {
  const btn = $("btn-fav");
  btn.textContent = state.currentFav ? "★ 已收藏" : "☆ 收藏";
  btn.classList.toggle("active", !!state.currentFav);
}

async function toggleFav() {
  const id = state.currentSongId;
  if (!id) return;
  const fav = !state.currentFav;
  await api("/api/songs/" + id + "/favorite", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ favorite: fav }),
  });
  state.currentFav = fav;
  updateFavBtn();
  refreshLibrary();
}

async function renameSong() {
  const id = state.currentSongId;
  if (!id) return;
  const t = prompt("新歌名：", $("song-title").textContent);
  if (!t || !t.trim()) return;
  const resp = await api("/api/songs/" + id, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: t.trim() }),
  });
  if (resp.ok) {
    $("song-title").textContent = t.trim();
    refreshLibrary();
  } else {
    const e = await resp.json().catch(() => ({}));
    alert(e.error || "改名失败");
  }
}

async function regenChorus() {
  const id = state.currentSongId;
  if (!id) return;
  const btn = $("btn-regen");
  btn.disabled = true;
  btn.textContent = "🎼 重写副歌中…";
  try {
    const resp = await api("/api/songs/" + id + "/regen_chorus", { method: "POST" });
    const d = await resp.json();
    if (d.song_id) {
      await loadSong(d.song_id, true);
      await refreshLibrary();
    } else {
      alert(d.error || "换副歌失败");
    }
  } finally {
    btn.disabled = false;
    btn.textContent = "🎼 换副歌";
  }
}

async function mvExport() {
  const id = state.currentSongId;
  if (!id) return;
  const btn = $("btn-mv");
  btn.disabled = true;
  btn.textContent = "🎬 合成中…";
  try {
    const r = await api("/api/songs/" + id + "/video.mp4");
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.error || "导出失败");
    }
    const blob = await r.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = ($("song-title").textContent || "mv") + ".mp4";
    document.body.appendChild(a);
    a.click();
    a.remove();
  } catch (e) {
    alert(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "🎬 MV";
  }
}

/* ================= 曲库 ================= */
async function refreshLibrary() {
  const resp = await api("/api/songs");
  if (!resp.ok) throw new Error("作品库加载失败（HTTP " + resp.status + "）");
  const data = await resp.json();
  state.songs = data.songs;
  renderLibrary();
}

function renderLibrary() {
  const list = $("song-list");
  const q = (($("lib-search") && $("lib-search").value) || "").trim().toLowerCase();
  const songs = (state.songs || []).filter((s) => !q
    || [s.title, s.genre, s.mood, labelOf("genres", s.genre), labelOf("moods", s.mood)]
      .join(" ").toLowerCase().includes(q));
  if ($("lib-count")) {
    const total = (state.songs || []).length;
    $("lib-count").textContent = total ? (q ? songs.length + " / " + total : "共 " + total + " 首") : "";
  }
  list.innerHTML = "";
  if (!songs.length) {
    list.innerHTML = '<div class="empty">' + (q ? "没有匹配的作品" : "还没有作品，快去生成第一首吧 ✨") + "</div>";
    return;
  }
  songs.forEach((song) => {
    const item = document.createElement("div");
    item.className = "song-item" + (song.favorite ? " fav" : "");
    const when = new Date(song.created_at * 1000);
    item.innerHTML = `
      <img class="lib-cover" alt="" src="${makeCover(song.id, song.title, 44)}">
      <div class="song-item-body">
        <div class="t">${song.favorite ? "★ " : ""}${escapeHtml(song.title)}</div>
        <div class="meta">
          <span>${labelOf("genres", song.genre)}</span>
          <span>${labelOf("moods", song.mood)}</span>
          <span>${song.bpm} BPM</span>
          <span>${fmtTime(song.duration_sec)}</span>
          <span>${providerIcon(song.provider)}</span>
          <span>${composerLabel(song.composer_provider)}</span>
          <span>${song.sing ? "🎤" : ""}</span>
          <span>${when.getMonth() + 1}/${when.getDate()} ${String(when.getHours()).padStart(2, "0")}:${String(when.getMinutes()).padStart(2, "0")}</span>
        </div>
      </div>
      <button class="del" type="button" title="删除作品" aria-label="删除《${escapeHtml(song.title)}》">✕</button>`;
    item.onclick = () => loadSong(song.id, true);
    item.querySelector(".del").onclick = async (e) => {
      e.stopPropagation();
      if (!confirm("删除《" + song.title + "》？")) return;
      await api("/api/songs/" + song.id, { method: "DELETE" });
      if (state.currentSongId === song.id) {
        $("audio").pause();
        $("player-panel").classList.add("hidden");
        const main = document.querySelector("main");
        if (main) main.classList.remove("has-player");
        state.currentSongId = null;
      }
      refreshLibrary();
    };
    list.appendChild(item);
  });
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ================= v5：音色 Persona / 重唱片段 / 专辑 ================= */
async function loadPersonas() {
  if (!$("persona-sel")) return;
  try {
    const data = await (await api("/api/personas")).json();
    state.personas = data.personas || [];
  } catch (e) { state.personas = []; }
  const sel = $("persona-sel");
  const cur = sel.value;
  sel.innerHTML = "";
  const none = document.createElement("option");
  none.value = "none";
  none.textContent = "不指定（每次新音色）";
  sel.appendChild(none);
  state.personas.forEach((p) => {
    const o = document.createElement("option");
    o.value = p.id;
    o.textContent = "🎨 " + p.name;
    sel.appendChild(o);
  });
  sel.value = cur && state.personas.some((p) => p.id === cur) ? cur : "none";
}

async function extractPersona() {
  const id = state.currentSongId;
  if (!id) return;
  const name = prompt("给这个音色起个名字（例如：夜班嗓）：", $("song-title").textContent || "");
  if (name === null) return;
  const btn = $("btn-persona-extract");
  btn.disabled = true;
  btn.textContent = "🎨 提取中约 1 分钟…";
  try {
    const resp = await api("/api/personas", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: "song", song_id: id, name: (name || "").trim() || "我的音色" }),
    });
    const d = await resp.json();
    if (d.persona) {
      await loadPersonas();
      $("persona-sel").value = d.persona.id;
      $("persona-row").scrollIntoView({ behavior: "smooth", block: "center" });
      showJob(true, "✅ 音色「" + d.persona.name + "」已提取，可在引擎配置里选用", 100);
      setTimeout(() => showJob(false), 3200);
    } else {
      alert(d.error || "提取失败");
    }
  } catch (e) {
    alert("提取失败：" + e.message);
  }
  btn.disabled = false;
  btn.textContent = "🎨 提取音色";
}

function openReseg() {
  const score = state.score;
  if (!score || !score.sections || !score.sections.length) {
    alert("当前作品没有段落信息");
    return;
  }
  const sel = $("reseg-section");
  sel.innerHTML = "";
  score.sections.forEach((sec, i) => {
    const first = (sec.lyrics || []).find((t) => t && t.trim()) || "（无歌词）";
    const o = document.createElement("option");
    o.value = i;
    o.textContent = (sec.label || sec.name) + " · " + sec.bars + " 小节 · " + String(first).slice(0, 10);
    o.selected = sec.name === "chorus";
    sel.appendChild(o);
  });
  $("reseg-lyrics").value = "";
  $("reseg-modal").classList.remove("hidden");
}

async function submitReseg() {
  const id = state.currentSongId;
  if (!id) return;
  const secIdx = parseInt($("reseg-section").value, 10);
  const lyrics = $("reseg-lyrics").value.trim();
  if (!lyrics) { alert("新歌词不能为空"); return; }
  const btn = $("reseg-submit");
  btn.disabled = true;
  btn.textContent = "提交中…";
  try {
    const resp = await api("/api/songs/" + id + "/reseg", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ section_index: secIdx, new_lyrics: lyrics,
                             full_remake: !!$("reseg-full").checked }),
    });
    const d = await resp.json();
    if (d.job_id) {
      $("reseg-modal").classList.add("hidden");
      $("btn-generate").disabled = true;
      showJob(true, "提交重唱任务…", 0);
      state.jobStartedAt = Date.now();
      state.activeJobIds = [d.job_id];
      $("btn-job-cancel").classList.remove("hidden");
      pollJobs([d.job_id]);
    } else {
      alert(d.error || "提交失败");
    }
  } catch (e) {
    alert("提交失败：" + e.message);
  }
  btn.disabled = false;
  btn.textContent = "开始重唱";
}

async function createAlbum() {
  const pid = $("persona-sel") ? $("persona-sel").value : "";
  if (!pid || pid === "none") { alert("请先选择一个音色 Persona"); return; }
  const pname = (state.personas.find((p) => p.id === pid) || {}).name || "";
  const theme = prompt("专辑主题（一句描述，会写给每首歌的歌词）：", pname);
  if (theme === null) return;
  const n = parseInt(prompt("生成几首？（1-5）", "2"), 10);
  if (!n || n < 1) return;
  const btn = $("btn-album");
  btn.disabled = true;
  btn.textContent = "📚 排队中…";
  try {
    const resp = await api("/api/album", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ persona_id: pid, count: n, theme: theme.trim() }),
    });
    const d = await resp.json();
    if (d.job_ids && d.job_ids.length) {
      $("btn-generate").disabled = true;
      showJob(true, "📚 专辑 " + d.count + " 首已排队（同一副嗓子）…", 0);
      state.jobStartedAt = Date.now();
      state.activeJobIds = d.job_ids;
      $("btn-job-cancel").classList.remove("hidden");
      pollJobs(d.job_ids);
    } else {
      alert(d.error || "排队失败");
    }
  } catch (e) {
    alert("排队失败：" + e.message);
  }
  btn.disabled = false;
  btn.textContent = "📚 专辑";
}

/* ================= 设置 ================= */
async function openSettings() {
  let resp;
  try { resp = await api("/api/config"); }
  catch (e) {
    $("cfg-status").textContent = "读取配置失败：" + e.message;
    $("settings-modal").classList.remove("hidden");
    return;
  }
  if (!resp.ok) {
    $("cfg-status").textContent = "读取配置失败（HTTP " + resp.status + "）";
    $("settings-modal").classList.remove("hidden");
    return;
  }
  const cfg = await resp.json();
  $("cfg-enabled").checked = !!cfg.llm.enabled;
  $("cfg-base-url").value = cfg.llm.base_url || "";
  $("cfg-model").value = cfg.llm.model || "";
  $("cfg-key").value = "";
  $("cfg-key").placeholder = cfg.llm.api_key_set ? ("已保存: " + cfg.llm.api_key_masked) : "粘贴 API Key";
  $("cfg-status").textContent = "配置来源: " + (cfg.source || "default");
  $("cfg-test-result").classList.add("hidden");
  $("cfg-test-result").className = "cfg-test-result hidden";
  const neural = (state.engines && state.engines.diffrhythm) || {};
  $("cfg-neural-status").textContent = neural.available
    ? "✅ " + (neural.reason || "就绪") + " · 采样步数 " + neural.steps
    : "❌ " + (neural.reason || "未安装（bash scripts/setup_neural.sh）");
  const mk = cfg.mureka || {};
  $("cfg-mureka-enabled").checked = !!mk.enabled;
  $("cfg-mureka-model").value = mk.model || "mureka-6";
  $("cfg-mureka-key").value = "";
  $("cfg-mureka-key").placeholder = mk.api_key_set
    ? ("已保存: " + mk.api_key_masked) : "粘贴 API Key";
  $("cfg-steps").value = (cfg.neural && cfg.neural.steps) || 16;
  $("cfg-steps-val").textContent = $("cfg-steps").value;
  $("cfg-mureka-status").textContent = "";
  $("settings-modal").classList.remove("hidden");
}

async function testConfig() {
  const btn = $("cfg-test");
  const result = $("cfg-test-result");
  btn.disabled = true;
  btn.textContent = "…";
  result.classList.remove("hidden", "ok", "fail");

  const body = {
    llm: {
      enabled: $("cfg-enabled").checked,
      base_url: $("cfg-base-url").value.trim(),
      model: $("cfg-model").value.trim(),
    },
  };
  const key = $("cfg-key").value.trim();
  if (key) body.llm.api_key = key;

  try {
    const resp = await api("/api/config/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (data.ok) {
      result.className = "cfg-test-result ok";
      result.textContent = "✅ 连接成功！模型: " + data.model + " · 耗时 " + data.elapsed_ms + "ms · 回复: " + (data.reply_preview || "(空)");
    } else {
      result.className = "cfg-test-result fail";
      result.textContent = "❌ " + (data.reason || "未知错误") + (data.base_url ? " · " + data.base_url : "");
    }
  } catch (e) {
    result.className = "cfg-test-result fail";
    result.textContent = "❌ 网络错误: " + e.message;
  }
  btn.disabled = false;
  btn.textContent = "🔌 测试连接";
}

async function saveSettings() {
  const body = {
    llm: {
      enabled: $("cfg-enabled").checked,
      base_url: $("cfg-base-url").value.trim() || undefined,
      model: $("cfg-model").value.trim() || undefined,
    },
  };
  const key = $("cfg-key").value.trim();
  if (key) body.llm.api_key = key;
  body.neural = { steps: parseInt($("cfg-steps").value, 10) || 16 };
  body.mureka = {
    enabled: $("cfg-mureka-enabled").checked,
    model: $("cfg-mureka-model").value.trim() || "mureka-6",
  };
  const mkey = $("cfg-mureka-key").value.trim();
  if (mkey) body.mureka.api_key = mkey;
  try {
    const resp = await api("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const cfg = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      $("cfg-status").textContent = "保存失败：" + (cfg.error || resp.statusText);
      return;
    }
    const mkOk = cfg.mureka && cfg.mureka.enabled;
    $("cfg-mureka-status").textContent = mkOk ? "✅ Mureka 云引擎已启用" : "Mureka 未启用";
    $("cfg-status").textContent = "✅ 已保存（来源: " + cfg.source + "）";
    await refreshConfigBadge();
    await loadMeta();
    setTimeout(() => $("settings-modal").classList.add("hidden"), 900);
  } catch (e) {
    $("cfg-status").textContent = "保存失败：" + e.message;
  }
}

async function refreshConfigBadge() {
  try {
    const cfg = await (await api("/api/config")).json();
    const badge = $("llm-badge");
    const composer = $("composer-provider");
    const option = composer && composer.querySelector('option[value="llm"]');
    if (!badge || !option) return;
    option.disabled = !cfg.llm.enabled;
    option.title = cfg.llm.enabled ? "使用已配置的远程 LLM" : "请先在引擎设置中配置并启用 LLM";
    if (cfg.llm.enabled) {
      badge.textContent = "LLM: " + cfg.llm.model + " ✓";
      badge.classList.add("on");
    } else {
      badge.textContent = "LLM 未配置（用离线引擎）";
      badge.classList.remove("on");
    }
  } catch (e) { console.error(e); }
}

/* ================= 神经引擎：参考音频 / 选项联动 ================= */
function onProviderChange() {
  const neural = $("provider").value === "diffrhythm";
  $("ref-audio-row").classList.toggle("hidden", !neural);
  $("ref-audio-hint").classList.toggle("hidden", !neural);
  // 完整版开关：神经引擎 + full 权重就绪才显示
  const fullOk = state.engines && state.engines.diffrhythm
    && state.engines.diffrhythm.full_available === true;
  $("full-song-row").classList.toggle("hidden", !neural || !fullOk);
  // 神经引擎自动决定音色/调性/演唱方式
  $("voice").disabled = neural;
  $("key-sel").disabled = neural;
  $("sing").disabled = neural;
}

async function uploadRefAudio() {
  const inp = $("ref-audio-file");
  const f = inp.files && inp.files[0];
  if (!f) {
    alert("请先选择音频文件（wav / mp3 / flac，建议 ≥10 秒）");
    return;
  }
  const btn = $("btn-ref-upload");
  btn.disabled = true;
  btn.textContent = "⬆ 上传中…";
  try {
    const resp = await api("/api/ref-audio", {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: f,
    });
    const d = await resp.json();
    if (d.ref_audio) {
      state.refAudio = d.ref_audio;
      $("ref-audio-name").textContent = "✅ " + f.name;
    } else {
      alert(d.error || "上传失败");
    }
  } catch (e) {
    alert("上传失败：" + e.message);
  }
  btn.disabled = false;
  btn.textContent = "⬆ 上传";
}

/* ================= 事件绑定 ================= */
function setMode(mode) {
  state.mode = mode;
  const adv = mode === "advanced";
  $("tab-simple").classList.toggle("active", !adv);
  $("tab-advanced").classList.toggle("active", adv);
  $("tab-simple").setAttribute("aria-selected", String(!adv));
  $("tab-advanced").setAttribute("aria-selected", String(adv));
  $("panel-simple").classList.toggle("hidden", adv);
  $("panel-adv-req").classList.toggle("hidden", !adv);
  $("panel-adv-lyrics").classList.toggle("hidden", !adv);
  $("advanced-extra").classList.toggle("hidden", !adv);
}

function bindEvents() {
  bindPlayer();
  $("btn-generate").onclick = () => generate();
  $("btn-inspire").onclick = () => {
    const list = (state.meta && state.meta.inspirations) || [];
    $("prompt").value = list[Math.floor(Math.random() * list.length)] || "";
  };
  $("tab-simple").onclick = () => setMode("simple");
  $("tab-advanced").onclick = () => setMode("advanced");
  $("btn-ai-lyrics").onclick = aiLyrics;
  $("duration").oninput = (e) => { $("dur-val").textContent = e.target.value + "s"; };
  $("bpm").oninput = (e) => { $("bpm-val").textContent = +e.target.value === 0 ? "自动" : e.target.value; };

  $("btn-remix").onclick = () => generate({ seed: Date.now() % 1e9 }, state.currentParams);
  $("btn-extend").onclick = () => {
    const base = state.currentParams || state.lastParams || collectParams();
    generate({
      seed: base.seed !== undefined && base.seed !== null ? base.seed : undefined,
      duration_sec: Math.min(180, (base.duration_sec || 30) + 30),
    }, base);
  };
  $("btn-fav").onclick = toggleFav;
  $("btn-rename").onclick = renameSong;
  $("btn-regen").onclick = regenChorus;
  $("btn-mv").onclick = mvExport;
  $("lib-search").oninput = renderLibrary;
  window.addEventListener("resize", () => {
    if (state.score) drawRoll();
    drawSpectrum();
  });

  $("cfg-steps").oninput = (e) => { $("cfg-steps-val").textContent = e.target.value; };
  $("btn-settings").onclick = openSettings;
  $("cfg-cancel").onclick = () => $("settings-modal").classList.add("hidden");
  $("cfg-save").onclick = saveSettings;
  $("cfg-test").onclick = testConfig;
  $("btn-job-cancel").onclick = cancelJob;
  $("btn-persona-extract").onclick = extractPersona;
  $("btn-persona-refresh").onclick = loadPersonas;
  $("btn-reseg").onclick = openReseg;
  $("reseg-cancel").onclick = () => $("reseg-modal").classList.add("hidden");
  $("reseg-submit").onclick = submitReseg;
  $("btn-album").onclick = createAlbum;
  $("reseg-modal").onclick = (e) => {
    if (e.target.id === "reseg-modal") $("reseg-modal").classList.add("hidden");
  };
  $("provider").onchange = onProviderChange;
  $("btn-ref-upload").onclick = uploadRefAudio;
  onProviderChange();
  $("candidate-a").onclick = () => loadSong(state.candidateSongIds[0], false);
  $("candidate-b").onclick = () => loadSong(state.candidateSongIds[1], false);
  $("settings-modal").onclick = (e) => {
    if (e.target.id === "settings-modal") $("settings-modal").classList.add("hidden");
  };
  const savedJobs = sessionStorage.getItem(ACTIVE_JOBS_KEY);
  if (savedJobs) {
    try {
      const ids = JSON.parse(savedJobs);
      if (Array.isArray(ids) && ids.length) {
        state.activeJobIds = ids;
        $("btn-generate").disabled = true;
        $("btn-job-cancel").classList.remove("hidden");
        pollJobs(ids);
      }
    } catch (e) {
      sessionStorage.removeItem(ACTIVE_JOBS_KEY);
    }
  }
}
