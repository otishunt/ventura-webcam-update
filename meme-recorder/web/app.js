// Meme Pose Recorder: participant-facing client.
// Flow: welcome → consent → setup → prompt ⇄ review → done
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  pid: null, code: null, sessionId: null, protocolVersion: null,
  consent: null, prompts: {}, queue: [], pos: 0, total: 0,
  stream: null, mime: null, recording: false, current: null,
  lastBlob: null, lastClipId: null, contact: "",
};

// ------------------------------------------------------------------ utils
function show(name) {
  document.querySelectorAll(".screen").forEach((s) =>
    s.classList.toggle("active", s.dataset.screen === name));
  window.scrollTo(0, 0);
}
document.querySelectorAll("[data-go]").forEach((b) =>
  b.addEventListener("click", () => {
    if (b.dataset.go === "consent") loadConsent();
    show(b.dataset.go);
  }));

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
  return data;
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function shuffle(a) {
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}
function esc(s) { const d = document.createElement("div"); d.textContent = s ?? ""; return d.innerHTML; }

// ------------------------------------------------------------------ consent
async function loadConsent() {
  if (state.consent) return;
  try {
    const c = await api("/api/consent");
    state.consent = c; state.contact = c.contact;
    $("consent-title").textContent = c.title;
    $("consent-study").textContent = `${c.study_name} · ${c.team}`;
    $("consent-body").innerHTML = c.paragraphs.map((p) => `<p>${esc(p)}</p>`).join("");
    $("consent-agreements").innerHTML = c.agreements.map((a) =>
      `<label class="check"><input type="checkbox" data-agree="${esc(a.id)}"> <span>${esc(a.text)}</span></label>`).join("");
    $("consent-version").textContent = c.version;
    $("consent-date").textContent = new Date().toLocaleDateString();
    document.querySelectorAll("[data-agree]").forEach((cb) => cb.addEventListener("change", checkConsent));
  } catch (e) { $("consent-error").textContent = e.message; }
}
function checkConsent() {
  const all = [...document.querySelectorAll("[data-agree]")].every((cb) => cb.checked);
  $("consent-submit").disabled = !(all && $("consent-name").value.trim().length >= 2);
}
$("consent-name").addEventListener("input", checkConsent);
$("consent-submit").addEventListener("click", async () => {
  $("consent-error").textContent = "";
  const agreements = {};
  document.querySelectorAll("[data-agree]").forEach((cb) => (agreements[cb.dataset.agree] = cb.checked));
  try {
    const r = await api("/api/participants", {
      consent_version: state.consent.version, consent_sha256: state.consent.text_sha256,
      agreements, signature_name: $("consent-name").value,
    });
    // The ID and code stay behind the scenes: they label the videos and
    // authenticate uploads, but participants never need to see them.
    state.pid = r.participant_id; state.code = r.resume_code;
    enterSetup();
  } catch (e) { $("consent-error").textContent = e.message; }
});


// ------------------------------------------------------------------ camera
function pickMime() {
  const opts = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm",
                "video/mp4;codecs=avc1", "video/mp4"];
  return opts.find((m) => window.MediaRecorder && MediaRecorder.isTypeSupported(m)) || "";
}

async function startCamera(deviceId) {
  if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
  const video = { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } };
  if (deviceId) video.deviceId = { exact: deviceId };
  state.stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
  $("setup-video").srcObject = state.stream;
  $("rec-video").srcObject = state.stream;
}

async function enterSetup() {
  show("setup");
  $("setup-error").textContent = "";
  if (!navigator.mediaDevices?.getUserMedia) {
    $("setup-error").textContent = "This browser can't access the camera here. The page must be opened over https:// (or on localhost).";
    return;
  }
  if (!window.MediaRecorder) {
    $("setup-error").textContent = "This browser can't record video. Please use a recent Chrome, Edge, Firefox, or Safari.";
    return;
  }
  try {
    await startCamera();
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
    const current = state.stream.getVideoTracks()[0].getSettings().deviceId;
    $("camera-select").innerHTML = devices.map((d, i) =>
      `<option value="${esc(d.deviceId)}" ${d.deviceId === current ? "selected" : ""}>${esc(d.label || `Camera ${i + 1}`)}</option>`).join("");
    $("setup-start").disabled = false;
    brightnessLoop();
  } catch (e) {
    $("setup-error").textContent = `Couldn't open the camera: ${e.message}. Check that you allowed camera access for this page.`;
  }
}
$("camera-select").addEventListener("change", (e) =>
  startCamera(e.target.value).catch((err) => ($("setup-error").textContent = err.message)));

const probe = document.createElement("canvas"); probe.width = 64; probe.height = 48;
function brightnessLoop() {
  const v = $("setup-video");
  if (!document.querySelector('[data-screen="setup"]').classList.contains("active")) return;
  if (v.videoWidth) {
    const ctx = probe.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(v, 0, 0, 64, 48);
    const px = ctx.getImageData(0, 0, 64, 48).data;
    let sum = 0;
    for (let i = 0; i < px.length; i += 4) sum += 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2];
    const mean = sum / (px.length / 4);
    $("bright-bar").style.width = `${Math.min(100, (mean / 255) * 100)}%`;
    $("bright-label").textContent = mean < 55 ? "too dark" : mean > 215 ? "too bright" : "good";
  }
  setTimeout(brightnessLoop, 500);
}

// ------------------------------------------------------------------ session
$("setup-start").addEventListener("click", async () => {
  $("setup-error").textContent = "";
  const track = state.stream.getVideoTracks()[0];
  const s = track.getSettings();
  state.mime = pickMime();
  try {
    const r = await api("/api/sessions", {
      participant_id: state.pid, resume_code: state.code,
      details: {
        setting: $("d-setting").value, camera_type: $("d-camera").value,
        lighting: $("d-lighting").value, glasses: $("d-glasses").value,
      },
      camera: { label: track.label, width: s.width, height: s.height, frameRate: s.frameRate, mime: state.mime },
      user_agent: navigator.userAgent,
    });
    state.sessionId = r.session_id;
    state.protocolVersion = r.protocol_version;
    state.prompts = Object.fromEntries(r.prompts.map((p) => [p.prompt_id, p]));
    buildQueue(r.remaining);
    if (!state.queue.length) return finish("You've already recorded every clip in the current protocol. Thank you!");
    showPrompt();
  } catch (e) { $("setup-error").textContent = e.message; }
});

function buildQueue(remaining) {
  const main = [], tail = [];
  for (const r of remaining) {
    const p = state.prompts[r.prompt_id];
    for (let i = 0; i < r.needed; i++) (p.kind === "free" ? tail : main).push({ promptId: p.prompt_id, attempt: 1 });
  }
  // Shuffle, then try to avoid the same prompt twice in a row.
  shuffle(main);
  for (let pass = 0; pass < 5; pass++) {
    for (let i = 1; i < main.length; i++) {
      if (main[i].promptId === main[i - 1].promptId) {
        const j = main.findIndex((x, k) => k > i && x.promptId !== main[i].promptId && (k + 1 >= main.length || main[k + 1].promptId !== main[i].promptId));
        if (j > 0) [main[i], main[j]] = [main[j], main[i]];
      }
    }
  }
  state.queue = [...main, ...tail];
  state.pos = 0; state.total = state.queue.length;
}

function showPrompt() {
  const item = state.queue[state.pos];
  const p = state.prompts[item.promptId];
  state.current = item;
  $("prog-label").textContent = `${state.pos + 1} / ${state.total}`;
  $("prog-bar").style.width = `${(state.pos / state.total) * 100}%`;
  $("p-kind").textContent = p.kind === "meme" ? "Meme pose" : p.kind === "free" ? "Last one: 1 minute" : "Everyday movement";
  $("p-name").textContent = p.kind === "meme" ? p.name : p.short;
  $("p-instruction").textContent = p.instruction;
  const img = $("p-image");
  if (p.image) { img.classList.remove("hidden"); img.src = `/meme-images/${encodeURIComponent(p.image)}`; img.onerror = () => img.classList.add("hidden"); }
  else img.classList.add("hidden");
  document.querySelector(".howto").classList.toggle("hidden", p.kind === "free");
  setOverlay("", "");
  $("p-record").disabled = false; $("p-skip").disabled = false;
  show("prompt");
}

function setOverlay(text, cls) {
  const o = $("rec-overlay");
  o.className = `overlay ${cls}`;
  o.textContent = text;
}

// ------------------------------------------------------------------ recording
async function record() {
  if (state.recording) return;
  const p = state.prompts[state.current.promptId];
  state.recording = true;
  $("p-record").disabled = true; $("p-skip").disabled = true;

  for (const n of [3, 2, 1]) { setOverlay(String(n), "count"); await sleep(800); }

  const chunks = [];
  const rec = new MediaRecorder(state.stream, { ...(state.mime ? { mimeType: state.mime } : {}), videoBitsPerSecond: 5_000_000 });
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  const started = new Promise((r) => (rec.onstart = r));
  const stopped = new Promise((r) => (rec.onstop = r));
  const recordedAt = new Date().toISOString();
  rec.start(500);
  await started;
  const t0 = performance.now();
  $("rec-dot").classList.add("on");

  let cueOn = null, cueOff = null;
  const total = p.clip_seconds * 1000;
  if (p.kind === "free") {
    cueOn = 0;
    while (performance.now() - t0 < total) {
      const left = Math.ceil((total - (performance.now() - t0)) / 1000);
      setOverlay(`Just act normally · ${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")} left`, "free");
      await sleep(250);
    }
    cueOff = Math.round(performance.now() - t0);
  } else {
    setOverlay("Neutral face…", "neutral");
    await sleep(p.cue_on * 1000 - (performance.now() - t0));
    cueOn = Math.round(performance.now() - t0);
    setOverlay(`NOW! ${p.short}`, "now");
    await sleep(p.cue_off * 1000 - (performance.now() - t0));
    cueOff = Math.round(performance.now() - t0);
    setOverlay("RELAX, back to neutral", "relax");
    await sleep(total - (performance.now() - t0));
  }
  rec.stop();
  await stopped;
  const duration = Math.round(performance.now() - t0);
  $("rec-dot").classList.remove("on");
  setOverlay("", "");
  state.recording = false;

  state.lastBlob = new Blob(chunks, { type: rec.mimeType || state.mime || "video/webm" });
  state.lastMeta = { cue_on_ms: cueOn, cue_off_ms: cueOff, duration_ms: duration, recorded_at: recordedAt };
  showReview();
}

function showReview() {
  const v = $("review-video");
  if (v.src) URL.revokeObjectURL(v.src);
  v.src = URL.createObjectURL(state.lastBlob);
  v.play().catch(() => {});
  $("review-qc").className = "qc"; $("review-qc").innerHTML = "";
  $("review-actions").classList.remove("hidden");
  $("qc-actions").classList.add("hidden");
  $("review-keep").disabled = false; $("review-redo").disabled = false;
  $("review-keep").textContent = "Keep it";
  show("review");
}

async function upload() {
  const item = state.current;
  const track = state.stream.getVideoTracks()[0];
  const s = track.getSettings();
  const meta = {
    participant_id: state.pid, resume_code: state.code, session_id: state.sessionId,
    prompt_id: item.promptId, attempt: item.attempt, mirrored_preview: true,
    camera: { width: s.width, height: s.height, frameRate: s.frameRate },
    ...state.lastMeta,
  };
  const ext = state.lastBlob.type.includes("mp4") ? "mp4" : "webm";
  const fd = new FormData();
  fd.append("meta", JSON.stringify(meta));
  fd.append("video", state.lastBlob, `clip.${ext}`);
  const res = await fetch("/api/clips", { method: "POST", body: fd });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Upload failed (${res.status})`);
  return data;
}

$("review-keep").addEventListener("click", async () => {
  $("review-keep").disabled = true; $("review-redo").disabled = true;
  const qcBox = $("review-qc");
  qcBox.className = "qc"; qcBox.textContent = "Uploading and checking quality…";
  try {
    const r = await upload();
    state.lastClipId = r.clip_id;
    if (r.status === "ok") {
      qcBox.className = "qc ok"; qcBox.textContent = "Saved ✓";
      await sleep(500);
      next();
    } else {
      qcBox.className = "qc warn";
      qcBox.innerHTML = `<strong>This clip might not be usable:</strong><ul>${r.qc.reasons.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`;
      $("review-actions").classList.add("hidden");
      $("qc-actions").classList.remove("hidden");
    }
  } catch (e) {
    qcBox.className = "qc bad"; qcBox.textContent = `${e.message}. Your clip is still here, so you can try again.`;
    $("review-keep").disabled = false; $("review-redo").disabled = false;
    $("review-keep").textContent = "Retry upload";
  }
});
function redo() { state.current.attempt += 1; showPrompt(); }
$("review-redo").addEventListener("click", redo);
$("qc-redo").addEventListener("click", redo);
$("qc-keep").addEventListener("click", async () => {
  try {
    await api(`/api/clips/${encodeURIComponent(state.lastClipId)}/accept`, { participant_id: state.pid, resume_code: state.code });
    next();
  } catch (e) { $("review-qc").textContent = e.message; }
});

$("p-record").addEventListener("click", record);
$("p-skip").addEventListener("click", () => next());
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && document.querySelector('[data-screen="prompt"]').classList.contains("active") && !$("p-record").disabled) {
    e.preventDefault(); record();
  }
});

function next() {
  state.pos += 1;
  if (state.pos >= state.queue.length) return finish();
  showPrompt();
}

function finish(msg) {
  if (msg) $("done-msg").textContent = msg;
  $("done-contact").textContent = state.contact || "the project team";
  if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
  state.sessionId = null;
  show("done");
}

window.addEventListener("beforeunload", (e) => {
  if (state.sessionId) { e.preventDefault(); e.returnValue = ""; }
});

show("welcome");
