// Bootstrap: wires store, timeline, stage, why-card, overlays, upload and export together.
import { bus } from "./bus.js";
import { store } from "./state.js";
import { Timeline, DEFAULTS } from "./timeline.js";
import { mountTimelineUI } from "./timeline_ui.js";
import { Morph } from "./morph.js";
import { Stage } from "./stage.js";
import { SCALES, scaleCss } from "./overlays.js";
import { renderWhy } from "./whycard.js";
import { mountUpload } from "./upload.js";
import { apiBase, playRecorded, ingest } from "./run.js";

const $ = (id) => document.getElementById(id);
const q = new URLSearchParams(location.search);
const MOCK = q.get("mock") === "1";
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ---------------------------------------------------------------------------- theme
// (the initial theme is set by the inline script in index.html, default dark)
function applyTheme(t) { document.documentElement.dataset.theme = t; }
$("btn-theme").onclick = () => {
  const dark = getComputedStyle(document.documentElement).colorScheme.includes("dark");
  const next = dark ? "light" : "dark";
  applyTheme(next);
  try { localStorage.setItem("metis-theme", next); } catch (_) { /* ignore */ }
};

// ------------------------------------------------------------------------ core objects
const timeline = new Timeline();
let stage;
try {
  stage = new Stage({ host: $("stage"), overlay: $("stage-fx") });
} catch (err) {
  console.error(err);
  $("stage-empty").innerHTML = `<div class="big">3D viewer unavailable</div><div class="small">${esc(err.message)}. Check your connection or vendor/3Dmol-min.js.</div>`;
  throw err;
}
let filter = "accepted";
window.metis = { bus, store, timeline, stage, ingest, perf: stage.perf };

const setStatus = (t) => { $("status").textContent = t; };

function setBadge(source) {
  const b = $("source-badge");
  b.hidden = !source;
  if (!source) return;
  b.className = "source-badge " + source;
  b.textContent = source === "mock" ? "Mock data" : source === "preview" ? "Preview" : source;
  b.title = source === "mock" ? "Scripted run with real RDKit descriptors, not produced by the GA" : "";
}

// ------------------------------------------------------------------------- keyframes
function seedKeyframe() {
  return { id: "seed", kind: "seed", morph: new Morph({ childBlock: store.seed.molblock }), morphMs: 0, holdMs: DEFAULTS.seedHoldMs, label: "Seed", smilesOut: store.seed.smiles };
}

function editKeyframe(c, prev) {
  const ref = prev && prev.smilesOut === c.parent_smiles ? prev.morph.childAtoms : null;
  let morph;
  try {
    morph = new Morph({ parentBlock: c.parent_molblock, childBlock: c.molblock, atomMap: c.atom_map, refAtoms: ref });
  } catch (err) {
    console.error("could not build morph for", c.id, err);
    return null;
  }
  return {
    id: c.id, kind: "edit", candidate: c, morph, label: c.edit?.name || "edit", rejected: !c.accepted, smilesOut: c.smiles,
    morphMs: morph.static ? 0 : DEFAULTS.morphMs, holdMs: DEFAULTS.holdMs,
  };
}

function buildAll() {
  if (!store.seed) return [];
  const kfs = [seedKeyframe()];
  for (const c of store.candidates) {
    if (filter === "accepted" && !c.accepted) continue;
    const kf = editKeyframe(c, kfs[kfs.length - 1]);
    if (kf) kfs.push(kf);
  }
  return kfs;
}

function rebuild(keepId) { timeline.replace(buildAll(), keepId); }

// -------------------------------------------------------------------------- run events
bus.on("run:created", () => setStatus("Run created, waiting for the first event…"));

bus.on("run:start", ({ payload }) => {
  timeline.clear();
  timeline.live = true;
  const kfs = buildAll();
  for (const kf of kfs) timeline.add(kf);
  $("card-start").classList.add("collapsed");
  $("stage-empty").hidden = true;
  setBadge(payload.source);
  setStatus(`Running · ${payload.config.generations} generations · population ${payload.config.population_size}${payload.source === "mock" ? " · MOCK DATA" : ""}`);
  timeline.seekFrame(0);
  if (q.get("autoplay") !== "0") timeline.play();
});

bus.on("run:candidate", ({ candidate }) => {
  if (filter === "accepted" && !candidate.accepted) return;
  const prev = timeline.keyframes[timeline.keyframes.length - 1];
  const kf = editKeyframe(candidate, prev);
  if (kf) timeline.add(kf);
});

bus.on("run:generation", ({ payload }) => {
  setStatus(`Generation ${payload.generation} · best ${payload.best.composite.toFixed(3)} · mean ${payload.mean_composite.toFixed(3)}${store.source === "mock" ? " · MOCK DATA" : ""}`);
});

bus.on("run:end", ({ payload }) => {
  timeline.live = false;
  setStatus(payload.headline);
  bus.emit("timeline:keyframes", { timeline });
});

bus.on("run:error", ({ payload }) => {
  timeline.live = false;
  setStatus("Error: " + payload.message);
  const m = $("validate-msg");
  m.hidden = false; m.className = "msg"; m.textContent = payload.message;
  $("btn-run").disabled = !store.seed;
});

// --------------------------------------------------------------- keyframe / tick -> views
let lastKf = null, legendKey = "";

function updateHud(kf, idx, total) {
  const t = $("hud-title");
  if (!kf) { t.innerHTML = ""; return; }
  if (kf.kind === "seed") {
    t.innerHTML = `<div class="hud-edit">Seed</div><div class="hud-sub">${esc(store.seed.smiles)}</div>`;
  } else {
    t.innerHTML = `<div class="hud-edit">${idx}. ${esc(kf.label)}</div><div class="hud-sub">${esc(kf.candidate.parent_smiles || "")} → ${esc(kf.candidate.smiles)}</div>`;
  }
  const preRun = kf.kind === "seed" && store.status === "ready";
  $("hud-hint").textContent = preRun ? "Click an atom to lock it (locked atoms are never edited) · drag to rotate · scroll to zoom" : "Drag to rotate · scroll to zoom · Space = play/pause · ←/→ = step";
}

function updateLegend() {
  const mode = store.overlay.mode, box = $("hud-legend"), side = $("ov-legend");
  const key = mode + "|" + (lastKf?.id || "") + "|" + stage.colorInfo.available + "|" + stage.colorInfo.approx;
  if (key === legendKey) return;
  legendKey = key;
  if (mode === "element") { box.innerHTML = ""; side.hidden = true; return; }
  const sc = SCALES[mode];
  let html;
  if (!stage.colorInfo.available) {
    html = `<div class="legend-ttl">${sc.title}</div><div class="legend-lbl"><span>No per-atom data for this step${lastKf?.kind === "seed" ? " (the seed event carries none)" : ""}</span></div>`;
  } else {
    html = `<div class="legend-ttl">${sc.title}</div><div class="legend-bar" style="background:${scaleCss(sc)}"></div><div class="legend-lbl"><span>${sc.negLabel} (−${sc.span})</span><span>(+${sc.span}) ${sc.posLabel}</span></div>` +
      (stage.colorInfo.approx ? `<div class="legend-lbl"><span>Parent values unavailable: unchanged atoms use child values</span></div>` : "");
  }
  box.innerHTML = html;
  side.hidden = true;
}

bus.on("timeline:keyframe", ({ index, keyframe }) => {
  lastKf = keyframe;
  stage.setKeyframe(keyframe);
  const total = timeline.keyframes.length - 1;
  $("why-body").innerHTML = renderWhy(keyframe, index, total);
  $("card-why").classList.remove("revealed");
  updateHud(keyframe, index, total);
  legendKey = "";
  $("stage-empty").hidden = true;
  bus.emit("view:keyframe", { index, keyframe });
});

bus.on("timeline:keyframes", () => {
  // the "n" in "Edit i / n" grows while streaming; refresh the badge text only
  const pill = document.querySelector("#why-body .pill");
  const s = timeline.state();
  if (pill && s.keyframe?.kind === "edit") pill.textContent = `Edit ${s.index} / ${timeline.keyframes.length - 1}`;
});

bus.on("timeline:tick", (s) => {
  if (!s.keyframe) return;
  stage.setProgress(s.keyframe.kind === "seed" ? 1 : s.u);
  $("card-why").classList.toggle("revealed", s.keyframe.kind === "seed" || s.u >= 0.55);
  updateLegend();
});

// ------------------------------------------------------------------------ overlays UI
function setMode(mode) {
  store.overlay.mode = mode;
  for (const b of $("ov-color").children) b.classList.toggle("on", b.dataset.mode === mode);
  legendKey = "";
  stage.invalidate();
}
$("ov-color").onclick = (e) => { const b = e.target.closest("button[data-mode]"); if (b) setMode(b.dataset.mode); };
$("ov-alerts").onchange = (e) => { store.overlay.alerts = e.target.checked; };
$("ov-highlight").onchange = (e) => { store.overlay.highlightEdit = e.target.checked; stage.invalidate(); };
$("ov-spin").onchange = (e) => { store.overlay.autorotate = e.target.checked; };
if (q.get("overlay")) setMode(q.get("overlay"));
if (q.get("spin") === "0") { store.overlay.autorotate = false; $("ov-spin").checked = false; }

// -------------------------------------------------------------------------- atom locks
function updateLockNote() {
  const n = $("lock-note"), ready = store.status === "ready";
  n.hidden = !ready;
  if (!ready) return;
  n.textContent = store.locked.size ? `🔒 ${store.locked.size} atom${store.locked.size === 1 ? "" : "s"} locked: ${[...store.locked].sort((a, b) => a - b).join(", ")}` : "Tip: click atoms in the 3D view to lock them before evolving.";
}
bus.on("stage:atomclick", ({ index }) => {
  if (store.status !== "ready") return;
  if (store.locked.has(index)) store.locked.delete(index); else store.locked.add(index);
  updateLockNote();
  bus.emit("lock:change", { locked: [...store.locked] });
});

// ----------------------------------------------------------------------------- upload
function showPreview(data, request) {
  const upload = $("validate-msg");
  store.reset();
  store.locked = new Set();
  const molblock = data.molblock || request?.molfile || null;
  if (!molblock) {
    timeline.clear();
    store.status = "ready";
    updateLockNote();
    upload.hidden = false; upload.className = "msg info";
    upload.textContent = "Molecule accepted. The server returned no 3D coordinates, so there is no preview and atoms cannot be locked.";
    setBadge(null);
    return;
  }
  store.source = "preview";
  store.seed = { smiles: data.canonical_smiles, molblock, score: data.score || null };
  store.status = "ready";
  if (data.score) store.remember(data.canonical_smiles, { score: data.score, alerts: data.score.alerts, composite: data.score.composite });
  timeline.live = false;
  timeline.clear();
  timeline.add(seedKeyframe());
  timeline.seekFrame(0);
  setBadge("preview");
  setStatus(`Previewing ${data.canonical_smiles}`);
  $("stage-empty").hidden = true;
  if (!data.molblock) {
    upload.hidden = false; upload.className = "msg info";
    upload.textContent = "Showing the coordinates from your file (the server returned none).";
  }
  updateLockNote();
}

const upload = mountUpload({ onPreview: showPreview, onStatus: setStatus });
$("card-start").querySelector("h2").onclick = () => $("card-start").classList.remove("collapsed");

bus.on("run:start", () => updateLockNote());
bus.on("run:end", () => updateLockNote());

// ------------------------------------------------------------------------------ timeline UI
mountTimelineUI(timeline, {
  onFilter: (f) => { filter = f; rebuild(); },
});

// -------------------------------------------------------------------------------- export
function exportPng() {
  if (!lastKf) return setStatus("Nothing to export yet.");
  const s = timeline.state();
  const name = lastKf.kind === "seed" ? "Seed" : `${s.index}. ${lastKf.label}`;
  const smi = lastKf.kind === "seed" ? store.seed.smiles : lastKf.candidate.smiles;
  const bg = getComputedStyle(document.documentElement).getPropertyValue("--stage-b").trim() || "#06080e";
  const url = stage.composeImage({ caption: `Metis Lead Studio · ${name} · ${smi}`, badge: store.source === "mock" ? "MOCK DATA" : "", bg });
  const a = document.createElement("a");
  a.href = url;
  a.download = `metis-${lastKf.kind === "seed" ? "seed" : "edit-" + s.index}.png`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  bus.emit("export:png", { url });
}
$("btn-export").onclick = exportPng;
window.metis.exportPng = exportPng;

// ------------------------------------------------------------------------------ keyboard
window.addEventListener("keydown", (e) => {
  if (e.target.matches("input, textarea, select") || e.metaKey || e.ctrlKey) return;
  if (e.key === " ") { timeline.toggle(); e.preventDefault(); }
  else if (e.key === "ArrowRight") timeline.step(1);
  else if (e.key === "ArrowLeft") timeline.step(-1);
  else if ("1234".includes(e.key) && e.key) timeline.setSpeed([0.5, 1, 2, 4][+e.key - 1]);
  else if (e.key === "s") exportPng();
});

// ----------------------------------------------------------------------------- perf HUD
if (q.get("perf") === "1") {
  const hud = $("hud-perf");
  hud.hidden = false;
  setInterval(() => {
    const p = stage.perf;
    hud.textContent = `${p.fps} fps · render ${p.avgMs.toFixed(1)} ms (worst ${p.worstMs.toFixed(1)})`;
  }, 500);
}

// ------------------------------------------------------------------------------ startup
async function startup() {
  const mockUrls = ["../mock/mock_run.json", "mock_run.json", "/studio/mock/mock_run.json"];
  if (MOCK) {
    setStatus("Loading mock run…");
    $("btn-mock").hidden = false;
    $("btn-mock").onclick = () => playRecorded(mockUrls).catch((err) => setStatus(err.message));
    try {
      await playRecorded(mockUrls);
    } catch (err) {
      setStatus(err.message);
      const m = $("validate-msg"); m.hidden = false; m.className = "msg"; m.textContent = err.message;
    }
    return;
  }
  try {
    const r = await fetch(apiBase() + "/api/health");
    const h = await r.json();
    setStatus(h.stub ? "Server connected (stub server: every run replays the scripted mock data)" : "Server connected");
  } catch (_) {
    setStatus("Server not reachable. Open with ?mock=1 to play the recorded mock run without a backend.");
  }
  if (q.get("smiles")) { upload.setSmiles(q.get("smiles")); upload.check(); }
}
startup();
