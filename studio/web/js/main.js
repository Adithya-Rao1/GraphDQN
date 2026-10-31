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
import { apiBase, playRecorded, ingest, stopStreams } from "./run.js";
import { parseMolblock } from "./molparse.js";
import * as API from "./api.js";
import { Pocket, centroid } from "./pocket.js";
import { PoseView, heavyMolblock } from "./pose.js";
import { mountStages, set as setStage, S, actions, loadTargets, startRun, restoreRun, closeStream, renderChips, affinityOf } from "./stages.js";

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
const pocket = new Pocket(stage);
const poseView = new PoseView({ stage, pocket });
window.metis = { bus, store, timeline, stage, ingest, perf: stage.perf, pocket, poseView };

// Inside the pocket scene the camera frames the binding site, not just the ligand.
const ligandFit = stage.fit.bind(stage);
stage.fit = () => (pocket.main && pocket.active && !poseView.active ? pocket.fit(0) : ligandFit());

const drawFx = stage._drawOverlay.bind(stage);
stage._drawOverlay = (now) => { drawFx(now); if (pocket.main && pocket.active && !poseView.active) pocket.drawAura(now); };

// Ligand coordinates from the server are in their own frame. When they are not already in the protein frame (no pose yet), the whole
// run is translated (rigidly) onto the pocket centre so it can be shown inside the scene. This is a schematic placement, not docking.
let placeOffset = [0, 0, 0], placed = false;
function shiftMolblock(block, d) {
  if (!block || (!d[0] && !d[1] && !d[2])) return block;
  const lines = block.replace(/\r/g, "").split("\n");
  const n = parseInt(lines[3].slice(0, 3), 10);
  for (let i = 0; i < n; i++) {
    const l = lines[4 + i];
    const f = (a, b, dv) => (parseFloat(l.slice(a, b)) + dv).toFixed(4).padStart(10);
    lines[4 + i] = f(0, 10, d[0]) + f(10, 20, d[1]) + f(20, 30, d[2]) + l.slice(30);
  }
  return lines.join("\n");
}
const place = (b) => shiftMolblock(b, placeOffset);
function updatePlacement() {
  placeOffset = [0, 0, 0]; placed = false;
  if (!pocket.main || !pocket.active || !store.seed?.molblock) return;
  try {
    const atoms = parseMolblock(store.seed.molblock).atoms;
    const c = centroid(atoms);
    const pose = S.pose && S.pose.smiles === store.seed.smiles ? centroid(parseMolblock(S.pose.ligandBlock).atoms) : null;
    const tgt = pose || pocket.center;
    const d = [tgt[0] - c[0], tgt[1] - c[1], tgt[2] - c[2]];
    if (Math.hypot(...d) < 4) return; // already in the protein frame
    placeOffset = d; placed = !pose;
  } catch (e) { console.warn("placement failed", e); }
}

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
  return { id: "seed", kind: "seed", morph: new Morph({ childBlock: place(store.seed.molblock) }), morphMs: 0, holdMs: DEFAULTS.seedHoldMs, label: "Seed", smilesOut: store.seed.smiles };
}

function editKeyframe(c, prev) {
  const ref = prev && prev.smilesOut === c.parent_smiles ? prev.morph.childAtoms : null;
  let morph;
  try {
    morph = new Morph({ parentBlock: place(c.parent_molblock), childBlock: place(c.molblock), atomMap: c.atom_map, refAtoms: ref });
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
  updatePlacement();
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
  updatePlacement();
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

// ----------------------------------------------------------------------------- pipeline wiring
const poseUi = { lastSmiles: null };
let structureText = null, sceneFlavor = null; // flavor: "apo" (target structure) | "pose" (DynamicBind protein)
const skeleton = (on, text) => { $("stage-skeleton").hidden = !on; if (text) $("sk-text").textContent = text; };

// scene loads are serialised: stage changes and run buttons can both ask for a scene at the same moment
let sceneLock = Promise.resolve(), viewToken = 0, docking = false;
const ensureScene = (flavor) => (sceneLock = sceneLock.then(() => ensureSceneNow(flavor)).catch((e) => console.error(e)));
async function ensureSceneNow(flavor) {
  const t = S.target;
  if (!t || !structureText) return;
  if (pocket.main && sceneFlavor === flavor && pocket.target?.id === t.id) return;
  const usePose = flavor === "pose" && S.pose?.proteinPdb;
  await pocket.load(t, usePose ? S.pose.proteinPdb : structureText);
  sceneFlavor = usePose ? "pose" : "apo";
  pocket.fit(0);
}

async function loadTargetScene(t) {
  skeleton(true, `Loading ${t.name}…`);
  setStatus(`Loading ${t.name}…`);
  poseView.leave();
  pocket.clear(); sceneFlavor = null; structureText = null;
  try {
    structureText = await API.getStructure(t);
    await pocket.load(t, structureText);
    sceneFlavor = "apo";
    setStatus(`${t.name} · ${t.pocket_residues?.length || 0} pocket residues`);
  } catch (e) {
    console.error(e);
    setStatus(`Could not load the structure of ${t.name}: ${e.message}`);
  }
  skeleton(false);
  await enterStage(S.stage);
}
actions.loadTarget = loadTargetScene;

async function previewSeed(smiles, block, tok = viewToken) {
  if (!smiles) return;
  try {
    let canonical = smiles, score = null;
    if (!block) {
      const v = await API.validate(smiles);
      if (!v?.ok || !v.molblock) throw new Error(v?.error || "no 3D coordinates");
      block = v.molblock; canonical = v.canonical_smiles || smiles; score = v.score || null;
    }
    if (tok !== viewToken || store.status === "running") return; // a run / other stage took over while the preview was loading
    $("stage-empty").hidden = true;
    timeline.live = false;
    showPreview({ canonical_smiles: canonical, molblock: block, score }, null);
  } catch (e) {
    setStatus(`Could not preview ${smiles}: ${e.message}`);
  }
}

/** Put the 3D scene and timeline into the right state for a stage (also used after a target change). */
async function enterStage(st) {
  const tok = ++viewToken;
  if (st !== "pose" && poseView.active) { poseView.leave(); lastKf && stage.setKeyframe(lastKf); }
  if (st === "pose") {
    timeline.pause();
    closeStream();
    $("stage-empty").hidden = true;
    await ensureScene("apo");
    if (!docking && S.pose && poseView.result && S.pose.smiles === (S.lead?.smiles || S.seedSmiles)) await playCachedPose();
    else if (!poseView.active) poseView._enter();
    paintPoseCard();
    return;
  }
  await ensureScene(st !== "binding" && S.pose ? "pose" : "apo");
  const rec = S.runs[st === "manual" ? "pharmacology" : st];
  const want = st === "binding" ? "binding" : "pharmacology";
  if (rec?.events.length && store.config?.stage !== want) {
    stopStreams();
    restoreRun(want === "binding" ? "binding" : "pharmacology");
    return;
  }
  if (store.config?.stage === want && store.status !== "ready") return; // already showing this stage's run
  if (rec?.events.length) return;
  const seed = want === "binding" ? S.seedSmiles : S.lead?.smiles || S.seedSmiles;
  const pose = S.pose && S.pose.smiles === seed ? S.pose : null;
  if (tok !== viewToken) return;
  store.reset();
  await previewSeed(seed, pose?.ligandBlock, tok);
}

mountStages();
window.MetisStages.onStageChange((st) => { enterStage(st).catch((e) => console.error(e)); });

async function playCachedPose() {
  skeleton(false);
  const p = S.pose;
  await poseView.replay();
  poseUi.lastSmiles = p.smiles;
}

function movedList(moved) {
  return (moved || []).filter((m) => m.d >= 0.5).slice(0, 5).map((m) => `${esc(m.resn)} ${m.resi} <b>${m.d.toFixed(1)} Å</b>`).join(" · ") || "no residue moved more than 0.5 Å";
}

function paintPoseCard() {
  const body = $("pose-body"), r = S.pose?.result, ctl = $("pose-controls");
  ctl.hidden = !r;
  if (!r) return;
  const f = (v, d = 2) => (typeof v === "number" ? v.toFixed(d) : "n/a");
  const lead = S.lead?.smiles === S.pose.smiles ? S.lead.affinity : null;
  body.innerHTML = `
    <div class="kv"><span>Ligand</span><code>${esc(S.pose.smiles)}</code></div>
    <div class="kv"><span>Pose confidence</span><b>${f(r.pose_confidence)}</b></div>
    <div class="kv"><span>DynamicBind affinity</span><b>${f(r.affinity, 1)}</b></div>
    <div class="kv"><span>PLAPT affinity (stage 1)</span><b>${typeof lead === "number" ? lead.toFixed(1) : "— (run stage 1)"}</b></div>
    <div class="kv"><span>Most moved residues</span><span>${movedList(S.pose.moved)}</span></div>
    <p class="plain">What this means: DynamicBind predicts <i>how</i> the drug sits in the pocket and how the protein bends around it. Its affinity and PLAPT's affinity are two separate model predictions, not measurements, and they can disagree. Treat both as hypotheses to test.</p>
    ${r.note ? `<p class="note">${esc(r.note)}</p>` : ""}`;
}

actions.dock = async (smiles) => {
  if (!S.target || !smiles) return;
  docking = true; viewToken++;
  if (S.stage !== "pose") setStage("pose");
  await ensureScene("apo");
  const lead = S.lead?.smiles === smiles ? S.lead.affinity : S.runs.binding?.done ? null : affinityOfSeed(smiles);
  $("pose-body").innerHTML = `<p class="muted">Predicting the pose of <code>${esc(smiles)}</code>…</p>`;
  $("pose-controls").hidden = true;
  $("btn-stage-run").disabled = true;
  const done = await poseView.start({
    target: S.target, smiles, structureText, leadAffinity: lead,
    onResult: (d) => {
      S.pose = { smiles, target_id: S.target.id, ligandBlock: d.ligandBlock, proteinPdb: d.proteinPdb, result: d.result, moved: d.moved };
      sceneFlavor = "apo";
      paintPoseCard(); renderChips(); bus.emit("pose:done", { pose: S.pose });
    },
  });
  docking = false;
  $("btn-stage-run").disabled = false;
  if (!done) paintPoseFailure();
};
const affinityOfSeed = (smiles) => (smiles === S.seedSmiles && store.seed ? affinityOf(store.seed.score) : null);
function paintPoseFailure() { if (!S.pose) $("pose-body").innerHTML = `<p class="muted">No pose yet. Try again with “Predict pose”.</p>`; }

actions.optimize = async (smiles) => {
  if (!smiles) return;
  viewToken++;
  if (S.stage !== "pharmacology") setStage("pharmacology");
  await ensureScene(S.pose ? "pose" : "apo");
  viewToken++;
  stopStreams();
  store.reset();
  await startRun("pharmacology", smiles);
};

bus.on("pose:done", () => { renderChips(); document.querySelectorAll("#stage-switcher .step")[1]?.classList.add("done"); });
$("pose-replay").onclick = () => poseView.replay();
$("pose-apo").onchange = (e) => poseView.setOverlay(e.target.checked);
$("pose-disp").onchange = (e) => poseView.setHighlight(e.target.checked);

// the custom-molecule card (stage 3): same run, plus the pipeline fields upload.js does not know about
$("btn-run").onclick = () => { if (store.seed) startRun("pharmacology", store.seed.smiles); };

// ------------------------------------------------------------------ banner, mode chip, help
function paintMode() {
  const chip = $("mode-chip"), m = API.api.mode;
  chip.hidden = m === "replay"; // the banner and the source badge already say so
  chip.className = "mode-chip " + m;
  chip.textContent = m === "live" ? "LIVE" : m === "fixtures" ? "FIXTURES" : "REPLAY";
  chip.title = m === "live" ? "Connected to the backend" : m === "fixtures" ? "Synthetic UI-development data (?fixtures=1)" : "Backend offline: recorded run";
  const b = $("offline-banner");
  b.hidden = m !== "replay";
  if (m === "replay") $("offline-detail").textContent = API.api.replay ? "Everything below is pre-recorded." : "No recording is available either, so there is nothing to show.";
}
$("btn-retry").onclick = () => location.reload();
const help = $("help");
const toggleHelp = (on) => { help.hidden = on === undefined ? !help.hidden : !on; };
$("btn-help").onclick = () => toggleHelp();
$("help-close").onclick = () => toggleHelp(false);
help.addEventListener("click", (e) => { if (e.target === help) toggleHelp(false); });

window.addEventListener("keydown", (e) => {
  if (e.target.matches("input, textarea, select") || e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key;
  const go = { q: "binding", w: "pose", e: "pharmacology", r: "manual" }[k.toLowerCase()];
  if (go) { setStage(S.stage === go && go === "manual" ? "pharmacology" : go); e.preventDefault(); }
  else if (k === "?") toggleHelp();
  else if (k === "Escape") toggleHelp(false);
  else if (k === "Enter" && e.target.tagName !== "BUTTON" && !$("btn-stage-run").disabled) $("btn-stage-run").click();
  else if (k.toLowerCase() === "d" && !$("btn-dock").disabled) $("btn-dock").click();
  else if (k.toLowerCase() === "p" && !$("btn-optimize").disabled) $("btn-optimize").click();
  else if (k.toLowerCase() === "a" && S.stage === "pose") { const c = $("pose-apo"); c.checked = !c.checked; c.onchange({ target: c }); }
});

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
  setStatus("Connecting…");
  const mode = await API.probe();
  paintMode();
  if (mode === "live") setStatus(API.api.health?.stub ? "Server connected (stub server: every run replays the scripted mock data)" : "Server connected");
  else if (mode === "fixtures") setStatus("Fixture mode: synthetic data for UI development");
  else setStatus("Backend offline: showing a recorded replay");
  await loadTargets();
  if (q.get("smiles")) { upload.setSmiles(q.get("smiles")); upload.check(); }
  const st = q.get("stage");
  if (st && ["binding", "pose", "pharmacology", "manual"].includes(st)) setStage(st);
}
startup();
