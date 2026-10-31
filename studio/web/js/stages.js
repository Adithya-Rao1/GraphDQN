// Pipeline UI: stepper (1 Binding -> 2 Pose -> 3 Pharmacology + Manual toggle), target / seed-drug pickers, status chips,
// the stage-1 leads table + live affinity chart, and the stage-3 "binding stays" badge.
// Exposes window.MetisStages = { current, onStageChange(cb), set(stage), state, actions }.
import { bus } from "./bus.js";
import { store } from "./state.js";
import { ingest } from "./run.js";
import * as API from "./api.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

export const STAGES = [
  { id: "binding", n: 1, name: "Binding", sub: "GA + PLAPT · predicted", key: "q" },
  { id: "pose", n: 2, name: "Pose", sub: "DynamicBind · predicted", key: "w" },
  { id: "pharmacology", n: 3, name: "Pharmacology", sub: "RDKit · interpretable", key: "e" },
];

/** Shared pipeline state. main.js fills `pose` once a pose has been produced. */
export const S = {
  targets: [], target: null, drug: null, seedSmiles: null, lead: null, pose: null,
  stage: "binding", manual: false,
  runs: { binding: null, pharmacology: null }, // {cfg, events[], done}
};

/** main.js assigns these; stages.js buttons call them. */
export const actions = { dock: null, optimize: null, loadTarget: null, restore: null };

const cbs = new Set();
let stream = null;

export const affinityOf = (score) => {
  const x = score?.extras;
  if (!x) return null;
  const v = x.plapt_affinity ?? x.plapt_pkd ?? null;
  return typeof v === "number" ? v : null;
};
const tanimotoOf = (score) => {
  const x = score?.extras;
  if (!x) return null;
  const v = x.tanimoto_to_seed ?? x.similarity_to_seed ?? x.tanimoto ?? null;
  return typeof v === "number" ? v : null;
};

// ------------------------------------------------------------------------------------ stepper
function renderStepper() {
  const nav = $("stage-switcher");
  nav.innerHTML = `
    <ol class="steps">${STAGES.map((s, i) => `
      <li class="step" data-stage="${s.id}">
        <button type="button" data-stage="${s.id}" title="${s.name} (${s.key.toUpperCase()})">
          <span class="num">${s.n}</span><span class="lbl"><b>${s.name}</b><small>${s.sub}</small></span><span class="tick" aria-hidden="true">✓</span>
        </button>${i < STAGES.length - 1 ? '<span class="link" aria-hidden="true"><i></i></span>' : ""}
      </li>`).join("")}
    </ol>
    <label class="manual-toggle" title="Edit the molecule by hand with live score deltas (R)">
      <input type="checkbox" id="manual-toggle"><span class="sw"></span><span><b>Manual</b><small>you are the policy</small></span>
    </label>`;
  nav.onclick = (e) => { const b = e.target.closest("button[data-stage]"); if (b) set(b.dataset.stage); };
  $("manual-toggle").onchange = (e) => set(e.target.checked ? "manual" : "pharmacology");
}

function paintStepper() {
  const active = S.stage === "manual" ? "pharmacology" : S.stage;
  const idx = STAGES.findIndex((s) => s.id === active);
  document.querySelectorAll("#stage-switcher .step").forEach((li, i) => {
    li.classList.toggle("on", i === idx);
    li.classList.toggle("past", i < idx);
    const id = STAGES[i].id;
    li.classList.toggle("done", (id === "binding" && !!S.runs.binding?.done) || (id === "pose" && !!S.pose) || (id === "pharmacology" && !!S.runs.pharmacology?.done));
  });
  const mt = $("manual-toggle");
  if (mt) mt.checked = S.stage === "manual";
}

export function set(stage) {
  if (stage === S.stage) return;
  const prev = S.stage;
  closeStream();
  S.stage = stage;
  document.body.dataset.stage = stage;
  $("panel-manual").hidden = stage !== "manual";
  paintStepper();
  paintRunButton();
  renderChips();
  for (const cb of [...cbs]) { try { cb(stage, prev); } catch (e) { console.error("onStageChange handler failed", e); } }
  bus.emit("stage:change", { stage, prev });
}

// ---------------------------------------------------------------------------- target picker
function fillTargets() {
  const sel = $("sel-target");
  sel.innerHTML = S.targets.map((t) => `<option value="${esc(t.id)}">${esc(t.name)}</option>`).join("");
  if (S.target) sel.value = S.target.id;
}

function fillDrugs() {
  const sel = $("sel-drug");
  sel.innerHTML = (S.target?.drugs || []).map((d, i) => `<option value="${i}">${esc(d.name)}</option>`).join("");
  S.drug = S.target?.drugs?.[0] || null;
  S.seedSmiles = S.drug?.smiles || null;
  $("seed-smiles").innerHTML = S.seedSmiles ? `<span class="muted">SMILES</span> <code>${esc(S.seedSmiles)}</code>${S.drug?.source ? ` <span class="muted">· ${esc(S.drug.source)}</span>` : ""}` : "";
}

function paintTargetMeta() {
  const t = S.target;
  $("target-meta").innerHTML = t
    ? `<span class="pdb">PDB ${esc(t.pdb_id)}${t.chain ? " · chain " + esc(t.chain) : ""}</span><span>${(t.pocket_residues || []).length} pocket residues</span><span>${t.sequence ? t.sequence.length + " aa" : ""}</span>${t.note ? `<div class="note">${esc(t.note)}</div>` : ""}`
    : "";
}

export async function selectTarget(id) {
  const t = S.targets.find((x) => x.id === id);
  if (!t) return;
  closeStream();
  delete document.body.dataset.run;
  S.target = t; S.lead = null; S.pose = null; S.runs = { binding: null, pharmacology: null };
  fillDrugs(); paintTargetMeta(); paintStepper(); paintRunButton(); renderLeads(); renderChips();
  bus.emit("ctx:target", { target: t });
  await actions.loadTarget?.(t);
}

function selectDrug(i) {
  closeStream();
  delete document.body.dataset.run;
  S.drug = S.target.drugs[i]; S.seedSmiles = S.drug.smiles; S.lead = null; S.pose = null; S.runs = { binding: null, pharmacology: null };
  $("seed-smiles").innerHTML = `<span class="muted">SMILES</span> <code>${esc(S.seedSmiles)}</code>${S.drug.source ? ` <span class="muted">· ${esc(S.drug.source)}</span>` : ""}`;
  paintStepper(); renderLeads(); renderChips();
  bus.emit("ctx:drug", { drug: S.drug });
}

export async function loadTargets() {
  $("target-skel").hidden = false; $("target-body").hidden = true;
  try {
    S.targets = await API.getTargets();
  } catch (e) {
    S.targets = [];
    showMsg(`Could not load targets: ${e.message}`);
  }
  $("target-skel").hidden = true; $("target-body").hidden = false;
  fillTargets();
  if (!S.targets.length) { $("sel-target").innerHTML = '<option>No targets available</option>'; return; }
  $("sel-target").onchange = (e) => selectTarget(e.target.value);
  $("sel-drug").onchange = (e) => selectDrug(+e.target.value);
  await selectTarget(S.targets[0].id);
}

const showMsg = (t) => { const m = $("stage-msg"); m.hidden = !t; m.textContent = t || ""; };

// ------------------------------------------------------------------------------------ chips
const chip = (txt, kind, title = "") => `<span class="schip ${kind}" title="${esc(title)}">${esc(txt)}</span>`;

export function renderChips() {
  const out = [];
  const mode = API.api.mode;
  if (mode === "replay") out.push(chip("REPLAY", "replay", "The backend is unreachable; this is a recorded run"));
  if (mode === "fixtures") out.push(chip("FIXTURE DATA", "mock", "?fixtures=1: synthetic UI-development data, not results"));
  const st = S.stage === "manual" ? "pharmacology" : S.stage;
  if (st === "binding") {
    out.push(chip("PREDICTED · PLAPT", "pred", "Binding affinity predicted by PLAPT from protein sequence + SMILES. Not measured."));
    if (store.source === "mock") out.push(chip("MOCK RUN", "mock", "Scripted run, not produced by the GA"));
  } else if (st === "pose") {
    const src = S.pose?.result?.source;
    if (src === "dynamicbind" || !src) out.push(chip("PREDICTED · DynamicBind", "pred", "Pose, protein conformation and affinity predicted by DynamicBind"));
    if (src === "precomputed") out.push(chip("PRECOMPUTED", "pre", "Seed-drug pose computed ahead of time with DynamicBind"));
    if (src === "fallback") out.push(chip("FALLBACK · not docked", "mock", "No DynamicBind result: placeholder placement"));
  } else {
    out.push(chip("RDKit descriptors", "pre", "Interpretable descriptor scoring"));
    if (S.pose) out.push(chip("POSE from stage 2", "pre", "Seed ligand starts from the stage-2 pose"));
    if (store.source === "mock") out.push(chip("MOCK RUN", "mock", "Scripted run, not produced by the GA"));
  }
  $("chips").innerHTML = out.join("");
}

// ------------------------------------------------------------------------------- run button
const RUN_LABEL = { binding: "Evolve for binding ▶", pose: "Predict pose ▶", pharmacology: "Optimize pharmacology ▶", manual: "" };
function paintRunButton() {
  const b = $("btn-stage-run");
  b.textContent = RUN_LABEL[S.stage] || "Run";
  b.hidden = S.stage === "manual";
  b.disabled = !S.seedSmiles;
}

export function closeStream() { stream?.close(); stream = null; }

function runConfig(stage, seed) {
  const g = (id, d) => parseInt($(id).value, 10) || d;
  if (stage === "binding") {
    const mw = parseFloat($("t-mw").value);
    return {
      seed_smiles: seed, stage: "binding", target_id: S.target.id, generations: g("t-gens", 8), population_size: g("t-pop", 16), rng_seed: 0,
      policy_guidance: 0, use_crossover: true, min_similarity_to_seed: parseFloat($("t-sim").value), max_mw: mw > 0 ? mw : null,
    };
  }
  const pose = S.pose && S.pose.smiles === seed ? S.pose : null;
  const cfg = {
    seed_smiles: seed, stage: "pharmacology", target_id: S.target?.id ?? null,
    generations: g("cfg-gens", 10), population_size: g("cfg-pop", 20), rng_seed: parseInt($("cfg-seed").value, 10) || 0,
    policy_guidance: parseFloat($("cfg-guid").value), locked_atoms: [...store.locked].sort((a, b) => a - b),
  };
  if (pose?.ligandBlock) cfg.seed_molblock = pose.ligandBlock;
  return cfg;
}

/** Create a run and stream it into the store (same ingest path as every other run). Events are cached per stage for later restore. */
export async function startRun(stage, seed) {
  closeStream();
  showMsg("");
  const cfg = runConfig(stage, seed);
  const rec = { cfg, events: [], done: false };
  S.runs[stage] = rec;
  paintStepper();
  $("btn-stage-run").disabled = true;
  try {
    const { run_id } = await API.createRun(cfg);
    bus.emit("run:created", { run_id, config: cfg });
    store.status = "running";
    stream = API.openRun(run_id, {
      onEvent: (ev) => { rec.events.push(ev); ingest(ev); if (ev.type === "run_end") { rec.done = true; paintStepper(); } },
      onDone: () => { $("btn-stage-run").disabled = false; },
      onStatus: (t) => { $("status").textContent = t; },
      onError: (msg) => { showMsg(msg); $("btn-stage-run").disabled = false; bus.emit("run:error", { payload: { message: msg, recoverable: false } }); },
    });
  } catch (e) {
    showMsg(e.status === 429 ? "The server is busy (run queue full). Try again in a moment." : e.message || String(e));
    $("btn-stage-run").disabled = false;
  }
}

/** Feed a cached stage run back through ingest (used when navigating back to a stage). */
export function restoreRun(stage) {
  const rec = S.runs[stage];
  if (!rec?.events.length) return false;
  for (const ev of rec.events) ingest(ev);
  return true;
}

// ------------------------------------------------------------------------------------ leads
function leadRows() {
  if (!store.seed || store.config?.stage !== "binding") return null;
  const min = store.config.min_similarity_to_seed ?? 0;
  const seed = { smiles: store.seed.smiles, affinity: affinityOf(store.seed.score), tc: 1, isSeed: true };
  const best = new Map();
  for (const c of store.candidates) {
    const aff = affinityOf(c.score), tc = tanimotoOf(c.score);
    if (aff == null) continue;
    if (tc != null && tc < min) continue;
    if (c.smiles === seed.smiles) continue;
    const cur = best.get(c.smiles);
    if (!cur || aff > cur.affinity) best.set(c.smiles, { smiles: c.smiles, affinity: aff, tc, id: c.id, gen: c.generation });
  }
  return { seed, leads: [...best.values()].sort((a, b) => b.affinity - a.affinity).slice(0, 5) };
}

const fmt = (v, d = 2) => (v == null ? "—" : v.toFixed(d));

export function renderLeads() {
  const tbl = $("leads-table"), data = leadRows();
  renderChart(data);
  const dock = $("btn-dock"), opt = $("btn-optimize");
  if (!data) {
    tbl.innerHTML = `<tbody><tr><td class="muted">Run the binding stage to see leads.</td></tr></tbody>`;
    dock.disabled = opt.disabled = !S.seedSmiles;
    return;
  }
  const rows = [data.seed, ...data.leads];
  const base = data.seed.affinity;
  tbl.innerHTML = `<thead><tr><th>#</th><th>Lead (SMILES)</th><th title="Tanimoto similarity to the seed drug">Tc</th><th title="PLAPT predicted affinity">Pred. aff.</th></tr></thead><tbody>${rows.map((r, i) => {
    const d = base != null && !r.isSeed ? r.affinity - base : null;
    const sel = S.lead?.smiles === r.smiles ? " sel" : "";
    return `<tr class="lead${sel}${r.isSeed ? " seedrow" : ""}" data-smiles="${esc(r.smiles)}" data-id="${esc(r.id || "")}" tabindex="0">
      <td>${r.isSeed ? "seed" : i}</td><td class="smi" title="${esc(r.smiles)}">${esc(r.smiles)}</td><td>${fmt(r.tc)}</td>
      <td class="aff">${fmt(r.affinity, 2)}${d != null ? ` <small class="${d >= 0 ? "up" : "dn"}">${d >= 0 ? "+" : ""}${d.toFixed(2)}</small>` : ""}</td></tr>`;
  }).join("")}</tbody>`;
  dock.disabled = opt.disabled = false;
  if (!S.lead) selectLead(data.leads[0] || data.seed, { quiet: true });
}

function selectLead(row, { quiet = false } = {}) {
  S.lead = { smiles: row.smiles, affinity: row.affinity, tc: row.tc, id: row.id };
  document.querySelectorAll("#leads-table tr.lead").forEach((tr) => tr.classList.toggle("sel", tr.dataset.smiles === row.smiles));
  if (!quiet) bus.emit("lead:select", { lead: S.lead });
}

function renderChart(data) {
  const svg = $("aff-chart");
  const W = 440, Hh = 170, L = 42, R = 14, T = 14, B = 28;
  const ps = [];
  if (data && data.seed.affinity != null) {
    ps.push({ g: 0, v: data.seed.affinity });
    const maxG = Math.max(0, ...store.candidates.map((c) => c.generation));
    let run = data.seed.affinity;
    for (let g = 1; g <= maxG; g++) {
      for (const c of store.candidates) if (c.generation === g) { const a = affinityOf(c.score); if (a != null && (tanimotoOf(c.score) ?? 1) >= (store.config.min_similarity_to_seed ?? 0) && a > run) run = a; }
      ps.push({ g, v: run });
    }
  }
  if (!ps.length) {
    svg.innerHTML = `<text x="${W / 2}" y="${Hh / 2}" text-anchor="middle" class="ax">best predicted affinity per generation appears here</text>`;
    return;
  }
  const gmax = Math.max(store.config?.generations || 1, ps[ps.length - 1].g, 1);
  const vs = ps.map((p) => p.v), lo = Math.min(...vs), hi = Math.max(...vs);
  const pad = Math.max((hi - lo) * 0.25, 0.4), y0 = lo - pad, y1 = hi + pad;
  const X = (g) => L + (g / gmax) * (W - L - R), Y = (v) => T + (1 - (v - y0) / (y1 - y0)) * (Hh - T - B);
  const line = ps.map((p, i) => `${i ? "L" : "M"}${X(p.g).toFixed(1)},${Y(p.v).toFixed(1)}`).join(" ");
  const area = `${line} L${X(ps[ps.length - 1].g).toFixed(1)},${Hh - B} L${X(0)},${Hh - B} Z`;
  const ticks = [0, 0.5, 1].map((f) => y0 + f * (y1 - y0));
  const last = ps[ps.length - 1];
  svg.innerHTML = `
    <defs><linearGradient id="affg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="var(--good)" stop-opacity=".35"/><stop offset="1" stop-color="var(--good)" stop-opacity="0"/></linearGradient></defs>
    ${ticks.map((v) => `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" class="grid"/><text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end" class="ax">${v.toFixed(1)}</text>`).join("")}
    <line x1="${L}" x2="${W - R}" y1="${Y(ps[0].v)}" y2="${Y(ps[0].v)}" class="seedline"/>
    <text x="${W - R}" y="${Y(ps[0].v) - 5}" text-anchor="end" class="ax seedlbl">seed drug</text>
    <path d="${area}" fill="url(#affg)"/>
    <path d="${line}" class="aff-line" pathLength="1"/>
    ${ps.map((p) => `<circle cx="${X(p.g).toFixed(1)}" cy="${Y(p.v).toFixed(1)}" r="${p === last ? 5 : 3}" class="${p === last ? "dot last" : "dot"}"/>`).join("")}
    <text x="${X(last.g).toFixed(1)}" y="${Y(last.v) - 10}" text-anchor="${last.g > gmax * 0.7 ? "end" : "start"}" class="best">best ${last.v.toFixed(2)}</text>
    <text x="${(L + W - R) / 2}" y="${Hh - 6}" text-anchor="middle" class="ax">generation</text>
    <text transform="translate(11 ${(T + Hh - B) / 2}) rotate(-90)" text-anchor="middle" class="ax">predicted affinity</text>`;
}

// -------------------------------------------------------------------- stage-3 binding badge
function paintBadge(kf) {
  const el = $("hud-binding");
  const live = S.stage === "pharmacology" || S.stage === "manual";
  const cur = kf ? (kf.kind === "seed" ? store.seed?.score : kf.candidate?.score) : null;
  const a = affinityOf(cur), a0 = affinityOf(store.seed?.score);
  if (!live || a == null) { el.hidden = true; return; }
  const d = a0 != null ? a - a0 : null;
  const kind = d == null ? "" : d >= -0.25 ? "ok" : d >= -0.75 ? "warn" : "bad";
  el.hidden = false;
  el.className = `stage-hud badge ${kind}`;
  el.innerHTML = `<span class="k">Binding stays?</span> <b>${a.toFixed(2)}</b>${d != null ? ` <span class="d">${d >= 0 ? "+" : ""}${d.toFixed(2)} vs seed</span>` : ""}<small>PLAPT · predicted</small>`;
}

// ------------------------------------------------------------------------------------- wiring
export function mountStages() {
  renderStepper();
  document.body.dataset.stage = S.stage;
  $("panel-manual").hidden = true;
  paintStepper();
  paintRunButton();
  renderChips();

  $("btn-stage-run").onclick = () => {
    if (!S.target) return;
    const seed = S.stage === "binding" ? S.seedSmiles : S.lead?.smiles || S.seedSmiles;
    if (S.stage === "binding") startRun("binding", seed);
    else if (S.stage === "pose") actions.dock?.(seed);
    else if (S.stage === "pharmacology") actions.optimize?.(seed);
  };
  $("btn-dock").onclick = () => { set("pose"); actions.dock?.(S.lead?.smiles || S.seedSmiles); };
  $("btn-optimize").onclick = () => { set("pharmacology"); actions.optimize?.(S.lead?.smiles || S.seedSmiles); };
  $("pose-opt").onclick = () => { set("pharmacology"); actions.optimize?.(S.pose?.smiles || S.lead?.smiles || S.seedSmiles); };
  $("t-sim").oninput = (e) => { $("t-sim-out").textContent = Number(e.target.value).toFixed(2); };
  $("leads-table").onclick = (e) => {
    const tr = e.target.closest("tr.lead"); if (!tr) return;
    const row = { smiles: tr.dataset.smiles, id: tr.dataset.id, affinity: parseFloat(tr.querySelector(".aff").textContent) };
    selectLead(row);
    const tl = window.metis?.timeline;
    const i = tl?.keyframes.findIndex((k) => k.id === row.id);
    if (tl && i > 0) tl.seekFrame(i, { end: true });
  };
  $("leads-table").onkeydown = (e) => { if (e.key === "Enter") e.target.click?.(); };

  const refresh = () => { if (store.config?.stage === "binding" || S.stage === "binding") renderLeads(); };
  bus.on("run:start", () => { document.body.dataset.run = "1"; renderLeads(); renderChips(); });
  bus.on("run:candidate", refresh);
  bus.on("run:generation", refresh);
  bus.on("run:end", () => { refresh(); renderChips(); });
  bus.on("view:keyframe", ({ keyframe }) => paintBadge(keyframe));
  bus.on("stage:change", () => paintBadge(window.metis?.timeline ? window.metis.timeline.keyframes[window.metis.timeline.state().index] : null));
  bus.on("api:mode", () => renderChips());

  window.MetisStages = {
    get current() { return S.stage; },
    onStageChange(cb) { cbs.add(cb); return () => cbs.delete(cb); },
    set, state: S, actions, STAGES,
  };
}
