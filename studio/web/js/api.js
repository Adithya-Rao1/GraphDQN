// The one API client. Every URL is RELATIVE ("/api/...") because a gateway proxies them (HF Space -> Lambda tunnel).
// Three modes, decided once at startup:
//   live      : the backend answers /api/health
//   fixtures  : ?fixtures=1, canned data from studio/web/fixtures/ (UI development; everything labelled fixture/mock)
//   replay    : backend unreachable -> recorded runs from studio/web/replay/index.json (labelled replay)
import { bus } from "./bus.js";

const Q = new URLSearchParams(location.search);
const ORIGIN = Q.get("api") || ""; // dev override only; production always uses same-origin relative URLs
const TYPES = ["run_start", "generation", "candidate", "policy_metrics", "run_end", "error"];

export const api = { mode: "live", fixtures: Q.get("fixtures") === "1", replay: null, health: null };

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const abs = (p) => ORIGIN + p;

async function fetchJson(url, opts = {}, timeoutMs = 20000) {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...opts, signal: ctl.signal });
    let data = null;
    const text = await res.text();
    try { data = text ? JSON.parse(text) : null; } catch (_) { data = { raw: text }; }
    if (!res.ok) {
      const d = data?.detail;
      const msg = typeof d === "string" ? d : Array.isArray(d) ? d.map((x) => x.msg).join("; ") : data?.error || `HTTP ${res.status}`;
      const err = new Error(msg); err.status = res.status; throw err;
    }
    return data;
  } finally { clearTimeout(t); }
}

// ----------------------------------------------------------------------------- health + mode
export async function probe() {
  if (api.fixtures) { api.mode = "fixtures"; api.health = { ok: true, fixtures: true }; bus.emit("api:mode", { mode: api.mode }); return api.mode; }
  try {
    api.health = await fetchJson(abs("/api/health"), {}, 5000);
    api.mode = "live";
  } catch (_) {
    api.health = null;
    api.mode = "replay";
    try { api.replay = await fetchJson("replay/index.json", {}, 8000); } catch (_e) { api.replay = null; }
  }
  bus.emit("api:mode", { mode: api.mode, hasReplay: !!api.replay });
  return api.mode;
}

// -------------------------------------------------------------------------------- targets
export async function getTargets() {
  if (api.mode === "fixtures") return fetchJson("fixtures/targets.json");
  if (api.mode === "replay") return api.replay?.targets || [];
  const d = await fetchJson(abs("/api/targets"));
  return Array.isArray(d) ? d : d?.targets || [];
}

/** Protein PDB text. structure_url is "a path under /api/data/"; the documented route is /api/data/structures/<id>.pdb. */
export async function getStructure(t) {
  const tries = [];
  if (api.mode === "fixtures") tries.push(t.structure_url.startsWith("fixtures/") ? t.structure_url : `fixtures/${t.id}.pdb`);
  else if (api.mode === "replay") tries.push(`replay/${t.id}.pdb`, `replay/structures/${t.id}.pdb`);
  else {
    const u = t.structure_url || "";
    if (/^https?:|^\//.test(u)) tries.push(abs(u));
    else if (u) tries.push(abs(`/api/data/${u.replace(/^\/+/, "")}`));
    tries.push(abs(`/api/data/structures/${t.id}.pdb`));
  }
  let last = null;
  for (const u of tries) {
    try { const r = await fetch(u); if (r.ok) return await r.text(); last = new Error(`${u}: HTTP ${r.status}`); } catch (e) { last = e; }
  }
  throw last || new Error("no structure URL");
}

// ----------------------------------------------------------------------------------- validate
export async function validate(smiles) {
  if (api.mode === "fixtures") {
    const p = await fetchJson("fixtures/pose_result.json");
    return { ok: true, canonical_smiles: smiles, molblock: p.ligand_sdf.split("$$$$")[0] };
  }
  if (api.mode === "replay") return { ok: false, error: "offline" };
  return fetchJson(abs("/api/molecules/validate"), jsonBody({ smiles }));
}

const jsonBody = (b) => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b) });

// ------------------------------------------------------------------------------------- runs
const fxRuns = new Map(); // run_id -> config (fixtures / replay)
let fxCounter = 0;

export async function createRun(cfg) {
  if (api.mode === "live") return fetchJson(abs("/api/runs"), jsonBody(cfg));
  const id = `${api.mode === "fixtures" ? "fx" : "rp"}-${++fxCounter}`;
  fxRuns.set(id, cfg);
  return { run_id: id };
}

async function localEvents(cfg) {
  if (api.mode === "fixtures") return fetchJson("fixtures/binding_run.json");
  const runs = api.replay?.runs || [];
  const stage = cfg.stage || "pharmacology";
  const m = runs.find((r) => r.stage === stage && r.target_id === cfg.target_id && r.seed_smiles === cfg.seed_smiles)
    || runs.find((r) => r.stage === stage && r.target_id === cfg.target_id)
    || runs.find((r) => r.stage === stage);
  if (!m) throw new Error(`No recorded ${stage} run available offline`);
  const list = await fetchJson(`replay/${m.file}`);
  return Array.isArray(list) ? list : list.events;
}

/**
 * Open the SSE stream of a run. Reconnects with backoff; the server replays from the start on reconnect
 * (no Last-Event-ID), so events with seq <= the last one seen are dropped.
 * Returns {close}. handlers: onEvent(ev), onDone(), onError(msg), onStatus(text)
 */
export function openRun(runId, { onEvent, onDone, onError, onStatus, paceMs = 130 }) {
  let closed = false, es = null, timer = 0, retries = 0, lastSeq = -1, sawAny = false;
  const finish = () => { closed = true; clearTimeout(timer); es?.close(); };

  if (api.mode !== "live") {
    (async () => {
      try {
        const events = await localEvents(fxRuns.get(runId) || {});
        const cfg = fxRuns.get(runId) || {};
        for (const raw of events) {
          if (closed) return;
          const ev = structuredClone(raw);
          if (ev.type === "run_start") {
            ev.payload.source = api.mode === "replay" ? "replay" : "mock";
            // the recording is of one particular seed; the config the user asked for wins for display only when it matches
            ev.payload.config = { ...ev.payload.config, stage: cfg.stage || ev.payload.config.stage, target_id: cfg.target_id ?? ev.payload.config.target_id };
          }
          onEvent(ev);
          if (ev.type === "run_end" || (ev.type === "error" && !ev.payload?.recoverable)) break;
          await sleep(ev.type === "candidate" ? paceMs : paceMs * 1.5);
        }
        if (!closed) onDone?.();
      } catch (e) { onError?.(e.message || String(e)); }
    })();
    return { close: finish };
  }

  const url = abs(`/api/runs/${encodeURIComponent(runId)}/events`);
  const open = () => {
    if (closed) return;
    es = new EventSource(url);
    for (const t of TYPES) {
      es.addEventListener(t, (e) => {
        if (t === "error" && !e.data) return; // connection-level error is handled by onerror below
        let ev;
        try { ev = JSON.parse(e.data); } catch (err) { console.error("bad SSE payload", err); return; }
        if (ev.seq <= lastSeq) return;
        lastSeq = ev.seq; sawAny = true; retries = 0;
        onEvent(ev);
        if (ev.type === "run_end" || (ev.type === "error" && !ev.payload?.recoverable)) { finish(); onDone?.(); }
      });
    }
    es.onerror = () => {
      if (closed) return;
      es.close();
      if (++retries > 6) { finish(); onError?.(sawAny ? "Lost connection to the run stream." : "Could not open the run event stream."); return; }
      onStatus?.(`Reconnecting to the run stream (attempt ${retries})…`);
      timer = setTimeout(open, Math.min(400 * 2 ** retries, 5000));
    };
  };
  open();
  return { close: finish };
}

// ------------------------------------------------------------------------------------- poses
const fxPoses = new Map(); // job_id -> {t0, req}
let poseCounter = 0;

export async function createPose(req) {
  if (api.mode === "live") return fetchJson(abs("/api/poses"), jsonBody(req));
  if (api.mode === "replay") {
    const hit = (api.replay?.poses || []).find((p) => p.target_id === req.target_id && p.smiles === req.smiles)
      || (api.replay?.poses || []).find((p) => p.target_id === req.target_id);
    if (!hit) throw new Error("No recorded pose available offline for this ligand");
    const id = `rp-pose-${++poseCounter}`;
    fxPoses.set(id, { t0: performance.now(), req, file: hit.file });
    return { job_id: id, status: "queued" };
  }
  const id = `fx-pose-${++poseCounter}`;
  fxPoses.set(id, { t0: performance.now(), req });
  return { job_id: id, status: "queued" };
}

export async function getPose(jobId) {
  if (api.mode === "live") return fetchJson(abs(`/api/poses/${encodeURIComponent(jobId)}`));
  const j = fxPoses.get(jobId);
  if (!j) throw new Error("unknown job");
  const base = { job_id: jobId, target_id: j.req.target_id, smiles: j.req.smiles };
  const dur = api.mode === "fixtures" ? 4200 : 1500; // simulated wait so the progress animation can be developed
  const el = performance.now() - j.t0;
  if (el < dur) return { ...base, status: el < 600 ? "queued" : "running", progress: Math.min(0.95, el / dur) };
  const full = await fetchJson(api.mode === "fixtures" ? "fixtures/pose_result.json" : `replay/${j.file}`);
  const out = { ...full, ...base, status: "done", progress: 1 };
  if (api.mode === "replay") out.source = full.source || "precomputed";
  return out;
}

// ------------------------------------------------------------------------------------- misc
export const apiUrl = (p) => abs(p);
export { fetchJson, jsonBody };
