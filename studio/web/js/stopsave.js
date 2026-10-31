// Stop button for running optimizations. Stopping keeps everything found so far: it cancels the server run and
// closes the stream, then ends the session with a story card built from the best candidate seen
// (labelled "stopped early"), so the usual "Save this run" button appears.
const m = () => window.metis;
let btn;

function best(store) {
  const cands = store.candidates || [];
  let b = null;
  for (const c of cands) if (!b || (c.score?.composite ?? 0) > (b.score?.composite ?? 0)) b = c;
  return b;
}

function pathTo(store, cand) {
  const byChild = new Map();
  for (const c of store.candidates) if (!byChild.has(c.smiles)) byChild.set(c.smiles, c);
  const out = [], seen = new Set();
  for (let c = cand; c && !seen.has(c.id); c = byChild.get(c.parent_smiles)) { seen.add(c.id); out.unshift(c.id); }
  return out;
}

async function stop() {
  const { store, ingest } = m();
  if (store.status !== "running") return;
  btn.disabled = true;
  const runId = store.runId;
  const { stopStreams, apiBase } = await import("./run.js");
  try { await fetch(`${apiBase()}/api/runs/${encodeURIComponent(runId)}`, { method: "DELETE", headers: { "X-Requested-With": "studio" }, credentials: "include" }); } catch (e) { /* the run may already be over */ }
  stopStreams();
  const cand = best(store);
  const seed = store.seed;
  if (!cand || !seed) { store.status = "ready"; sync(); return; }
  const rows = {};
  for (const k of Object.keys(seed.score.descriptors)) rows[k] = [seed.score.descriptors[k], cand.score.descriptors[k]];
  const n = pathTo(store, cand).length;
  ingest({
    type: "run_end", run_id: runId, seq: 1e9,
    payload: {
      seed_smiles: seed.smiles, final_smiles: cand.smiles, seed_score: seed.score, final_score: cand.score,
      edit_path: pathTo(store, cand),
      headline: `Stopped early: best of ${store.candidates.length} candidates so far, ${n} edit${n === 1 ? "" : "s"} from the seed, composite ${seed.score.composite.toFixed(2)} → ${cand.score.composite.toFixed(2)}`,
      descriptor_table: rows,
    },
  });
}

function sync() {
  if (!btn) return;
  btn.hidden = m()?.store.status !== "running";
  btn.disabled = false;
}

function init() {
  const bar = document.querySelector(".topbar");
  const bus = m()?.bus;
  if (!bar || !bus) return false;
  btn = document.createElement("button");
  btn.className = "btn"; btn.id = "btn-stop"; btn.hidden = true;
  btn.title = "Stop now and keep what has been found so far";
  btn.textContent = "■ Stop & keep results";
  btn.onclick = stop;
  bar.insertBefore(btn, document.getElementById("btn-export") || null);
  for (const e of ["run:start", "run:end", "run:error"]) bus.on(e, sync);
  sync();
  return true;
}

let tries = 0;
(function wait() { if (!init() && tries++ < 50) setTimeout(wait, 100); })();
