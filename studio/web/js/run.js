// Run ingestion: turns SSE events (or a recorded mock file) into store updates + bus events.
import { bus } from "./bus.js";
import { store } from "./state.js";

const TYPES = ["run_start", "generation", "candidate", "policy_metrics", "run_end", "error"];
let es = null;
let mockTimer = 0;

export function apiBase() {
  return new URLSearchParams(location.search).get("api") || "";
}

/** Apply one schema Event {type, run_id, seq, payload}. Exported so tests / wave-2 code can feed events. */
export function ingest(ev) {
  const p = ev.payload;
  switch (ev.type) {
    case "run_start":
      store.reset();
      store.runId = ev.run_id;
      store.source = p.source;
      store.config = p.config;
      store.status = "running";
      store.seed = { smiles: p.config.seed_smiles, molblock: p.seed_molblock, score: p.seed_score };
      store.remember(p.config.seed_smiles, { score: p.seed_score, alerts: p.seed_score.alerts, composite: p.seed_score.composite });
      store.locked = new Set(p.config.locked_atoms || []);
      bus.emit("run:start", { event: ev, payload: p });
      break;
    case "candidate":
      store.candidates.push(p);
      store.remember(p.smiles, {
        score: p.score, alerts: p.score.alerts, composite: p.score.composite,
        contributions: p.explanation?.atom_contributions || undefined,
      });
      bus.emit("run:candidate", { event: ev, candidate: p, index: store.candidates.length - 1 });
      break;
    case "generation":
      store.generations.push(p);
      bus.emit("run:generation", { event: ev, payload: p });
      break;
    case "policy_metrics":
      store.policy.push(p);
      bus.emit("run:policy", { event: ev, payload: p });
      break;
    case "run_end":
      store.end = p;
      store.status = "done";
      bus.emit("run:end", { event: ev, payload: p });
      break;
    case "error":
      store.status = p.recoverable ? store.status : "error";
      bus.emit("run:error", { event: ev, payload: p });
      break;
    default:
      console.warn("unknown event type", ev.type);
  }
}

export function stopStreams() {
  if (es) { es.close(); es = null; }
  clearTimeout(mockTimer);
}

/** Open the SSE stream of a created run. */
export function openStream(runId) {
  stopStreams();
  const q = new URLSearchParams(location.search);
  const delay = q.get("delay");
  es = new EventSource(`${apiBase()}/api/runs/${encodeURIComponent(runId)}/events${delay ? `?delay=${encodeURIComponent(delay)}` : ""}`);
  let sawAny = false;
  for (const t of TYPES) {
    es.addEventListener(t, (e) => {
      if (t === "error" && !e.data) return; // connection-level error, handled below
      sawAny = true;
      let ev;
      try { ev = JSON.parse(e.data); } catch (err) { console.error("bad SSE payload", err); return; }
      ingest(ev);
      if (t === "run_end" || (t === "error" && !ev.payload?.recoverable)) stopStreams();
    });
  }
  es.addEventListener("error", () => {
    // EventSource auto-reconnects after the server closes the stream; a finished run must not restart.
    if (store.status === "done" || store.status === "error") return stopStreams();
    if (es && es.readyState === EventSource.CLOSED) {
      store.status = "error";
      bus.emit("run:error", { payload: { message: sawAny ? "Lost connection to the run stream." : "Could not open the run event stream.", recoverable: false } });
    }
  });
}

/** Load a recorded event list (studio/mock/mock_run.json) and feed it through the same ingest path. */
export async function playRecorded(urls, { spacingMs = 120 } = {}) {
  stopStreams();
  let events = null, lastErr = null;
  for (const u of urls) {
    try {
      const r = await fetch(u);
      if (!r.ok) throw new Error(`${u}: HTTP ${r.status}`);
      events = await r.json();
      break;
    } catch (err) { lastErr = err; }
  }
  if (!events) throw new Error(`Could not load the mock run (${lastErr?.message || "no source"}). Serve the repo root, e.g. python -m http.server, and open /studio/web/index.html?mock=1`);
  let i = 0;
  const step = () => {
    if (i >= events.length) return;
    ingest(events[i++]);
    mockTimer = setTimeout(step, spacingMs);
  };
  step();
}
