// Single in-memory store for the current run. Panels read from it; only run.js / main.js write to it.
export const store = {
  runId: null,
  source: null, // "live" | "replay" | "mock" (as reported by run_start) or "preview" before a run
  status: "idle", // idle | validating | ready | running | done | error
  config: null,
  seed: null, // {smiles, molblock, score}
  candidates: [], // every candidate event payload, arrival order
  generations: [],
  policy: [],
  end: null,
  bySmiles: new Map(), // smiles -> {score, contributions, alerts}; lets an edit find its parent's atom data
  locked: new Set(), // seed atom indices the user locked (sent as RunConfig.locked_atoms)
  overlay: { mode: "element", alerts: true, highlightEdit: true, autorotate: true },
  reset() {
    this.runId = null; this.source = null; this.config = null; this.seed = null;
    this.candidates = []; this.generations = []; this.policy = []; this.end = null;
    this.bySmiles = new Map();
  },
  remember(smiles, info) {
    this.bySmiles.set(smiles, { ...(this.bySmiles.get(smiles) || {}), ...info });
  },
};
