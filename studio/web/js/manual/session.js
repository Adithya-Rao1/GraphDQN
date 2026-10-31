// Manual-mode session model (no DOM, no network): steps, undo / redo, lock tracking through atom maps, and the synthesis of the
// run events the shared viewer already understands (run_start / candidate / generation / run_end), so the 3D stage, why-card,
// timeline, lineage, charts and story panels all work for a hand-made path exactly as for a GA run.
// Transitions use chosen_by = "user" and are exportable as JSONL (also client-side, as a fallback for the server export).

const KEYS = ["qed", "logp", "tpsa", "mw", "hbd", "hba", "rotb", "fsp3", "alerts"];

export class ManualSession {
  constructor({ seedSmiles, seedMolblock, seedScore, targetId = null, locked = [], label = "seed" }) {
    this.runId = `manual-${Date.now().toString(36)}`;
    this.sessionId = null; // server session id, when the server accepted the session
    this.seed = { smiles: seedSmiles, molblock: seedMolblock, score: seedScore, label };
    this.targetId = targetId;
    this.locked = new Set(locked); // atom indices of the CURRENT molecule
    this.seedLocked = [...locked]; // the locks the run_start event carries (seed atom indices)
    this.steps = []; // applied steps
    this.redoStack = [];
    this.t0 = Date.now();
    this.logged = 0; // steps acknowledged by the server
    this.logFailed = 0;
  }

  get current() {
    const s = this.steps[this.steps.length - 1];
    return s ? { smiles: s.child_smiles, molblock: s.molblock, score: s.score } : this.seed;
  }

  /** step = {edit_id, name, category, site, rationale, preview: ManualPreview} */
  apply({ edit_id, name, category, site, preview }) {
    const parent = this.current;
    const step = {
      n: this.steps.length + 1,
      id: `${this.runId}-s${this.steps.length + 1}`,
      parent_smiles: parent.smiles, child_smiles: preview.child_smiles, molblock: preview.molblock, parent_molblock: parent.molblock,
      atom_map: preview.atom_map, score: preview.score, explanation: preview.explanation || null,
      edit_id, name, category, site: [...site],
      reward_delta: preview.score.composite - parent.score.composite,
      t: (Date.now() - this.t0) / 1000,
      lockedBefore: new Set(this.locked),
    };
    // carry locks through the edit: a locked atom keeps its identity via the (parent -> child) atom map
    const m = new Map(preview.atom_map);
    this.locked = new Set([...this.locked].filter((i) => m.has(i)).map((i) => m.get(i)));
    this.steps.push(step);
    this.redoStack = [];
    return step;
  }

  undo() {
    const s = this.steps.pop();
    if (!s) return null;
    this.locked = new Set(s.lockedBefore);
    this.redoStack.push(s);
    return s;
  }

  redo() {
    const s = this.redoStack.pop();
    if (!s) return null;
    const m = new Map(s.atom_map);
    this.locked = new Set([...this.locked].filter((i) => m.has(i)).map((i) => m.get(i)));
    this.steps.push(s);
    return s;
  }

  toggleLock(i) {
    if (this.locked.has(i)) this.locked.delete(i); else this.locked.add(i);
    return this.locked.has(i);
  }

  // -------------------------------------------------------------------- run events for the shared viewer
  ev(type, payload, seq) { return { type, run_id: this.runId, seq, payload }; }

  runStart() {
    return this.ev("run_start", {
      config: { seed_smiles: this.seed.smiles, objectives: {}, population_size: 1, generations: 0, rng_seed: 0, policy_guidance: 0, use_crossover: false,
        use_reactions: false, locked_atoms: [...this.seedLocked], stage: "manual", target_id: this.targetId, preset_id: null },
      seed_molblock: this.seed.molblock, seed_score: this.seed.score, source: "manual", schema_version: "0.1",
    }, 0);
  }

  candidate(step) {
    return this.ev("candidate", {
      id: step.id, generation: step.n, smiles: step.child_smiles, parent_smiles: step.parent_smiles, molblock: step.molblock, parent_molblock: step.parent_molblock,
      atom_map: step.atom_map,
      edit: { edit_id: step.edit_id, name: step.name, category: step.category, site: step.site, chosen_by: "user", parents: [step.parent_smiles] },
      score: step.score, explanation: step.explanation, accepted: true,
    }, step.n * 2 - 1);
  }

  generation(step) {
    const m = { smiles: step.child_smiles, composite: step.score.composite, candidate_id: step.id };
    return this.ev("generation", { generation: step.n, best: m, mean_composite: step.score.composite, population: [m], map_xy: null, map_method: null }, step.n * 2);
  }

  /** A StoryCard built from real values of the session (nothing invented). */
  runEnd() {
    const s0 = this.seed.score, last = this.steps[this.steps.length - 1];
    const s1 = last ? last.score : s0;
    const table = {};
    for (const k of KEYS) if (k in s0.descriptors) table[k] = [s0.descriptors[k], s1.descriptors[k]];
    const removed = (s0.alerts || []).filter((a) => !(s1.alerts || []).some((b) => b.name === a.name)).length;
    const n = this.steps.length;
    const headline = n
      ? `Manual session: composite ${s0.composite.toFixed(2)} → ${s1.composite.toFixed(2)}, QED ${s0.descriptors.qed.toFixed(2)} → ${s1.descriptors.qed.toFixed(2)} in ${n} hand-picked edit${n === 1 ? "" : "s"}${removed ? `, ${removed} alert${removed === 1 ? "" : "s"} removed` : ""}`
      : "Manual session: no edits applied";
    return this.ev("run_end", {
      seed_smiles: this.seed.smiles, final_smiles: this.current.smiles, seed_score: s0, final_score: s1,
      edit_path: this.steps.map((s) => s.id), headline, descriptor_table: table,
    }, this.steps.length * 2 + 1);
  }

  // ------------------------------------------------------------------------- training data
  /** ManualStep body for POST /api/manual/sessions/{id}/steps. */
  stepBody(step) {
    return { parent_smiles: step.parent_smiles, child_smiles: step.child_smiles, edit_id: step.edit_id, site: step.site, reward_delta: step.reward_delta, t: step.t };
  }

  /** Client-side JSONL (fallback when the server export is unreachable): one transition per applied step. */
  jsonl() {
    return this.steps.map((s) => JSON.stringify({
      parent_smiles: s.parent_smiles, edit_id: s.edit_id, site: s.site, child_smiles: s.child_smiles, reward: s.reward_delta,
      chosen_by: "user", target_id: this.targetId, session_id: this.sessionId || this.runId, t: +s.t.toFixed(2),
    })).join("\n") + (this.steps.length ? "\n" : "");
  }
}

/** Candidate-edit list helpers (pure). */
export function normalizeApplicable(body) {
  const list = Array.isArray(body) ? body : body?.edits || body?.items || body?.applicable || [];
  return list.map((e) => ({ edit_id: e.edit_id || e.id, name: e.name || e.edit_id || e.id, category: e.category || "", site: e.site || [], rationale: e.rationale || "", tags: e.tags || [] }));
}

export const previewKey = (smiles, e, targetId) => `${smiles}|${e.edit_id}|${e.site.join(",")}|${targetId || ""}`;
