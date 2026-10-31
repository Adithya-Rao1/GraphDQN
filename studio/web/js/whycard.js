// "Why" card: the edit, its rationale, descriptor deltas (with arrows) and structural alerts removed / added.
import { store } from "./state.js";

const ORDER = ["qed", "logp", "tpsa", "mw", "hbd", "hba", "rotb", "fsp3", "sa", "lipinski_violations", "alerts", "heavy_atoms"];
const LABEL = { qed: "QED", logp: "logP", tpsa: "TPSA", mw: "Mol. weight", hbd: "H-bond donors", hba: "H-bond acceptors", rotb: "Rotatable bonds", fsp3: "Fsp3", sa: "SA score", lipinski_violations: "Lipinski violations", alerts: "Structural alerts", heavy_atoms: "Heavy atoms" };
const DIGITS = { qed: 2, logp: 2, tpsa: 1, mw: 1, fsp3: 2, sa: 2 };
const ALWAYS = new Set(["qed", "logp", "tpsa", "mw", "alerts"]);
// without a user objective, only these have an unambiguous direction
const DEFAULT_DIR = { qed: "maximize", alerts: "minimize", lipinski_violations: "minimize", sa: "minimize" };
const BY = { policy: "DQN policy", random: "random edit", crossover: "crossover", reaction: "reaction", user: "manual" };

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (k, v) => (v == null ? "n/a" : Number(v).toFixed(DIGITS[k] ?? 0));

/** "good" | "bad" | "neutral" for a move of descriptor k from a to b, given the run's objectives. */
export function judge(k, a, b) {
  const spec = store.config?.objectives?.[k];
  const dir = spec?.direction || DEFAULT_DIR[k];
  if (!dir || a === b) return "neutral";
  if (dir === "maximize") return b > a ? "good" : "bad";
  if (dir === "minimize") return b < a ? "good" : "bad";
  const lo = spec?.low ?? -Infinity, hi = spec?.high ?? Infinity;
  const dist = (x) => (x < lo ? lo - x : x > hi ? x - hi : 0);
  return dist(b) < dist(a) ? "good" : dist(b) > dist(a) ? "bad" : "neutral";
}

function deltaCell(k, d, parentVal, childVal) {
  if (d == null || Math.abs(d) < 1e-9) return `<span class="d neutral">–</span>`;
  const cls = judge(k, parentVal, childVal);
  const arrow = d > 0 ? "▲" : "▼";
  const sign = d > 0 ? "+" : "−";
  return `<span class="d ${cls}">${arrow} ${sign}${Math.abs(d).toFixed(DIGITS[k] ?? 0)}</span>`;
}

function rows(desc, deltas) {
  const keys = ORDER.filter((k) => k in desc && (ALWAYS.has(k) || Math.abs(deltas?.[k] ?? 0) > 1e-9));
  return keys.map((k) => {
    const child = desc[k];
    const d = deltas?.[k];
    const parent = d == null || child == null ? child : child - d;
    return `<tr><td class="k">${LABEL[k] || k}</td><td class="v">${deltas ? `${fmt(k, parent)} → ` : ""}<b>${fmt(k, child)}</b></td>${deltas ? `<td class="d">${deltaCell(k, d, parent, child)}</td>` : ""}</tr>`;
  }).join("");
}

function alertChips(removed, added, remain) {
  const chips = [];
  for (const a of removed || []) chips.push(`<span class="chip removed" title="${esc(a.family)}">${esc(a.name)}</span>`);
  for (const a of added || []) chips.push(`<span class="chip added" title="${esc(a.family)}">+ ${esc(a.name)}</span>`);
  for (const a of remain || []) chips.push(`<span class="chip remain" title="${esc(a.family)}">${esc(a.name)}</span>`);
  if (!chips.length) return `<div class="alerts"><span class="lbl">Structural alerts</span><span class="chip" style="border-color:var(--line);color:var(--muted)">none</span></div>`;
  const lbl = [removed?.length ? `${removed.length} removed` : "", added?.length ? `${added.length} added` : ""].filter(Boolean).join(", ");
  return `<div class="alerts"><span class="lbl">Structural alerts${lbl ? ": " + lbl : ""}</span>${chips.join("")}</div>`;
}

/** HTML for a keyframe. */
export function renderWhy(kf, index, total) {
  if (!kf) return `<p class="muted">Edits will be explained here as the run plays.</p>`;
  if (kf.kind === "seed") {
    const s = store.seed?.score;
    return `
      <div class="why-head"><span class="pill">Seed molecule</span></div>
      <div class="edit-name">Starting point</div>
      <p class="rationale muted">${esc(store.seed?.smiles || "")}</p>
      ${s ? `<div class="score-line"><span class="big">${s.composite.toFixed(2)}</span><span class="muted">composite score</span></div><div class="bar"><i style="width:${Math.round(s.composite * 100)}%"></i></div>
      <table class="deltas">${rows(s.descriptors, null)}</table>${alertChips(null, null, s.alerts)}` : ""}`;
  }
  const c = kf.candidate, e = c.explanation, edit = c.edit;
  const desc = c.score.descriptors;
  const parentComposite = store.bySmiles.get(c.parent_smiles)?.composite;
  const dComposite = parentComposite == null ? null : c.score.composite - parentComposite;
  const cDelta = dComposite == null ? "" : `<span class="d ${dComposite > 0 ? "good" : dComposite < 0 ? "bad" : "neutral"}">${dComposite >= 0 ? "▲ +" : "▼ −"}${Math.abs(dComposite).toFixed(2)}</span>`;
  const remain = (store.bySmiles.get(c.parent_smiles)?.alerts || []).filter((a) => !(e?.alerts_removed || []).some((r) => r.name === a.name));
  return `
    <div class="why-head">
      <span class="pill">Edit ${index} / ${total}</span>
      <span class="pill ${edit?.chosen_by === "policy" ? "policy" : ""}">${esc(BY[edit?.chosen_by] || edit?.chosen_by || "edit")}</span>
      <span class="pill ${c.accepted ? "accepted" : "rejected"}">${c.accepted ? "accepted" : "not selected"}</span>
    </div>
    <div class="edit-name">${esc(edit?.name || "Edit")}</div>
    <div class="category">${esc(edit?.category || "")}${edit?.edit_id ? ` · <code>${esc(edit.edit_id)}</code>` : ""}${c.generation != null ? ` · generation ${c.generation}` : ""}</div>
    <p class="rationale">${esc(e?.rationale || "No rationale supplied.")}</p>
    ${e?.tags?.length ? `<div class="tags">${e.tags.map((t) => `<span class="tag">${esc(t.replace(/_/g, " "))}</span>`).join("")}</div>` : ""}
    <div class="reveal">
      <div class="score-line"><span class="big">${c.score.composite.toFixed(2)}</span><span class="muted">composite</span>${cDelta}</div>
      <div class="bar"><i style="width:${Math.round(c.score.composite * 100)}%"></i></div>
    </div>
    <div class="reveal"><table class="deltas">${rows(desc, e?.descriptor_deltas)}</table></div>
    <div class="reveal">${alertChips(e?.alerts_removed, e?.alerts_added, remain)}</div>`;
}
