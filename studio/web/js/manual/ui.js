// HTML builders for manual mode (strings only; index.js owns the DOM and the events).
import { esc, LABEL, DIGITS, fmt, fmtSigned, judge, bindingOf, scoreColor } from "../panels/util.js";

const SHOW = ["qed", "logp", "tpsa", "mw", "hbd", "hba", "rotb", "fsp3", "sa", "lipinski_violations", "alerts", "heavy_atoms"];
const ALWAYS = new Set(["qed", "logp", "tpsa", "mw", "alerts"]);
const CAT = { atom: "Atom swaps", bond: "Bond changes", bioisostere: "Bioisosteres", functional_group: "Functional groups", crossover: "Crossover", reaction: "Reactions" };
const arrow = (d) => (Math.abs(d) < 1e-9 ? "–" : d > 0 ? "▲" : "▼");
const tagText = (t) => esc(String(t).replace(/_/g, " "));

export function dockHtml(m) {
  const sel = m.selected;
  const head = `<header class="md-head"><span class="md-badge">Manual</span><b>Edit by hand</b><span class="md-spacer"></span>
      <button class="md-icon" data-act="collapse" title="Collapse / expand" aria-label="Collapse">${m.collapsed ? "▸" : "▾"}</button></header>`;
  if (m.collapsed) return head;
  if (!m.session) {
    return head + `<div class="md-body"><div class="md-start">
      <p class="md-lead">${m.seed ? "Pick atoms, preview every edit's effect, and apply the ones you like. Each step is saved as a training transition." : "Load a molecule first (SMILES or MOL file), then come back."}</p>
      ${m.seed ? `<code class="md-smi">${esc(m.seed.smiles)}</code>
      <button class="md-btn primary" data-act="start-seed">Start from this molecule</button>
      ${m.lead ? `<button class="md-btn" data-act="start-lead">Start from last run's final lead</button>` : ""}` : ""}
      ${m.error ? `<div class="md-err">${esc(m.error)}</div>` : ""}</div></div>`;
  }
  const s = m.session;
  const tools = `<div class="md-tools">
      <button class="md-btn" data-act="undo" ${s.steps.length ? "" : "disabled"} title="Undo (Ctrl/⌘+Z)">↶ Undo</button>
      <button class="md-btn" data-act="redo" ${s.redoStack.length ? "" : "disabled"} title="Redo (Ctrl/⌘+Shift+Z)">↷ Redo</button>
      <button class="md-btn ${sel.length && sel.every((i) => s.locked.has(i)) ? "on" : ""}" data-act="lock" ${sel.length ? "" : "disabled"} title="Lock the selected atom(s): locked atoms are never edited (L, or Shift-click an atom)">🔒 Lock</button>
      <button class="md-btn" data-act="export" ${s.steps.length ? "" : "disabled"} title="Download the session as training data (JSONL)">⇩ Export</button></div>`;
  let selLine;
  if (!sel.length) selLine = `<div class="md-sel hint">Click an atom in the 3D view to choose a site.<br><span>Ctrl/⌘-click a second atom to narrow to a bond · Shift-click to lock.</span></div>`;
  else selLine = `<div class="md-sel"><b>Site:</b> atom${sel.length > 1 ? "s" : ""} ${sel.map((i) => `<span class="md-atom">${i}</span>`).join("")}<button class="md-icon" data-act="clear-sel" title="Clear selection (Esc)">✕</button></div>`;
  let list;
  if (m.loadingList) list = `<div class="md-skel"><i></i><i></i><i></i></div>`;
  else if (m.listError) list = `<div class="md-err">${esc(m.listError)}</div>`;
  else if (!sel.length) {
    const n = m.edits.length;
    list = `<div class="md-empty">${n ? `${n} edit${n === 1 ? "" : "s"} possible on this molecule. Atoms with a blue glow have at least one.` : "No edits are available for this molecule (or all touch locked atoms)."}</div>`;
  } else if (!m.visible.length) list = `<div class="md-empty">No edit applies at ${sel.length > 1 ? "this combination of atoms" : "this atom"}. Try a neighbouring atom.</div>`;
  else {
    const groups = new Map();
    m.visible.forEach((e, i) => { const k = e.category || "other"; if (!groups.has(k)) groups.set(k, []); groups.get(k).push([e, i]); });
    list = [...groups].map(([cat, items]) => `<div class="md-grp"><div class="md-grp-h">${esc(CAT[cat] || cat)} <span>${items.length}</span></div>${items.map(([e, i]) => {
      const pv = m.previews.get(e._key);
      let badge = "";
      if (pv && pv.ok && pv.score) {
        const d = pv.score.composite - s.current.score.composite, cls = !pv.sane ? "bad" : d > 0.0005 ? "good" : d < -0.0005 ? "bad" : "neutral";
        badge = `<span class="md-delta ${cls}">${pv.sane ? `${arrow(d)} ${Math.abs(d).toFixed(2)}` : "⚠"}</span>`;
      } else if (pv && !pv.ok) badge = `<span class="md-delta bad" title="${esc(pv.reason || "")}">✕</span>`;
      return `<button class="md-edit ${m.hoverKey === e._key ? "hot" : ""}" data-i="${i}" role="option"><span class="md-edit-n">${esc(e.name)}</span>${badge}</button>`;
    }).join("")}</div>`).join("");
  }
  const foot = `<footer class="md-foot">${s.steps.length} edit${s.steps.length === 1 ? "" : "s"}${s.locked.size ? ` · ${s.locked.size} locked` : ""} · <span class="${s.sessionId ? (s.logFailed ? "bad" : "ok") : "warn"}" title="${esc(m.logNote || "")}">${m.logLabel}</span></footer>`;
  return head + `<div class="md-body">${tools}${selLine}<div class="md-list" role="listbox" aria-label="Applicable edits">${list}</div></div>` + foot;
}

function descRows(parent, child, deltas) {
  const keys = SHOW.filter((k) => k in child && (ALWAYS.has(k) || Math.abs(deltas?.[k] ?? (child[k] - (parent?.[k] ?? child[k]))) > 1e-9));
  return keys.map((k) => {
    const a = parent?.[k], b = child[k], d = deltas?.[k] ?? (a == null ? 0 : b - a);
    const cls = judge(k, a ?? b, b);
    return `<tr><td class="k">${LABEL[k] || k}</td><td class="v">${fmt(k, a)} → <b>${fmt(k, b)}</b></td><td class="d ${Math.abs(d) < 1e-9 ? "neutral" : cls}">${Math.abs(d) < 1e-9 ? "–" : `${arrow(d)} ${fmtSigned(k, d)}`}</td></tr>`;
  }).join("");
}

/** What-if card for a hovered edit. state: {edit, status: "loading"|"ok"|"fail", preview, ghost: counts, parent: ScoreResult} */
export function whatIfHtml({ edit, status, preview, ghost, parent, reason }) {
  if (!edit) return "";
  const head = `<div class="mw-head"><span class="md-badge sm">What if</span><span class="mw-cat">${esc(edit.category || "")}${edit.site?.length ? ` · site ${edit.site.join(", ")}` : ""}</span></div><div class="mw-name">${esc(edit.name)}</div>`;
  if (status === "loading") return head + `<div class="md-skel"><i></i><i></i></div>`;
  if (status === "fail" || !preview?.ok) return head + `<div class="md-err">${esc(reason || preview?.reason || "This edit could not be previewed.")}</div><p class="mw-why"><b>Why this edit?</b> ${esc(edit.rationale || "No rationale supplied.")}</p>`;
  const c = preview.score, e = preview.explanation, d = c.composite - parent.composite;
  const dcls = d > 0.0005 ? "good" : d < -0.0005 ? "bad" : "neutral";
  const bp = bindingOf(parent), bc = bindingOf(c);
  const bind = bp && bc ? (() => { const db = bc.value - bp.value, cl = db > 0 ? "good" : db < 0 ? "bad" : "neutral";
    return `<div class="mw-bind"><span>Binding (PLAPT) <i class="tag-pred">predicted</i></span><span>${bp.value.toFixed(2)} → <b>${bc.value.toFixed(2)}</b> <em class="${cl}">${arrow(db)} ${fmtSigned("bind", db)}</em></span></div>`; })() : "";
  const verdict = preview.sane
    ? `<div class="mw-verdict ok">✓ Chemistry sanity check passed</div>`
    : `<div class="mw-verdict bad">✕ Rejected by the sanity check: ${esc(preview.reason || "no reason given")}. This edit cannot be applied.</div>`;
  const al = [...(e?.alerts_removed || []).map((a) => `<span class="chip removed">${esc(a.name)}</span>`), ...(e?.alerts_added || []).map((a) => `<span class="chip added">+ ${esc(a.name)}</span>`)].join("");
  return head + verdict + `
    <div class="mw-score"><span class="mw-big" style="color:${scoreColor(c.composite)}">${c.composite.toFixed(2)}</span><span class="mw-sub">composite <small>(was ${parent.composite.toFixed(2)})</small></span>
      <span class="mw-d ${dcls}">${arrow(d)} ${Math.abs(d).toFixed(2)}</span></div>
    <table class="deltas mw-t">${descRows(parent.descriptors, c.descriptors, e?.descriptor_deltas)}</table>
    ${al ? `<div class="alerts mw-al">${al}</div>` : ""}${bind}
    ${ghost ? `<div class="mw-ghost muted">Ghost in 3D: <span class="g">+${ghost.added} atom${ghost.added === 1 ? "" : "s"}</span> <span class="r">−${ghost.removed}</span>${ghost.changed ? ` <span class="w">${ghost.changed} changed</span>` : ""}</div>` : ""}
    <p class="mw-why"><b>Why this edit?</b> ${esc(e?.rationale || edit.rationale || "No rationale supplied.")}</p>
    ${(e?.tags || edit.tags || []).length ? `<div class="tags">${(e?.tags || edit.tags).map((t) => `<span class="tag">${tagText(t)}</span>`).join("")}</div>` : ""}
    <div class="mw-hint">${preview.sane ? "Click the edit to apply it" : ""}</div>`;
}

/** Bottom panel: step strip + training log. */
export function stepsHtml(session, { sessionLabel, current }) {
  const steps = session.steps;
  const chips = [`<button class="ms-chip ${current === 0 ? "cur" : ""}" data-frame="0"><b>0</b><span>${esc(session.seed.label)}</span><em>${session.seed.score.composite.toFixed(2)}</em></button>`]
    .concat(steps.map((s, i) => `<button class="ms-chip ${current === i + 1 ? "cur" : ""}" data-frame="${i + 1}"><b>${s.n}</b><span>${esc(s.name)}</span><em class="${s.reward_delta > 0 ? "good" : s.reward_delta < 0 ? "bad" : ""}">${s.reward_delta >= 0 ? "▲" : "▼"} ${Math.abs(s.reward_delta).toFixed(2)}</em></button>`));
  const rows = steps.map((s) => `<tr><td>${s.n}</td><td class="mono">${esc(s.parent_smiles)} <span class="muted">→</span> ${esc(s.child_smiles)}</td><td><code>${esc(s.edit_id)}</code></td><td>${s.site.join(", ") || "–"}</td>
      <td class="${s.reward_delta > 0 ? "good" : s.reward_delta < 0 ? "bad" : ""}">${s.reward_delta >= 0 ? "+" : "−"}${Math.abs(s.reward_delta).toFixed(3)}</td><td><span class="pill">user</span></td></tr>`).join("");
  return `<section class="pcard msteps"><header class="pc-head"><h3>Manual session</h3><span class="pc-sub">${esc(sessionLabel)}</span>
    <button class="md-btn" data-act="replay" ${steps.length ? "" : "disabled"}>▶ Replay</button>
    <button class="md-btn primary" data-act="finish" ${steps.length ? "" : "disabled"} title="Show the before / after story card for this session">Finish &amp; show story</button></header>
    <div class="ms-strip">${chips.join('<i class="ms-link">›</i>')}</div>
    ${steps.length ? `<div class="ms-log"><div class="ms-log-h">Training log <span class="muted">one transition per step · chosen_by = user · reward = composite change</span></div>
      <table class="ms-table"><thead><tr><th>#</th><th>parent → child</th><th>edit</th><th>site</th><th>reward Δ</th><th>by</th></tr></thead><tbody>${rows}</tbody></table></div>` : `<div class="muted ms-empty">No edits yet. Apply an edit from the list in the 3D view.</div>`}
    <div class="pc-foot muted">Undone steps remain in the export: they were real transitions the environment scored. Rewards here are descriptor-based (RDKit), not measured activity.</div></section>`;
}
