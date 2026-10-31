// Manual CADD mode. See README.md for the contract. A manual session is replayed into the shared viewer as a pseudo run
// (run_start -> candidate / generation per applied edit -> run_end on "Finish"), with source "manual" and chosen_by "user", so the
// 3D stage, morph animation, why card, timeline, lineage, charts and story card all work for a hand-made path unchanged.
import { makeApi } from "./api.js";
import { ManualSession, normalizeApplicable, previewKey } from "./session.js";
import { ManualOverlay } from "./overlay.js";
import { Ghost } from "./ghost.js";
import { dockHtml, whatIfHtml, stepsHtml } from "./ui.js";
import { esc, toast, isMobile } from "../panels/util.js";

export function init(ctx = window.metis) {
  const { bus, store, timeline, stage } = ctx;
  const ingest = ctx.ingest || window.metis?.ingest;
  const api = makeApi({ fetchFn: ctx.fetch, base: ctx.apiBase || "" });
  const wrap = document.getElementById("stage-wrap");
  const mount = ctx.mounts?.manual || document.getElementById("panel-manual");
  if (!wrap || !stage) { console.warn("manual: no stage; manual mode disabled"); return null; }

  const m = { active: false, session: null, seed: null, lead: null, selected: [], edits: [], visible: [], previews: new Map(), hoverKey: null,
    loadingList: false, listError: null, collapsed: false, error: null, logLabel: "", logNote: "" };
  const listCache = new Map(); // `${smiles}|${locked}` -> normalized list
  const inflight = new Map();
  let overlay = null, ghost = null, dock = null, main = null, what = null, stepsEl = null, token = 0, hoverTimer = 0, busy = false, unsubs = [], moved = false;

  // ------------------------------------------------------------------ DOM
  function build() {
    if (dock) return;
    dock = document.createElement("aside");
    dock.className = "mdock";
    dock.setAttribute("aria-label", "Manual edit panel");
    main = document.createElement("div");
    main.className = "md-main";
    what = document.createElement("div");
    what.className = "mwhat";
    what.hidden = true;
    what.setAttribute("aria-live", "polite");
    dock.append(main, what);
    wrap.append(dock);
    stepsEl = document.createElement("div");
    stepsEl.className = "msteps-host";
    overlay = new ManualOverlay({ stage, wrap });
    ghost = new Ghost(stage);
    dock.addEventListener("click", onDockClick);
    dock.addEventListener("pointerover", (e) => { const b = e.target.closest(".md-edit"); if (b) hoverEdit(+b.dataset.i); });
    dock.addEventListener("pointerout", (e) => { if (e.target.closest(".md-edit") && !e.relatedTarget?.closest?.(".md-edit")) leaveEdit(); });
    dock.addEventListener("focusin", (e) => { const b = e.target.closest(".md-edit"); if (b) hoverEdit(+b.dataset.i, 0); });
    dock.addEventListener("focusout", (e) => { if (e.target.closest(".md-edit")) leaveEdit(); });
    stepsEl.addEventListener("click", onStepsClick);
    wrap.addEventListener("pointerdown", onDown, true);
    wrap.addEventListener("pointerup", onUp, true);
    wrap.addEventListener("pointermove", onMove, true);
    window.addEventListener("keydown", onKey);
    window.matchMedia("(max-width: 1000px)").addEventListener("change", place);
    place();
  }

  function place() {
    // wide screens: floating panels over the stage; narrow: below it, inside the manual mount
    const host = isMobile() && mount ? mount : wrap;
    dock.classList.toggle("static", host !== wrap);
    document.body.classList.toggle("manual-docked", host === wrap);
    if (host === wrap) wrap.append(dock); else mount.prepend(dock);
    if (mount && !stepsEl.isConnected) mount.append(stepsEl);
  }

  function teardown() {
    window.removeEventListener("keydown", onKey);
    document.body.classList.remove("manual-docked");
    dock?.remove(); stepsEl?.remove(); overlay?.stop(); overlay?.canvas.remove(); ghost?.clear();
    dock = main = what = stepsEl = overlay = ghost = null;
  }

  const cur = () => m.session?.current;
  const targetId = () => ctx.getTargetId?.() || store.config?.target_id || null;
  const latestIndex = () => timeline.keyframes.length - 1;
  const onLatest = () => { const s = timeline.state(); return s.index === latestIndex() && s.index === (m.session?.steps.length ?? 0); };

  function logLabel() {
    const s = m.session;
    if (!s) return "";
    if (!s.sessionId) return "not logged to server";
    return s.logFailed ? `logged ${s.logged}/${s.steps.length + s.redoStack.length} (some failed)` : `logged ${s.logged} to server`;
  }

  function renderDock() {
    if (!dock) return;
    m.visible = visibleEdits();
    m.logLabel = logLabel();
    m.logNote = m.session?.sessionId ? `session ${m.session.sessionId}` : "The server session could not be created; export falls back to a client-side file.";
    m.seed = store.seed && !m.session ? store.seed : m.seed;
    m.lead = !m.session && store.end && store.source !== "manual" ? store.candidates.find((c) => c.id === store.end.edit_path.at(-1)) || null : null;
    const sc = main.querySelector(".md-list")?.scrollTop;
    main.innerHTML = dockHtml(m);
    const l = main.querySelector(".md-list");
    if (l && sc) l.scrollTop = sc;
    overlay.selected = m.selected;
    overlay.locked = m.session?.locked || new Set();
    overlay.showLocks = m.session ? !(m.session.steps.length === 0 && timeline.state().keyframe?.kind === "seed") : true;
    overlay.density = density();
    renderSteps();
  }

  function renderSteps() {
    if (!stepsEl) return;
    if (!m.session) { stepsEl.innerHTML = ""; return; }
    stepsEl.innerHTML = stepsHtml(m.session, { sessionLabel: `${m.session.seed.label} · ${targetId() ? "target " + targetId() : "no target"} · ${m.session.sessionId ? "session " + m.session.sessionId : "local session"}`, current: timeline.state().index });
  }

  // ------------------------------------------------------------ applicable edits
  function density() {
    const d = new Map();
    for (const e of m.edits) for (const i of e.site) d.set(i, (d.get(i) || 0) + 1);
    return d;
  }
  function visibleEdits() {
    if (!m.selected.length) return [];
    return m.edits.filter((e) => m.selected.every((i) => e.site.includes(i)));
  }
  async function loadEdits() {
    const s = m.session;
    if (!s) return;
    const smiles = s.current.smiles, locked = [...s.locked].sort((a, b) => a - b);
    const key = `${smiles}|${locked}`;
    if (listCache.has(key)) { m.edits = listCache.get(key); m.listError = null; return renderDock(); }
    m.loadingList = true; m.edits = []; renderDock();
    try {
      const list = normalizeApplicable(await api.applicable(smiles, locked)).map((e) => ({ ...e, _key: previewKey(smiles, e, targetId()) }));
      listCache.set(key, list);
      if (s.current.smiles === smiles) { m.edits = list; m.listError = null; }
    } catch (err) {
      m.listError = `Could not load the applicable edits (${err.message}).`;
    }
    m.loadingList = false;
    renderDock();
  }

  // ---------------------------------------------------------------- previews
  function getPreview(e) {
    const smiles = cur().smiles, key = e._key || previewKey(smiles, e, targetId());
    if (m.previews.has(key)) return Promise.resolve(m.previews.get(key));
    if (inflight.has(key)) return inflight.get(key);
    const p = api.preview({ smiles, edit_id: e.edit_id, site: e.site, target_id: targetId() })
      .then((pv) => { m.previews.set(key, pv); return pv; })
      .catch((err) => ({ ok: false, reason: `Preview failed: ${err.message}`, sane: false }))
      .finally(() => inflight.delete(key));
    inflight.set(key, p);
    return p;
  }

  function hoverEdit(i, delay = 110) {
    const e = m.visible[i];
    if (!e) return;
    clearTimeout(hoverTimer);
    const my = ++token;
    m.hoverKey = e._key;
    dock.querySelectorAll(".md-edit.hot").forEach((b) => b.classList.remove("hot"));
    dock.querySelector(`.md-edit[data-i="${i}"]`)?.classList.add("hot");
    hoverTimer = setTimeout(async () => {
      showWhat({ edit: e, status: "loading", parent: cur().score });
      const pv = await getPreview(e);
      if (my !== token) return;
      let g = null;
      if (pv.ok && onLatest()) {
        const diff = ghost.show(pv);
        overlay.ghost = diff;
        g = diff ? diff.counts : null;
      } else if (pv.ok) toast("Viewing an earlier step. Jump to the latest step to preview 3D ghosts.");
      showWhat({ edit: e, status: pv.ok ? "ok" : "fail", preview: pv, ghost: g, parent: cur().score });
      refreshBadge(e, pv);
    }, delay);
  }

  function refreshBadge(e, pv) {
    const b = [...dock.querySelectorAll(".md-edit")].find((x) => m.visible[+x.dataset.i]?._key === e._key);
    if (!b || b.querySelector(".md-delta")) return;
    const d = pv.ok ? pv.score.composite - cur().score.composite : 0;
    const cls = !pv.ok || !pv.sane ? "bad" : d > 0.0005 ? "good" : d < -0.0005 ? "bad" : "neutral";
    b.insertAdjacentHTML("beforeend", `<span class="md-delta ${cls}" title="${esc(pv.reason || "")}">${!pv.ok ? "✕" : !pv.sane ? "⚠" : (d >= 0 ? "▲ " : "▼ ") + Math.abs(d).toFixed(2)}</span>`);
  }

  function leaveEdit() {
    clearTimeout(hoverTimer);
    token++;
    m.hoverKey = null;
    dock?.querySelectorAll(".md-edit.hot").forEach((b) => b.classList.remove("hot"));
    ghost?.clear();
    if (overlay) overlay.ghost = null;
    if (what) what.hidden = true;
  }

  function showWhat(state) {
    what.innerHTML = whatIfHtml(state);
    what.hidden = false;
  }

  // ------------------------------------------------------------------- apply
  async function applyEdit(i) {
    const e = m.visible[i];
    if (!e || busy || !m.session) return;
    busy = true;
    try {
      const pv = await getPreview(e);
      if (!pv.ok) return toast(pv.reason || "That edit could not be applied.");
      if (!pv.sane) return toast(`Rejected by the sanity check: ${pv.reason || "no reason given"}`);
      leaveEdit();
      ensureLive();
      const step = m.session.apply({ edit_id: e.edit_id, name: e.name, category: e.category, site: e.site, preview: pv });
      pushStep(step, true);
      afterStepChange();
      logStep(step);
    } finally { busy = false; }
  }

  /** Feed one step to the shared viewer and play its morph. */
  function pushStep(step, play) {
    ingest(m.session.candidate(step));
    ingest(m.session.generation(step));
    timeline.live = false;
    bus.emit("timeline:keyframes", { timeline });
    const i = timeline.keyframes.findIndex((k) => k.id === step.id);
    if (i >= 0) { timeline.seekFrame(i); if (play) timeline.play(); }
  }

  /** The pseudo run must be the one in the store (the user may have run something else, or finished the story). */
  function ensureLive() {
    if (store.runId === m.session.runId && store.status === "running") return;
    replay({ upto: m.session.steps.length });
  }

  /** Rebuild the shared viewer's run from the session: run_start then every step instantly; park the playhead at the end. */
  function replay({ upto = m.session.steps.length, quiet = true } = {}) {
    const s = m.session;
    const hold = s.steps.splice(upto); // temporarily hide steps that are not part of the replay (ensureLive is called before apply)
    store.locked = new Set(s.seedLocked);
    ingest(s.runStart());
    for (const st of s.steps) { ingest(s.candidate(st)); ingest(s.generation(st)); }
    s.steps.push(...hold);
    timeline.live = false;
    timeline.pause();
    bus.emit("timeline:keyframes", { timeline });
    timeline.seekFrame(timeline.keyframes.length - 1, { end: true });
    if (!quiet) timeline.play();
  }

  function afterStepChange() {
    m.selected = [];
    overlay.selected = m.selected;
    if (m.session.steps.length === 0) store.locked = new Set(m.session.locked);
    loadEdits();
    renderDock();
  }

  function undo() {
    const s = m.session;
    if (!s || !s.steps.length) return;
    leaveEdit();
    s.undo();
    replay({});
    afterStepChange();
  }
  function redo() {
    const s = m.session;
    if (!s || !s.redoStack.length) return;
    leaveEdit();
    ensureLive();
    const step = s.redo();
    pushStep(step, true);
    afterStepChange();
  }

  async function logStep(step) {
    const s = m.session;
    if (!s.sessionId) return;
    try { await api.appendStep(s.sessionId, s.stepBody(step)); s.logged++; } catch (err) { s.logFailed++; console.warn("manual: step not logged", err); }
    renderDock();
  }

  // ------------------------------------------------------------------- session
  async function start({ smiles, molblock, score, label = "seed" } = {}) {
    smiles = smiles || store.seed?.smiles; molblock = molblock || store.seed?.molblock;
    score = score || store.seed?.score;
    if (!smiles || !molblock) { m.error = "No 3D molecule to start from. Validate a SMILES or drop a MOL file first."; return renderDock(); }
    m.error = null;
    if (!score) {
      try { const v = await api.validate(smiles); score = v.score; } catch (err) { /* handled below */ }
      if (!score) { m.error = "Could not score the starting molecule (is the server reachable?)."; return renderDock(); }
    }
    const s = new ManualSession({ seedSmiles: smiles, seedMolblock: molblock, seedScore: score, targetId: targetId(), locked: [...store.locked], label });
    s.seedLocked = [...store.locked];
    m.session = s; m.selected = []; m.previews.clear(); listCache.clear();
    try { const r = await api.createSession({ seed_smiles: smiles, target_id: targetId() }); s.sessionId = r.session_id || null; } catch (err) { console.warn("manual: no server session", err); }
    replay({});
    renderDock();
    loadEdits();
    toast("Manual mode: click an atom, hover an edit to preview it, click to apply.");
  }

  async function exportJsonl() {
    const s = m.session;
    if (!s?.steps.length) return;
    let text = null, how = "server";
    if (s.sessionId) { try { text = await api.exportText(s.sessionId); } catch (err) { console.warn("manual: server export failed", err); } }
    if (text == null) { text = s.jsonl(); how = "client-side"; }
    const url = URL.createObjectURL(new Blob([text], { type: "application/x-ndjson" }));
    const a = document.createElement("a");
    a.href = url; a.download = `metis-manual-${s.sessionId || s.runId}.jsonl`;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
    toast(`Exported ${text.trim().split("\n").length} transition(s) (${how} JSONL)`);
    bus.emit("manual:export", { how, lines: text.trim().split("\n").length });
  }

  function saveSession() {
    const s = m.session;
    if (!s?.steps.length) return;
    window.dispatchEvent(new CustomEvent("metis:manual-export", {
      detail: { steps: s.steps.map((x) => s.stepBody(x)), jsonl: s.jsonl(), seed_smiles: s.seedSmiles, target_id: s.targetId },
    }));
  }

  function finish() {
    const s = m.session;
    if (!s || !s.steps.length) return;
    ensureLive();
    ingest(s.runEnd());
    document.getElementById("panel-story")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ------------------------------------------------------------------ events
  function onDockClick(ev) {
    const t = ev.target.closest("[data-act], .md-edit");
    if (!t) return;
    if (t.classList.contains("md-edit")) return void applyEdit(+t.dataset.i);
    switch (t.dataset.act) {
      case "collapse": m.collapsed = !m.collapsed; return renderDock();
      case "start-seed": return void start({});
      case "start-lead": { const c = m.lead; return void start({ smiles: c.smiles, molblock: c.molblock, score: c.score, label: "last run's lead" }); }
      case "undo": return undo();
      case "redo": return redo();
      case "lock": return toggleLock(m.selected);
      case "export": return void exportJsonl();
      case "save": return void saveSession();
      case "clear-sel": m.selected = []; return void renderDockAfterSel();
    }
  }
  function onStepsClick(ev) {
    const t = ev.target.closest("[data-act], [data-frame]");
    if (!t) return;
    if (t.dataset.frame != null) { timeline.pause(); timeline.seekFrame(+t.dataset.frame, { end: true }); return renderSteps(); }
    if (t.dataset.act === "replay") { ensureLive(); timeline.seekFrame(0); timeline.play(); }
    else if (t.dataset.act === "finish") finish();
  }

  function toggleLock(atoms) {
    const s = m.session;
    if (!s || !atoms.length) return;
    if (!onLatest()) return toast("Jump to the latest step to change locks.");
    const lock = !atoms.every((i) => s.locked.has(i));
    for (const i of atoms) { if (lock) s.locked.add(i); else s.locked.delete(i); }
    if (s.steps.length === 0) { store.locked = new Set(s.locked); s.seedLocked = [...s.locked]; bus.emit("lock:change", { locked: [...s.locked] }); }
    listCache.clear(); // applicability depends on the locks
    loadEdits();
    renderDock();
    toast(`${atoms.length} atom${atoms.length === 1 ? "" : "s"} ${lock ? "locked: edits will never touch " + (atoms.length === 1 ? "it" : "them") : "unlocked"}`);
  }

  function renderDockAfterSel() { leaveEdit(); renderDock(); }

  let down = null;
  function onDown(ev) { if (!ev.target.closest(".mdock, .story-chip")) down = { x: ev.clientX, y: ev.clientY, t: performance.now() }; }
  function onMove(ev) {
    if (!m.active || !m.session || ev.target.closest(".mdock")) return;
    overlay.hover = ev.buttons ? -1 : overlay.pick(ev.clientX, ev.clientY);
    stage.host.style.cursor = overlay.hover >= 0 ? "pointer" : "";
  }
  function onUp(ev) {
    const d = down; down = null;
    if (!d || !m.active || !m.session) return;
    if (Math.hypot(ev.clientX - d.x, ev.clientY - d.y) > 5 || performance.now() - d.t > 600) return;
    const i = overlay.pick(ev.clientX, ev.clientY);
    if (!onLatest()) {
      timeline.pause();
      timeline.seekFrame(latestIndex(), { end: true });
      return toast("Showing the latest step. Click an atom again to select it.");
    }
    if (i < 0) { if (m.selected.length) { m.selected = []; renderDockAfterSel(); } return; }
    if (ev.shiftKey || ev.altKey) return toggleLock([i]);
    if (ev.metaKey || ev.ctrlKey) m.selected = m.selected.includes(i) ? m.selected.filter((x) => x !== i) : [...m.selected, i].slice(-2);
    else m.selected = m.selected.length === 1 && m.selected[0] === i ? [] : [i];
    renderDockAfterSel();
  }
  function onKey(ev) {
    if (!m.active || ev.target.matches?.("input, textarea, select")) return;
    const mod = ev.metaKey || ev.ctrlKey;
    if (mod && ev.key.toLowerCase() === "z") { ev.preventDefault(); ev.shiftKey ? redo() : undo(); }
    else if (!mod && ev.key === "Escape") { m.selected = []; renderDockAfterSel(); }
    else if (!mod && ev.key.toLowerCase() === "l" && m.selected.length) toggleLock(m.selected);
  }

  // keep the dock's lock-ring switch and the steps strip in sync with the playhead
  const onKf = () => { if (!m.active) return; if (!m.session) return renderDock(); overlay.showLocks = !(m.session.steps.length === 0 && timeline.state().keyframe?.kind === "seed"); renderSteps(); };
  const onStore = () => { if (m.active && !m.session) renderDock(); };

  function activate() {
    if (m.active) return;
    m.active = true;
    build();
    overlay.start();
    document.body.classList.add("manual-on");
    unsubs = [bus.on("timeline:keyframe", onKf), bus.on("run:end", onStore), bus.on("view:keyframe", onKf)];
    renderDock();
    bus.emit("manual:activate", {});
  }
  function deactivate() {
    if (!m.active) return;
    m.active = false;
    leaveEdit();
    unsubs.forEach((u) => u());
    document.body.classList.remove("manual-on");
    teardown();
    bus.emit("manual:deactivate", {});
  }

  // integration points for ui-structure (any of them is enough)
  window.MetisStages?.onStageChange?.((stg) => (stg === "manual" ? activate() : deactivate()));
  bus.on("stage:change", (e) => ((e?.stage || e) === "manual" ? activate() : deactivate()));
  const api2 = { activate, deactivate, start, undo, redo, exportJsonl, finish, get session() { return m.session; }, get active() { return m.active; }, get _overlay() { return overlay; } };
  window.MetisManual = api2;
  return api2;
}
