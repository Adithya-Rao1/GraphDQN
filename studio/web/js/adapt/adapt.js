// "Adapt to my edits": self-injecting panel for manual mode. Add once to index.html:
//   <script type="module" src="js/adapt/adapt.js"></script>
// Inputs (all optional, first one wins):
//   bus events  manual:session {session_id}  |  manual:step {session_id?, steps?}  |  manual:steps {session_id?, steps:[...]|count}
//   or call     window.MetisAdapt.update({session_id, count})   // when the manual panel prefers a direct call
// Output: CustomEvent "metis:adapter-selected" {detail:{adapter_id}} and window.MetisStudio.extraConfig = {adapter_id}.
const MIN_STEPS = 5;
const API = new URLSearchParams(location.search).get("api") || "";
const css = new URL("../../css/adapt.css", import.meta.url).href;
if (!document.querySelector(`link[href="${css}"]`)) {
  const l = document.createElement("link"); l.rel = "stylesheet"; l.href = css; document.head.appendChild(l);
}

const st = { session: null, count: 0, job: null, result: null, err: null, progress: 0, stage: "", adapterId: null, use: false };
let root = null;

const pct = (x) => (x * 100).toFixed(0) + "%";
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function mount() {
  const host = document.getElementById("panel-manual");
  if (!host) return null;
  if (!root) { root = document.createElement("section"); root.className = "adapt-panel"; root.id = "adapt-panel"; host.appendChild(root); }
  return root;
}

function chart(ev) {
  const rows = (k, label) => {
    const group = [["base", "Base", "base"], ["adapted", "Adapted", "adapted"], ["random", "Random", "random"]].map(([key, name, cls]) => {
      const r = ev[key], v = r[`top${k}`], [lo, hi] = r[`top${k}_ci95`];
      return `<div class="adapt-row"><span>${name}</span><div class="track"><div class="fill ${cls}" style="width:${v * 100}%"></div>` +
        `<div class="ci" style="left:${lo * 100}%;width:${Math.max(0, hi - lo) * 100}%"></div></div>` +
        `<span class="v">${pct(v)} <small>[${pct(lo)}–${pct(hi)}]</small></span></div>`;
    }).join("");
    return `<div class="adapt-grp">${label}</div>${group}`;
  };
  return `<div class="adapt-chart">${rows(1, "Top-1 hit rate")}${rows(3, "Top-3 hit rate")}</div>` +
    `<div class="adapt-meta">n = ${ev.n} held-out edits · ${esc(ev.method)} · bars show the hit rate, whiskers the bootstrap 95% interval</div>` +
    `<div class="adapt-note">${esc(ev.note)}</div>`;
}

function render() {
  if (!mount()) return;
  const ready = st.count >= MIN_STEPS, busy = !!st.job && !st.result && !st.err;
  let h = `<h4>Adapt to my edits</h4><p class="adapt-sub">Fine-tunes a private copy of the edit policy on your logged manual edits, then checks whether it ranks your held-out choices higher. Predicted agreement with your past choices, not a measure of better chemistry.</p>`;
  if (!st.session && !st.count) h += `<p class="adapt-sub">Make at least ${MIN_STEPS} manual edits to enable this.</p>`;
  h += `<button class="adapt-btn" id="adapt-go" ${ready && !busy ? "" : "disabled"}>Adapt to my edits${ready ? "" : ` (${st.count}/${MIN_STEPS} edits)`}</button>`;
  if (busy) h += `<div class="adapt-bar"><i style="width:${st.progress * 100}%"></i></div><div class="adapt-meta">${esc(st.stage || "queued")}…</div>`;
  if (st.err) h += `<div class="adapt-err">${esc(st.err)}</div>`;
  if (st.result) {
    const r = st.result;
    h += chart(r.evaluation);
    if (r.dataset && r.dataset.dropped) h += `<div class="adapt-meta">${r.dataset.dropped} of ${r.dataset.received} logged steps were dropped as invalid (${esc(JSON.stringify(r.dataset.reasons))}).</div>`;
    h += `<label class="adapt-toggle"><input type="checkbox" id="adapt-use" ${st.use ? "checked" : ""}> Use my adapted policy for new runs</label>`;
  }
  root.innerHTML = h;
  root.querySelector("#adapt-go")?.addEventListener("click", start);
  root.querySelector("#adapt-use")?.addEventListener("change", (e) => select(e.target.checked));
}

function select(on) {
  st.use = on;
  window.MetisStudio = window.MetisStudio || {};
  if (on && st.adapterId) window.MetisStudio.extraConfig = { adapter_id: st.adapterId };
  else if (window.MetisStudio.extraConfig) delete window.MetisStudio.extraConfig.adapter_id;
  window.dispatchEvent(new CustomEvent("metis:adapter-selected", { detail: { adapter_id: on ? st.adapterId : null } }));
}

async function start() {
  Object.assign(st, { result: null, err: null, progress: 0, stage: "queued", adapterId: null });
  if (st.use) select(false);
  try {
    const r = await fetch(`${API}/api/adapt/jobs`, { method: "POST", headers: { "Content-Type": "application/json" }, credentials: "include",
      body: JSON.stringify({ manual_session_id: st.session }) });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || `HTTP ${r.status}`);
    st.job = j.job_id; render(); poll();
  } catch (e) { st.err = e.message; st.job = null; render(); }
}

async function poll() {
  const id = st.job;
  while (id === st.job) {
    try {
      const r = await fetch(`${API}/api/adapt/jobs/${id}`, { credentials: "include" });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || `HTTP ${r.status}`);
      st.progress = j.progress; st.stage = j.stage;
      if (j.status === "done") { st.result = j.result; st.adapterId = j.result.adapter_id; st.job = null; }
      else if (j.status === "failed") { st.err = j.error || "Adaptation failed"; st.job = null; }
    } catch (e) { st.err = e.message; st.job = null; }
    render();
    if (st.job) await new Promise((res) => setTimeout(res, 700));
  }
}

function update(d) {
  if (!d) return;
  const sid = d.session_id || d.session?.session_id;
  if (sid && sid !== st.session) { st.session = sid; st.result = null; st.err = null; if (st.use) select(false); }
  const steps = d.steps ?? d.session?.steps;
  if (Array.isArray(steps)) st.count = steps.length;
  else if (typeof steps === "number") st.count = steps;
  else if (typeof d.count === "number") st.count = d.count;
  else if (d.step || d.type === "step") st.count += 1;
  render();
}

function init() {
  window.MetisAdapt = { update, state: st };
  const bus = window.metis && window.metis.bus;
  if (bus) for (const ev of ["manual:session", "manual:step", "manual:steps"]) bus.on(ev, update);
  render();
}
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
