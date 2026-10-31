// Story card: shown on run_end. Headline, seed vs final structures (two synced 3Dmol views; atoms added / removed along the
// edit path are tinted), descriptor table with animated arrows, alerts removed, and PLAPT binding before/after (tagged predicted).
import { parseMolblock, superpose } from "../molparse.js";
import { ELEM_COLORS, elemRadius } from "../morph.js";
import { LABEL, esc, fmt, fmtSigned, judge, bindingOf, countUp, scoreColor, DIGITS } from "./util.js";
import { molSvg } from "./mol2d.js";

const ORDER = ["qed", "logp", "tpsa", "mw", "hbd", "hba", "rotb", "fsp3", "sa", "lipinski_violations", "alerts", "heavy_atoms"];
const ALWAYS = new Set(["qed", "logp", "tpsa", "mw", "alerts"]);
const GREEN = 0x34f5a1, RED = 0xff3b30;
const SOURCE_LABEL = { live: "Live run", replay: "Recorded replay", mock: "Mock data", manual: "Manual session" };

/** Compose the atom maps along edit_path: seed atom index -> final atom index. Null when the path is not a plain chain. */
export function composeMaps(store, seedAtomCount) {
  const path = store.end?.edit_path || [];
  const byId = new Map(store.candidates.map((c) => [c.id, c]));
  let cur = new Map(Array.from({ length: seedAtomCount }, (_, i) => [i, i]));
  let smiles = store.seed.smiles, last = null;
  for (const id of path) {
    const c = byId.get(id);
    // (the config's seed string may not be the engine's canonical SMILES, so the first hop is not compared)
    if (!c || (last && c.parent_smiles !== smiles)) return null;
    const step = new Map(c.atom_map.map(([p, ch]) => [p, ch]));
    const next = new Map();
    for (const [s, p] of cur) if (step.has(p)) next.set(s, step.get(p));
    cur = next; smiles = c.smiles; last = c;
  }
  return { map: cur, last };
}

function atomsOf(block) {
  const { atoms, bonds } = parseMolblock(block);
  return { atoms, bonds };
}

function makeViewer(host) {
  if (!window.$3Dmol) return null;
  return window.$3Dmol.createViewer(host, { backgroundAlpha: 0, antialias: true });
}

function showMolecule(viewer, { atoms, bonds }, tint) {
  viewer.removeAllModels();
  const model = viewer.addModel();
  model.addAtoms(atoms.map((a, i) => ({ elem: a.elem, x: a.x, y: a.y, z: a.z, serial: i, index: i, bonds: [], bondOrder: [] })));
  const A = model.atoms;
  for (const b of bonds) { A[b.a].bonds.push(b.b); A[b.a].bondOrder.push(b.order); A[b.b].bonds.push(b.a); A[b.b].bondOrder.push(b.order); }
  A.forEach((a, i) => {
    const hl = tint?.has(i);
    const col = hl ? tint.get(i) : ELEM_COLORS[a.elem] ?? 0xd0a8ff;
    a.style = { stick: { radius: 0.14, color: col }, sphere: { radius: elemRadius(a.elem) * (hl ? 1.05 : 0.9), color: col } };
  });
  viewer.zoomTo({}, 0);
  viewer.zoom(1.05, 0);
  viewer.render();
}

export function init(ctx = window.metis) {
  const { bus, store, timeline } = ctx;
  const mount = ctx.mounts?.story || document.getElementById("panel-story");
  if (!mount) return null;
  let built = null, rafId = 0, visible = true, lead = 0, autoScrolled = false, chip = null, io = null;

  function ensureChip() {
    if (chip) return chip;
    const wrap = document.getElementById("stage-wrap");
    if (!wrap) return null;
    chip = document.createElement("button");
    chip.className = "story-chip";
    chip.type = "button";
    chip.innerHTML = `<span class="dot"></span>Run complete · see the story <b>↓</b>`;
    chip.hidden = true;
    chip.onclick = () => mount.scrollIntoView({ behavior: "smooth", block: "start" });
    wrap.appendChild(chip);
    return chip;
  }

  function clear() {
    cancelAnimationFrame(rafId);
    built = null;
    mount.replaceChildren();
    if (chip) chip.hidden = true;
    autoScrolled = false;
  }

  function render(end) {
    const seed = store.seed, seedS = end.seed_score, finS = end.final_score;
    const nEdits = end.edit_path.length;
    const bSeed = bindingOf(seedS), bFin = bindingOf(finS);
    const source = store.source;
    const ringHtml = (id, v) => `<svg class="ring" viewBox="0 0 100 100" aria-hidden="true"><circle cx="50" cy="50" r="42" class="trk"/><circle cx="50" cy="50" r="42" class="val" id="${id}" pathLength="100" stroke-dasharray="0 100" style="stroke:${scoreColor(v)}"/></svg>`;
    const rows = [];
    const keys = ORDER.filter((k) => k in seedS.descriptors && (ALWAYS.has(k) || Math.abs((finS.descriptors[k] ?? 0) - (seedS.descriptors[k] ?? 0)) > 1e-9));
    const mk = (k, a, b, extra = "") => {
      const d = b - a;
      const cls = k === "composite" ? (d > 0 ? "good" : d < 0 ? "bad" : "neutral") : judge(k, a, b);
      const arrow = Math.abs(d) < 1e-9 ? "–" : d > 0 ? "▲" : "▼";
      return `<tr class="sr ${extra}" style="--i:${rows.length}"><td class="k">${LABEL[k] || k}</td>
        <td class="v"><span class="from">${fmt(k, a)}</span><span class="sep">→</span><b class="to" data-k="${k}" data-a="${a}" data-b="${b}">${fmt(k, a)}</b></td>
        <td class="d"><span class="arrow ${cls} ${d > 0 ? "up" : d < 0 ? "down" : ""}">${arrow}</span><span class="dv ${cls}" data-d="${d}" data-k="${k}">${Math.abs(d) < 1e-9 ? "" : fmtSigned(k, d)}</span></td></tr>`;
    };
    rows.push(mk("composite", seedS.composite, finS.composite, "hero"));
    for (const k of keys) rows.push(mk(k, seedS.descriptors[k], finS.descriptors[k]));

    const seedAl = seedS.alerts || [], finAl = finS.alerts || [];
    const removed = seedAl.filter((a) => !finAl.some((f) => f.name === a.name));
    const added = finAl.filter((a) => !seedAl.some((s) => s.name === a.name));
    const remain = finAl.filter((a) => seedAl.some((s) => s.name === a.name));
    const chips = [
      ...removed.map((a) => `<span class="chip removed" title="${esc(a.family)}">${esc(a.name)}</span>`),
      ...added.map((a) => `<span class="chip added" title="${esc(a.family)}">+ ${esc(a.name)}</span>`),
      ...remain.map((a) => `<span class="chip remain" title="${esc(a.family)}">${esc(a.name)}</span>`),
    ].join("") || `<span class="chip none">no alerts at start or end</span>`;

    let bindHtml = "";
    if (bSeed && bFin) {
      const d = bFin.value - bSeed.value, cls = d > 0 ? "good" : d < 0 ? "bad" : "neutral";
      bindHtml = `<div class="bind">
        <div class="bind-top"><span class="bind-ttl">Binding affinity (PLAPT)</span><span class="tag-pred" title="A model's prediction from sequence and SMILES, not a measurement">predicted</span></div>
        <div class="bind-vals"><span class="from">${bSeed.value.toFixed(2)}</span><span class="sep">→</span><b class="to bind-to" data-a="${bSeed.value}" data-b="${bFin.value}">${bSeed.value.toFixed(2)}</b>
          <span class="arrow ${cls} ${d > 0 ? "up" : d < 0 ? "down" : ""}">${d > 0 ? "▲" : d < 0 ? "▼" : "–"}</span><span class="dv ${cls}" data-d="${d}" data-k="bind">${Math.abs(d) < 1e-9 ? "" : fmtSigned("bind", d)}</span></div>
        <div class="bind-note">${esc(bSeed.key === "plapt_pkd" ? "pKd" : "affinity score")}, higher = tighter predicted binding. Model output only; verify experimentally.</div></div>`;
    }

    mount.innerHTML = `
    <article class="story ${source === "mock" ? "is-mock" : ""}">
      <header class="story-head">
        <div class="story-kicker">
          <span class="story-pill ok">Run complete</span>
          <span class="story-pill src ${esc(source)}">${esc(SOURCE_LABEL[source] || source || "")}</span>
          ${bSeed ? `<span class="story-pill pred">binding predicted</span>` : ""}
        </div>
        <h2 class="story-title">${esc(end.headline)}</h2>
        <p class="story-sub">${nEdits} accepted edit${nEdits === 1 ? "" : "s"} on the path from seed to final${store.generations.length ? ` · ${store.generations.length} generation${store.generations.length === 1 ? "" : "s"}` : ""}${removed.length ? ` · ${removed.length} structural alert${removed.length === 1 ? "" : "s"} removed` : ""}</p>
      </header>
      <div class="story-body">
        <div class="story-views">
          <figure class="sv" data-which="0"><div class="sv-view"></div>
            <figcaption><div class="sv-tag">Seed</div><div class="sv-ring">${ringHtml("ring-seed", seedS.composite)}<b class="rv" id="rv-seed">0.00</b></div><code class="sv-smi" title="${esc(end.seed_smiles)}">${esc(end.seed_smiles)}</code></figcaption></figure>
          <div class="sv-arrow" aria-hidden="true"><svg viewBox="0 0 60 24"><path d="M2 12h50m-10-8 10 8-10 8" /></svg></div>
          <figure class="sv" data-which="1"><div class="sv-view"></div>
            <figcaption><div class="sv-tag fin">Final</div><div class="sv-ring">${ringHtml("ring-fin", finS.composite)}<b class="rv" id="rv-fin">0.00</b></div><code class="sv-smi" title="${esc(end.final_smiles)}">${esc(end.final_smiles)}</code></figcaption></figure>
          <div class="sv-legend"><span><i class="g"></i>atoms added</span><span><i class="r"></i>atoms removed</span><span class="muted">drag either view to rotate both</span></div>
        </div>
        <div class="story-table">
          <table class="st"><tbody>${rows.join("")}</tbody></table>
          <div class="st-alerts"><span class="lbl">Structural alerts${removed.length ? `: ${removed.length} removed` : ""}${added.length ? `, ${added.length} added` : ""}</span><div class="chips">${chips}</div></div>
          ${bindHtml}
        </div>
      </div>
    </article>`;
    built = { end, rows: rows.length };
    animate();
    mountViews(end);
    const c = ensureChip();
    if (c) { c.hidden = false; }
  }

  function animate() {
    mount.querySelectorAll(".to").forEach((n, i) => {
      const a = +n.dataset.a, b = +n.dataset.b, k = n.dataset.k;
      countUp(n, b, { from: a, digits: n.classList.contains("bind-to") ? 2 : DIGITS[k] ?? 0, ms: 1000, delay: 350 + i * 90 });
    });
    mount.querySelectorAll(".dv").forEach((n, i) => {
      const d = +n.dataset.d, k = n.dataset.k;
      if (Math.abs(d) < 1e-9) return;
      countUp(n, d, { from: 0, digits: k === "bind" ? 2 : DIGITS[k] ?? 0, signed: true, ms: 1000, delay: 350 + i * 90 });
    });
    for (const [id, v, rv] of [["ring-seed", built.end.seed_score.composite, "rv-seed"], ["ring-fin", built.end.final_score.composite, "rv-fin"]]) {
      const c = mount.querySelector("#" + id);
      requestAnimationFrame(() => { c.style.transition = "stroke-dasharray 1.1s cubic-bezier(.2,.8,.2,1)"; c.setAttribute("stroke-dasharray", `${(v * 100).toFixed(1)} 100`); });
      countUp(mount.querySelector("#" + rv), v, { digits: 2, ms: 1100, delay: 100 });
    }
  }

  // ------------------------------------------------------------------ 3D views
  const viewers = [null, null];
  function mountViews(end) {
    const hosts = [...mount.querySelectorAll(".sv-view")];
    let seedD, finD, tintSeed = new Map(), tintFin = new Map();
    try {
      seedD = atomsOf(store.seed.molblock);
      const composed = composeMaps(store, seedD.atoms.length);
      const finBlock = composed?.last?.molblock || store.seed.molblock;
      finD = atomsOf(finBlock);
      if (composed && composed.last) {
        const pairs = [...composed.map].map(([s, f]) => [f, s]);
        superpose(seedD.atoms, finD.atoms, pairs); // orient the final like the seed so the two views read as before / after
        const kept = new Set(composed.map.keys()), keptF = new Set(composed.map.values());
        seedD.atoms.forEach((_, i) => { if (!kept.has(i)) tintSeed.set(i, RED); });
        finD.atoms.forEach((_, i) => { if (!keptF.has(i)) tintFin.set(i, GREEN); });
      }
    } catch (err) {
      console.error("story: could not prepare the 3D views", err);
      hosts.forEach((h, i) => { h.innerHTML = `<div class="sv-fallback">${molSvg(i ? (end.final_smiles && store.candidates.find((c) => c.id === end.edit_path.at(-1))?.molblock) || store.seed.molblock : store.seed.molblock, { w: 220, h: 150 })}</div>`; });
      return;
    }
    hosts.forEach((h, i) => {
      if (!viewers[i] || viewers[i].host !== h) {
        try { viewers[i] = makeViewer(h); if (viewers[i]) viewers[i].host = h; } catch (e) { viewers[i] = null; }
      }
      if (!viewers[i]) { h.innerHTML = `<div class="sv-fallback">${molSvg((i ? store.candidates.find((c) => c.id === end.edit_path.at(-1))?.molblock : null) || store.seed.molblock, { w: 220, h: 150 })}<div class="muted">3D unavailable</div></div>`; return; }
      showMolecule(viewers[i], i ? finD : seedD, i ? tintFin : tintSeed);
      h.onpointerdown = () => { lead = i; };
    });
    cancelAnimationFrame(rafId);
    let last = performance.now();
    const loop = (now) => {
      const dt = Math.min(now - last, 100); last = now;
      if (visible && viewers[0] && viewers[1]) {
        const L = viewers[lead], F = viewers[1 - lead];
        if (!window.metis?.store?.overlay || store.overlay.autorotate) L.rotate((10 * dt) / 1000, "y");
        const va = L.getView(), vb = F.getView();
        vb.splice(4, 4, ...va.slice(4, 8)); // copy only the rotation quaternion; each view keeps its own zoom / centre
        F.setView(vb);
        L.render(); F.render();
      }
      rafId = requestAnimationFrame(loop);
    };
    rafId = requestAnimationFrame(loop);
    requestAnimationFrame(() => viewers.forEach((v) => { v?.resize(); v?.zoomTo({}, 0); v?.zoom(1.05, 0); v?.render(); }));
  }

  io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      visible = e.isIntersecting;
      if (chip && built) chip.hidden = e.isIntersecting;
    }
  }, { threshold: 0.15 });
  io.observe(mount);

  bus.on("run:start", clear);
  bus.on("run:end", ({ payload }) => {
    try { render(payload); } catch (err) { console.error("story card failed", err); }
  });
  bus.on("timeline:pause", (s) => {
    // when playback has reached the end of a finished run, bring the story into view once
    if (!built || autoScrolled || store.status !== "done" || !s || s.t < s.duration - 1) return;
    autoScrolled = true;
    setTimeout(() => { if (!visible) mount.scrollIntoView({ behavior: "smooth", block: "start" }); }, 700);
  });
  // already finished before init (e.g. late init)?
  if (store.end) render(store.end);
  return { render, clear };
}
