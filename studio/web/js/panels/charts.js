// Run charts: hand-written SVG, no library. Composite (best + mean) per generation, best predicted binding affinity so far,
// and structural alerts of the best molecule per generation. Seed values are dashed baselines; the playhead's generation is marked.
import { esc, bindingOf, rafThrottle } from "./util.js";

const W = 360, H = 176, M = { l: 38, r: 16, t: 16, b: 28 };

const niceTicks = (lo, hi, n = 3) => {
  const span = hi - lo || 1, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
};

/** Pure: derive the series the charts draw from the store (exported for tests). */
export function deriveSeries(store) {
  const gens = [...store.generations].sort((a, b) => a.generation - b.generation);
  const seedScore = store.seed?.score || null;
  const best = [], mean = [], alerts = [], aff = [];
  const seedAff = bindingOf(seedScore)?.value ?? null;
  let runningAff = seedAff;
  if (seedScore) { best.push([0, seedScore.composite]); mean.push([0, seedScore.composite]); alerts.push([0, seedScore.descriptors.alerts]); }
  if (seedAff != null) aff.push([0, seedAff]);
  for (const g of gens) {
    best.push([g.generation, g.best.composite]);
    mean.push([g.generation, g.mean_composite]);
    const sc = store.bySmiles.get(g.best.smiles)?.score;
    if (sc) alerts.push([g.generation, sc.descriptors.alerts]);
    for (const c of store.candidates) {
      if (c.generation !== g.generation) continue;
      const b = bindingOf(c.score)?.value;
      if (b != null && (runningAff == null || b > runningAff)) runningAff = b;
    }
    if (runningAff != null && (aff.length || seedAff != null || store.candidates.some((c) => bindingOf(c.score)))) aff.push([g.generation, runningAff]);
  }
  return { gens, best, mean, alerts, aff, seed: { composite: seedScore?.composite ?? null, alerts: seedScore?.descriptors.alerts ?? null, aff: seedAff } };
}

function chartSvg(def, id, anim) {
  const { series, baseline, yFmt, step, domain, unit } = def;
  const all = series.flatMap((s) => s.pts);
  if (!all.length) return "";
  const xs = all.map((p) => p[0]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1);
  let lo = Math.min(...all.map((p) => p[1]), baseline?.value ?? Infinity), hi = Math.max(...all.map((p) => p[1]), baseline?.value ?? -Infinity);
  if (domain) { lo = domain[0] ?? lo; hi = domain[1] ?? hi; } else { const pad = (hi - lo || 1) * 0.15; lo -= pad; hi += pad; }
  if (def.floor0) lo = 0;
  const sx = (x) => M.l + ((x - x0) / (x1 - x0)) * (W - M.l - M.r);
  const sy = (y) => H - M.b - ((y - lo) / (hi - lo || 1)) * (H - M.t - M.b);
  let g = "";
  for (const t of niceTicks(lo, hi, 3)) g += `<line class="grid" x1="${M.l}" x2="${W - M.r}" y1="${sy(t).toFixed(1)}" y2="${sy(t).toFixed(1)}"/><text class="tick" x="${M.l - 6}" y="${(sy(t) + 3.5).toFixed(1)}" text-anchor="end">${yFmt(t)}</text>`;
  const xt = niceTicks(x0, x1, Math.min(6, x1 - x0));
  for (const t of xt) if (Number.isInteger(t)) g += `<text class="tick" x="${sx(t).toFixed(1)}" y="${H - M.b + 16}" text-anchor="middle">${t === 0 ? "seed" : t}</text>`;
  g += `<text class="axis" x="${(M.l + W - M.r) / 2}" y="${H - 2}" text-anchor="middle">generation</text>`;
  if (baseline && baseline.value != null) g += `<line class="base" x1="${M.l}" x2="${W - M.r}" y1="${sy(baseline.value).toFixed(1)}" y2="${sy(baseline.value).toFixed(1)}"/><text class="base-l" x="${M.l + 6}" y="${(sy(baseline.value) - 4).toFixed(1)}">${esc(baseline.label)}</text>`;
  const path = (pts) => {
    if (!step) return pts.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)} ${sy(p[1]).toFixed(1)}`).join(" ");
    return pts.map((p, i) => (i ? `H${sx(p[0]).toFixed(1)}V${sy(p[1]).toFixed(1)}` : `M${sx(p[0]).toFixed(1)} ${sy(p[1]).toFixed(1)}`)).join(" ");
  };
  for (const s of series) {
    if (s.area && s.pts.length > 1) g += `<path class="area" d="${path(s.pts)} V${sy(lo).toFixed(1)} H${sx(s.pts[0][0]).toFixed(1)} Z" style="fill:${s.color}"/>`;
    g += `<path class="ln ${anim ? "anim" : ""} ${s.dash ? "dash" : ""}" pathLength="1" d="${path(s.pts)}" style="stroke:${s.color}"/>`;
    if (!s.dash) for (const p of s.pts) g += `<circle class="pt" cx="${sx(p[0]).toFixed(1)}" cy="${sy(p[1]).toFixed(1)}" r="3" style="fill:${s.color}"/>`;
    const last = s.pts[s.pts.length - 1];
    g += `<text class="end-l" x="${Math.min(sx(last[0]) + 6, W - 2).toFixed(1)}" y="${(sy(last[1]) + 4).toFixed(1)}" style="fill:${s.color}" ${sx(last[0]) + 6 > W - 30 ? 'text-anchor="end" dx="-8" dy="-9"' : ""}>${yFmt(last[1])}</text>`;
  }
  g += `<line class="head" id="${id}-head" y1="${M.t - 4}" y2="${H - M.b}" x1="0" x2="0" style="display:none"/>`;
  g += `<line class="hov" id="${id}-hov" y1="${M.t}" y2="${H - M.b}" x1="0" x2="0" style="display:none"/>`;
  g += `<rect class="hit" x="${M.l}" y="${M.t}" width="${W - M.l - M.r}" height="${H - M.t - M.b}" fill="transparent" data-chart="${id}"/>`;
  return { svg: `<svg viewBox="0 0 ${W} ${H}" class="chart-svg" role="img" aria-label="${esc(def.title)}">${g}</svg>`, sx, x0, x1, series, def };
}

export function init(ctx = window.metis) {
  const { bus, store } = ctx;
  const mount = ctx.mounts?.charts || document.getElementById("panel-charts");
  if (!mount) return null;
  let host = null, built = {}, animNext = false, head = null;

  function shell() {
    if (host) return;
    mount.innerHTML = `<section class="pcard charts"><header class="pc-head"><h3>Progress</h3><span class="pc-sub" id="ch-sub"></span></header><div class="ch-grid" id="ch-grid"></div>
      <div class="chart-tip" id="ch-tip" hidden></div></section>`;
    host = mount;
    const grid = mount.querySelector("#ch-grid"), tip = mount.querySelector("#ch-tip");
    grid.addEventListener("pointermove", (e) => {
      const hit = e.target.closest?.("[data-chart]");
      if (!hit) return hideHover();
      const c = built[hit.dataset.chart];
      if (!c) return;
      const svg = hit.ownerSVGElement, r = svg.getBoundingClientRect(), x = ((e.clientX - r.left) / r.width) * W;
      const xs = [...new Set(c.series.flatMap((s) => s.pts.map((p) => p[0])))];
      const gx = xs.reduce((a, b) => (Math.abs(c.sx(b) - x) < Math.abs(c.sx(a) - x) ? b : a));
      const hov = svg.querySelector(`#${hit.dataset.chart}-hov`);
      hov.setAttribute("x1", c.sx(gx)); hov.setAttribute("x2", c.sx(gx)); hov.style.display = "";
      const rows = c.series.map((s) => { const p = s.pts.find((q) => q[0] === gx); return p ? `<div><i style="background:${s.color}"></i>${esc(s.name)} <b>${c.def.yFmt(p[1])}</b></div>` : ""; }).join("");
      tip.innerHTML = `<div class="t-h">${gx === 0 ? "Seed" : "Generation " + gx}</div>${rows}`;
      tip.hidden = false;
      const cr = mount.querySelector(".charts").getBoundingClientRect();
      tip.style.left = Math.min(e.clientX - cr.left + 14, cr.width - 170) + "px";
      tip.style.top = e.clientY - cr.top + 14 + "px";
    });
    grid.addEventListener("pointerleave", hideHover);
  }
  function hideHover() {
    mount.querySelectorAll(".hov").forEach((l) => (l.style.display = "none"));
    const t = mount.querySelector("#ch-tip");
    if (t) t.hidden = true;
  }

  const draw = rafThrottle(() => {
    if (!store.seed) return;
    const d = deriveSeries(store);
    if (d.gens.length === 0 && !d.best.length) return;
    if (d.gens.length === 0) return; // nothing to chart before the first generation event
    shell();
    const defs = [];
    defs.push({ id: "ch-comp", title: "Composite score per generation", yFmt: (v) => v.toFixed(2), domain: null,
      series: [{ name: "best", pts: d.best, color: "var(--good)", area: true }, { name: "population mean", pts: d.mean, color: "var(--accent)", dash: false }],
      baseline: d.seed.composite != null ? { value: d.seed.composite, label: `seed ${d.seed.composite.toFixed(2)}` } : null,
      head: "Composite score (best / mean)" });
    if (d.aff.length > 1 || d.seed.aff != null) {
      defs.push({ id: "ch-aff", title: "Best predicted binding affinity so far", yFmt: (v) => v.toFixed(2), domain: null,
        series: [{ name: "best predicted affinity", pts: d.aff, color: "var(--warn)", area: true }],
        baseline: d.seed.aff != null ? { value: d.seed.aff, label: `seed ${d.seed.aff.toFixed(2)}` } : null,
        head: "Best predicted affinity so far", tag: "predicted" });
    }
    if (d.alerts.length > 1) {
      defs.push({ id: "ch-alert", title: "Structural alerts of the best molecule per generation", yFmt: (v) => String(Math.round(v)), step: true, floor0: true,
        domain: [0, Math.max(1, ...d.alerts.map((p) => p[1]))], series: [{ name: "alerts (best)", pts: d.alerts, color: "var(--bad)" }],
        baseline: d.seed.alerts != null ? { value: d.seed.alerts, label: `seed ${d.seed.alerts}` } : null, head: "Structural alerts (best molecule)" });
    }
    built = {};
    const anim = animNext;
    animNext = false;
    mount.querySelector("#ch-grid").innerHTML = defs.map((def) => {
      const c = chartSvg(def, def.id, anim);
      if (!c) return "";
      built[def.id] = c;
      return `<figure class="chart"><figcaption>${esc(def.head)}${def.tag ? `<span class="tag-pred">${esc(def.tag)}</span>` : ""}</figcaption>${c.svg}</figure>`;
    }).join("");
    mount.querySelector("#ch-sub").textContent = `${d.gens.length} generation${d.gens.length === 1 ? "" : "s"}`;
    placeHead();
  });

  function placeHead() {
    if (!host) return;
    const gen = head == null ? null : head;
    for (const [id, c] of Object.entries(built)) {
      const l = mount.querySelector(`#${id}-head`);
      if (!l) continue;
      if (gen == null) { l.style.display = "none"; continue; }
      const x = c.sx(Math.min(Math.max(gen, c.x0), c.x1));
      l.setAttribute("x1", x); l.setAttribute("x2", x); l.style.display = "";
    }
  }

  bus.on("run:start", () => { head = 0; built = {}; mount.replaceChildren(); host = null; });
  bus.on("run:generation", () => draw());
  bus.on("run:end", () => { animNext = true; draw(); });
  bus.on("view:keyframe", ({ keyframe }) => {
    head = keyframe?.kind === "edit" ? keyframe.candidate.generation : 0;
    placeHead();
  });
  if (store.seed && store.generations.length) draw();
  return { draw };
}
