// Population map: generation.map_xy as a 2D scatter. Dots are coloured by the generation in which a molecule first entered the
// population; the seed, the best molecule, the edit path and the playhead's molecule are marked. The method (map_method) is stated.
import { esc, frameIndex, toast, rafThrottle } from "./util.js";
import { molSvg } from "./mol2d.js";

const W = 420, H = 340, PAD = 28;
const METHOD = {
  morgan_pca: { name: "Morgan fingerprints (2048-bit, radius 2) → PCA", short: "Morgan + PCA" },
  chemberta_pca: { name: "ChemBERTa embeddings → PCA", short: "ChemBERTa + PCA" },
};

/** Pure: collect map points from the store (exported for tests). */
export function collectPoints(store) {
  const pts = new Map(); // smiles -> point
  let method = null;
  for (const g of store.generations) {
    if (!g.map_xy || g.map_xy.length !== g.population.length) continue;
    method = g.map_method || method;
    g.population.forEach((m, i) => {
      const [x, y] = g.map_xy[i];
      const prev = pts.get(m.smiles);
      pts.set(m.smiles, { smiles: m.smiles, x, y, composite: m.composite, id: m.candidate_id || prev?.id || null, gen: prev ? prev.gen : g.generation, last: g.generation });
    });
  }
  return { pts: [...pts.values()], method };
}

const genColor = (g, gmax) => {
  const t = gmax <= 1 ? 1 : (g - 1) / (gmax - 1);
  return `hsl(${Math.round(212 - 70 * t)} ${Math.round(75 + 15 * t)}% ${Math.round(52 + 14 * t)}%)`;
};

export function init(ctx = window.metis) {
  const { bus, store, timeline } = ctx;
  const mount = ctx.mounts?.map || document.getElementById("panel-map");
  if (!mount) return null;
  let host = null, current = null, view = null;

  function shell() {
    if (host) return;
    mount.innerHTML = `<section class="pcard map"><header class="pc-head"><h3>Chemical-space map</h3><span class="pc-sub" id="map-sub"></span></header>
      <div class="map-wrap"><svg id="map-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Population map"></svg><div class="chart-tip map-tip" id="map-tip" hidden></div></div>
      <div class="map-legend"><span class="gbar" id="map-gbar"></span><span class="muted" id="map-gl"></span></div>
      <div class="pc-foot muted" id="map-foot"></div></section>`;
    host = mount;
    const svg = mount.querySelector("#map-svg"), tip = mount.querySelector("#map-tip");
    svg.addEventListener("pointermove", (e) => {
      const d = e.target.closest?.("[data-i]");
      if (!d) { tip.hidden = true; return; }
      const p = view.pts[+d.dataset.i];
      const cand = store.candidates.find((c) => c.id === p.id) || store.candidates.find((c) => c.smiles === p.smiles);
      tip.innerHTML = `${cand ? `<div class="tip-mol">${molSvg(cand.molblock, { w: 110, h: 70 })}</div>` : ""}<div class="t-h">${p.id === "seed" ? "Seed" : "Generation " + p.gen}</div><div class="mono">${esc(p.smiles)}</div><div>composite <b>${p.composite != null ? p.composite.toFixed(3) : "n/a"}</b></div>`;
      tip.hidden = false;
      const r = mount.querySelector(".map-wrap").getBoundingClientRect();
      tip.style.left = Math.min(e.clientX - r.left + 14, r.width - 240) + "px";
      tip.style.top = Math.max(e.clientY - r.top - 20, 0) + "px";
    });
    svg.addEventListener("pointerleave", () => { tip.hidden = true; });
    svg.addEventListener("click", (e) => {
      const d = e.target.closest?.("[data-i]");
      if (!d) return;
      const p = view.pts[+d.dataset.i];
      const id = p.smiles === store.seed?.smiles ? "seed" : p.id || store.candidates.find((c) => c.smiles === p.smiles)?.id;
      const i = id ? frameIndex(timeline, id) : -1;
      if (i < 0) return toast(id ? "That molecule is not in the timeline. Switch the filter to “All”." : "This population member has no candidate event to jump to.");
      timeline.seekFrame(i);
      if (!timeline.playing) timeline.play();
    });
  }

  const draw = rafThrottle(() => {
    const { pts, method } = collectPoints(store);
    if (!pts.length) return;
    shell();
    const gmax = Math.max(1, ...store.generations.map((g) => g.generation));
    // seed position: the engine anchors the map on the seed (origin); mock data has no such anchor
    const seedSmiles = store.seed?.smiles;
    let seedPt = pts.find((p) => p.smiles === seedSmiles);
    const real = store.source !== "mock";
    if (!seedPt && real) seedPt = { smiles: seedSmiles, x: 0, y: 0, composite: store.seed?.score?.composite ?? null, id: "seed", gen: 0, synthetic: true };
    const all = seedPt && !pts.includes(seedPt) ? [...pts, seedPt] : pts;
    const xs = all.map((p) => p.x), ys = all.map((p) => p.y);
    let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const sp = Math.max(x1 - x0, y1 - y0, 1e-6), cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    const sc = (Math.min(W, H) - 2 * PAD) / sp;
    const X = (x) => W / 2 + (x - cx) * sc, Y = (y) => H / 2 - (y - cy) * sc;
    view = { pts: all, X, Y };

    // best = highest composite among map points
    let best = all[0];
    for (const p of all) if ((p.composite ?? -1) > (best.composite ?? -1)) best = p;
    if (store.end) best = all.find((p) => p.smiles === store.end.final_smiles) || best;

    let g = `<defs><radialGradient id="map-bg" cx="50%" cy="50%" r="70%"><stop offset="0" stop-color="var(--accent)" stop-opacity=".10"/><stop offset="1" stop-color="var(--accent)" stop-opacity="0"/></radialGradient>
      <filter id="map-glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="3.2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
      <rect x="0" y="0" width="${W}" height="${H}" fill="url(#map-bg)" rx="12"/>
      <line class="mx" x1="${PAD}" x2="${W - PAD}" y1="${Y(0)}" y2="${Y(0)}"/><line class="mx" x1="${X(0)}" x2="${X(0)}" y1="${PAD}" y2="${H - PAD}"/>`;
    // edit path
    if (store.end && seedPt) {
      const byId = new Map(all.filter((p) => p.id).map((p) => [p.id, p]));
      const path = [seedPt, ...store.end.edit_path.map((id) => byId.get(id) || all.find((p) => p.smiles === store.candidates.find((c) => c.id === id)?.smiles)).filter(Boolean)];
      if (path.length > 1) g += `<polyline class="mpath" points="${path.map((p) => `${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join(" ")}" filter="url(#map-glow)"/>`;
    }
    const latest = Math.max(...pts.map((p) => p.last));
    pts.forEach((p, i) => {
      if (p === seedPt) return;
      g += `<circle class="md ${p.last === latest && store.status === "running" ? "new" : ""}" data-i="${i}" cx="${X(p.x).toFixed(1)}" cy="${Y(p.y).toFixed(1)}" r="${p === best ? 5.5 : 4}" style="fill:${genColor(p.gen, gmax)}"><title>${esc(p.smiles)}</title></circle>`;
    });
    if (current) {
      const p = all.find((q) => q.smiles === current);
      if (p) g += `<circle class="mcur" cx="${X(p.x).toFixed(1)}" cy="${Y(p.y).toFixed(1)}" r="10"/>`;
    }
    if (best && best !== seedPt) {
      g += `<g class="mbest" transform="translate(${X(best.x).toFixed(1)} ${Y(best.y).toFixed(1)})"><circle r="9" fill="none" stroke="var(--good)" stroke-width="2" filter="url(#map-glow)"/><text y="-14" text-anchor="middle">best</text></g>`;
    }
    if (seedPt) {
      const i = all.indexOf(seedPt), sx = X(seedPt.x), sy = Y(seedPt.y);
      const star = Array.from({ length: 10 }, (_, k) => { const a = (-Math.PI / 2) + (k * Math.PI) / 5, r = k % 2 ? 4.6 : 10; return `${(sx + r * Math.cos(a)).toFixed(1)},${(sy + r * Math.sin(a)).toFixed(1)}`; }).join(" ");
      g += `<polygon class="mseed" data-i="${i}" points="${star}"/><text class="mseed-l" x="${sx.toFixed(1)}" y="${(sy + 22).toFixed(1)}" text-anchor="middle">seed</text>`;
    }
    mount.querySelector("#map-svg").innerHTML = g;
    const m = METHOD[method];
    mount.querySelector("#map-sub").textContent = m ? m.short : method || "";
    mount.querySelector("#map-gbar").style.background = `linear-gradient(90deg, ${genColor(1, gmax)}, ${genColor(gmax, gmax)})`;
    mount.querySelector("#map-gl").textContent = `generation 1 → ${gmax}`;
    mount.querySelector("#map-foot").innerHTML = (store.source === "mock"
      ? `<b>Mock-labelled run:</b> coordinates may be scripted rather than computed from the structures. `
      : "") + `Method: ${esc(m ? m.name : method || "unknown")}. A visual aid only: axes have no physical meaning${real ? "; the seed sits at the origin and distance reads as structural dissimilarity" : ""}. Click a dot to jump to that molecule.`;
  });

  bus.on("run:start", () => { current = null; view = null; mount.replaceChildren(); host = null; });
  bus.on("run:generation", () => draw());
  bus.on("run:end", () => draw());
  bus.on("view:keyframe", ({ keyframe }) => {
    current = keyframe?.kind === "edit" ? keyframe.candidate.smiles : store.seed?.smiles;
    draw();
  });
  if (store.generations.length) draw();
  return { draw };
}
