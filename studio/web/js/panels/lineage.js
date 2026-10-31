// Lineage tree: the accepted edits from the seed as a clickable SVG tree (node = molecule thumbnail + composite, edge = edit name).
// Clicking a node seeks the shared timeline to that edit. The path to the final molecule glows; the playhead's node is ringed.
import { esc, scoreColor, frameIndex, toast, rafThrottle } from "./util.js";
import { molSvg } from "./mol2d.js";

const COL = 168, ROW = 62, NW = 74, NH = 50, PAD = 16;

/** Pure: build the tree from the store. Returns {nodes[], edges[], maxDepth, leaves}. Exported for tests. */
export function buildTree(store, only = null) {
  if (!store.seed) return null;
  const seedNode = { id: "seed", smiles: store.seed.smiles, block: store.seed.molblock, composite: store.seed.score?.composite ?? null, depth: 0, children: [], edit: null };
  const minGen = Math.min(...store.candidates.map((c) => c.generation), 1);
  const nodes = [seedNode], bySmiles = new Map([[seedNode.smiles, seedNode]]), edges = [];
  for (const c of store.candidates) {
    if (!c.accepted || bySmiles.has(c.smiles) || (only && !only.has(c.id))) continue;
    // the run config's seed string may differ from the engine's canonical SMILES, so first-generation edits hang off the seed
    const parent = bySmiles.get(c.parent_smiles) || (c.edit?.parents || []).map((s) => bySmiles.get(s)).find(Boolean) || (c.generation <= minGen ? seedNode : null);
    const p = parent || seedNode;
    const n = { id: c.id, smiles: c.smiles, block: c.molblock, composite: c.score.composite, depth: p.depth + 1, children: [], edit: c.edit, generation: c.generation, orphan: !parent, crossover: c.edit?.chosen_by === "crossover" };
    p.children.push(n);
    nodes.push(n);
    bySmiles.set(n.smiles, n);
    edges.push({ from: p, to: n, label: c.edit?.name || "edit", orphan: !parent });
  }
  // tidy layout: leaves get consecutive rows, parents sit at the mean of their children
  let row = 0;
  const place = (n) => {
    if (!n.children.length) { n.row = row++; return; }
    n.children.forEach(place);
    n.row = (n.children[0].row + n.children[n.children.length - 1].row) / 2;
  };
  place(seedNode);
  return { nodes, edges, maxDepth: Math.max(...nodes.map((n) => n.depth)), leaves: row, byId: new Map(nodes.map((n) => [n.id, n])) };
}

export function init(ctx = window.metis) {
  const { bus, store, timeline } = ctx;
  const mount = ctx.mounts?.lineage || document.getElementById("panel-lineage");
  if (!mount) return null;
  let host = null, current = "seed", showAll = false, tree = null;

  function shell() {
    if (host) return;
    mount.innerHTML = `<section class="pcard lineage">
      <header class="pc-head"><h3>Lineage</h3><span class="pc-sub" id="lin-sub"></span>
        <label class="pc-toggle" title="Also draw every other accepted edit (the whole surviving family tree)"><input type="checkbox" id="lin-all"> show all accepted</label></header>
      <div class="lin-scroll" id="lin-scroll"><svg id="lin-svg" role="img" aria-label="Lineage tree of accepted edits"></svg></div>
      <div class="pc-foot muted">Click a molecule to jump the 3D view to that edit. Node colour = composite score; green line = path to the final molecule; dashed outline = crossover child.</div>
    </section>`;
    host = mount;
    mount.querySelector("#lin-all").onchange = (e) => { showAll = e.target.checked; draw(); };
    mount.querySelector("#lin-svg").addEventListener("click", (e) => {
      const g = e.target.closest("[data-id]");
      if (!g) return;
      seekTo(g.dataset.id);
    });
    mount.querySelector("#lin-svg").addEventListener("keydown", (e) => {
      if ((e.key === "Enter" || e.key === " ") && e.target.closest("[data-id]")) { e.preventDefault(); seekTo(e.target.closest("[data-id]").dataset.id); }
    });
  }

  function seekTo(id) {
    const i = frameIndex(timeline, id);
    if (i < 0) { toast("That edit is not in the timeline. Switch the timeline filter to “All”."); return; }
    timeline.seekFrame(i);
    if (!timeline.playing) timeline.play();
  }

  function pathIds() {
    const set = new Set(["seed"]);
    const ids = store.end?.edit_path;
    if (ids) ids.forEach((id) => set.add(id));
    return set;
  }

  const draw = rafThrottle(() => {
    const hasEnd = !!store.end;
    // while streaming, draw everything accepted so the tree grows live; once finished show the winning path unless asked for all
    tree = buildTree(store, hasEnd && !showAll ? pathIds() : null);
    if (!tree) return;
    shell();
    const onPath = pathIds(), onlyPath = false;
    const W = PAD * 2 + tree.maxDepth * COL + NW, H = PAD * 2 + Math.max(tree.leaves - 1, 0) * ROW + NH;
    const X = (n) => PAD + n.depth * COL, Y = (n) => PAD + n.row * ROW;
    let svg = `<defs><filter id="lin-glow" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>`;
    const edgeSvg = (e, hot) => {
      const x1 = X(e.from) + NW, y1 = Y(e.from) + NH / 2, x2 = X(e.to), y2 = Y(e.to) + NH / 2, mx = (x1 + x2) / 2;
      const dim = onlyPath && hasEnd && !hot;
      return `<g class="le ${hot ? "hot" : ""} ${dim ? "dim" : ""} ${e.orphan ? "orphan" : ""}"><path d="M${x1} ${y1} C${mx} ${y1} ${mx} ${y2} ${x2} ${y2}" />
        <text x="${(x1 + x2) / 2}" y="${(y1 + y2) / 2 - 5}" text-anchor="middle">${esc(e.label.length > 20 ? e.label.slice(0, 19) + "…" : e.label)}<title>${esc(e.label)}</title></text></g>`;
    };
    // cold edges first, hot ones on top
    for (const e of tree.edges) if (!onPath.has(e.to.id)) svg += edgeSvg(e, false);
    for (const e of tree.edges) if (onPath.has(e.to.id)) svg += edgeSvg(e, true);
    for (const n of tree.nodes) {
      const x = X(n), y = Y(n), hot = hasEnd && onPath.has(n.id), dim = onlyPath && hasEnd && !hot;
      const col = n.composite == null ? "var(--muted)" : scoreColor(n.composite);
      const label = n.id === "seed" ? "seed" : n.composite == null ? "" : n.composite.toFixed(2);
      svg += `<g class="ln ${hot ? "hot" : ""} ${dim ? "dim" : ""} ${n.id === current ? "cur" : ""} ${n.crossover ? "xo" : ""}" data-id="${esc(n.id)}" tabindex="0" role="button" aria-label="${esc((n.edit?.name || "Seed") + " " + label)}" transform="translate(${x} ${y})">
        <rect class="halo" x="-3" y="-3" width="${NW + 6}" height="${NH + 6}" rx="12"/>
        <rect class="box" width="${NW}" height="${NH}" rx="10" style="stroke:${col}"/>
        <g class="thumb" style="color:var(--fg)">${molSvg(n.block, { w: NW - 6, h: NH - 16, pad: 3 }).replace("<svg ", '<svg x="3" y="2" ')}</g>
        <text class="sc" x="${NW / 2}" y="${NH - 5}" text-anchor="middle" style="fill:${col}">${label}</text>
        <title>${esc((n.edit?.name || "Seed molecule") + " · " + n.smiles + (n.composite != null ? " · composite " + n.composite.toFixed(3) : ""))}</title></g>`;
    }
    const sv = mount.querySelector("#lin-svg");
    sv.setAttribute("viewBox", `0 0 ${W} ${H}`);
    sv.setAttribute("width", W);
    sv.setAttribute("height", H);
    sv.innerHTML = svg;
    const total = store.candidates.filter((c) => c.accepted).length;
    mount.querySelector("#lin-sub").textContent = hasEnd && !showAll ? `winning path · ${tree.nodes.length - 1} edit${tree.nodes.length === 2 ? "" : "s"} (of ${total} accepted)` : `${tree.nodes.length - 1} accepted edit${tree.nodes.length === 2 ? "" : "s"} · depth ${tree.maxDepth}`;
    centreOnCurrent(false);
  });

  function centreOnCurrent(smooth = true) {
    const sc = mount.querySelector("#lin-scroll");
    const n = tree?.byId.get(current);
    if (!sc || !n) return;
    const x = PAD + n.depth * COL + NW / 2, y = PAD + n.row * ROW + NH / 2;
    sc.scrollTo({ left: Math.max(0, x - sc.clientWidth / 2), top: Math.max(0, y - sc.clientHeight / 2), behavior: smooth ? "smooth" : "auto" });
  }

  bus.on("run:start", () => { current = "seed"; draw(); });
  bus.on("run:candidate", () => draw());
  bus.on("run:end", () => draw());
  bus.on("view:keyframe", ({ keyframe }) => {
    current = keyframe?.id || "seed";
    mount.querySelectorAll(".ln.cur").forEach((n) => n.classList.remove("cur"));
    const g = [...mount.querySelectorAll(".ln")].find((n) => n.dataset.id === current);
    if (g) { g.classList.add("cur"); centreOnCurrent(); }
  });
  if (store.seed) draw();
  return { draw };
}
