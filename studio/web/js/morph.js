// Edit morph: builds a "union molecule" from parent + child (via atom_map) and evaluates it at progress p in [0, 1].
//  - mapped atoms interpolate between parent and (superposed) child coordinates
//  - removed atoms shrink, fade and tint red while collapsing towards the atom they hung off
//  - added atoms grow out of their attachment atom with a green glow
// Glow is not a real bloom (3Dmol has none): this module reports glow amounts, the stage draws them
// as additive radial gradients on a 2D canvas on top of the WebGL view.
import { parseMolblock, superpose } from "./molparse.js";

export const ELEM_COLORS = {
  C: 0xc9d4e0, H: 0xffffff, N: 0x5b8cff, O: 0xff6b6b, S: 0xffd23f, P: 0xff9f43,
  F: 0x7ee081, Cl: 0x3ddc84, Br: 0xc0713a, I: 0x9b59b6, B: 0xffb6a3, Si: 0xb8a58a,
};
const DEFAULT_COLOR = 0xd0a8ff;
// vdW-ish radii scaled for a ball-and-stick look (Angstrom).
const ELEM_RADIUS = { C: 0.42, H: 0.28, N: 0.42, O: 0.42, S: 0.52, P: 0.5, F: 0.38, Cl: 0.5, Br: 0.56, I: 0.62 };
export const RED = [1.0, 0.23, 0.19];
export const GREEN = [0.2, 0.96, 0.63];

const rgb = (hex) => [((hex >> 16) & 255) / 255, ((hex >> 8) & 255) / 255, (hex & 255) / 255];
export const elemRGB = (e) => rgb(ELEM_COLORS[e] ?? DEFAULT_COLOR);
export const elemRadius = (e) => ELEM_RADIUS[e] ?? 0.45;
export const hexOf = (c) => (Math.round(c[0] * 255) << 16) | (Math.round(c[1] * 255) << 8) | Math.round(c[2] * 255);

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const smooth = (t) => t * t * t * (t * (t * 6 - 15) + 10);
const win = (p, a, b) => clamp01((p - a) / (b - a));
const easeOutBack = (t) => { const c1 = 1.9, c3 = c1 + 1; return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2); };
const lerp = (a, b, t) => a + (b - a) * t;
const mix3 = (out, a, b, t) => { out[0] = lerp(a[0], b[0], t); out[1] = lerp(a[1], b[1], t); out[2] = lerp(a[2], b[2], t); return out; };

function nearestRoots(n, adj, isSource) {
  // multi-source BFS: root[i] = nearest source node
  const root = new Array(n).fill(-1);
  const queue = [];
  for (let i = 0; i < n; i++) if (isSource(i)) { root[i] = i; queue.push(i); }
  for (let h = 0; h < queue.length; h++) {
    const i = queue[h];
    for (const j of adj[i]) if (root[j] < 0) { root[j] = root[i]; queue.push(j); }
  }
  return root;
}

export class Morph {
  /** parentBlock may be null (static view of a single molecule, e.g. the seed). */
  constructor({ parentBlock, childBlock, atomMap = [], refAtoms = null }) {
    const child = parseMolblock(childBlock);
    const parent = parentBlock ? parseMolblock(parentBlock) : null;
    this.static = !parent;
    this.nodes = [];
    this.childAtoms = child.atoms; // final pose (aligned to the parent frame), for chaining the next edit
    this.parentToU = [];
    this.childToU = [];

    if (this.static) {
      child.atoms.forEach((a, c) => {
        const u = this.nodes.length;
        this.nodes.push({ kind: "mapped", pIdx: c, cIdx: c, pElem: a.elem, cElem: a.elem, pPos: a, cPos: a, root: u });
        this.parentToU[c] = u; this.childToU[c] = u;
      });
      this.rmsd = null;
    } else {
      const pairs = [];
      const seenP = new Set(), seenC = new Set();
      for (const [p, c] of atomMap) {
        if (p >= 0 && p < parent.atoms.length && c >= 0 && c < child.atoms.length && !seenP.has(p) && !seenC.has(c)) {
          pairs.push([p, c]); seenP.add(p); seenC.add(c);
        }
      }
      // keep the camera frame continuous: if this edit starts from the molecule the previous edit produced
      // (same atom count), rigidly move both conformers onto that pose
      if (refAtoms && refAtoms.length === parent.atoms.length) {
        superpose(refAtoms, parent.atoms, parent.atoms.map((_, i) => [i, i]), child.atoms);
      }
      // child -> parent frame (the engine aligns them, but we do not rely on it)
      this.rmsd = superpose(parent.atoms, child.atoms, pairs.map(([p, c]) => [c, p]));
      for (const [p, c] of pairs) {
        const u = this.nodes.length;
        this.nodes.push({ kind: "mapped", pIdx: p, cIdx: c, pElem: parent.atoms[p].elem, cElem: child.atoms[c].elem, pPos: parent.atoms[p], cPos: child.atoms[c] });
        this.parentToU[p] = u; this.childToU[c] = u;
      }
      parent.atoms.forEach((a, p) => {
        if (seenP.has(p)) return;
        const u = this.nodes.length;
        this.nodes.push({ kind: "removed", pIdx: p, cIdx: -1, pElem: a.elem, cElem: a.elem, pPos: a, cPos: a });
        this.parentToU[p] = u;
      });
      child.atoms.forEach((a, c) => {
        if (seenC.has(c)) return;
        const u = this.nodes.length;
        this.nodes.push({ kind: "added", pIdx: -1, cIdx: c, pElem: a.elem, cElem: a.elem, pPos: a, cPos: a });
        this.childToU[c] = u;
      });
    }

    const n = this.nodes.length;
    // union bonds
    const bmap = new Map();
    const key = (a, b) => (a < b ? `${a}-${b}` : `${b}-${a}`);
    if (!this.static) {
      for (const b of parent.bonds) {
        const a = this.parentToU[b.a], c = this.parentToU[b.b];
        if (a == null || c == null) continue;
        bmap.set(key(a, c), { a, b: c, pOrder: b.order, cOrder: 0 });
      }
    }
    for (const b of child.bonds) {
      const a = this.childToU[b.a], c = this.childToU[b.b];
      if (a == null || c == null) continue;
      const k = key(a, c);
      const e = bmap.get(k);
      if (e) e.cOrder = b.order;
      else bmap.set(k, { a, b: c, pOrder: this.static ? b.order : 0, cOrder: b.order });
    }
    this.bonds = [...bmap.values()];

    // adjacency per side for root search
    const adjP = Array.from({ length: n }, () => []);
    const adjC = Array.from({ length: n }, () => []);
    const adjAll = Array.from({ length: n }, () => []);
    for (const b of this.bonds) {
      if (b.pOrder) { adjP[b.a].push(b.b); adjP[b.b].push(b.a); }
      if (b.cOrder) { adjC[b.a].push(b.b); adjC[b.b].push(b.a); }
      adjAll[b.a].push(b.b); adjAll[b.b].push(b.a);
    }
    const mapped = (i) => this.nodes[i].kind === "mapped";
    const rootP = nearestRoots(n, adjP, mapped);
    const rootC = nearestRoots(n, adjC, mapped);
    this.nodes.forEach((nd, i) => { nd.root = nd.kind === "removed" ? rootP[i] : nd.kind === "added" ? rootC[i] : i; });

    // Kekule flips on ring bonds should not flicker: keep the parent's order when both are 1/2 and the bond is in a ring.
    const connectedWithout = (b) => {
      const seen = new Set([b.a]); const st = [b.a];
      while (st.length) {
        const i = st.pop();
        for (const j of adjAll[i]) {
          if ((i === b.a && j === b.b) || (i === b.b && j === b.a)) continue;
          if (j === b.b) return true;
          if (!seen.has(j)) { seen.add(j); st.push(j); }
        }
      }
      return false;
    };
    for (const b of this.bonds) {
      b.keepOrder = b.pOrder && b.cOrder && b.pOrder !== b.cOrder && b.pOrder <= 2 && b.cOrder <= 2 && connectedWithout(b);
    }

    this.cur = this.nodes.map(() => ({
      x: 0, y: 0, z: 0, scale: 1, alpha: 1, radius: 0.4,
      base: [1, 1, 1], tint: [1, 1, 1], tintAmt: 0, glowG: 0, glowR: 0,
    }));
    this.bondCur = this.bonds.map(() => ({ on: false, order: 1 }));
    this._baseA = this.nodes.map((nd) => elemRGB(nd.pElem));
    this._baseB = this.nodes.map((nd) => elemRGB(nd.cElem));
    this.counts = {
      mapped: this.nodes.filter((x) => x.kind === "mapped").length,
      removed: this.nodes.filter((x) => x.kind === "removed").length,
      added: this.nodes.filter((x) => x.kind === "added").length,
    };
  }

  /** Evaluate at progress p; `keepGlow` leaves a faint green glow on added atoms once settled. Returns {atoms, bonds}. */
  frame(p, { keepGlow = true } = {}) {
    const { nodes, cur } = this;
    // pass 1: mapped atoms first (others hang off them)
    for (let i = 0; i < nodes.length; i++) {
      const nd = nodes[i];
      if (nd.kind !== "mapped") continue;
      const o = cur[i];
      const w = this.static ? 1 : smooth(win(p, 0.28, 0.82));
      o.x = lerp(nd.pPos.x, nd.cPos.x, w); o.y = lerp(nd.pPos.y, nd.cPos.y, w); o.z = lerp(nd.pPos.z, nd.cPos.z, w);
      o.scale = 1; o.alpha = 1; o.glowR = 0; o.glowG = 0; o.tintAmt = 0;
      if (nd.pElem !== nd.cElem) {
        const e = smooth(win(p, 0.4, 0.7));
        mix3(o.base, this._baseA[i], this._baseB[i], e);
        o.radius = lerp(elemRadius(nd.pElem), elemRadius(nd.cElem), e);
        o.glowG = Math.sin(Math.PI * win(p, 0.3, 0.85)) * 0.8; // changed element
        o.tint = GREEN; o.tintAmt = 0;
      } else {
        o.base = this._baseA[i];
        o.radius = elemRadius(nd.pElem);
      }
    }
    // pass 2: removed / added
    for (let i = 0; i < nodes.length; i++) {
      const nd = nodes[i];
      if (nd.kind === "mapped") continue;
      const o = cur[i];
      const r = nd.root >= 0 ? cur[nd.root] : null;
      const rn = nd.root >= 0 ? nodes[nd.root] : null;
      if (nd.kind === "removed") {
        const e = smooth(win(p, 0.04, 0.5));
        const k = 1 - 0.55 * e;
        if (r) {
          o.x = r.x + (nd.pPos.x - rn.pPos.x) * k; o.y = r.y + (nd.pPos.y - rn.pPos.y) * k; o.z = r.z + (nd.pPos.z - rn.pPos.z) * k;
        } else { o.x = nd.pPos.x; o.y = nd.pPos.y; o.z = nd.pPos.z; }
        o.scale = 1 - e; o.alpha = 1 - e * 0.9;
        o.base = this._baseA[i]; o.radius = elemRadius(nd.pElem);
        o.tint = RED; o.tintAmt = smooth(win(p, 0, 0.22));
        o.glowR = smooth(win(p, 0, 0.12)) * (1 - smooth(win(p, 0.38, 0.62)));
        o.glowG = 0;
      } else {
        const w = win(p, 0.45, 0.95);
        const pos = smooth(w);
        if (r) {
          o.x = r.x + (nd.cPos.x - rn.cPos.x) * pos; o.y = r.y + (nd.cPos.y - rn.cPos.y) * pos; o.z = r.z + (nd.cPos.z - rn.cPos.z) * pos;
        } else { o.x = nd.cPos.x; o.y = nd.cPos.y; o.z = nd.cPos.z; }
        o.scale = Math.max(0, easeOutBack(w)) * (w > 0 ? 1 : 0);
        o.alpha = smooth(clamp01(w * 2.5));
        o.base = this._baseB[i]; o.radius = elemRadius(nd.cElem);
        o.tint = GREEN; o.tintAmt = 1 - smooth(win(p, 0.62, 1));
        o.glowG = smooth(win(p, 0.45, 0.62)) * (1 - (keepGlow ? 0.65 : 1) * smooth(win(p, 0.8, 1)));
        o.glowR = 0;
      }
    }
    // bonds
    const bc = this.bondCur;
    for (let k = 0; k < this.bonds.length; k++) {
      const b = this.bonds[k], o = bc[k];
      const A = cur[b.a], B = cur[b.b];
      let on, order;
      if (b.pOrder && b.cOrder) { on = true; order = b.keepOrder || p < 0.5 ? b.pOrder : b.cOrder; }
      else if (b.pOrder) { on = p < 0.5; order = b.pOrder; }
      else { on = p >= 0.5; order = b.cOrder; }
      o.on = on && A.scale > 0.06 && B.scale > 0.06;
      o.order = order || 1;
    }
    return { atoms: cur, bonds: bc };
  }
}

/** Centroid + radius of every union position (parent and child), for stable framing. */
export function unionBounds(morph) {
  const pts = [];
  for (const nd of morph.nodes) { pts.push(nd.pPos); pts.push(nd.cPos); }
  const c = { x: 0, y: 0, z: 0 };
  for (const p of pts) { c.x += p.x; c.y += p.y; c.z += p.z; }
  c.x /= pts.length; c.y /= pts.length; c.z /= pts.length;
  let r = 0;
  for (const p of pts) r = Math.max(r, Math.hypot(p.x - c.x, p.y - c.y, p.z - c.z));
  return { center: c, radius: r };
}
