// Per-atom overlays evaluated on the union molecule of a morph:
//  - colour by Crippen logP contribution (hydrophobicity) or Gasteiger charge, blended parent -> child
//  - structural-alert halos (red): alerts on parent atoms fade out, alerts on child atoms fade in
import { store } from "./state.js";

const lerp = (a, b, t) => a + (b - a) * t;
const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const smooth = (t) => t * t * t * (t * (t * 6 - 15) + 10);
const win = (p, a, b) => clamp01((p - a) / (b - a));
const hex = (h) => [((h >> 16) & 255) / 255, ((h >> 8) & 255) / 255, (h & 255) / 255];

// diverging scales (neutral in the middle), values clamped to +-SPAN
export const SCALES = {
  logp: { key: "crippen_logp", span: 0.5, neg: hex(0x3b82f6), mid: hex(0xd7dde6), pos: hex(0xff8a1f), negLabel: "hydrophilic", posLabel: "hydrophobic", title: "Crippen logP contribution" },
  charge: { key: "gasteiger", span: 0.5, neg: hex(0xff4d5a), mid: hex(0xd7dde6), pos: hex(0x4f8cff), negLabel: "negative", posLabel: "positive", title: "Gasteiger partial charge" },
};
export const NEUTRAL = hex(0x5b6573);

export function scaleColor(scale, v, out = [0, 0, 0]) {
  const t = Math.max(-1, Math.min(1, v / scale.span));
  const a = scale.mid, b = t < 0 ? scale.neg : scale.pos, k = Math.abs(t);
  out[0] = lerp(a[0], b[0], k); out[1] = lerp(a[1], b[1], k); out[2] = lerp(a[2], b[2], k);
  return out;
}

export function scaleCss(scale) {
  const c = (rgb) => `rgb(${rgb.map((v) => Math.round(v * 255)).join(",")})`;
  return `linear-gradient(90deg, ${c(scale.neg)}, ${c(scale.mid)}, ${c(scale.pos)})`;
}

/** Atom data for the parent/child sides of a keyframe: {parent: AtomContributions|null, child: ...|null}. */
export function contributionsFor(kf) {
  if (kf.kind === "seed") return { parent: null, child: kf.seedContributions || null };
  const c = kf.candidate;
  return {
    parent: store.bySmiles.get(c.parent_smiles)?.contributions || null,
    child: c.explanation?.atom_contributions || null,
  };
}

/**
 * Fill `out[i]` with an rgb colour for each union atom, or return null when the data for this keyframe
 * is unavailable. `info.approx` is set when the parent side had to be approximated by child values.
 */
export function atomColors(mode, kf, p, out, info = {}) {
  const scale = SCALES[mode];
  if (!scale) return null;
  const { parent, child } = contributionsFor(kf);
  const P = parent?.[scale.key], C = child?.[scale.key];
  if (!P && !C) return null;
  info.approx = !P || !C;
  const w = kf.kind === "seed" ? 1 : smooth(win(p, 0.28, 0.82));
  kf.morph.nodes.forEach((nd, i) => {
    const pv = P ? P[nd.pIdx] : undefined;
    const cv = C ? C[nd.cIdx] : undefined;
    let v;
    if (nd.kind === "removed") v = pv;
    else if (nd.kind === "added") v = cv;
    else v = lerp(pv ?? cv ?? 0, cv ?? pv ?? 0, w);
    out[i] = v == null ? NEUTRAL.slice() : scaleColor(scale, v, out[i] || [0, 0, 0]);
  });
  return out;
}

/** Union-atom flags for alert halos: {parent: Uint8Array, child: Uint8Array, names: {...}}; built once per keyframe. */
export function alertMasks(kf) {
  const n = kf.morph.nodes.length;
  const parent = new Uint8Array(n), child = new Uint8Array(n);
  const names = { parent: [], child: [] };
  const mark = (arr, hits, toU, bucket) => {
    for (const h of hits || []) {
      bucket.push(h.name);
      for (const a of h.atom_indices || []) if (toU[a] != null) arr[toU[a]] = 1;
    }
  };
  const m = kf.morph;
  if (kf.kind === "seed") {
    mark(child, store.seed?.score?.alerts, m.childToU, names.child);
    parent.set(child);
  } else {
    const c = kf.candidate;
    const known = store.bySmiles.get(c.parent_smiles)?.alerts;
    // parent alerts: the parent's own score if we saw it, else at least the ones this edit removed
    mark(parent, known ?? c.explanation?.alerts_removed, m.parentToU, names.parent);
    mark(child, c.score?.alerts, m.childToU, names.child);
  }
  return { parent, child, names };
}

/** Halo strength 0..1 for union atom i at progress p. */
export function haloAt(masks, i, p) {
  const w = smooth(win(p, 0.4, 0.8));
  return Math.max(masks.parent[i] * (1 - w), masks.child[i] * w);
}
