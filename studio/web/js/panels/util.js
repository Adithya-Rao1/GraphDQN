// Shared helpers for the analysis panels (story, lineage, charts, map) and manual mode.
import { judge } from "../whycard.js";
import { ELEM_COLORS } from "../morph.js";

export { judge };
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

export const LABEL = {
  composite: "Composite score", qed: "QED", logp: "logP", tpsa: "TPSA", mw: "Mol. weight", hbd: "H-bond donors", hba: "H-bond acceptors",
  rotb: "Rotatable bonds", fsp3: "Fsp3", sa: "SA score", lipinski_violations: "Lipinski violations", alerts: "Structural alerts", heavy_atoms: "Heavy atoms",
};
export const DIGITS = { composite: 2, qed: 2, logp: 2, tpsa: 1, mw: 1, fsp3: 2, sa: 2 };
export const fmt = (k, v) => (v == null || Number.isNaN(v) ? "n/a" : Number(v).toFixed(DIGITS[k] ?? 0));
export const fmtSigned = (k, d) => `${d > 0 ? "+" : "−"}${Math.abs(d).toFixed(DIGITS[k] ?? 0)}`;

// Predicted-binding keys a ScoreResult.extras may carry (the contract comment says plapt_pkd, the pivot brief plapt_affinity).
export const BINDING_KEYS = ["plapt_affinity", "plapt_pkd"];
export function bindingOf(score) {
  const ex = score?.extras;
  if (!ex) return null;
  for (const k of BINDING_KEYS) if (typeof ex[k] === "number") return { key: k, value: ex[k] };
  return null;
}

export const elemCss = (e) => "#" + (ELEM_COLORS[e] ?? 0xd0a8ff).toString(16).padStart(6, "0");

/** Composite score (0..1) -> a colour on a red-amber-green ramp that reads on dark and light panels. */
export function scoreColor(v) {
  const t = Math.min(Math.max(v ?? 0, 0), 1);
  const stops = [[255, 107, 99], [255, 176, 32], [52, 245, 161]];
  const x = t * 2, i = Math.min(Math.floor(x), 1), f = x - i;
  const c = stops[i].map((a, k) => Math.round(a + (stops[i + 1][k] - a) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

/** Tween a number into an element's text. */
export function countUp(node, to, { from = 0, ms = 900, digits = 2, signed = false, delay = 0 } = {}) {
  if (!node) return;
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) { node.textContent = fmtNum(to, digits, signed); return; }
  const t0 = performance.now() + delay;
  const ease = (t) => 1 - Math.pow(1 - t, 3);
  const step = (now) => {
    const t = Math.min(Math.max((now - t0) / ms, 0), 1);
    node.textContent = fmtNum(from + (to - from) * ease(t), digits, signed);
    if (t < 1) requestAnimationFrame(step);
  };
  node.textContent = fmtNum(from, digits, signed);
  requestAnimationFrame(step);
}
const fmtNum = (v, digits, signed) => (signed ? (v > 0.0000001 ? "+" : v < -0.0000001 ? "−" : "") + Math.abs(v).toFixed(digits) : v.toFixed(digits));

export const isMobile = () => matchMedia("(max-width: 1000px)").matches;

let toastEl = null, toastTimer = 0;
export function toast(msg, ms = 2600) {
  if (!toastEl) {
    toastEl = document.createElement("div");
    toastEl.className = "metis-toast";
    toastEl.setAttribute("role", "status");
    document.body.appendChild(toastEl);
  }
  toastEl.textContent = msg;
  toastEl.classList.add("on");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove("on"), ms);
}

/** Resolve the timeline keyframe index for a candidate id ("seed" for the seed); -1 when it is filtered out. */
export function frameIndex(timeline, id) {
  return timeline.keyframes.findIndex((k) => k.id === id);
}

/** Run `fn` at most once per animation frame. */
export function rafThrottle(fn) {
  let queued = false;
  return (...a) => { if (queued) return; queued = true; requestAnimationFrame(() => { queued = false; fn(...a); }); };
}
