// Tiny 2D depiction of a 3D mol block for thumbnails: project onto the plane of largest extent (PCA), draw bonds and
// coloured heteroatoms as inline SVG. Not a structure-diagram layout: it is the 3D conformer seen face-on, which is
// exactly what the 3D stage shows. No dependencies, no network.
import { parseMolblock } from "../molparse.js";
import { elemCss } from "./util.js";

const cache = new Map();

function top2Axes(pts) {
  const c = [0, 0, 0];
  for (const p of pts) { c[0] += p.x; c[1] += p.y; c[2] += p.z; }
  c.forEach((_, i) => (c[i] /= pts.length));
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const p of pts) {
    const v = [p.x - c[0], p.y - c[1], p.z - c[2]];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[i][j] += v[i] * v[j];
  }
  const mul = (M, v) => M.map((r) => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);
  const norm = (v) => { const n = Math.hypot(...v) || 1; return v.map((a) => a / n); };
  let a = norm([1, 0.3, 0.1]);
  for (let i = 0; i < 40; i++) a = norm(mul(C, a));
  // deflate and find the second axis
  const lam = mul(C, a).reduce((s, x, i) => s + x * a[i], 0);
  const D = C.map((r, i) => r.map((x, j) => x - lam * a[i] * a[j]));
  let b = norm([0.2, 1, 0.4]);
  for (let i = 0; i < 40; i++) b = norm(mul(D, b));
  const dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  b = norm(b.map((x, i) => x - dot * a[i]));
  return { c, a, b };
}

/** SVG markup (a <svg> element string) of the molecule, fitted into w x h. */
export function molSvg(molblock, { w = 64, h = 44, pad = 4, stroke = "currentColor", highlight = null } = {}) {
  const key = molblock + `|${w}|${h}|${stroke}|${highlight ? [...highlight].join(",") : ""}`;
  if (cache.has(key)) return cache.get(key);
  let svg = "";
  try {
    const { atoms, bonds } = parseMolblock(molblock);
    const { c, a, b } = top2Axes(atoms);
    const P = atoms.map((p) => {
      const v = [p.x - c[0], p.y - c[1], p.z - c[2]];
      return { x: v[0] * a[0] + v[1] * a[1] + v[2] * a[2], y: -(v[0] * b[0] + v[1] * b[1] + v[2] * b[2]), z: v[0] * (a[1] * b[2] - a[2] * b[1]) + v[1] * (a[2] * b[0] - a[0] * b[2]) + v[2] * (a[0] * b[1] - a[1] * b[0]) };
    });
    const xs = P.map((p) => p.x), ys = P.map((p) => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    const s = Math.min((w - 2 * pad) / Math.max(maxX - minX, 1), (h - 2 * pad) / Math.max(maxY - minY, 1));
    const ox = (w - (maxX - minX) * s) / 2 - minX * s, oy = (h - (maxY - minY) * s) / 2 - minY * s;
    const X = (i) => (P[i].x * s + ox).toFixed(1), Y = (i) => (P[i].y * s + oy).toFixed(1);
    const sw = Math.max(1, Math.min(2, s * 0.12));
    let body = "";
    for (const bd of bonds) {
      if (bd.order >= 2) {
        const dx = P[bd.b].x - P[bd.a].x, dy = P[bd.b].y - P[bd.a].y, L = Math.hypot(dx, dy) || 1;
        const nx = (-dy / L) * sw * 0.9, ny = (dx / L) * sw * 0.9;
        for (const sgn of bd.order === 2 ? [-1, 1] : [-1.6, 0, 1.6]) {
          body += `<line x1="${(+X(bd.a) + nx * sgn).toFixed(1)}" y1="${(+Y(bd.a) + ny * sgn).toFixed(1)}" x2="${(+X(bd.b) + nx * sgn).toFixed(1)}" y2="${(+Y(bd.b) + ny * sgn).toFixed(1)}"/>`;
        }
      } else body += `<line x1="${X(bd.a)}" y1="${Y(bd.a)}" x2="${X(bd.b)}" y2="${Y(bd.b)}"/>`;
    }
    let dots = "";
    atoms.forEach((at, i) => {
      const hl = highlight && highlight.has(i);
      if (at.elem !== "C" || hl) dots += `<circle cx="${X(i)}" cy="${Y(i)}" r="${hl ? sw * 2.3 : sw * 1.7}" fill="${hl ? "#34f5a1" : elemCss(at.elem)}"/>`;
    });
    svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" class="mol2d" aria-hidden="true"><g stroke="${stroke}" stroke-width="${sw.toFixed(2)}" stroke-linecap="round" opacity=".85">${body}</g>${dots}</svg>`;
  } catch (err) {
    svg = `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" class="mol2d"><text x="${w / 2}" y="${h / 2}" text-anchor="middle" font-size="9" fill="currentColor">?</text></svg>`;
  }
  if (cache.size > 400) cache.clear();
  cache.set(key, svg);
  return svg;
}

/** Inner-markup variant for embedding inside another <svg> at (x, y). */
export function molGroup(molblock, x, y, opts) {
  const s = molSvg(molblock, opts);
  return s.replace("<svg ", `<svg x="${x}" y="${y}" `);
}
