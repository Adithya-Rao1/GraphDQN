// The 3D stage: one 3Dmol.js viewer holding the "union molecule" of the current keyframe, plus a 2D canvas on top
// for glow / alert halos / lock rings (3Dmol has no bloom). The molecule is re-styled every frame while a morph runs.
import { bus } from "./bus.js";
import { store } from "./state.js";
import { hexOf } from "./morph.js";
import { atomColors, alertMasks, haloAt, SCALES } from "./overlays.js";

const STICK_R = 0.13;
const SPIN_DEG_PER_S = 9;
const BG = [0.04, 0.06, 0.1]; // approx. colour of the dark stage, used to fade atoms out

export class Stage {
  constructor({ host, overlay }) {
    this.host = host;
    this.overlay = overlay;
    this.ctx = overlay.getContext("2d");
    if (!window.$3Dmol) throw new Error("3Dmol.js failed to load");
    this.viewer = window.$3Dmol.createViewer(host, { backgroundAlpha: 0, antialias: true });
    this.kf = null;
    this.morph = null;
    this.p = 0;
    this.model = null;
    this.colors = [];
    this.masks = null;
    this.colorInfo = {};
    this.hover = -1;
    this.dirtyMol = true;
    this.dragging = false;
    this.lastFrame = performance.now();
    this.perf = { frames: [], renderMs: [], fps: 0, avgMs: 0, worstMs: 0, rebuilds: 0 };
    this._tmp = [0, 0, 0];

    host.addEventListener("pointerdown", () => { this.dragging = true; });
    window.addEventListener("pointerup", () => { this.dragging = false; });
    new ResizeObserver(() => this.resize()).observe(host);
    requestAnimationFrame((n) => this._loop(n));
  }

  resize() {
    const r = this.host.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.overlay.width = Math.max(1, Math.round(r.width * dpr));
    this.overlay.height = Math.max(1, Math.round(r.height * dpr));
    this.overlay.style.width = r.width + "px";
    this.overlay.style.height = r.height + "px";
    this.viewer.resize();
    if (this.model) this.fit();
    this.dirtyMol = true;
  }

  /** Show a keyframe (its Morph is already attached as kf.morph). */
  setKeyframe(kf, { zoom = true } = {}) {
    this.kf = kf;
    this.morph = kf.morph;
    this.viewer.removeAllModels();
    this.viewer.removeAllLabels?.();
    this.model = this.viewer.addModel();
    this.model.addAtoms(this.morph.nodes.map((nd, i) => ({
      elem: nd.cElem, x: nd.pPos.x, y: nd.pPos.y, z: nd.pPos.z, serial: i, index: i,
    })));
    this.viewer.setClickable({}, true, (atom) => {
      if (atom && this.kf?.kind === "seed") bus.emit("stage:atomclick", { index: atom.index });
    });
    this.viewer.setHoverable({}, true,
      (atom) => { if (this.kf?.kind === "seed" && atom) { this.hover = atom.index; this.host.style.cursor = "pointer"; } },
      () => { this.hover = -1; this.host.style.cursor = ""; });
    this.masks = alertMasks(kf);
    this.dirtyMol = true;
    this.p = kf.kind === "seed" ? 1 : 0;
    this._applyFrame();
    if (zoom) this.fit();
    this.viewer.render();
  }

  /** Frame the whole union molecule; narrower stages (phones, portrait) need more headroom. */
  fit() {
    const r = this.host.getBoundingClientRect();
    const aspect = r.width / Math.max(r.height, 1);
    this.viewer.zoomTo({}, 0);
    this.viewer.zoom(Math.min(1.25, 0.8 * aspect), 0);
  }

  /** Progress 0..1 of the current morph. */
  setProgress(p) {
    if (p !== this.p) { this.p = p; this.dirtyMol = true; }
  }

  invalidate() { this.dirtyMol = true; }

  _applyFrame() {
    if (!this.model || !this.morph) return;
    const t0 = performance.now();
    const p = this.p;
    const { atoms: cur, bonds: bcur } = this.morph.frame(p, { keepGlow: store.overlay.highlightEdit });
    const mode = store.overlay.mode;
    let cols = null;
    this.colorInfo = {};
    if (mode !== "element") cols = atomColors(mode, this.kf, p, this.colors, this.colorInfo);
    this.colorInfo.available = mode === "element" || !!cols;
    const atoms = this.model.atoms;
    for (let i = 0; i < atoms.length; i++) {
      const a = atoms[i], o = cur[i];
      a.x = o.x; a.y = o.y; a.z = o.z;
      a.bonds = []; a.bondOrder = [];
      if (o.scale < 0.04 || o.alpha < 0.03) { a.style = {}; continue; }
      let c = cols ? cols[i] : o.base;
      if (o.tintAmt > 0.001) {
        const t = o.tintAmt, k = this._tmp;
        k[0] = c[0] + (o.tint[0] - c[0]) * t; k[1] = c[1] + (o.tint[1] - c[1]) * t; k[2] = c[2] + (o.tint[2] - c[2]) * t;
        c = k;
      }
      if (o.alpha < 0.999) {
        // 3Dmol cannot vary opacity per atom inside one model ("opacity is ambiguous"), so fade towards the stage background instead
        const k = this._tmp, f = o.alpha;
        k[0] = c[0] * f + BG[0] * (1 - f); k[1] = c[1] * f + BG[1] * (1 - f); k[2] = c[2] * f + BG[2] * (1 - f);
        c = k;
      }
      const col = hexOf(c);
      a.color = col;
      a.style = {
        sphere: { radius: o.radius * o.scale, color: col },
        stick: { radius: STICK_R * Math.min(1, o.scale + 0.15), color: col },
      };
    }
    for (let k = 0; k < bcur.length; k++) {
      if (!bcur[k].on) continue;
      const b = this.morph.bonds[k], A = atoms[b.a], B = atoms[b.b];
      A.bonds.push(b.b); A.bondOrder.push(bcur[k].order);
      B.bonds.push(b.a); B.bondOrder.push(bcur[k].order);
    }
    this.model.molObj = null; // 3Dmol rebuilds the mesh on the next render
    this.dirtyMol = false;
    this._styleMs = performance.now() - t0;
  }

  _loop(now) {
    const dt = Math.min(now - this.lastFrame, 100);
    this.lastFrame = now;
    if (this.model) {
      const t0 = performance.now();
      const spin = store.overlay.autorotate && !this.dragging;
      let rendered = false;
      if (this.dirtyMol) {
        this._applyFrame();
        this.perf.rebuilds++;
      }
      if (spin) { this.viewer.rotate((SPIN_DEG_PER_S * dt) / 1000, "y"); rendered = true; }
      else if (this._needsRender || this.model.molObj === null) { this.viewer.render(); rendered = true; }
      this._needsRender = false;
      if (rendered) {
        const ms = performance.now() - t0;
        const pf = this.perf;
        pf.frames.push(now); pf.renderMs.push(ms);
        while (pf.frames.length && now - pf.frames[0] > 1000) { pf.frames.shift(); pf.renderMs.shift(); }
        pf.fps = pf.frames.length;
        pf.avgMs = pf.renderMs.reduce((s, v) => s + v, 0) / (pf.renderMs.length || 1);
        pf.worstMs = Math.max(...pf.renderMs, 0);
      }
      this._drawOverlay(now);
    }
    requestAnimationFrame((n) => this._loop(n));
  }

  // ------------------------------------------------------------------ 2D overlay (glow / halos / locks)
  _drawOverlay(now) {
    const ctx = this.ctx, dpr = window.devicePixelRatio || 1;
    const W = this.overlay.width / dpr, H = this.overlay.height / dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    if (!this.morph) return;
    const cur = this.morph.cur;
    const pts = this.viewer.modelToScreen(cur.map((o) => ({ x: o.x, y: o.y, z: o.z })));
    const rect = this.host.getBoundingClientRect();
    const ox = rect.left + window.pageXOffset - document.documentElement.clientLeft;
    const oy = rect.top + window.pageYOffset - document.documentElement.clientTop;
    // pixels per Angstrom: the largest screen/3D ratio over bonds (bonds perpendicular to the view)
    let ppa = 0;
    for (const b of this.morph.bonds) {
      const A = cur[b.a], B = cur[b.b];
      const d3 = Math.hypot(A.x - B.x, A.y - B.y, A.z - B.z);
      if (d3 < 0.5) continue;
      ppa = Math.max(ppa, Math.hypot(pts[b.a].x - pts[b.b].x, pts[b.a].y - pts[b.b].y) / d3);
    }
    if (!ppa) ppa = H / 12;
    const kf = this.kf, p = this.p;
    const pulse = 0.5 + 0.5 * Math.sin(now / 260);

    ctx.globalCompositeOperation = "lighter";
    cur.forEach((o, i) => {
      const x = pts[i].x - ox, y = pts[i].y - oy, r0 = Math.max(o.radius * ppa, 6);
      if (o.glowG > 0.01) this._glow(x, y, r0 * (2.4 + 2.2 * o.glowG), [52, 245, 161], 0.6 * o.glowG);
      if (o.glowR > 0.01) this._glow(x, y, r0 * (2.4 + 2.0 * o.glowR), [255, 59, 48], 0.6 * o.glowR);
      if (store.overlay.alerts) {
        const h = haloAt(this.masks, i, p) * Math.max(o.alpha, 0.0);
        if (h > 0.01) this._glow(x, y, r0 * (3.0 + 0.5 * pulse), [255, 59, 48], (0.28 + 0.14 * pulse) * h);
      }
    });
    ctx.globalCompositeOperation = "source-over";

    cur.forEach((o, i) => {
      const x = pts[i].x - ox, y = pts[i].y - oy, r0 = Math.max(o.radius * ppa, 6);
      if (store.overlay.alerts) {
        const h = haloAt(this.masks, i, p) * o.alpha;
        if (h > 0.02) {
          ctx.beginPath();
          ctx.setLineDash([5, 4]);
          ctx.lineDashOffset = -now / 60;
          ctx.strokeStyle = `rgba(255,92,84,${0.95 * h})`;
          ctx.lineWidth = 2.2;
          ctx.arc(x, y, r0 * 1.75, 0, Math.PI * 2);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
      if (kf.kind === "seed") {
        if (store.locked.has(i)) this._lock(x, y, r0, now);
        else if (i === this.hover) {
          ctx.beginPath();
          ctx.strokeStyle = "rgba(255,255,255,.75)";
          ctx.lineWidth = 2;
          ctx.arc(x, y, r0 * 1.7, 0, Math.PI * 2);
          ctx.stroke();
        }
      }
    });
    this._screen = { pts: pts.map((q) => ({ x: q.x - ox, y: q.y - oy })), ppa };
  }

  _glow(x, y, r, rgb, a) {
    const g = this.ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a})`);
    g.addColorStop(0.45, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a * 0.35})`);
    g.addColorStop(1, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0)`);
    this.ctx.fillStyle = g;
    this.ctx.beginPath();
    this.ctx.arc(x, y, r, 0, Math.PI * 2);
    this.ctx.fill();
  }

  _lock(x, y, r0, now) {
    const ctx = this.ctx, R = r0 * 1.9;
    ctx.beginPath();
    ctx.strokeStyle = "#ffb020";
    ctx.lineWidth = 3;
    ctx.arc(x, y, R, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.strokeStyle = "rgba(255,176,32,.35)";
    ctx.lineWidth = 7;
    ctx.arc(x, y, R, 0, Math.PI * 2);
    ctx.stroke();
    // padlock glyph at the upper right of the ring
    const gx = x + R * 0.78, gy = y - R * 0.78, s = Math.max(r0 * 0.55, 7);
    ctx.fillStyle = "#ffb020";
    ctx.strokeStyle = "#ffb020";
    ctx.lineWidth = Math.max(s * 0.28, 1.6);
    ctx.beginPath();
    ctx.arc(gx, gy - s * 0.35, s * 0.55, Math.PI, 0);
    ctx.stroke();
    ctx.fillRect(gx - s * 0.8, gy - s * 0.35, s * 1.6, s * 1.2);
  }

  // ---------------------------------------------------------------------------- export
  /** PNG data URL: WebGL canvas + 2D overlay + caption strip, on the theme background. */
  composeImage({ caption, badge, bg }) {
    this.viewer.render();
    const gl = this.viewer.getCanvas();
    const w = gl.width, h = gl.height, dpr = window.devicePixelRatio || 1;
    const out = document.createElement("canvas");
    const strip = Math.round(64 * dpr);
    out.width = w; out.height = h + strip;
    const c = out.getContext("2d");
    c.fillStyle = bg || "#0b0f17";
    c.fillRect(0, 0, out.width, out.height);
    c.drawImage(gl, 0, 0);
    c.drawImage(this.overlay, 0, 0, w, h);
    c.fillStyle = "rgba(255,255,255,.92)";
    c.font = `600 ${Math.round(20 * dpr)}px -apple-system, Helvetica, Arial, sans-serif`;
    c.textBaseline = "middle";
    c.fillText(caption, 20 * dpr, h + strip / 2);
    if (badge) {
      c.font = `700 ${Math.round(16 * dpr)}px -apple-system, Helvetica, Arial, sans-serif`;
      const tw = c.measureText(badge).width;
      c.fillStyle = "#ffb020";
      c.fillText(badge, w - tw - 20 * dpr, h + strip / 2);
    }
    return out.toDataURL("image/png");
  }
}
export { SCALES };
