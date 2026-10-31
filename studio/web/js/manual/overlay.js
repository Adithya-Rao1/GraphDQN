// 2D overlay on top of the 3D stage for manual mode: atom picking, hover / selection rings, lock rings, "has edits" dots and the
// ghost diff markers. Positions come from the public Stage surface only (stage.viewer.modelToScreen + stage.morph.cur), so it works
// for any keyframe (seed or edit), unlike the stage's own click handler which is seed-only.
export class ManualOverlay {
  constructor({ stage, wrap }) {
    this.stage = stage;
    this.wrap = wrap;
    this.canvas = document.createElement("canvas");
    this.canvas.className = "manual-fx";
    const fx = document.getElementById("stage-fx");
    (fx?.parentNode || wrap).insertBefore(this.canvas, fx ? fx.nextSibling : null);
    this.ctx = this.canvas.getContext("2d");
    this.hover = -1;
    this.selected = [];
    this.locked = new Set();
    this.showLocks = true; // false while the stage itself draws the (seed) locks
    this.density = new Map(); // current-molecule atom index -> number of applicable edits
    this.ghost = null; // {added: [{x,y,z,elem}], removed: Set(current-molecule idx), changed: Set(idx)}
    this.active = false;
    this.pts = null;
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(stage.host);
    this.resize();
  }

  resize() {
    const r = this.stage.host.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.round(r.width * dpr));
    this.canvas.height = Math.max(1, Math.round(r.height * dpr));
    this.canvas.style.width = r.width + "px";
    this.canvas.style.height = r.height + "px";
  }

  start() {
    if (this.active) return;
    this.active = true;
    const loop = (now) => { if (!this.active) return; this.draw(now); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }

  stop() {
    this.active = false;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  /** Screen positions (element-relative px) of every union node + pixels per Angstrom. */
  project() {
    const { viewer, morph, host } = this.stage;
    if (!viewer || !morph) return null;
    const cur = morph.cur;
    const pts = viewer.modelToScreen(cur.map((o) => ({ x: o.x, y: o.y, z: o.z })));
    const rect = host.getBoundingClientRect();
    const ox = rect.left + window.pageXOffset - document.documentElement.clientLeft;
    const oy = rect.top + window.pageYOffset - document.documentElement.clientTop;
    let ppa = 0;
    for (const b of morph.bonds) {
      const A = cur[b.a], B = cur[b.b], d3 = Math.hypot(A.x - B.x, A.y - B.y, A.z - B.z);
      if (d3 < 0.5) continue;
      ppa = Math.max(ppa, Math.hypot(pts[b.a].x - pts[b.b].x, pts[b.a].y - pts[b.b].y) / d3);
    }
    return { pts: pts.map((p) => ({ x: p.x - ox, y: p.y - oy })), ppa: ppa || rect.height / 12, ox, oy };
  }

  /** Current-molecule atom index -> screen {x, y, r}; null when not displayed. */
  atomScreen(i, P = this.project()) {
    const morph = this.stage.morph;
    if (!P || !morph) return null;
    const u = morph.childToU[i];
    if (u == null) return null;
    const o = morph.cur[u];
    return { x: P.pts[u].x, y: P.pts[u].y, r: Math.max(o.radius * P.ppa, 6) };
  }

  /** Index (in the current molecule) of the atom nearest to a client point, or -1. */
  pick(clientX, clientY) {
    const P = this.project(), morph = this.stage.morph;
    if (!P || !morph) return -1;
    const rect = this.stage.host.getBoundingClientRect();
    const x = clientX - rect.left, y = clientY - rect.top;
    let best = -1, bd = Infinity;
    morph.nodes.forEach((nd, u) => {
      if (nd.cIdx < 0 || nd.kind === "removed" || morph.cur[u].scale < 0.5) return;
      const r = Math.max(morph.cur[u].radius * P.ppa * 1.5, 16);
      const d = Math.hypot(P.pts[u].x - x, P.pts[u].y - y);
      if (d < r && d < bd) { bd = d; best = nd.cIdx; }
    });
    return best;
  }

  draw(now) {
    const ctx = this.ctx, dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width / dpr, this.canvas.height / dpr);
    const P = this.project();
    if (!P || !this.stage.morph) return;
    const pulse = 0.5 + 0.5 * Math.sin(now / 320);
    // atoms that have applicable edits: soft dots (bigger = more options)
    ctx.globalCompositeOperation = "lighter";
    for (const [i, n] of this.density) {
      const s = this.atomScreen(i, P);
      if (!s || this.selected.includes(i)) continue;
      this.glow(s.x, s.y, s.r * (1.5 + Math.min(n, 12) * 0.07 + 0.2 * pulse), [94, 161, 255], 0.16 + Math.min(n, 12) * 0.012);
    }
    if (this.ghost) {
      for (const a of this.ghost.screenAdded || []) this.glow(a.x, a.y, 24 + 4 * pulse, [52, 245, 161], 0.5);
      for (const i of this.ghost.removed) { const s = this.atomScreen(i, P); if (s) this.glow(s.x, s.y, s.r * (2.6 + 0.4 * pulse), [255, 59, 48], 0.5); }
      for (const i of this.ghost.changed) { const s = this.atomScreen(i, P); if (s) this.glow(s.x, s.y, s.r * (2.4 + 0.4 * pulse), [255, 176, 32], 0.5); }
    }
    ctx.globalCompositeOperation = "source-over";
    if (this.ghost) {
      for (const i of this.ghost.removed) { const s = this.atomScreen(i, P); if (s) this.ring(s, 1.9, "rgba(255,92,84,.95)", 2.4, [5, 4], now); }
    }
    if (this.showLocks) for (const i of this.locked) { const s = this.atomScreen(i, P); if (s) this.lock(s.x, s.y, s.r); }
    if (this.hover >= 0 && !this.selected.includes(this.hover)) { const s = this.atomScreen(this.hover, P); if (s) this.ring(s, 1.65, "rgba(255,255,255,.8)", 2); }
    this.selected.forEach((i, k) => {
      const s = this.atomScreen(i, P);
      if (!s) return;
      this.ring(s, 1.8 + 0.08 * pulse, "#5ea1ff", 3.2);
      this.ring(s, 2.3 + 0.12 * pulse, "rgba(94,161,255,.35)", 6);
      ctx.fillStyle = "#06101f"; ctx.strokeStyle = "#5ea1ff"; ctx.lineWidth = 2;
      const bx = s.x + s.r * 1.5, by = s.y - s.r * 1.5;
      ctx.beginPath(); ctx.arc(bx, by, 9, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#cfe3ff"; ctx.font = "700 11px -apple-system, Helvetica, Arial, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(String(i), bx, by + 0.5);
    });
    if (this.ghost?.added?.length) {
      const pts = this.stage.viewer.modelToScreen(this.ghost.added.map((a) => ({ x: a.x, y: a.y, z: a.z })));
      this.ghost.screenAdded = pts.map((p) => ({ x: p.x - P.ox, y: p.y - P.oy }));
    }
  }

  ring(s, k, color, w, dash = null, now = 0) {
    const ctx = this.ctx;
    ctx.beginPath();
    if (dash) { ctx.setLineDash(dash); ctx.lineDashOffset = -now / 60; }
    ctx.strokeStyle = color; ctx.lineWidth = w;
    ctx.arc(s.x, s.y, s.r * k, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  glow(x, y, r, rgb, a) {
    const g = this.ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a})`);
    g.addColorStop(0.5, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a * 0.3})`);
    g.addColorStop(1, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0)`);
    this.ctx.fillStyle = g;
    this.ctx.beginPath(); this.ctx.arc(x, y, r, 0, Math.PI * 2); this.ctx.fill();
  }

  lock(x, y, r0) {
    const ctx = this.ctx, R = r0 * 1.9;
    ctx.beginPath(); ctx.strokeStyle = "#ffb020"; ctx.lineWidth = 3; ctx.arc(x, y, R, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.strokeStyle = "rgba(255,176,32,.35)"; ctx.lineWidth = 7; ctx.arc(x, y, R, 0, Math.PI * 2); ctx.stroke();
    const gx = x + R * 0.78, gy = y - R * 0.78, s = Math.max(r0 * 0.55, 7);
    ctx.fillStyle = "#ffb020"; ctx.strokeStyle = "#ffb020"; ctx.lineWidth = Math.max(s * 0.28, 1.6);
    ctx.beginPath(); ctx.arc(gx, gy - s * 0.35, s * 0.55, Math.PI, 0); ctx.stroke();
    ctx.fillRect(gx - s * 0.8, gy - s * 0.35, s * 1.6, s * 1.2);
  }
}
