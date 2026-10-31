// Stage 2: DynamicBind pose. Request -> poll -> progress animation -> the ligand glides from outside the pocket into the predicted
// pose while the protein morphs from the apo structure to the predicted one (apo ghosted for comparison, moved residues highlighted).
import { bus } from "./bus.js";
import { store } from "./state.js";
import * as API from "./api.js";
import { parseMolblock, superpose, firstMolRecord } from "./molparse.js";
import { elemRGB, hexOf } from "./morph.js";
import { parsePdb, centroid, heat } from "./pocket.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const clamp01 = (x) => Math.max(0, Math.min(1, x));
const GLIDE_MS = 2600, MORPH_START = 900, MORPH_MS = 2200;

// ---------------------------------------------------------------- mol block helpers (heavy atoms only)
/** Drop hydrogens from a V2000 block; returns a new block string (so seed_molblock obeys the heavy-atoms-only contract). */
export function heavyMolblock(text) {
  const rec = firstMolRecord(text).molfile;
  const lines = rec.replace(/\r/g, "").split("\n");
  const nA = parseInt(lines[3].slice(0, 3), 10), nB = parseInt(lines[3].slice(3, 6), 10);
  const atoms = lines.slice(4, 4 + nA), bonds = lines.slice(4 + nA, 4 + nA + nB);
  const keep = atoms.map((l) => l.slice(31, 34).trim() !== "H");
  const newIdx = []; let k = 0;
  keep.forEach((o, i) => { newIdx[i] = o ? ++k : -1; });
  const nb = bonds.filter((l) => keep[parseInt(l.slice(0, 3), 10) - 1] && keep[parseInt(l.slice(3, 6), 10) - 1])
    .map((l) => String(newIdx[parseInt(l.slice(0, 3), 10) - 1]).padStart(3) + String(newIdx[parseInt(l.slice(3, 6), 10) - 1]).padStart(3) + l.slice(6));
  const tail = lines.slice(4 + nA + nB).filter((l) => !l.startsWith("M  CHG") && !l.startsWith("M  END"));
  const chg = [];
  lines.slice(4 + nA + nB).forEach((l) => {
    if (!l.startsWith("M  CHG")) return;
    const n = parseInt(l.slice(6, 9), 10);
    for (let q = 0; q < n; q++) {
      const idx = parseInt(l.slice(10 + 8 * q, 14 + 8 * q), 10) - 1, c = parseInt(l.slice(14 + 8 * q, 18 + 8 * q), 10);
      if (keep[idx]) chg.push([newIdx[idx], c]);
    }
  });
  const out = [lines[0], lines[1], lines[2], String(k).padStart(3) + String(nb.length).padStart(3) + lines[3].slice(6),
    ...atoms.filter((_, i) => keep[i]), ...nb];
  for (let i = 0; i < chg.length; i += 8) {
    const part = chg.slice(i, i + 8);
    out.push("M  CHG" + String(part.length).padStart(3) + part.map(([a, c]) => String(a).padStart(4) + String(c).padStart(4)).join(""));
  }
  out.push(...tail.filter((l) => l.trim() && !/^\$\$\$\$/.test(l)), "M  END");
  return out.join("\n") + "\n";
}

function pdbWithCoords(text, atoms) {
  let i = 0;
  const f = (v) => v.toFixed(3).padStart(8);
  return text.split("\n").map((l) => {
    if (!(l.startsWith("ATOM") || l.startsWith("HETATM"))) return l;
    const a = atoms[i++];
    return l.slice(0, 30) + f(a.x) + f(a.y) + f(a.z) + l.slice(54);
  }).join("\n");
}

// ------------------------------------------------------------------------------------ 3D ligand
function makeLigand(viewer, molblock) {
  const { atoms, bonds } = parseMolblock(molblock);
  const model = viewer.addModel();
  model.addAtoms(atoms.map((a, i) => ({ elem: a.elem, x: a.x, y: a.y, z: a.z, serial: i, index: i })));
  const ms = model.atoms;
  ms.forEach((m) => { m.bonds = []; m.bondOrder = []; });
  for (const b of bonds) { ms[b.a].bonds.push(b.b); ms[b.a].bondOrder.push(b.order); ms[b.b].bonds.push(b.a); ms[b.b].bondOrder.push(b.order); }
  ms.forEach((m) => { const c = hexOf(elemRGB(m.elem)); m.style = { sphere: { radius: 0.34, color: c }, stick: { radius: 0.15, color: c } }; });
  return { model, base: atoms.map((a) => ({ x: a.x, y: a.y, z: a.z })), n: atoms.length, atoms };
}

function setLigandPose(lig, fn) {
  const ms = lig.model.atoms;
  for (let i = 0; i < lig.n; i++) { const p = fn(lig.base[i]); ms[i].x = p[0]; ms[i].y = p[1]; ms[i].z = p[2]; }
  lig.model.molObj = null;
}

function rotAxis(axis, ang) {
  const [x, y, z] = axis, c = Math.cos(ang), s = Math.sin(ang), t = 1 - c;
  return [[t * x * x + c, t * x * y - s * z, t * x * z + s * y], [t * x * y + s * z, t * y * y + c, t * y * z - s * x], [t * x * z - s * y, t * y * z + s * x, t * z * z + c]];
}
const mul = (R, v) => [R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2], R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2], R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2]];

// ------------------------------------------------------------------------------------ view
export class PoseView {
  constructor({ stage, pocket }) {
    this.stage = stage; this.pocket = pocket; this.viewer = stage.viewer;
    this.run = 0; // generation counter: cancels stale async work
    this.raf = 0; this.layers = []; this.lig = null; this.result = null; this.shapes = []; this.labels = [];
    this.opts = { apo: true, disp: true };
  }

  get active() { return !!this._active; }

  // detach the ligand Stage (its keyframe model) so this view owns the scene
  _enter() {
    if (this._active) return;
    this._active = true;
    const st = this.stage;
    if (st.model) { try { this.viewer.removeModel(st.model); } catch (_) { /* ignore */ } }
    st.model = null; st.morph = null;
    st.ctx.clearRect(0, 0, st.overlay.width, st.overlay.height);
    $("hud-title").innerHTML = ""; $("hud-legend").innerHTML = "";
  }

  /** Remove everything this view added; the caller restores the Stage keyframe. */
  leave() {
    this.run++;
    cancelAnimationFrame(this.raf);
    this._clearScene();
    this._active = false;
    this.pocket.showMain?.();
    $("hud-pose").hidden = true; $("hud-progress").hidden = true;
  }

  _clearScene() {
    for (const l of this.layers) this.pocket.removeLayer(l);
    this.layers = [];
    if (this.lig) { this.viewer.removeModel(this.lig.model); this.lig = null; }
    this._clearMarks();
    this.viewer.render();
  }

  _clearMarks() {
    for (const s of this.shapes) { try { this.viewer.removeShape(s); } catch (_) { /* ignore */ } }
    for (const l of this.labels) { try { this.viewer.removeLabel(l); } catch (_) { /* ignore */ } }
    this.shapes = []; this.labels = [];
  }

  // ------------------------------------------------------------------ request -> poll -> play
  async start({ target, smiles, structureText, leadAffinity, onResult }) {
    const my = ++this.run;
    cancelAnimationFrame(this.raf);
    this._enter();
    this._clearScene();
    this.target = target; this.smiles = smiles; this.leadAffinity = leadAffinity;
    this.pocket.showMain?.();
    this.pocket.fit(700);
    $("hud-pose").hidden = true;
    this._progress(0, "Requesting pose…");
    this._startOrbit(smiles, my);
    let job;
    try {
      job = await API.createPose({ target_id: target.id, smiles });
    } catch (e) { return this._fail(my, e.message || String(e)); }
    let res = job;
    for (let i = 0; i < 400 && this.run === my; i++) {
      try { res = await API.getPose(job.job_id); } catch (e) { return this._fail(my, e.message || String(e)); }
      if (res.status === "done" || res.status === "failed") break;
      const label = res.status === "queued" ? "Queued for DynamicBind…" : "DynamicBind: sampling pose + protein flexibility…";
      this._progress(res.progress ?? 0, label);
      await new Promise((r) => setTimeout(r, 900));
    }
    if (this.run !== my) return;
    if (res.status !== "done") return this._fail(my, res.note || "Pose prediction failed");
    this._progress(1, "Pose ready");
    await new Promise((r) => setTimeout(r, 250));
    if (this.run !== my) return;
    this.result = res;
    const done = await this.play(res, structureText, my);
    if (done) onResult?.(done);
    return done;
  }

  _fail(my, msg) {
    if (this.run !== my) return null;
    this._stopOrbit();
    $("hud-progress").hidden = false;
    $("hud-progress").innerHTML = `<div class="prog-card fail"><b>Pose prediction failed</b><div>${esc(msg)}</div></div>`;
    bus.emit("pose:failed", { message: msg });
    return null;
  }

  _progress(p, label) {
    const el = $("hud-progress");
    el.hidden = false;
    const pct = Math.round(clamp01(p) * 100);
    el.innerHTML = `<div class="prog-card"><div class="ring" style="--p:${pct}"><span>${pct}%</span></div><div class="prog-txt"><b>${esc(label)}</b><div class="muted">${esc(this.smiles || "")}</div></div></div>`;
  }

  // ---- waiting animation: the ligand (server conformer) tumbles outside the pocket
  async _startOrbit(smiles, my) {
    this._stopOrbit();
    let molblock = null;
    try { const v = await API.validate(smiles); if (v?.ok && v.molblock) molblock = v.molblock; } catch (_) { /* decorative only */ }
    if (!molblock || this.run !== my || this._posePlaying) return;
    try {
      const lig = makeLigand(this.viewer, heavyMolblock(molblock));
      this.orbit = lig;
      const c = centroid(lig.base);
      const t0 = performance.now();
      const startPos = this._startPos();
      const loop = (now) => {
        if (this.orbit !== lig) return;
        const t = (now - t0) / 1000;
        const R = rotAxis([0.3, 0.8, 0.52], t * 0.9);
        setLigandPose(lig, (p) => { const v = mul(R, [p.x - c[0], p.y - c[1], p.z - c[2]]); return [startPos[0] + v[0], startPos[1] + v[1] + Math.sin(t * 1.6) * 0.8, startPos[2] + v[2]]; });
        this.viewer.rotate(0.15, "y");
        this.viewer.render();
        this.orbitRaf = requestAnimationFrame(loop);
      };
      this.orbitRaf = requestAnimationFrame(loop);
    } catch (e) { console.warn("orbit preview failed", e); }
  }

  _stopOrbit() {
    cancelAnimationFrame(this.orbitRaf);
    if (this.orbit) { try { this.viewer.removeModel(this.orbit.model); } catch (_) { /* ignore */ } this.orbit = null; }
  }

  _startPos() {
    const c = this.pocket.center, m = this.pocket.main;
    const all = m ? centroid(m.atoms.filter((a) => a.name === "CA")) : c;
    let d = [c[0] - all[0], c[1] - all[1], c[2] - all[2]];
    const n = Math.hypot(...d);
    d = n < 2 ? [1, 0.25, 0.1] : d.map((x) => x / n);
    const nn = Math.hypot(...d);
    return [c[0] + (d[0] / nn) * 24, c[1] + (d[1] / nn) * 24, c[2] + (d[2] / nn) * 24];
  }

  // ------------------------------------------------------------------------------- the glide + morph
  async play(res, structureText, my = this.run) {
    this._posePlaying = true;
    this._stopOrbit();
    $("hud-progress").hidden = true;
    this._clearScene();
    this.pocket.hideMain?.();

    const apoText = res.apo_pdb || structureText;
    const predText = res.protein_pdb || apoText;
    if (!apoText || !predText) return this._fail(my, "The result carries no protein structure.");
    const ligText = heavyMolblock(res.ligand_sdf || "");
    const apoA = parsePdb(apoText), predA = parsePdb(predText);
    const lig = parseMolblock(ligText);

    // align predicted protein (and its ligand) onto the apo frame using backbone CA, trimming the most mobile 30 %
    const key = (a) => `${a.chain}:${a.resi}:${a.name}`;
    const apoMap = new Map(apoA.map((a, i) => [key(a), i]));
    const caApo = apoA.filter((a) => a.name === "CA"), caPred = predA.filter((a) => a.name === "CA");
    const apoCaIdx = new Map(caApo.map((a, i) => [`${a.chain}:${a.resi}`, i]));
    let pairs = [];
    caPred.forEach((a, i) => { const j = apoCaIdx.get(`${a.chain}:${a.resi}`); if (j !== undefined) pairs.push([i, j]); });
    const extras = predA.filter((a) => a.name !== "CA").concat(lig.atoms);
    if (pairs.length >= 3) {
      superpose(caApo, caPred, pairs, extras);
      const res2 = pairs.map(([i, j]) => [Math.hypot(caPred[i].x - caApo[j].x, caPred[i].y - caApo[j].y, caPred[i].z - caApo[j].z), i, j]).sort((a, b) => a[0] - b[0]);
      const keep = res2.slice(0, Math.max(3, Math.floor(res2.length * 0.7))).map(([, i, j]) => [i, j]);
      superpose(caApo, caPred, keep, extras);
    }
    // per-residue CA displacement (predicted vs apo, Angstrom)
    const disp = new Map(); const moved = [];
    for (const [i, j] of pairs) {
      const d = Math.hypot(caPred[i].x - caApo[j].x, caPred[i].y - caApo[j].y, caPred[i].z - caApo[j].z);
      disp.set(caPred[i].resi, d);
      moved.push({ d, resi: caPred[i].resi, resn: caPred[i].resn, apo: caApo[j], pred: caPred[i] });
    }
    moved.sort((a, b) => b.d - a.d);

    const pocketRes = this.target?.pocket_residues || [];
    // ghost = apo; main = predicted topology, starting on the apo coordinates
    const ghost = this.pocket.addProtein(apoText, { ghost: true });
    const main = this.pocket.addProtein(pdbWithCoords(predText, predA), { pocket: pocketRes });
    this.layers = [ghost, main];
    this.pocket.enableHover(main);
    const mAtoms = main.model.atoms;
    const p0 = predA.map((a) => { const j = apoMap.get(key(a)); return j === undefined ? { x: a.x, y: a.y, z: a.z } : { x: apoA[j].x, y: apoA[j].y, z: apoA[j].z }; });
    const p1 = predA.map((a) => ({ x: a.x, y: a.y, z: a.z }));
    ghost.model.setStyle({}, { cartoon: { color: 0xc9d3e6, opacity: this.opts.apo ? 0.22 : 0, thickness: 0.16 } });
    if (!this.opts.apo) ghost.model.setStyle({}, {});

    // ligand: final pose = lig.atoms (aligned); start = rotated + pushed outside the pocket
    const ligModel = makeLigand(this.viewer, ligText);
    ligModel.base = lig.atoms.map((a) => ({ x: a.x, y: a.y, z: a.z }));
    this.lig = ligModel;
    const cFin = centroid(ligModel.base);
    const startPos = this._startPos();
    const off = [startPos[0] - this.pocket.center[0], startPos[1] - this.pocket.center[1], startPos[2] - this.pocket.center[2]];
    const axis = [0.3, 0.8, 0.52]; const nrm = Math.hypot(...axis); const ax = axis.map((x) => x / nrm);
    const ang0 = Math.PI * 0.85;

    this.pocket.fit(0);
    this.viewer.render();
    const t0 = performance.now();
    let lastMorph = -1, lastT = t0;
    const stage = this.stage;
    return new Promise((resolve) => {
      const frame = (now) => {
        if (this.run !== my) return resolve(null);
        const t = now - t0;
        const e1 = ease(clamp01(t / GLIDE_MS));
        const e2 = ease(clamp01((t - MORPH_START) / MORPH_MS));
        const R = rotAxis(ax, ang0 * (1 - e1));
        setLigandPose(ligModel, (p) => {
          const v = mul(R, [p.x - cFin[0], p.y - cFin[1], p.z - cFin[2]]);
          return [cFin[0] + v[0] + off[0] * (1 - e1), cFin[1] + v[1] + off[1] * (1 - e1), cFin[2] + v[2] + off[2] * (1 - e1)];
        });
        if (e2 - lastMorph > 0.04 || (e2 === 1 && lastMorph < 1)) {
          lastMorph = e2;
          for (let i = 0; i < mAtoms.length; i++) { mAtoms[i].x = p0[i].x + (p1[i].x - p0[i].x) * e2; mAtoms[i].y = p0[i].y + (p1[i].y - p0[i].y) * e2; mAtoms[i].z = p0[i].z + (p1[i].z - p0[i].z) * e2; }
          main.model.molObj = null;
        }
        if (store.overlay.autorotate && t < GLIDE_MS + 600) this.viewer.rotate(((now - lastT) * 9) / 1000, "y");
        lastT = now;
        this.viewer.render();
        this._fx(now, cFin, off, e1, t);
        if (t < GLIDE_MS + MORPH_START + 300) return void (this.raf = requestAnimationFrame(frame));
        // settled
        resolve(this._finish({ main, ghost, moved, disp, ligModel, ligText, predText: pdbWithCoords(predText, predA), my }));
        this._idle();
      };
      this.raf = requestAnimationFrame(frame);
    });
  }

  // 2D glow riding on the ligand + an arrival ring
  _fx(now, cFin, off, e1, t) {
    const st = this.stage, ctx = st.ctx, dpr = window.devicePixelRatio || 1;
    const W = st.overlay.width / dpr, Hh = st.overlay.height / dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, Hh);
    const pos = { x: cFin[0] + off[0] * (1 - e1), y: cFin[1] + off[1] * (1 - e1), z: cFin[2] + off[2] * (1 - e1) };
    const [pt] = this.viewer.modelToScreen([pos]);
    const rect = st.host.getBoundingClientRect();
    const x = pt.x - (rect.left + window.pageXOffset - document.documentElement.clientLeft);
    const y = pt.y - (rect.top + window.pageYOffset - document.documentElement.clientTop);
    ctx.globalCompositeOperation = "lighter";
    const a = 0.55 * (1 - e1 * 0.6);
    const g = ctx.createRadialGradient(x, y, 0, x, y, 110);
    g.addColorStop(0, `rgba(52,245,161,${a})`); g.addColorStop(0.4, `rgba(94,161,255,${a * 0.3})`); g.addColorStop(1, "rgba(94,161,255,0)");
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, 110, 0, Math.PI * 2); ctx.fill();
    const tt = (t - GLIDE_MS * 0.85) / 900; // arrival ring
    if (tt > 0 && tt < 1) {
      ctx.strokeStyle = `rgba(52,245,161,${0.8 * (1 - tt)})`; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(x, y, 30 + tt * 150, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.globalCompositeOperation = "source-over";
    this.pocket.drawAura(now);
  }

  _idle() {
    // keep the scene alive: slow rotation + render, until leave()/start()
    const my = this.run;
    let last = performance.now();
    const loop = (now) => {
      if (this.run !== my || !this._active) return;
      if (store.overlay.autorotate) this.viewer.rotate(((now - last) * 9) / 1000, "y"); else if (this._dirty) this.viewer.render();
      this._dirty = false;
      last = now;
      this.stage.ctx.clearRect(0, 0, this.stage.overlay.width, this.stage.overlay.height);
      this.pocket.drawAura(now);
      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
  }

  _finish({ main, ghost, moved, disp, ligModel, ligText, predText, my }) {
    this._posePlaying = false;
    this.main = main; this.ghost = ghost; this.moved = moved; this.disp = disp;
    this.pocket.style(main, { dispByResi: this.opts.disp ? disp : null });
    this.pocket.addPocketSurface(main);
    this._marks();
    this.viewer.render();
    this._tiles();
    this.finalLigand = ligText; this.finalProtein = predText;
    return { result: this.result, ligandBlock: ligText, proteinPdb: predText, moved: moved.slice(0, 5), disp };
  }

  /** Displacement lines + labels for the most moved residues. */
  _marks() {
    this._clearMarks();
    if (!this.moved) return;
    const v = this.viewer;
    const top = this.moved.filter((m) => m.d >= 0.8).slice(0, 24);
    if (this.opts.disp) {
      for (const m of top) {
        this.shapes.push(v.addCylinder({ start: { x: m.apo.x, y: m.apo.y, z: m.apo.z }, end: { x: m.pred.x, y: m.pred.y, z: m.pred.z }, radius: 0.07, color: heat(m.d), fromCap: 1, toCap: 2, alpha: 0.9 }));
      }
      for (const m of top.slice(0, 3)) {
        this.labels.push(v.addLabel(`${m.resn} ${m.resi}  moved ${m.d.toFixed(1)} Å`, {
          position: { x: m.pred.x, y: m.pred.y, z: m.pred.z }, fontSize: 13, fontColor: "#fff", backgroundColor: "#7a1f18", backgroundOpacity: 0.85, borderColor: "#ff6b63", borderThickness: 1, showBackground: true, inFront: true,
        }));
      }
    }
  }

  setOverlay(on) {
    this.opts.apo = on;
    if (!this.ghost) return;
    if (on) this.ghost.model.setStyle({}, { cartoon: { color: 0xc9d3e6, opacity: 0.22, thickness: 0.16 } }); else this.ghost.model.setStyle({}, {});
    this.viewer.render();
  }

  setHighlight(on) {
    this.opts.disp = on;
    if (!this.main) return;
    this.pocket.style(this.main, { dispByResi: on ? this.disp : null });
    this._marks();
    this.viewer.render();
  }

  _tiles() {
    const r = this.result;
    const el = $("hud-pose");
    const fmt = (v, d = 2) => (typeof v === "number" ? v.toFixed(d) : "n/a");
    const lead = this.leadAffinity;
    el.hidden = false;
    el.innerHTML = `
      <div class="tile"><div class="k">Pose confidence</div><div class="v">${fmt(r.pose_confidence)}</div><div class="s">DynamicBind · predicted</div></div>
      <div class="tile"><div class="k">DynamicBind affinity</div><div class="v">${fmt(r.affinity, 1)}</div><div class="s">independent model · predicted</div></div>
      <div class="tile alt"><div class="k">PLAPT affinity (stage 1)</div><div class="v">${typeof lead === "number" ? lead.toFixed(1) : "n/a"}</div><div class="s">sequence model · predicted</div></div>
      <div class="tile-note">Both affinities are model predictions, not measurements. They come from different models and need not agree. ${esc(r.note || "")}</div>`;
  }

  replay() {
    if (!this.result) return;
    return this.play(this.result, null, ++this.run);
  }
}
