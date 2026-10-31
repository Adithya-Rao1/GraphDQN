// Protein + pocket scene inside the SAME 3Dmol viewer the ligand Stage uses.
// Stage.setKeyframe() calls viewer.removeAllModels() and registers click/hover on ALL atoms, so this module patches those two
// viewer methods (on the instance, not the library) so that only the ligand models are affected and protein atoms are never "lockable".
import { bus } from "./bus.js";

export const H = (c) => `#${c.toString(16).padStart(6, "0")}`;

/** Minimal PDB ATOM/HETATM reader: [{chain, resi, resn, name, x, y, z, elem}] */
export function parsePdb(text) {
  const out = [];
  for (const l of String(text).split("\n")) {
    if (!(l.startsWith("ATOM") || l.startsWith("HETATM"))) continue;
    out.push({
      name: l.slice(12, 16).trim(), resn: l.slice(17, 20).trim(), chain: l[21] || "A", resi: parseInt(l.slice(22, 26), 10),
      x: parseFloat(l.slice(30, 38)), y: parseFloat(l.slice(38, 46)), z: parseFloat(l.slice(46, 54)),
      elem: (l.slice(76, 78).trim() || l.slice(12, 16).trim()[0] || "C").toUpperCase(),
    });
  }
  return out;
}

export const centroid = (pts) => {
  const n = Math.max(pts.length, 1);
  return [pts.reduce((s, p) => s + p.x, 0) / n, pts.reduce((s, p) => s + p.y, 0) / n, pts.reduce((s, p) => s + p.z, 0) / n];
};

const CARTOON_A = [0x8aa4d6, 0x6fd0c4]; // N -> C terminus gradient, soft blue to teal

function lerpHex(a, b, t) {
  const f = (s) => Math.round(((a >> s) & 255) * (1 - t) + ((b >> s) & 255) * t);
  return (f(16) << 16) | (f(8) << 8) | f(0);
}

export class Pocket {
  constructor(stage) {
    this.stage = stage;
    this.viewer = stage.viewer;
    this.layers = new Set(); // protein-ish models that removeAllModels must not touch
    this.target = null;
    this.main = null; // {model, atoms(parsed), pocket residues}
    this.center = [0, 0, 0];
    this.active = false;
    this._patch();
  }

  _patch() {
    const v = this.viewer;
    const origRemove = v.removeAllModels.bind(v);
    v.removeAllModels = () => {
      if (!this.layers.size) return origRemove();
      for (const m of [...v.getModelList()]) if (!this.layers.has(m)) v.removeModel(m);
    };
    const origClick = v.setClickable.bind(v), origHover = v.setHoverable.bind(v);
    const scoped = (sel) => (this.layers.size && this.stage.model ? { ...sel, model: this.stage.model } : sel);
    v.setClickable = (sel, ...rest) => origClick(scoped(sel), ...rest);
    v.setHoverable = (sel, ...rest) => origHover(scoped(sel), ...rest);
    this._origHover = origHover;
  }

  /** Add a protein model from PDB text. Returns a layer {model, atoms, pocket, handles[]}. */
  addProtein(pdbText, { pocket = [], ghost = false, quality = "full" } = {}) {
    const v = this.viewer;
    const model = v.addModel(pdbText, "pdb");
    this.layers.add(model);
    const atoms = parsePdb(pdbText);
    const layer = { model, atoms, pocket, ghost, shapes: [], surfaces: [], labelAtom: null };
    this.style(layer);
    return layer;
  }

  style(layer, { dispByResi = null } = {}) {
    const { model, ghost, pocket } = layer;
    model.setStyle({}, {});
    if (ghost) {
      model.setStyle({}, { cartoon: { color: 0xc9d3e6, opacity: 0.22, thickness: 0.16 } });
      return;
    }
    const resis = [...new Set(layer.atoms.map((a) => a.resi))].sort((x, y) => x - y);
    const BANDS = 24, per = Math.max(1, Math.ceil(resis.length / BANDS));
    const stick = { radius: 0.11, colorscheme: { prop: "elem", map: { C: 0xaec1e6, N: 0x6aa6ff, O: 0xff7a70, S: 0xffd45a } }, opacity: 0.85 };
    const pocketSet = new Set(pocket);
    for (let i = 0; i < resis.length; i += per) {
      const band = resis.slice(i, i + per);
      const col = lerpHex(CARTOON_A[0], CARTOON_A[1], i / Math.max(resis.length - 1, 1));
      model.setStyle({ resi: band }, { cartoon: { color: col, thickness: 0.22, arrows: true } });
    }
    if (pocket.length) {
      for (const r of pocket) {
        const col = dispByResi?.has(r) ? heat(dispByResi.get(r)) : null;
        model.setStyle({ resi: [r] }, { cartoon: { color: col ?? lerpHex(CARTOON_A[0], CARTOON_A[1], (r - resis[0]) / Math.max(resis[resis.length - 1] - resis[0], 1)), thickness: 0.26, arrows: true }, stick });
      }
    }
    if (dispByResi) {
      for (const [r, d] of dispByResi) {
        if (d < 0.5 || pocketSet.has(r)) continue;
        model.setStyle({ resi: [r] }, { cartoon: { color: heat(d), thickness: 0.26, arrows: true } });
      }
    }
  }

  /** Translucent surface around the pocket residues. 3Dmol builds surfaces asynchronously; resolves when added. */
  async addPocketSurface(layer) {
    if (!layer.pocket.length || layer.ghost) return;
    const v = this.viewer;
    try {
      const id = await v.addSurface(window.$3Dmol.SurfaceType.MS, { opacity: 0.22, color: "#5ea1ff" }, { model: layer.model, resi: layer.pocket });
      layer.surfaces.push(id);
      v.render();
    } catch (e) { console.warn("pocket surface failed", e); }
  }

  removeSurfaces(layer) {
    for (const id of layer.surfaces) { try { this.viewer.removeSurface(id); } catch (_) { /* ignore */ } }
    layer.surfaces = [];
  }

  /** Residue name label on hover (CA atoms only). */
  enableHover(layer) {
    const v = this.viewer;
    this._origHover({ model: layer.model, atom: "CA" }, true,
      (atom) => {
        if (!atom || atom._lbl) return;
        atom._lbl = v.addLabel(`${atom.resn} ${atom.resi}${layer.pocket.includes(atom.resi) ? " · pocket" : ""}`, {
          position: atom, fontSize: 14, fontColor: "#eaf0f8", backgroundColor: "#0b1220", backgroundOpacity: 0.88,
          borderColor: "#5ea1ff", borderThickness: 1, inFront: true, showBackground: true,
        });
        v.render();
      },
      (atom) => { if (atom?._lbl) { v.removeLabel(atom._lbl); atom._lbl = null; v.render(); } });
  }

  removeLayer(layer) {
    if (!layer) return;
    this.removeSurfaces(layer);
    for (const s of layer.shapes) { try { this.viewer.removeShape(s); } catch (_) { /* ignore */ } }
    layer.shapes = [];
    this.layers.delete(layer.model);
    this.viewer.removeModel(layer.model);
  }

  /** The main scene for stages 1 and 3: soft cartoon + pocket surface; the ligand Stage keeps working beside it. */
  async load(target, pdbText) {
    this.clear();
    this.target = target;
    const layer = this.addProtein(pdbText, { pocket: target.pocket_residues || [] });
    this.main = layer;
    const pk = layer.atoms.filter((a) => a.name === "CA" && layer.pocket.includes(a.resi));
    this.center = pk.length ? centroid(pk) : centroid(layer.atoms.filter((a) => a.name === "CA"));
    this.enableHover(layer);
    this.active = true;
    this.viewer.rotate(40, "x"); // a 3/4 view reads better than looking straight down an axis
    this.fit(0);
    await this.addPocketSurface(layer);
    this.viewer.render();
    bus.emit("pocket:loaded", { target });
    return layer;
  }

  clear() {
    if (this.main) this.removeLayer(this.main);
    this.main = null;
    this.active = false;
    this.viewer.render();
  }

  hideMain() { if (this.main) { this.removeSurfaces(this.main); this.main.model.setStyle({}, {}); this.viewer.render(); } }
  async showMain() {
    if (!this.main) return;
    this.style(this.main);
    await this.addPocketSurface(this.main);
    this.viewer.render();
  }

  /** Soft pulsing blue aura on the 2D overlay around the pocket centre (3Dmol surfaces cannot glow). Call after the overlay was cleared. */
  drawAura(now, strength = 1) {
    const st = this.stage, ctx = st.ctx, c = this.center;
    if (!this.main) return;
    const dpr = window.devicePixelRatio || 1;
    const W = st.overlay.width / dpr, Hh = st.overlay.height / dpr;
    const [pt] = this.viewer.modelToScreen([{ x: c[0], y: c[1], z: c[2] }]);
    const rect = st.host.getBoundingClientRect();
    const x = pt.x - (rect.left + window.pageXOffset - document.documentElement.clientLeft);
    const y = pt.y - (rect.top + window.pageYOffset - document.documentElement.clientTop);
    const r = 0.42 * Math.min(W, Hh) * (1 + 0.05 * Math.sin(now / 700));
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.globalCompositeOperation = "lighter";
    const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, `rgba(94,161,255,${0.2 * strength})`); g.addColorStop(0.55, `rgba(94,161,255,${0.07 * strength})`); g.addColorStop(1, "rgba(94,161,255,0)");
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  }

  /** Frame the pocket (not the whole protein) so the ligand and its binding site fill the stage. */
  fit(ms = 600) {
    const m = this.main;
    if (!m) return this.viewer.zoomTo({}, ms);
    const sel = m.pocket.length ? { model: m.model, resi: m.pocket } : { model: m.model };
    const r = this.stage.host.getBoundingClientRect();
    const aspect = r.width / Math.max(r.height, 1);
    this.viewer.zoomTo(sel, ms);
    this.viewer.zoom(Math.min(1.0, 0.62 * aspect), ms);
  }
}

/** white -> amber -> red for displacement in Angstrom (0..4) */
export function heat(d) {
  const t = Math.max(0, Math.min(1, d / 3.5));
  return t < 0.5 ? lerpHex(0xdfe7f5, 0xffb020, t * 2) : lerpHex(0xffb020, 0xff3b30, (t - 0.5) * 2);
}
