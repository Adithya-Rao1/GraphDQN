// Ghost of a previewed child: a second, translucent 3Dmol model drawn in the stage's own viewer, superposed on the displayed
// molecule through the preview's atom_map. Added atoms are green; unchanged atoms are a thin cyan skeleton (one uniform opacity: 3Dmol cannot vary it per atom inside a model).
// The stage calls removeAllModels() on every keyframe change, which drops the ghost; clear() tolerates that.
import { parseMolblock, superpose } from "../molparse.js";
import { elemRadius } from "../morph.js";

const GREEN = 0x34f5a1, SKEL = 0x7fd8ff;

export class Ghost {
  constructor(stage) { this.stage = stage; this.model = null; this.raf = 0; }

  /** preview: ManualPreview (ok); returns the diff markers for the overlay, or null if the ghost could not be placed. */
  show(preview) {
    this.clear();
    const { viewer, morph } = this.stage;
    if (!viewer || !morph || !preview?.molblock) return null;
    const child = parseMolblock(preview.molblock);
    const fixed = [];
    for (const [p] of preview.atom_map) {
      const u = morph.childToU[p];
      if (u == null) return null;
      fixed[p] = morph.nodes[u].cPos;
    }
    // the displayed (current) molecule is the PARENT of the previewed child; its atoms are indexed like the current molblock
    const pairs = preview.atom_map.filter(([p, c]) => fixed[p] && c < child.atoms.length).map(([p, c]) => [c, p]);
    superpose(fixed, child.atoms, pairs);
    const mapped = new Map(preview.atom_map.map(([p, c]) => [c, p]));
    const mappedP = new Set(preview.atom_map.map(([p]) => p));
    const added = child.atoms.map((a, i) => ({ ...a, i })).filter((a) => !mapped.has(a.i));
    const changed = new Set();
    for (const [c, p] of mapped) {
      const u = morph.childToU[p];
      if (u != null && morph.nodes[u].cElem !== child.atoms[c].elem) changed.add(p);
    }
    const removed = new Set();
    morph.childToU.forEach((u, p) => { if (u != null && !mappedP.has(p)) removed.add(p); });

    const model = viewer.addModel();
    model.addAtoms(child.atoms.map((a, i) => ({ elem: a.elem, x: a.x, y: a.y, z: a.z, serial: 10000 + i, index: i, bonds: [], bondOrder: [] })));
    const A = model.atoms;
    for (const b of child.bonds) { A[b.a].bonds.push(b.b); A[b.a].bondOrder.push(b.order); A[b.b].bonds.push(b.a); A[b.b].bondOrder.push(b.order); }
    this.model = model;
    this.t0 = performance.now();
    const style = (k) => {
      A.forEach((a, i) => {
        const isNew = !mapped.has(i);
        a.style = isNew
          ? { stick: { radius: 0.16, color: GREEN, opacity: 0.6 * k }, sphere: { radius: elemRadius(a.elem) * 0.95, color: GREEN, opacity: 0.6 * k } }
          : { stick: { radius: 0.07, color: SKEL, opacity: 0.6 * k } };
      });
      model.molObj = null;
      viewer.render();
    };
    const step = (now) => {
      const k = Math.min((now - this.t0) / 260, 1);
      if (this.model !== model) return;
      style(k);
      if (k < 1) this.raf = requestAnimationFrame(step);
    };
    this.raf = requestAnimationFrame(step);
    return { added: added.map((a) => ({ x: a.x, y: a.y, z: a.z, elem: a.elem })), removed, changed, counts: { added: added.length, removed: removed.size, changed: changed.size } };
  }

  clear() {
    cancelAnimationFrame(this.raf);
    if (this.model) {
      try { this.stage.viewer.removeModel(this.model); this.stage.viewer.render(); } catch (_) { /* already removed by a keyframe change */ }
      this.model = null;
    }
  }
}
