// Minimal MOL/SDF V2000 reader and a Kabsch-style superposition (Horn quaternion method).
// Only what the viewer needs: heavy-atom coordinates, elements, formal charges, bonds.

export function parseMolblock(text) {
  const lines = String(text).replace(/\r/g, "").split("\n");
  const counts = lines[3] || "";
  const nAtoms = parseInt(counts.slice(0, 3), 10);
  const nBonds = parseInt(counts.slice(3, 6), 10);
  if (!Number.isFinite(nAtoms) || !Number.isFinite(nBonds)) throw new Error("Unreadable MOL block");
  const atoms = [];
  for (let i = 0; i < nAtoms; i++) {
    const l = lines[4 + i];
    atoms.push({
      x: parseFloat(l.slice(0, 10)),
      y: parseFloat(l.slice(10, 20)),
      z: parseFloat(l.slice(20, 30)),
      elem: l.slice(31, 34).trim(),
      charge: 0,
    });
  }
  const bonds = [];
  for (let i = 0; i < nBonds; i++) {
    const l = lines[4 + nAtoms + i];
    const a = parseInt(l.slice(0, 3), 10) - 1;
    const b = parseInt(l.slice(3, 6), 10) - 1;
    const order = parseInt(l.slice(6, 9), 10);
    bonds.push({ a, b, order: order > 3 ? 1 : order });
  }
  for (const l of lines.slice(4 + nAtoms + nBonds)) {
    if (l.startsWith("M  CHG")) {
      const n = parseInt(l.slice(6, 9), 10);
      for (let k = 0; k < n; k++) {
        const idx = parseInt(l.slice(10 + 8 * k, 14 + 8 * k), 10) - 1;
        const chg = parseInt(l.slice(14 + 8 * k, 18 + 8 * k), 10);
        if (atoms[idx]) atoms[idx].charge = chg;
      }
    }
    if (l.startsWith("M  END")) break;
  }
  return { atoms, bonds };
}

/** First record of an SDF (or a plain MOL) as text. */
export function firstMolRecord(text) {
  const t = String(text).replace(/\r/g, "");
  const i = t.indexOf("\n$$$$");
  return { molfile: i >= 0 ? t.slice(0, i) + "\n" : t, multiple: i >= 0 && t.slice(i + 5).trim().length > 0 };
}

function centroid(pts) {
  const c = [0, 0, 0];
  for (const p of pts) { c[0] += p.x; c[1] += p.y; c[2] += p.z; }
  return c.map((v) => v / (pts.length || 1));
}

// Jacobi eigen-decomposition of a symmetric 4x4 matrix -> eigenvector of the largest eigenvalue.
function topEigenvector(M) {
  const n = 4;
  const A = M.map((r) => r.slice());
  const V = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
  for (let sweep = 0; sweep < 50; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
    if (off < 1e-18) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-15) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  let best = 0;
  for (let i = 1; i < n; i++) if (A[i][i] > A[best][best]) best = i;
  return [V[0][best], V[1][best], V[2][best], V[3][best]];
}

/**
 * Rotate+translate `movingAtoms` in place so the points listed in `pairs` ([movingIdx, fixedIdx])
 * superpose on `fixedAtoms`; `extras` (e.g. a second conformer) receive the same rigid transform. With fewer than 3 pairs only the centroids are matched.
 * Returns the RMSD over the pairs after superposition (null when there are no pairs).
 */
export function superpose(fixedAtoms, movingAtoms, pairs, extras = []) {
  const all = movingAtoms.concat(extras);
  if (!pairs.length) {
    const cf = centroid(fixedAtoms), cm = centroid(movingAtoms);
    for (const a of all) { a.x += cf[0] - cm[0]; a.y += cf[1] - cm[1]; a.z += cf[2] - cm[2]; }
    return null;
  }
  const F = pairs.map(([, f]) => fixedAtoms[f]);
  const Mv = pairs.map(([m]) => movingAtoms[m]);
  const cf = centroid(F), cm = centroid(Mv);
  let R = null;
  if (pairs.length >= 3) {
    const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < pairs.length; i++) {
      const m = [Mv[i].x - cm[0], Mv[i].y - cm[1], Mv[i].z - cm[2]];
      const f = [F[i].x - cf[0], F[i].y - cf[1], F[i].z - cf[2]];
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) S[r][c] += m[r] * f[c];
    }
    const [Sxx, Sxy, Sxz] = S[0], [Syx, Syy, Syz] = S[1], [Szx, Szy, Szz] = S[2];
    const N = [
      [Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx],
      [Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz],
      [Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy],
      [Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz],
    ];
    const [q0, q1, q2, q3] = topEigenvector(N);
    R = [
      [q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
      [2 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 - q0 * q1)],
      [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3],
    ];
  }
  for (const a of all) {
    let v = [a.x - cm[0], a.y - cm[1], a.z - cm[2]];
    if (R) v = [0, 1, 2].map((r) => R[r][0] * v[0] + R[r][1] * v[1] + R[r][2] * v[2]);
    a.x = v[0] + cf[0]; a.y = v[1] + cf[1]; a.z = v[2] + cf[2];
  }
  let s = 0;
  for (let i = 0; i < pairs.length; i++) {
    const m = movingAtoms[pairs[i][0]], f = fixedAtoms[pairs[i][1]];
    s += (m.x - f.x) ** 2 + (m.y - f.y) ** 2 + (m.z - f.z) ** 2;
  }
  return Math.sqrt(s / pairs.length);
}
