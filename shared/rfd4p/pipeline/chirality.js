/**
 * Idealized local geometry features for stereogenic centers and E/Z double bonds.
 *
 * Ports `rfproteina.utils.chirals` and `rfproteina.transforms.chirality_feature`.
 */

import { BOND_TYPE, METAL_ELEMENTS, getResidueSegments } from './cif.js';

export const MAX_OVERLAPPING_CENTERS = 6;

export const CHIRAL_TYPE = {
  CHI_UNSPECIFIED: 0,
  CHI_TETRAHEDRAL_CW: 1,
  CHI_TETRAHEDRAL_CCW: 2,
  CHI_OTHER: 3,
  CHI_TETRAHEDRAL: 4,
  CHI_ALLENE: 5,
  CHI_SQUAREPLANAR: 6,
  CHI_TRIGONALBIPYRAMIDAL: 7,
  CHI_OCTAHEDRAL: 8,
};

const INV_SQRT3 = 1 / Math.sqrt(3);
const HALF_SQRT3 = Math.sqrt(3) / 2;

export const IDEAL_POLYHEDRA = {
  tetrahedral: [
    [INV_SQRT3, INV_SQRT3, INV_SQRT3],
    [INV_SQRT3, -INV_SQRT3, -INV_SQRT3],
    [-INV_SQRT3, INV_SQRT3, -INV_SQRT3],
    [-INV_SQRT3, -INV_SQRT3, INV_SQRT3],
  ],
  square_planar: [
    [1, 0, 0],
    [0, 1, 0],
    [-1, 0, 0],
    [0, -1, 0],
  ],
  trigonal_bipyramidal: [
    [1, 0, 0],
    [-0.5, HALF_SQRT3, 0],
    [-0.5, -HALF_SQRT3, 0],
    [0, 0, 1],
    [0, 0, -1],
  ],
  octahedral: [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
  ],
};

export const DOUBLE_BOND_TEMPLATES = new Map();
for (const side of [1, -1]) {
  for (const nLeft of [1, 2]) {
    for (const nRight of [1, 2]) {
      const leftPts = [
        [-1, HALF_SQRT3, 0],
        [-1, -HALF_SQRT3, 0],
      ].slice(0, nLeft);
      const rightPts = [
        [1, side * HALF_SQRT3, 0],
        [1, -side * HALF_SQRT3, 0],
      ].slice(0, nRight);
      DOUBLE_BOND_TEMPLATES.set(
        `${side}:${nLeft}:${nRight}`,
        [[-0.5, 0, 0], [0.5, 0, 0], ...leftPts, ...rightPts]
      );
    }
  }
}

// ---------------------------------------------------------------------------
// 3x3 Matrix / Kabsch SVD Alignment (`Rotation.align_vectors(a, b)`)
// ---------------------------------------------------------------------------

function det3x3(m) {
  return (
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
  );
}

function matMul3x3(A, B) {
  const C = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      C[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j];
    }
  }
  return C;
}

function transpose3x3(A) {
  return [
    [A[0][0], A[1][0], A[2][0]],
    [A[0][1], A[1][1], A[2][1]],
    [A[0][2], A[1][2], A[2][2]],
  ];
}

function jacobiEigenSymmetric3x3(A) {
  const S = [
    [A[0][0], A[0][1], A[0][2]],
    [A[1][0], A[1][1], A[1][2]],
    [A[2][0], A[2][1], A[2][2]],
  ];
  const V = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];

  for (let iter = 0; iter < 50; iter++) {
    let p = 0, q = 1;
    let maxVal = Math.abs(S[0][1]);
    if (Math.abs(S[0][2]) > maxVal) { maxVal = Math.abs(S[0][2]); p = 0; q = 2; }
    if (Math.abs(S[1][2]) > maxVal) { maxVal = Math.abs(S[1][2]); p = 1; q = 2; }
    if (maxVal < 1e-15) break;

    const app = S[p][p];
    const aqq = S[q][q];
    const apq = S[p][q];
    const tau = (aqq - app) / (2 * apq);
    const t = tau >= 0 ? 1 / (tau + Math.sqrt(1 + tau * tau)) : -1 / (-tau + Math.sqrt(1 + tau * tau));
    const c = 1 / Math.sqrt(1 + t * t);
    const s = t * c;

    S[p][p] = app - t * apq;
    S[q][q] = aqq + t * apq;
    S[p][q] = 0;
    S[q][p] = 0;

    for (let r = 0; r < 3; r++) {
      if (r !== p && r !== q) {
        const srp = S[r][p];
        const srq = S[r][q];
        S[r][p] = c * srp - s * srq;
        S[p][r] = S[r][p];
        S[r][q] = s * srp + c * srq;
        S[q][r] = S[r][q];
      }
      const vrp = V[r][p];
      const vrq = V[r][q];
      V[r][p] = c * vrp - s * vrq;
      V[r][q] = s * vrp + c * vrq;
    }
  }

  // Sort eigenvalues descending
  const order = [0, 1, 2].sort((i, j) => S[j][j] - S[i][j]);
  const eigVals = order.map((i) => Math.max(0, S[i][i]));
  const eigVecs = [
    [V[0][order[0]], V[0][order[1]], V[0][order[2]]],
    [V[1][order[0]], V[1][order[1]], V[1][order[2]]],
    [V[2][order[0]], V[2][order[1]], V[2][order[2]]],
  ];
  if (det3x3(eigVecs) < 0) {
    eigVecs[0][2] = -eigVecs[0][2];
    eigVecs[1][2] = -eigVecs[1][2];
    eigVecs[2][2] = -eigVecs[2][2];
  }
  return { eigVals, eigVecs };
}

function cross3(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function normalize3(v) {
  const n = Math.hypot(v[0], v[1], v[2]);
  return n > 1e-12 ? [v[0] / n, v[1] / n, v[2] / n] : [0, 0, 0];
}

/**
 * Find optimal proper rotation matrix `R` (3x3) aligning `b` onto `a`
 * (minimizing `sum_i ||a_i - R b_i||^2`), matching `scipy.spatial.transform.Rotation.align_vectors(a, b)`.
 * Returns `{ rotation, rssd }`.
 */
export function alignVectors(a, b) {
  const k = a.length;
  // B = a^T @ b
  const B = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  let normSqSum = 0;
  for (let i = 0; i < k; i++) {
    const ai = a[i];
    const bi = b[i];
    normSqSum += ai[0] * ai[0] + ai[1] * ai[1] + ai[2] * ai[2];
    normSqSum += bi[0] * bi[0] + bi[1] * bi[1] + bi[2] * bi[2];
    for (let p = 0; p < 3; p++) {
      for (let q = 0; q < 3; q++) {
        B[p][q] += ai[p] * bi[q];
      }
    }
  }

  // SVD of B via eigendecomposition of B^T B
  const Bt = transpose3x3(B);
  const BtB = matMul3x3(Bt, B);
  const { eigVals, eigVecs: V } = jacobiEigenSymmetric3x3(BtB);
  const s = eigVals.map((v) => Math.sqrt(Math.max(0, v)));

  // Compute U = B V S^{-1} for non-zero singular values, and complete orthonormal basis with det(U) = +1
  const BV = matMul3x3(B, V);
  const uCols = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];

  if (s[0] > 1e-10) {
    uCols[0] = normalize3([BV[0][0] / s[0], BV[1][0] / s[0], BV[2][0] / s[0]]);
  } else {
    uCols[0] = [1, 0, 0];
  }

  if (s[1] > 1e-10) {
    // Orthogonalize against uCols[0]
    let v1 = [BV[0][1] / s[1], BV[1][1] / s[1], BV[2][1] / s[1]];
    const dot01 = v1[0] * uCols[0][0] + v1[1] * uCols[0][1] + v1[2] * uCols[0][2];
    v1 = [v1[0] - dot01 * uCols[0][0], v1[1] - dot01 * uCols[0][1], v1[2] - dot01 * uCols[0][2]];
    uCols[1] = normalize3(v1);
  } else {
    // Pick a unit vector orthogonal to uCols[0]
    const cand = Math.abs(uCols[0][0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    const dot = cand[0] * uCols[0][0] + cand[1] * uCols[0][1] + cand[2] * uCols[0][2];
    uCols[1] = normalize3([cand[0] - dot * uCols[0][0], cand[1] - dot * uCols[0][1], cand[2] - dot * uCols[0][2]]);
  }

  // Third column via right-hand rule u0 x u1 so det(U) = +1
  const u2Right = cross3(uCols[0], uCols[1]);
  // Check sign of <BV[:, 2], u2Right>
  const dot2 = BV[0][2] * u2Right[0] + BV[1][2] * u2Right[1] + BV[2][2] * u2Right[2];
  const sign2 = s[2] > 1e-10 && dot2 < 0 ? -1 : 1;
  uCols[2] = u2Right;

  // Since det(U) = +1 and det(V) = +1, R = U @ V^T is guaranteed proper rotation with det(R) = +1!
  const U = [
    [uCols[0][0], uCols[1][0], uCols[2][0]],
    [uCols[0][1], uCols[1][1], uCols[2][1]],
    [uCols[0][2], uCols[1][2], uCols[2][2]],
  ];
  const R = matMul3x3(U, transpose3x3(V));
  const trace = s[0] + s[1] + sign2 * s[2];
  const rssd = Math.sqrt(Math.max(0, normSqSum - 2 * trace));
  return { rotation: R, rssd };
}

export function applyRotation(R, pts) {
  return pts.map((p) => [
    R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2],
    R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2],
    R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2],
  ]);
}

function getPermutations(arr) {
  if (arr.length <= 1) return [arr.slice()];
  const out = [];
  for (let i = 0; i < arr.length; i++) {
    const rest = [...arr.slice(0, i), ...arr.slice(i + 1)];
    for (const p of getPermutations(rest)) {
      out.push([arr[i], ...p]);
    }
  }
  return out;
}

function enumerateStereoisomerTemplates(template, tol = 1e-5) {
  const reps = [];
  const indices = Array.from({ length: template.length }, (_, i) => i);
  for (const perm of getPermutations(indices)) {
    const candidate = perm.map((idx) => template[idx]);
    if (reps.some((existing) => alignVectors(candidate, existing).rssd < tol)) {
      continue;
    }
    reps.push(candidate);
  }
  return reps;
}

export const STEREOISOMER_TEMPLATES = new Map();
for (const [geom, full] of Object.entries(IDEAL_POLYHEDRA)) {
  for (let n = 3; n <= full.length; n++) {
    STEREOISOMER_TEMPLATES.set(`${geom}:${n}`, enumerateStereoisomerTemplates(full.slice(0, n)));
  }
}

export function classifyStereoisomer(centered, geometry) {
  const templates = STEREOISOMER_TEMPLATES.get(`${geometry}:${centered.length}`);
  if (!templates) return null;
  for (const pt of centered) {
    if (!Number.isFinite(pt[0]) || !Number.isFinite(pt[1]) || !Number.isFinite(pt[2])) return null;
  }
  const unit = centered.map((p) => {
    const n = Math.max(Math.hypot(p[0], p[1], p[2]), 1e-8);
    return [p[0] / n, p[1] / n, p[2] / n];
  });
  let bestIdx = 0;
  let bestRmsd = Infinity;
  const sqrtN = Math.sqrt(centered.length);
  for (let i = 0; i < templates.length; i++) {
    const rmsd = alignVectors(unit, templates[i]).rssd / sqrtN;
    if (rmsd < bestRmsd) {
      bestRmsd = rmsd;
      bestIdx = i;
    }
  }
  return bestIdx;
}

export function idealizeCoords(centered, geometry) {
  const idx = classifyStereoisomer(centered, geometry);
  if (idx === null) return centered.map((p) => [Math.fround(p[0]), Math.fround(p[1]), Math.fround(p[2])]);
  const template = STEREOISOMER_TEMPLATES.get(`${geometry}:${centered.length}`)[idx];
  const unit = centered.map((p) => {
    const n = Math.max(Math.hypot(p[0], p[1], p[2]), 1e-8);
    return [p[0] / n, p[1] / n, p[2] / n];
  });
  const { rotation } = alignVectors(unit, template);
  return applyRotation(rotation, template).map((p) => [
    Math.fround(p[0]),
    Math.fround(p[1]),
    Math.fround(p[2]),
  ]);
}

export function classifyDoubleBondStereoisomer(coords) {
  for (const pt of coords) {
    if (!Number.isFinite(pt[0]) || !Number.isFinite(pt[1]) || !Number.isFinite(pt[2])) return null;
  }
  const axisRaw = [
    coords[1][0] - coords[0][0],
    coords[1][1] - coords[0][1],
    coords[1][2] - coords[0][2],
  ];
  const len = Math.hypot(axisRaw[0], axisRaw[1], axisRaw[2]);
  if (len < 1e-6) return null;
  const axis = [axisRaw[0] / len, axisRaw[1] / len, axisRaw[2] / len];

  const sub0 = [coords[2][0] - coords[0][0], coords[2][1] - coords[0][1], coords[2][2] - coords[0][2]];
  const sub1 = [coords[3][0] - coords[1][0], coords[3][1] - coords[1][1], coords[3][2] - coords[1][2]];
  const dot0 = sub0[0] * axis[0] + sub0[1] * axis[1] + sub0[2] * axis[2];
  const dot1 = sub1[0] * axis[0] + sub1[1] * axis[1] + sub1[2] * axis[2];
  const proj0 = [sub0[0] - dot0 * axis[0], sub0[1] - dot0 * axis[1], sub0[2] - dot0 * axis[2]];
  const proj1 = [sub1[0] - dot1 * axis[0], sub1[1] - dot1 * axis[1], sub1[2] - dot1 * axis[2]];
  const l0 = Math.hypot(proj0[0], proj0[1], proj0[2]);
  const l1 = Math.hypot(proj1[0], proj1[1], proj1[2]);
  if (l0 < 1e-6 || l1 < 1e-6) return null;
  const cosine = (proj0[0] * proj1[0] + proj0[1] * proj1[1] + proj0[2] * proj1[2]) / (l0 * l1);
  if (Math.abs(cosine) < 0.5) return null;
  return cosine > 0 ? 1 : -1;
}

// ---------------------------------------------------------------------------
// Topological & 3D Stereocenter / Double-Bond Perception
// ---------------------------------------------------------------------------

function resolveStereoCoords(indices, crd, intcrd, groups, rawCoords) {
  if (
    crd &&
    indices.every((i) => Number.isFinite(crd[i][0]) && Number.isFinite(crd[i][1]) && Number.isFinite(crd[i][2]))
  ) {
    return crd;
  }
  if (
    intcrd &&
    groups &&
    indices.every(
      (i) =>
        Number.isFinite(intcrd[i][0]) &&
        Number.isFinite(intcrd[i][1]) &&
        Number.isFinite(intcrd[i][2]) &&
        groups[i] === groups[indices[0]]
    )
  ) {
    return intcrd;
  }
  return rawCoords;
}

/**
 * Compute topological Weisfeiler-Lehman symmetry classes on `atoms` excluding center `excludeAtom` (optional).
 */
function computeWlClasses(atoms, atomSubset, adjMap, excludeAtom = -1) {
  let labels = new Map();
  for (const u of atomSubset) {
    if (u === excludeAtom) continue;
    const nbs = adjMap.get(u) || [];
    const deg = nbs.filter((e) => e.to !== excludeAtom).length;
    labels.set(
      u,
      `${atoms.atomic_number?.[u] ?? 0}:${atoms.charge?.[u] ?? 0}:${atoms.is_aromatic?.[u] ? 1 : 0}:${deg}`
    );
  }
  for (let iter = 0; iter < 6; iter++) {
    const next = new Map();
    for (const u of atomSubset) {
      if (u === excludeAtom) continue;
      const nbs = (adjMap.get(u) || [])
        .filter((e) => e.to !== excludeAtom && labels.has(e.to))
        .map((e) => `${e.bt}:${labels.get(e.to)}`)
        .sort();
      next.set(u, `${labels.get(u)}(${nbs.join(',')})`);
    }
    labels = next;
  }
  return labels;
}

// Standard amino acid and nucleotide chiral centers (for `chiral_type` annotation completeness)
const STD_POLYMER_CHIRAL_CENTERS = {
  ALA: { CA: 2 },
  ARG: { CA: 2 },
  ASN: { CA: 2 },
  ASP: { CA: 2 },
  CYS: { CA: 2 },
  GLN: { CA: 2 },
  GLU: { CA: 2 },
  HIS: { CA: 2 },
  ILE: { CA: 2, CB: 2 },
  LEU: { CA: 2 },
  LYS: { CA: 2 },
  MET: { CA: 2 },
  PHE: { CA: 2 },
  PRO: { CA: 2 },
  SER: { CA: 2 },
  THR: { CA: 2, CB: 1 },
  TRP: { CA: 2 },
  TYR: { CA: 2 },
  VAL: { CA: 2 },
  DA: { "C1'": 1, "C3'": 2, "C4'": 1 },
  DC: { "C1'": 1, "C3'": 2, "C4'": 1 },
  DG: { "C1'": 1, "C3'": 2, "C4'": 1 },
  DT: { "C1'": 1, "C3'": 2, "C4'": 1 },
  A: { "C1'": 1, "C2'": 1, "C3'": 2, "C4'": 1 },
  C: { "C1'": 1, "C2'": 1, "C3'": 2, "C4'": 1 },
  G: { "C1'": 1, "C2'": 1, "C3'": 2, "C4'": 1 },
  U: { "C1'": 1, "C2'": 1, "C3'": 2, "C4'": 1 },
};

/**
 * Compute `chiral_type` and `stereogenic_bond_atoms` annotations on `atoms`,
 * and return `{ chiralCenters, doubleBonds }`.
 */
export function ensureStereochemistryAnnotations(atoms) {
  const n = atoms.nAtoms;
  const chiralType = new Array(n).fill(0);
  const stereogenicBondAtoms = new Array(n).fill(false);

  // Build adjacency list over heavy atoms
  const adjMap = new Map();
  for (let i = 0; i < n; i++) adjMap.set(i, []);
  for (const [u, v, bt] of atoms.bonds || []) {
    if (bt === BOND_TYPE.COORDINATION) continue;
    adjMap.get(u).push({ to: v, bt });
    adjMap.get(v).push({ to: u, bt });
  }

  const crd = atoms.annotations?.['condition_coordinate_1_atom'];
  const intcrd = atoms.annotations?.['condition_InternalCoordinate_1_atom'];
  const groups = atoms.annotations?.['condition_CoordinateGroup_1_atom'];

  const resStarts = getResidueSegments(atoms);
  for (let r = 0; r < resStarts.length - 1; r++) {
    const s = resStarts[r];
    const e = resStarts[r + 1];
    const resName = atoms.res_name[s];
    if (resName === '<M>' || /^\d+-\d+$/.test(resName)) continue;

    // Check standard polymer lookup first if not atomized
    const isAtomizedRes = atoms.atomize ? atoms.atomize[s] : !atoms.is_polymer[s];
    if (!isAtomizedRes && STD_POLYMER_CHIRAL_CENTERS[resName]) {
      const map = STD_POLYMER_CHIRAL_CENTERS[resName];
      for (let i = s; i < e; i++) {
        const ct = map[atoms.atom_name[i]];
        if (ct) chiralType[i] = ct;
      }
      continue;
    }

    const resIndices = Array.from({ length: e - s }, (_, k) => s + k);

    // Detect tetrahedral stereocenters in residue
    for (const c of resIndices) {
      if (atoms.is_aromatic?.[c]) continue;
      const nbs = adjMap.get(c) || [];
      if (nbs.length !== 3 && nbs.length !== 4) continue;
      if (nbs.some((edge) => edge.bt !== BOND_TYPE.SINGLE)) continue;

      const atomicNum = atoms.atomic_number?.[c] ?? 0;
      const charge = atoms.charge?.[c] ?? 0;
      // 3-heavy-neighbor atoms are sp3 tetrahedral if carbon (with 1 implicit H),
      // or trivalent pyramidal sulfur/phosphorus (with 1 lone pair / positive charge)
      if (nbs.length === 3) {
        const isTetra3 =
          (atomicNum === 6 && charge === 0) ||
          atomicNum === 16 ||
          atomicNum === 15;
        if (!isTetra3) continue;
      } else {
        if (![6, 7, 14, 15, 16].includes(atomicNum)) continue;
      }

      // _is_false_positive_tetrahedral:
      // 1. P/S bonded to 2+ oxygens
      if (
        (atomicNum === 15 || atomicNum === 16) &&
        nbs.filter((edge) => (atoms.atomic_number?.[edge.to] ?? 0) === 8).length >= 2
      ) {
        continue;
      }
      // 2. Ring atom bonded to metal
      if (
        atoms.is_aromatic?.[c] &&
        nbs.some((edge) => METAL_ELEMENTS.has((atoms.element?.[edge.to] || '').toUpperCase()))
      ) {
        continue;
      }

      // 3. Check topological distinctness of neighbors in residue graph minus c
      const wl = computeWlClasses(atoms, resIndices, adjMap, c);
      const nbLabels = nbs.map((edge) => wl.get(edge.to));
      if (new Set(nbLabels).size !== nbLabels.length) continue;

      // 4. Planarity check (normalized triple product >= 0.05)
      const nbIdxs = nbs.map((edge) => edge.to);
      const allIdxs = [c, ...nbIdxs];
      const srcCoords = resolveStereoCoords(allIdxs, crd, intcrd, groups, atoms.coord);
      const cCoord = srcCoords[c];
      if (!Number.isFinite(cCoord[0])) continue;
      const vecs = nbIdxs.slice(0, 3).map((nb) => [
        srcCoords[nb][0] - cCoord[0],
        srcCoords[nb][1] - cCoord[1],
        srcCoords[nb][2] - cCoord[2],
      ]);
      if (vecs.some((v) => !Number.isFinite(v[0]))) continue;
      const cr = cross3(vecs[1], vecs[2]);
      const tp = Math.abs(vecs[0][0] * cr[0] + vecs[0][1] * cr[1] + vecs[0][2] * cr[2]);
      const n0 = Math.hypot(vecs[0][0], vecs[0][1], vecs[0][2]);
      const n1 = Math.hypot(vecs[1][0], vecs[1][1], vecs[1][2]);
      const n2 = Math.hypot(vecs[2][0], vecs[2][1], vecs[2][2]);
      const normTp = tp / Math.max(n0 * n1 * n2, 1e-8);
      if (normTp < 0.05) continue;

      // Signed volume determines CW (1) vs CCW (2)
      const signedTp = vecs[0][0] * cr[0] + vecs[0][1] * cr[1] + vecs[0][2] * cr[2];
      chiralType[c] = signedTp >= 0 ? CHIRAL_TYPE.CHI_TETRAHEDRAL_CW : CHIRAL_TYPE.CHI_TETRAHEDRAL_CCW;
    }

    // Detect stereogenic double bonds in residue
    const wlFull = computeWlClasses(atoms, resIndices, adjMap, -1);
    for (const u of resIndices) {
      for (const edge of adjMap.get(u) || []) {
        const v = edge.to;
        if (v <= u || v < s || v >= e || edge.bt !== BOND_TYPE.DOUBLE) continue;
        if (atoms.is_aromatic?.[u] || atoms.is_aromatic?.[v]) continue;
        const leftNbs = (adjMap.get(u) || []).filter((e2) => e2.to !== v).map((e2) => e2.to);
        const rightNbs = (adjMap.get(v) || []).filter((e2) => e2.to !== u).map((e2) => e2.to);
        if (leftNbs.length < 1 || leftNbs.length > 2 || rightNbs.length < 1 || rightNbs.length > 2) continue;
        if (leftNbs.length === 2 && wlFull.get(leftNbs[0]) === wlFull.get(leftNbs[1])) continue;
        if (rightNbs.length === 2 && wlFull.get(rightNbs[0]) === wlFull.get(rightNbs[1])) continue;
        stereogenicBondAtoms[u] = true;
        stereogenicBondAtoms[v] = true;
      }
    }
  }

  atoms.chiral_type = chiralType;
  atoms.stereogenic_bond_atoms = stereogenicBondAtoms;
  return { chiralType, stereogenicBondAtoms };
}

export function identifyStereogenicCenters(atoms) {
  if (!atoms.chiral_type) ensureStereochemistryAnnotations(atoms);
  const n = atoms.nAtoms;
  const adj = Array.from({ length: n }, () => []);
  for (const [u, v] of atoms.bonds || []) {
    adj[u].push(v);
    adj[v].push(u);
  }

  const centers = [];
  for (let idx = 0; idx < n; idx++) {
    const ct = atoms.chiral_type[idx];
    if (!ct || ct === CHIRAL_TYPE.CHI_UNSPECIFIED || ct === CHIRAL_TYPE.CHI_OTHER) continue;
    let geometry = null;
    if (
      ct === CHIRAL_TYPE.CHI_TETRAHEDRAL_CW ||
      ct === CHIRAL_TYPE.CHI_TETRAHEDRAL_CCW ||
      ct === CHIRAL_TYPE.CHI_TETRAHEDRAL
    ) {
      geometry = 'tetrahedral';
    } else if (ct === CHIRAL_TYPE.CHI_SQUAREPLANAR) {
      geometry = 'square_planar';
    } else if (ct === CHIRAL_TYPE.CHI_TRIGONALBIPYRAMIDAL) {
      geometry = 'trigonal_bipyramidal';
    } else if (ct === CHIRAL_TYPE.CHI_OCTAHEDRAL) {
      geometry = 'octahedral';
    }
    if (!geometry) continue;
    const bondedSorted = adj[idx].slice().sort((a, b) => a - b);
    if (!STEREOISOMER_TEMPLATES.has(`${geometry}:${bondedSorted.length}`)) continue;
    centers.push({
      center_idx: idx,
      bonded_idxs: bondedSorted,
      geometry,
      chiral_type: ct,
    });
  }
  return centers;
}

export function identifyStereogenicDoubleBonds(atoms) {
  if (!atoms.bonds || atoms.bonds.length === 0) return [];
  if (!atoms.stereogenic_bond_atoms) ensureStereochemistryAnnotations(atoms);
  const n = atoms.nAtoms;
  const adj = Array.from({ length: n }, () => []);
  const doubleBonds = [];
  for (const [u, v, bt] of atoms.bonds) {
    adj[u].push(v);
    adj[v].push(u);
    if (bt === BOND_TYPE.DOUBLE) doubleBonds.push([Math.min(u, v), Math.max(u, v)]);
  }
  if (doubleBonds.length === 0) return [];

  const result = [];
  for (const [left, right] of doubleBonds) {
    if (!atoms.stereogenic_bond_atoms[left] || !atoms.stereogenic_bond_atoms[right]) continue;
    if (atoms.atomize && (!atoms.atomize[left] || !atoms.atomize[right])) continue;
    const leftNeighbors = adj[left].filter((idx) => idx !== right).sort((a, b) => a - b);
    const rightNeighbors = adj[right].filter((idx) => idx !== left).sort((a, b) => a - b);
    if (
      leftNeighbors.length < 1 ||
      leftNeighbors.length > 2 ||
      rightNeighbors.length < 1 ||
      rightNeighbors.length > 2
    ) {
      continue;
    }
    result.push({
      bond_idxs: [left, right],
      neighbor_idxs: [leftNeighbors, rightNeighbors],
      reference_idxs: [left, right, leftNeighbors[0], rightNeighbors[0]],
    });
  }
  return result;
}

/**
 * Compute `chirality`, `chiral_frame_atoms`, `chiral_slot_owner`, `tetrahedral_vertices`,
 * and `is_chiral_center` features for `atoms` (inference mode: deterministic).
 */
export function computeChiralityFeatures(atoms) {
  const n = atoms.nAtoms;
  const nBlocks = MAX_OVERLAPPING_CENTERS;
  const embedDim = nBlocks * 3;

  ensureStereochemistryAnnotations(atoms);

  const chirality = new Float32Array(n * embedDim);
  const chiralFrameAtoms = new Int32Array(n * nBlocks * 8).fill(-1);
  const chiralSlotOwner = new Int32Array(n * nBlocks).fill(-1);
  const tetrahedralVertices = new Int32Array(n * 4).fill(-1);

  const tokenIds = atoms.token_id;
  const isLigand = atoms.is_ligand || atoms.is_polymer.map((p) => !p);
  const hasAtomize = Array.isArray(atoms.atomize);

  // Count atoms per token_id
  const tokenCounts = new Map();
  if (tokenIds) {
    for (let i = 0; i < n; i++) {
      tokenCounts.set(tokenIds[i], (tokenCounts.get(tokenIds[i]) || 0) + 1);
    }
  }
  const atomized = Array.from({ length: n }, (_, i) =>
    Boolean((hasAtomize ? atoms.atomize[i] : true) && (!tokenIds || tokenCounts.get(tokenIds[i]) === 1))
  );

  const centers = identifyStereogenicCenters(atoms);
  if (tokenIds) {
    for (const center of centers) {
      const cidx = center.center_idx;
      const bonded = center.bonded_idxs;
      if (center.geometry !== 'tetrahedral' || (bonded.length !== 3 && bonded.length !== 4)) continue;
      const support = [cidx, ...bonded];
      if (support.every((idx) => isLigand[idx] && atomized[idx])) {
        const vIdxs = bonded.length === 3 ? [...bonded, cidx] : bonded;
        for (let k = 0; k < 4; k++) {
          tetrahedralVertices[cidx * 4 + k] = tokenIds[vIdxs[k]];
        }
      }
    }
  }

  const crd = atoms.annotations?.['condition_coordinate_1_atom'];
  const intcrd = atoms.annotations?.['condition_InternalCoordinate_1_atom'];
  const groups = atoms.annotations?.['condition_CoordinateGroup_1_atom'];

  const frameItems = [];

  // 1. Chiral centers
  for (const center of centers) {
    const cidx = center.center_idx;
    const bonded = center.bonded_idxs;
    if (hasAtomize && !atoms.atomize[cidx]) continue;
    const centerIdxs = [cidx, ...bonded];
    const sourceCoords = resolveStereoCoords(centerIdxs, crd, intcrd, groups, atoms.coord);
    const centered = bonded.map((b) => [
      sourceCoords[b][0] - sourceCoords[cidx][0],
      sourceCoords[b][1] - sourceCoords[cidx][1],
      sourceCoords[b][2] - sourceCoords[cidx][2],
    ]);
    const idealized = idealizeCoords(centered, center.geometry);
    frameItems.push({ isCenter: true, cidx, bonded, rotated: idealized });
  }

  // 2. Stereogenic double bonds
  for (const bond of identifyStereogenicDoubleBonds(atoms)) {
    const [left, right] = bond.bond_idxs;
    const [leftNeighbors, rightNeighbors] = bond.neighbor_idxs;
    const indices = [left, right, ...leftNeighbors, ...rightNeighbors];
    const sourceCoords = resolveStereoCoords(indices, crd, intcrd, groups, atoms.coord);
    const bondCoords = indices.map((idx) => sourceCoords[idx]);
    if (bondCoords.some((c) => !Number.isFinite(c[0]) || !Number.isFinite(c[1]) || !Number.isFinite(c[2]))) {
      continue;
    }
    const refCoords = bond.reference_idxs.map((idx) => sourceCoords[idx]);
    const side = classifyDoubleBondStereoisomer(refCoords);
    if (side === null) continue;
    const template = DOUBLE_BOND_TEMPLATES.get(`${side}:${leftNeighbors.length}:${rightNeighbors.length}`);
    const midpoint = [
      (sourceCoords[left][0] + sourceCoords[right][0]) / 2,
      (sourceCoords[left][1] + sourceCoords[right][1]) / 2,
      (sourceCoords[left][2] + sourceCoords[right][2]) / 2,
    ];
    const centered = bondCoords.map((c) => [c[0] - midpoint[0], c[1] - midpoint[1], c[2] - midpoint[2]]);
    const { rotation } = alignVectors(centered, template);
    const idealized = applyRotation(rotation, template).map((p) => [
      Math.fround(p[0]),
      Math.fround(p[1]),
      Math.fround(p[2]),
    ]);
    frameItems.push({ isCenter: false, cidx: left, bonded: indices, rotated: idealized });
  }

  const atomBlocks = new Map();

  for (const { isCenter, cidx, bonded, rotated } of frameItems) {
    for (let block = 0; block < nBlocks; block++) {
      const offset = block * 3;
      let anyOccupied = false;
      for (const b of bonded) {
        const base = b * embedDim + offset;
        if (chirality[base] !== 0 || chirality[base + 1] !== 0 || chirality[base + 2] !== 0) {
          anyOccupied = true;
          break;
        }
      }
      if (!anyOccupied) {
        const origin = isCenter ? [cidx, cidx] : [bonded[0], bonded[1]];
        const allSupport = [...origin, ...bonded];
        if (tokenIds && allSupport.every((idx) => atomized[idx])) {
          const frameBase = (cidx * nBlocks + block) * 8;
          for (let k = 0; k < allSupport.length; k++) {
            chiralFrameAtoms[frameBase + k] = tokenIds[allSupport[k]];
          }
          for (const b of bonded) {
            chiralSlotOwner[b * nBlocks + block] = tokenIds[cidx] * nBlocks + block;
          }
        }
        for (let k = 0; k < bonded.length; k++) {
          const b = bonded[k];
          const base = b * embedDim + offset;
          chirality[base] = rotated[k][0];
          chirality[base + 1] = rotated[k][1];
          chirality[base + 2] = rotated[k][2];
          if (!atomBlocks.has(b)) atomBlocks.set(b, []);
          atomBlocks.get(b).push(block);
        }
        break;
      }
    }
  }

  // Cycling fill across unoccupied blocks
  for (const [atomIdx, blocks] of atomBlocks.entries()) {
    const occupied = blocks.slice().sort((a, b) => a - b);
    const empty = [];
    for (let b = 0; b < nBlocks; b++) {
      if (!occupied.includes(b)) empty.push(b);
    }
    if (empty.length === 0) continue;
    for (let i = 0; i < empty.length; i++) {
      const targetBlock = empty[i];
      const sourceBlock = occupied[i % occupied.length];
      const dstBase = atomIdx * embedDim + targetBlock * 3;
      const srcBase = atomIdx * embedDim + sourceBlock * 3;
      chirality[dstBase] = chirality[srcBase];
      chirality[dstBase + 1] = chirality[srcBase + 1];
      chirality[dstBase + 2] = chirality[srcBase + 2];
      if (tokenIds) {
        chiralSlotOwner[atomIdx * nBlocks + targetBlock] = chiralSlotOwner[atomIdx * nBlocks + sourceBlock];
      }
    }
  }

  const isChiralCenter = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (atoms.chiral_type[i] !== 0 && (!hasAtomize || atoms.atomize[i])) {
      isChiralCenter[i] = 1;
    }
  }

  return {
    chirality: { data: chirality, shape: [n, embedDim] },
    chiral_frame_atoms: { data: chiralFrameAtoms, shape: [n, nBlocks, 8] },
    chiral_slot_owner: { data: chiralSlotOwner, shape: [n, nBlocks] },
    tetrahedral_vertices: { data: tetrahedralVertices, shape: [n, 4] },
    is_chiral_center: isChiralCenter,
  };
}
