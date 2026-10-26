/**
 * Output structure processing, unindexed motif insertion mapping (Hungarian algorithm),
 * internal-coordinate group grafting, input-frame alignment, and mmCIF / PDB writers
 * for RFD4-Proteina.
 *
 * Mirrors:
 * - `rfproteina/utils/structure_export.py` (`process_single_sample_to_atom_array`, `write_generated_structure_cif`)
 * - `rfproteina/utils/coordinate_transfer.py` (`copy_generated_coordinates_to_input`, `remove_terminal_oxygen_atoms`)
 * - `rfproteina/transforms/utils/conditioning_utils.py` (`insert_internal_coord_conditioned_groups`, `align_atom_array_to_input_frame`)
 * - `rfproteina/inference_utils/insertion_mapping.py` (`build_predicted_insertion_mapping`)
 * - `rfproteina/transforms/unindexing.py` (`annotate_unindexed_insertion`, `_restage_internal_coord_cwi_per_group`)
 */

import {
  UNIFIED_ATOM37_ENCODING,
  UNIFIED_ATOM37_TOKENS,
  UNIFIED_RESTYPE_ATOM37_MASK,
  inferElementSymbol,
} from './encoding.js';

function det3x3(m) {
  return (
    m[0] * (m[4] * m[8] - m[5] * m[7]) -
    m[1] * (m[3] * m[8] - m[5] * m[6]) +
    m[2] * (m[3] * m[7] - m[4] * m[6])
  );
}

function jacobiEigen3x3(sym) {
  const a = new Float64Array(sym);
  const v = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  const pairs = [
    [0, 1],
    [0, 2],
    [1, 2],
  ];
  for (let sweep = 0; sweep < 25; sweep += 1) {
    let maxOff = 0;
    for (const [p, q] of pairs) {
      maxOff = Math.max(maxOff, Math.abs(a[p * 3 + q]));
    }
    if (maxOff < 1e-13) break;
    for (const [p, q] of pairs) {
      const apq = a[p * 3 + q];
      if (Math.abs(apq) < 1e-13) continue;
      const app = a[p * 3 + p];
      const aqq = a[q * 3 + q];
      const tau = (aqq - app) / (2 * apq);
      const t = tau >= 0 ? 1 / (tau + Math.sqrt(1 + tau * tau)) : -1 / (-tau + Math.sqrt(1 + tau * tau));
      const c = 1 / Math.sqrt(1 + t * t);
      const s = t * c;

      a[p * 3 + p] = app - t * apq;
      a[q * 3 + q] = aqq + t * apq;
      a[p * 3 + q] = 0;
      a[q * 3 + p] = 0;

      const r = 3 - p - q;
      const arp = a[r * 3 + p];
      const arq = a[r * 3 + q];
      a[r * 3 + p] = c * arp - s * arq;
      a[p * 3 + r] = a[r * 3 + p];
      a[r * 3 + q] = s * arp + c * arq;
      a[q * 3 + r] = a[r * 3 + q];

      for (let k = 0; k < 3; k += 1) {
        const vkp = v[k * 3 + p];
        const vkq = v[k * 3 + q];
        v[k * 3 + p] = c * vkp - s * vkq;
        v[k * 3 + q] = s * vkp + c * vkq;
      }
    }
  }

  const order = [0, 1, 2].sort((i, j) => a[j * 3 + j] - a[i * 3 + i]);
  const values = new Float64Array(3);
  const vectors = new Float64Array(9);
  for (let col = 0; col < 3; col += 1) {
    const srcCol = order[col];
    values[col] = Math.max(0, a[srcCol * 3 + srcCol]);
    for (let row = 0; row < 3; row += 1) {
      vectors[row * 3 + col] = v[row * 3 + srcCol];
    }
  }
  if (det3x3(vectors) < 0) {
    vectors[0 * 3 + 2] = -vectors[0 * 3 + 2];
    vectors[1 * 3 + 2] = -vectors[1 * 3 + 2];
    vectors[2 * 3 + 2] = -vectors[2 * 3 + 2];
  }
  return { values, vectors };
}

/**
 * Kabsch superimposition: aligns `mobileCoords` (`[N * 3]`) onto `fixedCoords` (`[N * 3]`).
 */
export function superimposeCoords3D(fixedCoords, mobileCoords) {
  const n = Math.floor(Math.min(fixedCoords.length, mobileCoords.length) / 3);
  if (n === 0) {
    return {
      aligned: new Float32Array(0),
      rotation: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
      fixedCenter: new Float64Array(3),
      mobileCenter: new Float64Array(3),
      rmsd: 0,
    };
  }
  const fixedCenter = new Float64Array(3);
  const mobileCenter = new Float64Array(3);
  for (let i = 0; i < n; i += 1) {
    fixedCenter[0] += fixedCoords[i * 3 + 0];
    fixedCenter[1] += fixedCoords[i * 3 + 1];
    fixedCenter[2] += fixedCoords[i * 3 + 2];
    mobileCenter[0] += mobileCoords[i * 3 + 0];
    mobileCenter[1] += mobileCoords[i * 3 + 1];
    mobileCenter[2] += mobileCoords[i * 3 + 2];
  }
  fixedCenter[0] /= n;
  fixedCenter[1] /= n;
  fixedCenter[2] /= n;
  mobileCenter[0] /= n;
  mobileCenter[1] /= n;
  mobileCenter[2] /= n;

  const h = new Float64Array(9);
  for (let i = 0; i < n; i += 1) {
    const mx = mobileCoords[i * 3 + 0] - mobileCenter[0];
    const my = mobileCoords[i * 3 + 1] - mobileCenter[1];
    const mz = mobileCoords[i * 3 + 2] - mobileCenter[2];
    const fx = fixedCoords[i * 3 + 0] - fixedCenter[0];
    const fy = fixedCoords[i * 3 + 1] - fixedCenter[1];
    const fz = fixedCoords[i * 3 + 2] - fixedCenter[2];
    h[0] += mx * fx;
    h[1] += mx * fy;
    h[2] += mx * fz;
    h[3] += my * fx;
    h[4] += my * fy;
    h[5] += my * fz;
    h[6] += mz * fx;
    h[7] += mz * fy;
    h[8] += mz * fz;
  }

  const hth = new Float64Array(9);
  for (let r = 0; r < 3; r += 1) {
    for (let c = 0; c < 3; c += 1) {
      hth[r * 3 + c] =
        h[0 * 3 + r] * h[0 * 3 + c] +
        h[1 * 3 + r] * h[1 * 3 + c] +
        h[2 * 3 + r] * h[2 * 3 + c];
    }
  }
  const { values, vectors: V } = jacobiEigen3x3(hth);

  const U = new Float64Array(9);
  for (let col = 0; col < 2; col += 1) {
    const sigma = Math.sqrt(values[col]);
    if (sigma > 1e-9) {
      for (let row = 0; row < 3; row += 1) {
        U[row * 3 + col] =
          (h[row * 3 + 0] * V[0 * 3 + col] +
            h[row * 3 + 1] * V[1 * 3 + col] +
            h[row * 3 + 2] * V[2 * 3 + col]) / sigma;
      }
    } else if (col === 0) {
      U[0] = 1;
    } else {
      const u0x = U[0];
      const u0y = U[3];
      const u0z = U[6];
      let vx = Math.abs(u0x) < 0.9 ? 1 : 0;
      let vy = Math.abs(u0x) < 0.9 ? 0 : 1;
      let vz = 0;
      const dot = vx * u0x + vy * u0y + vz * u0z;
      vx -= dot * u0x;
      vy -= dot * u0y;
      vz -= dot * u0z;
      const norm = Math.hypot(vx, vy, vz) || 1;
      U[1] = vx / norm;
      U[4] = vy / norm;
      U[7] = vz / norm;
    }
  }
  U[0 * 3 + 2] = U[1 * 3 + 0] * U[2 * 3 + 1] - U[2 * 3 + 0] * U[1 * 3 + 1];
  U[1 * 3 + 2] = U[2 * 3 + 0] * U[0 * 3 + 1] - U[0 * 3 + 0] * U[2 * 3 + 1];
  U[2 * 3 + 2] = U[0 * 3 + 0] * U[1 * 3 + 1] - U[1 * 3 + 0] * U[0 * 3 + 1];

  // 🔴 NO det(H) SIGN HERE: V is right-handed (`jacobiEigen3x3` makes it so) and U's third
  // column is the cross product of its first two, so V U^T is already the best PROPER rotation -
  // the Kabsch correction is built in. Flipping the third axis by det(H)'s sign, as this did,
  // returned a REFLECTION exactly when the correction mattered (a design's motif fitted onto a
  // mirror-like reference): det -1 and an RMSD below the best a rotation can reach.
  const R = new Float64Array(9);
  for (let r = 0; r < 3; r += 1) {
    for (let c = 0; c < 3; c += 1) {
      R[r * 3 + c] =
        V[r * 3 + 0] * U[c * 3 + 0] +
        V[r * 3 + 1] * U[c * 3 + 1] +
        V[r * 3 + 2] * U[c * 3 + 2];
    }
  }

  const aligned = new Float32Array(n * 3);
  let sqErr = 0;
  for (let i = 0; i < n; i += 1) {
    const mx = mobileCoords[i * 3 + 0] - mobileCenter[0];
    const my = mobileCoords[i * 3 + 1] - mobileCenter[1];
    const mz = mobileCoords[i * 3 + 2] - mobileCenter[2];
    const ax = mx * R[0] + my * R[1] + mz * R[2] + fixedCenter[0];
    const ay = mx * R[3] + my * R[4] + mz * R[5] + fixedCenter[1];
    const az = mx * R[6] + my * R[7] + mz * R[8] + fixedCenter[2];
    aligned[i * 3 + 0] = ax;
    aligned[i * 3 + 1] = ay;
    aligned[i * 3 + 2] = az;
    const dx = ax - fixedCoords[i * 3 + 0];
    const dy = ay - fixedCoords[i * 3 + 1];
    const dz = az - fixedCoords[i * 3 + 2];
    sqErr += dx * dx + dy * dy + dz * dz;
  }

  return {
    aligned,
    rotation: R,
    fixedCenter,
    mobileCenter,
    rmsd: Math.sqrt(sqErr / n),
  };
}

/**
 * Standard CCD heavy-atom ordering (excluding terminal leaving oxygens `OXT` / `OP3`)
 * used by `get_empty_ccd_template` and `biotite.structure.info.standardize_order`
 * in `copy_generated_coordinates_to_input`.
 */
export const CCD_STANDARD_ATOM_ORDER = Object.freeze({
  // 20 canonical amino acids
  ALA: Object.freeze(['N', 'CA', 'C', 'O', 'CB']),
  ARG: Object.freeze(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2']),
  ASN: Object.freeze(['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2']),
  ASP: Object.freeze(['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2']),
  CYS: Object.freeze(['N', 'CA', 'C', 'O', 'CB', 'SG']),
  GLN: Object.freeze(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2']),
  GLU: Object.freeze(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2']),
  GLY: Object.freeze(['N', 'CA', 'C', 'O']),
  HIS: Object.freeze(['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2']),
  ILE: Object.freeze(['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1']),
  LEU: Object.freeze(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2']),
  LYS: Object.freeze(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ']),
  MET: Object.freeze(['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE']),
  PHE: Object.freeze(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']),
  PRO: Object.freeze(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD']),
  SER: Object.freeze(['N', 'CA', 'C', 'O', 'CB', 'OG']),
  THR: Object.freeze(['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2']),
  TRP: Object.freeze(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2']),
  TYR: Object.freeze(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH']),
  VAL: Object.freeze(['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2']),

  // 4 canonical DNA nucleotides
  DA: Object.freeze(["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"]),
  DC: Object.freeze(["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"]),
  DG: Object.freeze(["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"]),
  DT: Object.freeze(["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C7", "C6"]),

  // 4 canonical RNA nucleotides
  A: Object.freeze(["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"]),
  C: Object.freeze(["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"]),
  G: Object.freeze(["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"]),
  U: Object.freeze(["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"]),
});

/**
 * Return the ordered list of heavy atom names for a residue token (with terminal leaving
 * oxygens `OXT` and `OP3` removed, matching `remove_terminal_oxygen_atoms`).
 *
 * @param {string} resName
 * @returns {readonly string[]}
 */
export function getOrderedAtomNamesForResidue(resName) {
  if (CCD_STANDARD_ATOM_ORDER[resName]) {
    return CCD_STANDARD_ATOM_ORDER[resName];
  }
  const tokenAtoms = UNIFIED_ATOM37_ENCODING.token_atoms[resName];
  if (!tokenAtoms) return [];
  return tokenAtoms.filter((name) => Boolean(name) && name !== 'OXT' && name !== 'OP3');
}

/**
 * Exact Hungarian / Jonker-Volgenant linear sum assignment minimizing total cost
 * on a rectangular `nRows x nCols` matrix (`scipy.optimize.linear_sum_assignment(cost, maximize=False)`).
 *
 * @param {ArrayLike<number>} costMatrix Flat row-major `[nRows * nCols]` cost matrix
 * @param {number} nRows
 * @param {number} nCols
 * @returns {{ rowInd: Int32Array, colInd: Int32Array }}
 */
export function linearSumAssignmentMinimize(costMatrix, nRows, nCols) {
  if (nRows <= 0 || nCols <= 0) {
    return { rowInd: new Int32Array(0), colInd: new Int32Array(0) };
  }
  if (nRows > nCols) {
    const transposed = new Float64Array(nCols * nRows);
    for (let r = 0; r < nRows; r += 1) {
      for (let c = 0; c < nCols; c += 1) {
        transposed[c * nRows + r] = costMatrix[r * nCols + c];
      }
    }
    const { rowInd: cInd, colInd: rInd } = linearSumAssignmentMinimize(transposed, nCols, nRows);
    const pairs = [];
    for (let k = 0; k < cInd.length; k += 1) {
      pairs.push([rInd[k], cInd[k]]);
    }
    pairs.sort((a, b) => a[0] - b[0]);
    const rowInd = new Int32Array(pairs.length);
    const colInd = new Int32Array(pairs.length);
    for (let k = 0; k < pairs.length; k += 1) {
      rowInd[k] = pairs[k][0];
      colInd[k] = pairs[k][1];
    }
    return { rowInd, colInd };
  }

  // Replace non-finite +Infinity entries with a large finite penalty so dual potentials stay finite
  let maxFinite = 0;
  for (let i = 0; i < nRows * nCols; i += 1) {
    const v = Math.abs(costMatrix[i]);
    if (Number.isFinite(v) && v > maxFinite) maxFinite = v;
  }
  const infPenalty = (maxFinite + 1.0) * (nRows + 1) * 1e6;

  const u = new Float64Array(nRows + 1);
  const v = new Float64Array(nCols + 1);
  const p = new Int32Array(nCols + 1);
  const way = new Int32Array(nCols + 1);
  const minv = new Float64Array(nCols + 1);
  const used = new Uint8Array(nCols + 1);

  for (let i = 1; i <= nRows; i += 1) {
    p[0] = i;
    let j0 = 0;
    minv.fill(Infinity);
    used.fill(0);

    do {
      used[j0] = 1;
      const i0 = p[j0];
      let delta = Infinity;
      let j1 = 0;
      const rowOffset = (i0 - 1) * nCols;
      for (let j = 1; j <= nCols; j += 1) {
        if (!used[j]) {
          const raw = costMatrix[rowOffset + (j - 1)];
          const cost = Number.isFinite(raw) ? raw : infPenalty;
          const cur = cost - u[i0] - v[j];
          if (cur < minv[j]) {
            minv[j] = cur;
            way[j] = j0;
          }
          if (minv[j] < delta) {
            delta = minv[j];
            j1 = j;
          }
        }
      }
      for (let j = 0; j <= nCols; j += 1) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);

    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 !== 0);
  }

  const rowInd = new Int32Array(nRows);
  const colInd = new Int32Array(nRows);
  for (let r = 0; r < nRows; r += 1) rowInd[r] = r;
  for (let j = 1; j <= nCols; j += 1) {
    if (p[j] > 0) {
      colInd[p[j] - 1] = j - 1;
    }
  }
  return { rowInd, colInd };
}

/**
 * Exact Hungarian / Jonker-Volgenant linear sum assignment maximizing total score
 * on a rectangular `nRows x nCols` matrix (`scipy.optimize.linear_sum_assignment(score, maximize=True)`).
 *
 * @param {ArrayLike<number>} scoreMatrix Flat row-major `[nRows * nCols]` score matrix
 * @param {number} nRows
 * @param {number} nCols
 * @returns {{ rowInd: Int32Array, colInd: Int32Array }}
 */
export function linearSumAssignmentMaximize(scoreMatrix, nRows, nCols) {
  if (nRows <= 0 || nCols <= 0) {
    return { rowInd: new Int32Array(0), colInd: new Int32Array(0) };
  }
  let maxVal = -Infinity;
  for (let i = 0; i < nRows * nCols; i += 1) {
    const v = scoreMatrix[i];
    if (Number.isFinite(v) && v > maxVal) maxVal = v;
  }
  if (!Number.isFinite(maxVal)) maxVal = 0;

  const costMatrix = new Float64Array(nRows * nCols);
  for (let i = 0; i < nRows * nCols; i += 1) {
    const v = scoreMatrix[i];
    costMatrix[i] = Number.isFinite(v) ? maxVal - v : Infinity;
  }
  return linearSumAssignmentMinimize(costMatrix, nRows, nCols);
}

/**
 * Extract per-token metadata (`tokenStarts`, `tokenIds`, `hasIndex`) from `tokenMeta` or `inputAtoms`.
 */
function resolveTokenMeta(nValid, tokenMeta) {
  if (!tokenMeta) {
    const tokenIds = new Int32Array(nValid);
    const hasIndex = new Uint8Array(nValid);
    for (let i = 0; i < nValid; i += 1) {
      tokenIds[i] = i;
      hasIndex[i] = 1;
    }
    return { nValid, tokenIds, hasIndex };
  }

  // Case 1: inputAtoms dictionary from `<case>.atoms.json` (with `token_id` and `condition_index_1_residue`)
  if (Array.isArray(tokenMeta.token_id) || ArrayBuffer.isView(tokenMeta.token_id)) {
    const atomTokenIds = tokenMeta.token_id;
    const atomHasIdx = tokenMeta.condition_index_1_residue;
    const tokenStarts = [];
    let prevTok = -1;
    for (let i = 0; i < atomTokenIds.length; i += 1) {
      if (atomTokenIds[i] !== prevTok) {
        tokenStarts.push(i);
        prevTok = atomTokenIds[i];
      }
    }
    const count = nValid != null ? Math.min(nValid, tokenStarts.length) : tokenStarts.length;
    const tokenIds = new Int32Array(count);
    const hasIndex = new Uint8Array(count);
    const chains = new Array(count);
    for (let t = 0; t < count; t += 1) {
      const a0 = tokenStarts[t];
      tokenIds[t] = atomTokenIds[a0];
      hasIndex[t] = atomHasIdx ? (atomHasIdx[a0] ? 1 : 0) : 1;
      chains[t] = tokenMeta.chain_id?.[a0] ?? '';
    }
    return { nValid: count, tokenIds, hasIndex, chains };
  }

  // Case 2: `{ hasIndex, tokenIds }`
  if (tokenMeta.hasIndex) {
    const count = nValid != null ? nValid : tokenMeta.hasIndex.length;
    const tokenIds = new Int32Array(count);
    const hasIndex = new Uint8Array(count);
    for (let t = 0; t < count; t += 1) {
      tokenIds[t] = tokenMeta.tokenIds ? tokenMeta.tokenIds[t] : t;
      hasIndex[t] = tokenMeta.hasIndex[t] ? 1 : 0;
    }
    return { nValid: count, tokenIds, hasIndex };
  }

  // Case 3: Array of token descriptors `[{ tokenId, hasIndex, isUnindexed }]`
  if (Array.isArray(tokenMeta)) {
    const count = nValid != null ? Math.min(nValid, tokenMeta.length) : tokenMeta.length;
    const tokenIds = new Int32Array(count);
    const hasIndex = new Uint8Array(count);
    for (let t = 0; t < count; t += 1) {
      const item = tokenMeta[t] ?? {};
      tokenIds[t] = item.tokenId ?? item.token_id ?? t;
      if (item.hasIndex !== undefined) {
        hasIndex[t] = item.hasIndex ? 1 : 0;
      } else if (item.condition_index_1_residue !== undefined) {
        hasIndex[t] = item.condition_index_1_residue ? 1 : 0;
      } else if (item.isUnindexed !== undefined) {
        hasIndex[t] = item.isUnindexed ? 0 : 1;
      } else {
        hasIndex[t] = 1;
      }
    }
    return { nValid: count, tokenIds, hasIndex };
  }

  throw new Error('Unsupported tokenMeta format in buildPredictedInsertionMapping');
}

/**
 * Build `{ [unindexedTokenId]: targetIndexedTokenId }` from pairwise insertion logits
 * via rectangular Hungarian assignment (`build_predicted_insertion_mapping` in
 * `rfproteina/inference_utils/insertion_mapping.py`).
 *
 * @param {ArrayLike<number>} insertionLogits Flat `[nFull * nFull]` (or `[nValid * nValid]`) insertion logits
 * @param {number|null} nValid Number of non-padded tokens
 * @param {object|Array} tokenMeta Input atom table (`atoms.json`), `{ hasIndex, tokenIds }`, or token array
 * @param {number} [logitStride] Row stride of `insertionLogits` (defaults to `Math.round(Math.sqrt(insertionLogits.length))`)
 * @returns {Record<number, number>}
 */
export function buildPredictedInsertionMapping(insertionLogits, nValid, tokenMeta, logitStride = null) {
  if (!insertionLogits || insertionLogits.length === 0) return {};
  const meta = resolveTokenMeta(nValid, tokenMeta);
  const count = meta.nValid;
  const stride = logitStride ?? Math.round(Math.sqrt(insertionLogits.length));

  const unindexedPositions = [];
  const indexedPositions = [];
  for (let p = 0; p < count; p += 1) {
    if (meta.hasIndex[p]) {
      indexedPositions.push(p);
    } else {
      unindexedPositions.push(p);
    }
  }

  if (unindexedPositions.length === 0 || indexedPositions.length === 0) {
    return {};
  }

  // 🔴 ONE ASSIGNMENT PER CHAIN, as upstream solves it: an unindexed token maps onto a distinct
  // generated token of its OWN chain, never across chains (chains in sorted order, one with no
  // generated token left unmapped).
  const chainOf = (p) => meta.chains?.[p] ?? '';
  const mapping = {};
  for (const chain of [...new Set(unindexedPositions.map(chainOf))].sort()) {
    const rows = unindexedPositions.filter((p) => chainOf(p) === chain);
    const cols = indexedPositions.filter((p) => chainOf(p) === chain);
    if (cols.length === 0) continue;
    const scoreMatrix = new Float64Array(rows.length * cols.length);
    for (let r = 0; r < rows.length; r += 1) {
      for (let c = 0; c < cols.length; c += 1) {
        scoreMatrix[r * cols.length + c] = insertionLogits[rows[r] * stride + cols[c]];
      }
    }
    const { rowInd, colInd } = linearSumAssignmentMaximize(scoreMatrix, rows.length, cols.length);
    for (let k = 0; k < rowInd.length; k += 1) {
      mapping[meta.tokenIds[rows[rowInd[k]]]] = meta.tokenIds[cols[colInd[k]]];
    }
  }
  return mapping;
}

/**
 * Match inserted unindexed token atoms to target generated token atoms by atom name,
 * with element + distance Hungarian fallback (`_get_target_indices_corresponding_to_inserted`
 * in `rfproteina/transforms/unindexing.py`).
 *
 * @param {Array<{ atomName: string, element: string, x: number, y: number, z: number }>} srcAtoms
 * @param {Array<{ atomName: string, element: string, x: number, y: number, z: number }>} targetAtoms
 * @param {boolean} [allowElementFallback=true]
 * @returns {Int32Array|null} Target indices corresponding to each `srcAtoms` entry, or `null` on failure
 */
function matchInsertedAtomsToTarget(srcAtoms, targetAtoms, allowElementFallback = true) {
  const nSrc = srcAtoms.length;
  const nTgt = targetAtoms.length;
  const targetByName = new Map();
  for (let t = 0; t < nTgt; t += 1) {
    targetByName.set(targetAtoms[t].atomName, t);
  }

  const result = new Int32Array(nSrc).fill(-1);
  const usedTarget = new Uint8Array(nTgt);
  const unmatchedSrc = [];

  for (let s = 0; s < nSrc; s += 1) {
    const tIdx = targetByName.get(srcAtoms[s].atomName);
    if (tIdx !== undefined && !usedTarget[tIdx]) {
      result[s] = tIdx;
      usedTarget[tIdx] = 1;
    } else {
      unmatchedSrc.push(s);
    }
  }

  if (unmatchedSrc.length === 0) {
    return result;
  }
  if (!allowElementFallback) {
    return null;
  }

  const unmatchedTgt = [];
  for (let t = 0; t < nTgt; t += 1) {
    if (!usedTarget[t]) unmatchedTgt.push(t);
  }
  if (unmatchedSrc.length > unmatchedTgt.length) {
    return null;
  }

  const nRows = unmatchedSrc.length;
  const nCols = unmatchedTgt.length;
  const dists = new Float64Array(nRows * nCols);
  for (let r = 0; r < nRows; r += 1) {
    const sa = srcAtoms[unmatchedSrc[r]];
    for (let c = 0; c < nCols; c += 1) {
      const ta = targetAtoms[unmatchedTgt[c]];
      if (sa.element !== ta.element) {
        dists[r * nCols + c] = Infinity;
      } else {
        dists[r * nCols + c] = Math.hypot(sa.x - ta.x, sa.y - ta.y, sa.z - ta.z);
      }
    }
  }

  const { rowInd, colInd } = linearSumAssignmentMinimize(dists, nRows, nCols);
  for (let k = 0; k < rowInd.length; k += 1) {
    const r = rowInd[k];
    const c = colInd[k];
    if (!Number.isFinite(dists[r * nCols + c])) {
      return null;
    }
    result[unmatchedSrc[r]] = unmatchedTgt[c];
  }

  for (let s = 0; s < nSrc; s += 1) {
    if (result[s] < 0) return null;
  }
  return result;
}

/**
 * Superimpose internal-coordinate conditioned groups (`coordGroup > 0`) onto generated
 * coordinates and update `cwiCoords` (`SA_CWI`), matching `insert_internal_coord_conditioned_groups`
 * (`rfproteina/transforms/utils/conditioning_utils.py`) and `_restage_internal_coord_cwi_per_group`
 * (`rfproteina/transforms/unindexing.py`).
 *
 * @param {Array<{ x: number, y: number, z: number, coordGroup: number, intCrd: number[]|null }>} atoms
 * @param {Float32Array} cwiCoords Flat `[nAtoms * 3]` coordinates with insertion
 * @returns {Float32Array} Updated `cwiCoords`
 */
export function insertInternalCoordConditionedGroups(atoms, cwiCoords) {
  const byGroup = new Map();
  for (let i = 0; i < atoms.length; i += 1) {
    const a = atoms[i];
    const gid = a.coordGroup ?? -1;
    if (gid > 0 && a.intCrd && Number.isFinite(a.intCrd[0]) && Number.isFinite(a.intCrd[1]) && Number.isFinite(a.intCrd[2])) {
      if (!byGroup.has(gid)) byGroup.set(gid, []);
      byGroup.get(gid).push(i);
    }
  }

  for (const [, indices] of byGroup.entries()) {
    if (indices.length === 0) continue;
    const fixed = new Float32Array(indices.length * 3);
    const mobile = new Float32Array(indices.length * 3);
    for (let k = 0; k < indices.length; k += 1) {
      const idx = indices[k];
      const a = atoms[idx];
      fixed[k * 3 + 0] = a.x;
      fixed[k * 3 + 1] = a.y;
      fixed[k * 3 + 2] = a.z;
      mobile[k * 3 + 0] = a.intCrd[0];
      mobile[k * 3 + 1] = a.intCrd[1];
      mobile[k * 3 + 2] = a.intCrd[2];
    }
    const { aligned } = superimposeCoords3D(fixed, mobile);
    for (let k = 0; k < indices.length; k += 1) {
      const idx = indices[k];
      cwiCoords[idx * 3 + 0] = aligned[k * 3 + 0];
      cwiCoords[idx * 3 + 1] = aligned[k * 3 + 1];
      cwiCoords[idx * 3 + 2] = aligned[k * 3 + 2];
    }
  }
  return cwiCoords;
}

/**
 * Rigidly align all atom coordinates onto the input frame defined by `cwiCoords`
 * at `anchorIndices` (`C_CRD.mask(atom_array)`), matching `align_atom_array_to_input_frame`
 * (`rfproteina/transforms/utils/conditioning_utils.py`).
 *
 * @param {Array<{ x: number, y: number, z: number }>} atoms
 * @param {Float32Array} cwiCoords Flat `[nAtoms * 3]` reference coordinates (`SA_CWI`)
 * @param {number[]} anchorIndices Atom indices where `C_CRD` is finite
 */
export function alignAtomsToInputFrame(atoms, cwiCoords, anchorIndices) {
  // 🔴 THREE ANCHORS OR NONE, as upstream's `anchor_mask.sum() < 3` has it: one or two atoms
  // (a lone ion under C_CRD) do not fix a frame, and this turned the design about them anyway.
  // Upstream's fallback to the largest internal group is not here because it is the identity:
  // that group's reference was itself fitted onto the design (see the README).
  if (!anchorIndices || anchorIndices.length < 3) return atoms;

  const nAnchors = anchorIndices.length;
  const fixed = new Float32Array(nAnchors * 3);
  const mobile = new Float32Array(nAnchors * 3);
  for (let k = 0; k < nAnchors; k += 1) {
    const idx = anchorIndices[k];
    const a = atoms[idx];
    fixed[k * 3 + 0] = cwiCoords[idx * 3 + 0];
    fixed[k * 3 + 1] = cwiCoords[idx * 3 + 1];
    fixed[k * 3 + 2] = cwiCoords[idx * 3 + 2];
    mobile[k * 3 + 0] = a.x;
    mobile[k * 3 + 1] = a.y;
    mobile[k * 3 + 2] = a.z;
  }

  const { rotation: R, fixedCenter, mobileCenter } = superimposeCoords3D(fixed, mobile);
  for (let i = 0; i < atoms.length; i += 1) {
    const a = atoms[i];
    const mx = a.x - mobileCenter[0];
    const my = a.y - mobileCenter[1];
    const mz = a.z - mobileCenter[2];
    a.x = Math.fround(mx * R[0] + my * R[1] + mz * R[2] + fixedCenter[0]);
    a.y = Math.fround(mx * R[3] + my * R[4] + mz * R[5] + fixedCenter[1]);
    a.z = Math.fround(mx * R[6] + my * R[7] + mz * R[8] + fixedCenter[2]);
  }
  return atoms;
}

/**
 * Convert a single decoded sample (`coors`, `atomMask`, `residueType`, `paddingMask`, `insertionLogits`)
 * and `inputAtoms` (or `batch`) into the final aligned atom array, mirroring:
 * 1. `process_single_sample_to_atom_array` (`structure_export.py`)
 * 2. `copy_generated_coordinates_to_input` (`coordinate_transfer.py`)
 * 3. `annotate_unindexed_insertion` (`unindexing.py`)
 * 4. `align_atom_array_to_input_frame` (`conditioning_utils.py`)
 *
 * @param {object} params
 * @param {Float32Array} params.coors `[n, 37, 3]` coordinates in Å
 * @param {Uint8Array|Int32Array} params.atomMask `[n, 37]` boolean mask
 * @param {Int32Array} params.residueType `[n]` residue type indices (`-1` for atomized `<A>`)
 * @param {Uint8Array|Int32Array} [params.paddingMask] `[n]` valid token mask
 * @param {Float32Array|null} [params.insertionLogits] `[n, n]` insertion head logits
 * @param {object|null} [params.inputAtoms] Input atom table (`<case>.atoms.json`)
 * @param {object|null} [params.batch] Optional batch dictionary fallback when `inputAtoms` is absent
 * @returns {{
 *   atoms: Array<object>,
 *   predictedInsertionMapping: Record<number, number>,
 *   cwiCoords: Float32Array,
 * }}
 */
export function processSampleToAtoms({
  coors,
  atomMask,
  residueType,
  paddingMask = null,
  insertionLogits = null,
  inputAtoms = null,
  batch = null,
}) {
  const nTotal = residueType.length;

  // Fallback path when full `inputAtoms` table is not provided (e.g. standalone browser call)
  if (!inputAtoms || !inputAtoms.token_id) {
    const atoms = [];
    let serial = 1;
    for (let tid = 0; tid < nTotal; tid += 1) {
      if (paddingMask && !paddingMask[tid]) continue;
      const rt = residueType[tid];
      const isAtomized = rt < 0 || UNIFIED_ATOM37_TOKENS[rt] === '<A>';
      if (isAtomized) {
        const base = (tid * 37 + 1) * 3;
        atoms.push({
          serial: serial++,
          groupPdb: 'HETATM',
          isHetero: true,
          atomName: 'CA',
          resName: 'UNK',
          chainId: 'A',
          resId: tid + 1,
          insCode: '',
          x: coors[base + 0],
          y: coors[base + 1],
          z: coors[base + 2],
          rawX: coors[base + 0],
          rawY: coors[base + 1],
          rawZ: coors[base + 2],
          element: 'C',
          occupancy: 1.0,
          bFactor: NaN,
          charge: 0,
          entityId: 0,
          authSeqId: tid + 1,
          tokenId: tid,
          isUnindexedInsertion: false,
          crdSrcComponent: '.',
          srcComponent: '.',
          coordGroup: -1,
          crd: null,
          intCrd: null,
        });
      } else {
        const resName = UNIFIED_ATOM37_TOKENS[rt] || 'ALA';
        const slotMap = UNIFIED_ATOM37_ENCODING.token_slot_map[resName] || {};
        const order = getOrderedAtomNamesForResidue(resName);
        for (const atomName of order) {
          const slot = slotMap[atomName];
          if (slot === undefined) continue;
          const present = atomMask ? atomMask[tid * 37 + slot] !== 0 : Boolean(UNIFIED_RESTYPE_ATOM37_MASK[rt]?.[slot]);
          if (!present) continue;
          const base = (tid * 37 + slot) * 3;
          atoms.push({
            serial: serial++,
            groupPdb: 'ATOM',
            isHetero: false,
            atomName,
            resName,
            chainId: 'A',
            resId: tid + 1,
            insCode: '',
            x: coors[base + 0],
            y: coors[base + 1],
            z: coors[base + 2],
            rawX: coors[base + 0],
            rawY: coors[base + 1],
            rawZ: coors[base + 2],
            element: inferElementSymbol(atomName),
            occupancy: 1.0,
            bFactor: NaN,
            charge: 0,
            entityId: 0,
            authSeqId: tid + 1,
            tokenId: tid,
            isUnindexedInsertion: false,
            crdSrcComponent: '.',
            srcComponent: '.',
            coordGroup: -1,
            crd: null,
            intCrd: null,
          });
        }
      }
    }
    return {
      atoms,
      predictedInsertionMapping: {},
      cwiCoords: new Float32Array(atoms.length * 3),
    };
  }

  // Group `inputAtoms` indices by `token_id`
  const tokenMap = new Map();
  for (let i = 0; i < inputAtoms.token_id.length; i += 1) {
    const tid = inputAtoms.token_id[i];
    if (!tokenMap.has(tid)) tokenMap.set(tid, []);
    tokenMap.get(tid).push(i);
  }

  const nValid = tokenMap.size;
  const idxTids = [];
  const unidxTids = [];
  for (const [tid, aIndices] of tokenMap.entries()) {
    if (paddingMask && !paddingMask[tid]) continue;
    const a0 = aIndices[0];
    const hasIdx = inputAtoms.condition_index_1_residue ? Boolean(inputAtoms.condition_index_1_residue[a0]) : true;
    if (hasIdx) {
      idxTids.push(tid);
    } else {
      unidxTids.push(tid);
    }
  }

  // Build initial generated atom records per indexed token (`copy_generated_coordinates_to_input`)
  const tokensAtoms = new Map();
  const outAtoms = [];

  for (const tid of idxTids) {
    const aIndices = tokenMap.get(tid);
    const a0 = aIndices[0];
    const isAtomized = Boolean(inputAtoms.atomize?.[a0]);
    const tokenAtomList = [];

    if (isAtomized) {
      const base = (tid * 37 + 1) * 3;
      const crdRaw = inputAtoms.condition_coordinate_1_atom?.[a0] ?? null;
      const hasCrd = Array.isArray(crdRaw) && crdRaw[0] !== null && Number.isFinite(crdRaw[0]);
      const intCrdRaw = inputAtoms.condition_InternalCoordinate_1_atom?.[a0] ?? null;
      const hasIntCrd = Array.isArray(intCrdRaw) && intCrdRaw[0] !== null && Number.isFinite(intCrdRaw[0]);
      const isHet = inputAtoms.hetero ? Boolean(inputAtoms.hetero[a0]) : true;
      const srcComp = inputAtoms['annotation_src-component_1_atom']?.[a0] ?? '.';
      const rec = {
        serial: outAtoms.length + 1,
        groupPdb: isHet ? 'HETATM' : 'ATOM',
        isHetero: isHet,
        atomName: inputAtoms.atom_name[a0],
        resName: inputAtoms.res_name[a0],
        chainId: inputAtoms.chain_id[a0],
        resId: Number(inputAtoms.res_id[a0]),
        insCode: inputAtoms.ins_code?.[a0] || '',
        x: coors[base + 0],
        y: coors[base + 1],
        z: coors[base + 2],
        rawX: coors[base + 0],
        rawY: coors[base + 1],
        rawZ: coors[base + 2],
        element: inputAtoms.element?.[a0] || inferElementSymbol(inputAtoms.atom_name[a0]),
        occupancy: 1.0,
        bFactor: inputAtoms.b_factor?.[a0] != null ? Number(inputAtoms.b_factor[a0]) : NaN,
        charge: inputAtoms.charge?.[a0] != null ? Number(inputAtoms.charge[a0]) : 0,
        entityId: inputAtoms.label_entity_id?.[a0] != null ? Number(inputAtoms.label_entity_id[a0]) : 0,
        authSeqId: Number(inputAtoms.res_id[a0]),
        tokenId: tid,
        inputIndex: a0,
        tokenInputIndex: a0,
        isUnindexedInsertion: false,
        crdSrcComponent: hasCrd || hasIntCrd ? srcComp : '.',
        srcComponent: srcComp,
        coordGroup: inputAtoms.condition_CoordinateGroup_1_atom?.[a0] ?? -1,
        crd: hasCrd ? [Number(crdRaw[0]), Number(crdRaw[1]), Number(crdRaw[2])] : null,
        intCrd: hasIntCrd ? [Number(intCrdRaw[0]), Number(intCrdRaw[1]), Number(intCrdRaw[2])] : null,
      };
      outAtoms.push(rec);
      tokenAtomList.push(rec);
    } else {
      const rt = residueType[tid];
      const resName = UNIFIED_ATOM37_TOKENS[rt];
      const slotMap = UNIFIED_ATOM37_ENCODING.token_slot_map[resName] || {};
      const order = getOrderedAtomNamesForResidue(resName);

      // Map existing input atoms for this token by atom_name
      const inByName = new Map();
      for (const ai of aIndices) {
        const nm = inputAtoms.atom_name[ai];
        if (nm && nm !== 'UNK' && nm !== 'OXT' && nm !== 'OP3') {
          inByName.set(nm, ai);
        }
      }
      const wasMasked = inputAtoms.res_name[a0] === '<M>' || inputAtoms.res_name[a0] === 'UNK';
      const srcCompResidue = wasMasked ? '.' : (inputAtoms['annotation_src-component_1_atom']?.[a0] ?? '.');

      for (const atomName of order) {
        const slot = slotMap[atomName];
        if (slot === undefined || !atomMask[tid * 37 + slot]) continue;
        const base = (tid * 37 + slot) * 3;
        const ai = inByName.get(atomName);

        const crdRaw = ai !== undefined ? inputAtoms.condition_coordinate_1_atom?.[ai] : null;
        const hasCrd = Array.isArray(crdRaw) && crdRaw[0] !== null && Number.isFinite(crdRaw[0]);
        const intCrdRaw = ai !== undefined ? inputAtoms.condition_InternalCoordinate_1_atom?.[ai] : null;
        const hasIntCrd = Array.isArray(intCrdRaw) && intCrdRaw[0] !== null && Number.isFinite(intCrdRaw[0]);
        const coordGroup = ai !== undefined ? (inputAtoms.condition_CoordinateGroup_1_atom?.[ai] ?? -1) : -1;
        const isHet = wasMasked ? false : Boolean(inputAtoms.hetero?.[ai ?? a0]);
        const bFactor =
          ai !== undefined && inputAtoms.b_factor?.[ai] != null ? Number(inputAtoms.b_factor[ai]) : NaN;
        const occupancy =
          ai !== undefined && !wasMasked && inputAtoms.occupancy?.[ai] != null && inputAtoms.occupancy[ai] > 0
            ? Number(inputAtoms.occupancy[ai])
            : 1.0;

        const rec = {
          serial: outAtoms.length + 1,
          groupPdb: isHet ? 'HETATM' : 'ATOM',
          isHetero: isHet,
          atomName,
          resName,
          chainId: inputAtoms.chain_id[a0],
          resId: Number(inputAtoms.res_id[a0]),
          insCode: inputAtoms.ins_code?.[a0] || '',
          x: coors[base + 0],
          y: coors[base + 1],
          z: coors[base + 2],
          rawX: coors[base + 0],
          rawY: coors[base + 1],
          rawZ: coors[base + 2],
          element: (ai !== undefined && inputAtoms.element?.[ai] && inputAtoms.element[ai] !== 'X')
            ? inputAtoms.element[ai]
            : inferElementSymbol(atomName),
          occupancy,
          bFactor,
          charge: ai !== undefined && inputAtoms.charge?.[ai] != null ? Number(inputAtoms.charge[ai]) : 0,
          entityId: inputAtoms.label_entity_id?.[a0] != null ? Number(inputAtoms.label_entity_id[a0]) : 0,
          authSeqId: Number(inputAtoms.res_id[a0]),
          tokenId: tid,
          // The input atom this one IS (-1 where the generated residue added it), and the
          // token's first: what a metric reads the input's annotations through.
          inputIndex: ai ?? -1,
          tokenInputIndex: a0,
          isUnindexedInsertion: false,
          crdSrcComponent: hasCrd || hasIntCrd ? srcCompResidue : '.',
          srcComponent: srcCompResidue,
          coordGroup,
          crd: hasCrd ? [Number(crdRaw[0]), Number(crdRaw[1]), Number(crdRaw[2])] : null,
          intCrd: hasIntCrd ? [Number(intCrdRaw[0]), Number(intCrdRaw[1]), Number(intCrdRaw[2])] : null,
        };
        outAtoms.push(rec);
        tokenAtomList.push(rec);
      }
    }
    tokensAtoms.set(tid, tokenAtomList);
  }

  // Handle unindexed tokens (`build_predicted_insertion_mapping` + `annotate_unindexed_insertion`)
  let predictedInsertionMapping = {};
  if (unidxTids.length > 0) {
    if (insertionLogits && insertionLogits.length > 0) {
      predictedInsertionMapping = buildPredictedInsertionMapping(insertionLogits, nValid, inputAtoms);
    }

    // Attempt to validate / apply `predictedInsertionMapping` against target tokens (`annotate_unindexed_insertion`)
    const canApplyMapping = (mapping) => {
      const entries = Object.entries(mapping);
      if (entries.length === 0) return false;
      for (const [uStr, tTid] of entries) {
        const uTid = Number(uStr);
        const uAtomIndices = tokenMap.get(uTid);
        const tgtAtomList = tokensAtoms.get(Number(tTid));
        if (!uAtomIndices || !tgtAtomList) return false;
        const srcAtomList = uAtomIndices
          .filter((ai) => {
            const nm = inputAtoms.atom_name[ai];
            return nm !== 'OXT' && nm !== 'OP3';
          })
          .map((ai) => ({
            ai,
            atomName: inputAtoms.atom_name[ai],
            element: inputAtoms.element?.[ai] || inferElementSymbol(inputAtoms.atom_name[ai]),
            x: Number(inputAtoms.coord?.[ai]?.[0] ?? 0),
            y: Number(inputAtoms.coord?.[ai]?.[1] ?? 0),
            z: Number(inputAtoms.coord?.[ai]?.[2] ?? 0),
          }));
        const matched = matchInsertedAtomsToTarget(srcAtomList, tgtAtomList, true);
        if (!matched) return false;
      }
      return true;
    };

    // Apply `annotate_unindexed_insertion`: transfer conditions and reference coords onto mapped target tokens
    if (canApplyMapping(predictedInsertionMapping)) {
      for (const [uStr, tTid] of Object.entries(predictedInsertionMapping)) {
        const uTid = Number(uStr);
        const uAtomIndices = tokenMap.get(uTid);
        const tgtAtomList = tokensAtoms.get(Number(tTid));
        const srcFiltered = uAtomIndices.filter((ai) => {
          const nm = inputAtoms.atom_name[ai];
          return nm !== 'OXT' && nm !== 'OP3';
        });
        const srcAtomList = srcFiltered.map((ai) => ({
          ai,
          atomName: inputAtoms.atom_name[ai],
          element: inputAtoms.element?.[ai] || inferElementSymbol(inputAtoms.atom_name[ai]),
          x: Number(inputAtoms.coord?.[ai]?.[0] ?? 0),
          y: Number(inputAtoms.coord?.[ai]?.[1] ?? 0),
          z: Number(inputAtoms.coord?.[ai]?.[2] ?? 0),
        }));
        const matched = matchInsertedAtomsToTarget(srcAtomList, tgtAtomList, true);
        if (!matched) continue;

        for (const tgtRec of tgtAtomList) {
          tgtRec.isUnindexedInsertion = true;
        }

        const uA0 = srcFiltered[0];
        const tokenSrcId = inputAtoms['annotation_src-component_1_atom']?.[uA0] ?? String(uTid);

        for (let s = 0; s < srcFiltered.length; s += 1) {
          const ai = srcFiltered[s];
          const tgtRec = tgtAtomList[matched[s]];
          const crdRaw = inputAtoms.condition_coordinate_1_atom?.[ai] ?? null;
          const hasCrd = Array.isArray(crdRaw) && crdRaw[0] !== null && Number.isFinite(crdRaw[0]);
          const intCrdRaw = inputAtoms.condition_InternalCoordinate_1_atom?.[ai] ?? null;
          const hasIntCrd = Array.isArray(intCrdRaw) && intCrdRaw[0] !== null && Number.isFinite(intCrdRaw[0]);
          const srcGrp = inputAtoms.condition_CoordinateGroup_1_atom?.[ai] ?? -1;

          // Every 1-body condition is the SOURCE's on a grafted atom (`annotate_unindexed_insertion`)
          tgtRec.condSource = ai;
          if (srcGrp !== -1) {
            tgtRec.coordGroup = srcGrp;
          }
          if (hasCrd) {
            tgtRec.crd = [Number(crdRaw[0]), Number(crdRaw[1]), Number(crdRaw[2])];
            tgtRec.crdSrcComponent = tokenSrcId;
          }
          if (hasIntCrd) {
            tgtRec.intCrd = [Number(intCrdRaw[0]), Number(intCrdRaw[1]), Number(intCrdRaw[2])];
            tgtRec.crdSrcComponent = tokenSrcId;
          }
        }
      }
    }
  }

  // Build `SA_CWI` (`coord_with_insertion`) and run internal-coordinate group superposition
  const cwiCoords = new Float32Array(outAtoms.length * 3);
  const anchorIndices = [];
  for (let i = 0; i < outAtoms.length; i += 1) {
    const a = outAtoms[i];
    if (a.crd && Number.isFinite(a.crd[0])) {
      cwiCoords[i * 3 + 0] = a.crd[0];
      cwiCoords[i * 3 + 1] = a.crd[1];
      cwiCoords[i * 3 + 2] = a.crd[2];
      anchorIndices.push(i);
    } else {
      cwiCoords[i * 3 + 0] = a.x;
      cwiCoords[i * 3 + 1] = a.y;
      cwiCoords[i * 3 + 2] = a.z;
    }
  }

  insertInternalCoordConditionedGroups(outAtoms, cwiCoords);

  // Align generated coordinates onto the input frame (`align_atom_array_to_input_frame`)
  alignAtomsToInputFrame(outAtoms, cwiCoords, anchorIndices);

  return {
    atoms: outAtoms,
    predictedInsertionMapping,
    cwiCoords,
  };
}

/**
 * Format an atom name for mmCIF `_atom_site.label_atom_id` / `auth_atom_id`
 * (quoting names that contain single quotes such as `"O5'"`).
 */
function formatCifAtomName(atomName) {
  if (!atomName) return '.';
  if (atomName.includes("'") || /\s/.test(atomName)) {
    return `"${atomName}"`;
  }
  return atomName;
}

/**
 * Format a coordinate float for mmCIF `_atom_site.Cartn_{x,y,z}` matching NumPy/Biotite's
 * 3-decimal round-half-to-even formatting.
 */
function formatCifCoord(val) {
  const f = Math.fround(val);
  const scaled = f * 1000.0;
  const fl = Math.floor(scaled);
  const diff = scaled - fl;
  let roundedInt;
  if (Math.abs(diff - 0.5) < 1e-6) {
    roundedInt = (fl & 1) === 0 ? fl : fl + 1;
  } else {
    roundedInt = Math.round(scaled);
  }
  return (roundedInt / 1000.0).toFixed(3);
}

/**
 * Serialize processed atom records into a valid mmCIF string (`toCifString`).
 *
 * @param {Array<object>|{ atoms: Array<object> }} atomsOrResult
 * @param {object} [options]
 * @param {string} [options.dataBlockName='rfd4p_design']
 * @returns {string}
 */
export function toCifString(atomsOrResult, options = {}) {
  const atoms = Array.isArray(atomsOrResult) ? atomsOrResult : (atomsOrResult?.atoms ?? []);
  const dataBlockName = options.dataBlockName || 'rfd4p_design';

  const lines = [
    `data_${dataBlockName}`,
    '#',
    'loop_',
    '_atom_site.group_PDB',
    '_atom_site.id',
    '_atom_site.type_symbol',
    '_atom_site.label_atom_id',
    '_atom_site.label_alt_id',
    '_atom_site.label_comp_id',
    '_atom_site.label_asym_id',
    '_atom_site.label_entity_id',
    '_atom_site.label_seq_id',
    '_atom_site.pdbx_PDB_ins_code',
    '_atom_site.auth_seq_id',
    '_atom_site.auth_comp_id',
    '_atom_site.auth_asym_id',
    '_atom_site.auth_atom_id',
    '_atom_site.B_iso_or_equiv',
    '_atom_site.occupancy',
    '_atom_site.pdbx_formal_charge',
    '_atom_site.is_unindexed_insertion',
    '_atom_site.annotation_src-coordinate-conditioned-region_1_atom',
    '_atom_site.Cartn_x',
    '_atom_site.Cartn_y',
    '_atom_site.Cartn_z',
    '_atom_site.pdbx_PDB_model_num',
  ];

  for (let i = 0; i < atoms.length; i += 1) {
    const a = atoms[i];
    const groupPdb = a.groupPdb || (a.isHetero ? 'HETATM' : 'ATOM');
    const id = a.serial ?? i + 1;
    const elem = a.element || inferElementSymbol(a.atomName);
    const atomId = formatCifAtomName(a.atomName);
    const compId = a.resName || 'UNK';
    const asymId = a.chainId || 'A';
    const entityId = a.entityId ?? 0;
    const seqId = a.resId ?? 1;
    const insCode = a.insCode || '.';
    const authSeqId = a.authSeqId ?? seqId;
    const bIso = Number.isFinite(a.bFactor) ? String(a.bFactor) : 'nan';
    const occ = Number.isFinite(a.occupancy) ? a.occupancy.toFixed(1) : '1.0';
    const chargeStr =
      a.charge && a.charge !== 0 ? (a.charge > 0 ? `+${a.charge}` : `${a.charge}`) : '?';
    const unidxStr = a.isUnindexedInsertion ? 'True' : 'False';
    const crdSrcStr = a.crdSrcComponent || '.';
    const xStr = formatCifCoord(a.x);
    const yStr = formatCifCoord(a.y);
    const zStr = formatCifCoord(a.z);

    lines.push(
      `${groupPdb} ${id} ${elem} ${atomId} . ${compId} ${asymId} ${entityId} ${seqId} ${insCode} ${authSeqId} ${compId} ${asymId} ${atomId} ${bIso} ${occ} ${chargeStr} ${unidxStr} ${crdSrcStr} ${xStr} ${yStr} ${zStr} 1`
    );
  }
  lines.push('#');
  return lines.join('\n') + '\n';
}

/**
 * Format an atom name into the 4-character PDB columns 13-16.
 */
function formatPdbAtomField(atomName, element) {
  const name = String(atomName || 'CA').trim();
  if (name.length >= 4) return name.slice(0, 4);
  const elem = String(element || inferElementSymbol(name)).trim().toUpperCase();
  if (elem.length === 1 && name.startsWith(elem)) {
    return ` ${name.padEnd(3, ' ')}`;
  }
  return name.padEnd(4, ' ');
}

/**
 * Serialize processed atom records into a standard 80-column PDB string (`toPdbString`).
 *
 * @param {Array<object>|{ atoms: Array<object> }} atomsOrResult
 * @param {object} [options]
 * @param {string} [options.title='RFD4-Proteina Design']
 * @returns {string}
 */
export function toPdbString(atomsOrResult, options = {}) {
  const atoms = Array.isArray(atomsOrResult) ? atomsOrResult : (atomsOrResult?.atoms ?? []);
  const title = options.title || 'RFD4-Proteina Design';
  const lines = [
    'HEADER    DE NOVO DESIGN                          01-JAN-26   RFD4',
    `TITLE     ${title}`,
  ];

  let prevChainId = null;
  let prevIsHet = false;
  for (let i = 0; i < atoms.length; i += 1) {
    const a = atoms[i];
    const isHet = a.groupPdb ? a.groupPdb === 'HETATM' : Boolean(a.isHetero);
    const chainId = String(a.chainId || 'A').slice(0, 1);
    if (prevChainId !== null && prevChainId !== chainId && !prevIsHet) {
      lines.push('TER');
    }
    prevChainId = chainId;
    prevIsHet = isHet;

    const recName = isHet ? 'HETATM' : 'ATOM  ';
    const serial = String((a.serial ?? i + 1) % 100000).padStart(5, ' ');
    const elem = String(a.element || inferElementSymbol(a.atomName)).trim().toUpperCase();
    const atomField = formatPdbAtomField(a.atomName, elem);
    const resName = String(a.resName || 'UNK').slice(0, 3).padStart(3, ' ');
    const resSeq = String(a.resId ?? 1).slice(-4).padStart(4, ' ');
    const insCode = String(a.insCode || ' ').slice(0, 1);
    const xStr = Number(a.x).toFixed(3).padStart(8, ' ');
    const yStr = Number(a.y).toFixed(3).padStart(8, ' ');
    const zStr = Number(a.z).toFixed(3).padStart(8, ' ');
    const occStr = (Number.isFinite(a.occupancy) ? a.occupancy : 1.0).toFixed(2).padStart(6, ' ');
    const bVal = Number.isFinite(a.bFactor) ? a.bFactor : 0.0;
    const bStr = bVal.toFixed(2).padStart(6, ' ');
    const elemField = elem.slice(0, 2).padStart(2, ' ');

    lines.push(
      `${recName}${serial} ${atomField} ${resName} ${chainId}${resSeq}${insCode}   ${xStr}${yStr}${zStr}${occStr}${bStr}          ${elemField}  `
    );
  }
  lines.push('TER');
  lines.push('END');
  return lines.join('\n') + '\n';
}

/**
 * Split a whitespace-delimited mmCIF data row, respecting single- and double-quoted tokens.
 */
function splitCifTokens(line) {
  const tokens = [];
  let i = 0;
  const n = line.length;
  while (i < n) {
    while (i < n && /\s/.test(line[i])) i += 1;
    if (i >= n) break;
    if (line[i] === "'" || line[i] === '"') {
      const q = line[i];
      i += 1;
      const start = i;
      while (i < n && !(line[i] === q && (i + 1 === n || /\s/.test(line[i + 1])))) {
        i += 1;
      }
      tokens.push(line.slice(start, i));
      if (i < n) i += 1;
    } else {
      const start = i;
      while (i < n && !/\s/.test(line[i])) i += 1;
      tokens.push(line.slice(start, i));
    }
  }
  return tokens;
}

/**
 * Parse `_atom_site` records from an mmCIF string (correctly preserving `resId = 0`).
 *
 * @param {string} cifText
 * @returns {Array<{
 *   serial: number,
 *   groupPdb: string,
 *   atomName: string,
 *   resName: string,
 *   chainId: string,
 *   resId: number,
 *   element: string,
 *   x: number,
 *   y: number,
 *   z: number,
 *   isUnindexedInsertion: boolean,
 *   crdSrcComponent: string,
 * }>}
 */
export function parseAtomSiteCif(cifText) {
  const lines = String(cifText || '').split(/\r?\n/);
  const headers = [];
  const atoms = [];
  let inAtomSite = false;

  for (let i = 0; i < lines.length; i += 1) {
    const trimmed = lines[i].trim();
    if (!trimmed || trimmed.startsWith('#')) {
      if (inAtomSite && atoms.length > 0) break;
      continue;
    }
    if (trimmed === 'loop_') {
      if (inAtomSite && atoms.length > 0) break;
      headers.length = 0;
      inAtomSite = false;
      continue;
    }
    if (trimmed.startsWith('_atom_site.')) {
      inAtomSite = true;
      headers.push(trimmed.slice('_atom_site.'.length).trim());
      continue;
    }
    if (inAtomSite) {
      if (trimmed.startsWith('_')) break;
      const cols = splitCifTokens(trimmed);
      if (cols.length < headers.length) continue;
      const row = {};
      for (let c = 0; c < headers.length; c += 1) {
        row[headers[c]] = cols[c];
      }
      const seqStr =
        row.label_seq_id !== undefined && row.label_seq_id !== '.' && row.label_seq_id !== '?'
          ? row.label_seq_id
          : row.auth_seq_id;
      const parsedResId = Number.parseInt(seqStr, 10);
      const parsedId = Number.parseInt(row.id, 10);
      atoms.push({
        serial: Number.isFinite(parsedId) ? parsedId : atoms.length + 1,
        groupPdb: row.group_PDB || 'ATOM',
        atomName: row.label_atom_id || row.auth_atom_id || 'CA',
        resName: row.label_comp_id || row.auth_comp_id || 'UNK',
        chainId: row.label_asym_id || row.auth_asym_id || 'A',
        resId: Number.isFinite(parsedResId) ? parsedResId : atoms.length + 1,
        element: row.type_symbol || inferElementSymbol(row.label_atom_id || 'C'),
        x: Number.parseFloat(row.Cartn_x),
        y: Number.parseFloat(row.Cartn_y),
        z: Number.parseFloat(row.Cartn_z),
        isUnindexedInsertion: row.is_unindexed_insertion === 'True',
        crdSrcComponent: row['annotation_src-coordinate-conditioned-region_1_atom'] || '.',
      });
    }
  }
  return atoms;
}
