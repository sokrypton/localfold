// Geometric featurisation: backbone frames, the k-nearest-neighbour graph, and
// the edge/node embeddings that feed the encoder.
//
// This mirrors ProteinFeatures / ProteinFeaturesLigand in the reference
// implementation, including its small inconsistencies (the Ca-Ca block reuses
// the masked, adjusted distance from the neighbour search while the other 24
// blocks recompute a raw distance). Those quirks are load-bearing for parity.

import {
  ATOM_GROUP,
  ATOM_PERIOD,
  ATOM_TYPE_ONEHOT,
  CB_COEFF,
  MAX_RELATIVE_FEATURE,
  RBF,
  SIDE_CHAIN_ATOM_TYPES,
  SIDE_CHAIN_START,
} from "./constants.js";
import { argTopKSmallest, layerNorm, linear, tryEdgeFeatures } from "./ops.js";

const RBF_MU = new Float32Array(RBF.count);
for (let i = 0; i < RBF.count; i++) {
  RBF_MU[i] = RBF.min + (i * (RBF.max - RBF.min)) / (RBF.count - 1);
}
const RBF_SIGMA = (RBF.max - RBF.min) / RBF.count;
const RBF_INV_SIGMA = 1 / RBF_SIGMA;

/**
 * Below this, a radial basis value is flushed to zero.
 *
 * `exp(-z²)` underflows past float32's smallest normal (1.18e-38) for atom
 * pairs far from a basis centre, and about 4% of the entries in a real edge
 * feature block land in the denormal range. Denormal arithmetic is trapped to
 * microcode on x86, and wasm -- unlike every native BLAS, which sets
 * flush-to-zero -- is required to honour it: measured, that one detail cost
 * **21x** on the edge-embedding matmul, 0.9 GFLOP/s against 20.6.
 *
 * The threshold sits well above the denormal boundary so that products with the
 * weights cannot fall into it either. Discarding a feature of size 1e-30 next
 * to features of order 1 is not an approximation anyone can measure, and it is
 * what the reference implementation effectively does already.
 */
const RBF_FLUSH = 1e-30;

/** Write the 16-channel radial basis expansion of `d` at `out[off..off+16]`. */
export function rbfInto(out, off, d) {
  for (let c = 0; c < RBF.count; c++) {
    const z = (d - RBF_MU[c]) * RBF_INV_SIGMA;
    const v = Math.exp(-z * z);
    out[off + c] = v < RBF_FLUSH ? 0 : v;
  }
}

export function dist(a, ai, b, bi) {
  const dx = a[ai] - b[bi];
  const dy = a[ai + 1] - b[bi + 1];
  const dz = a[ai + 2] - b[bi + 2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz + 1e-6);
}

/**
 * Split [L, 4, 3] backbone coordinates into per-atom arrays and derive the
 * virtual C-beta.
 *
 * @param {Float32Array} X [L, 4, 3] ordered N, CA, C, O
 */
export function computeBackbone(X, L) {
  const N = new Float32Array(L * 3);
  const CA = new Float32Array(L * 3);
  const C = new Float32Array(L * 3);
  const O = new Float32Array(L * 3);
  const CB = new Float32Array(L * 3);
  for (let i = 0; i < L; i++) {
    const s = i * 12;
    const d = i * 3;
    for (let j = 0; j < 3; j++) {
      N[d + j] = X[s + j];
      CA[d + j] = X[s + 3 + j];
      C[d + j] = X[s + 6 + j];
      O[d + j] = X[s + 9 + j];
    }
    const bx = CA[d] - N[d];
    const by = CA[d + 1] - N[d + 1];
    const bz = CA[d + 2] - N[d + 2];
    const cx = C[d] - CA[d];
    const cy = C[d + 1] - CA[d + 1];
    const cz = C[d + 2] - CA[d + 2];
    const ax = by * cz - bz * cy;
    const ay = bz * cx - bx * cz;
    const az = bx * cy - by * cx;
    CB[d] = CB_COEFF.a * ax + CB_COEFF.b * bx + CB_COEFF.c * cx + CA[d];
    CB[d + 1] = CB_COEFF.a * ay + CB_COEFF.b * by + CB_COEFF.c * cy + CA[d + 1];
    CB[d + 2] = CB_COEFF.a * az + CB_COEFF.b * bz + CB_COEFF.c * cz + CA[d + 2];
  }
  return { N, CA, C, O, CB };
}

/**
 * k-nearest-neighbour graph over C-alpha.
 *
 * Reproduces `ProteinFeatures._dist` exactly, including its handling of masked
 * residues: a pair with either end masked has distance 0, and every such entry
 * is then pushed out to the row's maximum, so masked residues sort to the back.
 *
 * @returns {{EIdx: Int32Array, DNeighbors: Float32Array, K: number}}
 */
export function neighborGraph(CA, mask, L, topK) {
  const K = Math.min(topK, L);
  let unmasked = 0;
  for (let i = 0; i < L; i++) if (mask[i] !== 0) unmasked++;

  // The grid can only certify a row when the K nearest are guaranteed to be
  // unmasked residues, which needs strictly more than K of them. Otherwise the
  // row's answer depends on ties at the row maximum and the exact sweep decides.
  const grid = unmasked > K ? buildGrid(CA, mask, L, K) : null;

  const EIdx = new Int32Array(L * K);
  const DNeighbors = new Float32Array(L * K);
  const row = new Float32Array(L);

  for (let i = 0; i < L; i++) {
    const out = EIdx.subarray(i * K, (i + 1) * K);
    if (grid !== null && mask[i] !== 0) {
      gridNearest(grid, CA, i, K, out);
      for (let k = 0; k < K; k++) DNeighbors[i * K + k] = dist(CA, i * 3, CA, out[k] * 3);
      continue;
    }
    let rowMax = 0;
    for (let j = 0; j < L; j++) {
      const m = mask[i] * mask[j];
      const d = m === 0 ? 0 : dist(CA, i * 3, CA, j * 3);
      row[j] = d;
      if (d > rowMax) rowMax = d;
    }
    for (let j = 0; j < L; j++) {
      if (mask[i] * mask[j] === 0) row[j] += rowMax;
    }
    argTopKSmallest(row, L, K, out);
    for (let k = 0; k < K; k++) DNeighbors[i * K + k] = row[out[k]];
  }
  return { EIdx, DNeighbors, K };
}

/**
 * Uniform bucket grid over the unmasked C-alpha, so the neighbour search stops
 * being O(L²·K).
 *
 * The cell edge is chosen so a 3x3x3 block holds a few times K points, which is
 * usually enough to answer a query after one expansion.
 */
function buildGrid(CA, mask, L, K) {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  let n = 0;
  for (let i = 0; i < L; i++) {
    if (mask[i] === 0) continue;
    n++;
    const x = CA[i * 3], y = CA[i * 3 + 1], z = CA[i * 3 + 2];
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }
  const volume = Math.max((maxX - minX) * (maxY - minY) * (maxZ - minZ), 1);
  const cell = Math.max(Math.cbrt((volume * K) / (n * 4)), 2);

  const nx = Math.max(1, Math.ceil((maxX - minX) / cell) + 1);
  const ny = Math.max(1, Math.ceil((maxY - minY) / cell) + 1);
  const nz = Math.max(1, Math.ceil((maxZ - minZ) / cell) + 1);
  const cells = nx * ny * nz;

  const cellOf = new Int32Array(L).fill(-1);
  const counts = new Int32Array(cells + 1);
  for (let i = 0; i < L; i++) {
    if (mask[i] === 0) continue;
    const cx = Math.min(nx - 1, Math.floor((CA[i * 3] - minX) / cell));
    const cy = Math.min(ny - 1, Math.floor((CA[i * 3 + 1] - minY) / cell));
    const cz = Math.min(nz - 1, Math.floor((CA[i * 3 + 2] - minZ) / cell));
    const c = (cz * ny + cy) * nx + cx;
    cellOf[i] = c;
    counts[c + 1]++;
  }
  for (let c = 0; c < cells; c++) counts[c + 1] += counts[c];
  const start = Int32Array.from(counts);
  const items = new Int32Array(n);
  const cursor = Int32Array.from(counts);
  // Fill in ascending residue index, so a cell's members stay index-ordered and
  // ties resolve the same way the exact sweep resolves them.
  for (let i = 0; i < L; i++) {
    if (cellOf[i] >= 0) items[cursor[cellOf[i]]++] = i;
  }
  return { minX, minY, minZ, cell, nx, ny, nz, start, items };
}

/**
 * The K nearest unmasked residues to `i`, ascending, ties broken toward the
 * lower index -- the same order `argTopKSmallest` produces.
 *
 * Rings of cells are scanned outward. Everything left unscanned after ring r
 * lies at least `r · cell` away, so once the current K-th distance is below
 * that the answer cannot change.
 */
function gridNearest(grid, CA, i, K, out) {
  const { minX, minY, minZ, cell, nx, ny, nz, start, items } = grid;
  const cx = Math.min(nx - 1, Math.floor((CA[i * 3] - minX) / cell));
  const cy = Math.min(ny - 1, Math.floor((CA[i * 3 + 1] - minY) / cell));
  const cz = Math.min(nz - 1, Math.floor((CA[i * 3 + 2] - minZ) / cell));

  // Insertion-sorted top-K by (distance, index).
  const bestD = new Float64Array(K).fill(Infinity);
  const bestI = new Int32Array(K).fill(0x7fffffff);
  let filled = 0;

  const consider = (j) => {
    // Rounded to float32 before comparing. The exact sweep buffers its row in a
    // Float32Array, so that is the precision its ordering is decided at; two
    // distances that differ as doubles can tie once rounded, and then the tie
    // must break by index the same way. Without this the graphs diverge on a
    // couple of edges in a few thousand.
    const d = Math.fround(dist(CA, i * 3, CA, j * 3));
    if (filled === K && (d > bestD[K - 1] || (d === bestD[K - 1] && j > bestI[K - 1]))) return;
    let p = Math.min(filled, K - 1);
    while (p > 0 && (d < bestD[p - 1] || (d === bestD[p - 1] && j < bestI[p - 1]))) {
      bestD[p] = bestD[p - 1];
      bestI[p] = bestI[p - 1];
      p--;
    }
    bestD[p] = d;
    bestI[p] = j;
    if (filled < K) filled++;
  };

  const maxRing = Math.max(nx, ny, nz);
  for (let r = 0; ; r++) {
    const x0 = Math.max(0, cx - r), x1 = Math.min(nx - 1, cx + r);
    const y0 = Math.max(0, cy - r), y1 = Math.min(ny - 1, cy + r);
    const z0 = Math.max(0, cz - r), z1 = Math.min(nz - 1, cz + r);
    for (let z = z0; z <= z1; z++) {
      for (let y = y0; y <= y1; y++) {
        // Only the shell added by this ring, except on the first pass.
        const onShellZY = r === 0 || z === cz - r || z === cz + r || y === cy - r || y === cy + r;
        for (let x = x0; x <= x1; x++) {
          if (!onShellZY && x !== cx - r && x !== cx + r) continue;
          const c = (z * ny + y) * nx + x;
          for (let t = start[c]; t < start[c + 1]; t++) consider(items[t]);
        }
      }
    }
    // Strict, so a point exactly on the boundary cannot be missed.
    if (filled === K && bestD[K - 1] < r * cell) break;
    if (r >= maxRing) break;
  }
  out.set(bestI.subarray(0, K));
}

// The 24 recomputed distance blocks, in the reference's order. `null` marks the
// leading Ca-Ca block, which reuses the neighbour-search distance instead.
const RBF_PAIRS = [
  null,
  ["N", "N"], ["C", "C"], ["O", "O"], ["CB", "CB"],
  ["CA", "N"], ["CA", "C"], ["CA", "O"], ["CA", "CB"],
  ["N", "C"], ["N", "O"], ["N", "CB"],
  ["CB", "C"], ["CB", "O"], ["O", "C"],
  ["N", "CA"], ["C", "CA"], ["O", "CA"], ["CB", "CA"],
  ["C", "N"], ["O", "N"], ["CB", "N"],
  ["C", "CB"], ["O", "CB"], ["C", "O"],
];

export const EDGE_IN_DIM = 16 + RBF.count * 25; // 416

/**
 * Edge embeddings E = LayerNorm(W_edge · [positional ‖ 25 × RBF]).
 *
 * Built in row chunks so the intermediate [L, K, 416] block never has to exist
 * all at once -- at L = 1000, K = 48 that would be 160 MB.
 *
 * @returns {Float32Array} [L, K, 128]
 */
export function edgeFeatures(w, bb, EIdx, DNeighbors, residueIdx, chainLabels, L, K, chunk = 64) {
  const edgeEmbed = w.linear(`${w.featurePrefix}.edge_embedding`);
  const normEdges = w.norm(`${w.featurePrefix}.norm_edges`);
  const posLinear = w.linear(`${w.featurePrefix}.embeddings.linear`);
  const posIn = 2 * MAX_RELATIVE_FEATURE + 2; // 66
  const hidden = edgeEmbed.shape[0];

  const E = new Float32Array(L * K * hidden);
  const scratch = new Float32Array(chunk * K * EDGE_IN_DIM);
  const projected = new Float32Array(chunk * K * hidden);
  const pos16 = new Float32Array(chunk * K * 16);
  const rowOf = new Int32Array(chunk);
  // N, CA, C, O, CB interleaved, which is the layout the fused kernel indexes.
  const xyz = new Float32Array(L * 15);
  for (let i = 0; i < L; i++) {
    const atoms = [bb.N, bb.CA, bb.C, bb.O, bb.CB];
    for (let a = 0; a < 5; a++) {
      xyz[i * 15 + a * 3] = atoms[a][i * 3];
      xyz[i * 15 + a * 3 + 1] = atoms[a][i * 3 + 1];
      xyz[i * 15 + a * 3 + 2] = atoms[a][i * 3 + 2];
    }
  }

  for (let start = 0; start < L; start += chunk) {
    const end = Math.min(L, start + chunk);
    const rows = end - start;

    // The relative-position block is a one-hot times a Linear, so it is a
    // column read either way; it stays here and is handed to the kernel.
    for (let i = start; i < end; i++) {
      rowOf[i - start] = i;
      for (let k = 0; k < K; k++) {
        const j = EIdx[i * K + k];
        const sameChain = chainLabels[i] === chainLabels[j] ? 1 : 0;
        const offset = residueIdx[i] - residueIdx[j];
        const clipped = Math.min(
          Math.max(offset + MAX_RELATIVE_FEATURE, 0),
          2 * MAX_RELATIVE_FEATURE,
        );
        const d = sameChain ? clipped : 2 * MAX_RELATIVE_FEATURE + 1;
        const base = ((i - start) * K + k) * 16;
        for (let o = 0; o < 16; o++) {
          pos16[base + o] = posLinear.weight[o * posIn + d] + posLinear.bias[o];
        }
      }
    }

    const dest = E.subarray(start * K * hidden, end * K * hidden);
    if (tryEdgeFeatures(
      pos16, xyz, EIdx.subarray(start * K), DNeighbors.subarray(start * K), rowOf,
      edgeEmbed.weight, normEdges.gamma, normEdges.beta, rows, K, hidden, dest,
    )) continue;

    scratch.fill(0, 0, rows * K * EDGE_IN_DIM);
    for (let i = start; i < end; i++) {
      for (let k = 0; k < K; k++) {
        const j = EIdx[i * K + k];
        const base = ((i - start) * K + k) * EDGE_IN_DIM;
        for (let o = 0; o < 16; o++) scratch[base + o] = pos16[((i - start) * K + k) * 16 + o];
        let off = base + 16;
        rbfInto(scratch, off, DNeighbors[i * K + k]);
        off += RBF.count;
        for (let p = 1; p < RBF_PAIRS.length; p++) {
          const [an, bn] = RBF_PAIRS[p];
          rbfInto(scratch, off, dist(bb[an], i * 3, bb[bn], j * 3));
          off += RBF.count;
        }
      }
    }
    linear(scratch, edgeEmbed.weight, edgeEmbed.bias, rows * K, EDGE_IN_DIM, hidden, projected);
    layerNorm(projected, normEdges.gamma, normEdges.beta, rows * K, hidden, dest);
  }
  return E;
}

// ---------------------------------------------------------------------------
// Ligand context
// ---------------------------------------------------------------------------

const { type: N_TYPE, group: N_GROUP, period: N_PERIOD, total: N_ATOM_FEAT } = ATOM_TYPE_ONEHOT;

/**
 * Apply a Linear to the concatenated [type ‖ group ‖ period] one-hot without
 * materialising the 147-wide vector: three column reads plus the bias.
 */
function atomTypeLinear(out, off, lin, cout, t) {
  const g = N_TYPE + ATOM_GROUP[t];
  const p = N_TYPE + N_GROUP + ATOM_PERIOD[t];
  for (let o = 0; o < cout; o++) {
    const row = o * N_ATOM_FEAT;
    out[off + o] = lin.weight[row + t] + lin.weight[row + g] + lin.weight[row + p]
      + (lin.bias === null ? 0 : lin.bias[o]);
  }
}

/**
 * For each residue, pick the `M` ligand atoms nearest its C-beta.
 *
 * Mirrors `get_nearest_neighbours`: masked pairs are pushed out to a distance
 * of 1000 Å², and residues with fewer than M reachable atoms are zero padded.
 *
 * @param {Float32Array} ligXyz [A, 3]
 * @param {Int32Array}   ligType [A] atom-type index
 * `Yg` records which global ligand atom landed in each slot, or -1 for the
 * zero padding used when a residue has fewer than M atoms in reach. Everything
 * downstream that depends only on *which* atoms are involved -- the atom-pair
 * distances, and therefore the whole first round of atom-to-atom message
 * passing -- is deduplicated through it.
 *
 * @returns {{Y: Float32Array, Yt: Int32Array, Ym: Float32Array,
 *            Yg: Int32Array, Dclosest: Float32Array}}
 */
export function nearestLigandAtoms(CB, mask, ligXyz, ligType, ligMask, L, M) {
  const A = ligType.length;
  const Y = new Float32Array(L * M * 3);
  const Yt = new Int32Array(L * M);
  const Ym = new Float32Array(L * M);
  const Yg = new Int32Array(L * M).fill(-1);
  const Dclosest = new Float32Array(L);
  if (A === 0) return { Y, Yt, Ym, Yg, Dclosest };

  const take = Math.min(M, A);
  const row = new Float32Array(A);
  const pick = new Int32Array(take);

  for (let i = 0; i < L; i++) {
    for (let a = 0; a < A; a++) {
      const m = mask[i] * ligMask[a];
      const dx = CB[i * 3] - ligXyz[a * 3];
      const dy = CB[i * 3 + 1] - ligXyz[a * 3 + 1];
      const dz = CB[i * 3 + 2] - ligXyz[a * 3 + 2];
      const l2 = dx * dx + dy * dy + dz * dz;
      row[a] = l2 * m + (1 - m) * 1000.0;
    }
    argTopKSmallest(row, A, take, pick);
    Dclosest[i] = Math.sqrt(row[pick[0]]);
    for (let s = 0; s < take; s++) {
      const a = pick[s];
      Y[(i * M + s) * 3] = ligXyz[a * 3];
      Y[(i * M + s) * 3 + 1] = ligXyz[a * 3 + 1];
      Y[(i * M + s) * 3 + 2] = ligXyz[a * 3 + 2];
      Yt[i * M + s] = ligType[a];
      Ym[i * M + s] = ligMask[a];
      Yg[i * M + s] = a;
    }
  }
  return { Y, Yt, Ym, Yg, Dclosest };
}

/** How many backbone neighbours contribute their side chains. */
export const SIDE_CHAIN_NEIGHBORS = 16;
/** Side-chain slots per residue: everything in the 37-atom set past C-beta. */
const SC_SLOTS = SIDE_CHAIN_ATOM_TYPES.length;

/**
 * The same selection as `nearestLigandAtoms`, but with the side chains of
 * nearby *fixed* residues thrown into the candidate pool first.
 *
 * This is LigandMPNN's `--ligand_mpnn_use_side_chain_context`. Each residue
 * looks at its 16 nearest backbone neighbours, takes their 32 side-chain slots,
 * masks out any belonging to a residue that is being designed -- its side chain
 * is about to change, so showing it would be cheating -- and appends the ligand
 * slots this residue already selected. The M nearest of that pool to C-beta
 * become the atom context.
 *
 * The ligand half is `lig`, the [L, M] output of `nearestLigandAtoms`, not the
 * raw atom list: the reference's `featurize` always runs that selection first
 * and `forward` concatenates onto its result, padding slots included. With a
 * ligand larger than M the two differ.
 *
 * Two consequences worth knowing. The encoder now depends on which positions
 * are fixed, so changing the selection invalidates the encoding; and the atom
 * pool is no longer a handful of ligand atoms, so the pair-dedup table below
 * has more distinct pairs to hold.
 *
 * @param {Float32Array} chainMask [L] 1 = being designed, 0 = fixed
 * @param {{Y: Float32Array, Yt: Int32Array, Ym: Float32Array, Yg: Int32Array}} lig
 * @returns {{Y: Float32Array, Yt: Int32Array, Ym: Float32Array,
 *            Yg: Int32Array, Dclosest: Float32Array, poolSize: number}}
 */
export function sideChainContext(
  CB, mask, EIdx, K, xyz37, xyz37Mask, chainMask, lig, L, M,
) {
  const nNbr = Math.min(SIDE_CHAIN_NEIGHBORS, K);
  const nCand = nNbr * SC_SLOTS + M;
  // Global atom ids: residue r slot s -> r * SC_SLOTS + s, ligand atom a ->
  // L * SC_SLOTS + 1 + a, with a padding slot at L * SC_SLOTS. The pair table
  // keys on these, so they have to be unique across the whole structure, not
  // just within one residue's candidate list.
  let maxLigId = 0;
  for (let n = 0; n < lig.Yg.length; n++) maxLigId = Math.max(maxLigId, lig.Yg[n] + 1);
  const poolSize = L * SC_SLOTS + 1 + maxLigId;

  const Y = new Float32Array(L * M * 3);
  const Yt = new Int32Array(L * M);
  const Ym = new Float32Array(L * M);
  const Yg = new Int32Array(L * M).fill(-1);
  const Dclosest = new Float32Array(L);

  const take = Math.min(M, nCand);
  if (take === 0) return { Y, Yt, Ym, Yg, Dclosest, poolSize };

  const row = new Float32Array(nCand);
  const cm = new Float32Array(nCand);
  const xyz = new Float32Array(nCand * 3);
  const type = new Int32Array(nCand);
  const gid = new Int32Array(nCand);
  const pick = new Int32Array(take);

  for (let i = 0; i < L; i++) {
    let c = 0;
    for (let k = 0; k < nNbr; k++) {
      const j = EIdx[i * K + k];
      // A designed neighbour contributes nothing: `xyz_37_m * (1 - chain_mask)`.
      const keep = chainMask[j] > 0 ? 0 : 1;
      for (let s = 0; s < SC_SLOTS; s++) {
        const slot = j * 37 + SIDE_CHAIN_START + s;
        xyz[c * 3] = xyz37[slot * 3];
        xyz[c * 3 + 1] = xyz37[slot * 3 + 1];
        xyz[c * 3 + 2] = xyz37[slot * 3 + 2];
        type[c] = SIDE_CHAIN_ATOM_TYPES[s];
        gid[c] = j * SC_SLOTS + s;
        cm[c] = xyz37Mask[slot] * keep;
        c++;
      }
    }
    for (let m = 0; m < M; m++) {
      const src = i * M + m;
      xyz[c * 3] = lig.Y[src * 3];
      xyz[c * 3 + 1] = lig.Y[src * 3 + 1];
      xyz[c * 3 + 2] = lig.Y[src * 3 + 2];
      type[c] = lig.Yt[src];
      gid[c] = L * SC_SLOTS + 1 + lig.Yg[src];
      cm[c] = lig.Ym[src];
      c++;
    }

    for (let n = 0; n < nCand; n++) {
      const m = mask[i] * cm[n];
      const dx = CB[i * 3] - xyz[n * 3];
      const dy = CB[i * 3 + 1] - xyz[n * 3 + 1];
      const dz = CB[i * 3 + 2] - xyz[n * 3 + 2];
      row[n] = (dx * dx + dy * dy + dz * dz) * m + (1 - m) * 10000.0;
    }
    argTopKSmallest(row, nCand, take, pick);
    Dclosest[i] = Math.sqrt(row[pick[0]]);

    for (let s = 0; s < take; s++) {
      const n = pick[s];
      const dst = i * M + s;
      Y[dst * 3] = xyz[n * 3];
      Y[dst * 3 + 1] = xyz[n * 3 + 1];
      Y[dst * 3 + 2] = xyz[n * 3 + 2];
      Yt[dst] = type[n];
      // The candidate's own mask, not the adjusted one -- the reference gathers
      // `Y_m` straight through, so a masked residue i keeps its neighbours'
      // masks even though its distances were all pushed to the padding value.
      Ym[dst] = cm[n];
      Yg[dst] = gid[n];
    }
  }
  return { Y, Yt, Ym, Yg, Dclosest, poolSize };
}

/**
 * Deduplicate the M x M atom pairs each residue sees down to the distinct
 * global atom pairs across the whole structure.
 *
 * The atom-pair distance depends only on which two ligand atoms are involved,
 * never on the residue looking at them, so a structure with a 16-atom ligand
 * has at most 17 x 17 distinct pairs (padding included) where the naive loop
 * computes L x 25 x 25 of them -- 289 against 75625 for streptavidin. Every
 * quantity derived purely from the pair inherits the same reduction, which for
 * the first atom-context layer is its entire message.
 *
 * @param {Int32Array} Yg [L, M] global atom index per slot, -1 for padding
 * @param {number} A total ligand atoms, used to key the pair map
 * @returns {{pairId: Int32Array, count: number, dist: Float32Array}}
 *          `pairId` is [L, M, M]; `dist` is one distance per distinct pair
 */
export function ligandPairTable(Yg, Y, L, M, A) {
  const pairId = new Int32Array(L * M * M);
  const seen = new Map();
  const symSeen = new Map();
  const symOfOrdered = [];
  const dist2Of = [];
  let count = 0;
  let symCount = 0;

  for (let i = 0; i < L; i++) {
    for (let m = 0; m < M; m++) {
      const ga = Yg[i * M + m];
      const ao = (i * M + m) * 3;
      for (let n = 0; n < M; n++) {
        const gb = Yg[i * M + n];
        // Padding always sits at the origin, the same point for every residue,
        // so (-1, b) is a well-defined pair and shares one entry.
        const key = (ga + 1) * (A + 1) + (gb + 1);
        let id = seen.get(key);
        if (id === undefined) {
          id = count++;
          seen.set(key, id);
          // Everything derived from the pair's *distance* is the same for
          // (a, b) and (b, a) -- `dist` squares the componentwise differences,
          // so the two are not merely close but the identical float. Give them
          // one id and the RBF, its embedding and the second layer's edge
          // projection are computed once instead of twice.
          const lo = Math.min(ga, gb);
          const hi = Math.max(ga, gb);
          const symKey = (lo + 1) * (A + 1) + (hi + 1);
          let sym = symSeen.get(symKey);
          if (sym === undefined) {
            sym = symCount++;
            symSeen.set(symKey, sym);
            dist2Of.push(dist(Y, ao, Y, (i * M + n) * 3));
          }
          symOfOrdered.push(sym);
        }
        pairId[(i * M + m) * M + n] = id;
      }
    }
  }
  return {
    pairId,
    count,
    /** [count] ordered pair id -> the unordered id its distance lives at. */
    symOf: Int32Array.from(symOfOrdered),
    symCount,
    dist: Float32Array.from(dist2Of),
  };
}

/** RBF expansion of one distance per distinct pair. */
export function pairRbf(dists, count, out) {
  out = out ?? new Float32Array(count * RBF.count);
  for (let p = 0; p < count; p++) rbfInto(out, p * RBF.count, dists[p]);
  return out;
}

/**
 * The atom-type embedding, one row per element rather than per (residue, slot).
 *
 * @returns {Float32Array} [120, hidden]
 */
export function atomTypeEmbedding(w, prefix, cout) {
  const lin = w.linear(prefix);
  const out = new Float32Array(N_TYPE * cout);
  for (let t = 0; t < N_TYPE; t++) atomTypeLinear(out, t * cout, lin, cout, t);
  return out;
}

/**
 * Per-residue ligand node features.
 *
 * V is the projected [5 × RBF ‖ atom-type ‖ frame angles] block; YNodes is the
 * atom-type embedding used by the ligand-to-ligand message passing.
 *
 * @returns {{V: Float32Array, YNodes: Float32Array}} both [L, M, 128]
 */
export function ligandNodeFeatures(w, bb, Y, Yt, L, M) {
  const projectDown = w.linear("features.node_project_down");
  const normNodes = w.norm("features.norm_nodes");
  const typeLinear = w.linear("features.type_linear");
  const yNodesLin = w.linear("features.y_nodes");
  const normYNodes = w.norm("features.norm_y_nodes");
  const hidden = projectDown.shape[0];
  const inDim = 5 * RBF.count + 64 + 4; // 148

  const feats = new Float32Array(L * M * inDim);
  const yNodesRaw = new Float32Array(L * M * hidden);
  const atoms = [bb.N, bb.CA, bb.C, bb.O, bb.CB];

  for (let i = 0; i < L; i++) {
    // Local frame: e1 along N-CA, e2 in the N-CA-C plane, e3 completing it.
    const d = i * 3;
    const v1x = bb.N[d] - bb.CA[d];
    const v1y = bb.N[d + 1] - bb.CA[d + 1];
    const v1z = bb.N[d + 2] - bb.CA[d + 2];
    const v2x = bb.C[d] - bb.CA[d];
    const v2y = bb.C[d + 1] - bb.CA[d + 1];
    const v2z = bb.C[d + 2] - bb.CA[d + 2];
    const n1 = Math.max(Math.sqrt(v1x * v1x + v1y * v1y + v1z * v1z), 1e-12);
    const e1x = v1x / n1, e1y = v1y / n1, e1z = v1z / n1;
    const dot = e1x * v2x + e1y * v2y + e1z * v2z;
    const u2x = v2x - e1x * dot, u2y = v2y - e1y * dot, u2z = v2z - e1z * dot;
    const n2 = Math.max(Math.sqrt(u2x * u2x + u2y * u2y + u2z * u2z), 1e-12);
    const e2x = u2x / n2, e2y = u2y / n2, e2z = u2z / n2;
    const e3x = e1y * e2z - e1z * e2y;
    const e3y = e1z * e2x - e1x * e2z;
    const e3z = e1x * e2y - e1y * e2x;

    for (let m = 0; m < M; m++) {
      const yo = (i * M + m) * 3;
      const base = (i * M + m) * inDim;
      for (let a = 0; a < 5; a++) {
        rbfInto(feats, base + a * RBF.count, dist(atoms[a], d, Y, yo));
      }
      atomTypeLinear(feats, base + 5 * RBF.count, typeLinear, 64, Yt[i * M + m]);

      // Direction of the ligand atom in the residue frame, as (cosφ, sinφ, cosθ, sinθ).
      const rx = Y[yo] - bb.CA[d];
      const ry = Y[yo + 1] - bb.CA[d + 1];
      const rz = Y[yo + 2] - bb.CA[d + 2];
      const lx = rx * e1x + ry * e1y + rz * e1z;
      const ly = rx * e2x + ry * e2y + rz * e2z;
      const lz = rx * e3x + ry * e3y + rz * e3z;
      const rxy = Math.sqrt(lx * lx + ly * ly + 1e-8);
      const rxyz = Math.sqrt(lx * lx + ly * ly + lz * lz) + 1e-8;
      const fo = base + 5 * RBF.count + 64;
      feats[fo] = lx / rxy;
      feats[fo + 1] = ly / rxy;
      feats[fo + 2] = rxy / rxyz;
      feats[fo + 3] = lz / rxyz;

      atomTypeLinear(yNodesRaw, (i * M + m) * hidden, yNodesLin, hidden, Yt[i * M + m]);
    }
  }

  const V = new Float32Array(L * M * hidden);
  linear(feats, projectDown.weight, projectDown.bias, L * M, inDim, hidden, V);
  layerNorm(V, normNodes.gamma, normNodes.beta, L * M, hidden, V);
  const YNodes = layerNorm(yNodesRaw, normYNodes.gamma, normYNodes.beta, L * M, hidden);
  return { V, YNodes };
}

/**
 * Ligand-atom pair edges for residues `[start, end)`, already projected and
 * normalised: LayerNorm(W_y_edges · RBF(‖Y_m - Y_m'‖)).
 *
 * Computed per chunk because the full tensor is [L, M, M, 128] -- 160 MB at
 * L = 500, M = 25.
 *
 * @returns {Float32Array} [end - start, M, M, 128]
 */
export function ligandEdgeChunk(w, Y, M, start, end, scratch, out) {
  const yEdges = w.linear("features.y_edges");
  const normYEdges = w.norm("features.norm_y_edges");
  const hidden = yEdges.shape[0];
  const rows = end - start;
  const pairs = rows * M * M;

  scratch = scratch ?? new Float32Array(pairs * RBF.count);
  out = out ?? new Float32Array(pairs * hidden);

  let at = 0;
  for (let i = start; i < end; i++) {
    for (let m = 0; m < M; m++) {
      const ao = (i * M + m) * 3;
      for (let n = 0; n < M; n++) {
        const bo = (i * M + n) * 3;
        rbfInto(scratch, at, dist(Y, ao, Y, bo));
        at += RBF.count;
      }
    }
  }
  linear(scratch, yEdges.weight, yEdges.bias, pairs, RBF.count, hidden, out);
  layerNorm(out, normYEdges.gamma, normYEdges.beta, pairs, hidden, out);
  return out;
}
