/**
 * Token-level bond features (`token_bonds`, `token_bond_orders`, `token_bond_is_aromatic`,
 * `token_bond_path_length`, `canonical_polymer_ordering`) and gap-filled shortest-path BFS.
 *
 * Ports `rfproteina.datapipes.bonds`.
 */

import { BOND_TYPE, getResidueSegments, getPolymerizationAtoms } from './cif.js';
import { getDefaultIndexGroups } from './spec.js';

export const TOKEN_TYPE = {
  PROTEIN: 0,
  RNA: 1,
  DNA: 2,
  ATOMIZED: 3,
  UNKNOWN: 4,
  GLYCAN: 5,
  NCAA: 6,
};

const BOND_TYPE_TO_ORDER = {
  [BOND_TYPE.ANY]: 1,
  [BOND_TYPE.SINGLE]: 1,
  [BOND_TYPE.DOUBLE]: 2,
  [BOND_TYPE.TRIPLE]: 3,
  [BOND_TYPE.QUADRUPLE]: 1,
  [BOND_TYPE.AROMATIC_SINGLE]: 1,
  [BOND_TYPE.AROMATIC_DOUBLE]: 2,
  [BOND_TYPE.AROMATIC_TRIPLE]: 3,
  [BOND_TYPE.COORDINATION]: 4,
  [BOND_TYPE.AROMATIC]: 1,
};

const AROMATIC_BOND_SET = new Set([
  BOND_TYPE.AROMATIC_SINGLE,
  BOND_TYPE.AROMATIC_DOUBLE,
  BOND_TYPE.AROMATIC_TRIPLE,
  BOND_TYPE.AROMATIC,
]);

/**
 * Get the start atom indices of every token in `atoms` (plus optional exclusive stop `nAtoms`).
 */
export function getTokenStarts(atoms, addExclusiveStop = false) {
  const n = atoms.nAtoms;
  if (n === 0) return addExclusiveStop ? [0] : [];

  if (atoms.token_id) {
    const starts = [0];
    for (let i = 1; i < n; i++) {
      if (atoms.token_id[i] !== atoms.token_id[i - 1]) starts.push(i);
    }
    if (addExclusiveStop) starts.push(n);
    return starts;
  }

  const resStarts = getResidueSegments(atoms);
  const starts = [];
  for (let r = 0; r < resStarts.length - 1; r++) {
    const s = resStarts[r];
    const e = resStarts[r + 1];
    if (atoms.atomize && atoms.atomize[s]) {
      for (let i = s; i < e; i++) starts.push(i);
    } else {
      starts.push(s);
    }
  }
  if (addExclusiveStop) starts.push(n);
  return starts;
}

function getPolymerizationAtomNamesForToken(resName, tokenType, chainType) {
  let [leaving, entering] = getPolymerizationAtoms(resName, chainType);
  if (!leaving && !entering) {
    if (tokenType === TOKEN_TYPE.PROTEIN || tokenType === TOKEN_TYPE.NCAA) {
      leaving = 'C';
      entering = 'N';
    } else if (tokenType === TOKEN_TYPE.RNA || tokenType === TOKEN_TYPE.DNA) {
      leaving = "O3'";
      entering = 'P';
    }
  }
  return [leaving || '', entering || ''];
}

function findPolymerizationAtomIdx(atoms, resStart, resStop, tokenType, exiting) {
  if (resStop - resStart === 1 && atoms.res_name[resStart] === '<M>') {
    return resStart;
  }
  const [exitingName, enteringName] = getPolymerizationAtomNamesForToken(
    atoms.res_name[resStart],
    tokenType,
    atoms.chain_type?.[resStart]
  );
  const targetName = exiting ? exitingName : enteringName;
  let match = -1;
  let count = 0;
  for (let i = resStart; i < resStop; i++) {
    if (atoms.atom_name[i] === targetName) {
      match = i;
      count++;
    }
  }
  return count === 1 ? match : null;
}

export function bondFlankingResidues(atoms, prevRes, nextRes, resBounds = null) {
  const bounds = resBounds || getResidueSegments(atoms);
  const sPrev = bounds[prevRes];
  const ePrev = bounds[prevRes + 1];
  const sNext = bounds[nextRes];
  const eNext = bounds[nextRes + 1];
  if (atoms.atomize && (atoms.atomize[sPrev] || atoms.atomize[sNext])) return;
  const ttPrev = atoms.token_type ? atoms.token_type[sPrev] : TOKEN_TYPE.PROTEIN;
  const ttNext = atoms.token_type ? atoms.token_type[sNext] : TOKEN_TYPE.PROTEIN;
  const leftAtom = findPolymerizationAtomIdx(atoms, sPrev, ePrev, ttPrev, true);
  const rightAtom = findPolymerizationAtomIdx(atoms, sNext, eNext, ttNext, false);
  if (leftAtom !== null && rightAtom !== null) {
    const a = Math.min(leftAtom, rightAtom);
    const b = Math.max(leftAtom, rightAtom);
    if (!atoms.bonds.some(([u, v]) => Math.min(u, v) === a && Math.max(u, v) === b)) {
      atoms.bonds.push([a, b, BOND_TYPE.SINGLE]);
    }
  }
}

/**
 * Multi-source unweighted BFS for shortest-path distances between `sourceIndices` in an undirected graph.
 */
export function computeTokenPathLengths(tok1, tok2, sourceIndices, pathLengthCutoff = 64) {
  const nSources = sourceIndices.length;
  const dist = new Int16Array(nSources * nSources);
  if (nSources === 0 || tok1.length === 0) return dist;

  let maxNode = 0;
  for (let i = 0; i < tok1.length; i++) {
    if (tok1[i] > maxNode) maxNode = tok1[i];
    if (tok2[i] > maxNode) maxNode = tok2[i];
  }
  for (let i = 0; i < nSources; i++) {
    if (sourceIndices[i] > maxNode) maxNode = sourceIndices[i];
  }
  const nNodes = maxNode + 1;

  const adj = Array.from({ length: nNodes }, () => []);
  for (let i = 0; i < tok1.length; i++) {
    const u = tok1[i];
    const v = tok2[i];
    adj[u].push(v);
    adj[v].push(u);
  }

  const dRow = new Int32Array(nNodes);
  const queue = new Int32Array(nNodes);

  for (let sIdx = 0; sIdx < nSources; sIdx++) {
    const srcNode = sourceIndices[sIdx];
    dRow.fill(-1);
    dRow[srcNode] = 0;
    let head = 0;
    let tail = 0;
    queue[tail++] = srcNode;

    while (head < tail) {
      const u = queue[head++];
      const du = dRow[u];
      const nbs = adj[u];
      for (let k = 0; k < nbs.length; k++) {
        const v = nbs[k];
        if (dRow[v] === -1) {
          dRow[v] = du + 1;
          queue[tail++] = v;
        }
      }
    }

    const rowOffset = sIdx * nSources;
    for (let tIdx = 0; tIdx < nSources; tIdx++) {
      const d = dRow[sourceIndices[tIdx]];
      if (d > 0) {
        dist[rowOffset + tIdx] =
          pathLengthCutoff !== null && d > pathLengthCutoff ? pathLengthCutoff + 1 : d;
      }
    }
  }

  return dist;
}

/**
 * Compute token-level bond features matching `ComputeTokenLevelBondFeaturesLikeAF3`:
 * - `token_bonds` (bool `[n_tokens, n_tokens]`)
 * - `token_bond_orders` (int16 `[n_tokens, n_tokens]`)
 * - `token_bond_is_aromatic` (bool `[n_tokens, n_tokens]`)
 * - `token_bond_path_length` (int16 `[n_tokens, n_tokens]`)
 * - `canonical_polymer_ordering` (int8 `[n_tokens, n_tokens]`)
 */
export function computeTokenBondFeatures(atoms, { pathLengthCutoff = 64, fillResIdGaps = true } = {}) {
  const nAtoms = atoms.nAtoms;
  const tokenBounds = getTokenStarts(atoms, true);
  const tokenStarts = tokenBounds.slice(0, -1);
  const nTotal = tokenStarts.length;

  const atomToToken = new Int32Array(nAtoms);
  for (let t = 0; t < nTotal; t++) {
    for (let a = tokenBounds[t]; a < tokenBounds[t + 1]; a++) {
      atomToToken[a] = t;
    }
  }

  const insSrc = atoms.insertion_source_idx || new Array(nAtoms).fill(-1);
  const srcPerTok = tokenStarts.map((s) => insSrc[s]);
  const isTokWithSrc = srcPerTok.map((s) => s !== -1);
  const nDirect = isTokWithSrc.filter((x) => !x).length;

  const toksWithSrc = [];
  const srcToks = [];
  for (let t = 0; t < nTotal; t++) {
    if (isTokWithSrc[t]) {
      toksWithSrc.push(t);
      srcToks.push(srcPerTok[t]);
    }
  }

  const idxGrpAnnot = atoms.annotations?.['condition_index-group_1_token'] || getDefaultIndexGroups(atoms);
  const tokGroups = tokenStarts.map((s) => idxGrpAnnot[s]);

  const isDirectAtom = new Uint8Array(nAtoms);
  for (let t = 0; t < nTotal; t++) {
    if (!isTokWithSrc[t]) {
      for (let a = tokenBounds[t]; a < tokenBounds[t + 1]; a++) isDirectAtom[a] = 1;
    }
  }

  // Inter-token bond edges on direct atoms
  const edgeTok1 = [];
  const edgeTok2 = [];
  const edgeBt = [];
  const edgeAtom1 = [];
  const edgeAtom2 = [];

  for (const [u, v, bt] of atoms.bonds || []) {
    if (!isDirectAtom[u] || !isDirectAtom[v]) continue;
    const t1 = atomToToken[u];
    const t2 = atomToToken[v];
    if (t1 !== t2) {
      edgeTok1.push(t1);
      edgeTok2.push(t2);
      edgeBt.push(bt);
      edgeAtom1.push(u);
      edgeAtom2.push(v);
    }
  }

  const directBondMask = new Uint8Array(nDirect * nDirect);
  const ORDER_SENTINEL = 32767;
  const orderWork = new Int16Array(nDirect * nDirect).fill(ORDER_SENTINEL);
  const directIsAromatic = new Uint8Array(nDirect * nDirect);

  for (let e = 0; e < edgeTok1.length; e++) {
    const t1 = edgeTok1[e];
    const t2 = edgeTok2[e];
    const bt = edgeBt[e];
    const ord = BOND_TYPE_TO_ORDER[bt] ?? 1;
    const arom = AROMATIC_BOND_SET.has(bt) ? 1 : 0;

    directBondMask[t1 * nDirect + t2] = 1;
    directBondMask[t2 * nDirect + t1] = 1;

    if (ord < orderWork[t1 * nDirect + t2]) orderWork[t1 * nDirect + t2] = ord;
    if (ord < orderWork[t2 * nDirect + t1]) orderWork[t2 * nDirect + t1] = ord;

    if (arom) {
      directIsAromatic[t1 * nDirect + t2] = 1;
      directIsAromatic[t2 * nDirect + t1] = 1;
    }
  }

  const directBondOrder = new Int16Array(nDirect * nDirect);
  for (let i = 0; i < directBondOrder.length; i++) {
    directBondOrder[i] = orderWork[i] < ORDER_SENTINEL ? orderWork[i] : 0;
  }
  for (let t = 0; t < nDirect; t++) {
    directBondMask[t * nDirect + t] = 0;
    directBondOrder[t * nDirect + t] = 0;
    directIsAromatic[t * nDirect + t] = 0;
  }

  const directTokStarts = tokenStarts.filter((_, t) => !isTokWithSrc[t]);
  const isPoly = directTokStarts.map((s) => !(atoms.atomize && atoms.atomize[s]));
  const directTokGroups = tokGroups.filter((_, t) => !isTokWithSrc[t]);

  const tokenTypePerDirect = directTokStarts.map((s) => {
    const tt = atoms.token_type ? atoms.token_type[s] : TOKEN_TYPE.PROTEIN;
    return tt === TOKEN_TYPE.NCAA ? TOKEN_TYPE.PROTEIN : tt;
  });
  const isCanonical = tokenTypePerDirect.map(
    (tt) => tt === TOKEN_TYPE.PROTEIN || tt === TOKEN_TYPE.RNA || tt === TOKEN_TYPE.DNA
  );

  // Upstream direction of each inter-token edge
  const segmin = atoms.annotations?.['annotation_expsegmin_1_residue'];
  const standsIn = Array.from({ length: nAtoms }, (_, i) => Boolean(segmin && segmin[i] !== -1));
  const anyStandsIn = standsIn.some(Boolean);

  const atomTokenTypes = Array.from({ length: nAtoms }, (_, i) =>
    atoms.token_type ? atoms.token_type[i] : TOKEN_TYPE.PROTEIN
  );

  const isLeavingAndEntering = (atomIdx) => {
    const [leaving, entering] = getPolymerizationAtomNamesForToken(
      atoms.res_name[atomIdx],
      atomTokenTypes[atomIdx],
      atoms.chain_type?.[atomIdx]
    );
    const an = atoms.atom_name[atomIdx];
    const s = standsIn[atomIdx];
    return [an === leaving || s, an === entering || s];
  };

  const resIdPerDirectTok = directTokStarts.map((s) => atoms.res_id[s]);
  const molPerDirectTok = directTokStarts.map((s) => atoms.molecule_iid[s]);

  const upToks = [];
  const dnToks = [];

  for (let e = 0; e < edgeTok1.length; e++) {
    const a1 = edgeAtom1[e];
    const a2 = edgeAtom2[e];
    const t1 = edgeTok1[e];
    const t2 = edgeTok2[e];
    const [leaves1, enters1] = isLeavingAndEntering(a1);
    const [leaves2, enters2] = isLeavingAndEntering(a2);
    let fwd = leaves1 && enters2;
    let rev = leaves2 && enters1;

    if (anyStandsIn && standsIn[a1] && standsIn[a2]) {
      const sameMol = molPerDirectTok[t1] === molPerDirectTok[t2];
      const closesRing = sameMol && Math.abs(resIdPerDirectTok[t1] - resIdPerDirectTok[t2]) > 1;
      const tok1Upstream = Boolean((t1 < t2) ^ (closesRing ? 1 : 0));
      fwd = tok1Upstream;
      rev = !tok1Upstream;
    }

    if (fwd) {
      upToks.push(t1);
      dnToks.push(t2);
    }
    if (rev) {
      upToks.push(t2);
      dnToks.push(t1);
    }
  }

  const directCanonical = new Int8Array(nDirect * nDirect);
  const suppress = new Uint8Array(nDirect * nDirect);
  for (let k = 0; k < upToks.length; k++) {
    const up = upToks[k];
    const dn = dnToks[k];
    directCanonical[dn * nDirect + up] = 1;
    directCanonical[up * nDirect + dn] = -1;
    suppress[up * nDirect + dn] = 1;
    suppress[dn * nDirect + up] = 1;
  }

  for (let i = 0; i < nDirect; i++) {
    for (let j = 0; j < nDirect; j++) {
      const idx = i * nDirect + j;
      const polyPoly = isPoly[i] && isPoly[j];
      const sameGroup = directTokGroups[i] === directTokGroups[j];
      const eligible =
        polyPoly &&
        isCanonical[i] &&
        isCanonical[j] &&
        tokenTypePerDirect[i] === tokenTypePerDirect[j] &&
        sameGroup &&
        directBondMask[idx] === 1;
      if (!eligible) directCanonical[idx] = 0;

      const sup = (suppress[idx] === 1 && polyPoly) || !sameGroup;
      if (sup) {
        directBondMask[idx] = 0;
        directBondOrder[idx] = 0;
        directIsAromatic[idx] = 0;
      }
    }
  }

  // Compute path lengths (with optional res_id gap filling for inference)
  let pathTok1 = edgeTok1.slice();
  let pathTok2 = edgeTok2.slice();
  let sourceIndices = Array.from({ length: nDirect }, (_, i) => i);

  if (fillResIdGaps) {
    // Replicate `_fill_res_id_gaps`: find unbonded same-(group, chain_iid) polymer residues with res_id gap > 1
    // that are adjacent in the atom array, and add virtual filler token nodes between their polymerization atoms.
    const resStarts = getResidueSegments(atoms);
    const nRes = resStarts.length - 1;
    const atomToRes = new Int32Array(nAtoms);
    for (let r = 0; r < nRes; r++) {
      for (let a = resStarts[r]; a < resStarts[r + 1]; a++) atomToRes[a] = r;
    }

    // Connected components over residues
    const resComp = Array.from({ length: nRes }, (_, r) => r);
    const findComp = (x) => {
      while (resComp[x] !== x) {
        resComp[x] = resComp[resComp[x]];
        x = resComp[x];
      }
      return x;
    };
    const unionComp = (a, b) => {
      const ra = findComp(a);
      const rb = findComp(b);
      if (ra !== rb) resComp[rb] = ra;
    };
    for (const [u, v] of atoms.bonds || []) {
      unionComp(atomToRes[u], atomToRes[v]);
    }

    const resGroups = resStarts.slice(0, nRes).map((s) => idxGrpAnnot[s]);
    const resChainIids = resStarts.slice(0, nRes).map((s) => atoms.chain_iid[s]);
    const resIds = resStarts.slice(0, nRes).map((s) => atoms.res_id[s]);
    const resIsPoly = resStarts.slice(0, nRes).map((s) => atoms.is_polymer[s]);

    // Map all atoms (not only direct) to tokens in the full token graph
    const allEdgeTok1 = [];
    const allEdgeTok2 = [];
    for (const [u, v] of atoms.bonds || []) {
      const t1 = atomToToken[u];
      const t2 = atomToToken[v];
      if (t1 !== t2) {
        allEdgeTok1.push(t1);
        allEdgeTok2.push(t2);
      }
    }

    let nextVirtualToken = nTotal;
    const uniqueGroups = Array.from(new Set(resGroups)).sort((a, b) => a - b);
    for (const grp of uniqueGroups) {
      const uniqueChains = Array.from(
        new Set(resChainIids.filter((_, r) => resGroups[r] === grp))
      ).sort();
      for (const cid of uniqueChains) {
        const members = [];
        for (let r = 0; r < nRes; r++) {
          if (resIsPoly[r] && resGroups[r] === grp && resChainIids[r] === cid) {
            members.push(r);
          }
        }
        // Stable sort by res_id
        members.sort((a, b) => resIds[a] - resIds[b] || a - b);
        for (let m = 0; m < members.length - 1; m++) {
          const prevRes = members[m];
          const nextRes = members[m + 1];
          if (findComp(prevRes) === findComp(nextRes)) continue;
          const gap = Math.max(resIds[nextRes] - resIds[prevRes], 1);
          unionComp(prevRes, nextRes);
          const nFillers = gap - 1;
          if (nFillers <= 0) continue;
          if (resStarts[prevRes + 1] !== resStarts[nextRes]) continue;

          // Wire virtual filler tokens between linkAtom(prevRes, exiting=true) and linkAtom(nextRes, exiting=false)
          const firstV = nextVirtualToken;
          for (let f = 0; f < nFillers - 1; f++) {
            allEdgeTok1.push(nextVirtualToken + f);
            allEdgeTok2.push(nextVirtualToken + f + 1);
          }
          nextVirtualToken += nFillers;
          const lastV = nextVirtualToken - 1;

          const leftAtom = findPolymerizationAtomIdx(
            atoms,
            resStarts[prevRes],
            resStarts[prevRes + 1],
            atomTokenTypes[resStarts[prevRes]],
            true
          );
          const rightAtom = findPolymerizationAtomIdx(
            atoms,
            resStarts[nextRes],
            resStarts[nextRes + 1],
            atomTokenTypes[resStarts[nextRes]],
            false
          );
          if (leftAtom !== null) {
            allEdgeTok1.push(atomToToken[leftAtom]);
            allEdgeTok2.push(firstV);
          }
          if (rightAtom !== null) {
            allEdgeTok1.push(lastV);
            allEdgeTok2.push(atomToToken[rightAtom]);
          }
        }
      }
    }

    pathTok1 = allEdgeTok1;
    pathTok2 = allEdgeTok2;
    sourceIndices = [];
    for (let t = 0; t < nTotal; t++) {
      if (!isTokWithSrc[t]) sourceIndices.push(t);
    }
  }

  const directPathLength = computeTokenPathLengths(pathTok1, pathTok2, sourceIndices, pathLengthCutoff);
  for (let i = 0; i < nDirect; i++) {
    for (let j = 0; j < nDirect; j++) {
      if (i === j || directTokGroups[i] !== directTokGroups[j]) {
        directPathLength[i * nDirect + j] = 0;
      }
    }
  }

  // Assemble [nTotal x nTotal] outputs
  const outBondMask = new Uint8Array(nTotal * nTotal);
  const outBondOrder = new Int16Array(nTotal * nTotal);
  const outIsAromatic = new Uint8Array(nTotal * nTotal);
  const outPathLength = new Int16Array(nTotal * nTotal);
  const outCanonical = new Int8Array(nTotal * nTotal);

  for (let i = 0; i < nDirect; i++) {
    for (let j = 0; j < nDirect; j++) {
      const srcIdx = i * nDirect + j;
      const dstIdx = i * nTotal + j;
      outBondMask[dstIdx] = directBondMask[srcIdx];
      outBondOrder[dstIdx] = directBondOrder[srcIdx];
      outIsAromatic[dstIdx] = directIsAromatic[srcIdx];
      outPathLength[dstIdx] = directPathLength[srcIdx];
      outCanonical[dstIdx] = directCanonical[srcIdx];
    }
  }

  // Copy features for unindexed borrower tokens within each IndexGroup
  const borrowerGroups = new Set(toksWithSrc.map((t) => tokGroups[t]));
  for (const grp of borrowerGroups) {
    const grpToks = [];
    const grpSrcs = [];
    for (let k = 0; k < toksWithSrc.length; k++) {
      if (tokGroups[toksWithSrc[k]] === grp) {
        grpToks.push(toksWithSrc[k]);
        grpSrcs.push(srcToks[k]);
      }
    }
    for (let i = 0; i < grpToks.length; i++) {
      for (let j = 0; j < grpToks.length; j++) {
        const dstIdx = grpToks[i] * nTotal + grpToks[j];
        const srcIdx = grpSrcs[i] * nDirect + grpSrcs[j];
        outBondMask[dstIdx] = directBondMask[srcIdx];
        outBondOrder[dstIdx] = directBondOrder[srcIdx];
        outIsAromatic[dstIdx] = directIsAromatic[srcIdx];
        outPathLength[dstIdx] = directPathLength[srcIdx];
        outCanonical[dstIdx] = directCanonical[srcIdx];
      }
    }
  }

  return {
    bond_mask: { data: outBondMask, shape: [nTotal, nTotal] },
    bond_order: { data: outBondOrder, shape: [nTotal, nTotal] },
    is_aromatic: { data: outIsAromatic, shape: [nTotal, nTotal] },
    token_bond_path_length: { data: outPathLength, shape: [nTotal, nTotal] },
    canonical_polymer_ordering: { data: outCanonical, shape: [nTotal, nTotal] },
  };
}
