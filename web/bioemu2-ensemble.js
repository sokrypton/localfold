/**
 * A BioEmu2 ensemble as a distribution: where each sample sits, and what the samples agree on.
 *
 * WHAT IT OWNS: one representative atom a token (its CA or C1', else its first atom - a ligand
 * is one atom a token anyway), the PCA of the samples' pairwise distances between those atoms,
 * and the fraction of samples in which two tokens are in contact. The page puts the PCA on
 * py2Dmol's scatter plot (one point a frame, the current one highlighted, a click jumps to it)
 * and orders the play bar along PC1, so the animation sweeps across the distribution rather
 * than through the order the samples happened to be drawn.
 *
 * 🔴 DISTANCES, NOT COORDINATES. A distance matrix needs no superposition, so the plot does not
 * depend on which frame anything was aligned to, and a sample that is the same shape rotated is
 * the same point.
 */

/** Pairs within this many angstroms count as a contact - the 8 A CA rule the contact maps use. */
export const ENSEMBLE_CONTACT_CUTOFF = 8;
/** Above this many representatives the PCA strides over them, so the features stay ~45k wide. */
const PCA_MAX_POINTS = 300;

/**
 * Each token's representative atom, in the featuriser's atom order.
 * @param topology the featuriser's `topology` (names, tokenIdxAtAtom)
 * @param tokens the token count
 * @returns {Int32Array} an atom index a token, -1 for a token with no atom
 */
export function representativeAtoms(topology, tokens) {
  const first = new Int32Array(tokens).fill(-1);
  const centre = new Int32Array(tokens).fill(-1);
  for (let a = 0; a < topology.names.length; a += 1) {
    const token = topology.tokenIdxAtAtom[a];
    if (token < 0 || token >= tokens) continue;
    if (first[token] < 0) first[token] = a;
    if (centre[token] < 0 && (topology.names[a] === "CA" || topology.names[a] === "C1'")) centre[token] = a;
  }
  return centre.map((a, token) => (a >= 0 ? a : first[token]));
}

/**
 * How often each token pair is in contact across the samples: a tokens x tokens matrix of
 * fractions, the shape `contactMapFor` takes. A token with no atom reads 0 everywhere.
 */
export function ensembleContacts(samples, reps, cutoff = ENSEMBLE_CONTACT_CUTOFF) {
  const T = reps.length;
  const out = new Float32Array(T * T);
  if (samples.length === 0) return out;
  const cutoff2 = cutoff * cutoff;
  for (const x of samples) {
    for (let i = 0; i < T; i += 1) {
      const a = reps[i];
      if (a < 0) continue;
      for (let j = i; j < T; j += 1) {
        const b = reps[j];
        if (b < 0) continue;
        const dx = x[a * 3] - x[b * 3];
        const dy = x[a * 3 + 1] - x[b * 3 + 1];
        const dz = x[a * 3 + 2] - x[b * 3 + 2];
        if (dx * dx + dy * dy + dz * dz < cutoff2) {
          out[i * T + j] += 1;
          if (i !== j) out[j * T + i] += 1;
        }
      }
    }
  }
  for (let k = 0; k < out.length; k += 1) out[k] /= samples.length;
  return out;
}

/** One sample's upper-triangle distances between the given atoms. */
export function distanceFeatures(x, atoms) {
  const n = atoms.length;
  const out = new Float64Array((n * (n - 1)) / 2);
  let k = 0;
  for (let i = 0; i < n; i += 1) {
    const a = atoms[i] * 3;
    for (let j = i + 1; j < n; j += 1) {
      const b = atoms[j] * 3;
      out[k] = Math.hypot(x[a] - x[b], x[a + 1] - x[b + 1], x[a + 2] - x[b + 2]);
      k += 1;
    }
  }
  return out;
}

/**
 * The first two principal components of `rows` (one feature vector a sample), through the
 * samples' Gram matrix - n x n for n samples, however wide the features are.
 *
 * Each component's sign is fixed so its most extreme sample scores positive, which keeps the
 * plot from flipping as samples are added.
 *
 * @returns {{scores: [number, number][], explained: [number, number]}} per-sample (PC1, PC2), and
 *   each component's share of the total variance; null below three samples
 */
export function pca2(rows) {
  const n = rows.length;
  if (n < 3) return null;
  const d = rows[0].length;
  const mean = new Float64Array(d);
  for (const row of rows) for (let k = 0; k < d; k += 1) mean[k] += row[k];
  for (let k = 0; k < d; k += 1) mean[k] /= n;
  const centred = rows.map((row) => {
    const c = new Float64Array(d);
    for (let k = 0; k < d; k += 1) c[k] = row[k] - mean[k];
    return c;
  });
  const gram = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i += 1) {
    for (let j = i; j < n; j += 1) {
      let s = 0;
      const a = centred[i];
      const b = centred[j];
      for (let k = 0; k < d; k += 1) s += a[k] * b[k];
      gram[i][j] = s;
      gram[j][i] = s;
    }
  }
  let trace = 0;
  for (let i = 0; i < n; i += 1) trace += gram[i][i];
  const components = [];
  for (let c = 0; c < 2; c += 1) {
    const { vector, value } = topEigen(gram, components);
    components.push({ vector, value: Math.max(0, value) });
  }
  const scores = Array.from({ length: n }, (_, i) => components.map(({ vector, value }) => vector[i] * Math.sqrt(value)));
  for (let c = 0; c < 2; c += 1) {
    let extreme = 0;
    for (let i = 1; i < n; i += 1) if (Math.abs(scores[i][c]) > Math.abs(scores[extreme][c])) extreme = i;
    if (scores[extreme][c] < 0) for (const score of scores) score[c] = -score[c];
  }
  const share = (value) => (trace > 0 ? value / trace : 0);
  return { scores, explained: [share(components[0].value), share(components[1].value)] };
}

/** The largest eigenpair of a symmetric matrix, orthogonal to the vectors already found. */
function topEigen(matrix, found) {
  const n = matrix.length;
  let v = Float64Array.from({ length: n }, (_, i) => 1 + i / n);
  const deflate = (u) => {
    for (const { vector } of found) {
      let dot = 0;
      for (let i = 0; i < n; i += 1) dot += u[i] * vector[i];
      for (let i = 0; i < n; i += 1) u[i] -= dot * vector[i];
    }
  };
  const normalise = (u) => {
    let norm = 0;
    for (let i = 0; i < n; i += 1) norm += u[i] * u[i];
    norm = Math.sqrt(norm);
    if (norm === 0) return false;
    for (let i = 0; i < n; i += 1) u[i] /= norm;
    return true;
  };
  deflate(v);
  if (!normalise(v)) return { vector: new Float64Array(n), value: 0 };
  let value = 0;
  for (let iteration = 0; iteration < 1000; iteration += 1) {
    const next = new Float64Array(n);
    for (let i = 0; i < n; i += 1) {
      let s = 0;
      for (let j = 0; j < n; j += 1) s += matrix[i][j] * v[j];
      next[i] = s;
    }
    deflate(next);
    let rayleigh = 0;
    for (let i = 0; i < n; i += 1) rayleigh += v[i] * next[i];
    if (!normalise(next)) return { vector: v, value: 0 };
    let change = 0;
    for (let i = 0; i < n; i += 1) change = Math.max(change, Math.abs(Math.abs(next[i]) - Math.abs(v[i])));
    v = next;
    value = rayleigh;
    if (change < 1e-10) break;
  }
  return { vector: v, value };
}

/**
 * The ensemble's PCA: the representatives' pairwise distances, strided past PCA_MAX_POINTS.
 * @returns the pca2 result, or null below three samples or three representatives
 */
export function ensemblePca(samples, reps) {
  const atoms = Array.from(reps).filter((a) => a >= 0);
  const stride = Math.max(1, Math.ceil(atoms.length / PCA_MAX_POINTS));
  const used = atoms.filter((_, i) => i % stride === 0);
  if (used.length < 3) return null;
  return pca2(samples.map((x) => distanceFeatures(x, used)));
}

/** Sample indices in PC1 order, lowest first; draw order where there is no PCA. */
export function orderAlongPc1(pca, count) {
  const order = Array.from({ length: count }, (_, i) => i);
  if (pca === null) return order;
  return order.sort((a, b) => pca.scores[a][0] - pca.scores[b][0] || a - b);
}
