/**
 * Superposing a model on a crystal, and the two numbers that come out of it.
 *
 * 🔴 LIFTED VERBATIM OUT OF fold-opendde.js, NOT REWRITTEN. A first attempt at
 * a complex scorer hand-rolled this again and garbled the Newton polar
 * iteration - the rotation is `R <- (R + (R^-1)^T) / 2` and the reinvention had
 * the transpose in the wrong place, which returns a plausible matrix and a
 * plausible RMSD. Numerics that already exist and are already trusted get
 * imported.
 *
 * 🔴 AND `superposeInto` EXISTS FOR COMPLEXES. Fitting each chain separately
 * reports two perfect chains that are nowhere near each other as a perfect
 * answer, so a complex is fitted ONCE over every chain together and each chain
 * is then scored in that shared frame - which is what `place` is for.
 */

export function superpose(model, truth) {
  const pairs = model.map((p, i) => [p, truth[i]]).filter(([a, b]) => a && b);
  const n = pairs.length;
  const centre = (which) => {
    const c = [0, 0, 0];
    for (const pair of pairs) for (let d = 0; d < 3; d += 1) c[d] += pair[which][d] / n;
    return c;
  };
  const cm = centre(0);
  const ct = centre(1);
  const covariance = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const [a, b] of pairs) {
    for (let i = 0; i < 3; i += 1) {
      for (let j = 0; j < 3; j += 1) covariance[i][j] += (a[i] - cm[i]) * (b[j] - ct[j]);
    }
  }
  // Rotation by iterative polar decomposition - enough for a score, and it
  // avoids a second SVD in the tree.
  let rotation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const multiply = (x, y) => x.map((row, i) => y[0].map((_, j) =>
    row.reduce((s, v, k) => s + v * y[k][j], 0)));
  const transpose = (m) => m[0].map((_, j) => m.map((row) => row[j]));
  const inverse3 = (m) => {
    const det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
      - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
      + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    const c = (i, j) => {
      const rows = [0, 1, 2].filter((r) => r !== i);
      const cols = [0, 1, 2].filter((cc) => cc !== j);
      return ((i + j) % 2 ? -1 : 1)
        * (m[rows[0]][cols[0]] * m[rows[1]][cols[1]] - m[rows[0]][cols[1]] * m[rows[1]][cols[0]]);
    };
    return [0, 1, 2].map((i) => [0, 1, 2].map((j) => c(j, i) / det));
  };
  rotation = covariance;
  for (let iteration = 0; iteration < 40; iteration += 1) {
    const next = transpose(inverse3(rotation)).map((row, i) => row.map((v, j) =>
      0.5 * (rotation[i][j] + v)));
    rotation = next;
  }
  let squared = 0;
  const d0 = 1.24 * Math.cbrt(Math.max(n - 15, 1)) - 1.8;
  let tm = 0;
  const deviations = [];
  for (const [a, b] of pairs) {
    const moved = [0, 1, 2].map((i) =>
      [0, 1, 2].reduce((s, k) => s + (a[k] - cm[k]) * rotation[k][i], 0) + ct[i]);
    const d2 = [0, 1, 2].reduce((s, i) => s + (moved[i] - b[i]) ** 2, 0);
    squared += d2;
    deviations.push(Math.sqrt(d2));
    tm += 1 / (1 + d2 / (d0 * d0));
  }
  return { rmsd: Math.sqrt(squared / n), tm: tm / n, pairs: n, deviations,
           // The fitted transform, so a caller can score SUBSETS of the same
           // pairing in this one frame - see the note above on complexes.
           place: (p) => [0, 1, 2].map((i) =>
             [0, 1, 2].reduce((s, k) => s + (p[k] - cm[k]) * rotation[k][i], 0) + ct[i]),
           d0 };
}

/**
 * RMSD and TM over a SLICE of an already-fitted pairing.
 *
 * `d0` is the whole complex's, deliberately: TM-score's length term is a
 * property of the target being scored, and re-deriving it per chain would make
 * a two-chain complex's per-chain numbers incomparable with its total.
 */
export function scoreSlice(model, truth, place, d0, from, count) {
  let squared = 0;
  let tm = 0;
  let n = 0;
  for (let at = from; at < from + count; at += 1) {
    const a = model[at];
    const b = truth[at];
    if (!a || !b) continue;
    const moved = place(a);
    const d2 = [0, 1, 2].reduce((s, i) => s + (moved[i] - b[i]) ** 2, 0);
    squared += d2;
    tm += 1 / (1 + d2 / (d0 * d0));
    n += 1;
  }
  return n === 0 ? null : { residues: n, rmsd: Math.sqrt(squared / n), tm: tm / n };
}

/**
 * The model's alpha carbons, one per token, in token order - `null` where a
 * token has no CA slot (a ligand, a nucleotide).
 *
 * 🔴 IT FINDS THE SLOT BY NAME AND THE NAME IS PACKED FOUR CHARACTERS TO A
 * TOKEN, offset by 32. Every caller that wanted a CA had written this loop
 * again; `fold-opendde.js` had it inline and the first complex scorer copied
 * it. A batch's dense slot count is `batch.dense` and it is NOT four - the
 * slot a CA lands in differs by residue.
 */
export function modelAlphaCarbons(batch, positions) {
  const out = [];
  for (let token = 0; token < batch.tokens; token += 1) {
    let slot = -1;
    for (let s = 0; s < batch.dense; s += 1) {
      const base = (token * batch.dense + s) * 4;
      const name = [0, 1, 2, 3].map((c) => {
        const v = batch.refAtomNameChars[base + c];
        return v > 0 ? String.fromCharCode(v + 32) : "";
      }).join("");
      if (name === "CA") { slot = s; break; }
    }
    out.push(slot < 0 ? null : [
      positions[(token * batch.dense + slot) * 3],
      positions[(token * batch.dense + slot) * 3 + 1],
      positions[(token * batch.dense + slot) * 3 + 2]]);
  }
  return out;
}

/**
 * Every relabelling of INTERCHANGEABLE chains, as arrays mapping a model chain
 * slot to the truth chain it should be scored against.
 *
 * 🔴 A HOMODIMER'S CHAINS ARE INTERCHANGEABLE AND A SCORE THAT IGNORES THAT IS
 * WRONG, NOT CONSERVATIVE. 5CAJ's A and B are the same 261-residue sequence, so
 * a perfect prediction with the two labels the other way round superposes as a
 * total failure - and the tool would report a placement defect that is not
 * there. AlphaFold 3 does this itself and calls it chain permutation alignment.
 *
 * 🔴 AND ONLY IDENTICAL SEQUENCES MAY SWAP. Permuting two DIFFERENT chains is
 * not a relabelling, it is a different answer, and taking the best over those
 * would let a scorer report a fold that put the wrong protein in the right
 * place as correct.
 *
 * @param {string[]} sequences one per chain, in model order
 * @param {{limit?: number}} [options] refuse above this many assignments
 * @returns {number[][]} each is `assignment[modelSlot] = truthSlot`; the first
 *   is always the identity, so a caller can report the unpermuted score beside
 *   the best one.
 */
export function chainAssignments(sequences, options = {}) {
  const limit = options.limit ?? 720;
  const groups = new Map();
  sequences.forEach((sequence, index) => {
    if (!groups.has(sequence)) groups.set(sequence, []);
    groups.get(sequence).push(index);
  });
  const permutationsOf = (items) => (items.length <= 1 ? [items]
    : items.flatMap((item, index) => permutationsOf(
      [...items.slice(0, index), ...items.slice(index + 1)]).map((rest) => [item, ...rest])));
  let assignments = [[]];
  for (const [, members] of groups) {
    const orders = permutationsOf(members);
    // 🔴 CHECKED AS IT GROWS, NOT AT THE END. Eight identical chains is 40,320
    // assignments and enumerating them to find out is the hang this refuses.
    if (assignments.length * orders.length > limit) {
      throw new Error(`${assignments.length * orders.length} chain permutations to try;`
        + ` this scorer enumerates them and refuses above ${limit}`);
    }
    assignments = assignments.flatMap((partial) => orders.map((order) => {
      const next = [...partial];
      members.forEach((slot, index) => { next[slot] = order[index]; });
      return next;
    }));
  }
  return assignments;
}

/**
 * The same fit, in the shape `fittedPdb` asks a viewer library for.
 *
 * 🔴 TWO CONVENTIONS FOR ONE KABSCH, AND THE ADAPTER IS WHERE THEY MEET.
 * py2Dmol publishes `superpose(mobile, reference, {from, to})` - index arrays
 * naming the points the fit is COMPUTED from, with the transform applied to
 * every point of `mobile` - and `superpose` above pairs by position and hands
 * back a `place`. A runtime with no page has no py2Dmol, so without this a
 * streamed AlphaFold 3 trajectory TUMBLES: the sampler re-augments the whole
 * system every step, and unfitted frames differ by a rigid motion far larger
 * than anything the denoiser did.
 *
 * @param {number[][]} mobile every point of the frame being placed
 * @param {number[][]} reference the frame it is placed onto
 * @param {{from?: number[], to?: number[]}} [slots] which points to fit on
 */
export function superposeApi(mobile, reference, slots = {}) {
  const from = slots.from ?? mobile.map((_, index) => index);
  const to = slots.to ?? reference.map((_, index) => index);
  // Paired by position, so the two subsets must be the same length - which is
  // what `fittedPdb` passes, the same slot list for both.
  const { place } = superpose(from.map((i) => mobile[i]), to.map((i) => reference[i]));
  return mobile.map((point) => place(point));
}
