/**
 * Which residue of one sequence is which residue of another.
 *
 * WHY THIS EXISTS. Superposing two predictions means pairing their residues off,
 * and while an edit only ever substitutes, position i is position i and there is
 * nothing to work out. Insert or delete one residue and every position after it
 * shifts: pairing by index would fit the tail of one structure onto the wrong
 * part of the other and swing the whole thing round to do it. So before an indel
 * can be superposed at all, something has to say which positions correspond.
 *
 * SMITH-WATERMAN, local rather than global. The two sequences are the same
 * protein a moment apart, so a global alignment would usually do - but it forces
 * the ends to pair even when they should not, and a truncation or a run of new
 * residues at one terminus would drag the whole fit round to honour a
 * correspondence that is not there. Local alignment answers the question a
 * superposition actually asks: which stretch do these two genuinely share?
 *
 * MATCH SCORES 2, NOT 1, and the reason is narrow but real. Local alignment
 * floors its running score at zero, so a mismatch is survivable only if enough
 * score has been banked before it. At match 1 a substitution in the SECOND
 * residue costs exactly what the first one earned, the score hits the floor, and
 * the alignment restarts past it - dropping the N-terminal residue from the fit.
 * Mid-sequence there is plenty banked and either score runs through; it is the
 * first residue or two where the difference shows. At 2 against -1 a mismatch
 * costs less than a match earns, so one bad residue can never zero the walk.
 *
 * A MISMATCH IS STILL A PAIR. The point is to fit the backbone, and a substituted
 * residue has one: it is the residue whose side chain changed, which is usually
 * the very thing the reader wants to look at. Only a gap goes unpaired.
 */

const DEFAULTS = { match: 2, mismatch: -1, gap: -2 };

/**
 * Align two sequences and report the positions that pair up.
 *
 * @param {string} a
 * @param {string} b
 * @param {{match?: number, mismatch?: number, gap?: number}} [scores]
 * @returns {Array<[number, number]>} `[indexInA, indexInB]`, ascending, gaps omitted
 */
export function alignPositions(a, b, scores = {}) {
  const { match, mismatch, gap } = { ...DEFAULTS, ...scores };
  const n = a.length;
  const m = b.length;
  if (n === 0 || m === 0) return [];

  // The traceback: 0 diagonal (a pair), 1 up (a gap in b), 2 left (a gap in a),
  // 3 the floor - a cell no better than starting afresh, where a walk stops.
  const score = new Float64Array((n + 1) * (m + 1));
  const from = new Uint8Array((n + 1) * (m + 1));
  const at = (i, j) => i * (m + 1) + j;
  let bestScore = 0;
  let bestI = 0;
  let bestJ = 0;

  for (let i = 1; i <= n; i += 1) {
    for (let j = 1; j <= m; j += 1) {
      const diagonal = score[at(i - 1, j - 1)] + (a[i - 1] === b[j - 1] ? match : mismatch);
      const up = score[at(i - 1, j)] + gap;
      const left = score[at(i, j - 1)] + gap;
      // ...AND ZERO, which is what makes this local: a cell worth less than
      // nothing is not worth continuing from, so the alignment may start again.
      let best = 0;
      let direction = 3;
      if (diagonal > best) { best = diagonal; direction = 0; }
      if (up > best) { best = up; direction = 1; }
      if (left > best) { best = left; direction = 2; }
      score[at(i, j)] = best;
      from[at(i, j)] = direction;
      if (best > bestScore) { bestScore = best; bestI = i; bestJ = j; }
    }
  }

  // ...walked back from the BEST cell, not the corner, and stopping at the floor.
  const pairs = [];
  let i = bestI;
  let j = bestJ;
  while (i > 0 && j > 0) {
    const direction = from[at(i, j)];
    if (direction === 3) break;
    if (direction === 0) { pairs.push([i - 1, j - 1]); i -= 1; j -= 1; }
    else if (direction === 1) { i -= 1; }
    else { j -= 1; }
  }
  return pairs.reverse();
}

/**
 * The alignment as it is actually needed: two index lists, same length.
 *
 * Separate from alignPositions because the common case - same length, no indel -
 * does not need a table at all, and running one would be a lot of arithmetic to
 * rediscover that i pairs with i.
 *
 * @returns {{from: number[], to: number[], identical: boolean}}
 */
export function correspondence(a, b) {
  if (a.length === b.length) {
    const index = Array.from({ length: a.length }, (_, i) => i);
    return { from: index, to: index, identical: true };
  }
  const pairs = alignPositions(a, b);
  return {
    from: pairs.map(([i]) => i),
    to: pairs.map(([, j]) => j),
    identical: false,
  };
}
