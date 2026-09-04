// Two predictions of the same sequence length, superposed and interpolated, so
// that applying a mutation shows you what MOVED rather than replacing one
// picture with another.

/** Slot of the alpha carbon in the atom37 layout, and of the beta carbon. */
const CA = 1;
const CB = 3;

/**
 * The alpha carbons, as a flat list of atom37 slot indices.
 *
 * ONE PER RESIDUE AND ALWAYS PRESENT - every residue has a CA, glycine
 * included - which is what makes them the right atoms to fit on. The side
 * chains are exactly what a mutation changes, so a fit that used them would be
 * pulled about by the thing it is meant to be measuring.
 *
 * @param {number} length residues
 * @returns {number[]} indices into a length*37 atom list
 */
export function alphaCarbons(length) {
  return Array.from({ length }, (_, residue) => residue * 37 + CA);
}

/** atom37 as a list of [x, y, z], one entry per slot, present or not. */
function toPoints(atom37, length) {
  const out = new Array(length * 37);
  for (let i = 0; i < out.length; i += 1) {
    out[i] = [atom37[i * 3], atom37[i * 3 + 1], atom37[i * 3 + 2]];
  }
  return out;
}

/** ...and back. */
function toAtom37(points) {
  const out = new Float32Array(points.length * 3);
  for (let i = 0; i < points.length; i += 1) {
    out[i * 3] = points[i][0];
    out[i * 3 + 1] = points[i][1];
    out[i * 3 + 2] = points[i][2];
  }
  return out;
}

/**
 * Where each of the NEW structure's atoms starts from.
 *
 * 🔴 A MUTATION CHANGES ONE RESIDUE'S ATOM LIST, so the two structures do not
 * have the same atoms and a straight pairwise interpolation is undefined
 * exactly where the interesting thing is happening. Every frame of the morph
 * carries the NEW atom set - anything else would make the frames different
 * lengths, which py2Dmol cannot align and cannot cache per field - and an atom
 * the old structure did not have starts at its residue's ANCHOR: the old CB,
 * or the old CA where there was no CB (glycine, and glycine is the common
 * case, since mutating away from it is how you get a side chain at all).
 *
 * So a new side chain grows out of the stub that was there, which is both the
 * cheapest thing to compute and the truest picture of what happened. The
 * alternative - holding those atoms at their final positions while everything
 * else moves - puts a finished tryptophan in the scene from frame one.
 *
 * @param {{atom37: Float32Array, atom37Mask: Float32Array}} from the old structure
 * @param {{atom37: Float32Array, atom37Mask: Float32Array}} to the new one, already superposed
 * @param {number} length residues
 * @returns {Array<Array<number>>} start points, one per atom37 slot of `to`
 */
export function anchoredStart(from, to, length) {
  const oldPoints = toPoints(from.atom37, length);
  const newPoints = toPoints(to.atom37, length);
  const start = new Array(length * 37);
  for (let residue = 0; residue < length; residue += 1) {
    const base = residue * 37;
    const anchor = from.atom37Mask[base + CB] >= 0.5
      ? oldPoints[base + CB] : oldPoints[base + CA];
    for (let atom = 0; atom < 37; atom += 1) {
      const index = base + atom;
      start[index] = from.atom37Mask[index] >= 0.5 ? oldPoints[index]
        : (to.atom37Mask[index] >= 0.5 ? anchor : newPoints[index]);
    }
  }
  return start;
}

/**
 * The morph, as structures a PDB writer will take.
 *
 * `steps` frames INCLUSIVE OF BOTH ENDS, so 2 is a hard cut and 12 is about a
 * second at the play bar's rate. Linear in Cartesian space: the two structures
 * are superposed and differ by a few Angstrom over most residues, so the
 * straight line between two atoms and the arc a real motion would take are
 * indistinguishable at this scale - and an arc would need a torsion-space
 * model of a move that is not physical anyway.
 *
 * 🔴 THE INTERMEDIATE FRAMES CARRY NO pLDDT, AND THE RAMP PAINTS THAT RED.
 * They used to interpolate it between the two ends, which colours a structure
 * that never existed with a confidence nothing measured - the two ends are
 * predictions and everything between them is an animation. The ends keep their
 * own real values, so the morph starts and finishes on the truth and is
 * honestly blank in between.
 *
 * @param {{atom37: Float32Array, atom37Mask: Float32Array}} from
 * @param {{atom37: Float32Array, atom37Mask: Float32Array}} to superposed onto `from`
 * @param {Float32Array} plddtFrom
 * @param {Float32Array} plddtTo
 * @param {number} length residues
 * @param {number} steps frames, both ends included
 * @returns {Array<{structure: object, confidence: {plddt: Float32Array}}>}
 */
export function morphFrames(from, to, plddtFrom, plddtTo, length, steps) {
  if (!Number.isInteger(steps) || steps < 2) {
    throw new RangeError("a morph needs at least its two ends");
  }
  const start = anchoredStart(from, to, length);
  const end = toPoints(to.atom37, length);
  const frames = [];
  for (let step = 0; step < steps; step += 1) {
    // SMOOTHSTEP, not a straight ramp: a morph that starts and stops abruptly
    // reads as a jump cut with extra frames in the middle.
    const linear = step / (steps - 1);
    const t = linear * linear * (3 - 2 * linear);
    const points = start.map((a, i) => [
      a[0] + (end[i][0] - a[0]) * t,
      a[1] + (end[i][1] - a[1]) * t,
      a[2] + (end[i][2] - a[2]) * t,
    ]);
    const plddt = new Float32Array(length);
    const ends = step === 0 ? plddtFrom : (step === steps - 1 ? plddtTo : null);
    if (ends !== null) for (let r = 0; r < length; r += 1) plddt[r] = ends[r];
    frames.push({
      // THE NEW MASK THROUGHOUT, which is what keeps every frame the same
      // length - see anchoredStart.
      structure: { atom37: toAtom37(points), atom37Mask: to.atom37Mask },
      confidence: { plddt },
    });
  }
  return frames;
}

/**
 * Superpose `to` onto `from` on their alpha carbons.
 *
 * py2Dmol's own `addFrame` superposes a frame onto the one before it, but only
 * when the two have the same number of positions - and a mutation makes them
 * differ. `py2Dmol.superpose` fits on the subset and moves everything, which is
 * why this can hand it the CAs and get the whole structure back.
 *
 * @returns {{atom37: Float32Array, atom37Mask: Float32Array}} a new structure
 */
/** How many residues an atom37 array holds: 37 slots of x, y, z apiece. */
function residueCount(structure) {
  return structure.atom37.length / (37 * 3);
}

/**
 * `to`, moved onto `from`.
 *
 * @param {object} pairing [optional] which residue of `to` answers to which of
 *   `from`, as the two index lists web/align.js produces. Only "i with i" while
 *   the sequences are the same length: an insertion or a deletion shifts every
 *   position after it, and fitting index for index across a shift swings the
 *   whole structure round trying to reconcile a tail with the wrong part of the
 *   other one. Absent, the two are assumed to correspond position for position.
 */
export function superposeOnto(api, to, from, length, pairing) {
  const identity = Array.from({ length }, (_, i) => i);
  const caOf = (index) => index * 37 + CA;
  const moved = api.superpose(
    toPoints(to.atom37, length),
    toPoints(from.atom37, residueCount(from)),
    { from: (pairing?.from ?? identity).map(caOf), to: (pairing?.to ?? identity).map(caOf) },
  );
  return { atom37: toAtom37(moved), atom37Mask: to.atom37Mask };
}
