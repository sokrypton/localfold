/**
 * The six template GEOMETRY features, from a template's dense atoms.
 *
 * These are what `src/af3/template-reference.js` used to refuse to compute, and
 * the reason it refused was good: with no template all six are identically
 * zero, so nothing here could tell a correct implementation from a wrong one.
 * `tools/oracle/dump_af3_trunk.py --template <pdb>` now produces a reference
 * with a real one in it, which is what makes this file writable.
 *
 * They are separate from the embedder because they are ARITHMETIC OVER
 * COORDINATES and nothing else - no weights, no channels, no layer norm - so
 * they can be checked against the oracle's captured template arrays directly,
 * before any of the embedding is involved. A wrong unit vector and a wrong
 * projection look identical once they have been summed into 64 channels.
 *
 * Transcribed from `construct_input` in
 * alphafold3/src/alphafold3/model/network/template_modules.py, whose feature
 * ORDER is the index in `template_pair_embedding_{i}`:
 *
 *     0  distogram of pseudo-beta positions, 39 bins, x the mask
 *     1  pseudo-beta mask, 2D
 *     2  template aatype along j        (in the embedder, not here)
 *     3  template aatype along i        (in the embedder, not here)
 *     4  unit vector x, x the backbone mask
 *     5  unit vector y
 *     6  unit vector z
 *     7  backbone mask, 2D
 *     8  the query pair, layer-normed   (in the embedder, not here)
 */

/** Dense atom slots per residue in AF3's protein layout. */
export const NUM_DENSE = 24;

/** Distogram bins, from AF3's DistogramFeaturesConfig. */
export const DGRAM_BINS = 39;
export const DGRAM_MIN = 3.25;
export const DGRAM_MAX = 50.75;

/**
 * Which dense slot carries each residue type's pseudo-beta atom.
 *
 * 🔴 CB FOR EVERYTHING BUT GLYCINE, WHICH HAS NONE AND TAKES CA. That is the
 * whole of the protein half of this table - slot 4 is CB, slot 1 is CA, and
 * aatype 7 is glycine - and getting it wrong puts one residue in twenty a
 * bond-length away from where AF3 puts it, which is a distogram bin at short
 * range and nothing at long range. AF3's own
 * `protein_data_processing.RESTYPE_PSEUDOBETA_INDEX`, in the residue order
 * ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR
 * VAL UNK - A G C U DA DG DC DT N.
 */
export const PSEUDO_BETA_SLOT = [
  4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  0, 0, 22, 23, 14, 14, 21, 22, 13, 13, 0,
];

/**
 * The three dense slots forming each residue type's backbone frame, as
 * (C, CA, N).
 *
 * 🔴 (C, CA, N) AND NOT (N, CA, C), which is the order the same three atoms
 * take everywhere else in this repository. AF3's comment says why: the
 * side-chain frame convention is (C, CA, N) and the backbone one is (N, CA, C),
 * and `make_backbone_rigid` reads the SIDE-CHAIN table's group 0 - so it names
 * its locals `c, b, a` to undo the swap. Reading this table as (N, CA, C)
 * gives a frame reflected through its own CA, and a unit vector that is
 * plausible, smoothly varying, and mirrored.
 */
export const BACKBONE_SLOTS = [
  [2, 1, 0], [2, 1, 0], [2, 1, 0], [2, 1, 0], [2, 1, 0], [2, 1, 0], [2, 1, 0],
  [2, 1, 0], [2, 1, 0], [2, 1, 0], [2, 1, 0], [2, 1, 0], [2, 1, 0], [2, 1, 0],
  [2, 1, 0], [2, 1, 0], [2, 1, 0], [2, 1, 0], [2, 1, 0], [2, 1, 0],
  [0, 0, 0], [0, 0, 0],
  [12, 8, 6], [12, 8, 6], [12, 8, 6], [12, 8, 6],
  [12, 8, 6], [12, 8, 6], [12, 8, 6], [12, 8, 6],
  [0, 0, 0],
];

const slotOf = (table, code) => (code >= 0 && code < table.length ? table[code] : table[20]);

/**
 * Pseudo-beta positions and their mask.
 *
 * @param {ArrayLike<number>} aatype [tokens]
 * @param {ArrayLike<number>} positions [tokens, 24, 3], already masked
 * @param {ArrayLike<number>} mask [tokens, 24]
 * @param {number} tokens
 * @returns {{positions: Float32Array, mask: Float32Array}} [tokens, 3], [tokens]
 */
export function pseudoBeta(aatype, positions, mask, tokens) {
  const out = new Float32Array(tokens * 3);
  const present = new Float32Array(tokens);
  for (let token = 0; token < tokens; token += 1) {
    const slot = slotOf(PSEUDO_BETA_SLOT, aatype[token]);
    const base = (token * NUM_DENSE + slot) * 3;
    out[token * 3] = positions[base];
    out[token * 3 + 1] = positions[base + 1];
    out[token * 3 + 2] = positions[base + 2];
    present[token] = mask[token * NUM_DENSE + slot] > 0 ? 1 : 0;
  }
  return { positions: out, mask: present };
}

/**
 * The 39-bin distogram of a set of positions.
 *
 * 🔴 A PAIR CLOSER THAN 3.25 A IS IN NO BIN AT ALL. AF3 builds this as
 * `(d2 > lower) * (d2 < upper)` over `lower = linspace(3.25, 50.75, 39)^2`, so
 * a distance below the first break fails the first test and every other one -
 * the row is all zero rather than falling into bin 0. That is not the usual
 * clamped bucketisation and writing the usual one puts every close contact,
 * which is every pair the model cares most about, one bin too high.
 *
 * The last bin catches everything past 50.75 A, because its upper break is
 * 1e8 rather than the next value.
 *
 * @param {ArrayLike<number>} positions [tokens, 3]
 * @param {number} tokens
 * @returns {Uint8Array} [tokens, tokens, 39], one-hot
 */
export function distogram(positions, tokens) {
  const lower = new Float64Array(DGRAM_BINS);
  for (let bin = 0; bin < DGRAM_BINS; bin += 1) {
    const edge = DGRAM_MIN + (DGRAM_MAX - DGRAM_MIN) * (bin / (DGRAM_BINS - 1));
    lower[bin] = edge * edge;
  }
  const out = new Uint8Array(tokens * tokens * DGRAM_BINS);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      const dx = positions[i * 3] - positions[j * 3];
      const dy = positions[i * 3 + 1] - positions[j * 3 + 1];
      const dz = positions[i * 3 + 2] - positions[j * 3 + 2];
      const d2 = dx * dx + dy * dy + dz * dz;
      const base = (i * tokens + j) * DGRAM_BINS;
      for (let bin = 0; bin < DGRAM_BINS; bin += 1) {
        const upper = bin + 1 < DGRAM_BINS ? lower[bin + 1] : 1e8;
        if (d2 > lower[bin] && d2 < upper) { out[base + bin] = 1; break; }
      }
    }
  }
  return out;
}

/**
 * Each residue's backbone frame, and which residues have one.
 *
 * The rotation is Gram-Schmidt on (C - CA) and (N - CA), which is AF3's
 * `Rot3Array.from_two_vectors(frame[2] - frame[1], frame[0] - frame[1])` with
 * its `c, b, a` naming resolved: e1 along the first vector, e2 the part of the
 * second orthogonal to it, e3 their cross product.
 *
 * @returns {{rotations: Float32Array, translations: Float32Array,
 *            mask: Float32Array}} [tokens, 9] row-major, [tokens, 3], [tokens]
 */
export function backboneFrames(aatype, positions, mask, tokens) {
  const rotations = new Float32Array(tokens * 9);
  const translations = new Float32Array(tokens * 3);
  const present = new Float32Array(tokens);
  const at = (token, slot, axis) => positions[(token * NUM_DENSE + slot) * 3 + axis];
  for (let token = 0; token < tokens; token += 1) {
    const [c, b, a] = slotOf(BACKBONE_SLOTS, aatype[token]);
    present[token] = (mask[token * NUM_DENSE + a] > 0 && mask[token * NUM_DENSE + b] > 0
      && mask[token * NUM_DENSE + c] > 0) ? 1 : 0;
    for (let axis = 0; axis < 3; axis += 1) translations[token * 3 + axis] = at(token, b, axis);

    // e1 along C - CA; e2 the part of N - CA orthogonal to it; e3 = e1 x e2.
    const e1 = [at(token, c, 0) - at(token, b, 0), at(token, c, 1) - at(token, b, 1),
                at(token, c, 2) - at(token, b, 2)];
    const v2 = [at(token, a, 0) - at(token, b, 0), at(token, a, 1) - at(token, b, 1),
                at(token, a, 2) - at(token, b, 2)];
    const n1 = Math.hypot(e1[0], e1[1], e1[2]) || 1;
    for (let axis = 0; axis < 3; axis += 1) e1[axis] /= n1;
    const dot = e1[0] * v2[0] + e1[1] * v2[1] + e1[2] * v2[2];
    const e2 = [v2[0] - dot * e1[0], v2[1] - dot * e1[1], v2[2] - dot * e1[2]];
    const n2 = Math.hypot(e2[0], e2[1], e2[2]) || 1;
    for (let axis = 0; axis < 3; axis += 1) e2[axis] /= n2;
    const e3 = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0]];
    // 🔴 THE COLUMNS ARE THE AXES, so the matrix maps frame coordinates to
    // world ones and its TRANSPOSE is what takes a world point into the frame -
    // which is the direction the unit vector needs. Storing it the other way
    // round gives a rotation that is still orthonormal and still smooth.
    const row = token * 9;
    for (let axis = 0; axis < 3; axis += 1) {
      rotations[row + axis * 3] = e1[axis];
      rotations[row + axis * 3 + 1] = e2[axis];
      rotations[row + axis * 3 + 2] = e3[axis];
    }
  }
  return { rotations, translations, mask: present };
}

/**
 * All six geometry features for one template slot.
 *
 * 🔴 THE UNIT VECTOR IS `R_i^-1 (t_j - t_i)`: "where is j, seen from i". The
 * FRAME belongs to the row index and the POINT to the column, which is the one
 * thing here that cannot be read off the source without care - AF3 writes it
 * `rigid[:, None].inverse().apply_to_point(points)`, and it is the broadcast
 * that decides: `rigid[:, None]` is (N, 1) and `points` is (N,), so axis 0
 * comes from the frames and axis 1 from the points.
 *
 * Written the other way round it is the exact TRANSPOSE - still unit length,
 * still smooth, still masked correctly, and wrong. Measured against AF3's own
 * `make_backbone_rigid` on a 16-residue template: the transposed reading
 * scores relRMS 1.5 against the reference and the correct one 1.3e-7, while
 * the distogram and both masks are bit-exact either way. That is the whole
 * error, and it is invisible to every check that does not have a real
 * template in it - which is every check this repository had.
 *
 * 🔴 AND EVERY MASK IS MULTIPLIED BY multichainMask2d. A template covers ONE
 * chain, so a distance between two chains of the query is not something it
 * knows - and left unmasked it is a real number computed from two structures
 * that were never in the same frame.
 *
 * @param {object} template `aatype` [tokens], `atomPositions` [tokens, 24, 3],
 *   `atomMask` [tokens, 24]
 * @param {ArrayLike<number>} multichainMask2d [tokens, tokens]
 * @param {number} tokens
 * @returns {{distogram: Uint8Array, pseudoBetaMask2d: Float32Array,
 *            unitVector: Float32Array, backboneMask2d: Float32Array}}
 *   `unitVector` is [tokens, tokens, 3], already masked.
 */
export function templateGeometry(template, multichainMask2d, tokens) {
  const { aatype, atomMask } = template;
  // 🔴 THE POSITIONS ARE MASKED FIRST, which AF3 does as
  // `dense_atom_positions *= dense_atom_mask[..., None]` before anything reads
  // them. An unresolved atom's coordinates are whatever the featuriser left
  // there, and they reach the distogram through the pseudo-beta gather even
  // though its mask is zero - the mask only zeroes the OUTPUT, and a bin index
  // computed from a stale coordinate is still a bin index.
  const positions = new Float32Array(tokens * NUM_DENSE * 3);
  for (let slot = 0; slot < tokens * NUM_DENSE; slot += 1) {
    const keep = atomMask[slot] > 0 ? 1 : 0;
    positions[slot * 3] = template.atomPositions[slot * 3] * keep;
    positions[slot * 3 + 1] = template.atomPositions[slot * 3 + 1] * keep;
    positions[slot * 3 + 2] = template.atomPositions[slot * 3 + 2] * keep;
  }

  const beta = pseudoBeta(aatype, positions, atomMask, tokens);
  const dgram = distogram(beta.positions, tokens);
  const frames = backboneFrames(aatype, positions, atomMask, tokens);

  const pseudoBetaMask2d = new Float32Array(tokens * tokens);
  const backboneMask2d = new Float32Array(tokens * tokens);
  const unitVector = new Float32Array(tokens * tokens * 3);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      const pair = i * tokens + j;
      const chain = multichainMask2d[pair];
      pseudoBetaMask2d[pair] = beta.mask[i] * beta.mask[j] * chain;
      const backbone = frames.mask[i] * frames.mask[j] * chain;
      backboneMask2d[pair] = backbone;

      // ...j's translation, expressed in i's frame. See the note above.
      const dx = frames.translations[j * 3] - frames.translations[i * 3];
      const dy = frames.translations[j * 3 + 1] - frames.translations[i * 3 + 1];
      const dz = frames.translations[j * 3 + 2] - frames.translations[i * 3 + 2];
      const row = i * 9;
      const x = frames.rotations[row] * dx + frames.rotations[row + 3] * dy
        + frames.rotations[row + 6] * dz;
      const y = frames.rotations[row + 1] * dx + frames.rotations[row + 4] * dy
        + frames.rotations[row + 7] * dz;
      const z = frames.rotations[row + 2] * dx + frames.rotations[row + 5] * dy
        + frames.rotations[row + 8] * dz;
      // 🔴 NORMALISED WITH AN EPSILON, and the diagonal is why: i == j is the
      // zero vector, whose norm is zero, and AF3's `Vec3Array.normalized()`
      // divides by `max(norm, 1e-6)` rather than by the norm. Without that the
      // whole diagonal is NaN and the NaN reaches the pair representation
      // through a weight, which turns the entire fold into NaN two blocks
      // later with nothing pointing back here.
      const norm = Math.max(Math.sqrt(x * x + y * y + z * z), 1e-6);
      unitVector[pair * 3] = (x / norm) * backbone;
      unitVector[pair * 3 + 1] = (y / norm) * backbone;
      unitVector[pair * 3 + 2] = (z / norm) * backbone;
    }
  }

  // The distogram is masked last, exactly as AF3 does: `dgram *= mask[..., None]`.
  const masked = new Float32Array(tokens * tokens * DGRAM_BINS);
  for (let pair = 0; pair < tokens * tokens; pair += 1) {
    if (pseudoBetaMask2d[pair] === 0) continue;
    for (let bin = 0; bin < DGRAM_BINS; bin += 1) {
      masked[pair * DGRAM_BINS + bin] = dgram[pair * DGRAM_BINS + bin];
    }
  }
  return { distogram: masked, pseudoBetaMask2d, unitVector, backboneMask2d };
}

/**
 * Which pairs a template slot is allowed to speak about.
 *
 * 🔴 AF3 MASKS ACROSS CHAINS, AND THE REASON IS PROVENANCE, NOT MODELLING. Its
 * `Template` is documented as one protein chain, and a complex's chains are
 * templated by SEPARATE searches - so the coordinates for chain A and chain B
 * were never in one frame and a distance between them is a number computed
 * from two unrelated structures. Masking is the only correct thing to do with
 * it.
 *
 * 🔴 BUT IT IS NOT ALWAYS TRUE, AND THAT IS WORTH UNLOCKING. When both chains
 * come from ONE file - a real co-crystal, or a complex this page predicted -
 * they ARE in one frame, and the cross-chain distances are exactly the
 * interface geometry a binder method wants. So spanning is a property of where
 * the coordinates came from, which is why it is a flag on the SLOT rather than
 * a setting on the model: a slot built from one structure may span, and two
 * slots built from two structures may not, in the same fold.
 *
 * Measured on a two-chain query with a template on each chain: masking as AF3
 * does reproduces it to relRMS 5.5e-7, and leaving the cross-chain pairs open
 * scores 1.09 - so this is most of the module's answer, not a correction.
 *
 * @param {ArrayLike<number>} asymId [tokens], one id per chain
 * @param {number} tokens
 * @param {{coverage?: ArrayLike<number>, spanChains?: boolean}} [options]
 *   `coverage` is 1 where THIS slot has a residue; spanning opens only the
 *   pairs it covers at both ends, because a pair with one end outside the
 *   template is still two frames apart.
 * @returns {Float32Array} [tokens, tokens]
 */
export function multichainMaskFor(asymId, tokens, options = {}) {
  const { coverage, spanChains = false } = options;
  const mask = new Float32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      const same = asymId[i] === asymId[j];
      const spanned = spanChains && coverage !== undefined
        && coverage[i] > 0 && coverage[j] > 0;
      mask[i * tokens + j] = same || spanned ? 1 : 0;
    }
  }
  return mask;
}

/**
 * Which query tokens a slot actually has a residue at.
 *
 * Derived from the atom mask rather than carried alongside it, because two
 * statements of the same thing drift and this one cannot: a token with no
 * atoms contributes nothing to any feature whatever a coverage array says.
 *
 * @param {{atomMask: ArrayLike<number>}} template
 * @param {number} tokens
 * @returns {Float32Array}
 */
export function coverageOf(template, tokens) {
  const covered = new Float32Array(tokens);
  for (let token = 0; token < tokens; token += 1) {
    for (let slot = 0; slot < NUM_DENSE; slot += 1) {
      if (template.atomMask[token * NUM_DENSE + slot] > 0) { covered[token] = 1; break; }
    }
  }
  return covered;
}
