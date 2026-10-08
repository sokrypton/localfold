/**
 * AF3's per-atom conditioning: chemistry in, 128 channels per atom out.
 *
 * This is the bottom of the model. Everything else in the trunk reads token
 * representations; this reads a REFERENCE CONFORMER - where each atom of each
 * residue sits in an idealised copy of it, what element it is, what charge it
 * carries and what it is called - and turns that into the per-atom features the
 * atom transformer runs on. Its output is where 384 of `target_feat`'s 447
 * columns come from.
 *
 * THE LAYOUT IS DENSE PER TOKEN, not a flat atom list: (num_tokens, 24, ...)
 * for a protein, where 24 is the widest residue and the unused slots are masked
 * off. That is what makes this tractable in a browser - the whole thing is a
 * small table lookup per residue plus five matrix multiplies.
 *
 * 🔴 FIVE EMBEDDINGS ARE SUMMED AND ALL FIVE ARE BIAS-FREE, so a missing term
 * is not a constant offset that a later LayerNorm absorbs - it is a direction
 * removed from every atom. The one most likely to be got wrong silently is the
 * charge: AF3 feeds arcsinh(charge), not the charge, and the two are IDENTICAL
 * AT ZERO. Every standard amino acid conformer in the CCD is neutral, so a test
 * on protein alone can never tell them apart; it first matters on a ligand.
 */
import { linear } from "../trunk/pairformer-reference.js";

/**
 * @param {{positions: Float32Array, mask: Float32Array, element: ArrayLike<number>,
 *          charge: Float32Array, atomNameChars: ArrayLike<number>}} reference
 *   positions   tokens * dense * 3
 *   mask        tokens * dense
 *   element     tokens * dense          atomic number, 0..127
 *   charge      tokens * dense
 *   atomNameChars tokens * dense * 4    ASCII minus 32, 0..63
 * @param {number} tokens
 * @param {number} dense    atom slots per token (24 for a protein)
 * @param {object} weights
 * @returns {Float32Array} tokens * dense * channels
 */
export function perAtomConditioning(reference, tokens, dense, weights, dialect) {
  const channels = weights.channels;
  const rows = tokens * dense;
  // 🔴 ONE PASS A ROW, THE SAME FLOATS. This was seven passes over
  // `rows x channels` - a `linear` for the position, the mask and the charge,
  // each allocating its own array, and an `add` for each - which is 46 ms of
  // host time a fold at 255 tokens, run twice (the target-feat encoder and the
  // diffusion head). Every element below sees the same operations in the same
  // order and is rounded to f32 at the same points the separate passes rounded
  // it, so the result is bit-identical:
  //
  //   - the position is `linear`'s float64 dot product from zero over
  //     ascending columns, rounded once;
  //   - boltz2's single bias over the whole concatenation, added once;
  //   - the mask and the charge are one-column `linear`s, so each is
  //     `fround(0 + x * w)` added to the running value;
  //   - the element is a one-hot over the periodic table, so an atomic number
  //     indexes its row directly (adding the selected row IS the matmul's float:
  //     every other term is an exact zero);
  //   - 🔴 arcsinh of the charge, NOT the charge, for AlphaFold 3 - identical at
  //     zero, so no protein-only check can see it - and the raw charge for
  //     RAW_REF_CHARGE dialects (boltz2, chai1, rosettafold3, ESMFold2);
  //   - the atom NAME, four characters each a 64-way one-hot of ASCII minus 32,
  //     summed in CHARACTER order in a float64 and rounded once BEFORE it reaches
  //     the running value, because the matmul committed `act + (a+b+c+d)` and
  //     `(((act+a)+b)+c)+d` differed in 169,390 of 696,512 floats;
  //   - and masked last, so an absent atom contributes nothing downstream even
  //     though four of the five embeddings above are non-zero for it.
  const fround = Math.fround;
  const raw = dialect?.rawRefCharge === true;
  const pos = weights.embedRefPos;
  const maskWeight = weights.embedRefMask;
  const chargeWeight = weights.embedRefCharge;
  const elementWeight = weights.embedRefElement;
  const nameWeight = weights.embedRefAtomName;
  const bias = weights.embedAtomFeaturesBias ?? null;
  const act = new Float32Array(rows * channels);
  const nameSum = new Float64Array(channels);
  for (let index = 0; index < rows; index += 1) {
    const p0 = reference.positions[index * 3];
    const p1 = reference.positions[index * 3 + 1];
    const p2 = reference.positions[index * 3 + 2];
    const mask = fround(reference.mask[index]);
    const charge = fround(raw ? reference.charge[index] : Math.asinh(reference.charge[index]));
    const atomicNumber = reference.element[index];
    const elementBase = atomicNumber >= 0 && atomicNumber < 128 ? atomicNumber * channels : -1;
    nameSum.fill(0);
    for (let character = 0; character < 4; character += 1) {
      const code = reference.atomNameChars[index * 4 + character];
      if (code < 0 || code >= 64) continue;
      const weightBase = (character * 64 + code) * channels;
      for (let c = 0; c < channels; c += 1) nameSum[c] += nameWeight[weightBase + c];
    }
    const keep = reference.mask[index];
    const actBase = index * channels;
    for (let c = 0; c < channels; c += 1) {
      let value = fround(((0 + p0 * pos[c]) + p1 * pos[channels + c]) + p2 * pos[2 * channels + c]);
      if (bias !== null) value = fround(value + bias[c]);
      value = fround(value + fround(0 + mask * maskWeight[c]));
      if (elementBase >= 0) value = fround(value + elementWeight[elementBase + c]);
      value = fround(value + fround(0 + charge * chargeWeight[c]));
      value = fround(value + fround(nameSum[c]));
      act[actBase + c] = value * keep;
    }
  }
  return act;
}

/**
 * The per-atom PAIR conditioning, within each token's own atoms.
 *
 * @param {Float32Array} act  the single conditioning, tokens * dense * channels
 * @returns {Float32Array} tokens * dense * dense * pairChannels
 */
export function perAtomPairConditioning(reference, act, tokens, dense, weights) {
  const channels = weights.channels;
  const pairChannels = weights.pairChannels;
  const rows = tokens * dense;

  // ...through a relu first, which is not a detail: the single conditioning is
  // signed and roughly centred, so the relu discards about half of it.
  const rectified = new Float32Array(act.length);
  for (let index = 0; index < act.length; index += 1) {
    rectified[index] = act[index] > 0 ? act[index] : 0;
  }
  const row = linear(rectified, rows, channels, pairChannels, weights.singleToPairCondRow);
  const column = linear(rectified, rows, channels, pairChannels, weights.singleToPairCondCol);

  const output = new Float32Array(tokens * dense * dense * pairChannels);
  const offsets = new Float32Array(3);
  for (let token = 0; token < tokens; token += 1) {
    for (let a = 0; a < dense; a += 1) {
      for (let b = 0; b < dense; b += 1) {
        const base = ((token * dense + a) * dense + b) * pairChannels;
        const rowBase = (token * dense + a) * pairChannels;
        const columnBase = (token * dense + b) * pairChannels;
        for (let c = 0; c < pairChannels; c += 1) {
          output[base + c] = row[rowBase + c] + column[columnBase + c];
        }
        let squared = 0;
        for (let axis = 0; axis < 3; axis += 1) {
          const difference = reference.positions[(token * dense + a) * 3 + axis]
            - reference.positions[(token * dense + b) * 3 + axis];
          offsets[axis] = difference;
          squared += difference * difference;
        }
        for (let c = 0; c < pairChannels; c += 1) {
          let total = 0;
          for (let axis = 0; axis < 3; axis += 1) {
            total += offsets[axis] * weights.embedPairOffsets[axis * pairChannels + c];
          }
          // ...INVERSE squared distance, so nearby atoms give a large feature
          // and distant ones tend to zero rather than growing without bound.
          output[base + c] += total
            + weights.embedPairDistances[c] / (1 + squared);
        }
      }
    }
  }
  return output;
}
