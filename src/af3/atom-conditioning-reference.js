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
import { linear } from "./pairformer-reference.js";

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
export function perAtomConditioning(reference, tokens, dense, weights) {
  const channels = weights.channels;
  const rows = tokens * dense;

  const act = linear(reference.positions, rows, 3, channels, weights.embedRefPos);

  const add = (contribution) => {
    for (let index = 0; index < act.length; index += 1) act[index] += contribution[index];
  };

  const maskColumn = new Float32Array(rows);
  for (let index = 0; index < rows; index += 1) maskColumn[index] = reference.mask[index];
  add(linear(maskColumn, rows, 1, channels, weights.embedRefMask));

  // ...the element as a one-hot over the periodic table, which is why the
  // weight is 128 rows: an atomic number indexes it directly.
  //
  // 🔴 AND "INDEXES IT DIRECTLY" IS THE IMPLEMENTATION, NOT JUST THE READING.
  // Building the one-hot and multiplying by it is rows * 128 * channels
  // multiply-adds of which all but rows * channels are by zero. Adding the row
  // the one-hot selects is the SAME FLOAT: `linear` accumulates from zero over
  // ascending columns, so the sum is a run of exact zeros around one term.
  // This and the atom name below are 384 of this function's 389 input columns
  // and were 99% of its arithmetic - 72 ms at 59 tokens, 220 at 150, 343 at
  // 300, once per fold, all of it spent multiplying by zero.
  for (let index = 0; index < rows; index += 1) {
    const atomicNumber = reference.element[index];
    if (atomicNumber < 0 || atomicNumber >= 128) continue;
    const weightBase = atomicNumber * channels;
    const actBase = index * channels;
    for (let c = 0; c < channels; c += 1) {
      act[actBase + c] += weights.embedRefElement[weightBase + c];
    }
  }

  // 🔴 arcsinh, NOT the charge. See the note at the top: identical at zero, so
  // no protein-only check can catch this.
  const charge = new Float32Array(rows);
  for (let index = 0; index < rows; index += 1) {
    charge[index] = Math.asinh(reference.charge[index]);
  }
  add(linear(charge, rows, 1, channels, weights.embedRefCharge));

  // ...the atom's NAME, as four characters, each a 64-way one-hot of its ASCII
  // code minus 32, flattened to 256 columns. "CA" is padded, so the trailing
  // slots are the one-hot of character 0 rather than nothing at all.
  //
  // Four one-hots, so four gathered rows summed - and summed in CHARACTER
  // order, which is ascending column order, because that is the order the
  // matmul added them in and floating-point addition does not commute.
  //
  // 🔴 THE FOUR ROWS ARE SUMMED BEFORE THEY REACH `act`, AND THAT IS NOT
  // TIDINESS. The matmul form accumulated a row's whole dot product from zero and
  // `add` then committed it in one step, so `act + (a+b+c+d)` is what it
  // computed; adding the four to `act` in turn is `(((act+a)+b)+c)+d`, which
  // floating-point addition does not promise is the same number. Measured
  // across four shapes it differed in 169,390 of 696,512 floats by up to
  // 4.8e-7 - physically nothing, but this file is the reference the GPU
  // kernels are checked against, and a reference that moves under them is a
  // tolerance nobody chose. Summed first, every float is identical.
  // 🔴 AND IT IS A Float64Array, WHICH IS THE SECOND HALF OF THE SAME POINT.
  // `linear` accumulates a dot product in a JS number - float64 - and rounds
  // ONCE, when it stores the result. A Float32Array scratch rounds after every
  // addition instead, and that alone left 12,641 floats differing by up to
  // 1.2e-7. Accumulate wide, round where the matmul rounded.
  const nameSum = new Float64Array(channels);
  for (let index = 0; index < rows; index += 1) {
    nameSum.fill(0);
    for (let character = 0; character < 4; character += 1) {
      const code = reference.atomNameChars[index * 4 + character];
      if (code < 0 || code >= 64) continue;
      const weightBase = (character * 64 + code) * channels;
      for (let c = 0; c < channels; c += 1) {
        nameSum[c] += weights.embedRefAtomName[weightBase + c];
      }
    }
    const actBase = index * channels;
    for (let c = 0; c < channels; c += 1) act[actBase + c] += Math.fround(nameSum[c]);
  }

  // ...and masked last, so an absent atom contributes nothing downstream even
  // though four of the five embeddings above are non-zero for it.
  for (let index = 0; index < rows; index += 1) {
    const keep = reference.mask[index];
    for (let c = 0; c < channels; c += 1) act[index * channels + c] *= keep;
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
