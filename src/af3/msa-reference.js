/**
 * One AF3 MSA block (an "evoformer iteration"), on the CPU.
 *
 * The trunk is two stacks: four of these, then forty-eight pairformer blocks.
 * They share their whole pair half - both triangle multiplications, both
 * triangle attentions, the pair transition - so this file adds only the three
 * operations that touch the MSA, and imports the rest from the block that is
 * already checked exact against AF3.
 *
 * THE BLOCK (config `alphafold3`, no dropout):
 *
 *     pair += outerProductMean(msa)          <- the MSA BEFORE its update
 *     msa  += msaAttention(msa, pair)        <- the pair AFTER the OPM
 *     msa  += transition(msa)
 *     pair += triangleMultiplication(pair, "outgoing")
 *     pair += triangleMultiplication(pair, "incoming")
 *     pair += gridSelfAttention(pair, transpose = false)
 *     pair += gridSelfAttention(pair, transpose = true)
 *     pair += transition(pair)
 *
 * 🔴 THE FIRST TWO LINES CROSS, AND TWO MODELS OF THE LINEAGE SWAP THEM. AF3
 * takes the outer product of the MSA as it arrived and then updates the MSA
 * against the pair that outer product just changed; OpenDDE and Boltz-2 update
 * the MSA first and take the outer product of the result. Same modules, same
 * weights, different function - and the difference compounds over four blocks.
 * Reading either line alone tells you nothing; the order is the content.
 *
 * 🔴 THE MSA ATTENTION HAS NO QUERIES AND NO KEYS. Every other attention here
 * builds its weights from the thing it attends over. This one builds them
 * ENTIRELY from the pair representation - `pair_logits` is the whole score -
 * and the MSA supplies only values and a gate. AF3 calls it pair-weighted
 * averaging, and it is the reason the MSA stack is 3 M parameters against the
 * pairformer's 147 M.
 */
import {
  gridSelfAttention, layerNorm, linear, transition, triangleMultiplication,
} from "./pairformer-reference.js";

const sigmoid = (value) => 1 / (1 + Math.exp(-value));

/** Softmax over the last axis of `rows` rows of `width`, in place. */
function softmaxRows(values, rows, width) {
  for (let row = 0; row < rows; row += 1) {
    const base = row * width;
    let largest = -Infinity;
    for (let i = 0; i < width; i += 1) {
      if (values[base + i] > largest) largest = values[base + i];
    }
    let total = 0;
    for (let i = 0; i < width; i += 1) {
      const value = Math.exp(values[base + i] - largest);
      values[base + i] = value;
      total += value;
    }
    for (let i = 0; i < width; i += 1) values[base + i] /= total;
  }
  return values;
}

/**
 * The outer product mean: the MSA's only route into the pair representation.
 *
 * @param {Float32Array} msa      sequences * tokens * msaChannels
 * @param {Float32Array} msaMask  sequences * tokens
 * @param {number} sequences
 * @param {number} tokens
 * @param {number} msaChannels
 * @param {number} pairChannels
 * @param {object} weights
 * @returns {Float32Array} tokens * tokens * pairChannels
 */
export function outerProductMean(msa, msaMask, sequences, tokens, msaChannels,
                                 pairChannels, weights) {
  const outer = weights.outerChannels;
  const rows = sequences * tokens;
  const normalised = layerNorm(msa, rows, msaChannels, weights.layerNormInputScale,
                               weights.layerNormInputOffset);
  const left = linear(normalised, rows, msaChannels, outer, weights.leftProjection);
  const right = linear(normalised, rows, msaChannels, outer, weights.rightProjection);
  // ...masked AFTER the projection, on both sides. The product is bilinear, so
  // a masked row contributes nothing to either factor.
  for (let row = 0; row < rows; row += 1) {
    const keep = msaMask[row];
    for (let c = 0; c < outer; c += 1) {
      left[row * outer + c] *= keep;
      right[row * outer + c] *= keep;
    }
  }

  const output = new Float32Array(tokens * tokens * pairChannels);
  // The contraction, one token pair at a time: accumulate the outer product
  // over sequences, then map it through output_w.
  const product = new Float32Array(outer * outer);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      product.fill(0);
      let norm = 0;
      for (let s = 0; s < sequences; s += 1) {
        norm += msaMask[s * tokens + i] * msaMask[s * tokens + j];
        const leftBase = (s * tokens + i) * outer;
        const rightBase = (s * tokens + j) * outer;
        for (let c = 0; c < outer; c += 1) {
          const leftValue = left[leftBase + c];
          if (leftValue === 0) continue;
          for (let e = 0; e < outer; e += 1) {
            product[c * outer + e] += leftValue * right[rightBase + e];
          }
        }
      }
      const outputBase = (i * tokens + j) * pairChannels;
      // 🔴 THE MEAN IS TAKEN AFTER THE PROJECTION, and its denominator is
      // 1e-3 + the number of sequences covering BOTH tokens - not the sequence
      // count. Dividing earlier, or by `sequences`, differs on any MSA with a
      // gap and agrees on every toy input that has none.
      const scale = 1 / (1e-3 + norm);
      for (let f = 0; f < pairChannels; f += 1) {
        let total = weights.outputB[f];
        for (let c = 0; c < outer; c += 1) {
          for (let e = 0; e < outer; e += 1) {
            total += product[c * outer + e] * weights.outputW[(c * outer + e) * pairChannels + f];
          }
        }
        output[outputBase + f] = total * scale;
      }
    }
  }
  return output;
}

/**
 * MSA pair-weighted averaging: values from the MSA, weights from the pair.
 *
 * @param {Float32Array} msa      sequences * tokens * msaChannels
 * @param {Float32Array} msaMask  sequences * tokens
 * @param {Float32Array} pair     tokens * tokens * pairChannels
 */
export function msaAttention(msa, msaMask, pair, sequences, tokens, msaChannels,
                             pairChannels, weights) {
  const heads = weights.heads;
  const dimension = weights.dimension;
  const width = heads * dimension;
  const rows = sequences * tokens;
  const normalised = layerNorm(msa, rows, msaChannels, weights.actNormScale,
                               weights.actNormOffset);
  const pairNormalised = layerNorm(pair, tokens * tokens, pairChannels,
                                   weights.pairNormScale, weights.pairNormOffset);
  const flat = linear(pairNormalised, tokens * tokens, pairChannels, heads,
                      weights.pairLogits);

  // ...one mask row for all sequences: AF3 takes the maximum over the MSA
  // depth, so a token any sequence covers is attendable by every sequence.
  const keyMask = new Float32Array(tokens);
  for (let t = 0; t < tokens; t += 1) {
    let largest = 0;
    for (let s = 0; s < sequences; s += 1) {
      if (msaMask[s * tokens + t] > largest) largest = msaMask[s * tokens + t];
    }
    keyMask[t] = largest;
  }

  const weightsByHead = new Float32Array(heads * tokens * tokens);
  for (let head = 0; head < heads; head += 1) {
    for (let i = 0; i < tokens; i += 1) {
      const base = head * tokens * tokens + i * tokens;
      for (let j = 0; j < tokens; j += 1) {
        weightsByHead[base + j] = flat[(i * tokens + j) * heads + head]
          + 1e9 * (keyMask[j] - 1);
      }
      softmaxRows(weightsByHead.subarray(base, base + tokens), 1, tokens);
    }
  }

  const values = linear(normalised, rows, msaChannels, width, weights.vProjection);
  const averaged = new Float32Array(rows * width);
  for (let s = 0; s < sequences; s += 1) {
    for (let i = 0; i < tokens; i += 1) {
      const outputBase = (s * tokens + i) * width;
      for (let head = 0; head < heads; head += 1) {
        const weightBase = head * tokens * tokens + i * tokens;
        for (let d = 0; d < dimension; d += 1) {
          let total = 0;
          for (let j = 0; j < tokens; j += 1) {
            total += weightsByHead[weightBase + j]
              * values[(s * tokens + j) * width + head * dimension + d];
          }
          averaged[outputBase + head * dimension + d] = total;
        }
      }
    }
  }

  const gate = linear(normalised, rows, msaChannels, width, weights.gatingQuery);
  for (let index = 0; index < averaged.length; index += 1) {
    averaged[index] *= sigmoid(gate[index]);
  }
  return linear(averaged, rows, width, msaChannels, weights.outputProjection);
}

/**
 * One MSA block: msa and pair in, msa and pair out.
 *
 * @param {{msa: Float32Array, pair: Float32Array, msaMask: Float32Array,
 *          pairMask: Float32Array, sequences: number, tokens: number}} state
 * @param {object} weights
 * @param {{swapTransposedBias: boolean}} dialect
 */
export function msaBlock(state, weights, dialect) {
  const { msaMask, pairMask, sequences, tokens } = state;
  const pairChannels = weights.pairChannels;
  const msaChannels = weights.msaChannels;
  const rows = sequences * tokens;
  let pair = Float32Array.from(state.pair);
  let msa = Float32Array.from(state.msa);

  const addPair = (delta) => {
    for (let index = 0; index < pair.length; index += 1) pair[index] += delta[index];
  };
  const addMsa = (delta) => {
    for (let index = 0; index < msa.length; index += 1) msa[index] += delta[index];
  };

  // ...the outer product of the MSA AS IT ARRIVED, before the update below.
  addPair(outerProductMean(msa, msaMask, sequences, tokens, msaChannels, pairChannels,
                           weights.outerProductMean));
  // ...and then the MSA update, against the pair the outer product just changed.
  addMsa(msaAttention(msa, msaMask, pair, sequences, tokens, msaChannels, pairChannels,
                      weights.msaAttention1));
  addMsa(transition(msa, rows, msaChannels, weights.msaTransition));

  addPair(triangleMultiplication(pair, pairMask, tokens, pairChannels, "outgoing",
                                 weights.triangleMultiplicationOutgoing));
  addPair(triangleMultiplication(pair, pairMask, tokens, pairChannels, "incoming",
                                 weights.triangleMultiplicationIncoming));
  addPair(gridSelfAttention(pair, pairMask, tokens, pairChannels, false,
                            weights.pairAttention1, dialect));
  addPair(gridSelfAttention(pair, pairMask, tokens, pairChannels, true,
                            weights.pairAttention2, dialect));
  addPair(transition(pair, tokens * tokens, pairChannels, weights.pairTransition));

  return { msa, pair };
}
