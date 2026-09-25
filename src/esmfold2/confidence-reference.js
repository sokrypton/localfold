/**
 * Synthyra's ESMFold2 confidence head, on the host.
 *
 * 🔴 THIS CHECKPOINT DID NOT HAVE ONE. biohub ships ESMFold2 with
 * `confidence_head.enabled: false` and zero confidence tensors, which is why
 * `src/esmfold2/aligned-error.js` exists at all - a pAE derived from the
 * distogram's certainty, measured and withheld in docs/EF2FAST.md. Synthyra
 * froze that trunk and trained a head on it; the trunk underneath is ours byte
 * for byte, so this module is the whole of the difference.
 *
 * 🔴 AND ITS FOUR BLOCKS ARE THE TRUNK'S BLOCK, PAIR-ONLY. No single track
 * crosses them - `singleFromPair` makes one at the END out of the finished
 * pair - so a port that threaded a single through would be a different head
 * that runs. The block is composed here the way tools/check-esmfold2-trunk.js
 * composes the trunk's, for the reason that file gives: `pairformerBlock`
 * always runs a single attention and a single transition, and feeding it a
 * synthesised zero single checks arithmetic ESMFold2 never does.
 *
 * 🔴 AND `pair + foldingTrunk(pair)` ADDS THE INPUT TWICE, WHICH IS THEIR LINE
 * AND NOT A SLIP. Their `FoldingTrunk.forward` returns the UPDATED pair - each
 * block's residuals are already inside it - and the head then writes
 * `pair = pair + self.folding_trunk(pair, ...)`. So the stack's input appears
 * once through the blocks' own residual chain and once more on top. A port
 * that read that line as an ordinary residual is short exactly one copy of the
 * input, which is a PAE that looks entirely reasonable and is wrong.
 */
import { layerNorm, linear, transition, triangleMultiplication }
  from "../af3/trunk/pairformer-reference.js";

/**
 * The expectation of a categorical distribution over evenly spaced bins.
 *
 * Their `_categorical_mean`: `linspace(start, end, bins + 1)` and the MIDPOINTS
 * of those edges, so 64 bins over [0, 32] are centred 0.25 to 31.75 and 50 over
 * [0, 1] are centred 0.01 to 0.99. pLDDT comes out on [0, 1] here; the page's
 * 0-100 is the caller's multiply.
 */
export function categoricalMean(logits, rows, bins, start, end) {
  const width = (end - start) / bins;
  const out = new Float32Array(rows);
  for (let row = 0; row < rows; row += 1) {
    const at = row * bins;
    let top = -Infinity;
    for (let bin = 0; bin < bins; bin += 1) top = Math.max(top, logits[at + bin]);
    let total = 0;
    let sum = 0;
    for (let bin = 0; bin < bins; bin += 1) {
      const weight = Math.exp(logits[at + bin] - top);
      total += weight;
      sum += weight * (start + width * (bin + 0.5));
    }
    out[row] = sum / total;
  }
  return out;
}

/** Row-attention pooling: a scalar per pair, a masked softmax along j, then 256 -> 384. */
export function singleFromPair(pair, tokenMask, tokens, pairChannels, singleChannels, weights) {
  const scores = linear(pair, tokens * tokens, pairChannels, 1, weights.poolingAttention);
  const pooled = new Float32Array(tokens * pairChannels);
  for (let i = 0; i < tokens; i += 1) {
    let top = -Infinity;
    for (let j = 0; j < tokens; j += 1) {
      // 🔴 THE MASK IS A BIAS OF -1e9 AND NOT A ZEROED WEIGHT: an all-masked row
      // would divide by zero the other way, and their softmax is over the
      // biased scores so a masked column still contributes exp(-1e9) = 0.
      const value = tokenMask[j] > 0.5 ? scores[i * tokens + j] : -1e9;
      if (value > top) top = value;
    }
    let total = 0;
    for (let j = 0; j < tokens; j += 1) {
      const value = tokenMask[j] > 0.5 ? scores[i * tokens + j] : -1e9;
      total += Math.exp(value - top);
    }
    for (let j = 0; j < tokens; j += 1) {
      const value = tokenMask[j] > 0.5 ? scores[i * tokens + j] : -1e9;
      const weight = Math.exp(value - top) / total;
      if (weight === 0) continue;
      const from = (i * tokens + j) * pairChannels;
      const into = i * pairChannels;
      for (let c = 0; c < pairChannels; c += 1) pooled[into + c] += weight * pair[from + c];
    }
  }
  return linear(pooled, tokens, pairChannels, singleChannels, weights.poolingOutput);
}

/**
 * The head, input to output.
 *
 * @param {{tokens: number, atoms: number, sInputs: Float32Array,
 *          pair: Float32Array, coordinates: Float32Array, repAtom: Int32Array,
 *          atomToToken: Int32Array, tokenMask: Float32Array,
 *          atomMask: Float32Array}} inputs
 * @param {object} weights  from `confidenceHeadWeights`
 */
export function esmfold2Confidence(inputs, weights) {
  const { tokens, atoms } = inputs;
  const dPair = weights.pairChannels;
  const dSingle = weights.singleChannels;
  const dInputs = weights.singleInputs;

  const single = layerNorm(inputs.sInputs, tokens, dInputs,
                           weights.sInputsNormScale, weights.sInputsNormOffset);
  let pair = layerNorm(inputs.pair, tokens * tokens, dPair,
                       weights.zNormScale, weights.zNormOffset);

  const rows = linear(single, tokens, dInputs, dPair, weights.sToZ);
  const columns = linear(single, tokens, dInputs, dPair, weights.sToZTranspose);
  const left = linear(single, tokens, dInputs, dPair, weights.sToZProdIn1);
  const right = linear(single, tokens, dInputs, dPair, weights.sToZProdIn2);
  const product = new Float32Array(tokens * tokens * dPair);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      const into = (i * tokens + j) * dPair;
      for (let c = 0; c < dPair; c += 1) {
        product[into + c] = left[i * dPair + c] * right[j * dPair + c];
      }
    }
  }
  const projected = linear(product, tokens * tokens, dPair, dPair, weights.sToZProdOut);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      const into = (i * tokens + j) * dPair;
      for (let c = 0; c < dPair; c += 1) {
        pair[into + c] += rows[i * dPair + c] + columns[j * dPair + c] + projected[into + c];
      }
    }
  }

  // 🔴 A BUCKET COUNT, NOT A ONE-HOT MATMUL. `dist_bin_pairwise_embed` is an
  // `nn.Embedding(128, 256)` looked up by how many of the 127 boundaries the
  // distance exceeds, so the table's rows are read directly and the tensor is
  // the one matrix in this head that the exporter does NOT transpose.
  const edges = weights.boundaries;
  const distances = new Float32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    const a = inputs.repAtom[i] * 3;
    for (let j = 0; j < tokens; j += 1) {
      const b = inputs.repAtom[j] * 3;
      const dx = inputs.coordinates[a] - inputs.coordinates[b];
      const dy = inputs.coordinates[a + 1] - inputs.coordinates[b + 1];
      const dz = inputs.coordinates[a + 2] - inputs.coordinates[b + 2];
      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
      distances[i * tokens + j] = distance;
      let bucket = 0;
      for (let edge = 0; edge < edges.length; edge += 1) if (distance > edges[edge]) bucket += 1;
      const from = bucket * dPair;
      const into = (i * tokens + j) * dPair;
      for (let c = 0; c < dPair; c += 1) pair[into + c] += weights.distanceEmbedding[from + c];
    }
  }

  const pairMask = new Float32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      pairMask[i * tokens + j] = inputs.tokenMask[i] * inputs.tokenMask[j];
    }
  }
  // See the note at the top: the stack's answer is ADDED to its own input.
  const initial = Float32Array.from(pair);
  let stack = Float32Array.from(pair);
  const add = (delta) => {
    for (let index = 0; index < stack.length; index += 1) stack[index] += delta[index];
  };
  for (const block of weights.blocks) {
    add(triangleMultiplication(stack, pairMask, tokens, dPair, "outgoing",
                               block.triangleMultiplicationOutgoing));
    add(triangleMultiplication(stack, pairMask, tokens, dPair, "incoming",
                               block.triangleMultiplicationIncoming));
    add(transition(stack, tokens * tokens, dPair, block.pairTransition));
  }
  for (let index = 0; index < pair.length; index += 1) pair[index] += stack[index];

  const pooled = singleFromPair(pair, inputs.tokenMask, tokens, dPair, dSingle, weights);

  // 🔴 THE pLDDT TABLE IS INDEXED BY THE ATOM'S SLOT WITHIN ITS TOKEN, and
  // their `_compute_intra_token_idx` is a running count that RESETS at each
  // token boundary - atoms of one token are contiguous - clamped to the
  // table's 23 rows. Indexing it by the atom's global position instead reads a
  // different residue's weights and still returns a number.
  const slots = weights.maxAtomsPerToken;
  const bins = weights.plddtBins;
  const normed = layerNorm(
    gatherTokenToAtom(pooled, inputs.atomToToken, atoms, dSingle), atoms, dSingle,
    weights.plddtNormScale, weights.plddtNormOffset);
  const plddtLogits = new Float32Array(atoms * bins);
  let slot = 0;
  for (let atom = 0; atom < atoms; atom += 1) {
    if (atom > 0 && inputs.atomToToken[atom] !== inputs.atomToToken[atom - 1]) slot = 0;
    const table = Math.min(slot, slots - 1) * dSingle * bins;
    for (let bin = 0; bin < bins; bin += 1) {
      let sum = 0;
      for (let c = 0; c < dSingle; c += 1) {
        sum += normed[atom * dSingle + c] * weights.plddtWeight[table + c * bins + bin];
      }
      plddtLogits[atom * bins + bin] = sum;
    }
    slot += 1;
  }
  const plddtPerAtom = categoricalMean(plddtLogits, atoms, bins, 0, 1);

  const plddtSum = new Float32Array(tokens);
  const plddtCount = new Float32Array(tokens);
  for (let atom = 0; atom < atoms; atom += 1) {
    const token = inputs.atomToToken[atom];
    plddtSum[token] += plddtPerAtom[atom] * inputs.atomMask[atom];
    plddtCount[token] += inputs.atomMask[atom];
  }
  const plddt = new Float32Array(tokens);
  for (let token = 0; token < tokens; token += 1) {
    plddt[token] = plddtSum[token] / Math.max(plddtCount[token], 1e-6);
  }
  const plddtCa = new Float32Array(tokens);
  for (let token = 0; token < tokens; token += 1) plddtCa[token] = plddtPerAtom[inputs.repAtom[token]];
  let weighted = 0;
  let total = 0;
  for (let atom = 0; atom < atoms; atom += 1) {
    weighted += plddtPerAtom[atom] * inputs.atomMask[atom];
    total += inputs.atomMask[atom];
  }

  const paeBins = weights.paeBins;
  const paeLogits = linear(pair, tokens * tokens, dPair, paeBins, weights.pae);
  const pae = categoricalMean(paeLogits, tokens * tokens, paeBins, 0, 32);

  return { initial, pair, single: pooled, distances, plddtLogits, plddtPerAtom, plddt, plddtCa,
           complexPlddt: weighted / (total + 1e-8), paeLogits, pae };
}

function gatherTokenToAtom(tokenValues, atomToToken, atoms, channels) {
  const out = new Float32Array(atoms * channels);
  for (let atom = 0; atom < atoms; atom += 1) {
    const from = atomToToken[atom] * channels;
    for (let c = 0; c < channels; c += 1) out[atom * channels + c] = tokenValues[from + c];
  }
  return out;
}
