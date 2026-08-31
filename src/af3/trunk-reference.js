/**
 * AF3's whole trunk, on the CPU: embedder, both stacks, and the distogram head.
 *
 *     embed  ->  4 x msaBlock  ->  48 x pairformerBlock  ->  distogramHead
 *
 * This is the assembly, not new arithmetic - every piece is checked
 * individually by the scripts in tools/oracle. What it adds is the thing no
 * per-piece check can show: that the pieces fit in the order AF3 runs them, and
 * that fifty-two blocks of a residual stack do not turn a correct block into a
 * wrong answer.
 *
 * 🔴 STILL STUBBED, and the caller has to supply both: `targetFeat`, whose 384
 * atom-derived columns need the atom transformer encoder, and the template
 * embedding, which contributes even when every template slot is empty. See
 * embedder-reference.js.
 */
import { embed } from "./embedder-reference.js";
import { msaBlock } from "./msa-reference.js";
import { pairformerBlock } from "./pairformer-reference.js";
import { linear } from "./pairformer-reference.js";

const FIRST_BREAK = 2.3125;
const LAST_BREAK = 21.6875;
const NUM_BINS = 64;
const CONTACT_THRESHOLD = 8.0 + 1e-3;

/** The distogram bin edges: 63 of them, evenly spaced. */
export function binEdges() {
  const breaks = new Float32Array(NUM_BINS - 1);
  for (let index = 0; index < NUM_BINS - 1; index += 1) {
    breaks[index] = FIRST_BREAK
      + (LAST_BREAK - FIRST_BREAK) * index / (NUM_BINS - 2);
  }
  return breaks;
}

/**
 * The distogram head: logits, and the contact probability read off them.
 *
 * 🔴 THE HEAD IS SYMMETRISED BY A SUM, NOT A MEAN. AF3 computes one half and
 * adds its transpose, so the logits are twice the size a mean would give -
 * which is not a rescaling once a softmax sees them. (chai-1's post-hoc head
 * does use a mean, and swapping the two silently sharpens or flattens every
 * contact probability.)
 */
export function distogramHead(pair, pairMask, tokens, pairChannels, weights) {
  const half = linear(pair, tokens * tokens, pairChannels, NUM_BINS,
                      weights.halfLogits);
  const logits = new Float32Array(tokens * tokens * NUM_BINS);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      const forward = (i * tokens + j) * NUM_BINS;
      const backward = (j * tokens + i) * NUM_BINS;
      for (let b = 0; b < NUM_BINS; b += 1) {
        logits[forward + b] = half[forward + b] + half[backward + b];
      }
    }
  }

  // ...the bins that count as contact: every one whose TOP edge is at or below
  // 8 A. The last bin's top is extrapolated by one spacing, since the 63 breaks
  // describe 64 bins and the final one is open-ended.
  const breaks = binEdges();
  const spacing = breaks[breaks.length - 1] - breaks[breaks.length - 2];
  const contactBin = new Float32Array(NUM_BINS);
  for (let b = 0; b < NUM_BINS; b += 1) {
    const top = b < NUM_BINS - 1 ? breaks[b] : breaks[breaks.length - 1] + spacing;
    contactBin[b] = top <= CONTACT_THRESHOLD ? 1 : 0;
  }

  const contactProbs = new Float32Array(tokens * tokens);
  for (let index = 0; index < tokens * tokens; index += 1) {
    const base = index * NUM_BINS;
    let largest = -Infinity;
    for (let b = 0; b < NUM_BINS; b += 1) {
      if (logits[base + b] > largest) largest = logits[base + b];
    }
    let total = 0;
    let contact = 0;
    for (let b = 0; b < NUM_BINS; b += 1) {
      const probability = Math.exp(logits[base + b] - largest);
      total += probability;
      if (contactBin[b] === 1) contact += probability;
    }
    contactProbs[index] = pairMask[index] * (contact / total);
  }
  return { logits, contactProbs, binEdges: breaks };
}

/**
 * The whole trunk.
 *
 * @param {object} input     see embedder-reference.js, plus pairMask/seqMask
 * @param {object} weights   `{embedder, msaBlocks[], pairformerBlocks[], distogram}`
 * @param {{swapTransposedBias: boolean}} dialect
 * @param {(stage: string, index: number, state: object) => void} [onBlock]
 */
export function runTrunk(input, weights, dialect, onBlock) {
  const { tokens, pairMask, seqMask, msaMask, sequences } = input;
  const embedded = embed(input, weights.embedder);

  let msaState = {
    msa: embedded.msa, pair: embedded.pair, msaMask, pairMask, sequences, tokens,
  };
  for (let index = 0; index < weights.msaBlocks.length; index += 1) {
    const next = msaBlock(msaState, weights.msaBlocks[index], dialect);
    msaState = { ...msaState, msa: next.msa, pair: next.pair };
    if (onBlock) onBlock("msa", index, msaState);
  }

  let state = {
    pair: msaState.pair, single: embedded.single, pairMask, seqMask, tokens,
  };
  for (let index = 0; index < weights.pairformerBlocks.length; index += 1) {
    const next = pairformerBlock(state, weights.pairformerBlocks[index], dialect);
    state = { ...state, pair: next.pair, single: next.single };
    if (onBlock) onBlock("pairformer", index, state);
  }

  return {
    pair: state.pair,
    single: state.single,
    ...distogramHead(state.pair, pairMask, tokens,
                     weights.embedder.pairChannels, weights.distogram),
  };
}
