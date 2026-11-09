/**
 * AF3's whole trunk: the embedder, the template stack, both block stacks and the
 * distogram head.
 *
 *     embed  ->  template  ->  4 x msaBlock  ->  48 x pairformerBlock  ->  distogram
 *
 * "This is the assembly, not new arithmetic - every piece is checked
 * individually... What it adds is the thing no per-piece check can show: that
 * the pieces fit in the order AF3 runs them, and that fifty-two blocks of a
 * residual stack do not turn a correct block into a wrong answer."
 *
 * 🔴 THE TEMPLATE STACK READS THE PAIR AS THE EMBEDDER LEFT IT, and its term is
 * added back to that same pair - which is why neither module owns the
 * composition and this one does. `cpu/af3/trunk/embedder.js` takes the term as a
 * FUNCTION of the pair for exactly this reason.
 *
 * 🔴 AND boltz2 ADDS THE PRE-MSA PAIR BACK: "its MSAModule returns the updated z
 * and its caller adds z to that, so what reaches the pairformer is
 * `2 * z_in + delta`." `msaDoubleAddPair` is that term, and it needs the
 * embedder's pair held across the whole MSA stack.
 *
 * 🔴 AND THE DISTOGRAM HEAD IS SYMMETRISED BY A SUM, NOT A MEAN. "AF3 computes
 * one half and adds its transpose, so the logits are twice the size a mean would
 * give - which is not a rescaling once a softmax sees them. (chai-1's post-hoc
 * head does use a mean, and swapping the two silently sharpens or flattens every
 * contact probability.)"
 */
import { ONE_TARGET, PRELUDE, sized, memoSource } from "../../runtime/execution.js";
import { add } from "../../runtime/elementwise.js";
import { embed } from "./embedder.js";
import { linear } from "../../kernels/linear.js";
import { msaBlock } from "./msa-block.js";
import { pairformerBlock } from "./pairformer-block.js";
import { drain } from "../../runtime/steps.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */

export const FIRST_BREAK = 2.3125;
export const LAST_BREAK = 21.6875;
export const DISTOGRAM_BINS = 64;

/** The distogram bin edges: `bins - 1` of them, evenly spaced (64 bins for AF3, 65 for rosettafold3). */
export function binEdges(bins = DISTOGRAM_BINS) {
  const breaks = new Float32Array(bins - 1);
  for (let index = 0; index < bins - 1; index += 1) {
    breaks[index] = FIRST_BREAK
      + (LAST_BREAK - FIRST_BREAK) * index / (bins - 2);
  }
  return breaks;
}

/**
 * `logits[i][j][b] = half[i][j][b] + half[j][i][b]` - the sum, not the mean.
 */
const symmetriseSourceUncached = (tokens, bins = DISTOGRAM_BINS) => sized({ L: tokens, ELEMENTS: tokens * tokens * bins }, `${PRELUDE}${ONE_TARGET}
const int BINS = ${bins};
uniform sampler2D halfTex;

// element by element, so a bin count that is not a whole number of texels (rosettafold3's 65) needs no padding
float symmetric(int at) {
  int bin = at % BINS;
  int pair = at / BINS;
  int i = pair / L;
  int j = pair - i * L;
  return fetch1(halfTex, (i * L + j) * BINS + bin) + fetch1(halfTex, (j * L + i) * BINS + bin);
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(symmetric(base),
    base + 1 < ELEMENTS ? symmetric(base + 1) : 0.0,
    base + 2 < ELEMENTS ? symmetric(base + 2) : 0.0,
    base + 3 < ELEMENTS ? symmetric(base + 3) : 0.0);
}`);

/**
 * `contactProbs[i][j] = pairMask[i][j] * sum(softmax(logits)[b] for b < bins[i][j])`
 *
 * 🔴 THE CONTACT BIN COUNT IS PER PAIR, NOT A CONSTANT. `af3ContactBins` takes
 * one class a token and gives each pair its own threshold, so the kernel reads
 * a per-pair count rather than baking one in.
 */
const contactSourceUncached = (tokens, bins = DISTOGRAM_BINS) => sized({ L: tokens, ELEMENTS: tokens * tokens }, `${PRELUDE}${ONE_TARGET}
const int BINS = ${bins};
uniform sampler2D logitsTex;
uniform sampler2D maskTex;
uniform sampler2D binsTex;

float contactAt(int pair) {
  int base = pair * BINS;
  float largest = -1e30;
  for (int b = 0; b < BINS; ++b) largest = max(largest, fetch1(logitsTex, base + b));
  float total = 0.0;
  float contact = 0.0;
  int counted = int(fetch1(binsTex, pair));
  for (int b = 0; b < BINS; ++b) {
    float weight = exp(fetch1(logitsTex, base + b) - largest);
    total += weight;
    if (b < counted) contact += weight;
  }
  return fetch1(maskTex, pair) * (contact / total);
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    contactAt(base),
    base + 1 < ELEMENTS ? contactAt(base + 1) : 0.0,
    base + 2 < ELEMENTS ? contactAt(base + 2) : 0.0,
    base + 3 < ELEMENTS ? contactAt(base + 3) : 0.0);
}`);

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{pair: Tensor, pairMask: Tensor, contactBins: Tensor, halfLogits: Tensor,
 *   tokens: number, pairChannels: number}} call
 * @returns {{logits: Tensor, contactProbs: Tensor}} the caller owns both
 */
export function distogramHead(execution, call) {
  const { pair, pairMask, contactBins, halfLogits, halfLogitsBias, tokens, pairChannels } = call;
  const bins = call.bins ?? DISTOGRAM_BINS;
  const pairs = tokens * tokens;
  // 🔴 A TRAINED BIAS (protenix2, mol1, boltz2, rosettafold3: `distogramBias`) RIDES IN THE HALF LOGITS, so the
  // symmetrising SUM applies it twice - which is what those checkpoints trained with (webgpu/af3/trunk/trunk.js adds
  // `2 * bias`). It was dropped here: the contact map and the confidence head's input were the bias-free ones.
  const half = linear(execution, {
    x: pair, weight: halfLogits, bias: halfLogitsBias, rows: pairs,
    inChannels: pairChannels, outChannels: bins, weightLayout: "in-major",
  });
  const logits = execution.allocate(pairs * bins, "af3.distogram.logits");
  execution.dispatch({
    source: symmetriseSource(tokens, bins),
    output: logits,
    inputs: { halfTex: half },
    label: `af3.distogram.symmetrise:${tokens}`,
  });
  execution.release(half);
  const contactProbs = execution.allocate(pairs, "af3.distogram.contacts");
  execution.dispatch({
    source: contactSource(tokens, bins),
    output: contactProbs,
    inputs: { logitsTex: logits, maskTex: pairMask, binsTex: contactBins },
    label: `af3.distogram.contacts:${tokens}`,
  });
  logits.label = "af3.distogram.logits";
  contactProbs.label = "af3.distogram.contacts";
  return { logits, contactProbs };
}

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{embedded: {pair: Tensor, msa: Tensor, single: Tensor},
 *   templateUpdate?: Tensor,
 *   msaMask: Tensor, pairMask: Tensor, seqMask: Tensor, contactBins: Tensor,
 *   halfLogits: Tensor,
 *   msaBlocks: any[], pairformerBlocks: any[],
 *   tokens: number, sequences: number, pairChannels: number,
 *   msaUpdateBeforeOuterProduct: boolean, msaDoubleAddPair?: boolean,
 *   swapTransposedBias?: boolean, divideByLength?: boolean,
 *   opmBiasAfterNorm?: boolean, epsilon?: number,
 *   onBlock?: (stack: string, index: number) => void}} call
 * @returns {{pair: Tensor, single: Tensor, logits: Tensor, contactProbs: Tensor}}
 *   the caller owns these four and NOTHING from `embedded`, which this consumes
 */
/** `runTrunkSteps`, run straight through: the trunk's tensors. */
export function runTrunk(execution, call) {
  return drain(runTrunkSteps(execution, call));
}

/**
 * The trunk as a generator: yields `{ stage, index }` after each MSA and
 * pairformer block, so a page can let go of its thread between blocks
 * (webgl/runtime/steps.js); returns what `runTrunk` returns.
 */
export function* runTrunkSteps(execution, call) {
  const {
    embedded, templateUpdate, msaMask, pairMask, seqMask, contactBins, halfLogits, halfLogitsBias, bins,
    msaBlocks, pairformerBlocks, tokens, sequences, pairChannels,
    msaUpdateBeforeOuterProduct, msaDoubleAddPair = false,
    swapTransposedBias = false, divideByLength = false, opmBiasAfterNorm = false,
    epsilon = 1e-5,
  } = call;
  // 🔴 THE SEAMS, UNDER webgpu/af3/trunk/trunk.js's TAP NAMES, read back only when a caller asks (`onSeam`) - so a fold
  // tool can hold each stage to the WGSL trunk's rather than only the finished pair (tools/gpu/fold-af3-webgl.js --seams)
  const seam = (name, tensor) => {
    if (call.onSeam !== undefined) call.onSeam(name, Float32Array.from(execution.read(tensor)));
  };

  // 🔴 THIS CONSUMES THE EMBEDDER'S THREE TENSORS, which has to be said rather
  // than left to the reader: every one of them is superseded inside the stacks
  // (the pair by the template residual or the first MSA block, the MSA by that
  // block, the single by the first pairformer block), and leaving them to the
  // caller leaked three the first time this ran. The caller keeps nothing from
  // `embed` after calling this.
  let pair = embedded.pair;
  let ownPair = true;
  seam("tap.z_init_generic", pair);
  // (Mol-1's is after its PairMixer, as the reference taps it)
  if ((call.pairMixerBlocks ?? []).length === 0) seam("tap.trunk_in_single", embedded.single);
  if (templateUpdate !== undefined) {
    const withTemplate = add(execution, pair, templateUpdate);
    execution.release(pair);
    pair = withTemplate;
  }
  seam("tap.z_after_template", pair);
  // ...and boltz2 needs that same pair back after the MSA stack, so it is held.
  const beforeMsa = msaDoubleAddPair ? pair : undefined;

  let msa = embedded.msa;
  let ownMsa = true;
  for (const [index, weights] of msaBlocks.entries()) {
    const next = msaBlock(execution, {
      msa, pair, msaMask, pairMask, sequences, tokens, weights,
      msaUpdateBeforeOuterProduct, swapTransposedBias, divideByLength,
      opmBiasAfterNorm, epsilon,
    });
    if (ownMsa) execution.release(msa);
    // ...the embedder's pair is released only once nothing still needs it.
    if (ownPair && pair !== beforeMsa) execution.release(pair);
    msa = next.msa;
    pair = next.pair;
    ownMsa = true;
    ownPair = true;
    call.onBlock?.("msa", index);
    yield { stage: "msa", index };
  }
  execution.release(msa);

  if (beforeMsa !== undefined) {
    const doubled = add(execution, pair, beforeMsa);
    if (ownPair) execution.release(pair);
    execution.release(beforeMsa);
    pair = doubled;
    ownPair = true;
  }
  // (after boltz2's double add, where webgpu/'s tap is)
  seam("tap.z_after_msa", pair);

  let single = embedded.single;
  let ownSingle = true;
  // 🔴 Mol-1: its PairMixer blocks at the pair's width, then the pairformer on a NARROWER pair of its own between two
  // bias-free adapters, the full pair rewritten from it after (shared/af3/dialect.js MOL1; webgpu/'s trunk the same)
  const mixer = call.pairMixerBlocks ?? [];
  for (const weights of mixer) {
    const next = pairformerBlock(execution, {
      pair, single, pairMask, seqMask, tokens, weights, swapTransposedBias, divideByLength, epsilon,
    });
    if (ownPair) execution.release(pair);
    if (ownSingle) execution.release(single);
    pair = next.pair;
    single = next.single;
    ownPair = true;
    ownSingle = true;
    yield { stage: "pairmixer" };
  }
  const widths = mixer.length > 0 ? call.pairWidth : undefined;
  if (widths !== undefined) {
    const narrow = linear(execution, {
      x: pair, weight: widths.trunkPairIn, rows: tokens * tokens,
      inChannels: pairChannels, outChannels: widths.trunkChannels, weightLayout: "in-major",
    });
    if (ownPair) execution.release(pair);
    pair = narrow;
    ownPair = true;
    seam("tap.trunk_in_pair", pair);
    seam("tap.trunk_in_single", single);
  }
  for (const [index, weights] of pairformerBlocks.entries()) {
    const next = pairformerBlock(execution, {
      pair, single, pairMask, seqMask, tokens, weights,
      swapTransposedBias, divideByLength, epsilon,
    });
    if (ownPair) execution.release(pair);
    if (ownSingle) execution.release(single);
    pair = next.pair;
    single = next.single;
    ownPair = true;
    ownSingle = true;
    call.onBlock?.("pairformer", index);
    yield { stage: "pairformer", index };
  }

  seam("tap.trunk_out_pair", pair);
  if (widths !== undefined) {
    const wide = linear(execution, {
      x: pair, weight: widths.trunkPairOut, rows: tokens * tokens,
      inChannels: widths.trunkChannels, outChannels: pairChannels, weightLayout: "in-major",
    });
    execution.release(pair);
    pair = wide;
  }
  const { logits, contactProbs } = distogramHead(execution, {
    pair, pairMask, contactBins, halfLogits, halfLogitsBias, bins, tokens, pairChannels,
  });
  pair.label = "af3.trunk.pair";
  single.label = "af3.trunk.single";
  return { pair, single, logits, contactProbs };
}

// Every builder above, memoised on its arguments (`memoSource` in webgl/runtime/execution.js).
export const symmetriseSource = memoSource(symmetriseSourceUncached);
export const contactSource = memoSource(contactSourceUncached);
