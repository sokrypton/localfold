/**
 * AF3's diffusion conditioning: what the denoiser knows besides the atoms.
 *
 *     pair   = Linear(LayerNorm([trunkPair | relativeEncoding]))
 *     pair  += transition(pair) x 2                     UNCONDITIONED
 *     single = Linear(LayerNorm([trunkSingle | targetFeat]))
 *     single += Linear(LayerNorm(fourier(log(sigma / sigma_data) / 4)))
 *     single += transition(single) x 2                  UNCONDITIONED
 *
 * 🔴 THE PAIR CONDITIONING IS ONE CONCATENATION OR TWO COMPRESSIONS, AND THE
 * NORM'S LENGTH SAYS WHICH. AF3 concatenates the trunk pair with the RAW
 * relative encoding and normalises the lot (128 + 139 = 267); OpenDDE compresses
 * each to the pair width SEPARATELY and concatenates those (128 + 128 = 256);
 * protenix2 and boltz2 project the relative encoding alone and concatenate it
 * with the RAW trunk pair. "The joint LayerNorm over the widened concatenation
 * couples the two terms, so this is a different function and not a
 * re-association."
 *
 * 🔴 AND GETTING THAT WIDTH WRONG IS SILENT IN ONE DIRECTION AND LOUD IN THE
 * OTHER, WHICH IS WHY THIS PORT REFUSES RATHER THAN COMPUTES. The reference
 * records both: boltz2's 267 "is LONGER than its scale, so the LayerNorm read
 * past the end and every one of 73728 elements came out NaN. protenix2's 395 is
 * SHORTER than its 512, so it read a prefix, stayed finite, and the GPU made the
 * identical mistake - so check-af3-diffusion-conditioning compared two wrong
 * computations and passed at 3.20e-7. A checker agreeing with itself is the
 * failure this repository keeps finding, and the only reason it surfaced is that
 * a second model rounded the other way." So the derived width is checked against
 * the LayerNorm's own length here, and a disagreement is an error.
 *
 * 🔴 AND THE SINGLE IT READS IS NOT THE WIDTH IT WRITES. "AF3's projection is
 * [831, 384] and its trunk single is also 384, so `seqChannels` served as both
 * and this line was right by coincidence. boltz2's is [768, 768] - 768 out, 384
 * in - and reading one number for two gave 1152 against a LayerNorm of 768."
 *
 * 🔴 AND THE FOUR TRANSITIONS HERE ARE UNCONDITIONED, which is the distinction
 * boltz2's up-gate is gated on: its plain Transition has no up-gate and only its
 * adaLN-conditioned block does.
 *
 * 🔴 AND THE NOISE LEVEL IS SCALED BY SIGMA_DATA BEFORE THE LOG, with the
 * Fourier weight and bias coming from the MODEL always: "AF3 keeps them as
 * frozen constants in its SOURCE rather than its checkpoint, while every ported
 * model of the lineage trained its own - so a table compiled in here would be
 * correct for exactly one checkpoint."
 */
import { ONE_TARGET, PRELUDE, sized, memoSource } from "../../runtime/execution.js";
import { conditionedTransition } from "./transformer.js";
import { layerNorm } from "../../kernels/layer-norm.js";
import { linear } from "../../kernels/linear.js";
import { singleCondPadding, singleCondSource } from "../../../shared/af3/dialect.js";
import { sumSource } from "./atom-attention.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */

/** AF3's assumed data scale, in angstroms. Every noise level is relative to it. */
export const SIGMA_DATA = 16.0;

/**
 * The two coefficients that turn a network output into a denoised structure,
 * plus what its input is divided by.
 *
 * "At a noise level far above SIGMA_DATA the skip term vanishes and the update
 * carries everything; far below, the reverse. This is what makes the same
 * network usable at every step of the schedule."
 */
export function scalings(noiseLevel) {
  const denominator = noiseLevel * noiseLevel + SIGMA_DATA * SIGMA_DATA;
  return {
    skip: SIGMA_DATA * SIGMA_DATA / denominator,
    out: noiseLevel * SIGMA_DATA / Math.sqrt(denominator),
    in: 1 / Math.sqrt(denominator),
  };
}

/**
 * The Fourier embedding of a noise level, on the host: it is ONE row, so there
 * is nothing for a GPU to parallelise over.
 *
 * @param {number} scaledNoiseLevel the level ALREADY divided by SIGMA_DATA
 */
export function noiseEmbedding(scaledNoiseLevel, weight, bias) {
  if (weight === undefined || bias === undefined) {
    throw new Error(
      "the Fourier embedding's weight and bias must come from the model: AF3 keeps them"
      + " in its source and ported models train their own, so there is no correct default");
  }
  const t = Math.log(scaledNoiseLevel) / 4;
  return Float32Array.from(weight, (w, index) => Math.cos(2 * Math.PI * (t * w + bias[index])));
}

/**
 * The single conditioning's concatenation, with the openfold3 lineage's two
 * ZERO columns inserted where the dialect asks for them.
 *
 * 🔴 THE PADDED COLUMNS SIT AFTER THE **TRUNK SINGLE** BLOCK, not at the end:
 * "the concatenation is [trunkSingle, targetFeat]", and the two inserts land one
 * after the restype block and one after the profile block in the PADDED index
 * space - "so the second already counts the first". The mapping comes from
 * `shared/af3/dialect.js`'s own `singleCondSource` rather than being re-derived,
 * which is the point of that function existing: "written as a loop over
 * `singleCondPadding` rather than arithmetic so that the CPU reference and the
 * generated WGSL below cannot express it differently".
 */
const singleConcatSourceUncached = (rows, leftWidth, rightWidth, padding) => {
  const width = leftWidth + rightWidth + padding.length;
  // The per-channel source, resolved on the host from `singleCondSource` and
  // written out as one branch per RUN of channels that read the same tensor at
  // the same offset, so nothing here can disagree with it.
  // 🔴 NOT ONE LINE A CHANNEL: that was an 833-way select, inlined four times
  // by `at` below, and ANGLE's Metal compiler on an M2 never finished linking
  // it - the first diffusion step of every fold hung the tab.
  const runs = [];
  for (let channel = 0; channel < width; channel += 1) {
    const source = singleCondSource(padding, channel);
    const tex = source < 0 ? null : source < leftWidth ? "left" : "right";
    const offset = tex === null ? 0 : (tex === "left" ? source : source - leftWidth) - channel;
    const last = runs[runs.length - 1];
    if (last !== undefined && last.tex === tex && last.offset === offset) last.end = channel + 1;
    else runs.push({ tex, offset, end: channel + 1 });
  }
  const cases = runs.map(({ tex, offset, end }) => {
    if (tex === null) return `  if (channel < ${end}) return 0.0;`;
    const stride = tex === "left" ? leftWidth : rightWidth;
    return `  if (channel < ${end}) return fetch1(${tex}Tex, row * ${stride} + channel + (${offset}));`;
  }).join("\n");
  return sized({ ELEMENTS: rows * width }, `${PRELUDE}${ONE_TARGET}
const int WIDTH = ${width};
uniform sampler2D leftTex;
uniform sampler2D rightTex;

float channelAt(int row, int channel) {
${cases}
  return 0.0;
}

float at(int element) {
  int channel = element % WIDTH;
  int row = element / WIDTH;
  return channelAt(row, channel);
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    at(base),
    base + 1 < ELEMENTS ? at(base + 1) : 0.0,
    base + 2 < ELEMENTS ? at(base + 2) : 0.0,
    base + 3 < ELEMENTS ? at(base + 3) : 0.0);
}`);
};

/** `out[row][c] = c < leftWidth ? left[row][c] : right[row][c - leftWidth]`. */
const concatenateSourceUncached = (rows, leftWidth, rightWidth) => sized({ ELEMENTS: rows * (leftWidth + rightWidth) }, `${PRELUDE}${ONE_TARGET}
const int LEFT = ${leftWidth};
const int RIGHT = ${rightWidth};
const int WIDTH = ${leftWidth + rightWidth};
uniform sampler2D leftTex;
uniform sampler2D rightTex;

float at(int element) {
  int channel = element % WIDTH;
  int row = element / WIDTH;
  return channel < LEFT
    ? fetch1(leftTex, row * LEFT + channel)
    : fetch1(rightTex, row * RIGHT + (channel - LEFT));
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    at(base),
    base + 1 < ELEMENTS ? at(base + 1) : 0.0,
    base + 2 < ELEMENTS ? at(base + 2) : 0.0,
    base + 3 < ELEMENTS ? at(base + 3) : 0.0);
}`);

/** `out[row][c] = x[row][c] + row0[c]` - the noise term, broadcast per token. */
const broadcastRowSourceUncached = (rows, channels) => sized({ ELEMENTS: rows * channels }, `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
uniform sampler2D xTex;
uniform sampler2D rowTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = fetch4(xTex, texel) + fetch4(rowTex, texel % C4);
}`);

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{trunkSingle: Tensor, trunkPair: Tensor, targetFeat: Tensor,
 *   identity: Tensor, noiseLevel: number, tokens: number, weights: any,
 *   relativeWidth: number,
 *   fourierWeight: Float32Array, fourierBias: Float32Array,
 *   padUnknownDna: boolean, epsilon?: number,
 *   onStage?: (stage: string, tensor: Tensor) => void}} call
 * @returns {{single: Tensor, pair: Tensor}} the caller owns both
 */
/**
 * The PAIR half alone, which does not depend on the noise level.
 *
 * 🔴 AND THAT IS WHY IT IS ITS OWN FUNCTION: a sampler calls the head once a
 * STEP, and only the single track reads sigma - through the Fourier embedding.
 * The pair half is the trunk pair, the relative encoding, a joint LayerNorm over
 * the concatenation and two transitions, every one of them a function of the
 * TRUNK, so recomputing it 200 times computes the same tensor 200 times. At 59
 * tokens that is a 3481x267 ragged LayerNorm and a 267-wide projection a step.
 *
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{trunkPair: Tensor, identity: Tensor, tokens: number, weights: any,
 *   relativeWidth: number, epsilon?: number,
 *   onStage?: (stage: string, tensor: Tensor) => void}} call
 * @returns {Tensor} `tokens^2 * pairChannels`, the caller's to release
 */
export function pairConditioning(execution, call) {
  const { trunkPair, identity, tokens, weights: w, relativeWidth, epsilon = 1e-5 } = call;
  const pairs = tokens * tokens;
  const pairChannels = w.pairChannels;
  const trunkPairChannels = w.trunkPairChannels ?? pairChannels;

  const split = w.zTrunkProjection !== undefined;
  const projectedRelpos = !split && w.relpeProjection !== undefined;
  const width = split ? 2 * pairChannels
    : projectedRelpos ? trunkPairChannels + pairChannels
      : trunkPairChannels + relativeWidth;
  // 🔴 THE DERIVED WIDTH IS CHECKED AGAINST THE NORM'S OWN LENGTH, and this port
  // refuses rather than reading a prefix. See the note at the top: a short read
  // stays finite and two implementations can make the same mistake.
  if (w.pairCondInitialNormScale.elements !== width) {
    throw new Error(
      `af3 diffusion conditioning: the pair concatenation is ${width} channels`
      + ` (${split ? "two compressions" : projectedRelpos
        ? "a raw trunk pair and a projected relative encoding"
        : "a raw trunk pair and the raw relative encoding"})`
      + ` but its LayerNorm scale is ${w.pairCondInitialNormScale.elements}:`
      + " the dialect and the weights disagree, and a short read would stay finite");
  }

  // 🔴 ALLOCATED AFTER THE WIDTH CHECK, so a refusal leaks nothing. The first
  // version built the relative encoding first and the refusal arm then left one
  // tensor live - which `Storage.stats()` turned into a failing gate rather than
  // a slow page.
  //
  // The relative encoding is projected straight to the pair width where the
  // dialect compresses it and kept raw where AF3 concatenates it, so the kernel
  // runs over the full 139 columns either way.
  const relative = execution.allocate(pairs * relativeWidth, "af3.diffcond.relative");
  execution.dispatch({
    source: rawRelativeSource(tokens, relativeWidth),
    output: relative,
    inputs: { identityTex: identity },
    label: `af3.diffcond.relative:${tokens}x${relativeWidth}`,
  });

  let left;
  let right;
  let leftWidth;
  let rightWidth;
  const temporary = [];
  if (projectedRelpos) {
    left = trunkPair;
    leftWidth = trunkPairChannels;
    rightWidth = pairChannels;
    right = linear(execution, {
      x: relative, weight: w.relpeProjection, rows: pairs,
      inChannels: relativeWidth, outChannels: pairChannels, weightLayout: "in-major",
    });
    temporary.push(right);
  } else if (split) {
    leftWidth = pairChannels;
    rightWidth = pairChannels;
    const normalised = layerNorm(execution, {
      x: trunkPair, weight: w.zTrunkNormScale, bias: w.zTrunkNormOffset,
      rows: pairs, channels: trunkPairChannels, epsilon,
    });
    left = linear(execution, {
      x: normalised, weight: w.zTrunkProjection, rows: pairs,
      inChannels: trunkPairChannels, outChannels: pairChannels, weightLayout: "in-major",
    });
    execution.release(normalised);
    right = linear(execution, {
      x: relative, weight: w.relpeProjection, rows: pairs,
      inChannels: relativeWidth, outChannels: pairChannels, weightLayout: "in-major",
    });
    temporary.push(left, right);
  } else {
    left = trunkPair;
    right = relative;
    leftWidth = trunkPairChannels;
    rightWidth = relativeWidth;
  }
  const features2d = execution.allocate(pairs * width, "af3.diffcond.features2d");
  execution.dispatch({
    source: concatenateSource(pairs, leftWidth, rightWidth),
    output: features2d,
    inputs: { leftTex: left, rightTex: right },
    label: `af3.diffcond.concat2d:${pairs}x${width}`,
  });
  for (const tensor of temporary) execution.release(tensor);
  execution.release(relative);

  const normalised2d = layerNorm(execution, {
    x: features2d, weight: w.pairCondInitialNormScale, bias: w.pairCondInitialNormOffset,
    rows: pairs, channels: width, epsilon,
  });
  execution.release(features2d);
  let pair = linear(execution, {
    x: normalised2d, weight: w.pairCondInitialProjection, rows: pairs,
    inChannels: width, outChannels: pairChannels, weightLayout: "in-major",
  });
  execution.release(normalised2d);
  call.onStage?.("conditioning.pairInitial", pair);
  for (const transitionWeights of w.pairTransitions) {
    const delta = conditionedTransition(execution, {
      x: pair, rows: pairs, channels: pairChannels, factor: 2,
      weights: transitionWeights, prefix: "", epsilon,
    });
    const next = execution.allocate(pairs * pairChannels, "af3.diffcond.pair");
    execution.dispatch({
      source: sumSource(pairs * pairChannels),
      output: next,
      inputs: { aTex: pair, bTex: delta },
      label: `af3.diffcond.pair-residual:${pairs}x${pairChannels}`,
    });
    execution.release(delta);
    execution.release(pair);
    pair = next;
  }

  pair.label = "af3.diffcond.pair";
  return pair;
}

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{trunkSingle: Tensor, trunkPair: Tensor, targetFeat: Tensor,
 *   identity: Tensor, noiseLevel: number, tokens: number, weights: any,
 *   relativeWidth: number, fourierWeight: Float32Array, fourierBias: Float32Array,
 *   padUnknownDna: boolean, reusePair?: Tensor, epsilon?: number,
 *   onStage?: (stage: string, tensor: Tensor) => void}} call
 *   `reusePair` is a pair conditioning computed earlier - see `pairConditioning`
 *   for why a sampler has one - and is BORROWED: the result says so, and
 *   releasing it is the caller's.
 * @returns {{single: Tensor, pair: Tensor, pairIsBorrowed: boolean}}
 */
export function diffusionConditioning(execution, call) {
  const {
    trunkSingle, targetFeat, noiseLevel, tokens, weights: w, epsilon = 1e-5,
  } = call;
  if (typeof call.padUnknownDna !== "boolean") {
    throw new Error(
      "af3 diffusion conditioning: padUnknownDna has no default - stock AF3 is false"
      + " and the openfold3 lineage true, and the LayerNorm's length is what asserts"
      + " the dialect and the weights agree");
  }
  const seqChannels = w.seqChannels;
  const pair = call.reusePair ?? pairConditioning(execution, call);
  const pairIsBorrowed = call.reusePair !== undefined;

  // ...and the trunk single with target_feat.
  const trunkSingleChannels = w.trunkSingleChannels ?? seqChannels;
  const targetFeatWidth = w.targetFeatWidth;
  const padding = singleCondPadding({ padSingleCondUnknownDna: call.padUnknownDna },
    trunkSingleChannels);
  const singleWidth = trunkSingleChannels + targetFeatWidth + padding.length;
  if (w.singleCondInitialNormScale.elements !== singleWidth) {
    throw new Error(
      `af3 diffusion conditioning: the single concatenation is ${singleWidth} channels`
      + ` but its LayerNorm scale is ${w.singleCondInitialNormScale.elements}`);
  }
  const features1d = execution.allocate(tokens * singleWidth, "af3.diffcond.features1d");
  execution.dispatch({
    source: singleConcatSource(tokens, trunkSingleChannels, targetFeatWidth, padding),
    output: features1d,
    inputs: { leftTex: trunkSingle, rightTex: targetFeat },
    label: `af3.diffcond.concat1d:${tokens}x${singleWidth}`,
  });
  const normalised1d = layerNorm(execution, {
    x: features1d, weight: w.singleCondInitialNormScale,
    bias: w.singleCondInitialNormOffset,
    rows: tokens, channels: singleWidth, epsilon,
  });
  execution.release(features1d);
  let single = linear(execution, {
    x: normalised1d, weight: w.singleCondInitialProjection,
    bias: w.singleCondInitialProjectionBias,
    rows: tokens, inChannels: singleWidth, outChannels: seqChannels,
    weightLayout: "in-major",
  });
  execution.release(normalised1d);
  call.onStage?.("conditioning.singleInitial", single);

  // The noise term: one row, embedded on the HOST because there is nothing for
  // a GPU to parallelise over - and its weight and bias arrive as host arrays
  // for the same reason, which is why they are on the call rather than among the
  // uploaded weights.
  const embedded = noiseEmbedding(
    noiseLevel / SIGMA_DATA, call.fourierWeight, call.fourierBias);
  const embeddedTensor = execution.upload(embedded, "af3.diffcond.fourier");
  const normalisedNoise = layerNorm(execution, {
    x: embeddedTensor, weight: w.noiseEmbeddingInitialNormScale,
    bias: w.noiseEmbeddingInitialNormOffset,
    rows: 1, channels: embedded.length, epsilon,
  });
  execution.release(embeddedTensor);
  const projectedNoise = linear(execution, {
    x: normalisedNoise, weight: w.noiseEmbeddingInitialProjection, rows: 1,
    inChannels: embedded.length, outChannels: seqChannels, weightLayout: "in-major",
  });
  execution.release(normalisedNoise);
  const withNoise = execution.allocate(tokens * seqChannels, "af3.diffcond.single");
  execution.dispatch({
    source: broadcastRowSource(tokens, seqChannels),
    output: withNoise,
    inputs: { xTex: single, rowTex: projectedNoise },
    label: `af3.diffcond.noise:${tokens}x${seqChannels}`,
  });
  execution.release(projectedNoise);
  execution.release(single);
  single = withNoise;

  for (const transitionWeights of w.singleTransitions) {
    const delta = conditionedTransition(execution, {
      x: single, rows: tokens, channels: seqChannels, factor: 2,
      weights: transitionWeights, prefix: "", epsilon,
    });
    const next = execution.allocate(tokens * seqChannels, "af3.diffcond.single");
    execution.dispatch({
      source: sumSource(tokens * seqChannels),
      output: next,
      inputs: { aTex: single, bTex: delta },
      label: `af3.diffcond.single-residual:${tokens}x${seqChannels}`,
    });
    execution.release(delta);
    execution.release(single);
    single = next;
  }

  single.label = "af3.diffcond.single";
  return { single, pair, pairIsBorrowed };
}

/**
 * The relative encoding as its RAW 139 one-hot columns.
 *
 * 🔴 THIS IS THE ONE PLACE THE 139 COLUMNS ARE MATERIALISED, and it is because
 * AF3 concatenates them with the trunk pair and normalises the lot - so the
 * columns have to exist as columns. `trunk/embedder.js` folds the same four
 * one-hots into a projection instead, which is right there and wrong here.
 */
const rawRelativeSourceUncached = (tokens, relativeWidth) => sized({ L: tokens, ELEMENTS: tokens * tokens * relativeWidth }, `${PRELUDE}${ONE_TARGET}
const int WIDTH = ${relativeWidth};
const int MAX_IDX = 32;
const int MAX_CHAIN = 2;
const int POSITION_BINS = 66;
uniform sampler2D identityTex;

float at(int element) {
  int channel = element % WIDTH;
  int pair = element / WIDTH;
  int i = pair / L;
  int j = pair - i * L;

  float residueI = fetch1(identityTex, i);
  float residueJ = fetch1(identityTex, j);
  float tokenI = fetch1(identityTex, L + i);
  float tokenJ = fetch1(identityTex, L + j);
  bool sameChain = fetch1(identityTex, 2 * L + i) == fetch1(identityTex, 2 * L + j);
  bool sameEntity = fetch1(identityTex, 3 * L + i) == fetch1(identityTex, 3 * L + j);
  float symI = fetch1(identityTex, 4 * L + i);
  float symJ = fetch1(identityTex, 4 * L + j);

  int offset = clamp(int(residueI) - int(residueJ) + MAX_IDX, 0, 2 * MAX_IDX);
  int positionRow = sameChain ? offset : 2 * MAX_IDX + 1;
  bool sameResidue = sameChain && residueI == residueJ;
  int tokenOffset = clamp(int(tokenI) - int(tokenJ) + MAX_IDX, 0, 2 * MAX_IDX);
  int tokenRow = POSITION_BINS + (sameResidue ? tokenOffset : 2 * MAX_IDX + 1);
  int relativeChain = clamp(int(symI) - int(symJ) + MAX_CHAIN, 0, 2 * MAX_CHAIN);
  int chainRow = POSITION_BINS * 2 + 1
    + (sameEntity ? relativeChain : 2 * MAX_CHAIN + 1);

  if (channel == positionRow) return 1.0;
  if (channel == tokenRow) return 1.0;
  if (channel == POSITION_BINS * 2 && sameEntity) return 1.0;
  if (channel == chainRow) return 1.0;
  return 0.0;
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    at(base),
    base + 1 < ELEMENTS ? at(base + 1) : 0.0,
    base + 2 < ELEMENTS ? at(base + 2) : 0.0,
    base + 3 < ELEMENTS ? at(base + 3) : 0.0);
}`);

// Every builder above, memoised on its arguments (`memoSource` in webgl/runtime/execution.js).
export const singleConcatSource = memoSource(singleConcatSourceUncached);
export const concatenateSource = memoSource(concatenateSourceUncached);
export const broadcastRowSource = memoSource(broadcastRowSourceUncached);
export const rawRelativeSource = memoSource(rawRelativeSourceUncached);
