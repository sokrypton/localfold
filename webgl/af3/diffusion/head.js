/**
 * AF3's diffusion head: one denoising step, noisy positions in, the model's
 * estimate of clean ones out.
 *
 *     cond    = diffusionConditioning(trunk, target_feat, sigma)
 *     scaled  = positionsNoisy * atomMask * scale.in
 *     encoded = atomCrossAttentionEncoder(..., scaled, trunkSingle, cond.pair)
 *     act     = encoded.tokenAct + Linear(LayerNorm(cond.single))
 *     act     = diffusionTransformer(act, cond.single, cond.pair)
 *     update  = atomDecoder(LayerNorm(act), encoded)
 *     out     = (scale.skip * positionsNoisy + scale.out * update) * atomMask
 *
 * 🔴 THE OUTPUT IS A BLEND, NOT A PREDICTION. "AF3 returns
 * `skip * positions_noisy + out * update`, where the two coefficients depend on
 * the noise level: at high noise the update dominates, at low noise the input
 * does. Returning the update alone type-checks, runs, and produces a structure
 * that is wrong in a way that looks like a bad model rather than a bug."
 *
 * 🔴 AND THE POSITIONS ARE MASKED AND THEN RESCALED BY THE NOISE LEVEL, "so the
 * encoder always sees something of order one however far down the schedule it
 * is. Feeding raw angstroms works at low noise and saturates at high."
 *
 * 🔴 AND THE TRUNK'S SINGLE CONDITIONS THE ATOMS WHILE THE HEAD'S OWN
 * CONDITIONING CONDITIONS THE TOKEN TRANSFORMER. "They are different tensors and
 * AF3 uses both" - the encoder gets `trunkSingle` and the transformer gets
 * `cond.single`, and swapping them is a shape-compatible mistake.
 *
 * 🔴 AND THE DECODER'S SKIP CONNECTION IS THE ENCODER'S OUTPUT, NOT ITS INPUT.
 * "AF3 broadcasts the token activation to every atom and then ADDS what the
 * encoder's own transformer produced, so the decoder starts from a
 * representation that has already seen the atoms. Skipping it leaves the decoder
 * guessing the local geometry from a token average."
 *
 * 🔴 AND THE DECODER HAS ITS OWN PAIR LAYERNORM AND PROJECTION BUT READS THE
 * **ENCODER'S** PAIR CONDITIONING: "the atom geometry did not change between
 * them."
 */
import { ONE_TARGET, PRELUDE, sized } from "../../runtime/execution.js";
import { atomCrossAttentionEncoder } from "./atom-encoder.js";
import { atomPairLogits, crossAttentionBlock, gather, sumSource } from "./atom-attention.js";
import { diffusionConditioning, scalings } from "./conditioning.js";
import { diffusionTransformerSteps } from "./transformer.js";
import { drain } from "../../runtime/steps.js";
import { layerNorm } from "../../kernels/layer-norm.js";
import { linear } from "../../kernels/linear.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */

/** `out[token][atom][c] = projected[token][c]` - one token's features to its atoms. */
export const broadcastToAtomsSource = (tokens, dense, channels) => sized({ ELEMENTS: tokens * dense * channels }, `${PRELUDE}${ONE_TARGET}
const int DENSE = ${dense};
const int C4 = ${channels / 4};
uniform sampler2D projectedTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int slot = texel / C4;
  int token = slot / DENSE;
  result = fetch4(projectedTex, token * C4 + group);
}`);

/** `out[row][c] = (a[row][c] + b[row][c]) * mask[row]`. */
export const sumAndMaskSource = (rows, channels) => sized({ ELEMENTS: rows * channels }, `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
uniform sampler2D aTex;
uniform sampler2D bTex;
uniform sampler2D maskTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int row = texel / C4;
  result = (fetch4(aTex, texel) + fetch4(bTex, texel)) * fetch1(maskTex, row);
}`);

/** `out[row][c] = x[row][c] * mask[row]`, over a 3-wide row stored at stride 4. */
export const scalePositionsSource = (atoms) => sized({ ELEMENTS: atoms * 3 }, `${PRELUDE}${ONE_TARGET}
uniform sampler2D positionsTex;
uniform sampler2D maskTex;
uniform float factor;

float at(int element) {
  int atom = element / 3;
  return fetch1(positionsTex, element) * fetch1(maskTex, atom) * factor;
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

/**
 * `out[i] = (skip * noisy[i] + out * update[i]) * atomMask[atom]`
 *
 * The blend, which is the whole output of the head - see the note at the top.
 */
export const blendSource = (atoms) => sized({ ELEMENTS: atoms * 3 }, `${PRELUDE}${ONE_TARGET}
uniform sampler2D noisyTex;
uniform sampler2D updateTex;
uniform sampler2D maskTex;
uniform float skip;
uniform float outScale;

float at(int element) {
  int atom = element / 3;
  return (skip * fetch1(noisyTex, element) + outScale * fetch1(updateTex, element))
    * fetch1(maskTex, atom);
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

/** A three-channel gather with both strides given; see the encoder's note. */
export const positionGatherSource = (count, sourceStride, outStride) => sized({ ELEMENTS: count * outStride }, `${PRELUDE}${ONE_TARGET}
const int OUT = ${outStride};
const int STRIDE = ${sourceStride};
uniform sampler2D sourceTex;
uniform sampler2D indicesTex;
uniform sampler2D maskTex;

float at(int element) {
  int row = element / OUT;
  int axis = element - row * OUT;
  if (axis >= 3) return 0.0;
  if (fetch1(maskTex, row) == 0.0) return 0.0;
  return fetch1(sourceTex, int(fetch1(indicesTex, row)) * STRIDE + axis);
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

/** @typedef {{indices: Tensor, mask: Tensor}} Gather */
/**
 * @typedef {{tokenAtomsToQueries: Gather, queriesToKeys: Gather,
 *   queriesToTokenAtoms: Gather, tokensToQueries: Gather, tokensToKeys: Gather}} Gathers
 */
/**
 * @typedef {{tokenAct: Tensor, skipConnection: Tensor, queriesMask: Tensor,
 *   keysMask: Tensor, queriesCond: Tensor, keysCond: Tensor, pairCond: Tensor}} Encoded
 */

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{tokenAct: Tensor, encoded: Encoded, gathers: Gathers,
 *   shape: {tokens: number, dense: number, subsets: number, queries: number,
 *     keys: number},
 *   weights: any, keyMaskedAtomAttention: boolean, chainedAtomLayerNorm: boolean,
 *   diffusionNoResidual: boolean, pairNormPerBlock: boolean,
 *   maskAtomActPerBlock: boolean, epsilon?: number}} call
 * @returns {Tensor} `tokens * dense * 3`, the position update
 */
export function atomDecoder(execution, call) {
  const { tokenAct, encoded, gathers, weights: w, epsilon = 1e-5 } = call;
  const { tokens, dense, subsets, queries, keys } = call.shape;
  const channels = w.channels;
  const queryRows = subsets * queries;

  const projected = linear(execution, {
    x: tokenAct, weight: w.projectTokenFeaturesForBroadcast, rows: tokens,
    inChannels: w.perTokenChannels, outChannels: channels, weightLayout: "in-major",
  });
  const perAtom = execution.allocate(tokens * dense * channels, "af3.decoder.per-atom");
  execution.dispatch({
    source: broadcastToAtomsSource(tokens, dense, channels),
    output: perAtom,
    inputs: { projectedTex: projected },
    label: `af3.decoder.broadcast:${tokens}x${dense}x${channels}`,
  });
  execution.release(projected);
  const onQueries = gather(execution, {
    source: perAtom, indices: gathers.tokenAtomsToQueries.indices,
    mask: gathers.tokenAtomsToQueries.mask, count: queryRows, channels,
  });
  execution.release(perAtom);
  // ...plus the encoder's OUTPUT, then masked.
  let act = execution.allocate(queryRows * channels, "af3.decoder.act");
  execution.dispatch({
    source: sumAndMaskSource(queryRows, channels),
    output: act,
    inputs: {
      aTex: onQueries, bTex: encoded.skipConnection, maskTex: encoded.queriesMask,
    },
    label: `af3.decoder.skip:${queryRows}x${channels}`,
  });
  execution.release(onQueries);

  // The decoder's own pair LayerNorm and projection, over the ENCODER's pair.
  const logits = atomPairLogits(execution, {
    pair: encoded.pairCond, subsets, queries, keys,
    pairChannels: w.pairChannels, heads: w.heads, blocks: w.blocks.length,
    weights: w, pairNormPerBlock: call.pairNormPerBlock, epsilon,
  });
  for (const [index, block] of w.blocks.entries()) {
    const next = crossAttentionBlock(execution, {
      queriesAct: act, queriesToKeys: gathers.queriesToKeys.indices,
      queriesMask: encoded.queriesMask, keysMask: encoded.keysMask,
      queriesCond: encoded.queriesCond, keysCond: encoded.keysCond,
      pairLogits: logits[index],
      subsets, queries, keys, channels, heads: w.heads, dimension: w.dimension,
      weights: block,
      keyMaskedAtomAttention: call.keyMaskedAtomAttention,
      chainedAtomLayerNorm: call.chainedAtomLayerNorm,
      diffusionNoResidual: call.diffusionNoResidual,
      epsilon,
    });
    execution.release(act);
    act = next;
  }
  for (const tensor of logits) execution.release(tensor);

  const masked = execution.allocate(queryRows * channels, "af3.decoder.masked");
  execution.dispatch({
    source: maskOnlySource(queryRows, channels),
    output: masked,
    inputs: { xTex: act, maskTex: encoded.queriesMask },
    label: `af3.decoder.mask:${queryRows}x${channels}`,
  });
  execution.release(act);
  const normalised = layerNorm(execution, {
    x: masked, weight: w.atomFeaturesLayerNormScale, bias: w.atomFeaturesLayerNormOffset,
    rows: queryRows, channels, epsilon,
  });
  execution.release(masked);
  // ...THREE channels, through `linear`'s ragged-output path - the same one AF2's
  // 50 pLDDT bins take. Projecting four and dropping one would read a weight
  // column that does not exist: this tensor is `channels * 3` floats.
  const update = linear(execution, {
    x: normalised, weight: w.atomFeaturesToPositionUpdate, rows: queryRows,
    inChannels: channels, outChannels: 3, weightLayout: "in-major",
  });
  execution.release(normalised);
  // ...and back to the token-atom layout the caller's coordinates live in.
  const onAtoms = execution.allocate(tokens * dense * 3, "af3.decoder.update");
  execution.dispatch({
    source: positionGatherSource(tokens * dense, 3, 3),
    output: onAtoms,
    inputs: {
      sourceTex: update, indicesTex: gathers.queriesToTokenAtoms.indices,
      maskTex: gathers.queriesToTokenAtoms.mask,
    },
    label: `af3.decoder.to-atoms:${tokens * dense}`,
  });
  execution.release(update);
  onAtoms.label = "af3.decoder.update";
  return onAtoms;
}

/** `out[row][c] = x[row][c] * mask[row]`. */
export const maskOnlySource = (rows, channels) => sized({ ELEMENTS: rows * channels }, `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
uniform sampler2D xTex;
uniform sampler2D maskTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = fetch4(xTex, texel) * fetch1(maskTex, texel / C4);
}`);

/**
 * @typedef {{positionsNoisy: Tensor, noiseLevel: number, atomMask: Tensor,
 *   seqMask: Tensor, trunkSingle: Tensor, trunkPair: Tensor, targetFeat: Tensor,
 *   conditioning: Tensor, refPos: Tensor, refSpaceUid: Tensor, identity: Tensor,
 *   gathers: Gathers,
 *   shape: {tokens: number, dense: number, subsets: number, queries: number,
 *     keys: number},
 *   weights: any, relativeWidth: number,
 *   fourierWeight: Float32Array, fourierBias: Float32Array,
 *   padUnknownDna: boolean, maskPaddedKeys: boolean,
 *   keysUidFromQueriesLayout: boolean, keyMaskedAtomAttention: boolean,
 *   chainedAtomLayerNorm: boolean, diffusionNoResidual: boolean,
 *   pairNormPerBlock: boolean, maskAtomActPerBlock: boolean,
 *   preTrunkQuery?: boolean, epsilon?: number, conditioningPair?: Tensor,
 *   onStage?: (stage: string, tensor: Tensor) => void}} HeadCall
 */

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {HeadCall} call
 * @returns {Tensor} `tokens * dense * 3`, the denoised positions
 */
export function diffusionHead(execution, call) {
  return drain(diffusionHeadSteps(execution, call));
}

/** `diffusionHead` as a generator: yields after the atom encoder and each transformer super-block. */
export function* diffusionHeadSteps(execution, call) {
  const {
    positionsNoisy, noiseLevel, atomMask, seqMask, trunkSingle, trunkPair, targetFeat,
    conditioning, refPos, refSpaceUid, identity, gathers, weights: w,
    fourierWeight, fourierBias, relativeWidth, epsilon = 1e-5,
  } = call;
  const { tokens, dense } = call.shape;
  const atoms = tokens * dense;
  const scale = scalings(noiseLevel);
  const dialect = {
    maskPaddedKeys: call.maskPaddedKeys,
    keysUidFromQueriesLayout: call.keysUidFromQueriesLayout,
    preTrunkQuery: call.preTrunkQuery,
    keyMaskedAtomAttention: call.keyMaskedAtomAttention,
    chainedAtomLayerNorm: call.chainedAtomLayerNorm,
    diffusionNoResidual: call.diffusionNoResidual,
    pairNormPerBlock: call.pairNormPerBlock,
    maskAtomActPerBlock: call.maskAtomActPerBlock,
  };

  // 🔴 THE PAIR CONDITIONING IS NOISE-INDEPENDENT AND A SAMPLER HOISTS IT. Only
  // the single track reads sigma, so a caller that denoises 200 times passes the
  // pair in and this computes it once - see `pairConditioning`. Borrowed, so it
  // is not released here.
  const cond = diffusionConditioning(execution, {
    trunkSingle, trunkPair, targetFeat, identity, noiseLevel, tokens,
    weights: w.conditioning, relativeWidth, fourierWeight, fourierBias,
    padUnknownDna: call.padUnknownDna, epsilon,
    ...(call.conditioningPair === undefined ? {} : { reusePair: call.conditioningPair }),
    onStage: call.onStage,
  });
  call.onStage?.("conditioning.single", cond.single);
  call.onStage?.("conditioning.pair", cond.pair);

  const scaled = execution.allocate(atoms * 3, "af3.head.scaled");
  execution.dispatch({
    source: scalePositionsSource(atoms),
    output: scaled,
    inputs: { positionsTex: positionsNoisy, maskTex: atomMask },
    floats: { factor: scale.in },
    label: `af3.head.scale:${atoms}`,
  });
  call.onStage?.("scaled positions", scaled);

  const encoded = atomCrossAttentionEncoder(execution, {
    conditioning, atomMask, refPos, refSpaceUid, gathers,
    shape: call.shape, weights: w.encoder,
    tokenAtomsAct: scaled,
    // The TRUNK's single conditions the atoms; the head's own conditions the
    // token transformer. Different tensors, and AF3 uses both.
    trunkSingleCond: trunkSingle,
    trunkPairCond: cond.pair,
    ...dialect, epsilon,
    onStage: call.onStage,
  });
  execution.release(scaled);
  call.onStage?.("encoder.tokenAct", encoded.tokenAct);
  // The decoder reads more of the encoder than tokenAct, and a NaN in any of
  // these reaches the coordinates without ever touching tokenAct.
  call.onStage?.("encoder.skipConnection", encoded.skipConnection);
  call.onStage?.("encoder.pairCond", encoded.pairCond);
  call.onStage?.("encoder.keysCond", encoded.keysCond);
  call.onStage?.("encoder.queriesMask", encoded.queriesMask);
  yield { stage: "encoder" };

  const normalisedSingle = layerNorm(execution, {
    x: cond.single, weight: w.singleCondEmbeddingNormScale,
    bias: w.singleCondEmbeddingNormOffset,
    rows: tokens, channels: w.seqChannels, epsilon,
  });
  const projectedSingle = linear(execution, {
    x: normalisedSingle, weight: w.singleCondEmbeddingProjection, rows: tokens,
    inChannels: w.seqChannels, outChannels: w.perTokenChannels, weightLayout: "in-major",
  });
  execution.release(normalisedSingle);
  let act = execution.allocate(tokens * w.perTokenChannels, "af3.head.act");
  execution.dispatch({
    source: sumSource(tokens * w.perTokenChannels),
    output: act,
    inputs: { aTex: encoded.tokenAct, bTex: projectedSingle },
    label: `af3.head.single-add:${tokens}x${w.perTokenChannels}`,
  });
  execution.release(projectedSingle);
  execution.release(encoded.tokenAct);
  call.onStage?.("after single projection", act);

  const transformed = yield* diffusionTransformerSteps(execution, {
    act, cond: cond.single, pairCond: cond.pair, mask: seqMask, tokens,
    weights: w.transformer, epsilon,
  });
  execution.release(act);
  call.onStage?.("transformer", transformed);
  const normalised = layerNorm(execution, {
    x: transformed, weight: w.outputNormScale, bias: w.outputNormOffset,
    rows: tokens, channels: w.perTokenChannels, epsilon,
  });
  execution.release(transformed);
  call.onStage?.("output-norm", normalised);

  const update = atomDecoder(execution, {
    tokenAct: normalised, encoded, gathers, shape: call.shape,
    weights: w.decoder, ...dialect, epsilon,
  });
  execution.release(normalised);
  call.onStage?.("decoder", update);
  for (const key of ["skipConnection", "queriesMask", "keysMask", "queriesCond",
    "keysCond", "pairCond"]) {
    execution.release(encoded[key]);
  }
  execution.release(cond.single);
  if (!cond.pairIsBorrowed) execution.release(cond.pair);

  const output = execution.allocate(atoms * 3, "af3.head.output");
  execution.dispatch({
    source: blendSource(atoms),
    output,
    inputs: { noisyTex: positionsNoisy, updateTex: update, maskTex: atomMask },
    floats: { skip: scale.skip, outScale: scale.out },
    label: `af3.head.blend:${atoms}`,
  });
  execution.release(update);
  output.label = "af3.head.denoised";
  return output;
}
