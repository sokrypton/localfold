/**
 * An AlphaFold 2 Evoformer block: nine sub-layers, each a residual.
 *
 * `webgpu/af2/evoformer/block.js`'s `encodeEvoformerBlock` is the order, and it
 * is not negotiable - the MSA half feeds the pair half through the outer product
 * mean, and the pair half's four sub-layers run on a representation the previous
 * one has already updated:
 *
 *     msa  += rowAttention(msa, bias from pair)
 *     msa  += columnAttention(msa)
 *     msa  += transition(msa)
 *     pair += outerProductMean(msa)
 *     pair += triangleMultiplication(pair, "outgoing")
 *     pair += triangleMultiplication(pair, "incoming")
 *     pair += triangleAttention(pair, "starting")
 *     pair += triangleAttention(pair, "ending")
 *     pair += transition(pair)
 *
 * 🔴 EVERY RESIDUAL COSTS A TENSOR HERE, WHICH IS THE ONE STRUCTURAL DIFFERENCE
 * FROM THE WebGPU BLOCK. That one calls `addInPlace` and writes the update back
 * into `msa` or `pair`; a fragment shader may not sample the texture its
 * framebuffer is attached to, so each residual allocates a destination and the
 * previous representation is released immediately after. The peak is therefore
 * two representations plus the current sub-layer's scratch rather than one plus
 * scratch - and `Storage.stats()` is what makes that measurable rather than
 * assumed.
 *
 * 🔴 AND TWO SUB-LAYERS NEED AN EXPLICIT TRANSPOSE WHERE THE WebGPU ONE PASSES A
 * FLAG. Column attention attends over the sequences and the ending-node
 * triangle attention over the other pair axis; `encodeAttention` expresses both
 * by reading its source through `source_row`, which a fragment shader writing
 * one texel cannot do. So both transpose in, attend, and transpose the update
 * back before it is added - three extra draws each, and the reason
 * `webgl/runtime/transpose.js` exists.
 */
import { attentionBlock } from "../../kernels/attention-block.js";
import { outerProductMean } from "../../kernels/outer-product-mean.js";
import { pairBias as projectPairBias } from "../../kernels/pair-bias.js";
import { transition } from "../../kernels/transition.js";
import { triangleMultiplication } from "../../kernels/triangle-multiplication.js";
import { add } from "../../runtime/elementwise.js";
import { globalAttention } from "../../kernels/global-attention.js";
import { transposeLeading } from "../../runtime/transpose.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */
/** @typedef {Record<string, Tensor>} Group a named set of uploaded weights */
/**
 * @typedef {{heads: number, attention: Group, pairBias?: Group}} AttentionGroup
 * @typedef {{msaRowAttention: AttentionGroup, msaColumnAttention: AttentionGroup,
 *   msaTransition: Group, outerProductMean: Group,
 *   triangleMultiplicationOutgoing: Group, triangleMultiplicationIncoming: Group,
 *   triangleAttentionStarting: AttentionGroup, triangleAttentionEnding: AttentionGroup,
 *   pairTransition: Group}} BlockWeights
 * @typedef {Omit<BlockWeights, "msaColumnAttention">
 *   & {msaColumnGlobalAttention: Group & {heads: number}}} ExtraBlockWeights
 */

const ATTENTION_NAMES = [
  "queryNormScale", "queryNormOffset", "queryWeight", "keyWeight", "valueWeight",
  "gatingWeight", "gatingBias", "outputWeight", "outputBias",
];
const TRANSITION_NAMES = [
  "layerNormScale", "layerNormOffset", "firstWeight", "firstBias", "secondWeight", "secondBias",
];
const OPM_NAMES = [
  "layerNormScale", "layerNormOffset", "leftWeight", "leftBias",
  "rightWeight", "rightBias", "outputWeight", "outputBias",
];
const TRIANGLE_NAMES = [
  "layerNormInWeight", "layerNormInBias",
  "linearAPWeight", "linearAPBias", "linearAGWeight", "linearAGBias",
  "linearBPWeight", "linearBPBias", "linearBGWeight", "linearBGBias",
  "layerNormOutWeight", "layerNormOutBias",
  "linearZWeight", "linearZBias", "linearGWeight", "linearGBias",
];

/**
 * Upload a block's weights once, in the structure `webgpu/af2/evoformer` uses:
 * `{heads, attention: {...}, pairBias: {...}}` for an attention, a flat group
 * for a transition, the outer product mean and a triangle multiplication.
 *
 * @param {import("../../runtime/execution.js").Execution} execution
 */
export function uploadEvoformerBlockWeights(execution, weights) {
  /** @type {(names: string[], source: any, label: string) => Group} */
  const group = (names, source, label) => {
    /** @type {Record<string, Tensor>} */
    const out = {};
    for (const name of names) {
      const values = source[name];
      if (values === undefined) throw new Error(`${label}: ${name} is missing`);
      out[name] = execution.upload(values, `w.${label}.${name}`);
    }
    return out;
  };
  const attention = (source, label, biasNames) => ({
    heads: source.heads,
    attention: group(ATTENTION_NAMES, source.attention, `${label}.attention`),
    pairBias: group(biasNames, source.pairBias, `${label}.pairBias`),
  });
  // The extra stack's column attention is GLOBAL and its weights are a flat
  // group rather than `{heads, attention}`, because that kernel has no separate
  // pair-bias or per-head key and value; see encodeExtraMsaBlock.
  const column = weights.msaColumnGlobalAttention !== undefined
    ? {
      msaColumnGlobalAttention: Object.assign(
        group(ATTENTION_NAMES, weights.msaColumnGlobalAttention, "msa-column-global"),
        { heads: weights.msaColumnGlobalAttention.heads }),
    }
    : {
      msaColumnAttention: {
        heads: weights.msaColumnAttention.heads,
        attention: group(ATTENTION_NAMES, weights.msaColumnAttention.attention, "msa-column"),
      },
    };
  return {
    msaRowAttention: attention(weights.msaRowAttention, "msa-row",
      ["layerNormScale", "layerNormOffset", "projectionWeight"]),
    ...column,
    msaTransition: group(TRANSITION_NAMES, weights.msaTransition, "msa-transition"),
    outerProductMean: group(OPM_NAMES, weights.outerProductMean, "opm"),
    triangleMultiplicationOutgoing: group(
      TRIANGLE_NAMES, weights.triangleMultiplicationOutgoing, "tri-out"),
    triangleMultiplicationIncoming: group(
      TRIANGLE_NAMES, weights.triangleMultiplicationIncoming, "tri-in"),
    triangleAttentionStarting: attention(
      weights.triangleAttentionStarting, "tri-att-start", ["projectionWeight"]),
    triangleAttentionEnding: attention(
      weights.triangleAttentionEnding, "tri-att-end", ["projectionWeight"]),
    pairTransition: group(TRANSITION_NAMES, weights.pairTransition, "pair-transition"),
  };
}

export function releaseEvoformerBlockWeights(execution, weights) {
  const walk = (value) => {
    if (value === undefined || value === null) return;
    if (typeof value === "number") return;
    if (typeof value === "object" && "texture" in value) {
      execution.release(/** @type {Tensor} */ (value));
      return;
    }
    for (const member of Object.values(value)) walk(member);
  };
  walk(weights);
}

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * The three precision options exist because the shipped block's own choices are
 * not uniform and are not switchable there: `msaAttentionPrecision` and
 * `triangleAttentionPrecision` default to f16 because its attentions store
 * their normalised input, projections and output that way, while
 * `trianglePrecision` defaults to f32 because its triangle multiplications do
 * not. Each was settled by sweeping it against the WGSL block, not by reading.
 *
 * @param {{msa: Tensor, pair: Tensor, msaMask: Tensor, pairMask: Tensor,
 *   sequences: number, length: number, cM: number, cZ: number, cOuter: number,
 *   triangleHidden: number, msaHidden: number, pairHidden: number,
 *   weights: BlockWeights, epsilon?: number,
 *   msaAttentionPrecision?: "f16" | "f32",
 *   triangleAttentionPrecision?: "f16" | "f32",
 *   trianglePrecision?: "f16" | "f32"}} call
 * @returns {{msa: Tensor, pair: Tensor}} new tensors; the caller's inputs are untouched
 */
export function evoformerBlock(execution, call) {
  const {
    msaMask, pairMask, sequences, length, cM, cZ, cOuter,
    triangleHidden, msaHidden, pairHidden, weights: w, epsilon = 1e-5,
  } = call;
  const pairs = length * length;
  let msa = call.msa;
  let pair = call.pair;
  let ownMsa = false;
  let ownPair = false;

  /** Add an update into the running representation and release what it replaces. */
  const residual = (current, update, own) => {
    const next = add(execution, current, update);
    execution.release(update);
    if (own) execution.release(current);
    return next;
  };

  // 1. MSA row attention, biased by the pair.
  {
    const row = w.msaRowAttention;
    const bias = projectPairBias(execution, {
      pair, weights: row.pairBias, length, pairChannels: cZ, heads: row.heads, epsilon,
    });
    const update = attentionBlock(execution, {
      activations: msa, mask: msaMask, weights: row.attention,
      batch: sequences, queryLength: length, channels: cM, heads: row.heads,
      pairBias: bias, epsilon, storagePrecision: call.msaAttentionPrecision ?? "f16",
    });
    execution.release(bias);
    msa = residual(msa, update, ownMsa);
    ownMsa = true;
  }

  // 2. MSA column attention: over the sequences, so the MSA is transposed in
  //    and the update transposed back.
  {
    const column = w.msaColumnAttention;
    const transposed = transposeLeading(execution, {
      x: msa, outer: sequences, inner: length, channels: cM,
    });
    const updateT = attentionBlock(execution, {
      activations: transposed, mask: msaMask, weights: column.attention,
      batch: length, queryLength: sequences, channels: cM, heads: column.heads,
      transpose: true, epsilon, storagePrecision: call.msaAttentionPrecision ?? "f16",
    });
    execution.release(transposed);
    const update = transposeLeading(execution, {
      x: updateT, outer: length, inner: sequences, channels: cM,
    });
    execution.release(updateT);
    msa = residual(msa, update, ownMsa);
    ownMsa = true;
  }

  // 3. MSA transition.
  {
    const update = transition(execution, {
      activations: msa, weights: w.msaTransition,
      rows: sequences * length, channels: cM, hiddenChannels: msaHidden, epsilon,
    });
    msa = residual(msa, update, ownMsa);
    ownMsa = true;
  }

  pair = pairHalf(execution, call, msa, pair, ownPair, residual);
  msa.label = "block.msa";
  pair.label = "block.pair";
  return { msa, pair };
}

/**
 * The six sub-layers from the outer product mean down, shared by the main stack
 * and the extra one - `encodeEvoformerPairBlock` is the same split on the WGSL
 * side, and for the same reason: the extra stack differs only in its MSA half.
 */
function pairHalf(execution, call, msa, pairIn, ownPairIn, residual) {
  const {
    msaMask, pairMask, sequences, length, cM, cZ, cOuter,
    triangleHidden, pairHidden, weights: w, epsilon = 1e-5,
  } = call;
  const pairs = length * length;
  let pair = pairIn;
  let ownPair = ownPairIn;

  // 1. The MSA's one path into the pair track.
  {
    const update = outerProductMean(execution, {
      activations: msa, mask: msaMask, weights: w.outerProductMean,
      sequences, length, cM, cOuter, cZ, layerNormEpsilon: epsilon,
    });
    pair = residual(pair, update, ownPair);
    ownPair = true;
  }

  // 2, 3. The two triangle multiplications, in order: the second reads what the
  //       first wrote.
  /** @type {[("outgoing" | "incoming"), Group][]} */
  const multiplications = [
    ["outgoing", w.triangleMultiplicationOutgoing],
    ["incoming", w.triangleMultiplicationIncoming],
  ];
  for (const [direction, group] of multiplications) {
    const update = triangleMultiplication(execution, {
      z: pair, mask: pairMask, weights: group,
      shape: { length, cZ, cHidden: triangleHidden },
      direction, epsilon,
      projectionPrecision: call.trianglePrecision ?? "f32",
    });
    pair = residual(pair, update, ownPair);
    ownPair = true;
  }

  // 4. Triangle attention, starting node: the bias comes off the block's own
  //    normalised pair.
  {
    const starting = w.triangleAttentionStarting;
    const update = attentionBlock(execution, {
      activations: pair, mask: pairMask, weights: starting.attention,
      batch: length, queryLength: length, channels: cZ, heads: starting.heads,
      pairBiasProjection: starting.pairBias.projectionWeight, epsilon,
      storagePrecision: call.triangleAttentionPrecision ?? "f16",
      ...(call.triangleAttentionParts ?? {}),
    });
    pair = residual(pair, update, ownPair);
    ownPair = true;
  }

  // 5. Triangle attention, ending node: the other pair axis.
  {
    const ending = w.triangleAttentionEnding;
    const transposed = transposeLeading(execution, {
      x: pair, outer: length, inner: length, channels: cZ,
    });
    const updateT = attentionBlock(execution, {
      activations: transposed, mask: pairMask, weights: ending.attention,
      batch: length, queryLength: length, channels: cZ, heads: ending.heads,
      transpose: true, pairBiasProjection: ending.pairBias.projectionWeight, epsilon,
      storagePrecision: call.triangleAttentionPrecision ?? "f16",
      ...(call.triangleAttentionParts ?? {}),
    });
    execution.release(transposed);
    const update = transposeLeading(execution, {
      x: updateT, outer: length, inner: length, channels: cZ,
    });
    execution.release(updateT);
    pair = residual(pair, update, ownPair);
    ownPair = true;
  }

  // 6. Pair transition.
  {
    const update = transition(execution, {
      activations: pair, weights: w.pairTransition,
      rows: pairs, channels: cZ, hiddenChannels: pairHidden, epsilon,
    });
    pair = residual(pair, update, ownPair);
  }
  return pair;
}

/**
 * The EXTRA-MSA block: the same thing with global column attention in place of
 * ordinary column attention, which is the only difference
 * (`encodeExtraMsaBlock`). The extra stack runs thousands of alignment rows, so
 * attending over the sequences pairwise is unaffordable and its query is a
 * masked column mean instead; see `webgl/kernels/global-attention.js`.
 *
 * @param {Omit<Parameters<typeof evoformerBlock>[1], "weights">
 *   & {weights: ExtraBlockWeights}} call
 * @returns {{msa: Tensor, pair: Tensor}}
 */
export function extraMsaBlock(execution, call) {
  const {
    msaMask, pairMask, sequences, length, cM, msaHidden, weights: w, epsilon = 1e-5,
  } = call;
  let msa = call.msa;
  let pair = call.pair;
  let ownMsa = false;

  const residual = (current, update, own) => {
    const next = add(execution, current, update);
    execution.release(update);
    if (own) execution.release(current);
    return next;
  };

  // 1. MSA row attention, biased by the pair - the same as the main stack's.
  {
    const row = w.msaRowAttention;
    const bias = projectPairBias(execution, {
      pair, weights: row.pairBias, length, pairChannels: call.cZ, heads: row.heads, epsilon,
    });
    const update = attentionBlock(execution, {
      activations: msa, mask: msaMask, weights: row.attention,
      batch: sequences, queryLength: length, channels: cM, heads: row.heads,
      pairBias: bias, epsilon, storagePrecision: call.msaAttentionPrecision ?? "f16",
    });
    execution.release(bias);
    msa = residual(msa, update, ownMsa);
    ownMsa = true;
  }

  // 2. Global column attention, which transposes and transposes back itself.
  {
    const global = w.msaColumnGlobalAttention;
    const update = globalAttention(execution, {
      msa, mask: msaMask, weights: global,
      sequences, length, channels: cM, heads: global.heads, epsilon,
    });
    msa = residual(msa, update, ownMsa);
    ownMsa = true;
  }

  // 3. MSA transition.
  {
    const update = transition(execution, {
      activations: msa, weights: w.msaTransition,
      rows: sequences * length, channels: cM, hiddenChannels: msaHidden, epsilon,
    });
    msa = residual(msa, update, ownMsa);
  }

  pair = pairHalf(execution, call, msa, pair, false, residual);
  msa.label = "extra.msa";
  pair.label = "extra.pair";
  return { msa, pair };
}

const TEMPLATE_BLOCK_GROUPS = [
  "triangleAttentionStarting", "triangleAttentionEnding",
  "triangleMultiplicationOutgoing", "triangleMultiplicationIncoming", "pairTransition",
];

/**
 * A TEMPLATE pair block's weights: the pair half of an Evoformer block's,
 * without the outer product mean.
 *
 * It is here rather than in `template.js` so that the four name lists above
 * stay the one statement of what a group holds - `releaseEvoformerBlockWeights`
 * walks whatever it is given and releases both kinds.
 */
export function uploadTemplatePairBlockWeights(execution, weights) {
  /** @type {(names: string[], source: any, label: string) => Group} */
  const group = (names, source, label) => {
    /** @type {Record<string, Tensor>} */
    const out = {};
    for (const name of names) {
      const values = source[name];
      if (values === undefined) throw new Error(`${label}: ${name} is missing`);
      out[name] = execution.upload(values, `w.${label}.${name}`);
    }
    return out;
  };
  const attention = (source, label) => ({
    heads: source.heads,
    attention: group(ATTENTION_NAMES, source.attention, `${label}.attention`),
    pairBias: group(["projectionWeight"], source.pairBias, `${label}.pairBias`),
  });
  for (const name of TEMPLATE_BLOCK_GROUPS) {
    if (weights[name] === undefined) throw new Error(`template block weights: ${name} is missing`);
  }
  return {
    triangleAttentionStarting: attention(
      weights.triangleAttentionStarting, "template.tri-att-start"),
    triangleAttentionEnding: attention(
      weights.triangleAttentionEnding, "template.tri-att-end"),
    triangleMultiplicationOutgoing: group(
      TRIANGLE_NAMES, weights.triangleMultiplicationOutgoing, "template.tri-out"),
    triangleMultiplicationIncoming: group(
      TRIANGLE_NAMES, weights.triangleMultiplicationIncoming, "template.tri-in"),
    pairTransition: group(TRANSITION_NAMES, weights.pairTransition, "template.pair-transition"),
  };
}

/**
 * One block of AF2's template pair stack: five sub-layers, each a residual.
 *
 *     pair += triangleAttention(pair, "starting")
 *     pair += triangleAttention(pair, "ending")
 *     pair += triangleMultiplication(pair, "outgoing")
 *     pair += triangleMultiplication(pair, "incoming")
 *     pair += transition(pair)
 *
 * 🔴 IT IS NOT `pairHalf` MINUS THE OUTER PRODUCT MEAN - THE ORDER DIFFERS.
 * `encodeTemplatePairBlock` runs both ATTENTIONS first and the two
 * multiplications after, where `encodeEvoformerPairBlock` runs the
 * multiplications first. Every sub-layer reads what the previous one wrote, so
 * the two orders compute different tensors from the same weights, and reusing
 * `pairHalf` with the mean deleted would have been a plausible wrong answer.
 *
 * The pair track is `templateChannels` wide here (64, not 128), and there is no
 * MSA: the template's own representation is what the stack refines.
 *
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{pair: Tensor, pairMask: Tensor, length: number, cZ: number,
 *   triangleHidden: number, pairHidden: number,
 *   weights: ReturnType<typeof uploadTemplatePairBlockWeights>, epsilon?: number,
 *   triangleAttentionPrecision?: "f16" | "f32",
 *   triangleAttentionParts?: Record<string, "f16" | "f32">,
 *   trianglePrecision?: "f16" | "f32"}} call
 * @returns {Tensor} a new tensor; the caller's `pair` is untouched
 */
export function templatePairBlock(execution, call) {
  const {
    pairMask, length, cZ, triangleHidden, pairHidden, weights: w, epsilon = 1e-5,
  } = call;
  let pair = call.pair;
  let own = false;
  const residual = (current, update) => {
    const next = add(execution, current, update);
    execution.release(update);
    if (own) execution.release(current);
    own = true;
    return next;
  };

  // 1. Triangle attention, starting node: the bias comes off the block's own
  //    normalised pair, exactly as in the main stack.
  {
    const starting = w.triangleAttentionStarting;
    pair = residual(pair, attentionBlock(execution, {
      activations: pair, mask: pairMask, weights: starting.attention,
      batch: length, queryLength: length, channels: cZ, heads: starting.heads,
      pairBiasProjection: starting.pairBias.projectionWeight, epsilon,
      storagePrecision: call.triangleAttentionPrecision ?? "f16",
      ...(call.triangleAttentionParts ?? {}),
    }));
  }

  // 2. Triangle attention, ending node: the other pair axis, so transpose in
  //    and transpose the update back.
  {
    const ending = w.triangleAttentionEnding;
    const transposed = transposeLeading(execution, {
      x: pair, outer: length, inner: length, channels: cZ,
    });
    const updateT = attentionBlock(execution, {
      activations: transposed, mask: pairMask, weights: ending.attention,
      batch: length, queryLength: length, channels: cZ, heads: ending.heads,
      transpose: true, pairBiasProjection: ending.pairBias.projectionWeight, epsilon,
      storagePrecision: call.triangleAttentionPrecision ?? "f16",
      ...(call.triangleAttentionParts ?? {}),
    });
    execution.release(transposed);
    const update = transposeLeading(execution, {
      x: updateT, outer: length, inner: length, channels: cZ,
    });
    execution.release(updateT);
    pair = residual(pair, update);
  }

  // 3, 4. The two triangle multiplications, in order.
  /** @type {[("outgoing" | "incoming"), Group][]} */
  const multiplications = [
    ["outgoing", w.triangleMultiplicationOutgoing],
    ["incoming", w.triangleMultiplicationIncoming],
  ];
  for (const [direction, group] of multiplications) {
    pair = residual(pair, triangleMultiplication(execution, {
      z: pair, mask: pairMask, weights: group,
      shape: { length, cZ, cHidden: triangleHidden },
      direction, epsilon,
      projectionPrecision: call.trianglePrecision ?? "f32",
    }));
  }

  // 5. Pair transition.
  pair = residual(pair, transition(execution, {
    activations: pair, weights: w.pairTransition,
    rows: length * length, channels: cZ, hiddenChannels: pairHidden, epsilon,
  }));
  pair.label = "template.pair";
  return pair;
}
