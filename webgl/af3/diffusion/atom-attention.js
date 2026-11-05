/**
 * AF3's atom cross-attention block, and the AdaLN pieces every diffusion stage
 * is built from.
 *
 *     xq       = adaptiveLayerNorm(queries, queryCond)
 *     keysAct  = gather(queriesToKeys, AF3 ? queries : xq)
 *     xk       = adaptiveLayerNorm(keysAct, keyCond)
 *     gathered = softmax_k(q.k * scale + maskBias + pairLogits) . v
 *     attended = adaptiveZeroInit(gathered * sigmoid(gate), queryCond)
 *     out      = queries + attended + transition(...) * sigmoid(condGate)
 *
 * 🔴 198 M OF THE DIFFUSION'S 203 M PARAMETERS ARE THIS BLOCK AT ANOTHER WIDTH.
 * "The 24-block transformer... is the AdaLN-conditioned attention already
 * written for the atom stack at different widths. Its atom encoder and decoder
 * are 0.9 M each and are literally atom-encoder.js with other weights." So this
 * one module is most of the head.
 *
 * 🔴 THE CONDITIONING IS NOT THE SAME WIDTH AS THE ACTIVATION. "In the atom
 * stack both are 128 and the distinction is invisible; in the diffusion
 * transformer the conditioning is 384 and the activation 768, so the two
 * projections are 384->768 rather than square. Assuming square runs on the atom
 * stack and reads the diffusion weights at the wrong stride."
 *
 * 🔴 THE TWO ADAPTIVE LAYERNORMS ARE PARALLEL OR CHAINED, AND THAT IS THE MODEL.
 * "AlphaFold 3 normalises the RAW activation twice, once per side. OpenDDE
 * reassigns... so the gather onto the key layout happens BETWEEN the two, and
 * the second norm re-centres and re-scales a tensor whose statistics the first
 * already fixed - while seeing the first one's learned scale. Two chained
 * LayerNorms are not one, and neither arrangement changes a shape."
 *
 * 🔴 THE MASK BIAS IS A PRODUCT, NOT A SUM - UNDER AlphaFold 3. "It penalises a
 * pair only when the query AND the key are padded, so a real query can still
 * attend to a padded key. OpenDDE, Protenix and RoseTTAFold3 ADD the two
 * instead, which is an OR... The difference is large in a mostly-empty window
 * and it is nothing at all when every key is real."
 *
 * 🔴 AND THE TRANSITION READS THE POST-ATTENTION ACTIVATION, except in
 * rosettafold3, whose "`no_residual_connection_between_attention_and_transition`
 * hands the transition the PRE-attention activation and adds both through ONE
 * residual. The two are not a reordering - the transition sees a different
 * tensor - and it is worth the atom encoder's whole skip connection."
 *
 * Every one of those four has no default on either side.
 */
import { ONE_TARGET, PRELUDE } from "../../runtime/execution.js";
import { layerNorm } from "../../kernels/layer-norm.js";
import { linear } from "../../kernels/linear.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */

/**
 * `out[row][c] = sigmoid(scale[row][c]) * LayerNorm(x)[row][c] + shift[row][c]`
 *
 * The LayerNorm takes neither a scale nor an offset - the conditioning supplies
 * both - and the conditioning's own norm takes a scale with NO offset.
 */
export const adaptiveBlendSource = (rows, channels) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${rows * channels};
uniform sampler2D normalisedTex;
uniform sampler2D scaleTex;
uniform sampler2D shiftTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  vec4 scale = fetch4(scaleTex, texel);
  result = (vec4(1.0) / (vec4(1.0) + exp(-scale))) * fetch4(normalisedTex, texel)
    + fetch4(shiftTex, texel);
}`;

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{x: Tensor, cond: Tensor, rows: number, channels: number,
 *   condChannels?: number, weights: any, prefix: string, epsilon?: number}} call
 * @returns {Tensor}
 */
export function adaptiveLayerNorm(execution, call) {
  const { x, cond, rows, channels, weights: w, prefix, epsilon = 1e-5 } = call;
  const condChannels = call.condChannels ?? channels;
  // Neither a scale nor an offset: the conditioning supplies both below.
  const normalised = layerNorm(execution, { x, rows, channels, epsilon });
  // ...and the conditioning gets a scale but NO offset before it is projected.
  const condNormalised = layerNorm(execution, {
    x: cond, weight: w[`${prefix}SingleCondLayerNormScale`],
    rows, channels: condChannels, epsilon,
  });
  const scale = linear(execution, {
    x: condNormalised, weight: w[`${prefix}SingleCondScaleWeights`],
    bias: w[`${prefix}SingleCondScaleBias`],
    rows, inChannels: condChannels, outChannels: channels, weightLayout: "in-major",
  });
  const shift = linear(execution, {
    x: condNormalised, weight: w[`${prefix}SingleCondBias`],
    rows, inChannels: condChannels, outChannels: channels, weightLayout: "in-major",
  });
  execution.release(condNormalised);
  const out = execution.allocate(rows * channels, `af3.adaln.${prefix || "block"}`);
  execution.dispatch({
    source: adaptiveBlendSource(rows, channels),
    output: out,
    inputs: { normalisedTex: normalised, scaleTex: scale, shiftTex: shift },
    label: `af3.adaln.blend:${rows}x${channels}`,
  });
  for (const tensor of [normalised, scale, shift]) execution.release(tensor);
  return out;
}

/**
 * The AdaLN-zero output gate: project, then gate on the conditioning.
 *
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{x: Tensor, cond: Tensor, rows: number, channels: number,
 *   condChannels?: number, inChannels?: number, weights: any, prefix: string}} call
 * @returns {Tensor}
 */
export function adaptiveZeroInit(execution, call) {
  const { x, cond, rows, channels, weights: w, prefix } = call;
  const condChannels = call.condChannels ?? channels;
  const inChannels = call.inChannels ?? channels;
  const projected = linear(execution, {
    x, weight: w[`${prefix}Transition2`], rows,
    inChannels, outChannels: channels, weightLayout: "in-major",
  });
  const gate = linear(execution, {
    x: cond, weight: w[`${prefix}AdaptiveZeroCondWeights`],
    bias: w[`${prefix}AdaptiveZeroCondBias`],
    rows, inChannels: condChannels, outChannels: channels,
    weightLayout: "in-major", activation: "sigmoid",
  });
  const out = execution.allocate(rows * channels, "af3.adaln.zero");
  execution.dispatch({
    source: elementwiseProductSource(rows * channels),
    output: out,
    inputs: { aTex: projected, bTex: gate },
    label: `af3.adaln.zero:${rows}x${channels}`,
  });
  execution.release(projected);
  execution.release(gate);
  return out;
}

/** `out[i] = a[i] * b[i]`, declared here so the gate's name says what it gates. */
export const elementwiseProductSource = (elements) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${elements};
uniform sampler2D aTex;
uniform sampler2D bTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = fetch4(aTex, texel) * fetch4(bTex, texel);
}`;

/**
 * One of AF3's atom-layout gathers.
 *
 * 🔴 WHERE THE MASK IS FALSE THE DESTINATION IS ZERO rather than whatever index
 * 0 holds, "which matters because padded slots index 0 legitimately".
 */
export const gatherSource = (count, channels) => `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int ELEMENTS = ${count * channels};
uniform sampler2D sourceTex;
uniform sampler2D indicesTex;
uniform sampler2D maskTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int row = texel / C4;
  if (fetch1(maskTex, row) == 0.0) { result = vec4(0.0); return; }
  int from = int(fetch1(indicesTex, row));
  result = fetch4(sourceTex, from * C4 + group);
}`;

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{source: Tensor, indices: Tensor, mask: Tensor, count: number,
 *   channels: number}} call
 * @returns {Tensor}
 */
export function gather(execution, call) {
  const { source, indices, mask, count, channels } = call;
  if (channels % 4 !== 0) {
    throw new RangeError(`af3 gather: channels must be a multiple of 4, got ${channels}`);
  }
  const out = execution.allocate(count * channels, "af3.gather");
  execution.dispatch({
    source: gatherSource(count, channels),
    output: out,
    inputs: { sourceTex: source, indicesTex: indices, maskTex: mask },
    label: `af3.gather:${count}x${channels}`,
  });
  return out;
}

/**
 * The windowed cross-attention itself: `subsets` windows of `queries` queries
 * over `keys` keys, with a per-(subset, head, query, key) bias.
 *
 * 🔴 IT IS NOT `kernels/attention-flash.js`, AND THE REASON IS THE SHAPE. That
 * kernel assumes the query count and the key count are the same - its bias
 * index is `(head * Q + q) * Q + k` and its mask index `b * Q + k` - where this
 * one attends 32 queries over 128 keys with a bias indexed
 * `((subset * heads + head) * queries + query) * keys + key`. Reusing it would
 * have meant lying about one of the two counts.
 *
 * One fragment owns four consecutive head-dimension channels of one
 * (subset, query, head) and sweeps every key, keeping the online softmax in
 * registers - the same formulation the flash kernel uses, without its
 * multi-target spread, because `keys` here is 128 at most.
 */
export const atomAttentionSource = (shape, keyMasked) => {
  const { subsets, queries, keys, heads, dimension } = shape;
  const width = heads * dimension;
  return `${PRELUDE}${ONE_TARGET}
const int QUERIES = ${queries};
const int KEYS = ${keys};
const int HEADS = ${heads};
const int D = ${dimension};
const int D4 = ${dimension / 4};
const int WIDTH = ${width};
const int ELEMENTS = ${subsets * queries * width};
uniform sampler2D qTex;
uniform sampler2D kTex;
uniform sampler2D vTex;
uniform sampler2D queryMaskTex;
uniform sampler2D keyMaskTex;
uniform sampler2D pairLogitsTex;
uniform float scale;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % D4;
  int rest = texel / D4;
  int head = rest % HEADS;
  int row = rest / HEADS;
  int query = row % QUERIES;
  int subset = row / QUERIES;

  int queryIndex = subset * QUERIES + query;
  int queryBase = queryIndex * WIDTH + head * D;
  int biasBase = ((subset * HEADS + head) * QUERIES + query) * KEYS;
  float queryMask = fetch1(queryMaskTex, queryIndex);

  float runningMax = -1e30;
  float runningSum = 0.0;
  vec4 acc = vec4(0.0);
  for (int key = 0; key < KEYS; ++key) {
    int keyIndex = subset * KEYS + key;
    int keyBase = keyIndex * WIDTH + head * D;
    float dot = 0.0;
    for (int d = 0; d < D; ++d) {
      dot += fetch1(qTex, queryBase + d) * fetch1(kTex, keyBase + d);
    }
    float keyMask = fetch1(keyMaskTex, keyIndex);
    // The product form penalises a pair only when BOTH are padded; the sum form
    // is an OR. Neither has a default.
    float maskBias = ${keyMasked
    ? "-1e9 * ((1.0 - queryMask) + (1.0 - keyMask))"
    : "1e9 * (queryMask - 1.0) * (keyMask - 1.0)"};
    float logit = dot * scale + maskBias + fetch1(pairLogitsTex, biasBase + key);
    float newMax = max(runningMax, logit);
    float previousScale = exp(runningMax - newMax);
    float weight = exp(logit - newMax);
    runningSum = runningSum * previousScale + weight;
    runningMax = newMax;
    int valueBase = keyBase + group * 4;
    acc = acc * previousScale + weight * vec4(
      fetch1(vTex, valueBase), fetch1(vTex, valueBase + 1),
      fetch1(vTex, valueBase + 2), fetch1(vTex, valueBase + 3));
  }
  result = acc / runningSum;
}`;
};

/** `out[i] = base[i] + attended[i] + transition[i] * sigmoid(gate[i])`. */
export const blockOutputSource = (elements) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${elements};
uniform sampler2D baseTex;
uniform sampler2D attendedTex;
uniform sampler2D transitionTex;
uniform sampler2D gateTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  vec4 gate = fetch4(gateTex, texel);
  result = fetch4(baseTex, texel) + fetch4(attendedTex, texel)
    + fetch4(transitionTex, texel) * (vec4(1.0) / (vec4(1.0) + exp(-gate)));
}`;

/** `swish(wide[i]) * wide[intermediate + i]`, the SwiGLU over a blocked row. */
export const swiGluSource = (rows, intermediate) => `${PRELUDE}${ONE_TARGET}
const int INTER4 = ${intermediate / 4};
const int ELEMENTS = ${rows * intermediate};
uniform sampler2D wideTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % INTER4;
  int row = texel / INTER4;
  int base = row * 2 * INTER4;
  vec4 gate = fetch4(wideTex, base + group);
  vec4 value = fetch4(wideTex, base + INTER4 + group);
  result = (gate / (vec4(1.0) + exp(-gate))) * value;
}`;

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{queriesAct: Tensor, queriesToKeys: Tensor, queriesMask: Tensor,
 *   keysMask: Tensor, queriesCond: Tensor, keysCond: Tensor, pairLogits: Tensor,
 *   subsets: number, queries: number, keys: number, channels: number,
 *   heads: number, dimension: number, condChannels?: number,
 *   weights: any, keyMaskedAtomAttention: boolean, chainedAtomLayerNorm: boolean,
 *   diffusionNoResidual: boolean, epsilon?: number}} call
 * @returns {Tensor} a new activation; the caller's `queriesAct` is untouched
 */
export function crossAttentionBlock(execution, call) {
  const {
    queriesAct, queriesToKeys, queriesMask, keysMask, queriesCond, keysCond, pairLogits,
    subsets, queries, keys, channels, heads, dimension, weights: w, epsilon = 1e-5,
  } = call;
  const condChannels = call.condChannels ?? channels;
  for (const name of
    ["keyMaskedAtomAttention", "chainedAtomLayerNorm", "diffusionNoResidual"]) {
    if (typeof call[name] !== "boolean") {
      throw new Error(
        `af3 atom block: ${name} has no default - the four dialect choices in this block`
        + " each change the answer without changing a shape");
    }
  }
  const queryRows = subsets * queries;
  const keyRows = subsets * keys;
  const width = heads * dimension;

  const xq = adaptiveLayerNorm(execution, {
    x: queriesAct, cond: queriesCond, rows: queryRows, channels, condChannels,
    weights: w, prefix: "q", epsilon,
  });
  // AF3 gathers the RAW activation onto the key layout; OpenDDE gathers the
  // already-normalised one, so its second norm sees the first one's scale.
  const keysAct = gather(execution, {
    source: call.chainedAtomLayerNorm ? xq : queriesAct,
    indices: queriesToKeys, mask: keysMask, count: keyRows, channels,
  });
  const xk = adaptiveLayerNorm(execution, {
    x: keysAct, cond: keysCond, rows: keyRows, channels, condChannels,
    weights: w, prefix: "k", epsilon,
  });
  execution.release(keysAct);

  const query = linear(execution, {
    x: xq, weight: w.qProjection, bias: w.qBias, rows: queryRows,
    inChannels: channels, outChannels: width, weightLayout: "in-major",
  });
  const key = linear(execution, {
    x: xk, weight: w.kProjection, rows: keyRows,
    inChannels: channels, outChannels: width, weightLayout: "in-major",
  });
  const value = linear(execution, {
    x: xk, weight: w.vProjection, rows: keyRows,
    inChannels: channels, outChannels: width, weightLayout: "in-major",
  });
  execution.release(xk);

  const gathered = execution.allocate(queryRows * width, "af3.atom.gathered");
  execution.dispatch({
    source: atomAttentionSource(
      { subsets, queries, keys, heads, dimension }, call.keyMaskedAtomAttention),
    output: gathered,
    inputs: {
      qTex: query, kTex: key, vTex: value,
      queryMaskTex: queriesMask, keyMaskTex: keysMask, pairLogitsTex: pairLogits,
    },
    floats: { scale: 1 / Math.sqrt(dimension) },
    label: `af3.atom.attend:${subsets}x${queries}x${keys}x${width}`
      + `${call.keyMaskedAtomAttention ? ":or" : ":and"}`,
  });
  for (const tensor of [query, key, value]) execution.release(tensor);

  const gate = linear(execution, {
    x: xq, weight: w.gatingQuery, rows: queryRows,
    inChannels: channels, outChannels: width,
    weightLayout: "in-major", activation: "sigmoid",
  });
  const gatedAttention = execution.allocate(queryRows * width, "af3.atom.gated");
  execution.dispatch({
    source: elementwiseProductSource(queryRows * width),
    output: gatedAttention,
    inputs: { aTex: gathered, bTex: gate },
    label: `af3.atom.gate:${queryRows}x${width}`,
  });
  execution.release(gathered);
  execution.release(gate);
  const attention = adaptiveZeroInit(execution, {
    x: gatedAttention, cond: queriesCond, rows: queryRows, channels, condChannels,
    inChannels: width, weights: w, prefix: "",
  });
  execution.release(gatedAttention);
  execution.release(xq);

  // The transition reads the POST-attention activation - except in
  // rosettafold3, which hands it the PRE-attention one.
  let transitionInput;
  if (call.diffusionNoResidual) {
    transitionInput = queriesAct;
  } else {
    transitionInput = execution.allocate(queryRows * channels, "af3.atom.after-attention");
    execution.dispatch({
      source: sumSource(queryRows * channels),
      output: transitionInput,
      inputs: { aTex: queriesAct, bTex: attention },
      label: `af3.atom.residual:${queryRows}x${channels}`,
    });
  }
  const normalised = adaptiveLayerNorm(execution, {
    x: transitionInput, cond: queriesCond, rows: queryRows, channels, condChannels,
    weights: w, prefix: "ffw", epsilon,
  });
  if (transitionInput !== queriesAct) execution.release(transitionInput);

  const intermediate = channels * 2;
  const wide = linear(execution, {
    x: normalised, weight: w.ffwTransition1, rows: queryRows,
    inChannels: channels, outChannels: intermediate * 2, weightLayout: "in-major",
  });
  let gatedWide = execution.allocate(queryRows * intermediate, "af3.atom.swiglu");
  execution.dispatch({
    source: swiGluSource(queryRows, intermediate),
    output: gatedWide,
    inputs: { wideTex: wide },
    label: `af3.atom.swiglu:${queryRows}x${intermediate}`,
  });
  execution.release(wide);
  // 🔴 boltz2's EXTRA UP-GATE, reading the SAME adaptive-LayerNorm output the
  // SwiGLU reads - not the block input. Present only where the weight is.
  if (w.ffwAToB !== undefined && w.ffwAToB !== null) {
    const upGate = linear(execution, {
      x: normalised, weight: w.ffwAToB, rows: queryRows,
      inChannels: channels, outChannels: intermediate, weightLayout: "in-major",
    });
    const multiplied = execution.allocate(queryRows * intermediate, "af3.atom.up-gated");
    execution.dispatch({
      source: elementwiseProductSource(queryRows * intermediate),
      output: multiplied,
      inputs: { aTex: gatedWide, bTex: upGate },
      label: `af3.atom.up-gate:${queryRows}x${intermediate}`,
    });
    execution.release(gatedWide);
    execution.release(upGate);
    gatedWide = multiplied;
  }
  execution.release(normalised);
  const projected = linear(execution, {
    x: gatedWide, weight: w.ffwTransition2, rows: queryRows,
    inChannels: intermediate, outChannels: channels, weightLayout: "in-major",
  });
  execution.release(gatedWide);
  const transitionGate = linear(execution, {
    x: queriesCond, weight: w.ffwAdaptiveZeroCondWeights, bias: w.ffwAdaptiveZeroCondBias,
    rows: queryRows, inChannels: condChannels, outChannels: channels,
    weightLayout: "in-major",
  });
  const out = execution.allocate(queryRows * channels, "af3.atom.block");
  execution.dispatch({
    source: blockOutputSource(queryRows * channels),
    output: out,
    inputs: {
      baseTex: queriesAct, attendedTex: attention,
      transitionTex: projected, gateTex: transitionGate,
    },
    label: `af3.atom.out:${queryRows}x${channels}`,
  });
  for (const tensor of [attention, projected, transitionGate]) execution.release(tensor);
  out.label = "af3.atom.block";
  return out;
}

/** `out[i] = a[i] + b[i]` - here rather than `runtime/elementwise.js` only
 * because this file's other fused writes read the same way. */
export const sumSource = (elements) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${elements};
uniform sampler2D aTex;
uniform sampler2D bTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = fetch4(aTex, texel) + fetch4(bTex, texel);
}`;
