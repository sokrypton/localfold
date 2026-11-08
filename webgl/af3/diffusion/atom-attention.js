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
import { elementwiseOpSource } from "../../runtime/elementwise.js";
import { ONE_TARGET, PRELUDE, sized, part, memoSource } from "../../runtime/execution.js";
import { layerNorm } from "../../kernels/layer-norm.js";
import { linear } from "../../kernels/linear.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */

/**
 * `out[row][c] = sigmoid(scale[row][c]) * LayerNorm(x)[row][c] + shift[row][c]`
 *
 * The LayerNorm takes neither a scale nor an offset - the conditioning supplies
 * both - and the conditioning's own norm takes a scale with NO offset.
 */
const adaptiveBlendSourceUncached = (rows, channels) => part({ ELEMENTS: rows * channels }, {
  uniforms: ["sampler2D normalisedTex", "sampler2D scaleTex", "sampler2D shiftTex"],
  body: `  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  vec4 scale = fetch4(scaleTex, texel);
  result = (vec4(1.0) / (vec4(1.0) + exp(-scale))) * fetch4(normalisedTex, texel)
    + fetch4(shiftTex, texel);`,
});

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
const elementwiseProductSourceUncached = (elements) => elementwiseOpSource("multiply", elements);

/**
 * One of AF3's atom-layout gathers.
 *
 * 🔴 WHERE THE MASK IS FALSE THE DESTINATION IS ZERO rather than whatever index
 * 0 holds, "which matters because padded slots index 0 legitimately".
 */
const gatherSourceUncached = (count, channels) => part({ ELEMENTS: count * channels, C4: channels / 4 }, {
  uniforms: ["sampler2D sourceTex", "sampler2D indicesTex", "sampler2D maskTex"],
  body: `  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int row = texel / C4;
  if (fetch1(maskTex, row) == 0.0) { result = vec4(0.0); return; }
  int from = int(fetch1(indicesTex, row));
  result = fetch4(sourceTex, from * C4 + group);`,
});

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
const atomAttentionSourceUncached = (shape, keyMasked, lanewise = false) => {
  const { subsets, queries, keys, heads, dimension } = shape;
  const width = heads * dimension;
  // 🔴 `fetch1` IS A WHOLE `texelFetch` PLUS A COMPONENT SELECT. Both bases are
  // a row offset plus `head * D`, and `D4` below already states that four
  // divides the dimension, so four channels come in one fetch. The products
  // still accumulate in ascending `d` order, one scalar at a time, rather than
  // as a `dot`. `--no-attend-dot4` is the control arm.
  const wide = !lanewise && dimension % 4 === 0 && width % 4 === 0;
  return sized({ ELEMENTS: subsets * queries * width, QUERIES: queries, KEYS: keys, HEADS: heads, D: dimension, D4: dimension / 4, WIDTH: width }, `${PRELUDE}${ONE_TARGET}
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
${wide
    ? `    for (int dd = 0; dd < D4; ++dd) {
      vec4 q4 = fetch4(qTex, queryBase / 4 + dd);
      vec4 k4 = fetch4(kTex, keyBase / 4 + dd);
      dot += q4.x * k4.x; dot += q4.y * k4.y;
      dot += q4.z * k4.z; dot += q4.w * k4.w;
    }`
    : `    for (int d = 0; d < D; ++d) {
      dot += fetch1(qTex, queryBase + d) * fetch1(kTex, keyBase + d);
    }`}
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
}`);
};

/** `out[i] = base[i] + attended[i] + transition[i] * sigmoid(gate[i])`. */
const blockOutputSourceUncached = (elements) => part({ ELEMENTS: elements }, {
  uniforms: ["sampler2D baseTex", "sampler2D attendedTex", "sampler2D transitionTex", "sampler2D gateTex"],
  body: `  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  vec4 gate = fetch4(gateTex, texel);
  result = fetch4(baseTex, texel) + fetch4(attendedTex, texel)
    + fetch4(transitionTex, texel) * (vec4(1.0) / (vec4(1.0) + exp(-gate)));`,
});

/** `swish(wide[i]) * wide[intermediate + i]`, the SwiGLU over a blocked row. */
const swiGluSourceUncached = (rows, intermediate) => part({ ELEMENTS: rows * intermediate, INTER4: intermediate / 4 }, {
  uniforms: ["sampler2D wideTex"],
  body: `  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % INTER4;
  int row = texel / INTER4;
  int base = row * 2 * INTER4;
  vec4 gate = fetch4(wideTex, base + group);
  vec4 value = fetch4(wideTex, base + INTER4 + group);
  result = (gate / (vec4(1.0) + exp(-gate))) * value;`,
});

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
      { subsets, queries, keys, heads, dimension }, call.keyMaskedAtomAttention,
      execution.noAttendDot4),
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
const sumSourceUncached = (elements) => elementwiseOpSource("add", elements);

/**
 * The per-block attention biases, for every block of an atom stack at once.
 *
 * 🔴 ONE LAYERNORM AND ONE PROJECTION SERVE THE WHOLE STACK UNDER AF3, "which is
 * why the projection's output is (blocks, heads) rather than heads. Recomputing
 * it per block would read the same weights and give the same answer; splitting
 * it the wrong way round gives every block the biases meant for another."
 *
 * 🔴 ...AND OpenDDE NORMALISES AND PROJECTS PER BLOCK, with its own scale and
 * its own matrix each time: "`pair_input_layer_norm/scale` is [3, 16] there and
 * [16] here, and the projection [3, 16, 4] against [16, 12]. The MEAN and
 * VARIANCE are the same either way (they come from the pair, which does not
 * change), so the difference is entirely in which scale and which matrix a block
 * reads. That makes it invisible to every shape check and to any test that runs
 * one block."
 *
 * The output is one tensor a block, each already in the
 * `[subset][head][query][key]` order the attention kernel reads - because the
 * flat projection is `[subset][query][key][block][head]` and the transpose is
 * exactly the sort of thing that is easy to get backwards.
 */
const pairLogitsLayoutSourceUncached = (shape) => {
  const { subsets, queries, keys, heads, blocks, block } = shape;
  return part({ BLOCKS: blocks, ELEMENTS: subsets * heads * queries * keys, QUERIES: queries, KEYS: keys, HEADS: heads, BLOCK: block }, {
  uniforms: ["sampler2D flatTex"],
  helpers: `float logitsLayoutAt(int element) {
  int key = element % KEYS;
  int rest = element / KEYS;
  int query = rest % QUERIES;
  int rest2 = rest / QUERIES;
  int head = rest2 % HEADS;
  int subset = rest2 / HEADS;
  int source = (((subset * QUERIES + query) * KEYS) + key) * BLOCKS * HEADS
    + BLOCK * HEADS + head;
  return fetch1(flatTex, source);
}`,
  body: `  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    logitsLayoutAt(base),
    base + 1 < ELEMENTS ? logitsLayoutAt(base + 1) : 0.0,
    base + 2 < ELEMENTS ? logitsLayoutAt(base + 2) : 0.0,
    base + 3 < ELEMENTS ? logitsLayoutAt(base + 3) : 0.0);`,
});
};

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{pair: Tensor, subsets: number, queries: number, keys: number,
 *   pairChannels: number, heads: number, blocks: number, weights: any,
 *   pairNormPerBlock: boolean, epsilon?: number}} call
 * @returns {Tensor[]} one bias tensor a block, in the attention's own order
 */
export function atomPairLogits(execution, call) {
  const {
    pair, subsets, queries, keys, pairChannels, heads, blocks, weights: w,
    epsilon = 1e-5,
  } = call;
  if (typeof call.pairNormPerBlock !== "boolean") {
    throw new Error(
      "af3 atom pair logits: pairNormPerBlock has no default - AF3 normalises the"
      + " atom-pair conditioning once for the stack and OpenDDE once per block, and"
      + " the difference is which scale and which matrix a block reads");
  }
  const pairRows = subsets * queries * keys;

  // The flat `[row][blocks * heads]` projection, however it was produced.
  let flat;
  if (call.pairNormPerBlock) {
    flat = execution.allocate(pairRows * blocks * heads, "af3.atom.logits-flat");
    const perBlock = [];
    for (let block = 0; block < blocks; block += 1) {
      const normalised = layerNorm(execution, {
        x: pair, weight: w.pairInputLayerNormScales[block],
        rows: pairRows, channels: pairChannels, epsilon,
      });
      perBlock.push(linear(execution, {
        x: normalised, weight: w.pairLogitsProjections[block], rows: pairRows,
        inChannels: pairChannels, outChannels: heads, weightLayout: "in-major",
      }));
      execution.release(normalised);
    }
    execution.dispatch({
      source: interleaveBlocksSource(pairRows, blocks, heads),
      output: flat,
      inputs: Object.fromEntries(perBlock.map((t, b) => [`block${b}`, t])),
      label: `af3.atom.logits-interleave:${pairRows}x${blocks}x${heads}`,
    });
    for (const tensor of perBlock) execution.release(tensor);
  } else {
    const normalised = layerNorm(execution, {
      x: pair, weight: w.pairInputLayerNormScale,
      rows: pairRows, channels: pairChannels, epsilon,
    });
    flat = linear(execution, {
      x: normalised, weight: w.pairLogitsProjection, rows: pairRows,
      inChannels: pairChannels, outChannels: blocks * heads, weightLayout: "in-major",
    });
    execution.release(normalised);
  }

  const out = [];
  for (let block = 0; block < blocks; block += 1) {
    const perBlock = execution.allocate(
      subsets * heads * queries * keys, `af3.atom.logits${block}`);
    execution.dispatch({
      source: pairLogitsLayoutSource({ subsets, queries, keys, heads, blocks, block }),
      output: perBlock,
      inputs: { flatTex: flat },
      label: `af3.atom.logits-layout:${subsets}x${heads}x${queries}x${keys}`,
    });
    out.push(perBlock);
  }
  execution.release(flat);
  return out;
}

/**
 * `flat[row][block * heads + head] = perBlock[block][row][head]`.
 *
 * Only the per-block arm needs it: there the projections are separate tensors
 * and the attention reads one interleaved array, so the two have to meet
 * somewhere. GLSL ES forbids a dynamic sampler index, so the select is unrolled
 * - the same reason `interleavePlanesSource` in the flash kernel is.
 */
const interleaveBlocksSourceUncached = (rows, blocks, heads) => {
  const declarations = Array.from({ length: blocks },
    (_, b) => `uniform sampler2D block${b};`).join("\n");
  const select = Array.from({ length: blocks - 1 },
    (_, b) => `  if (block == ${b}) return fetch1(block${b}, row * HEADS + head);`).join("\n");
  return sized({ BLOCKS: blocks, ELEMENTS: rows * blocks * heads, HEADS: heads }, `${PRELUDE}${ONE_TARGET}
${declarations}

float at(int element) {
  int head = element % HEADS;
  int rest = element / HEADS;
  int block = rest % BLOCKS;
  int row = rest / BLOCKS;
${select}
  return fetch1(block${blocks - 1}, row * HEADS + head);
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

// Every builder above, memoised on its arguments (`memoSource` in webgl/runtime/execution.js).
export const adaptiveBlendSource = memoSource(adaptiveBlendSourceUncached);
export const elementwiseProductSource = memoSource(elementwiseProductSourceUncached);
export const gatherSource = memoSource(gatherSourceUncached);
export const atomAttentionSource = memoSource(atomAttentionSourceUncached);
export const blockOutputSource = memoSource(blockOutputSourceUncached);
export const swiGluSource = memoSource(swiGluSourceUncached);
export const sumSource = memoSource(sumSourceUncached);
export const pairLogitsLayoutSource = memoSource(pairLogitsLayoutSourceUncached);
export const interleaveBlocksSource = memoSource(interleaveBlocksSourceUncached);
