/**
 * AF3's triangle ("grid") self-attention over the pair representation.
 *
 *     normalised = LayerNorm(pair)                   AF3's FAST variance
 *     bias[h][i][j] = Linear(normalised)[i][j][h]    from the UNTRANSPOSED act
 *     act        = transpose(normalised)             for the column direction
 *     q, k, gate = Linear(act)                       transpose_weights
 *     v          = Linear(act)                       and v does not
 *     gathered   = softmax_j(q.k * scale + bias + maskPenalty) . v
 *     out        = Linear(gathered * sigmoid(gate))
 *     out        = transpose(out)                    back, for the column one
 *
 * 🔴 THE ATTENTION ITSELF IS `kernels/attention-flash.js` UNCHANGED, AND THAT IS
 * A FINDING RATHER THAN A CONVENIENCE. That kernel was written for AF2 and its
 * contract turns out to be AF3's grid attention exactly: q/k/v/gate laid out
 * `[batch][query][heads][headDim]` is `[row][i][head][d]`, its pair bias at
 * `(head * Q + q) * Q + k` is AF3's `bias[head][i][j]` to the element, and its
 * `transposeMask` is AF3's "the mask is over the KEY position, and it is the
 * transposed pair mask". So the column direction needs no second kernel on
 * either side.
 *
 * 🔴 THE BIAS COMES FROM THE UNTRANSPOSED ACTIVATION IN EVERY DIALECT, and the
 * openfold3 lineage then SWAPS ITS INDICES for the transposed direction where
 * stock AF3 does not. "Same weights, same shapes, silently different answer -
 * so the dialect has to be passed in rather than assumed." openfold3, opendde,
 * boltz2 and protenix2 swap; alphafold3 and rosettafold3 do not.
 *
 * 🔴 AND IT IS THE NORMALISED ACTIVATION THAT IS TRANSPOSED, NOT THE PAIR. AF3
 * normalises once at the top and every reader below - q, k, v, the gate - sees
 * that. "Reading the raw pair here instead costs no error: it multiplies the
 * attention inputs by the ~450x the LayerNorm was removing, and the op returns a
 * finite tensor about eighty times too large."
 *
 * 🔴 AND q, k AND THE GATE CARRY `transpose_weights` WHERE v AND THE OUTPUT DO
 * NOT. The exported shapes say so: `(heads, dimension, channels)` against
 * `(channels, heads, dimension)`. Same element count either way, so nothing in
 * the tensor distinguishes them - which is why `weightLayout` has no default in
 * this port's `linear`.
 *
 * 🔴 AND rosettafold3's GATE AND OUTPUT PROJECTION CARRY TRAINED BIASES, the
 * gate's being the one that matters: "its `to_g.bias` initialises to 1.0 against
 * a ZERO-initialised weight, so the gate is bias-DOMINATED: dropping it roughly
 * halves the gate, and this runs 96 times in an rf3 trunk." No other checkpoint
 * has a slot for either, and absent means the term does not exist.
 */
import { ONE_TARGET, PRELUDE, sized, targetDeclarations, memoSource } from "../../runtime/execution.js";
import { attentionFlash } from "../../kernels/attention-flash.js";
import { layerNorm } from "../../kernels/layer-norm.js";
import { linear } from "../../kernels/linear.js";
import { add, multiply } from "../../runtime/elementwise.js";
import { transposeLeading } from "../../runtime/transpose.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */

export const GRID_ATTENTION_WEIGHTS = [
  "actNormScale", "actNormOffset", "pairBiasProjection",
  "qProjection", "kProjection", "vProjection", "gatingQuery", "outputProjection",
];
/** ...and the two rosettafold3 alone has. */
export const GRID_ATTENTION_OPTIONAL = ["gatingQueryBias", "outputProjectionBias"];

/**
 * `[pairs][heads]` to `[heads][pairs]`, with the dialect's index swap folded in.
 *
 * The flash kernel reads its bias at `(head * Q + i) * Q + j`, which is
 * head-major - and the projection that produces it is pair-major, so the relayout
 * is a draw of its own. Doing the swap here rather than downstream keeps it next
 * to the comment that explains it.
 */
const biasPlanesSourceUncached = (length, heads, swap) => sized({ L: length, PAIRS: length * length, ELEMENTS: heads * length * length }, `${PRELUDE}${ONE_TARGET}
const int HEADS = ${heads};
uniform sampler2D rawTex;

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  vec4 value = vec4(0.0);
  // Every lane resolves its own head, because a texel straddles two head planes
  // whenever L*L is not a multiple of four.
  for (int lane = 0; lane < 4; ++lane) {
    int element = base + lane;
    if (element >= ELEMENTS) break;
    int head = element / PAIRS;
    int pair = element - head * PAIRS;
    int i = pair / L;
    int j = pair - i * L;
    ${swap ? "int source = j * L + i;" : "int source = pair;"}
    float v = fetch1(rawTex, source * HEADS + head);
    if (lane == 0) value.x = v;
    else if (lane == 1) value.y = v;
    else if (lane == 2) value.z = v;
    else value.w = v;
  }
  result = value;
}`);

/** `out[row][w] += bias[w]`, the one AF3 variant that has a projection bias. */
const addRowBiasSourceUncached = (rows, width) => sized({ ELEMENTS: rows * width }, `${PRELUDE}${ONE_TARGET}
const int W4 = ${width / 4};
uniform sampler2D xTex;
uniform sampler2D biasTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = fetch4(xTex, texel) + fetch4(biasTex, texel % W4);
}`);

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 */
/** `base[i][j] + update[j][i]`, a `[L][L][channels]` pair: the column attention's residual with its transpose folded in. */
const transposedAddSourceUncached = (length, channels) => sized({ ROWS: length * length },
  `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int L = ${length};
uniform sampler2D baseTex;
uniform sampler2D updateTex;

void main() {
  int texel = linearTexel();
  if (texel >= ROWS * C4) { result = vec4(0.0); return; }
  int row = texel / C4;
  int swapped = (row % L) * L + row / L;
  result = fetch4(baseTex, texel) + fetch4(updateTex, swapped * C4 + texel % C4);
}`);

/**
 * The fused projection `[pairs][4 x width]` split into query, key, value and gate (the logistic applied), and - for the
 * column direction - TRANSPOSED on the way: output row `i * L + j` reads the projection of row `j * L + i`. A projection
 * is per row, so projecting the untransposed activation and gathering its rows here is the same arithmetic as
 * transposing the activation first, without the `transposeLeading` draw over the whole pair.
 */
const splitGridQkvgSourceUncached = (length, width, transpose) => sized({ ROWS: length * length },
  `${PRELUDE}${targetDeclarations(4)}
const int W4 = ${width / 4};
const int L = ${length};
uniform sampler2D fusedTex;

void main() {
  int texel = linearTexel();
  if (texel >= ROWS * W4) {
    result0 = vec4(0.0); result1 = vec4(0.0); result2 = vec4(0.0); result3 = vec4(0.0);
    return;
  }
  int row = texel / W4;
  ${transpose ? "row = (row % L) * L + row / L;" : ""}
  int base = row * 4 * W4 + texel % W4;
  result0 = fetch4(fusedTex, base);
  result1 = fetch4(fusedTex, base + W4);
  result2 = fetch4(fusedTex, base + 2 * W4);
  result3 = vec4(1.0) / (vec4(1.0) + exp(-fetch4(fusedTex, base + 3 * W4)));
}`);

/**
 * The four attention projections as ONE in-major `[channels][4 x width]` weight - query, key, value, gate along the
 * output axis - and its bias (the gate's own, where the dialect has one, else zeros): what `gridSelfAttention` reads as
 * `qkvgProjection`, one projection and one split where there were four.
 *
 * 🔴 THREE OF THE FOUR ARE OUT-MAJOR AND ONE IS IN-MAJOR, so they are transposed here, once, on the host. Null when a
 * shape is not the one this expects.
 */
export function fuseGridProjections(weights) {
  const { qProjection: q, kProjection: k, vProjection: v, gatingQuery: g } = weights;
  if (![q, k, v, g].every((part) => part instanceof Float32Array && part.length === q.length)) return null;
  const width = weights.heads * weights.dimension;
  if (!Number.isInteger(width) || width % 4 !== 0 || q.length % width !== 0) return null;
  const channels = q.length / width;
  const fused = new Float32Array(channels * 4 * width);
  for (let c = 0; c < channels; c += 1) {
    const row = c * 4 * width;
    for (let o = 0; o < width; o += 1) {
      fused[row + o] = q[o * channels + c];                 // out-major [width][channels]
      fused[row + width + o] = k[o * channels + c];
      fused[row + 2 * width + o] = v[c * width + o];        // in-major [channels][width]
      fused[row + 3 * width + o] = g[o * channels + c];
    }
  }
  const bias = new Float32Array(4 * width);
  const gateBias = weights.gatingQueryBias;
  if (gateBias !== undefined && gateBias !== null) {
    if (!(gateBias instanceof Float32Array) || gateBias.length !== width) return null;
    bias.set(gateBias, 3 * width);
  }
  return { weight: fused, bias };
}

export function uploadGridAttentionWeights(execution, weights) {
  /** @type {any} */
  const out = { heads: weights.heads, dimension: weights.dimension };
  const fused = execution.noGridFusedProjection ? null : fuseGridProjections(weights);
  if (fused !== null) {
    out.qkvgProjection = execution.uploadWeight(fused.weight, "w.af3.grid.qkvg");
    out.qkvgBias = execution.uploadWeight(fused.bias, "w.af3.grid.qkvg-bias");
  }
  const fusedAway = new Set(fused === null ? [] : ["qProjection", "kProjection", "vProjection", "gatingQuery"]);
  for (const name of GRID_ATTENTION_WEIGHTS) {
    if (fusedAway.has(name)) continue;
    const values = weights[name];
    if (values === undefined) throw new Error(`af3 grid attention weights: ${name} is missing`);
    out[name] = execution.uploadWeight(values, `w.af3.grid.${name}`);
  }
  // Absent means the term does not exist, which is not the same as zero: a
  // zero gate bias is a gate at sigmoid(0), and rf3's is bias-dominated.
  for (const name of GRID_ATTENTION_OPTIONAL) {
    if (fused !== null && name === "gatingQueryBias") continue; // ...it is the fused bias's gate part
    if (weights[name] !== undefined && weights[name] !== null) {
      out[name] = execution.uploadWeight(weights[name], `w.af3.grid.${name}`);
    }
  }
  return out;
}

export function releaseGridAttentionWeights(execution, weights) {
  for (const name of [...GRID_ATTENTION_WEIGHTS, ...GRID_ATTENTION_OPTIONAL, "qkvgProjection", "qkvgBias"]) {
    if (weights[name] !== undefined) execution.release(weights[name]);
  }
}

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{pair: Tensor, mask: Tensor, weights: any, length: number, channels: number,
 *   transpose?: boolean, swapTransposedBias?: boolean, epsilon?: number}} call
 * @returns {Tensor} `length^2 * channels`, the update; the caller adds it
 */
export function gridSelfAttention(execution, call) {
  const {
    pair, mask, weights: w, length, channels,
    transpose = false, swapTransposedBias = false, epsilon = 1e-5, residualOf,
  } = call;
  const pairs = length * length;
  const heads = w.heads;
  const dimension = w.dimension;
  const width = heads * dimension;
  if (pair.elements !== pairs * channels) {
    throw new RangeError(
      `af3 grid attention: pair holds ${pair.elements} floats, not`
      + ` ${length}^2 x ${channels}`);
  }
  if (width % 4 !== 0) {
    throw new RangeError(
      `af3 grid attention: heads * dimension must be a multiple of 4, got ${width}`);
  }

  const normalised = layerNorm(execution, {
    x: pair, weight: w.actNormScale, bias: w.actNormOffset,
    rows: pairs, channels, epsilon, variance: "fast",
  });
  // The bias, from the UNTRANSPOSED activation in every dialect.
  const rawBias = linear(execution, {
    x: normalised, weight: w.pairBiasProjection, rows: pairs,
    inChannels: channels, outChannels: heads, weightLayout: "in-major",
  });
  const bias = execution.allocate(heads * pairs, "af3.grid.bias");
  execution.dispatch({
    source: biasPlanesSource(length, heads, transpose && swapTransposedBias),
    output: bias,
    inputs: { rawTex: rawBias },
    label: `af3.grid.bias:${length}x${heads}${transpose && swapTransposedBias ? ":swapped" : ""}`,
  });
  execution.release(rawBias);

  // ...and the activation itself, transposed for the column direction - except on the fused path, whose split gathers
  // the projected rows in transposed order instead (`splitGridQkvgSource`).
  const fusedPath = w.qkvgProjection !== undefined;
  const act = transpose && !fusedPath
    ? transposeLeading(execution, { x: normalised, outer: length, inner: length, channels })
    : normalised;

  const project = (weight, layout, activation) => linear(execution, {
    x: act, weight, rows: pairs, inChannels: channels, outChannels: width,
    weightLayout: layout, ...(activation === undefined ? {} : { activation }),
  });
  // ...one fused projection and a four-target split where the weights were fused at upload (`fuseGridProjections`)
  let query, key, value, gate;
  if (fusedPath) {
    const fusedOut = linear(execution, {
      x: act, weight: w.qkvgProjection, bias: w.qkvgBias, rows: pairs, inChannels: channels,
      outChannels: 4 * width, weightLayout: "in-major",
    });
    [query, key, value, gate] = ["q", "k", "v", "g"].map((part) => execution.allocate(pairs * width, `af3.grid.${part}`));
    execution.dispatch({
      source: splitGridQkvgSource(length, width, transpose),
      outputs: [query, key, value, gate],
      inputs: { fusedTex: fusedOut },
      label: `af3.grid.split-qkvg:${pairs}x${width}${transpose ? ":transposed" : ""}`,
    });
    execution.release(fusedOut);
  } else {
  query = project(w.qProjection, "out-major");
  key = project(w.kProjection, "out-major");
  value = project(w.vProjection, "in-major");
  // The gate: transpose_weights, then its optional bias, THEN the sigmoid - the
  // flash kernel multiplies by the gate it is handed, so the logistic has to be
  // applied here and cannot be folded into the projection when a bias follows.
  if (w.gatingQueryBias === undefined) {
    gate = project(w.gatingQuery, "out-major", "sigmoid");
  } else {
    const raw = project(w.gatingQuery, "out-major");
    const biased = execution.allocate(pairs * width, "af3.grid.gate-biased");
    execution.dispatch({
      source: addRowBiasSource(pairs, width),
      output: biased,
      inputs: { xTex: raw, biasTex: w.gatingQueryBias },
      label: `af3.grid.gate-bias:${pairs}x${width}`,
    });
    execution.release(raw);
    // One more draw for the logistic, which `linear` would have folded in.
    gate = execution.allocate(pairs * width, "af3.grid.gate");
    execution.dispatch({
      source: logisticSource(pairs * width),
      output: gate,
      inputs: { xTex: biased },
      label: `af3.grid.gate-logistic:${pairs}x${width}`,
    });
    execution.release(biased);
  }
  }

  // ...`attentionFlash` returns `{output, plan}`: the plan is how many draw
  // buffers the head dimension took, which this stage does not need.
  const { output: gathered } = attentionFlash(execution, {
    query, key, value, gate, mask,
    pairBias: bias,
    batch: length, queries: length, heads, headDim: dimension,
    // AF3's "the mask is over the KEY position, and it is the transposed pair
    // mask" - which is exactly this flag.
    transpose,
  });
  for (const tensor of [query, key, value, gate, bias]) execution.release(tensor);
  if (act !== normalised) execution.release(act);
  execution.release(normalised);

  let projected = linear(execution, {
    x: gathered, weight: w.outputProjection, rows: pairs,
    inChannels: width, outChannels: channels, weightLayout: "in-major",
  });
  execution.release(gathered);
  if (w.outputProjectionBias !== undefined) {
    const biased = execution.allocate(pairs * channels, "af3.grid.output-biased");
    execution.dispatch({
      source: addRowBiasSource(pairs, channels),
      output: biased,
      inputs: { xTex: projected, biasTex: w.outputProjectionBias },
      label: `af3.grid.output-bias:${pairs}x${channels}`,
    });
    execution.release(projected);
    projected = biased;
  }
  // `residualOf`: return `residualOf + update` rather than the update - which, transposed, is ONE draw reading the
  // update's rows in transposed order where it was a `transposeLeading` and an add (`transposedAddSource`).
  if (residualOf !== undefined) {
    let next;
    if (transpose) {
      next = execution.allocate(pairs * channels, "af3.grid.residual");
      execution.dispatch({
        source: transposedAddSource(length, channels),
        output: next,
        inputs: { baseTex: residualOf, updateTex: projected },
        label: `af3.grid.transposed-add:${pairs}x${channels}`,
      });
    } else {
      next = add(execution, residualOf, projected);
    }
    execution.release(projected);
    return next;
  }
  if (!transpose) {
    projected.label = "af3.grid";
    return projected;
  }
  // ...and back, so the residual lands on the orientation it came from.
  const restored = transposeLeading(
    execution, { x: projected, outer: length, inner: length, channels });
  execution.release(projected);
  restored.label = "af3.grid";
  return restored;
}

/** `sigmoid(x)`, needed only where a bias sits between the projection and it. */
const logisticSourceUncached = (elements) => sized({ ELEMENTS: elements }, `${PRELUDE}${ONE_TARGET}
uniform sampler2D xTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  vec4 x = fetch4(xTex, texel);
  result = vec4(1.0) / (vec4(1.0) + exp(-x));
}`);

// Every builder above, memoised on its arguments (`memoSource` in webgl/runtime/execution.js).
export const biasPlanesSource = memoSource(biasPlanesSourceUncached);
export const splitGridQkvgSource = memoSource(splitGridQkvgSourceUncached);
export const transposedAddSource = memoSource(transposedAddSourceUncached);
export const addRowBiasSource = memoSource(addRowBiasSourceUncached);
export const logisticSource = memoSource(logisticSourceUncached);
