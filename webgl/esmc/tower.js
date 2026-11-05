/**
 * ESM-C's tower and ESMFold2's language-model shim, in WebGL2.
 *
 *     h    = LN(x; attn_norm) @ qkv            -> [q | k | v]
 *     q, k = LN(q; q_norm), LN(k; k_norm)      // over the FULL d_model
 *     q, k = RoPE(q, k)                        // head_dim 64, base 10000
 *     x    = x + attn(q, k, v) @ attn_out / residualScale
 *     x    = x + swiglu(LN(x; ffn_norm) @ fc1) @ fc2 / residualScale
 *
 * 🔴 THE QK LayerNorm IS OVER THE WHOLE d_model, NOT PER HEAD. It runs on the
 * 1152-wide q and k BEFORE they are split into 18 heads of 64, "so normalising
 * per head is a different model that still runs, still folds, and is wrong by an
 * amount no shape check can see".
 *
 * 🔴 AND RoPE HERE IS THE SPLIT-HALVES CONVENTION, not the interleaved one:
 * `[x0..x31, x32..x63]` rotates as halves against each other, where interleaving
 * adjacent pairs is the other common reading and is also silent.
 *
 * 🔴 AND THE RESIDUAL SCALE IS DERIVED FROM THE DEPTH, `sqrt(layers / 36)`,
 * because "a manifest's figure was 1.0 for every tower, and 300M's is 0.9129".
 * It divides both residual branches; dropping it leaves a tower that runs and a
 * representation about a tenth too large at 300M.
 *
 * 🔴 AND ONLY THE LAST STATE IS FINAL-NORMED. ESMFold2 mixes all 37 - the
 * embedding plus 36 blocks - and the last one is post the stack's final
 * LayerNorm while the other 36 are the raw residual stream. "Returning the
 * pre-norm value there reads corr 0.909 against native on that layer where every
 * other layer is >= 0.9998."
 *
 * 🔴 AND THE 37 STATES NEVER HAVE TO EXIST AT ONCE. The mix is constant and the
 * shim's norm and projection are shared across k, so the sum is a running
 * accumulator - which is what this port does, because holding a
 * (37, tokens, 1152) tensor is 37 textures where one will do. The CPU reference
 * materialises them because a reference should be obvious.
 *
 * 🔴 AND THE ATTENTION HAS ITS OWN KERNEL RATHER THAN AF2's. `attention-flash.js`
 * serves four AF3 stages unchanged, and its contract carries a mask, a pair bias
 * and a GATE; ESM-C has none of the three, and a ones-gate plus a ones-mask is
 * two textures bound to say nothing. The arithmetic here is 30 lines without
 * them.
 *
 * 🔴 AND GELU IS THE erf FORM, NOT THE tanh ONE. The reference uses
 * Abramowitz-Stegun 7.1.26 (about 1.5e-7), and the tanh approximation differs by
 * ~1e-3 - which is three orders above any bar here, so the series is transcribed
 * rather than approximated.
 */
import { ONE_TARGET, PRELUDE } from "../runtime/execution.js";
import { layerNorm } from "../kernels/layer-norm.js";
import { linear } from "../kernels/linear.js";

/** @typedef {import("../runtime/storage.js").Tensor} Tensor */

/** ESM-C's alphabet, in id order; 0/1/2/32 are BOS/PAD/EOS/MASK. */
export const VOCAB = ("<cls> <pad> <eos> <unk> L A G V S E R T I D P K Q N F Y "
  + "M H W C X B U Z O . - | <mask>").split(" ");
const TOKEN = new Map(VOCAB.map((name, index) => [name, index]));
export const BOS = 0;
export const PAD = 1;
export const EOS = 2;
export const UNK = /** @type {number} */ (TOKEN.get("<unk>"));
export const MASK = 32;

/** One-letter sequence -> token ids, with BOS and EOS attached. */
export function sequenceIds(sequence) {
  const ids = new Int32Array(sequence.length + 2);
  ids[0] = BOS;
  for (let i = 0; i < sequence.length; i += 1) {
    const code = TOKEN.get(sequence[i].toUpperCase());
    ids[i + 1] = code === undefined ? UNK : code;
  }
  ids[ids.length - 1] = EOS;
  return ids;
}

/** The architecture's residual scale, from the DEPTH and not from a manifest. */
export function residualScaleFor(dims) {
  return Number.isInteger(dims.layers) && dims.layers > 0
    ? Math.sqrt(dims.layers / 36) : (dims.residualScale ?? 1);
}

/** `out[row][c] = table[ids[row]][c]`. */
export const embedSource = (rows, model) => `${PRELUDE}${ONE_TARGET}
const int C4 = ${model / 4};
const int ELEMENTS = ${rows * model};
uniform sampler2D tableTex;
uniform sampler2D idsTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int row = texel / C4;
  int id = int(fetch1(idsTex, row));
  result = fetch4(tableTex, id * C4 + group);
}`;

/** One of the three `[q | k | v]` segments of a `3 * model` projection. */
export const qkvSliceSource = (rows, model, which) => `${PRELUDE}${ONE_TARGET}
const int C4 = ${model / 4};
const int ELEMENTS = ${rows * model};
const int OFFSET = ${which * (model / 4)};
uniform sampler2D projectedTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int row = texel / C4;
  result = fetch4(projectedTex, row * 3 * C4 + OFFSET + group);
}`;

/**
 * Rotary embedding on a `(rows, heads, headDim)` tensor, split halves.
 *
 * Each fragment owns four consecutive channels of one head, and all four are on
 * the same side of the split because `headDim / 2` is a multiple of four for
 * every ESM-C width - which the caller asserts.
 */
export const ropeSource = (rows, heads, headDim, base) => `${PRELUDE}${ONE_TARGET}
const int HEAD_DIM = ${headDim};
const int HALF = ${headDim >> 1};
const int ELEMENTS = ${rows * heads * headDim};
const float BASE = ${base.toFixed(1)};
uniform sampler2D xTex;

float at(int element) {
  int d = element % HEAD_DIM;
  int row = element / (HEAD_DIM * ${heads});
  bool low = d < HALF;
  int pairIndex = low ? d : d - HALF;
  float frequency = float(row) / pow(BASE, (2.0 * float(pairIndex)) / float(HEAD_DIM));
  float c = cos(frequency);
  float s = sin(frequency);
  float first = fetch1(xTex, element + (low ? 0 : -HALF));
  float second = fetch1(xTex, element + (low ? HALF : 0));
  return low ? first * c - second * s : first * s + second * c;
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    at(base),
    base + 1 < ELEMENTS ? at(base + 1) : 0.0,
    base + 2 < ELEMENTS ? at(base + 2) : 0.0,
    base + 3 < ELEMENTS ? at(base + 3) : 0.0);
}`;

/**
 * Full self-attention over `rows`, no mask, no bias and no gate.
 *
 * q/k/v are `(rows, heads, headDim)` and the output is `(rows, heads * headDim)`
 * - the same buffer shape, which is why the projection after it reads it
 * directly.
 */
export const attentionSource = (rows, heads, headDim) => `${PRELUDE}${ONE_TARGET}
const int ROWS = ${rows};
const int HEADS = ${heads};
const int HEAD_DIM = ${headDim};
const int WIDTH = ${heads * headDim};
const int ELEMENTS = ${rows * heads * headDim};
const float SCALE = ${(1 / Math.sqrt(headDim)).toFixed(10)};
uniform sampler2D qTex;
uniform sampler2D kTex;
uniform sampler2D vTex;

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  // Four consecutive channels of one (row, head), so they share one softmax.
  int head = (base % WIDTH) / HEAD_DIM;
  int row = base / WIDTH;
  int qBase = (row * HEADS + head) * HEAD_DIM;

  float largest = -1e30;
  for (int j = 0; j < ROWS; ++j) {
    int kBase = (j * HEADS + head) * HEAD_DIM;
    float dot = 0.0;
    for (int d = 0; d < HEAD_DIM; ++d) {
      dot += fetch1(qTex, qBase + d) * fetch1(kTex, kBase + d);
    }
    largest = max(largest, dot * SCALE);
  }
  float total = 0.0;
  vec4 gathered = vec4(0.0);
  for (int j = 0; j < ROWS; ++j) {
    int kBase = (j * HEADS + head) * HEAD_DIM;
    float dot = 0.0;
    for (int d = 0; d < HEAD_DIM; ++d) {
      dot += fetch1(qTex, qBase + d) * fetch1(kTex, kBase + d);
    }
    float weight = exp(dot * SCALE - largest);
    total += weight;
    // Both terms are multiples of four - kBase steps by HEAD_DIM and the
    // fragment owns four aligned channels - so this is one texel of v.
    gathered += weight * fetch4(vTex, (kBase + base % HEAD_DIM) / 4);
  }
  result = gathered / total;
}`;

/** `out[row][c] = silu(wide[row][c]) * wide[row][ffn + c]`, the BLOCKED split. */
export const swiGluSource = (rows, ffn) => `${PRELUDE}${ONE_TARGET}
const int F4 = ${ffn / 4};
const int ELEMENTS = ${rows * ffn};
uniform sampler2D wideTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % F4;
  int row = texel / F4;
  vec4 gate = fetch4(wideTex, row * 2 * F4 + group);
  vec4 value = fetch4(wideTex, row * 2 * F4 + F4 + group);
  result = (gate / (vec4(1.0) + exp(-gate))) * value;
}`;

/** `out = a + b * factor`, which is both residual branches and the shim's sum. */
export const addScaledSource = (elements) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${elements};
uniform sampler2D aTex;
uniform sampler2D bTex;
uniform float factor;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = fetch4(aTex, texel) + factor * fetch4(bTex, texel);
}`;

/** `out[i][j][c] = single[i][c] * single[j][c]`, then `single[i][c] - single[j][c]`. */
export const pairJoinSource = (rows, pair) => `${PRELUDE}${ONE_TARGET}
const int ROWS = ${rows};
const int P4 = ${pair / 4};
const int WIDE4 = ${pair / 2};
const int ELEMENTS = ${rows * rows * 2 * pair};
uniform sampler2D singleTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % WIDE4;
  int cell = texel / WIDE4;
  int i = cell / ROWS;
  int j = cell - i * ROWS;
  bool product = group < P4;
  int channelGroup = product ? group : group - P4;
  vec4 a = fetch4(singleTex, i * P4 + channelGroup);
  vec4 b = fetch4(singleTex, j * P4 + channelGroup);
  result = product ? a * b : a - b;
}`;

/**
 * `out = gelu(x + bias)`, with GELU as the reference's erf series.
 *
 * Abramowitz-Stegun 7.1.26: about 1.5e-7, which is below every bar here. The
 * tanh approximation is ~1e-3 away and is a different function.
 */
export const geluBiasSource = (rows, channels) => `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int ELEMENTS = ${rows * channels};
uniform sampler2D xTex;
uniform sampler2D biasTex;

float erfOf(float x) {
  float sign = x < 0.0 ? -1.0 : 1.0;
  float z = abs(x);
  float t = 1.0 / (1.0 + 0.3275911 * z);
  float series = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
    + t * (-1.453152027 + t * 1.061405429))));
  return sign * (1.0 - series * exp(-z * z));
}

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  vec4 x = fetch4(xTex, texel) + fetch4(biasTex, texel % C4);
  result = vec4(
    0.5 * x.x * (1.0 + erfOf(x.x * 0.70710678118654752440)),
    0.5 * x.y * (1.0 + erfOf(x.y * 0.70710678118654752440)),
    0.5 * x.z * (1.0 + erfOf(x.z * 0.70710678118654752440)),
    0.5 * x.w * (1.0 + erfOf(x.w * 0.70710678118654752440)));
}`;

/**
 * `out[row][c] = x[row + from][c]`, which drops the BOS and EOS rows.
 *
 * 🔴 THE SHIM'S SINGLE IS THE SEQUENCE'S ROWS AND NOT THE TOWER'S. ESM-C runs
 * with BOS and EOS attached and ESMFold2 folds what is between them, so the
 * reference mixes `state.subarray(model, length - model)` at `rows - 2`. Every
 * step between the states and the single is per ROW - two LayerNorms, two
 * projections and a scaled sum - so dropping the two rows at the END computes
 * the same numbers for the rows that are kept, and keeps the mix one draw a
 * state instead of one draw a state per row.
 */
export const sliceRowsSource = (rows, channels, from) => `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int ELEMENTS = ${rows * channels};
const int FROM = ${from};
uniform sampler2D xTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int row = texel / C4;
  result = fetch4(xTex, (row + FROM) * C4 + group);
}`;

/** `out[row][c] = x[row][c] + bias[c]`. */
export const addRowBiasSource = (rows, channels) => `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int ELEMENTS = ${rows * channels};
uniform sampler2D xTex;
uniform sampler2D biasTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = fetch4(xTex, texel) + fetch4(biasTex, texel % C4);
}`;

/**
 * One ESM-C block.
 *
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{x: Tensor, rows: number, weights: any,
 *   dims: {model: number, heads: number, ffn: number, layers?: number,
 *     residualScale?: number},
 *   layer: number, epsilon?: number,
 *   onStage?: (stage: string, tensor: Tensor) => void}} call
 * @returns {Tensor} `rows * model`, the caller's to release
 */
export function esmcBlock(execution, call) {
  const { x, rows, weights, dims, layer, epsilon = 1e-5 } = call;
  const { model, heads, ffn } = dims;
  const headDim = model / heads;
  if (headDim % 8 !== 0) {
    throw new RangeError(
      `esmc: head_dim ${headDim} - this port's RoPE kernel owns four channels a`
      + " fragment and needs both halves aligned to four, so head_dim must be a"
      + " multiple of eight");
  }
  const inverseScale = 1 / residualScaleFor(dims);
  const at = (leaf) => {
    const tensor = weights[`blocks/${layer}/${leaf}`];
    if (tensor === undefined) {
      throw new Error(`esmc block ${layer}: blocks/${layer}/${leaf} is missing`);
    }
    return tensor;
  };

  const normed = layerNorm(execution, {
    x, weight: at("attn_norm/scale"), bias: at("attn_norm/offset"),
    rows, channels: model, epsilon,
  });
  call.onStage?.("attn-normed", normed);
  const projected = linear(execution, {
    x: normed, weight: at("qkv/weights"), rows,
    inChannels: model, outChannels: 3 * model, weightLayout: "in-major",
  });
  execution.release(normed);

  const slice = (which, label) => {
    const out = execution.allocate(rows * model, `esmc.${label}`);
    execution.dispatch({
      source: qkvSliceSource(rows, model, which),
      output: out,
      inputs: { projectedTex: projected },
      label: `esmc.qkv-slice:${rows}x${model}:${label}`,
    });
    return out;
  };
  const rawQuery = slice(0, "q");
  const rawKey = slice(1, "k");
  const value = slice(2, "v");
  execution.release(projected);

  // Over the FULL width, before the heads exist. No offset on either.
  const normQuery = layerNorm(execution, {
    x: rawQuery, weight: at("q_norm/scale"), rows, channels: model, epsilon,
  });
  const normKey = layerNorm(execution, {
    x: rawKey, weight: at("k_norm/scale"), rows, channels: model, epsilon,
  });
  execution.release(rawQuery);
  execution.release(rawKey);
  const rope = (input, label) => {
    const out = execution.allocate(rows * model, `esmc.${label}`);
    execution.dispatch({
      source: ropeSource(rows, heads, headDim, 10000),
      output: out,
      inputs: { xTex: input },
      label: `esmc.rope:${rows}x${heads}x${headDim}`,
    });
    return out;
  };
  const query = rope(normQuery, "q-rope");
  const key = rope(normKey, "k-rope");
  execution.release(normQuery);
  execution.release(normKey);
  call.onStage?.("query", query);
  call.onStage?.("key", key);
  call.onStage?.("value", value);

  const gathered = execution.allocate(rows * model, "esmc.attention");
  execution.dispatch({
    source: attentionSource(rows, heads, headDim),
    output: gathered,
    inputs: { qTex: query, kTex: key, vTex: value },
    label: `esmc.attend:${rows}x${heads}x${headDim}`,
  });
  for (const tensor of [query, key, value]) execution.release(tensor);
  const attention = linear(execution, {
    x: gathered, weight: at("attn_out/weights"), rows,
    inChannels: model, outChannels: model, weightLayout: "in-major",
  });
  execution.release(gathered);

  const afterAttention = execution.allocate(rows * model, "esmc.after-attention");
  execution.dispatch({
    source: addScaledSource(rows * model),
    output: afterAttention,
    inputs: { aTex: x, bTex: attention },
    floats: { factor: inverseScale },
    label: `esmc.attn-residual:${rows}x${model}`,
  });
  execution.release(attention);
  call.onStage?.("after-attention", afterAttention);

  const ffnNormed = layerNorm(execution, {
    x: afterAttention, weight: at("ffn_norm/scale"), bias: at("ffn_norm/offset"),
    rows, channels: model, epsilon,
  });
  call.onStage?.("ffn-normed", ffnNormed);
  const wide = linear(execution, {
    x: ffnNormed, weight: at("fc1/weights"), rows,
    inChannels: model, outChannels: 2 * ffn, weightLayout: "in-major",
  });
  execution.release(ffnNormed);
  const gated = execution.allocate(rows * ffn, "esmc.swiglu");
  execution.dispatch({
    source: swiGluSource(rows, ffn),
    output: gated,
    inputs: { wideTex: wide },
    label: `esmc.swiglu:${rows}x${ffn}`,
  });
  execution.release(wide);
  const projectedOut = linear(execution, {
    x: gated, weight: at("fc2/weights"), rows,
    inChannels: ffn, outChannels: model, weightLayout: "in-major",
  });
  execution.release(gated);

  const out = execution.allocate(rows * model, "esmc.block");
  execution.dispatch({
    source: addScaledSource(rows * model),
    output: out,
    inputs: { aTex: afterAttention, bTex: projectedOut },
    floats: { factor: inverseScale },
    label: `esmc.ffn-residual:${rows}x${model}`,
  });
  execution.release(afterAttention);
  execution.release(projectedOut);
  out.label = `esmc.block.${layer}`;
  return out;
}

/** softmax over the mix logits; a constant, so it folds once on the host. */
export function layerMix(combine) {
  let largest = -Infinity;
  for (const value of combine) if (value > largest) largest = value;
  const out = new Float32Array(combine.length);
  let total = 0;
  for (let i = 0; i < combine.length; i += 1) {
    out[i] = Math.exp(combine[i] - largest);
    total += out[i];
  }
  for (let i = 0; i < out.length; i += 1) out[i] /= total;
  return out;
}

/**
 * The tower and the shim in ONE pass, accumulating as it goes.
 *
 * 🔴 THE STATES ARE MIXED AS THEY ARE PRODUCED, which is the whole reason this
 * is one function: the mix is a constant and the shim's norm and projection are
 * shared across k, so state k can be normed, projected, scaled and added the
 * moment the block that made it returns - and then released. Holding 37 of them
 * is 37 textures of `rows * 1152` where one accumulator will do.
 *
 * 🔴 AND THE LAST STATE IS FINAL-NORMED BEFORE IT IS MIXED, not after.
 *
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{ids: Int32Array, weights: any,
 *   dims: {model: number, heads: number, ffn: number, layers: number,
 *     pair: number, residualScale?: number},
 *   combine: Float32Array, blocks?: number, stripTerminals?: boolean,
 *   epsilon?: number,
 *   onState?: (index: number, tensor: Tensor) => void,
 *   onStage?: (stage: string, tensor: Tensor) => void}} call
 *   `blocks` runs a PREFIX of the tower, for a gate that bisects; the mix is
 *   still normalised over every logit the bundle carries, as the reference's is.
 * @returns {{single: Tensor, lastState: Tensor}} both the caller's to release
 */
export function towerSingle(execution, call) {
  const { ids, weights, dims, epsilon = 1e-5 } = call;
  const { model, pair } = dims;
  const rows = ids.length;
  const layers = call.blocks ?? dims.layers;
  // 🔴 THE MIX IS A HOST ARRAY AND NOT A TEXTURE. It is 37 numbers, constant for
  // a bundle, and every one of them is a scalar uniform in the draw that uses
  // it - uploading them would be a texture fetch to read a number the host
  // already has. `combine` is the bundle's raw logits; the softmax is here.
  if (!(call.combine instanceof Float32Array)) {
    throw new Error(
      "esmc tower: `combine` is the bundle's lm/combine logits as a HOST array -"
      + " the mix is a constant and each weight is a uniform");
  }
  const mix = layerMix(call.combine);
  const idsTensor = execution.upload(Float32Array.from(ids), "esmc.ids");
  let state = execution.allocate(rows * model, "esmc.embedded");
  execution.dispatch({
    source: embedSource(rows, model),
    output: state,
    inputs: { tableTex: weights["embed/weights"], idsTex: idsTensor },
    label: `esmc.embed:${rows}x${model}`,
  });
  execution.release(idsTensor);

  // Starts at zero, which `Storage.allocate` guarantees.
  let accumulated = execution.allocate(rows * pair, "esmc.shim.accumulator");
  /** Norm, project, scale and add one state - then it can go. */
  const mixIn = (values, index, isLast) => {
    const source = isLast
      ? layerNorm(execution, {
        x: values, weight: weights["final_norm/scale"],
        rows, channels: model, epsilon,
      })
      : values;
    const normed = layerNorm(execution, {
      x: source, weight: weights["lm/norm/scale"], bias: weights["lm/norm/offset"],
      rows, channels: model, epsilon,
    });
    const projected = linear(execution, {
      x: normed, weight: weights["lm/projection/weights"], rows,
      inChannels: model, outChannels: pair, weightLayout: "in-major",
    });
    execution.release(normed);
    const next = execution.allocate(rows * pair, "esmc.shim.accumulator");
    execution.dispatch({
      source: addScaledSource(rows * pair),
      output: next,
      inputs: { aTex: accumulated, bTex: projected },
      floats: { factor: mix[index] },
      label: `esmc.shim.mix:${rows}x${pair}:${index}`,
    });
    execution.release(projected);
    if (source !== values) execution.release(source);
    return next;
  };
  call.onState?.(0, state);
  {
    const previous = accumulated;
    accumulated = mixIn(state, 0, layers === 0);
    execution.release(previous);
  }
  for (let layer = 0; layer < layers; layer += 1) {
    const next = esmcBlock(execution, {
      x: state, rows, weights, dims, layer, epsilon,
      ...(layer === 0 ? { onStage: call.onStage } : {}),
    });
    execution.release(state);
    state = next;
    call.onState?.(layer + 1, state);
    const previous = accumulated;
    accumulated = mixIn(state, layer + 1, layer === layers - 1);
    execution.release(previous);
  }

  // The downprojection is affine and sits OUTSIDE the sum: its bias must not be
  // added once per state.
  const down = linear(execution, {
    x: accumulated, weight: weights["lm/downproject/weights"], rows,
    inChannels: pair, outChannels: pair, weightLayout: "in-major",
  });
  execution.release(accumulated);
  const biased = execution.allocate(rows * pair, "esmc.shim.biased");
  execution.dispatch({
    source: addRowBiasSource(rows, pair),
    output: biased,
    inputs: { xTex: down, biasTex: weights["lm/downproject/bias"] },
    label: `esmc.shim.down-bias:${rows}x${pair}`,
  });
  execution.release(down);
  let single = biased;
  if (call.stripTerminals === true) {
    single = execution.allocate((rows - 2) * pair, "esmc.shim.single");
    execution.dispatch({
      source: sliceRowsSource(rows - 2, pair, 1),
      output: single,
      inputs: { xTex: biased },
      label: `esmc.shim.strip:${rows - 2}x${pair}`,
    });
    execution.release(biased);
  }
  single.label = "esmc.shim.single";
  return { single, lastState: state };
}

/**
 * `(rows, pair)` -> `(rows, rows, pair)`: the outer product carries BOTH a
 * product and a difference, so the pair representation sees magnitude and
 * direction.
 *
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{single: Tensor, rows: number, weights: any, pair: number,
 *   epsilon?: number}} call
 * @returns {Tensor} `rows^2 * pair`
 */
export function shimPair(execution, call) {
  const { single, rows, weights, pair, epsilon = 1e-5 } = call;
  const cells = rows * rows;
  const joined = execution.allocate(cells * 2 * pair, "esmc.shim.joined");
  execution.dispatch({
    source: pairJoinSource(rows, pair),
    output: joined,
    inputs: { singleTex: single },
    label: `esmc.shim.join:${rows}x${pair}`,
  });
  const first = linear(execution, {
    x: joined, weight: weights["lm/pair_mlp_1/weights"], rows: cells,
    inChannels: 2 * pair, outChannels: pair, weightLayout: "in-major",
  });
  execution.release(joined);
  const activated = execution.allocate(cells * pair, "esmc.shim.gelu");
  execution.dispatch({
    source: geluBiasSource(cells, pair),
    output: activated,
    inputs: { xTex: first, biasTex: weights["lm/pair_mlp_1/bias"] },
    label: `esmc.shim.gelu:${cells}x${pair}`,
  });
  execution.release(first);
  const second = linear(execution, {
    x: activated, weight: weights["lm/pair_mlp_2/weights"], rows: cells,
    inChannels: pair, outChannels: pair, weightLayout: "in-major",
  });
  execution.release(activated);
  const biased = execution.allocate(cells * pair, "esmc.shim.biased");
  execution.dispatch({
    source: addRowBiasSource(cells, pair),
    output: biased,
    inputs: { xTex: second, biasTex: weights["lm/pair_mlp_2/bias"] },
    label: `esmc.shim.pair-bias:${cells}x${pair}`,
  });
  execution.release(second);
  const out = layerNorm(execution, {
    x: biased, weight: weights["lm/pair_norm/scale"], bias: weights["lm/pair_norm/offset"],
    rows: cells, channels: pair, epsilon,
  });
  execution.release(biased);
  out.label = "esmc.shim.pair";
  return out;
}
