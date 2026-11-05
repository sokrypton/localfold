/**
 * ESMFold2's diffusion head in WebGL2: conditioning, twelve token blocks, and
 * one EDM denoise step.
 *
 *     z = z_proj(z_input_norm([z_trunk | rel_pos]))    512 -> 256
 *     z = z + transition(z), twice
 *     s = s_proj(s_input_norm(s_inputs))               451 -> 768
 *     t = 0.25 * log(max(t_hat / sigma_data, 1e-20))
 *     s = s + noise_proj(noise_norm(cos(2*pi*(t*w + b))))
 *     s = s + transition(s), twice
 *
 * 🔴 THERE IS NO `s_trunk` ANYWHERE IN THIS MODEL. "The conditioning takes one
 * and it is handed None: the trunk carries no single track, so the only single
 * representation in the whole graph is `s_inputs`." An AF3-shaped port that
 * reaches for a trunk single will not find one, and synthesising a zero would be
 * a different model.
 *
 * 🔴 AND `z` IS CACHED ACROSS THE SAMPLER'S STEPS WHILE `s` IS NOT. Only `s`
 * depends on the noise level. "Caching the pair and the single together would
 * freeze the noise level at step zero - eleven steps that all think they are the
 * first, which still converges to something."
 *
 * 🔴 AND THE PAIR IS CONCATENATED WITH THE RELATIVE ENCODING, NOT ADDED. The two
 * are the same shape and adding them conforms; the norm is 2 * c_z wide, which
 * is what says so.
 *
 * 🔴 AND THE TWO TRANSITIONS IN THIS FILE ARE PACKED THE TWO DIFFERENT WAYS. The
 * conditioning's `TransitionLayer` has `a_proj` and `b_proj` as two separate
 * Linears - `out_proj(silu(a) * b)`, so `a` is the gate - and the denoiser's
 * `lin_swish` is ONE Linear of `2 * hidden` split in half, gate first. "Both are
 * 'a SwiGLU transition in the diffusion module' and they are packed the two
 * different ways"; reading either as the other indexes one matrix at half its
 * stride.
 *
 * 🔴 AND adaLN's TWO LayerNorms ARE NOT THE SAME KIND: the activation's is
 * affine-FREE and the conditioning's has a learned SCALE and no bias. "There is
 * one weight vector between them and it belongs to `s`."
 *
 * 🔴 AND THE ATTENTION HAS TWO GATES, FROM DIFFERENT THINGS: `g_proj` gates the
 * per-head context from the ADALN-MODULATED activation, and `out_gate` gates the
 * whole output from the CONDITIONING SINGLE - and only the second has a bias,
 * "initialised to -2 upstream so a fresh block starts nearly closed".
 *
 * 🔴 AND THE LAST LINE OF A STEP IS THE EDM PRECONDITIONING, NOT A RESIDUAL:
 * `sigma^2/(sigma^2+t^2) * x_noisy + sigma*t/sqrt(sigma^2+t^2) * r`. "Writing it
 * as `x_noisy + r` runs, converges to something, and is a different sampler."
 */
import { ONE_TARGET, PRELUDE } from "../runtime/execution.js";
import { layerNorm } from "../kernels/layer-norm.js";
import { linear } from "../kernels/linear.js";
import { swiGluSource } from "../esmc/tower.js";
import { sumSource } from "./atom-transformer.js";

/** @typedef {import("../runtime/storage.js").Tensor} Tensor */

export const SIGMA_DATA = 16;

/** `[left | right]`, two tensors of the same width. */
export const concatenateSource = (rows, channels) => `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int ELEMENTS = ${rows * channels * 2};
uniform sampler2D leftTex;
uniform sampler2D rightTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % (2 * C4);
  int row = texel / (2 * C4);
  result = group < C4
    ? fetch4(leftTex, row * C4 + group)
    : fetch4(rightTex, row * C4 + group - C4);
}`;

/** `out = silu(a) * b`, the conditioning transition's SEPARATE projections. */
export const gatedProductSource = (elements) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${elements};
uniform sampler2D aTex;
uniform sampler2D bTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  vec4 a = fetch4(aTex, texel);
  result = (a / (vec4(1.0) + exp(-a))) * fetch4(bTex, texel);
}`;

/** `out[row][c] = x[row][c] + row0[c]`: one noise level for the whole structure. */
export const addBroadcastRowSource = (rows, channels) => `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int ELEMENTS = ${rows * channels};
uniform sampler2D xTex;
uniform sampler2D rowTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = fetch4(xTex, texel) + fetch4(rowTex, texel % C4);
}`;

/**
 * `out = sigmoid(gate + gateBias) * rms(activation) + shift`, adaLN-Zero.
 *
 * The activation's norm is affine-free and is computed here rather than through
 * `kernels/layer-norm.js` so that the whole modulation is one draw: the gate and
 * the shift are already per element.
 */
export const adaptiveSource = (rows, channels) => `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int CHANNELS = ${channels};
const int ELEMENTS = ${rows * channels};
uniform sampler2D activationTex;
uniform sampler2D gateTex;
uniform sampler2D shiftTex;
uniform sampler2D gateBiasTex;
uniform float epsilon;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int row = texel / C4;
  float total = 0.0;
  for (int i = 0; i < C4; ++i) {
    vec4 v = fetch4(activationTex, row * C4 + i);
    total += v.x + v.y + v.z + v.w;
  }
  float mean = total / float(CHANNELS);
  float spread = 0.0;
  for (int i = 0; i < C4; ++i) {
    vec4 centred = fetch4(activationTex, row * C4 + i) - mean;
    spread += dot(centred, centred);
  }
  float inverse = inversesqrt(spread / float(CHANNELS) + epsilon);
  vec4 normalised = (fetch4(activationTex, texel) - vec4(mean)) * inverse;
  vec4 gate = fetch4(gateTex, texel) + fetch4(gateBiasTex, group);
  result = normalised / (vec4(1.0) + exp(-gate)) + fetch4(shiftTex, texel);
}`;

/** `out = x * sigmoid(gate + bias)`, the output gate from the conditioning. */
export const gateWithBiasSource = (rows, channels) => `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int ELEMENTS = ${rows * channels};
uniform sampler2D xTex;
uniform sampler2D gateTex;
uniform sampler2D biasTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  vec4 gate = fetch4(gateTex, texel) + fetch4(biasTex, texel % C4);
  result = fetch4(xTex, texel) / (vec4(1.0) + exp(-gate));
}`;

/** `out = x * sigmoid(gate)`, the per-head context gate. */
export const gateSource = (elements) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${elements};
uniform sampler2D xTex;
uniform sampler2D gateTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  vec4 gate = fetch4(gateTex, texel);
  result = fetch4(xTex, texel) / (vec4(1.0) + exp(-gate));
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
 * Attention over tokens, biased per (i, j, head) by the pair representation.
 *
 * 🔴 THE SOFTMAX IS OVER THE KEY AXIS of an (i, j, head) tensor - `dim=-2`
 * upstream and not the last axis. "A softmax over the heads instead sums to one
 * and returns a plausible tensor."
 *
 * 🔴 AND THE VALUE IS THE SECOND HALF OF kv, so it starts a whole `channels`
 * further along the row.
 */
export const attentionSource = (tokens, channels, heads) => `${PRELUDE}${ONE_TARGET}
const int TOKENS = ${tokens};
const int CHANNELS = ${channels};
const int HEADS = ${heads};
const int HEAD_DIM = ${channels / heads};
const int ELEMENTS = ${tokens * channels};
const float SCALE = ${(1 / Math.sqrt(channels / heads)).toFixed(10)};
uniform sampler2D queryTex;
uniform sampler2D kvTex;
uniform sampler2D biasTex;

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  int i = base / CHANNELS;
  int c = base - i * CHANNELS;
  int head = c / HEAD_DIM;
  int queryBase = i * CHANNELS + head * HEAD_DIM;

  float largest = -1e30;
  for (int j = 0; j < TOKENS; ++j) {
    int keyBase = j * CHANNELS * 2 + head * HEAD_DIM;
    float total = 0.0;
    for (int d = 0; d < HEAD_DIM; ++d) {
      total += fetch1(queryTex, queryBase + d) * fetch1(kvTex, keyBase + d);
    }
    largest = max(largest, total * SCALE + fetch1(biasTex, (i * TOKENS + j) * HEADS + head));
  }
  float sum = 0.0;
  vec4 gathered = vec4(0.0);
  for (int j = 0; j < TOKENS; ++j) {
    int keyBase = j * CHANNELS * 2 + head * HEAD_DIM;
    float total = 0.0;
    for (int d = 0; d < HEAD_DIM; ++d) {
      total += fetch1(queryTex, queryBase + d) * fetch1(kvTex, keyBase + d);
    }
    float weight = exp(total * SCALE
      + fetch1(biasTex, (i * TOKENS + j) * HEADS + head) - largest);
    sum += weight;
    // ...the value half: a whole CHANNELS further along the row.
    int valueBase = j * CHANNELS * 2 + CHANNELS + head * HEAD_DIM;
    gathered += weight * fetch4(kvTex, (valueBase + c % HEAD_DIM) / 4);
  }
  result = gathered / sum;
}`;

/** `out = keep * noisy + take * update`, the EDM preconditioning. */
export const preconditionSource = (elements) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${elements};
uniform sampler2D noisyTex;
uniform sampler2D updateTex;
uniform float keep;
uniform float take;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = keep * fetch4(noisyTex, texel) + take * fetch4(updateTex, texel);
}`;

/** `out = x * factor`. */
export const scaleSource = (elements) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${elements};
uniform sampler2D xTex;
uniform float factor;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = fetch4(xTex, texel) * factor;
}`;

/** `cos(2 * pi * (t * w + b))`, one row, on the host. */
export function fourierEmbedding(t, weights, offsets) {
  const out = new Float32Array(weights.length);
  for (let i = 0; i < weights.length; i += 1) {
    out[i] = Math.cos(2 * Math.PI * (t * weights[i] + offsets[i]));
  }
  return out;
}

/**
 * One `TransitionLayer` - the conditioning's, with its two SEPARATE projections.
 *
 * @param {import("../runtime/execution.js").Execution} execution
 * @returns {Tensor} the update; the caller adds it
 */
export function transitionLayer(execution, { x, rows, channels, hidden, weights, epsilon = 1e-5 }) {
  const normalised = layerNorm(execution, {
    x, weight: weights.normScale, bias: weights.normOffset, rows, channels, epsilon,
  });
  const a = linear(execution, {
    x: normalised, weight: weights.aProjection, rows,
    inChannels: channels, outChannels: hidden, weightLayout: "in-major",
  });
  const b = linear(execution, {
    x: normalised, weight: weights.bProjection, rows,
    inChannels: channels, outChannels: hidden, weightLayout: "in-major",
  });
  execution.release(normalised);
  const gated = execution.allocate(rows * hidden, "ef2.diffusion.gated");
  execution.dispatch({
    source: gatedProductSource(rows * hidden),
    output: gated,
    inputs: { aTex: a, bTex: b },
    label: `ef2.diffusion.transition-gate:${rows}x${hidden}`,
  });
  execution.release(a);
  execution.release(b);
  const out = linear(execution, {
    x: gated, weight: weights.outProjection, rows,
    inChannels: hidden, outChannels: channels, weightLayout: "in-major",
  });
  execution.release(gated);
  return out;
}

/**
 * The denoiser's two conditioning tensors.
 *
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{zTrunk: Tensor, relPos: Tensor, sInputs: Tensor, tHat: number,
 *   shape: any, weights: any, hostFourier: {weights: Float32Array, offsets: Float32Array},
 *   cachedPair?: Tensor, sigmaData?: number, epsilon?: number,
 *   onStage?: (stage: string, tensor: Tensor) => void}} call
 * @returns {{single: Tensor, pair: Tensor, pairIsBorrowed: boolean}}
 */
export function diffusionConditioning(execution, call) {
  const {
    zTrunk, relPos, sInputs, tHat, shape, weights: w, epsilon = 1e-5,
  } = call;
  const sigmaData = call.sigmaData ?? SIGMA_DATA;
  const { tokens, pairChannels, singleInputs, tokenChannels, multiplier } = shape;
  const pairs = tokens * tokens;

  let pair = call.cachedPair;
  const pairIsBorrowed = pair !== undefined;
  if (pair === undefined) {
    const joined = execution.allocate(pairs * pairChannels * 2, "ef2.diffusion.joined");
    execution.dispatch({
      source: concatenateSource(pairs, pairChannels),
      output: joined,
      inputs: { leftTex: zTrunk, rightTex: relPos },
      label: `ef2.diffusion.concat:${pairs}x${pairChannels * 2}`,
    });
    const normalised = layerNorm(execution, {
      x: joined, weight: w.zInputNormScale, bias: w.zInputNormOffset,
      rows: pairs, channels: pairChannels * 2, epsilon,
    });
    execution.release(joined);
    pair = linear(execution, {
      x: normalised, weight: w.zProjection, rows: pairs,
      inChannels: pairChannels * 2, outChannels: pairChannels, weightLayout: "in-major",
    });
    execution.release(normalised);
    for (const block of w.zTransitions) {
      const delta = transitionLayer(execution, {
        x: pair, rows: pairs, channels: pairChannels,
        hidden: pairChannels * multiplier, weights: block, epsilon,
      });
      const next = execution.allocate(pairs * pairChannels, "ef2.diffusion.pair");
      execution.dispatch({
        source: sumSource(pairs * pairChannels),
        output: next,
        inputs: { aTex: pair, bTex: delta },
        label: `ef2.diffusion.pair-residual:${pairs}x${pairChannels}`,
      });
      execution.release(delta);
      execution.release(pair);
      pair = next;
    }
    call.onStage?.("conditioning.pair", pair);
  }

  const normalisedSingle = layerNorm(execution, {
    x: sInputs, weight: w.sInputNormScale, bias: w.sInputNormOffset,
    rows: tokens, channels: singleInputs, epsilon,
  });
  let single = linear(execution, {
    x: normalisedSingle, weight: w.sProjection, rows: tokens,
    inChannels: singleInputs, outChannels: tokenChannels, weightLayout: "in-major",
  });
  execution.release(normalisedSingle);

  // 🔴 THE NOISE LEVEL IS LOGGED AND QUARTERED, AND THE CLAMP IS NOT DECORATION:
  // "the sampler's last step takes t to 4e-4 and a schedule that reached zero
  // would give -Infinity here".
  const tNoise = 0.25 * Math.log(Math.max(tHat / sigmaData, 1e-20));
  const fourier = fourierEmbedding(
    tNoise, call.hostFourier.weights, call.hostFourier.offsets);
  const fourierTensor = execution.upload(fourier, "ef2.diffusion.fourier");
  const normalisedNoise = layerNorm(execution, {
    x: fourierTensor, weight: w.noiseNormScale, bias: w.noiseNormOffset,
    rows: 1, channels: fourier.length, epsilon,
  });
  execution.release(fourierTensor);
  const noise = linear(execution, {
    x: normalisedNoise, weight: w.noiseProjection, rows: 1,
    inChannels: fourier.length, outChannels: tokenChannels, weightLayout: "in-major",
  });
  execution.release(normalisedNoise);
  const withNoise = execution.allocate(tokens * tokenChannels, "ef2.diffusion.single");
  execution.dispatch({
    source: addBroadcastRowSource(tokens, tokenChannels),
    output: withNoise,
    inputs: { xTex: single, rowTex: noise },
    label: `ef2.diffusion.noise:${tokens}x${tokenChannels}`,
  });
  execution.release(noise);
  execution.release(single);
  single = withNoise;

  for (const block of w.sTransitions) {
    const delta = transitionLayer(execution, {
      x: single, rows: tokens, channels: tokenChannels,
      hidden: tokenChannels * multiplier, weights: block, epsilon,
    });
    const next = execution.allocate(tokens * tokenChannels, "ef2.diffusion.single");
    execution.dispatch({
      source: sumSource(tokens * tokenChannels),
      output: next,
      inputs: { aTex: single, bTex: delta },
      label: `ef2.diffusion.single-residual:${tokens}x${tokenChannels}`,
    });
    execution.release(delta);
    execution.release(single);
    single = next;
  }
  call.onStage?.("conditioning.single", single);
  return { single, pair, pairIsBorrowed };
}

/**
 * adaLN-Zero: the activation normalised affine-free, scaled and shifted from
 * the conditioning single.
 *
 * @param {import("../runtime/execution.js").Execution} execution
 * @returns {Tensor} `rows * channels`
 */
export function adaptiveLayerNorm(execution, { activation, single, rows, channels, weights, epsilon = 1e-5 }) {
  // 🔴 THE CONDITIONING'S NORM HAS A LEARNED SCALE AND NO BIAS, and the
  // activation's has neither - which `kernels/layer-norm.js` supports by
  // generating a source without the samplers rather than binding ones.
  const normalisedSingle = layerNorm(execution, {
    x: single, weight: weights.singleScale, rows, channels, epsilon,
  });
  const gate = linear(execution, {
    x: normalisedSingle, weight: weights.gateWeights, rows,
    inChannels: channels, outChannels: channels, weightLayout: "in-major",
  });
  const shift = linear(execution, {
    x: normalisedSingle, weight: weights.shiftWeights, rows,
    inChannels: channels, outChannels: channels, weightLayout: "in-major",
  });
  execution.release(normalisedSingle);
  const out = execution.allocate(rows * channels, "ef2.diffusion.adaln");
  execution.dispatch({
    source: adaptiveSource(rows, channels),
    output: out,
    inputs: {
      activationTex: activation, gateTex: gate, shiftTex: shift,
      gateBiasTex: weights.gateBias,
    },
    floats: { epsilon },
    label: `ef2.diffusion.adaln:${rows}x${channels}`,
  });
  execution.release(gate);
  execution.release(shift);
  return out;
}

/**
 * Attention biased by the pair representation, gated twice.
 *
 * @param {import("../runtime/execution.js").Execution} execution
 * @returns {Tensor} `tokens * channels`, the update; the caller adds it
 */
export function attentionPairBias(execution, call) {
  const {
    activation, single, pair, tokens, channels, pairChannels, heads,
    weights: w, epsilon = 1e-5,
  } = call;
  const pairs = tokens * tokens;
  const x = adaptiveLayerNorm(execution, {
    activation, single, rows: tokens, channels, weights: w.adaln, epsilon,
  });
  const rawQuery = linear(execution, {
    x, weight: w.queryWeights, rows: tokens,
    inChannels: channels, outChannels: channels, weightLayout: "in-major",
  });
  const query = execution.allocate(tokens * channels, "ef2.diffusion.query");
  execution.dispatch({
    source: addRowBiasSource(tokens, channels),
    output: query,
    inputs: { xTex: rawQuery, biasTex: w.queryBias },
    label: `ef2.diffusion.query-bias:${tokens}x${channels}`,
  });
  execution.release(rawQuery);
  const kv = linear(execution, {
    x, weight: w.kvWeights, rows: tokens,
    inChannels: channels, outChannels: 2 * channels, weightLayout: "in-major",
  });
  const gate = linear(execution, {
    x, weight: w.gateWeights, rows: tokens,
    inChannels: channels, outChannels: channels, weightLayout: "in-major",
  });
  execution.release(x);

  const normalisedPair = layerNorm(execution, {
    x: pair, weight: w.pairNormScale, bias: w.pairNormOffset,
    rows: pairs, channels: pairChannels, epsilon,
  });
  const bias = linear(execution, {
    x: normalisedPair, weight: w.pairBiasWeights, rows: pairs,
    inChannels: pairChannels, outChannels: heads, weightLayout: "in-major",
  });
  execution.release(normalisedPair);

  const context = execution.allocate(tokens * channels, "ef2.diffusion.context");
  execution.dispatch({
    source: attentionSource(tokens, channels, heads),
    output: context,
    inputs: { queryTex: query, kvTex: kv, biasTex: bias },
    label: `ef2.diffusion.attend:${tokens}x${channels}:${heads}`,
  });
  execution.release(query);
  execution.release(kv);
  execution.release(bias);

  const gated = execution.allocate(tokens * channels, "ef2.diffusion.gated");
  execution.dispatch({
    source: gateSource(tokens * channels),
    output: gated,
    inputs: { xTex: context, gateTex: gate },
    label: `ef2.diffusion.context-gate:${tokens}x${channels}`,
  });
  execution.release(context);
  execution.release(gate);
  const projected = linear(execution, {
    x: gated, weight: w.outWeights, rows: tokens,
    inChannels: channels, outChannels: channels, weightLayout: "in-major",
  });
  execution.release(gated);
  // ...and the second gate, from the CONDITIONING SINGLE, with its bias.
  const outGate = linear(execution, {
    x: single, weight: w.outGateWeights, rows: tokens,
    inChannels: channels, outChannels: channels, weightLayout: "in-major",
  });
  const out = execution.allocate(tokens * channels, "ef2.diffusion.attention");
  execution.dispatch({
    source: gateWithBiasSource(tokens, channels),
    output: out,
    inputs: { xTex: projected, gateTex: outGate, biasTex: w.outGateBias },
    label: `ef2.diffusion.out-gate:${tokens}x${channels}`,
  });
  execution.release(projected);
  execution.release(outGate);
  return out;
}

/**
 * The conditioned transition - the one that FUSES its gate.
 *
 * @param {import("../runtime/execution.js").Execution} execution
 * @returns {Tensor} `tokens * channels`, the update
 */
export function conditionedTransition(execution, call) {
  const {
    activation, single, tokens, channels, hidden, weights: w, epsilon = 1e-5,
  } = call;
  const x = adaptiveLayerNorm(execution, {
    activation, single, rows: tokens, channels, weights: w.adaln, epsilon,
  });
  const wide = linear(execution, {
    x, weight: w.swishWeights, rows: tokens,
    inChannels: channels, outChannels: 2 * hidden, weightLayout: "in-major",
  });
  execution.release(x);
  const gated = execution.allocate(tokens * hidden, "ef2.diffusion.swiglu");
  execution.dispatch({
    source: swiGluSource(tokens, hidden),
    output: gated,
    inputs: { wideTex: wide },
    label: `ef2.diffusion.swiglu:${tokens}x${hidden}`,
  });
  execution.release(wide);
  const projected = linear(execution, {
    x: gated, weight: w.outWeights, rows: tokens,
    inChannels: hidden, outChannels: channels, weightLayout: "in-major",
  });
  execution.release(gated);
  const outGate = linear(execution, {
    x: single, weight: w.outGateWeights, rows: tokens,
    inChannels: channels, outChannels: channels, weightLayout: "in-major",
  });
  const out = execution.allocate(tokens * channels, "ef2.diffusion.transition");
  execution.dispatch({
    source: gateWithBiasSource(tokens, channels),
    output: out,
    inputs: { xTex: projected, gateTex: outGate, biasTex: w.outGateBias },
    label: `ef2.diffusion.transition-gate:${tokens}x${channels}`,
  });
  execution.release(projected);
  execution.release(outGate);
  return out;
}

/**
 * The denoiser's twelve blocks.
 *
 * 🔴 THE ATTENTIONS AND THE TRANSITIONS ARE TWO SEPARATE ModuleLists, ZIPPED:
 * block i is `attn_blocks[i]` then `transition_blocks[i]`, "not a single list of
 * alternating modules. A flat export that interleaved them would load the right
 * count of the wrong things."
 *
 * @param {import("../runtime/execution.js").Execution} execution
 * @returns {Tensor} `tokens * channels`
 */
export function tokenTransformer(execution, call) {
  const {
    activation, single, pair, tokens, channels, pairChannels, heads, hidden,
    blocks, epsilon = 1e-5,
  } = call;
  let current = activation;
  let own = false;
  const residual = (update) => {
    const next = execution.allocate(tokens * channels, "ef2.diffusion.token-act");
    execution.dispatch({
      source: sumSource(tokens * channels),
      output: next,
      inputs: { aTex: current, bTex: update },
      label: `ef2.diffusion.residual:${tokens}x${channels}`,
    });
    execution.release(update);
    if (own) execution.release(current);
    current = next;
    own = true;
  };
  for (const [index, block] of blocks.entries()) {
    residual(attentionPairBias(execution, {
      activation: current, single, pair, tokens, channels, pairChannels, heads,
      weights: block.attention, epsilon,
    }));
    residual(conditionedTransition(execution, {
      activation: current, single, tokens, channels, hidden,
      weights: block.transition, epsilon,
    }));
    call.onStage?.(`token-block${index}`, current);
  }
  return current;
}

/**
 * One EDM denoise step: noisy coordinates in, denoised coordinates out.
 *
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{noisy: Tensor, tHat: number, sInputs: Tensor, zTrunk: Tensor,
 *   relPos: Tensor, shape: any, weights: any,
 *   hostFourier: {weights: Float32Array, offsets: Float32Array},
 *   embed: (scaled: Tensor) => any, decode: (act: Tensor, embedded: any) => Tensor,
 *   cachedPair?: Tensor, epsilon?: number,
 *   onStage?: (stage: string, tensor: Tensor) => void}} call
 *   `embed` and `decode` are the atom transformer's two halves, injected so this
 *   file does not import the encoder's whole surface to call it once.
 * @returns {{coordinates: Tensor, pair: Tensor, pairIsBorrowed: boolean}}
 */
export function denoiseStep(execution, call) {
  const { noisy, tHat, shape, weights: w, epsilon = 1e-5 } = call;
  const sigma = shape.sigmaData ?? SIGMA_DATA;
  const conditioning = diffusionConditioning(execution, {
    zTrunk: call.zTrunk, relPos: call.relPos, sInputs: call.sInputs, tHat,
    shape, weights: w.conditioning, hostFourier: call.hostFourier,
    cachedPair: call.cachedPair, sigmaData: sigma, epsilon,
    onStage: call.onStage,
  });
  const { single, pair } = conditioning;

  const denominator = Math.sqrt(tHat * tHat + sigma * sigma);
  const scaled = execution.allocate(noisy.elements, "ef2.diffusion.scaled");
  execution.dispatch({
    source: scaleSource(noisy.elements),
    output: scaled,
    inputs: { xTex: noisy },
    floats: { factor: 1 / denominator },
    label: `ef2.diffusion.scale:${noisy.elements}`,
  });
  const embedded = call.embed(scaled);
  execution.release(scaled);

  // ...the conditioning single, projected onto the tokens the encoder pooled.
  const normalisedStep = layerNorm(execution, {
    x: single, weight: w.stepNormScale, bias: w.stepNormOffset,
    rows: shape.tokens, channels: shape.tokenChannels, epsilon,
  });
  const stepped = linear(execution, {
    x: normalisedStep, weight: w.singleToToken, rows: shape.tokens,
    inChannels: shape.tokenChannels, outChannels: shape.tokenChannels,
    weightLayout: "in-major",
  });
  execution.release(normalisedStep);
  const act = execution.allocate(
    shape.tokens * shape.tokenChannels, "ef2.diffusion.act");
  execution.dispatch({
    source: sumSource(shape.tokens * shape.tokenChannels),
    output: act,
    inputs: { aTex: embedded.tokenAct, bTex: stepped },
    label: `ef2.diffusion.step-add:${shape.tokens}x${shape.tokenChannels}`,
  });
  execution.release(stepped);

  const transformed = tokenTransformer(execution, {
    activation: act, single, pair, tokens: shape.tokens,
    channels: shape.tokenChannels, pairChannels: shape.pairChannels,
    heads: shape.tokenHeads, hidden: shape.tokenChannels * shape.multiplier,
    blocks: w.tokenBlocks, epsilon, onStage: call.onStage,
  });
  execution.release(act);
  const normalised = layerNorm(execution, {
    x: transformed, weight: w.tokenNormScale, bias: w.tokenNormOffset,
    rows: shape.tokens, channels: shape.tokenChannels, epsilon,
  });
  execution.release(transformed);
  call.onStage?.("token-norm", normalised);

  const update = call.decode(normalised, embedded);
  execution.release(normalised);
  execution.release(single);

  const sigma2 = sigma * sigma;
  const t2 = tHat * tHat;
  const out = execution.allocate(noisy.elements, "ef2.diffusion.denoised");
  execution.dispatch({
    source: preconditionSource(noisy.elements),
    output: out,
    inputs: { noisyTex: noisy, updateTex: update },
    floats: { keep: sigma2 / (sigma2 + t2), take: (sigma * tHat) / Math.sqrt(sigma2 + t2) },
    label: `ef2.diffusion.precondition:${noisy.elements}`,
  });
  execution.release(update);
  out.label = "ef2.diffusion.denoised";
  return { coordinates: out, pair, pairIsBorrowed: conditioning.pairIsBorrowed };
}
