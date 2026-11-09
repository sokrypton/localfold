/**
 * AF3's 24-block token transformer: the 198 M parameters that are 98% of the
 * diffusion head.
 *
 *     per block:  act += conditionedSelfAttention(act, cond, pairLogits)
 *                 act += conditionedTransition(act, cond)
 *
 * 🔴 THE BLOCKS ARE NESTED SIX BY FOUR, AND THE PAIR LOGITS FOLLOW THAT
 * NESTING. "The LayerNorm over the pair conditioning is computed ONCE and
 * shared, but each of the six SUPER-BLOCKS then projects it to its own four
 * blocks' worth of head biases. So there are six projections, not one and not
 * twenty-four, and a flat reading of the stack indexes the wrong weights for
 * every block after the fourth."
 *
 * 🔴 THE SCALE IS THE PER-HEAD DIMENSION, TAKEN AFTER THE DIVISION BY THE HEAD
 * COUNT. "AF3 writes `key_dim = key_dim // num_head` and only then
 * `key_dim ** -0.5`. Using the full 768 instead of 48 is a factor of four on
 * every logit, which softmax turns into a much flatter attention."
 *
 * 🔴 AND A MISSING HEAD COUNT IS NOT AN ERROR IN JAVASCRIPT, IT IS A ZERO. The
 * reference records what that cost: it "read `weights.heads` while the head
 * count lives on the transformer config rather than the per-block weights, so
 * the loop below ran zero times and the whole attention branch returned an array
 * of zeros - no exception, no NaN, just a residual stream missing half its
 * updates. It cost a bisection down to the block to find, past four stages that
 * were all exact." So the shape is validated here rather than trusted.
 *
 * 🔴 AND THE CONDITIONING IS WIDER THAN THE ACTIVATION HERE - 384 against 768 -
 * which is the case `crossAttentionBlock` cannot be asked about and this one is
 * built for.
 *
 * The attention itself is `kernels/attention-flash.js` at `batch = 1`, which is
 * the same reuse AF3's single-track attention gets: that kernel's mask index
 * collapses to the key position and its bias layout is already `[head][i][j]`.
 */
import { ONE_TARGET, PRELUDE, sized, targetDeclarations, memoSource } from "../../runtime/execution.js";
import { drain } from "../../runtime/steps.js";
import { adaptiveLayerNorm, adaptiveZeroInit, elementwiseProductSource, sumSource }
  from "./atom-attention.js";
import { attentionFlash } from "../../kernels/attention-flash.js";
import { layerNorm } from "../../kernels/layer-norm.js";
import { linear, linearPaddedStride4, planeFetchGlsl, planeInputs } from "../../kernels/linear.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */

/** `swish(wide[i]) * wide[intermediate + i]` over a blocked row. */
const swiGluSourceUncached = (rows, intermediate) => sized({ ELEMENTS: rows * intermediate }, `${PRELUDE}${ONE_TARGET}
const int INTER4 = ${intermediate / 4};
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
}`);

/** `swiGluSource` reading the up projection's kept row planes (`planeFetchGlsl`) rather than a repacked tensor. */
const swiGluPlanesSourceUncached = (rows, intermediate, blockRows) => sized({ ELEMENTS: rows * intermediate }, `${PRELUDE}${ONE_TARGET}
const int INTER4 = ${intermediate / 4};
${planeFetchGlsl(blockRows, intermediate / 2)}
void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % INTER4;
  int row = texel / INTER4;
  vec4 gate = planeFetch(row, group);
  vec4 value = planeFetch(row, INTER4 + group);
  result = (gate / (vec4(1.0) + exp(-gate))) * value;
}`);

/** `splitQkvgSource` reading the fused projection's kept row planes. */
const splitQkvgPlanesSourceUncached = (rows, width, blockRows) => sized({ ROWS: rows }, `${PRELUDE}${targetDeclarations(4)}
const int W4 = ${width / 4};
${planeFetchGlsl(blockRows, width)}
void main() {
  int texel = linearTexel();
  if (texel >= ROWS * W4) {
    result0 = vec4(0.0); result1 = vec4(0.0); result2 = vec4(0.0); result3 = vec4(0.0);
    return;
  }
  int row = texel / W4;
  int group = texel % W4;
  result0 = planeFetch(row, group);
  result1 = planeFetch(row, group + W4);
  result2 = planeFetch(row, group + 2 * W4);
  result3 = vec4(1.0) / (vec4(1.0) + exp(-planeFetch(row, group + 3 * W4)));
}`);

/**
 * The whole AdaLN in ONE draw: the row's LayerNorm (no scale, no offset, two-pass variance - `layerNorm`'s default)
 * and the blend `sigmoid(scale) * normalised + shift`, with scale and shift read from one row of the step's batched
 * conditioning projections (`condBatchWeights`) at their column offsets.
 *
 * 🔴 EVERY FRAGMENT RECOMPUTES ITS ROW'S STATISTICS, which the split LayerNorm exists to avoid on a long row count -
 * and which is free here: the denoiser's rows are its TOKENS, 59 at 59 residues, and a draw costs ~47 us before it
 * computes anything (`probe-webgl-draw-floor.js`). Three draws (statistics, apply, blend) become one, twice a block.
 */
const blendSliceSourceUncached = (rows, channels, stride, parts = 0, outStride4 = channels / 4) => sized({ ROWS: rows }, `${PRELUDE}${ONE_TARGET}
const int CHANNELS = ${channels};
const int C4 = ${channels / 4};
const int OUT4 = ${outStride4};
const int STRIDE4 = ${stride / 4};
const int PARTS = ${Math.max(parts, 1)};
uniform sampler2D xTex;
uniform sampler2D condTex;
${parts > 0 ? "uniform sampler2D sumTex;\nuniform sampler2D squareTex;" : ""}
uniform int scaleAt4;
uniform int shiftAt4;
uniform float epsilon;

void main() {
  int target = linearTexel();
  if (target >= ROWS * OUT4) { result = vec4(0.0); return; }
  int row = target / OUT4;
  int c4 = target % OUT4;
  // ...an output laid at a wider stride than its row (the next projection's padded x): the tail is zeros
  if (c4 >= C4) { result = vec4(0.0); return; }
  int texel = row * C4 + c4;
  int base = row * C4;
${parts > 0 ? `  float total = 0.0;
  float spread = 0.0;
  for (int p = 0; p < PARTS; ++p) {
    total += fetch1(sumTex, row * PARTS + p);
    spread += fetch1(squareTex, row * PARTS + p);
  }
  float mean = total / float(CHANNELS);` : `  float total = 0.0;
  for (int i = 0; i < C4; ++i) {
    vec4 v = fetch4(xTex, base + i);
    total += v.x + v.y + v.z + v.w;
  }
  float mean = total / float(CHANNELS);
  float spread = 0.0;
  for (int i = 0; i < C4; ++i) {
    vec4 centred = fetch4(xTex, base + i) - mean;
    spread += dot(centred, centred);
  }`}
  float inverseStd = inversesqrt(spread / float(CHANNELS) + epsilon);
  vec4 normalised = (fetch4(xTex, texel) - mean) * inverseStd;
  vec4 scale = fetch4(condTex, row * STRIDE4 + scaleAt4 + c4);
  result = (vec4(1.0) / (vec4(1.0) + exp(-scale))) * normalised
    + fetch4(condTex, row * STRIDE4 + shiftAt4 + c4);
}`);

/**
 * A row's statistics in parts, for `blendSliceSource`'s `parted` form: `PARTS` fragments a row, each summing its slice
 * of `C4 / PARTS` texels - `centred` 0 the plain sum, 1 the sum of squares about the row mean (from the plain parts). The
 * same two-pass formula as the one-fragment loop, summed in `PARTS` groups rather than in one sequence, so every blend
 * fragment reads `2 x PARTS` floats where it walked its whole row twice.
 */
const rowPartsSourceUncached = (rows, channels, parts, centred) => sized({ ROWS: rows }, `${PRELUDE}${ONE_TARGET}
const int CHANNELS = ${channels};
const int C4 = ${channels / 4};
const int PARTS = ${parts};
const int SLICE = ${channels / 4 / parts};
uniform sampler2D xTex;
${centred ? "uniform sampler2D sumTex;" : ""}

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ROWS * PARTS) { result = vec4(0.0); return; }
  vec4 out4 = vec4(0.0);
  for (int lane = 0; lane < 4; ++lane) {
    int element = texel * 4 + lane;
    if (element >= ROWS * PARTS) break;
    int row = element / PARTS;
    int part = element - row * PARTS;
${centred ? `    float total = 0.0;
    for (int p = 0; p < PARTS; ++p) total += fetch1(sumTex, row * PARTS + p);
    float mean = total / float(CHANNELS);` : ""}
    float acc = 0.0;
    int base = row * C4 + part * SLICE;
    for (int i = 0; i < SLICE; ++i) {
      vec4 v = fetch4(xTex, base + i);
${centred ? `      vec4 c = v - mean;
      acc += dot(c, c);` : "      acc += v.x + v.y + v.z + v.w;"}
    }
    if (lane == 0) out4.x = acc;
    else if (lane == 1) out4.y = acc;
    else if (lane == 2) out4.z = acc;
    else out4.w = acc;
  }
  result = out4;
}`);

/**
 * AdaLN-zero's gate - `projected * gate` - with the (already sigmoid) gate read from the step's batched gates, and,
 * `withResidual`, the block's residual added in the same draw: `residual + projected * gate`.
 */
const gateSliceSourceUncached = (rows, channels, stride, withResidual = false) => sized({ ROWS: rows }, `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int STRIDE4 = ${stride / 4};
uniform sampler2D projectedTex;
uniform sampler2D gatesTex;
${withResidual ? "uniform sampler2D residualTex;" : ""}
uniform int gateAt4;

void main() {
  int texel = linearTexel();
  if (texel >= ROWS * C4) { result = vec4(0.0); return; }
  result = ${withResidual ? "fetch4(residualTex, texel) + " : ""}fetch4(projectedTex, texel)
    * fetch4(gatesTex, (texel / C4) * STRIDE4 + gateAt4 + texel % C4);
}`);

/** The most floats `pairLogitsCache` holds - every block's pair biases, 24 x 16 heads x L^2 - before it recomputes them. */
const PAIR_LOGITS_CACHE_FLOATS = 32 * 2 ** 20;

/** Release what a walk's `pairLogitsCache` holds. */
export function releasePairLogitsCache(execution, cache) {
  for (const tensor of cache?.logits ?? []) execution.release(tensor);
  if (cache !== undefined) cache.logits = null;
}

/** The two AdaLNs of a block, in `condBatchWeights`' column order: attention then transition. */
const ADALN_PREFIXES = ["", "ffw"];

/**
 * Every block's conditioning projections as two wide weights, so a denoiser step computes them in two draws rather
 * than 8 a block (192 at 24 blocks):
 *
 *     scaleShift = LayerNorm(cond) @ [ (gamma * scale) | (gamma * shift) ] per block, per AdaLN    [cond][4 C x blocks]
 *     gates      = sigmoid(cond @ [ zero-gate ] per block, per AdaLN-zero + bias)                    [cond][2 C x blocks]
 *
 * 🔴 THE CONDITIONING'S LAYERNORM SCALE IS FOLDED INTO THE WEIGHTS: `(LN(c) * gamma) @ W = LN(c) @ (gamma (.) W)`, so
 * one unscaled LayerNorm serves every block - which is what makes ONE projection possible. Not bit-identical: gamma
 * is applied in float32 before the weights are rounded to float16, not after.
 *
 * 🔴 ALL THE CONDITIONING A BLOCK READS DEPENDS ONLY ON THE STEP. `--skip` priced these eight draws a block at 0.57 s
 * of a 4.1 s diffusion, as 59 x 384 -> 768 projections too narrow to take the blocked kernel; batched they are wide.
 *
 * Returns null unless every block carries all of them in the expected shape.
 *
 * @param {Array<Record<string, any>>} blocks the host blocks, in order
 * @param {number} channels the token channels C
 * @returns {{scaleShiftWeight: Float32Array, scaleShiftBias: Float32Array,
 *   gateWeight: Float32Array, gateBias: Float32Array, condChannels: number} | null}
 */
export function condBatchWeights(blocks, channels) {
  if (blocks.length === 0 || channels % 4 !== 0) return null;
  const first = blocks[0].SingleCondScaleWeights;
  if (!(first instanceof Float32Array) || first.length % channels !== 0) return null;
  const condChannels = first.length / channels;
  const f32 = (value, length) => value instanceof Float32Array && value.length === length;
  for (const block of blocks) {
    for (const p of ADALN_PREFIXES) {
      if (!f32(block[`${p}SingleCondScaleWeights`], condChannels * channels)
        || !f32(block[`${p}SingleCondBias`], condChannels * channels)
        || !f32(block[`${p}AdaptiveZeroCondWeights`], condChannels * channels)
        || !f32(block[`${p}AdaptiveZeroCondBias`], channels)) return null;
      const gamma = block[`${p}SingleCondLayerNormScale`];
      if (gamma !== undefined && gamma !== null && !f32(gamma, condChannels)) return null;
      const bias = block[`${p}SingleCondScaleBias`];
      if (bias !== undefined && bias !== null && !f32(bias, channels)) return null;
    }
  }
  const scaleWidth = 4 * channels * blocks.length;
  const gateWidth = 2 * channels * blocks.length;
  const scaleShiftWeight = new Float32Array(condChannels * scaleWidth);
  const scaleShiftBias = new Float32Array(scaleWidth);
  const gateWeight = new Float32Array(condChannels * gateWidth);
  const gateBias = new Float32Array(gateWidth);
  for (const [b, block] of blocks.entries()) {
    for (const [a, p] of ADALN_PREFIXES.entries()) {
      const gamma = block[`${p}SingleCondLayerNormScale`] ?? null;
      const scale = block[`${p}SingleCondScaleWeights`];
      const shift = block[`${p}SingleCondBias`];
      const zero = block[`${p}AdaptiveZeroCondWeights`];
      const scaleAt = (b * 4 + a * 2) * channels;
      const shiftAt = scaleAt + channels;
      const gateAt = (b * 2 + a) * channels;
      for (let k = 0; k < condChannels; k += 1) {
        const g = gamma === null ? 1 : gamma[k];
        const row = k * channels;
        for (let o = 0; o < channels; o += 1) {
          scaleShiftWeight[k * scaleWidth + scaleAt + o] = g * scale[row + o];
          scaleShiftWeight[k * scaleWidth + shiftAt + o] = g * shift[row + o];
          gateWeight[k * gateWidth + gateAt + o] = zero[row + o];
        }
      }
      const bias = block[`${p}SingleCondScaleBias`];
      if (bias !== undefined && bias !== null) scaleShiftBias.set(bias, scaleAt);
      gateBias.set(block[`${p}AdaptiveZeroCondBias`], gateAt);
    }
  }
  return { scaleShiftWeight, scaleShiftBias, gateWeight, gateBias, condChannels };
}

/** The host keys `condBatchWeights` takes over, which a batched block no longer uploads. */
export const COND_BATCH_KEYS = ADALN_PREFIXES.flatMap((p) => [
  `${p}SingleCondLayerNormScale`, `${p}SingleCondScaleWeights`, `${p}SingleCondScaleBias`, `${p}SingleCondBias`,
  `${p}AdaptiveZeroCondWeights`, `${p}AdaptiveZeroCondBias`,
]);

/** `adaptiveLayerNorm` from a block's slice of the step's batched projections. */
function blendFromBatch(execution, { x, rows, channels, epsilon, batch, which, outStride4 = 0 }) {
  // `outStride4`: written as the next projection's padded x (`linearPaddedStride4`), float16 where its pad would be
  const out = outStride4 === 0 ? execution.allocate(rows * channels, "af3.adaln.batched")
    : execution.halfActivations ? execution.allocateHalf(rows * outStride4 * 4, "af3.adaln.batched-padded")
      : execution.allocate(rows * outStride4 * 4, "af3.adaln.batched-padded");
  const scaleAt = (batch.index * 4 + which * 2) * channels;
  // ...the row statistics in parts first (`rowPartsSource`), where the row divides into them
  const parts = !execution.noAdalnParts && (channels / 4) % ADALN_PARTS === 0 ? ADALN_PARTS : 0;
  let sums;
  let squares;
  if (parts > 0) {
    sums = execution.allocate(rows * parts, "af3.adaln.sums");
    execution.dispatch({
      source: rowPartsSource(rows, channels, parts, false), output: sums, inputs: { xTex: x },
      label: `af3.adaln.row-sums:${rows}x${channels}`,
    });
    squares = execution.allocate(rows * parts, "af3.adaln.squares");
    execution.dispatch({
      source: rowPartsSource(rows, channels, parts, true), output: squares, inputs: { xTex: x, sumTex: sums },
      label: `af3.adaln.row-squares:${rows}x${channels}`,
    });
  }
  execution.dispatch({
    source: blendSliceSource(rows, channels, batch.scaleShiftWidth, parts, outStride4 || channels / 4),
    output: out,
    inputs: { xTex: x, condTex: batch.scaleShift, ...(parts > 0 ? { sumTex: sums, squareTex: squares } : {}) },
    ints: { scaleAt4: scaleAt / 4, shiftAt4: (scaleAt + channels) / 4 },
    floats: { epsilon },
    label: `af3.adaln.fused:${rows}x${channels}`,
  });
  if (parts > 0) {
    execution.release(sums);
    execution.release(squares);
  }
  return out;
}

/** `adaptiveZeroInit` from a block's slice of the step's batched gates. */
function zeroFromBatch(execution, { x, rows, channels, inChannels, weight, batch, which, residual }) {
  // ...the gate (and the residual) in the projection's own last draw - its epilogue, or its repack - where it was a
  // separate `zero-slice` draw (`linear`'s `gateSlice`)
  if (residual !== undefined && channels % 4 === 0 && !execution.noLinearGateSlice
    && !execution.noLinearEpilogue && !execution.noRepackResidual) {
    return linear(execution, {
      x, weight, rows, inChannels, outChannels: channels, weightLayout: "in-major", residual,
      gateSlice: { gates: batch.gates, stride4: batch.gateWidth / 4, at4: ((batch.index * 2 + which) * channels) / 4 },
    });
  }
  const projected = linear(execution, {
    x, weight, rows, inChannels, outChannels: channels, weightLayout: "in-major",
  });
  const out = execution.allocate(rows * channels, "af3.adaln.zero");
  execution.dispatch({
    source: gateSliceSource(rows, channels, batch.gateWidth, residual !== undefined),
    output: out,
    inputs: { projectedTex: projected, gatesTex: batch.gates, ...(residual === undefined ? {} : { residualTex: residual }) },
    ints: { gateAt4: ((batch.index * 2 + which) * channels) / 4 },
    label: `af3.adaln.zero-slice:${rows}x${channels}`,
  });
  execution.release(projected);
  return out;
}

/**
 * One `[rows][4 x width]` projection split into query, key, value and the sigmoid gate - one draw, four targets.
 * The sigmoid is `linear`'s own.
 */
const splitQkvgSourceUncached = (rows, width) => sized({ ROWS: rows }, `${PRELUDE}${targetDeclarations(4)}
const int W4 = ${width / 4};
uniform sampler2D fusedTex;

void main() {
  int texel = linearTexel();
  if (texel >= ROWS * W4) {
    result0 = vec4(0.0); result1 = vec4(0.0); result2 = vec4(0.0); result3 = vec4(0.0);
    return;
  }
  int base = (texel / W4) * 4 * W4 + texel % W4;
  result0 = fetch4(fusedTex, base);
  result1 = fetch4(fusedTex, base + W4);
  result2 = fetch4(fusedTex, base + 2 * W4);
  result3 = vec4(1.0) / (vec4(1.0) + exp(-fetch4(fusedTex, base + 3 * W4)));
}`);

/**
 * A diffusion block's four attention projections as ONE `[in][4 x width]` weight (query, key, value, gate along the
 * output axis) and its bias (the query's, then zeros) - what `conditionedSelfAttention` reads as `qkvgProjection`.
 *
 * 🔴 FOUR 59 x 768 -> 768 PROJECTIONS ARE SQUARE AND ONE 768 -> 3072 IS WIDE, and on a 59-row operand only a wide one
 * takes the blocked projection (`blockedApplies`): `--skip` priced the four plus the output projection at 0.89 s of a
 * 4.1 s diffusion. Returns null when the block does not carry all four in the shape this expects.
 *
 * @param {Record<string, any>} block a host block, in-major `[in][out]` weights
 * @returns {{weight: Float32Array, bias: Float32Array} | null}
 */
export function fuseQkvgWeights(block) {
  const parts = [block.qProjection, block.kProjection, block.vProjection, block.gatingQuery];
  if (parts.some((part) => !(part instanceof Float32Array)) || !(block.qBias instanceof Float32Array)) return null;
  const width = block.qBias.length;
  if (width % 4 !== 0 || parts.some((part) => part.length !== parts[0].length || part.length % width !== 0)) return null;
  const inChannels = parts[0].length / width;
  const weight = new Float32Array(inChannels * 4 * width);
  for (let k = 0; k < inChannels; k += 1) {
    for (let part = 0; part < 4; part += 1) {
      weight.set(parts[part].subarray(k * width, (k + 1) * width), (k * 4 + part) * width);
    }
  }
  const bias = new Float32Array(4 * width);
  bias.set(block.qBias, 0);
  return { weight, bias };
}

/**
 * `logits[head][i][j] = flat[(i * tokens + j) * perSuper * heads + inner * heads + head]`
 *
 * The nesting made explicit: a super-block's projection carries `perSuper`
 * blocks' biases interleaved by head, and this pulls out one block's.
 */
const superBlockLogitsSourceUncached = (tokens, heads, perSuper, inner) => sized({ L: tokens, INNER: inner, ELEMENTS: heads * tokens * tokens }, `${PRELUDE}${ONE_TARGET}
const int HEADS = ${heads};
const int PER_SUPER = ${perSuper};
uniform sampler2D flatTex;

float at(int element) {
  int j = element % L;
  int rest = element / L;
  int i = rest % L;
  int head = rest / L;
  int source = (i * L + j) * PER_SUPER * HEADS + INNER * HEADS + head;
  return fetch1(flatTex, source);
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
 * A transition block with AdaLN conditioning, or none.
 *
 * 🔴 WITH `cond` ABSENT IT DEGRADES TO THE PLAIN FORM, which is what the
 * diffusion conditioning's own transitions use - and boltz2's extra up-gate is
 * gated on `cond` AS WELL AS on the weight, because "boltz2's plain Transition -
 * the one the diffusion conditioning's four transitions use - has NO up-gate,
 * and only its adaLN-conditioned ConditionedTransitionBlock does".
 *
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{x: Tensor, cond?: Tensor, rows: number, channels: number, factor: number,
 *   weights: any, prefix: string, condChannels?: number, epsilon?: number}} call
 * @returns {Tensor}
 */
export function conditionedTransition(execution, call) {
  const { x, cond, rows, channels, factor, weights: w, prefix, epsilon = 1e-5 } = call;
  const condChannels = call.condChannels ?? channels;
  const intermediate = channels * factor;
  // 🔴 THE KEY NAMES FOLLOW `adaptiveLayerNorm`'S, "which prefixes with 'ffw'
  // and then capitalises the leaf - so a conditioned transition reads
  // `ffwTransition1`, not `FfwTransition1`. Getting it wrong hands `linear` an
  // undefined weight, which at least throws; the danger is 'fixing' it by
  // renaming here and silently splitting the convention in two."
  const batch = call.condBatch;
  // ...the batched AdaLN writes the up projection's padded x itself, where that projection would pad one - 🔴 BUT NOT
  // WHEN boltz2's UP-GATE (`ffwAToB`) READS THE SAME x, which it takes unpadded: "linear: x holds 60416 floats, not 59 x
  // 768" was the first boltz2 fold here, and AF3's path never has the gate, so nothing else saw it
  const upGated = w[`${prefix}ffwAToB`] !== undefined && w[`${prefix}ffwAToB`] !== null;
  const xStride4 = cond !== undefined && batch !== undefined && !execution.noPrepaddedX && !upGated
    ? linearPaddedStride4(execution, { rows, inChannels: channels, outChannels: intermediate * 2, weightLayout: "in-major" })
    : 0;
  const normalised = cond === undefined
    ? layerNorm(execution, {
      x, weight: w[`${prefix}ffwLayerNormScale`], bias: w[`${prefix}ffwLayerNormOffset`],
      rows, channels, epsilon,
    })
    : batch !== undefined ? blendFromBatch(execution, { x, rows, channels, epsilon, batch, which: 1, outStride4: xStride4 })
      : adaptiveLayerNorm(execution, {
        x, cond, rows, channels, condChannels, weights: w, prefix: `${prefix}ffw`, epsilon,
      });
  const wide = linear(execution, {
    x: normalised, weight: w[`${prefix}ffwTransition1`], rows,
    inChannels: channels, outChannels: intermediate * 2, weightLayout: "in-major", keepPlanes: true, xStride4,
  });
  let gated = execution.allocate(rows * intermediate, "af3.difftx.swiglu");
  // ...reading the projection's row planes where it kept them, with no repack draw between
  const kept = /** @type {any} */ (wide).planes !== undefined ? /** @type {any} */ (wide) : null;
  execution.dispatch({
    source: kept === null ? swiGluSource(rows, intermediate) : swiGluPlanesSource(rows, intermediate, kept.blockRows),
    output: gated,
    inputs: kept === null ? { wideTex: wide } : planeInputs(kept),
    label: `af3.difftx.swiglu:${rows}x${intermediate}${kept === null ? "" : ":planes"}`,
  });
  if (kept === null) execution.release(wide);
  else for (const plane of kept.planes) execution.release(plane);
  if (cond !== undefined && w[`${prefix}ffwAToB`] !== undefined
      && w[`${prefix}ffwAToB`] !== null) {
    const upGate = linear(execution, {
      x: normalised, weight: w[`${prefix}ffwAToB`], rows,
      inChannels: channels, outChannels: intermediate, weightLayout: "in-major",
    });
    const multiplied = execution.allocate(rows * intermediate, "af3.difftx.up-gated");
    execution.dispatch({
      source: elementwiseProductSource(rows * intermediate),
      output: multiplied,
      inputs: { aTex: gated, bTex: upGate },
      label: `af3.difftx.up-gate:${rows}x${intermediate}`,
    });
    execution.release(gated);
    execution.release(upGate);
    gated = multiplied;
  }
  if (cond === undefined) {
    execution.release(normalised);
    const out = linear(execution, {
      x: gated, weight: w[`${prefix}ffwTransition2`], rows,
      inChannels: intermediate, outChannels: channels, weightLayout: "in-major",
    });
    execution.release(gated);
    return out;
  }
  execution.release(normalised);
  const out = batch !== undefined
    ? zeroFromBatch(execution, {
      x: gated, rows, channels, inChannels: intermediate, weight: w[`${prefix}ffwTransition2`], batch, which: 1,
      residual: call.residual,
    })
    : adaptiveZeroInit(execution, {
      x: gated, cond, rows, channels, condChannels, inChannels: intermediate,
      weights: w, prefix: `${prefix}ffw`,
    });
  execution.release(gated);
  return out;
}

/**
 * Self-attention over tokens, conditioned by AdaLN and biased by the pair.
 *
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{act: Tensor, cond: Tensor, pairLogits: Tensor, mask: Tensor,
 *   tokens: number, channels: number, condChannels: number, heads: number,
 *   dimension: number, weights: any, epsilon?: number}} call
 * @returns {Tensor}
 */
export function conditionedSelfAttention(execution, call) {
  const {
    act, cond, pairLogits, mask, tokens, channels, condChannels, heads, dimension,
    weights: w, epsilon = 1e-5,
  } = call;
  if (!Number.isInteger(heads) || !Number.isInteger(dimension)) {
    throw new Error(
      `af3 diffusion attention needs heads and dimension, got ${heads} and ${dimension}`
      + " - a missing head count is not an error in JavaScript, it is a zero, and the"
      + " whole attention branch then returns zeros with no exception and no NaN");
  }
  const width = heads * dimension;
  const batch = call.condBatch;
  // ...the batched AdaLN writes the fused projection's padded x itself, where that projection would pad one
  const xStride4 = batch !== undefined && w.qkvgProjection !== undefined && !execution.noPrepaddedX
    ? linearPaddedStride4(execution, { rows: tokens, inChannels: channels, outChannels: 4 * width, weightLayout: "in-major" })
    : 0;
  const x = batch !== undefined
    ? blendFromBatch(execution, { x: act, rows: tokens, channels, epsilon, batch, which: 0, outStride4: xStride4 })
    : adaptiveLayerNorm(execution, {
      x: act, cond, rows: tokens, channels, condChannels, weights: w, prefix: "", epsilon,
    });
  const project = (weight, bias, activation) => linear(execution, {
    x, weight, rows: tokens, inChannels: channels, outChannels: width,
    weightLayout: "in-major",
    ...(bias === undefined ? {} : { bias }),
    ...(activation === undefined ? {} : { activation }),
  });
  let query, key, value, gate;
  if (w.qkvgProjection !== undefined) {
    // ...one wide projection and one split, where four square ones did not block (`fuseQkvgWeights`)
    const fused = linear(execution, {
      x, weight: w.qkvgProjection, bias: w.qkvgBias, rows: tokens, inChannels: channels, outChannels: 4 * width,
      weightLayout: "in-major", keepPlanes: true, xStride4,
    });
    [query, key, value, gate] = ["q", "k", "v", "g"].map((part) => execution.allocate(tokens * width, `af3.difftx.${part}`));
    // ...reading the projection's row planes where it kept them, with no repack draw between
    const kept = /** @type {any} */ (fused).planes !== undefined ? /** @type {any} */ (fused) : null;
    execution.dispatch({
      source: kept === null ? splitQkvgSource(tokens, width) : splitQkvgPlanesSource(tokens, width, kept.blockRows),
      outputs: [query, key, value, gate],
      inputs: kept === null ? { fusedTex: fused } : planeInputs(kept),
      label: `af3.difftx.split-qkvg:${tokens}x${width}${kept === null ? "" : ":planes"}`,
    });
    if (kept === null) execution.release(fused);
    else for (const plane of kept.planes) execution.release(plane);
  } else {
    query = project(w.qProjection, w.qBias);
    key = project(w.kProjection);
    value = project(w.vProjection);
    gate = project(w.gatingQuery, undefined, "sigmoid");
  }
  // 🔴 rosettafold3's kq_norm: a TRAINED LayerNorm on q and k over the flattened heads x dimension, after the projection
  // and before the scaling (shared/af3/weights/diffusion-weights.js) - read off the weights, which only rf3 carries.
  // Missing here, its whole denoiser attended on unnormalised queries and keys.
  if (w.queryLayerNormScale !== undefined && w.queryLayerNormScale !== null) {
    const normalise = (tensor, which) => {
      const out = layerNorm(execution, {
        x: tensor, weight: w[`${which}LayerNormScale`], bias: w[`${which}LayerNormOffset`],
        rows: tokens, channels: width, epsilon,
      });
      execution.release(tensor);
      return out;
    };
    query = normalise(query, "query");
    key = normalise(key, "key");
  }
  // ...the scale is the PER-HEAD dimension, which `attentionFlash` defaults to.
  const { output: gathered } = attentionFlash(execution, {
    query, key, value, gate, mask, pairBias: pairLogits,
    batch: 1, queries: tokens, heads, headDim: dimension,
  });
  for (const tensor of [query, key, value, gate]) execution.release(tensor);
  execution.release(x);
  const out = batch !== undefined
    ? zeroFromBatch(execution, {
      x: gathered, rows: tokens, channels, inChannels: width, weight: w.Transition2, batch, which: 0,
      residual: call.residual,
    })
    : adaptiveZeroInit(execution, {
      x: gathered, cond, rows: tokens, channels, condChannels, inChannels: width,
      weights: w, prefix: "",
    });
  execution.release(gathered);
  return out;
}

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{act: Tensor, cond: Tensor, pairCond: Tensor, mask: Tensor, tokens: number,
 *   weights: any, epsilon?: number,
 *   onBlock?: (superBlock: number, inner: number) => void}} call
 * @returns {Tensor} a new activation; the caller's `act` is untouched
 */
export function diffusionTransformer(execution, call) {
  return drain(diffusionTransformerSteps(execution, call));
}

/**
 * `diffusionTransformer` as a generator: yields after each super-block's draws, so a caller that
 * waits for the GPU there (sampleOnWebGl2Steps) hands it the head in pieces rather than as one
 * submission every other WebGL context on the page has to wait behind.
 */
export function* diffusionTransformerSteps(execution, call) {
  const { act, cond, pairCond, mask, tokens, weights: w, epsilon = 1e-5 } = call;
  const channels = w.channels;
  const condChannels = w.condChannels;
  const heads = w.heads;
  const perSuper = w.blocksPerSuperBlock;
  const pairs = tokens * tokens;

  // 🔴 THE PAIR BIASES ARE THE SAME EVERY STEP: they come from the pair conditioning alone, which the sampler hoists
  // for the whole walk. A caller that passes `pairLogitsCache` (denoiserFromHead) gets them computed on the first
  // step - one LayerNorm and a projection a super-block over every PAIR, and a slice a block - and read thereafter.
  const cache = call.pairLogitsCache;
  const cached = cache?.logits ?? null;
  const keep = cache !== undefined && cached === null
    && w.superBlocks.length * perSuper * heads * pairs <= PAIR_LOGITS_CACHE_FLOATS;
  if (keep) cache.logits = [];
  // One LayerNorm over the pair conditioning, shared by every super-block.
  const normalisedPair = cached !== null ? null : layerNorm(execution, {
    x: pairCond, weight: w.pairInputLayerNormScale,
    rows: pairs, channels: w.pairChannels, epsilon,
  });

  // ...and, where the weights were batched at upload (`condBatchWeights`), every block's conditioning projections in
  // two draws for the whole step
  let condBatch;
  if (w.condBatch !== undefined) {
    const cb = w.condBatch;
    const condNormalised = layerNorm(execution, { x: cond, rows: tokens, channels: condChannels, epsilon });
    const scaleShiftWidth = cb.scaleShiftBias.elements;
    const gateWidth = cb.gateBias.elements;
    const scaleShift = linear(execution, {
      x: condNormalised, weight: cb.scaleShiftWeight, bias: cb.scaleShiftBias, rows: tokens,
      inChannels: condChannels, outChannels: scaleShiftWidth, weightLayout: "in-major",
    });
    execution.release(condNormalised);
    const gates = linear(execution, {
      x: cond, weight: cb.gateWeight, bias: cb.gateBias, rows: tokens,
      inChannels: condChannels, outChannels: gateWidth, weightLayout: "in-major", activation: "sigmoid",
    });
    condBatch = { scaleShift, gates, scaleShiftWidth, gateWidth };
  }
  let current = act;
  let own = false;
  for (const [superBlock, group] of w.superBlocks.entries()) {
    // ...and one projection a SUPER-BLOCK, carrying its four blocks' biases.
    const flat = cached !== null ? null : linear(execution, {
      x: normalisedPair, weight: group.pairLogitsProjection, rows: pairs,
      inChannels: w.pairChannels, outChannels: perSuper * heads, weightLayout: "in-major",
    });
    for (let inner = 0; inner < perSuper; inner += 1) {
      let pairLogits = cached?.[superBlock * perSuper + inner] ?? null;
      if (pairLogits === null) {
        pairLogits = execution.allocate(heads * pairs, "af3.difftx.logits");
        execution.dispatch({
          source: superBlockLogitsSource(tokens, heads, perSuper, inner),
          output: pairLogits,
          inputs: { flatTex: flat },
          label: `af3.difftx.logits:${tokens}x${heads}:${inner}/${perSuper}`,
        });
        if (keep) cache.logits.push(pairLogits);
      }
      const block = group.blocks[inner];
      const blockBatch = condBatch === undefined ? undefined
        : { ...condBatch, index: superBlock * perSuper + inner };
      // ...batched, each residual add is the gate draw's (`gateSliceSource`'s `withResidual`): two draws a block fewer
      const fused = blockBatch !== undefined;
      const attention = conditionedSelfAttention(execution, {
        act: current, cond, pairLogits, mask, tokens, channels, condChannels,
        heads, dimension: w.dimension, weights: block, epsilon, condBatch: blockBatch,
        ...(fused ? { residual: current } : {}),
      });
      if (cached === null && !keep) execution.release(pairLogits);
      let afterAttention = attention;
      if (!fused) {
        afterAttention = execution.allocate(tokens * channels, "af3.difftx.after-attention");
        execution.dispatch({
          source: sumSource(tokens * channels),
          output: afterAttention,
          inputs: { aTex: current, bTex: attention },
          label: `af3.difftx.residual:${tokens}x${channels}`,
        });
        execution.release(attention);
      }
      // 🔴 rosettafold3's BLOCK IS PARALLEL (`diffusionNoResidual`, `w.noResidual`): its transition reads the
      // PRE-attention activation and the block stores `act + attention(act) + transition(act)`, where AF3's reads
      // `act + attention`. The atom stacks had the convention here and this transformer did not - rf3's WebGL2 fold was
      // 1.1 A from WebGPU's after superposition with the trunk exact (webgpu/af3/diffusion/diffusion-transformer.js).
      const parallel = w.noResidual === true;
      if (own && !parallel) execution.release(current);
      const transitioned = conditionedTransition(execution, {
        x: parallel ? current : afterAttention, cond, rows: tokens, channels,
        factor: w.transitionFactor, weights: block, prefix: "", condChannels, epsilon, condBatch: blockBatch,
        ...(fused ? { residual: afterAttention } : {}),
      });
      let next = transitioned;
      if (!fused) {
        next = execution.allocate(tokens * channels, "af3.difftx.block");
        execution.dispatch({
          source: sumSource(tokens * channels),
          output: next,
          inputs: { aTex: afterAttention, bTex: transitioned },
          label: `af3.difftx.residual2:${tokens}x${channels}`,
        });
        execution.release(transitioned);
      }
      if (own && parallel) execution.release(current);
      execution.release(afterAttention);
      current = next;
      own = true;
      call.onBlock?.(superBlock, inner);
    }
    if (flat !== null) execution.release(flat);
    yield { stage: "transformer", superBlock };
  }
  if (normalisedPair !== null) execution.release(normalisedPair);
  if (condBatch !== undefined) {
    execution.release(condBatch.scaleShift);
    execution.release(condBatch.gates);
  }
  current.label = "af3.difftx";
  return current;
}

// Every builder above, memoised on its arguments (`memoSource` in webgl/runtime/execution.js).
export const swiGluSource = memoSource(swiGluSourceUncached);
export const blendSliceSource = memoSource(blendSliceSourceUncached);
export const gateSliceSource = memoSource(gateSliceSourceUncached);
export const splitQkvgSource = memoSource(splitQkvgSourceUncached);
export const rowPartsSource = memoSource(rowPartsSourceUncached);
/** Fragments a row for the AdaLN's statistics: 768 channels are 192 texels, 12 a part. */
const ADALN_PARTS = 16;
export const splitQkvgPlanesSource = memoSource(splitQkvgPlanesSourceUncached);
export const swiGluPlanesSource = memoSource(swiGluPlanesSourceUncached);
export const superBlockLogitsSource = memoSource(superBlockLogitsSourceUncached);
