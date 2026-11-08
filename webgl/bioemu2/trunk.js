/**
 * BioEmu2's trunk in WebGL2: a PairMixer of four MSA layers and 48 pair-only
 * blocks, run four times.
 *
 *     per pass:  s = s_init + s_recycle(LN s_prev)
 *                z = z_init + z_recycle(LN z_prev)
 *                four times:  m += pairWeightedAveraging(m, z)
 *                             m += msaTransition(m)
 *                             z += outerProductMean(m)
 *                             z  = pairOnlyBlock(z)
 *                then 48 more pair-only blocks
 *
 * 🔴 THE SINGLE TRACK IS TOUCHED BY THE RECYCLE AND BY NOTHING ELSE. All 52
 * blocks are pair-only, so `s` is `s_init + s_recycle(LN s_prev)` and then
 * constant for the pass - which is why a port that threaded a single through the
 * blocks would be a different model that runs, and why the only thing a pass
 * does to `s` is read it on the next one.
 *
 * 🔴 AND THE MSA LAYER RUNS BEFORE ITS PAIR BLOCK, NOT BESIDE IT. The reference
 * drives the pair stack with `beforeBlock(l)`, so layer l's MSA half runs and
 * then pair block l - the first four blocks of 52 have an MSA half and the other
 * 48 do not.
 *
 * 🔴 AND THE PAIR-ONLY BLOCK IS ESMFold2's, which is why this file is short: the
 * two triangle multiplications and the transition are `esmfold2/trunk.js`'s
 * `foldingTrunk` on weights the bundle carries in AF3's pairformer layout.
 *
 * 🔴 AND THE OUTER PRODUCT MEAN DIVIDES BY THE NUMBER OF VALID ROWS, a SCALAR,
 * where AF2's divides by each pair's own mask product. Using AF2's kernel here
 * would be the same arithmetic with a different denominator, which is why this
 * one is written out.
 *
 * 🔴 AND THE PAIR ACCUMULATOR IS f32 EVERYWHERE, which this backend gets for
 * free and the WGSL one had to be told: "BioEmu2's pair reaches ~3e5 (6MRR,
 * float32 arm) - past float16's 65,504 - where ESMFold2's, which this trunk was
 * tuned for, stays small", and a device default of f16 accumulation returned a
 * fold of NaN. There is no f16 in WebGL2 without an extension nobody can rely
 * on, so the hazard does not exist here; it is worth knowing that it exists at
 * all before anyone adds one.
 */
import { ONE_TARGET, PRELUDE, memoSource } from "../runtime/execution.js";
import { BIOEMU2 } from "../../shared/bioemu2/weights.js";
import { layerNorm } from "../kernels/layer-norm.js";
import { linear } from "../kernels/linear.js";
import { foldingTrunk } from "../esmfold2/trunk.js";
import { productSource, sumSource } from "./denoiser.js";

/** @typedef {import("../runtime/storage.js").Tensor} Tensor */

/**
 * The pair-weighted averaging's weights: a softmax over the KEY token, per head.
 *
 * 🔴 THE LOGITS ARE HEAD-FIRST, `(head, i, j)`, which is how the reference lays
 * them out and the softmax then runs over the last axis. Laid out `(i, j, head)`
 * the softmax would normalise over the heads and still sum to one.
 */
const averagingWeightsSourceUncached = (tokens, heads) => `${PRELUDE}${ONE_TARGET}
const int L = ${tokens};
const int HEADS = ${heads};
const int ELEMENTS = ${heads * tokens * tokens};
uniform sampler2D logitsTex;

float at(int element) {
  int j = element % L;
  int row = element / L;
  int base = row * L;
  float largest = -1e30;
  for (int k = 0; k < L; ++k) largest = max(largest, fetch1(logitsTex, base + k));
  float total = 0.0;
  for (int k = 0; k < L; ++k) total += exp(fetch1(logitsTex, base + k) - largest);
  return exp(fetch1(logitsTex, base + j) - largest) / total;
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
 * `out[s][i][head][d] = sum_j w[head][i][j] * v[s][j][head][d]`.
 *
 * The weights are shared by every MSA row - they come from the pair - and the
 * value is per row, which is what makes this a gather over j rather than an
 * attention.
 */
const averagedValuesSourceUncached = (sequences, tokens, heads, headDim) => `${PRELUDE}${ONE_TARGET}
const int L = ${tokens};
const int HEADS = ${heads};
const int HEAD_DIM = ${headDim};
const int WIDTH = ${heads * headDim};
const int ELEMENTS = ${sequences * tokens * heads * headDim};
uniform sampler2D weightsTex;
uniform sampler2D valuesTex;

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  int row = base / WIDTH;
  int c = base - row * WIDTH;
  int head = c / HEAD_DIM;
  int sequence = row / L;
  int i = row - sequence * L;

  vec4 total = vec4(0.0);
  for (int j = 0; j < L; ++j) {
    float weight = fetch1(weightsTex, (head * L + i) * L + j);
    total += weight * fetch4(valuesTex, ((sequence * L + j) * WIDTH + c) / 4);
  }
  result = total;
}`;

/**
 * `outer[i][j][c * 32 + d] = sum_s a[s][i][c] * b[s][j][d] / valid`.
 *
 * `a` is the first half of each row's 64 channels and `b` the second.
 */
const outerSourceUncached = (sequences, tokens, half) => `${PRELUDE}${ONE_TARGET}
const int L = ${tokens};
const int S = ${sequences};
const int HALF = ${half};
const int WIDTH = ${half * half};
const int ELEMENTS = ${tokens * tokens * half * half};
uniform sampler2D abTex;
uniform float inverseValid;

float at(int element) {
  int channel = element % WIDTH;
  int pair = element / WIDTH;
  int i = pair / L;
  int j = pair - i * L;
  int c = channel / HALF;
  int d = channel - c * HALF;
  float total = 0.0;
  for (int s = 0; s < S; ++s) {
    total += fetch1(abTex, ((s * L + i) * 2 * HALF) + c)
      * fetch1(abTex, ((s * L + j) * 2 * HALF) + HALF + d);
  }
  return total * inverseValid;
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
 * One MSA layer's MSA half and its pair half.
 *
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{msa: Tensor, pair: Tensor, sequences: number, tokens: number,
 *   valid: number, weights: any, epsilon?: number}} call
 * @returns {{msa: Tensor, pair: Tensor}} new tensors; the caller's are untouched
 */
export function msaLayer(execution, call) {
  const {
    msa: msaIn, pair, sequences, tokens, valid, weights: w, epsilon = 1e-5,
  } = call;
  const msaChannels = BIOEMU2.msa;
  const pairChannels = BIOEMU2.pair;
  const heads = BIOEMU2.msaHeads;
  const headDim = pairChannels / heads;
  const rows = sequences * tokens;
  const pairs = tokens * tokens;

  // --- pair-weighted averaging ------------------------------------------
  const normalisedMsa = layerNorm(execution, {
    x: msaIn, weight: w.pwa.normMScale, bias: w.pwa.normMOffset,
    rows, channels: msaChannels, epsilon,
  });
  const values = linear(execution, {
    x: normalisedMsa, weight: w.pwa.projM, rows,
    inChannels: msaChannels, outChannels: pairChannels, weightLayout: "out-major",
  });
  const rawGate = linear(execution, {
    x: normalisedMsa, weight: w.pwa.projG, rows,
    inChannels: msaChannels, outChannels: pairChannels, weightLayout: "out-major",
  });
  execution.release(normalisedMsa);
  const gate = execution.allocate(rows * pairChannels, "bioemu2.pwa.gate");
  execution.dispatch({
    source: logisticSource(rows * pairChannels),
    output: gate,
    inputs: { xTex: rawGate },
    label: `bioemu2.pwa.gate:${rows}x${pairChannels}`,
  });
  execution.release(rawGate);
  const normalisedPair = layerNorm(execution, {
    x: pair, weight: w.pwa.normZScale, bias: w.pwa.normZOffset,
    rows: pairs, channels: pairChannels, epsilon,
  });
  // The logits, head-first: one scalar per (head, i, j).
  const flatLogits = linear(execution, {
    x: normalisedPair, weight: w.pwa.projZ, rows: pairs,
    inChannels: pairChannels, outChannels: heads, weightLayout: "out-major",
  });
  execution.release(normalisedPair);
  const headFirst = execution.allocate(heads * pairs, "bioemu2.pwa.logits");
  execution.dispatch({
    source: headFirstSource(tokens, heads),
    output: headFirst,
    inputs: { flatTex: flatLogits },
    label: `bioemu2.pwa.head-first:${tokens}x${heads}`,
  });
  execution.release(flatLogits);
  const weights = execution.allocate(heads * pairs, "bioemu2.pwa.weights");
  execution.dispatch({
    source: averagingWeightsSource(tokens, heads),
    output: weights,
    inputs: { logitsTex: headFirst },
    label: `bioemu2.pwa.softmax:${tokens}x${heads}`,
  });
  execution.release(headFirst);
  const averaged = execution.allocate(rows * pairChannels, "bioemu2.pwa.averaged");
  execution.dispatch({
    source: averagedValuesSource(sequences, tokens, heads, headDim),
    output: averaged,
    inputs: { weightsTex: weights, valuesTex: values },
    label: `bioemu2.pwa.average:${rows}x${pairChannels}`,
  });
  execution.release(weights);
  execution.release(values);
  const gated = execution.allocate(rows * pairChannels, "bioemu2.pwa.gated");
  execution.dispatch({
    source: productSource(rows * pairChannels),
    output: gated,
    inputs: { aTex: averaged, bTex: gate },
    label: `bioemu2.pwa.gate-apply:${rows}x${pairChannels}`,
  });
  execution.release(averaged);
  execution.release(gate);
  const update = linear(execution, {
    x: gated, weight: w.pwa.projO, rows,
    inChannels: pairChannels, outChannels: msaChannels, weightLayout: "out-major",
  });
  execution.release(gated);
  let msa = execution.allocate(rows * msaChannels, "bioemu2.msa");
  execution.dispatch({
    source: sumSource(rows * msaChannels),
    output: msa,
    inputs: { aTex: msaIn, bTex: update },
    label: `bioemu2.pwa.residual:${rows}x${msaChannels}`,
  });
  execution.release(update);

  // --- the MSA transition, with its own norm ----------------------------
  {
    const normalised = layerNorm(execution, {
      x: msa, weight: w.transition.normScale, bias: w.transition.normOffset,
      rows, channels: msaChannels, epsilon,
    });
    const a = linear(execution, {
      x: normalised, weight: w.transition.fc1, rows,
      inChannels: msaChannels, outChannels: 4 * msaChannels, weightLayout: "out-major",
    });
    const b = linear(execution, {
      x: normalised, weight: w.transition.fc2, rows,
      inChannels: msaChannels, outChannels: 4 * msaChannels, weightLayout: "out-major",
    });
    execution.release(normalised);
    const activated = execution.allocate(rows * 4 * msaChannels, "bioemu2.msa.silu");
    execution.dispatch({
      source: siluProductSource(rows * 4 * msaChannels),
      output: activated,
      inputs: { aTex: a, bTex: b },
      label: `bioemu2.msa.transition-gate:${rows}x${4 * msaChannels}`,
    });
    execution.release(a);
    execution.release(b);
    const projected = linear(execution, {
      x: activated, weight: w.transition.fc3, rows,
      inChannels: 4 * msaChannels, outChannels: msaChannels, weightLayout: "out-major",
    });
    execution.release(activated);
    const next = execution.allocate(rows * msaChannels, "bioemu2.msa");
    execution.dispatch({
      source: sumSource(rows * msaChannels),
      output: next,
      inputs: { aTex: msa, bTex: projected },
      label: `bioemu2.msa.residual:${rows}x${msaChannels}`,
    });
    execution.release(projected);
    execution.release(msa);
    msa = next;
  }

  // --- the outer product mean -------------------------------------------
  const normalisedForOpm = layerNorm(execution, {
    x: msa, weight: w.opm.normScale, bias: w.opm.normOffset,
    rows, channels: msaChannels, epsilon,
  });
  const half = 32;
  const ab = linear(execution, {
    x: normalisedForOpm, weight: w.opm.projAb, rows,
    inChannels: msaChannels, outChannels: 2 * half, weightLayout: "out-major",
  });
  execution.release(normalisedForOpm);
  const outer = execution.allocate(pairs * half * half, "bioemu2.opm.outer");
  execution.dispatch({
    source: outerSource(sequences, tokens, half),
    output: outer,
    inputs: { abTex: ab },
    floats: { inverseValid: 1 / valid },
    label: `bioemu2.opm.outer:${pairs}x${half * half}`,
  });
  execution.release(ab);
  const pairUpdate = linear(execution, {
    x: outer, weight: w.opm.projO, bias: w.opm.projOBias, rows: pairs,
    inChannels: half * half, outChannels: pairChannels, weightLayout: "out-major",
  });
  execution.release(outer);
  const nextPair = execution.allocate(pairs * pairChannels, "bioemu2.z");
  execution.dispatch({
    source: sumSource(pairs * pairChannels),
    output: nextPair,
    inputs: { aTex: pair, bTex: pairUpdate },
    label: `bioemu2.opm.residual:${pairs}x${pairChannels}`,
  });
  execution.release(pairUpdate);
  return { msa, pair: nextPair };
}

/** `out = sigmoid(x)`. */
const logisticSourceUncached = (elements) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${elements};
uniform sampler2D xTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(1.0) / (vec4(1.0) + exp(-fetch4(xTex, texel)));
}`;

/** `out = silu(a) * b`. */
const siluProductSourceUncached = (elements) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${elements};
uniform sampler2D aTex;
uniform sampler2D bTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  vec4 a = fetch4(aTex, texel);
  result = (a / (vec4(1.0) + exp(-a))) * fetch4(bTex, texel);
}`;

/** `(pair, head)` to `(head, pair)`, which is the axis the softmax runs over. */
const headFirstSourceUncached = (tokens, heads) => `${PRELUDE}${ONE_TARGET}
const int PAIRS = ${tokens * tokens};
const int HEADS = ${heads};
const int ELEMENTS = ${heads * tokens * tokens};
uniform sampler2D flatTex;

float at(int element) {
  int head = element / PAIRS;
  int pair = element - head * PAIRS;
  return fetch1(flatTex, pair * HEADS + head);
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
 * One trunk pass: the recycle, then 52 pair blocks with the first four carrying
 * an MSA layer.
 *
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{zInit: Tensor, sInit: Tensor, m0: Tensor, zPrev: Tensor,
 *   sPrev: Tensor, pairMask: Tensor, tokens: number, sequences: number,
 *   valid: number, weights: any, msaBlocks: any[], mixerBlocks: any[],
 *   epsilon?: number, onBlock?: (index: number) => void}} call
 * @returns {{pair: Tensor, single: Tensor}} the pass's z and s, the caller's to
 *   release - and its inputs for the next pass
 */
export function trunkPass(execution, call) {
  const {
    zInit, sInit, m0, zPrev, sPrev, pairMask, tokens, sequences, valid,
    weights: w, msaBlocks, mixerBlocks, epsilon = 1e-5,
  } = call;
  const pairs = tokens * tokens;
  const pairChannels = BIOEMU2.pair;
  const single = BIOEMU2.single;

  // s = s_init + s_recycle(LN s_prev); z = z_init + z_recycle(LN z_prev).
  const normalisedSingle = layerNorm(execution, {
    x: sPrev, weight: w.sNormScale, bias: w.sNormOffset,
    rows: tokens, channels: single, epsilon,
  });
  const singleUpdate = linear(execution, {
    x: normalisedSingle, weight: w.sRecycle, rows: tokens,
    inChannels: single, outChannels: single, weightLayout: "out-major",
  });
  execution.release(normalisedSingle);
  const s = execution.allocate(tokens * single, "bioemu2.s");
  execution.dispatch({
    source: sumSource(tokens * single),
    output: s,
    inputs: { aTex: sInit, bTex: singleUpdate },
    label: `bioemu2.recycle.single:${tokens}x${single}`,
  });
  execution.release(singleUpdate);
  const normalisedPair = layerNorm(execution, {
    x: zPrev, weight: w.zNormScale, bias: w.zNormOffset,
    rows: pairs, channels: pairChannels, epsilon,
  });
  const pairUpdate = linear(execution, {
    x: normalisedPair, weight: w.zRecycle, rows: pairs,
    inChannels: pairChannels, outChannels: pairChannels, weightLayout: "out-major",
  });
  execution.release(normalisedPair);
  let z = execution.allocate(pairs * pairChannels, "bioemu2.z");
  execution.dispatch({
    source: sumSource(pairs * pairChannels),
    output: z,
    inputs: { aTex: zInit, bTex: pairUpdate },
    label: `bioemu2.recycle.pair:${pairs}x${pairChannels}`,
  });
  execution.release(pairUpdate);

  // The MSA state starts from m0 every pass - the recycle carries z and s, not m.
  let msa = execution.allocate(sequences * tokens * BIOEMU2.msa, "bioemu2.m");
  execution.dispatch({
    source: copySource(sequences * tokens * BIOEMU2.msa),
    output: msa,
    inputs: { xTex: m0 },
    label: `bioemu2.m0:${sequences * tokens}x${BIOEMU2.msa}`,
  });

  // 🔴 THE MSA HALF RUNS BEFORE ITS PAIR BLOCK, and only the first four blocks
  // have one.
  for (const [index, block] of msaBlocks.entries()) {
    const stepped = msaLayer(execution, {
      msa, pair: z, sequences, tokens, valid, weights: w.msa[index], epsilon,
    });
    execution.release(msa);
    execution.release(z);
    msa = stepped.msa;
    z = stepped.pair;
    const next = foldingTrunk(execution, {
      pair: z, mask: pairMask, tokens, channels: pairChannels,
      blocks: [block], epsilon,
    });
    execution.release(z);
    z = next;
    call.onBlock?.(index);
  }
  execution.release(msa);
  for (const [index, block] of mixerBlocks.entries()) {
    const next = foldingTrunk(execution, {
      pair: z, mask: pairMask, tokens, channels: pairChannels,
      blocks: [block], epsilon,
    });
    execution.release(z);
    z = next;
    call.onBlock?.(msaBlocks.length + index);
  }
  z.label = "bioemu2.z";
  s.label = "bioemu2.s";
  return { pair: z, single: s };
}

/** `out = x`. */
const copySourceUncached = (elements) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${elements};
uniform sampler2D xTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = fetch4(xTex, texel);
}`;

// Every builder above, memoised on its arguments (`memoSource` in webgl/runtime/execution.js).
export const averagingWeightsSource = memoSource(averagingWeightsSourceUncached);
export const averagedValuesSource = memoSource(averagedValuesSourceUncached);
export const outerSource = memoSource(outerSourceUncached);
export const logisticSource = memoSource(logisticSourceUncached);
export const siluProductSource = memoSource(siluProductSourceUncached);
export const headFirstSource = memoSource(headFirstSourceUncached);
export const copySource = memoSource(copySourceUncached);
