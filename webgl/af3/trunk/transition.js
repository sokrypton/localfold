/**
 * AF3's transition block: a SwiGLU, not AF2's two Linears with a ReLU.
 *
 *     normalised = LayerNorm(x)                      AF3's FAST variance
 *     wide       = Linear(normalised)                [rows, 2 * intermediate]
 *     gated[i]   = swish(wide[i]) * wide[intermediate + i]
 *     out        = Linear(gated)                     [rows, channels]
 *
 * 🔴 THE SPLIT IS BLOCKED, AND ITS TWIN FORTY LINES AWAY IS INTERLEAVED.
 * `cpu/af3/trunk/pairformer.js` states both: "transition BLOCKED. transition1 is
 * (in, 2 * intermediate) and is reshaped (in, 2, intermediate), so the gate half
 * is the FIRST `intermediate` outputs and the value half the second" - where the
 * triangle multiplication's projection is TRANSPOSED before its reshape and so
 * interleaves. "Getting either backwards costs no error and no NaN. It permutes
 * channels and returns a plausible tensor."
 *
 * 🔴 AND THE WIDENING FACTOR IS NOT ALWAYS FOUR. The trunk's transitions use 4
 * (128 -> 512) and the TEMPLATE stack's use 2 (64 -> 128). Both are "a
 * transition block" and both read `transition1` and `transition2`; only the
 * shape of the weights says which, "so a wrong factor here reads them at the
 * wrong stride rather than failing". It is derived from the weight below rather
 * than defaulted, which turns that silent misread into a refusal.
 *
 * 🔴 AND THERE ARE NO BIASES. AF3 writes `bias_init=1.0` on its gates, which
 * reads as "this gate starts open" and is inert, because those Linears are
 * bias-free and the initialiser never materialises a parameter. "A reference
 * that helpfully added the missing biases would open every gate and be wrong
 * everywhere" - and so would a port.
 */
import { ONE_TARGET, PRELUDE, sized } from "../../runtime/execution.js";
import { layerNorm } from "../../kernels/layer-norm.js";
import { linear } from "../../kernels/linear.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */

export const TRANSITION_WEIGHTS = [
  "inputLayerNormScale", "inputLayerNormOffset", "transition1", "transition2",
];

/**
 * `swish(gate) * value` over a blocked `[rows, 2 * intermediate]` row.
 *
 * One draw rather than folding it into the second projection: a fragment there
 * owns four OUTPUT channels and sweeps the whole intermediate axis, so it would
 * recompute the swish `channels / 4` times over.
 */
export const swiGluSource = (rows, intermediate) => sized({ ELEMENTS: rows * intermediate, INTER4: intermediate / 4 }, `${PRELUDE}${ONE_TARGET}
uniform sampler2D wideTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % INTER4;
  int row = texel / INTER4;
  // The gate half is the FIRST intermediate outputs and the value half the
  // second - blocked, not interleaved. No backticks in a GLSL comment: it lives
  // inside a JS template literal and one would end it.
  int base = row * 2 * INTER4;
  vec4 gate = fetch4(wideTex, base + group);
  vec4 value = fetch4(wideTex, base + INTER4 + group);
  result = (gate / (vec4(1.0) + exp(-gate))) * value;
}`);

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 */
export function uploadTransitionWeights(execution, weights) {
  /** @type {Record<string, Tensor>} */
  const out = {};
  for (const name of TRANSITION_WEIGHTS) {
    const values = weights[name];
    if (values === undefined) throw new Error(`af3 transition weights: ${name} is missing`);
    out[name] = execution.uploadWeight(values, `w.af3.transition.${name}`);
  }
  return out;
}

export function releaseTransitionWeights(execution, weights) {
  for (const name of TRANSITION_WEIGHTS) execution.release(weights[name]);
}

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{x: Tensor, weights: Record<string, Tensor>, rows: number, channels: number,
 *   epsilon?: number}} call
 * @returns {Tensor} `rows * channels`, the update; the caller adds it
 */
export function transition(execution, call) {
  const { x, weights: w, rows, channels, epsilon = 1e-5 } = call;
  if (x.elements !== rows * channels) {
    throw new RangeError(
      `af3 transition: x holds ${x.elements} floats, not ${rows} x ${channels}`);
  }
  // 🔴 DERIVED FROM THE WEIGHT, NEVER DEFAULTED. `transition1` is
  // `[channels, 2 * intermediate]`, so the factor falls out of its length - and
  // a bundle whose factor is 2 rather than 4 is then read correctly instead of
  // at the wrong stride.
  const intermediate = w.transition1.elements / (channels * 2);
  if (!Number.isInteger(intermediate) || intermediate % 4 !== 0) {
    throw new RangeError(
      `af3 transition: transition1 holds ${w.transition1.elements} floats, which is not`
      + ` ${channels} x 2 x a multiple of four`);
  }
  if (w.transition2.elements !== intermediate * channels) {
    throw new RangeError(
      `af3 transition: transition2 holds ${w.transition2.elements} floats, not`
      + ` ${intermediate} x ${channels}`);
  }

  const normalised = layerNorm(execution, {
    x, weight: w.inputLayerNormScale, bias: w.inputLayerNormOffset,
    rows, channels, epsilon, variance: "fast",
  });
  const wide = linear(execution, {
    x: normalised, weight: w.transition1, rows,
    inChannels: channels, outChannels: intermediate * 2, weightLayout: "in-major",
  });
  execution.release(normalised);
  const gated = execution.allocate(rows * intermediate, "af3.transition.gated");
  execution.dispatch({
    source: swiGluSource(rows, intermediate),
    output: gated,
    inputs: { wideTex: wide },
    label: `af3.transition.swiglu:${rows}x${intermediate}`,
  });
  execution.release(wide);
  const out = linear(execution, {
    x: gated, weight: w.transition2, rows,
    inChannels: intermediate, outChannels: channels, weightLayout: "in-major",
  });
  execution.release(gated);
  out.label = "af3.transition";
  return out;
}
