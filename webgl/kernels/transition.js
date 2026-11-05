/**
 * The transition: LayerNorm, expand with a ReLU, contract back.
 *
 *     x   = LayerNorm(activations)                 rows x channels
 *     h   = relu(x @ firstWeight + firstBias)      rows x hiddenChannels
 *     out = h @ secondWeight + secondBias          rows x channels
 *
 * Every stack in every model here runs one: AF2's MSA and pair transitions,
 * AF3's single and pair transitions, ESMFold2's and bioemu2's. It is three
 * draws made of kernels that already exist, which is the point of having built
 * them first.
 *
 * 🔴 ITS WEIGHTS ARE `[in][out]`, LIKE THE ATTENTION BLOCK'S AND UNLIKE THE
 * TRIANGLE'S. Read off the WGSL shader: `let w_row = parameters.weight_offset +
 * weight_k * parameters.columns;`. Two of the three operations ported so far
 * take `[in][out]` and one takes `[out][in]`, which is why `linear` refuses to
 * have a default.
 *
 * 🔴 THE HIDDEN ACTIVATION IS ROUNDED TO HALF, BECAUSE THE SHIPPED PATH DOES.
 * `webgpu/af2/evoformer/block.js` sets `hiddenStorage` to `"f16"` whenever the
 * hidden width is a multiple of four - every AF2 shape - and that is not gated
 * by `halfPrecisionAvailable` or by any knob, so a port that keeps f32 there
 * disagrees with the primary path by ~1e-4 on every transition. Measured
 * exactly that: a whole-block comparison read 1.045e-4 with this off, against a
 * 1.9e-8 floor. `hiddenPrecision: "f32"` is the arm for asking what the rounding
 * costs; the default matches.
 *
 * 🔴 AND THE RESIDUAL IS THE CALLER'S. `TransitionGpu` returns the transition's
 * own output, not the input plus it, and so does this - a block adds it. Worth
 * stating because a fragment shader cannot add in place (see
 * `webgl/runtime/elementwise.js`), so the caller needs a destination and
 * discovering that inside a stack is worse than reading it here.
 */
import { layerNorm } from "./layer-norm.js";
import { linear } from "./linear.js";

/** @typedef {import("../runtime/storage.js").Tensor} Tensor */

/** In the WGSL side's own spelling, and in its pack order. */
export const TRANSITION_WEIGHTS = [
  "layerNormScale", "layerNormOffset", "firstWeight", "firstBias", "secondWeight", "secondBias",
];

/**
 * @param {import("../runtime/execution.js").Execution} execution
 */
export function uploadTransitionWeights(execution, weights) {
  /** @type {Record<string, Tensor>} */
  const out = {};
  for (const name of TRANSITION_WEIGHTS) {
    const values = weights[name];
    if (values === undefined) throw new Error(`transition weights: ${name} is missing`);
    out[name] = execution.uploadWeight(values, `w.transition.${name}`);
  }
  return out;
}

export function releaseTransitionWeights(execution, weights) {
  for (const name of TRANSITION_WEIGHTS) execution.release(weights[name]);
}

/**
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{activations: Tensor, weights: Record<string, Tensor>, rows: number,
 *   channels: number, hiddenChannels: number, epsilon?: number,
 *   hiddenPrecision?: "f16" | "f32"}} call
 * @returns {Tensor} `rows x channels`; the caller owns it
 */
export function transition(execution, call) {
  const {
    activations, weights: w, rows, channels, hiddenChannels, epsilon = 1e-5,
    hiddenPrecision = hiddenChannels % 4 === 0 ? "f16" : "f32",
  } = call;
  if (activations.elements !== rows * channels) {
    throw new RangeError(
      `transition: activations hold ${activations.elements} floats, not ${rows} x ${channels}`);
  }

  const normalised = layerNorm(execution, {
    x: activations, weight: w.layerNormScale, bias: w.layerNormOffset,
    rows, channels, epsilon,
  });
  const hidden = linear(execution, {
    x: normalised, weight: w.firstWeight, bias: w.firstBias,
    rows, inChannels: channels, outChannels: hiddenChannels,
    activation: "relu", weightLayout: "in-major",
    roundHalf: hiddenPrecision === "f16",
  });
  execution.release(normalised);
  const output = linear(execution, {
    x: hidden, weight: w.secondWeight, bias: w.secondBias,
    rows, inChannels: hiddenChannels, outChannels: channels,
    weightLayout: "in-major",
  });
  execution.release(hidden);
  output.label = "transition";
  return output;
}
