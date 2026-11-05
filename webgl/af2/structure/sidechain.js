/**
 * The sidechain torsion head: seven angles a residue, as normalised (x, y)
 * pairs.
 *
 *     act  = Linear(relu(final))  +  Linear(relu(initial))
 *     act += Linear(relu(Linear(relu(act))))        twice, two weight sets
 *     raw  = Linear(relu(act))                      length x 14
 *     angle = (x, y) * inversesqrt(max(1e-12, x*x + y*y))
 *
 * 🔴 IT READS BOTH SINGLE REPRESENTATIONS, THE FINAL AND THE INITIAL, and they
 * are the same shape. The initial one is the NORMALISED projection from
 * `structure/initialize.js` - not its activations, which have been through one
 * more projection - and the two enter through DIFFERENT weights that are then
 * added. Passing the same tensor twice, or the wrong one of the two, is a
 * plausible set of torsions.
 *
 * 🔴 AND EVERY ReLU IS ON A LINEAR'S INPUT, NOT ITS OUTPUT. Everywhere else in
 * these models the activation belongs to the projection that produced the value
 * and `linear` folds it in for free; here the pattern is `Linear(relu(x))` with
 * `x` the output of an ADD, so folding it into the previous projection would put
 * it on the wrong side of the residual.
 *
 * 🔴 AND THE ANGLE NORMALISATION GUARDS AT 1e-12, NOT AT ZERO.
 * `inverseSqrt(max(1e-12, x*x + y*y))` - so a torsion whose two components are
 * both zero comes out as (0, 0) rather than NaN, and the structure that reads it
 * gets an unrotated bond instead of a hole. `unnormalizedAngles` is returned
 * too because AF2's loss reads it and the page reports it.
 */
import { ONE_TARGET, PRELUDE } from "../../runtime/execution.js";
import { linear } from "../../kernels/linear.js";
import { add, relu } from "../../runtime/elementwise.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */
/** @typedef {{weight: Tensor, bias: Tensor}} Projection */
/**
 * @typedef {{inputWeight: Tensor, inputBias: Tensor, initialInputWeight: Tensor,
 *   initialInputBias: Tensor, angleWeight: Tensor, angleBias: Tensor,
 *   residual1: Projection[], residual2: Projection[]}} SidechainWeights
 */

export const SIDECHAIN_WEIGHTS = [
  "inputWeight", "inputBias", "initialInputWeight", "initialInputBias",
  "angleWeight", "angleBias",
];

/**
 * Two angles to a texel, which is exact: there are 14 components a residue and a
 * texel starts on an even element, so its four always cover two whole pairs.
 */
export const normaliseAnglesSource = (length) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${length * 14};
uniform sampler2D rawTex;

vec2 unit(vec2 v) {
  return v * inversesqrt(max(1e-12, dot(v, v)));
}

void main() {
  int texel = linearTexel();
  int base = texel * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  vec4 raw = fetch4(rawTex, texel);
  vec2 first = unit(raw.xy);
  vec2 second = base + 2 < ELEMENTS ? unit(raw.zw) : vec2(0.0);
  result = vec4(first, second);
}`;

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 */
export function uploadSidechainWeights(execution, weights) {
  /** @type {any} */
  const out = {};
  for (const name of SIDECHAIN_WEIGHTS) {
    const values = weights[name];
    if (values === undefined) throw new Error(`sidechain weights: ${name} is missing`);
    out[name] = execution.upload(values, `w.sidechain.${name}`);
  }
  // Two residual blocks, two projections each, as arrays on the WGSL side.
  out.residual1 = weights.residual1Weights.map((value, index) => ({
    weight: execution.upload(value, `w.sidechain.r1w${index}`),
    bias: execution.upload(weights.residual1Biases[index], `w.sidechain.r1b${index}`),
  }));
  out.residual2 = weights.residual2Weights.map((value, index) => ({
    weight: execution.upload(value, `w.sidechain.r2w${index}`),
    bias: execution.upload(weights.residual2Biases[index], `w.sidechain.r2b${index}`),
  }));
  return out;
}

export function releaseSidechainWeights(execution, weights) {
  for (const name of SIDECHAIN_WEIGHTS) execution.release(weights[name]);
  /** @type {Projection[][]} */
  for (const group of [weights.residual1, weights.residual2]) {
    for (const { weight, bias } of group) {
      execution.release(weight);
      execution.release(bias);
    }
  }
}

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{activations: Tensor, initialActivations: Tensor, weights: SidechainWeights,
 *   length: number, inputChannels: number, hiddenChannels: number}} call
 * @returns {{angles: Tensor, unnormalizedAngles: Tensor}} the caller owns both
 */
export function sidechainAngles(execution, call) {
  const {
    activations, initialActivations, weights: w, length, inputChannels, hiddenChannels,
  } = call;
  /** @type {[string, Tensor][]} */
  const inputs = [["activations", activations], ["initialActivations", initialActivations]];
  for (const [name, tensor] of inputs) {
    if (tensor.elements !== length * inputChannels) {
      throw new RangeError(
        `sidechainAngles: ${name} holds ${tensor.elements} floats, not `
        + `${length} x ${inputChannels}`);
    }
  }

  /** `Linear(relu(x))`, the pattern this head is made of. */
  const project = (x, weight, bias, inChannels, outChannels) => {
    const activated = relu(execution, x);
    const out = linear(execution, {
      x: activated, weight, bias, rows: length, inChannels, outChannels,
      weightLayout: "in-major",
    });
    execution.release(activated);
    return out;
  };

  const fromFinal = project(
    activations, w.inputWeight, w.inputBias, inputChannels, hiddenChannels);
  const fromInitial = project(
    initialActivations, w.initialInputWeight, w.initialInputBias, inputChannels, hiddenChannels);
  let act = add(execution, fromFinal, fromInitial);
  execution.release(fromFinal);
  execution.release(fromInitial);

  for (const group of [w.residual1, w.residual2]) {
    const first = project(
      act, group[0].weight, group[0].bias, hiddenChannels, hiddenChannels);
    const second = project(
      first, group[1].weight, group[1].bias, hiddenChannels, hiddenChannels);
    execution.release(first);
    const next = add(execution, act, second);
    execution.release(second);
    execution.release(act);
    act = next;
  }

  const unnormalizedAngles = project(act, w.angleWeight, w.angleBias, hiddenChannels, 14);
  execution.release(act);
  const angles = execution.allocate(length * 14, "sidechain.angles");
  execution.dispatch({
    source: normaliseAnglesSource(length),
    output: angles,
    inputs: { rawTex: unnormalizedAngles },
    label: `sidechain.normalise:${length}`,
  });
  unnormalizedAngles.label = "sidechain.unnormalized";
  return { angles, unnormalizedAngles };
}
