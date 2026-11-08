/**
 * One structure-module iteration: the IPA update folded in, a transition, and
 * the backbone frame advanced.
 *
 *     normalised = LayerNorm(activations + ipaUpdate)
 *     t          = Linear(relu(.)) x2 then Linear(.)      three projections
 *     activations' = LayerNorm(normalised + t)
 *     update     = Linear(activations')                   six floats a residue
 *     affine'    = advance(affine, update)
 *
 * 🔴 THE FRAME UPDATE IS A QUATERNION PRODUCT WITH THE REAL PART IMPLIED, NOT A
 * ROTATION BY A GIVEN QUATERNION. The update's first three floats are a vector
 * `v`, and the increment is `(-dot(q.yzw, v), q.x * v + cross(q.yzw, v))` - the
 * product of `q` with `(1, v)`, minus `q` - added to `q` and then NORMALISED.
 * Reading those three floats as a rotation, or forgetting the normalise, gives a
 * frame that drifts off the unit sphere over eight iterations and a fold that
 * degrades with length rather than failing.
 *
 * 🔴 AND THE TRANSLATION IS ROTATED BY THE OLD FRAME, NOT THE NEW ONE.
 * `translation + rotate(q, update[3..6])` with `q` as it was - which is what
 * makes the update a local displacement in the frame that produced it.
 *
 * The add-then-normalise is two draws here where the WGSL fuses it into one: the
 * sum is a single f32 operation either way, so the values are the same, and a
 * fragment shader cannot hold a row's statistics across invocations anyway.
 */
import { ONE_TARGET, PRELUDE, memoSource } from "../../runtime/execution.js";
import { layerNorm } from "../../kernels/layer-norm.js";
import { linear } from "../../kernels/linear.js";
import { add, relu } from "../../runtime/elementwise.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */

export const ITERATION_WEIGHTS = [
  "attentionNormScale", "attentionNormOffset",
  "transitionNormScale", "transitionNormOffset",
  "affineWeight", "affineBias",
];

const advanceAffineSourceUncached = (length) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${length * 7};
uniform sampler2D affineTex;
uniform sampler2D updateTex;

float componentOf(int element) {
  int residue = element / 7;
  int slot = element - residue * 7;
  int affineBase = residue * 7;
  int updateBase = residue * 6;
  vec4 q = vec4(fetch1(affineTex, affineBase), fetch1(affineTex, affineBase + 1),
    fetch1(affineTex, affineBase + 2), fetch1(affineTex, affineBase + 3));
  if (slot < 4) {
    vec3 v = vec3(fetch1(updateTex, updateBase), fetch1(updateTex, updateBase + 1),
      fetch1(updateTex, updateBase + 2));
    // The product of q with (1, v), less q: the real part is implied.
    vec4 delta = vec4(-dot(q.yzw, v), q.x * v + cross(q.yzw, v));
    vec4 next = normalize(q + delta);
    return slot == 0 ? next.x : slot == 1 ? next.y : slot == 2 ? next.z : next.w;
  }
  vec3 translation = vec3(fetch1(affineTex, affineBase + 4),
    fetch1(affineTex, affineBase + 5), fetch1(affineTex, affineBase + 6));
  vec3 local = vec3(fetch1(updateTex, updateBase + 3),
    fetch1(updateTex, updateBase + 4), fetch1(updateTex, updateBase + 5));
  // Rotated by the OLD frame - the update is a displacement in the frame that
  // produced it.
  vec3 u = q.yzw;
  vec3 rotated = 2.0 * dot(u, local) * u + (q.x * q.x - dot(u, u)) * local
    + 2.0 * q.x * cross(u, local);
  vec3 next = translation + rotated;
  return slot == 4 ? next.x : slot == 5 ? next.y : next.z;
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    componentOf(base),
    base + 1 < ELEMENTS ? componentOf(base + 1) : 0.0,
    base + 2 < ELEMENTS ? componentOf(base + 2) : 0.0,
    base + 3 < ELEMENTS ? componentOf(base + 3) : 0.0);
}`;

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 */
export function uploadIterationWeights(execution, weights) {
  /** @type {any} */
  const out = {};
  for (const name of ITERATION_WEIGHTS) {
    const values = weights[name];
    if (values === undefined) throw new Error(`iteration weights: ${name} is missing`);
    out[name] = execution.uploadWeight(values, `w.iteration.${name}`);
  }
  out.transition = weights.transitionWeights.map((value, index) => ({
    weight: execution.uploadWeight(value, `w.iteration.t${index}`),
    bias: execution.uploadWeight(weights.transitionBiases[index], `w.iteration.tb${index}`),
  }));
  return out;
}

export function releaseIterationWeights(execution, weights) {
  for (const name of ITERATION_WEIGHTS) execution.release(weights[name]);
  for (const { weight, bias } of weights.transition) {
    execution.release(weight);
    execution.release(bias);
  }
}

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{activations: Tensor, update: Tensor, affine: Tensor, weights: any,
 *   length: number, channels: number, epsilon?: number}} call
 * @returns {{activations: Tensor, affine: Tensor}} new tensors; the inputs are untouched
 */
export function structureIteration(execution, call) {
  const { activations, update, affine, weights: w, length, channels, epsilon = 1e-5 } = call;

  const addNormalise = (left, right, scale, offset) => {
    const summed = add(execution, left, right);
    const out = layerNorm(execution, {
      x: summed, weight: scale, bias: offset, rows: length, channels, epsilon,
    });
    execution.release(summed);
    return out;
  };

  const normalised = addNormalise(
    activations, update, w.attentionNormScale, w.attentionNormOffset);

  // Three projections, a ReLU on the input of the second and third.
  let hidden = linear(execution, {
    x: normalised, weight: w.transition[0].weight, bias: w.transition[0].bias,
    rows: length, inChannels: channels, outChannels: channels,
    activation: "relu", weightLayout: "in-major",
  });
  for (const index of [1, 2]) {
    const next = linear(execution, {
      x: hidden, weight: w.transition[index].weight, bias: w.transition[index].bias,
      rows: length, inChannels: channels, outChannels: channels,
      ...(index === 1 ? { activation: "relu" } : {}),
      weightLayout: "in-major",
    });
    execution.release(hidden);
    hidden = next;
  }

  const nextActivations = addNormalise(
    normalised, hidden, w.transitionNormScale, w.transitionNormOffset);
  execution.release(normalised);
  execution.release(hidden);

  const affineUpdate = linear(execution, {
    x: nextActivations, weight: w.affineWeight, bias: w.affineBias,
    rows: length, inChannels: channels, outChannels: 6, weightLayout: "in-major",
  });
  const nextAffine = execution.allocate(length * 7, "structure.affine");
  execution.dispatch({
    source: advanceAffineSource(length),
    output: nextAffine,
    inputs: { affineTex: affine, updateTex: affineUpdate },
    label: `structure.advance-affine:${length}`,
  });
  execution.release(affineUpdate);
  nextActivations.label = "structure.activations";
  nextAffine.label = "structure.affine";
  return { activations: nextActivations, affine: nextAffine };
}

// Every builder above, memoised on its arguments (`memoSource` in webgl/runtime/execution.js).
export const advanceAffineSource = memoSource(advanceAffineSourceUncached);
