/**
 * The structure module's starting point: the single representation from the
 * MSA's first row, and every residue's frame at the identity.
 *
 *     single     = Linear(msaFirstRow)                 length x structureChannels
 *     initial    = LayerNorm(single)
 *     activations = Linear(initial)
 *     affine[i]  = (1, 0, 0, 0, 0, 0, 0)
 *
 * 🔴 THREE TENSORS COME OUT AND ALL THREE ARE READ LATER, which is why this
 * returns them rather than just the activations. `single` is the projection
 * before the norm; `initialRepresentation` is the NORMALISED one, and the
 * sidechain head reads exactly that, not the activations - AF2's torsion head
 * takes the initial single representation alongside the final one. Returning the
 * wrong one of the two would be a sidechain head reading a tensor of the right
 * shape that has been through one more projection.
 *
 * 🔴 AND AN AFFINE IS SEVEN FLOATS: A QUATERNION THEN A TRANSLATION. The
 * identity is `(1, 0, 0, 0)` and not `(0, 0, 0, 1)`, which is the other
 * convention in common use and would start every residue rotated by 180 degrees
 * about an axis the data never names. `webgpu/af2/structure/initialize.js`'s
 * `AFFINE_INIT_SHADER` writes 1.0 at `base` and zero across the rest.
 */
import { ONE_TARGET, PRELUDE, memoSource } from "../../runtime/execution.js";
import { layerNorm } from "../../kernels/layer-norm.js";
import { linear } from "../../kernels/linear.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */

export const STRUCTURE_INITIALIZE_WEIGHTS = [
  "singleProjectionWeight", "singleProjectionBias",
  "singleNormScale", "singleNormOffset",
  "initialProjectionWeight", "initialProjectionBias",
];

/** `affine[i] = (1, 0, 0, 0, 0, 0, 0)` - quaternion identity, zero translation. */
const affineIdentitySourceUncached = (length) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${length * 7};
void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    base % 7 == 0 ? 1.0 : 0.0,
    base + 1 < ELEMENTS && (base + 1) % 7 == 0 ? 1.0 : 0.0,
    base + 2 < ELEMENTS && (base + 2) % 7 == 0 ? 1.0 : 0.0,
    base + 3 < ELEMENTS && (base + 3) % 7 == 0 ? 1.0 : 0.0);
}`;

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 */
export function uploadStructureInitializeWeights(execution, weights) {
  /** @type {Record<string, Tensor>} */
  const out = {};
  for (const name of STRUCTURE_INITIALIZE_WEIGHTS) {
    const values = weights[name];
    if (values === undefined) throw new Error(`structure initialize weights: ${name} is missing`);
    out[name] = execution.uploadWeight(values, `w.structure.init.${name}`);
  }
  return out;
}

export function releaseStructureInitializeWeights(execution, weights) {
  for (const name of STRUCTURE_INITIALIZE_WEIGHTS) execution.release(weights[name]);
}

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{msaFirstRow: Tensor, weights: Record<string, Tensor>, length: number,
 *   msaChannels: number, structureChannels: number, epsilon?: number}} call
 * @returns {{single: Tensor, initialRepresentation: Tensor, activations: Tensor,
 *   affine: Tensor}} the caller owns all four
 */
export function structureInitialize(execution, call) {
  const {
    msaFirstRow, weights: w, length, msaChannels, structureChannels, epsilon = 1e-5,
  } = call;
  if (msaFirstRow.elements !== length * msaChannels) {
    throw new RangeError(
      `structureInitialize: the MSA row holds ${msaFirstRow.elements} floats, not `
      + `${length} x ${msaChannels}`);
  }

  const single = linear(execution, {
    x: msaFirstRow, weight: w.singleProjectionWeight, bias: w.singleProjectionBias,
    rows: length, inChannels: msaChannels, outChannels: structureChannels,
    weightLayout: "in-major",
  });
  const initialRepresentation = layerNorm(execution, {
    x: single, weight: w.singleNormScale, bias: w.singleNormOffset,
    rows: length, channels: structureChannels, epsilon,
  });
  const activations = linear(execution, {
    x: initialRepresentation, weight: w.initialProjectionWeight,
    bias: w.initialProjectionBias,
    rows: length, inChannels: structureChannels, outChannels: structureChannels,
    weightLayout: "in-major",
  });
  const affine = execution.allocate(length * 7, "structure.affine");
  execution.dispatch({
    source: affineIdentitySource(length),
    output: affine,
    label: `structure.affine-identity:${length}`,
  });
  single.label = "structure.single";
  initialRepresentation.label = "structure.initial";
  activations.label = "structure.activations";
  return { single, initialRepresentation, activations, affine };
}

// Every builder above, memoised on its arguments (`memoSource` in webgl/runtime/execution.js).
export const affineIdentitySource = memoSource(affineIdentitySourceUncached);
