/**
 * Triangle multiplication, whole: the first complete AlphaFold operation in
 * this backend rather than a primitive.
 *
 * Nine draws for what the WGSL side fuses into rather fewer, in the order the
 * CPU reference states it (`webgpu/kernels/triangle/cpu-reference.js`, which is
 * this repository's statement of the operation):
 *
 *     zn        = LayerNorm(z)                       pairs x cZ
 *     a         = Linear(zn) * sigmoid(Linear(zn)) * mask
 *     b         = Linear(zn) * sigmoid(Linear(zn)) * mask
 *     contract  = sum_k a[i,k,h] * b[j,k,h]          pairs x cHidden
 *     out       = Linear(LayerNorm(contract)) * sigmoid(Linear(zn))
 *
 * 🔴 THE SIGMOIDS ARE FOLDED INTO THE PROJECTIONS AND THE MASK IS NOT. An
 * activation costs nothing inside the kernel that already has the value in a
 * register, so `ag`, `bg` and the output gate are `activation: "sigmoid"`
 * linears rather than separate passes. The mask cannot be: it is one float a
 * PAIR against cHidden floats of activation, so it is a broadcast
 * (`maskRows`), and it is a draw of its own until something measures that as a
 * cost worth fusing.
 *
 * 🔴 AND EVERY INTERMEDIATE IS RELEASED THE MOMENT ITS LAST READER IS DONE,
 * which is the whole reason `webgl/runtime/storage.js` counts. At cZ 128 and
 * cHidden 128 a 400-token block's `zn` alone is 78 MiB, so holding all nine at
 * once is most of a small GPU's memory for no reason. `zn` is the one that
 * lives longest - the output gate reads it after the contraction is finished -
 * and the ordering below is written around that.
 *
 * 🔴 NOTHING HERE IS WIRED TO A WEIGHT BUNDLE. `uploadTriangleWeights` takes
 * the plain arrays the CPU reference and `createDeterministicTriangleInput`
 * use, so a gate can drive this against the reference today; reading
 * `model-af2-monomer-int5` is a later stage, and a transposed weight is a
 * plausible-looking wrong answer, so that stage gets its own differential.
 * The layout this expects is the reference's: `weight[out * inChannels + in]`.
 */
import { layerNorm } from "./layer-norm.js";
import { linear } from "./linear.js";
import { triangleContract } from "./triangle-contract.js";
import { maskRows, multiply } from "../runtime/elementwise.js";
/** @typedef {import("../runtime/storage.js").Tensor} Tensor */


/** The weight names this operation reads, in the reference's own spelling. */
export const TRIANGLE_WEIGHTS = [
  "layerNormInWeight", "layerNormInBias",
  "linearAPWeight", "linearAPBias", "linearAGWeight", "linearAGBias",
  "linearBPWeight", "linearBPBias", "linearBGWeight", "linearBGBias",
  "layerNormOutWeight", "layerNormOutBias",
  "linearZWeight", "linearZBias", "linearGWeight", "linearGBias",
];

/**
 * Upload a host weight set once, to be kept across blocks.
 *
 * @param {import("../runtime/execution.js").Execution} execution
 */
export function uploadTriangleWeights(execution, weights) {
  /** @type {Record<string, Tensor>} */
  const out = {};
  for (const name of TRIANGLE_WEIGHTS) {
    const values = weights[name];
    if (values === undefined) throw new Error(`triangle weights: ${name} is missing`);
    out[name] = execution.upload(values, `w.triangle.${name}`);
  }
  return out;
}

export function releaseTriangleWeights(execution, weights) {
  for (const name of TRIANGLE_WEIGHTS) execution.release(weights[name]);
}

/**
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{z: Tensor, mask: Tensor, weights: Record<string, Tensor>,
 *   shape: {length: number, cZ: number, cHidden: number},
 *   direction?: "outgoing" | "incoming", epsilon?: number}} call
 * @returns {Tensor} the output, `pairs x cZ`; the caller owns it
 */
export function triangleMultiplication(execution, call) {
  const { z, mask, weights: w, shape, direction = "outgoing", epsilon = 1e-5 } = call;
  const { length, cZ, cHidden } = shape;
  const pairs = length * length;
  if (z.elements !== pairs * cZ) {
    throw new RangeError(`triangleMultiplication: z holds ${z.elements} floats, not ${pairs} x ${cZ}`);
  }
  if (mask.elements !== pairs) {
    throw new RangeError(`triangleMultiplication: mask holds ${mask.elements} floats, not ${pairs}`);
  }

  const zn = layerNorm(execution, {
    x: z, weight: w.layerNormInWeight, bias: w.layerNormInBias,
    rows: pairs, channels: cZ, epsilon,
  });

  /** One of the two contraction operands: project, gate, mask. */
  const side = (projectWeight, projectBias, gateWeight, gateBias, label) => {
    const projected = linear(execution, {
      x: zn, weight: projectWeight, bias: projectBias,
      rows: pairs, inChannels: cZ, outChannels: cHidden,
    });
    const gate = linear(execution, {
      x: zn, weight: gateWeight, bias: gateBias,
      rows: pairs, inChannels: cZ, outChannels: cHidden, activation: "sigmoid",
    });
    const gated = multiply(execution, projected, gate);
    execution.release(projected);
    execution.release(gate);
    const masked = maskRows(execution, { x: gated, mask, rows: pairs, channels: cHidden });
    execution.release(gated);
    masked.label = `triangle.${label}`;
    return masked;
  };

  const a = side(w.linearAPWeight, w.linearAPBias, w.linearAGWeight, w.linearAGBias, "a");
  const b = side(w.linearBPWeight, w.linearBPBias, w.linearBGWeight, w.linearBGBias, "b");

  const contracted = triangleContract(execution, { a, b, length, channels: cHidden, direction });
  execution.release(a);
  execution.release(b);

  const normalised = layerNorm(execution, {
    x: contracted, weight: w.layerNormOutWeight, bias: w.layerNormOutBias,
    rows: pairs, channels: cHidden, epsilon,
  });
  execution.release(contracted);

  const projected = linear(execution, {
    x: normalised, weight: w.linearZWeight, bias: w.linearZBias,
    rows: pairs, inChannels: cHidden, outChannels: cZ,
  });
  execution.release(normalised);

  // ...the output gate reads `zn`, not the contraction, which is why zn has
  // been held all the way down here.
  const gate = linear(execution, {
    x: zn, weight: w.linearGWeight, bias: w.linearGBias,
    rows: pairs, inChannels: cZ, outChannels: cZ, activation: "sigmoid",
  });
  execution.release(zn);

  const output = multiply(execution, projected, gate);
  execution.release(projected);
  execution.release(gate);
  output.label = `triangle.${direction}`;
  return output;
}
