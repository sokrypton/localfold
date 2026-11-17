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
import { linear, planeFetchGlsl, planeInputs } from "./linear.js";
import { alignedPlaneStride, contractInto, paddedPairLength, triangleContract } from "./triangle-contract.js";
import { add, maskRows, multiply } from "../runtime/elementwise.js";
import { PRELUDE, sized, memoSource } from "../runtime/execution.js";
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
    out[name] = execution.uploadWeight(values, `w.triangle.${name}`);
  }
  return out;
}

export function releaseTriangleWeights(execution, weights) {
  for (const name of [...TRIANGLE_WEIGHTS, ...FUSED_WEIGHTS]) {
    if (weights[name] !== undefined) execution.release(weights[name]);
  }
}

/** The four input projections as one, uploaded in their place by `uploadTriangleGroup`. */
const FUSED_WEIGHTS = ["projectionGate", "projectionGateBias"];
const FUSED_FROM = [
  ["linearAPWeight", "linearAPBias"], ["linearAGWeight", "linearAGBias"],
  ["linearBPWeight", "linearBPBias"], ["linearBGWeight", "linearBGBias"],
];

/**
 * AF2's four `[H][cZ]` input projections - a's projection and gate, then b's - as ONE `[4H][cZ]` out-major weight and
 * one `[4H]` bias, which `triangleMultiplication` runs as a single projection whose row planes the operand draw reads
 * directly (`linear`'s `keepPlanes`). Out-major, so the fusion is a concatenation. Null where the shapes disagree.
 *
 * 🔴 UPLOADED IN float32 WHATEVER `halfWeights` SAYS: the four parts are 16,384 floats each at cZ 128, under the
 * float16 floor, and their concatenation is exactly AT it - fusing would silently halve their precision.
 */
export function fuseTriangleProjections(weights) {
  const parts = FUSED_FROM.map(([w, b]) => [weights[w], weights[b]]);
  if (parts.some(([w, b]) => w === undefined || b === undefined)) return null;
  const size = parts[0][0].length;
  const hidden = parts[0][1].length;
  if (parts.some(([w, b]) => w.length !== size || b.length !== hidden)) return null;
  const weight = new Float32Array(4 * size);
  const bias = new Float32Array(4 * hidden);
  parts.forEach(([w, b], index) => { weight.set(w, index * size); bias.set(b, index * hidden); });
  return { weight, bias };
}

/**
 * Upload one triangle multiplication's weights the way `triangleMultiplication` reads them fastest: the four input
 * projections fused (`fuseTriangleProjections`) unless `execution.noAf2TriangleBlock` asks for the nine-draw form.
 *
 * @param {import("../runtime/execution.js").Execution} execution
 */
export function uploadTriangleGroup(execution, source, label) {
  const fused = execution.noAf2TriangleBlock ? null : fuseTriangleProjections(source);
  /** @type {Record<string, Tensor>} */
  const out = {};
  for (const name of TRIANGLE_WEIGHTS) {
    const values = source[name];
    if (values === undefined) throw new Error(`${label}: ${name} is missing`);
    if (fused !== null && FUSED_FROM.some((pair) => pair.includes(name))) continue;
    out[name] = execution.uploadWeight(values, `w.${label}.${name}`);
  }
  if (fused !== null) {
    out.projectionGate = execution.upload(fused.weight, `w.${label}.projection-gate`);
    out.projectionGateBias = execution.upload(fused.bias, `w.${label}.projection-gate-bias`);
  }
  return out;
}

/**
 * The two gated, masked operands, written CHANNEL-MAJOR into the padded (and, incoming, transposed) planes the
 * pair-blocked contraction reads (`contractInto`, ./triangle-contract.js) - AF3's `operandsLaidOutSource`, with
 * AF2's split: the fused projection's row is `[aP | aG | bP | bG]`, BLOCKED, where AF3 interleaves its halves.
 *
 *     a[c][i][j] = aP[pair][c] * sigmoid(aG[pair][c]) * mask[pair]
 *
 * in the order the nine-draw form computes it - the product, then the mask - so the operands agree with it to the
 * rounding of `exp`.
 */
const operandsSourceUncached = (length, channels, paddedLength, transposed, blockRows,
  planeStride = paddedLength * paddedLength) => sized(
  { L: length, LP: paddedLength, PLANE: planeStride, ELEMENTS: channels * planeStride }, `${PRELUDE}
layout(location = 0) out vec4 aOut;
layout(location = 1) out vec4 bOut;
const int H = ${channels};
const int ROWW = ${4 * channels};
${blockRows > 0 ? `${planeFetchGlsl(blockRows, channels)}
float projectionAt(int element) { int row = element / ROWW; int col = element - row * ROWW; return planeFetch(row, col / 4)[col % 4]; }`
    : `uniform sampler2D projectionTex;
float projectionAt(int element) { return fetch1(projectionTex, element); }`}
uniform sampler2D maskTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { aOut = vec4(0.0); bOut = vec4(0.0); return; }
  // LP and PLANE are multiples of four, so the four lanes share one plane and one row and only j advances.
  int element = texel * 4;
  int c = element / PLANE;
  int within = element - c * PLANE;
  int i = within / LP;
  int j0 = within - i * LP;
  vec4 a = vec4(0.0);
  vec4 b = vec4(0.0);
  // ...and past LP rows a plane's tail, when its stride is wider than LP * LP, which nothing reads
  if (i < L) {
    for (int lane = 0; lane < 4; ++lane) {
      int j = j0 + lane;
      if (j >= L) break;
      int index = ${transposed ? "j * L + i" : "i * L + j"};
      int wide = index * ROWW + c;
      float m = fetch1(maskTex, index);
      float left = projectionAt(wide) * (1.0 / (1.0 + exp(-projectionAt(wide + H)))) * m;
      float right = projectionAt(wide + 2 * H) * (1.0 / (1.0 + exp(-projectionAt(wide + 3 * H)))) * m;
      if (lane == 0) { a.x = left; b.x = right; }
      else if (lane == 1) { a.y = left; b.y = right; }
      else if (lane == 2) { a.z = left; b.z = right; }
      else { a.w = left; b.w = right; }
    }
  }
  aOut = a;
  bOut = b;
}`);
export const operandsSource = memoSource(operandsSourceUncached);

/**
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{z: Tensor, mask: Tensor, weights: Record<string, Tensor>,
 *   shape: {length: number, cZ: number, cHidden: number},
 *   direction?: "outgoing" | "incoming", epsilon?: number,
 *   projectionPrecision?: "f32" | "f16", residualOf?: Tensor}} call
 * @returns {Tensor} the output, `pairs x cZ` - `residualOf + update` where that is given; the caller owns it
 */
export function triangleMultiplication(execution, call) {
  const {
    z, mask, weights: w, shape, direction = "outgoing", epsilon = 1e-5,
    projectionPrecision = "f32",
  } = call;
  const halfProjections = projectionPrecision === "f16";
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

  if (w.projectionGate !== undefined) {
    if (halfProjections) throw new Error("triangleMultiplication: the fused projections are float32 only");
    return blockedTriangle(execution, call, zn);
  }

  /** One of the two contraction operands: project, gate, mask. */
  const side = (projectWeight, projectBias, gateWeight, gateBias, label) => {
    const projected = linear(execution, {
      x: zn, weight: projectWeight, bias: projectBias,
      rows: pairs, inChannels: cZ, outChannels: cHidden, roundHalf: halfProjections,
    });
    const gate = linear(execution, {
      x: zn, weight: gateWeight, bias: gateBias,
      rows: pairs, inChannels: cZ, outChannels: cHidden, activation: "sigmoid",
      roundHalf: halfProjections,
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
  if (call.residualOf !== undefined) {
    const summed = add(execution, call.residualOf, output);
    execution.release(output);
    return summed;
  }
  return output;
}

/**
 * The same operation in five draws plus AF3's contraction: one fused input projection, one operand draw writing both
 * gated operands straight into the contraction's padded channel-major planes, the pair-blocked contraction
 * (`contractInto`), AF2's own LayerNorm, and the output projection with its gate - and the caller's residual - in its
 * epilogue.
 */
function blockedTriangle(execution, call, zn) {
  const { mask, weights: w, shape, direction = "outgoing", epsilon = 1e-5, residualOf } = call;
  const { length, cZ, cHidden } = shape;
  const pairs = length * length;
  const projection = linear(execution, {
    x: zn, weight: w.projectionGate, bias: w.projectionGateBias,
    rows: pairs, inChannels: cZ, outChannels: 4 * cHidden, keepPlanes: true,
  });
  const kept = /** @type {any} */ (projection).planes !== undefined ? /** @type {any} */ (projection) : null;
  const padded = paddedPairLength(execution, length, cHidden);
  const transposed = direction === "incoming";
  const stride = execution.noAlignedPlanes ? padded * padded : alignedPlaneStride(padded);
  const a = execution.allocate(cHidden * stride, "triangle.a");
  const b = execution.allocate(cHidden * stride, "triangle.b");
  execution.dispatch({
    source: operandsSource(length, cHidden, padded, transposed, kept?.blockRows ?? 0, stride),
    outputs: [a, b],
    inputs: kept === null ? { projectionTex: projection, maskTex: mask } : { ...planeInputs(kept), maskTex: mask },
    label: `triangle.operands${transposed ? ":transposed" : ""}:${length}x${cHidden}`,
  });
  if (kept !== null) for (const plane of kept.planes) execution.release(plane);
  else execution.release(projection);
  const contracted = execution.allocate(pairs * cHidden, "triangle.contract");
  // Incoming is outgoing over the transposed planes, handed bT and aT: `sum_k b[k][i] a[k][j]`.
  contractInto(execution, contracted, transposed ? b : a, transposed ? a : b, length, cHidden, false,
    transposed ? "incoming-transposed" : "outgoing", padded, "triangle", stride);
  execution.release(a);
  execution.release(b);
  const normalised = layerNorm(execution, {
    x: contracted, weight: w.layerNormOutWeight, bias: w.layerNormOutBias,
    rows: pairs, channels: cHidden, epsilon,
  });
  execution.release(contracted);
  const gate = linear(execution, {
    x: zn, weight: w.linearGWeight, bias: w.linearGBias,
    rows: pairs, inChannels: cZ, outChannels: cZ, activation: "sigmoid",
  });
  execution.release(zn);
  let output;
  if (residualOf !== undefined) {
    // ...the gate read per row in the repack (`gateSlice` at the output's own stride), where a multiplier costs a
    // draw after it; `--no-linear-gate-slice` is the arm
    output = linear(execution, {
      x: normalised, weight: w.linearZWeight, bias: w.linearZBias,
      rows: pairs, inChannels: cHidden, outChannels: cZ, residual: residualOf,
      ...(execution.noLinearGateSlice ? { multiplier: gate }
        : { gateSlice: { gates: gate, stride4: cZ / 4, at4: 0 } }),
    });
  } else {
    const projected = linear(execution, {
      x: normalised, weight: w.linearZWeight, bias: w.linearZBias,
      rows: pairs, inChannels: cHidden, outChannels: cZ,
    });
    output = multiply(execution, projected, gate);
    execution.release(projected);
  }
  execution.release(normalised);
  execution.release(gate);
  output.label = `triangle.${direction}`;
  return output;
}
