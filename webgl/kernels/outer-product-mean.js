/**
 * The outer product mean: the only path by which the MSA reaches the pair
 * track, and the one kernel here whose arithmetic does not fit one draw.
 *
 *     m        = LayerNorm(activations)                        [s][i][cM]
 *     left     = mask[s][i] * (m @ leftWeight  + leftBias)     [s][i][cOuter]
 *     right    = mask[s][i] * (m @ rightWeight + rightBias)    [s][i][cOuter]
 *     acc[i][j][z] = sum_s sum_cl sum_cr left[s][i][cl] * right[s][j][cr]
 *                                        * outputWeight[(cl * cOuter + cr) * cZ + z]
 *     count[i][j]  = sum_s mask[s][i] * mask[s][j]
 *     out[i][j][z] = (acc + outputBias[z]) / (normalizationEpsilon + count[i][j])
 *
 * 🔴 FOUR CONVENTIONS, EVERY ONE READ OFF THE WGSL SHADER RATHER THAN ASSUMED,
 * because each would produce a plausible tensor if guessed wrong:
 *
 *   - The MASK is applied to the PROJECTIONS, not to the product
 *     (`left[index] = keep_r * acc_left...`), so the numerator carries
 *     `mask[s][i] * mask[s][j]` exactly as the denominator does.
 *   - `outputWeight` is `[c_left][c_right][cZ]`:
 *     `((outer_left * p.c_outer + outer_right) * p.c_z) + z`. Transposing the
 *     two channel axes is invisible in the tensor's shape, which is
 *     `cOuter * cOuter * cZ` either way, and `cOuter` is square.
 *   - `outputBias` is added BEFORE the division, so the bias is divided too:
 *     `(output[index] + weights[p.output_bias + z]) / (p.normalization_epsilon
 *     + count)`. Adding it after is a different function of the mask.
 *
 *     🔴 AND boltz2 ADDS IT AFTER, AND CLAMPS RATHER THAN NUDGES.
 *     `cpu/af3/trunk/msa.js` prices it: "AF3 is `(product @ W + b) / (1e-3 +
 *     count)`; boltz2 is `product @ W / max(count, 1) + b`. The difference is
 *     worth exactly `(1 - 1/n) * b` - a per-channel CONSTANT on every pair - so
 *     it is invisible at MSA depth 1, where it is zero, and worth 15/16 of the
 *     bias at depth 16." That is why `biasAfterNorm` is an option and why a
 *     depth-1 gate could never see it.
 *   - The normalisation epsilon is **1e-3**, not the LayerNorm's 1e-5, and they
 *     are separate inputs on the WGSL side for that reason.
 *
 * 🔴 AND IT IS TWO DRAWS BECAUSE ONE WOULD BE `cOuter^2` PER OUTPUT ELEMENT.
 * Contracted directly, a fragment owning four `cZ` channels of one pair loops
 * `sequences * cOuter * cOuter` - a thousand multiply-adds a sequence at
 * cOuter 32, for every pair. The WGSL side splits it and so does this:
 *
 *     intermediate[s][i][cr][z] = sum_cl left[s][i][cl] * outputWeight[...]
 *     out[i][j][z]              = sum_s sum_cr intermediate[s][i][cr][z] * right[s][j][cr]
 *
 * which turns `cOuter^2` per pair into `cOuter` per pair plus one pass over
 * `sequences * length * cOuter * cZ`. The intermediate is the memory cost -
 * 2.1 M floats at 8 sequences, 64 residues, cOuter 32, cZ 128 - and it is
 * released as soon as the second draw is done.
 */
import { PRELUDE, ONE_TARGET, sized } from "../runtime/execution.js";
import { layerNorm } from "./layer-norm.js";
import { linear } from "./linear.js";
import { maskRows } from "../runtime/elementwise.js";

/** @typedef {import("../runtime/storage.js").Tensor} Tensor */

export const OUTER_PRODUCT_MEAN_WEIGHTS = [
  "layerNormScale", "layerNormOffset", "leftWeight", "leftBias",
  "rightWeight", "rightBias", "outputWeight", "outputBias",
];

/**
 * `intermediate[s][i][cr][z] = sum_cl left[s][i][cl] * outputWeight[(cl*cOuter+cr)*cZ+z]`
 */
export const opmIntermediateSource = (rows, cOuter, cZ) => sized({ ROWS: rows }, `${PRELUDE}${ONE_TARGET}
const int C_OUTER = ${cOuter};
const int CZ4 = ${cZ / 4};
uniform sampler2D leftTex;
uniform sampler2D weightTex;

void main() {
  int texel = linearTexel();
  if (texel >= ROWS * C_OUTER * CZ4) { result = vec4(0.0); return; }
  int z4 = texel % CZ4;
  int rest = texel / CZ4;
  int cr = rest % C_OUTER;
  int si = rest / C_OUTER;

  vec4 sum = vec4(0.0);
  for (int cl = 0; cl < C_OUTER; ++cl) {
    // One scalar of left against one texel of the weight: the weight's four cZ
    // channels for this (cl, cr) are contiguous because cZ is the fastest axis
    // of [c_left][c_right][cZ]. (No backticks in here - this comment lives
    // inside a JS template literal, and one would end it. See
    // test/modules-parse.test.js, which exists because that shipped once.)
    sum += fetch1(leftTex, si * C_OUTER + cl)
      * fetch4(weightTex, (cl * C_OUTER + cr) * CZ4 + z4);
  }
  result = sum;
}`);

/**
 * `out[i][j][z] = (sum_s sum_cr intermediate[s][i][cr][z] * right[s][j][cr]
 *                  + outputBias[z]) / (epsilon + sum_s mask[s][i] * mask[s][j])`
 */
export const opmContractSource = (
  sequences, length, cOuter, cZ, biasAfterNorm = false,
) => sized({ L: length }, `${PRELUDE}${ONE_TARGET}
const int SEQUENCES = ${sequences};
const int C_OUTER = ${cOuter};
const int CZ4 = ${cZ / 4};
uniform sampler2D intermediateTex;
uniform sampler2D rightTex;
uniform sampler2D maskTex;
uniform sampler2D biasTex;
uniform float normalizationEpsilon;

void main() {
  int texel = linearTexel();
  if (texel >= L * L * CZ4) { result = vec4(0.0); return; }
  int z4 = texel % CZ4;
  int pair = texel / CZ4;
  int j = pair % L;
  int i = pair / L;

  vec4 sum = vec4(0.0);
  float count = 0.0;
  for (int s = 0; s < SEQUENCES; ++s) {
    int row = s * L;
    count += fetch1(maskTex, row + i) * fetch1(maskTex, row + j);
    int leftBase = (row + i) * C_OUTER;
    int rightBase = (row + j) * C_OUTER;
    for (int cr = 0; cr < C_OUTER; ++cr) {
      sum += fetch1(rightTex, rightBase + cr)
        * fetch4(intermediateTex, (leftBase + cr) * CZ4 + z4);
    }
  }
  // ...the bias is inside the division, which is the WGSL side's arrangement
  // and not an obvious one.
${biasAfterNorm
    ? "  result = sum / max(count, 1.0) + fetch4(biasTex, z4);"
    : "  result = (sum + fetch4(biasTex, z4)) / (normalizationEpsilon + count);"}
}`);

/**
 * @param {import("../runtime/execution.js").Execution} execution
 */
export function uploadOuterProductMeanWeights(execution, weights) {
  /** @type {Record<string, Tensor>} */
  const out = {};
  for (const name of OUTER_PRODUCT_MEAN_WEIGHTS) {
    const values = weights[name];
    if (values === undefined) throw new Error(`outer product mean weights: ${name} is missing`);
    out[name] = execution.uploadWeight(values, `w.opm.${name}`);
  }
  return out;
}

export function releaseOuterProductMeanWeights(execution, weights) {
  for (const name of OUTER_PRODUCT_MEAN_WEIGHTS) execution.release(weights[name]);
}

/**
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{activations: Tensor, mask: Tensor, weights: Record<string, Tensor>,
 *   sequences: number, length: number, cM: number, cOuter: number, cZ: number,
 *   layerNormEpsilon?: number, normalizationEpsilon?: number,
 *   variance?: "two-pass" | "fast", biasAfterNorm?: boolean}} call
 * @returns {Tensor} `length^2 x cZ`; the caller owns it
 */
export function outerProductMean(execution, call) {
  const {
    activations, mask, weights: w, sequences, length, cM, cOuter, cZ,
    layerNormEpsilon = 1e-5, normalizationEpsilon = 1e-3,
    // 🔴 AF2 CENTRES AND AF3 DOES NOT - see kernels/layer-norm.js. The default
    // is AF2's, because this kernel was written for AF2 and a silent change of
    // formula is the kind of thing that moves a fold by 1e-4 and nothing else.
    variance = "two-pass",
    biasAfterNorm = false,
  } = call;
  const rows = sequences * length;
  if (activations.elements !== rows * cM) {
    throw new RangeError(
      `outerProductMean: activations hold ${activations.elements} floats, not `
      + `${sequences} x ${length} x ${cM}`);
  }
  if (mask.elements !== rows) {
    throw new RangeError(`outerProductMean: mask holds ${mask.elements} floats, not ${rows}`);
  }
  /** @type {[string, number][]} */
  const widths = [["cOuter", cOuter], ["cZ", cZ]];
  for (const [name, value] of widths) {
    if (value % 4 !== 0) {
      throw new RangeError(`outerProductMean: ${name} must be a multiple of 4, got ${value}`);
    }
  }

  const normalised = layerNorm(execution, {
    x: activations, weight: w.layerNormScale, bias: w.layerNormOffset,
    rows, channels: cM, epsilon: layerNormEpsilon, variance,
  });

  /** Project, then mask - the order the WGSL shader applies them in. */
  const side = (weight, bias, label) => {
    const projected = linear(execution, {
      x: normalised, weight, bias, rows, inChannels: cM, outChannels: cOuter,
      weightLayout: "in-major",
    });
    const masked = maskRows(execution, { x: projected, mask, rows, channels: cOuter });
    execution.release(projected);
    masked.label = `opm.${label}`;
    return masked;
  };
  const left = side(w.leftWeight, w.leftBias, "left");
  const right = side(w.rightWeight, w.rightBias, "right");
  execution.release(normalised);

  const intermediate = execution.allocate(rows * cOuter * cZ, "opm.intermediate");
  execution.dispatch({
    source: opmIntermediateSource(rows, cOuter, cZ),
    output: intermediate,
    inputs: { leftTex: left, weightTex: w.outputWeight },
    label: `opm.intermediate:${rows}x${cOuter}x${cZ}`,
  });
  execution.release(left);

  const output = execution.allocate(length * length * cZ, "opm.output");
  execution.dispatch({
    source: opmContractSource(sequences, length, cOuter, cZ, biasAfterNorm),
    output,
    inputs: {
      intermediateTex: intermediate, rightTex: right, maskTex: mask, biasTex: w.outputBias,
    },
    // 🔴 NOT PASSED IN boltz2's FORM, BECAUSE THE SHADER NO LONGER READS IT.
    // That variant clamps the count rather than nudging it, so the compiler
    // strips the uniform and `dispatch` refuses it by name - which is the check
    // working: "no float uniform `normalizationEpsilon`".
    ...(biasAfterNorm ? {} : { floats: { normalizationEpsilon } }),
    label: `opm.contract:${sequences}x${length}x${cOuter}x${cZ}`,
  });
  execution.release(intermediate);
  execution.release(right);
  return output;
}
