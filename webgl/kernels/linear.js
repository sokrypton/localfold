/**
 * A dense projection: `out[row][o] = sum_i x[row][i] * weight[o][i] + bias[o]`,
 * with the activation folded in.
 *
 * This is the kernel the rest of an Evoformer block is made of - the q/k/v/gate
 * projections, both halves of a transition, the triangle's four projections -
 * and it is where the FLOPs are.
 *
 * 🔴 WEIGHT IS `[out][in]`, ROW-MAJOR, AND THAT IS A CHOICE THE CALLER MUST
 * MATCH. A fragment owns four consecutive OUTPUT channels of one row, so it
 * walks four weight rows against one input row; with `[out][in]` those four
 * reads are four contiguous texels and the input's are shared. `[in][out]`
 * would make the weight reads strided by `out` and the input's unshared. The
 * bundles on disk are not necessarily in this order - wiring them is the next
 * stage's job, and a transposed weight is a plausible-looking wrong answer, so
 * `check-webgl-kernels.js` compares against a reference that indexes the same
 * way and nothing here guesses.
 *
 * 🔴 AND THE INTENSITY IS 3.2 FLOPS A FETCH, WHICH IS A QUARTER OF WHAT THIS
 * API REACHES. Four outputs a fragment is four weight texels plus one input
 * texel for sixteen multiply-adds. docs/A100.md's GEMM reaches 21.3 on the
 * fragment path by giving one invocation EIGHT outputs across eight draw
 * buffers, and 42.7 through transform feedback. This kernel is written for
 * correctness first, as AGENTS.md requires - "Establish numerical correctness
 * before optimizing performance" - and the geometry is the first thing to
 * revisit once a block folds.
 */
import { ONE_TARGET, PRELUDE } from "../runtime/execution.js";
/** @typedef {import("../runtime/storage.js").Tensor} Tensor */


const ACTIVATIONS = {
  none: "value",
  relu: "max(value, vec4(0.0))",
  sigmoid: "vec4(1.0) / (vec4(1.0) + exp(-value))",
};

export const linearSource = (rows, inChannels, outChannels, activation, useBias) => `${PRELUDE}${ONE_TARGET}
const int ROWS = ${rows};
const int IN = ${inChannels};
const int IN4 = ${inChannels / 4};
const int OUT = ${outChannels};
const int OUT4 = ${outChannels / 4};
uniform sampler2D xTex;
uniform sampler2D weightTex;
${useBias ? "uniform sampler2D biasTex;" : ""}

void main() {
  int texel = linearTexel();
  if (texel >= ROWS * OUT4) { result = vec4(0.0); return; }
  int row = texel / OUT4;
  int group = texel - row * OUT4;
  int xBase = row * IN4;
  // Four output channels: four weight rows, each IN long, each starting on a
  // texel boundary because IN is a multiple of four.
  int wBase = group * 4 * IN4;

  vec4 sum = vec4(0.0);
  for (int i = 0; i < IN4; ++i) {
    vec4 x = fetch4(xTex, xBase + i);
    sum.x += dot(x, fetch4(weightTex, wBase + i));
    sum.y += dot(x, fetch4(weightTex, wBase + IN4 + i));
    sum.z += dot(x, fetch4(weightTex, wBase + 2 * IN4 + i));
    sum.w += dot(x, fetch4(weightTex, wBase + 3 * IN4 + i));
  }
  vec4 value = sum${useBias ? " + fetch4(biasTex, group)" : ""};
  result = ${ACTIVATIONS[activation]};
}`;

/**
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{x: Tensor, weight: Tensor, bias?: Tensor, rows: number, inChannels: number,
 *   outChannels: number, activation?: "none" | "relu" | "sigmoid"}} call
 * @param {Tensor} [into]
 * @returns {Tensor}
 */
export function linear(execution, call, into) {
  const { x, weight, bias, rows, inChannels, outChannels, activation = "none" } = call;
  if (!(activation in ACTIVATIONS)) {
    throw new RangeError(`linear: no activation \`${activation}\``);
  }
  /** @type {[string, number][]} */
  const widths = [["inChannels", inChannels], ["outChannels", outChannels]];
  for (const [name, value] of widths) {
    if (value % 4 !== 0) throw new RangeError(`linear: ${name} must be a multiple of 4, got ${value}`);
  }
  if (x.elements !== rows * inChannels) {
    throw new RangeError(`linear: x holds ${x.elements} floats, not ${rows} x ${inChannels}`);
  }
  if (weight.elements !== outChannels * inChannels) {
    throw new RangeError(
      `linear: weight holds ${weight.elements} floats, not ${outChannels} x ${inChannels}`);
  }
  if (bias !== undefined && bias.elements !== outChannels) {
    throw new RangeError(`linear: bias holds ${bias.elements} floats, not ${outChannels}`);
  }
  const output = into ?? execution.allocate(rows * outChannels, "linear");
  if (output === x) {
    throw new Error("linear: a fragment shader cannot read the texture it draws into");
  }
  return execution.dispatch({
    source: linearSource(rows, inChannels, outChannels, activation, bias !== undefined),
    output,
    inputs: bias === undefined
      ? { xTex: x, weightTex: weight }
      : { xTex: x, weightTex: weight, biasTex: bias },
    label: `linear:${rows}x${inChannels}x${outChannels}:${activation}`,
  });
}
