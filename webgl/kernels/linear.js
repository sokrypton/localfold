/**
 * A dense projection: `out[row][o] = sum_i x[row][i] * weight[o][i] + bias[o]`,
 * with the activation folded in.
 *
 * This is the kernel the rest of an Evoformer block is made of - the q/k/v/gate
 * projections, both halves of a transition, the triangle's four projections -
 * and it is where the FLOPs are.
 *
 * 🔴 BOTH WEIGHT LAYOUTS ARE SUPPORTED BECAUSE THIS MODEL USES BOTH, AND A
 * TRANSPOSED WEIGHT IS A PLAUSIBLE-LOOKING WRONG ANSWER. The triangle's
 * reference projects with `weight[out * inChannels + in]` (`project` in
 * webgpu/kernels/triangle/cpu-reference.js) while AF2's attention projects with
 * `weight[in * outChannels + out]` - `let weight_row = weight_c * projected;`
 * then `weight_row + output_hd` in webgpu/kernels/attention.js. Same model,
 * opposite conventions, and nothing in either tensor's SHAPE distinguishes them
 * because both are `in * out` floats. So `weightLayout` is explicit and has no
 * safe default: `"out-major"` is `[out][in]` and `"in-major"` is `[in][out]`.
 *
 * Neither costs more than the other here. A fragment owns four consecutive
 * OUTPUT channels of one row: with `[out][in]` it reads four contiguous texels
 * of four different weight rows against one shared input texel, and with
 * `[in][out]` it reads one texel per input channel, each already holding the
 * four outputs it wants. Five fetches for sixteen multiply-adds either way.
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

/** `half4` is in the prelude; see `webgl/runtime/execution.js` for why. */
const ROUND_HALF = (expression) => `half4(${expression})`;

export const linearSource = (
  rows, inChannels, outChannels, activation, useBias, weightLayout, roundHalf = false,
) => {
  // `[out][in]`: four weight rows, each IN long and each starting on a texel
  // boundary because IN is a multiple of four, against one shared input texel.
  const outMajor = `  int wBase = group * 4 * IN4;
  for (int i = 0; i < IN4; ++i) {
    vec4 x = fetch4(xTex, xBase + i);
    sum.x += dot(x, fetch4(weightTex, wBase + i));
    sum.y += dot(x, fetch4(weightTex, wBase + IN4 + i));
    sum.z += dot(x, fetch4(weightTex, wBase + 2 * IN4 + i));
    sum.w += dot(x, fetch4(weightTex, wBase + 3 * IN4 + i));
  }`;
  // `[in][out]`: one texel per input channel, already holding the four outputs
  // this fragment owns, scaled by that channel's input value.
  const inMajor = `  for (int i = 0; i < IN4; ++i) {
    vec4 x = fetch4(xTex, xBase + i);
    int wRow = (i * 4) * OUT4 + group;
    sum += x.x * fetch4(weightTex, wRow);
    sum += x.y * fetch4(weightTex, wRow + OUT4);
    sum += x.z * fetch4(weightTex, wRow + 2 * OUT4);
    sum += x.w * fetch4(weightTex, wRow + 3 * OUT4);
  }`;
  // 🔴 AND A WIDTH THAT IS NOT A MULTIPLE OF FOUR NEEDS A DIFFERENT KERNEL,
  // BECAUSE A TEXEL THEN STRADDLES TWO ROWS. AF2's pLDDT head projects onto **50**
  // bins: row r occupies elements [50r, 50r + 50), so a fragment's four
  // consecutive elements can belong to two different rows with two different
  // input vectors. The aligned kernel above shares one input sweep across its
  // four outputs and cannot; this one resolves each component separately, four
  // dot products for four outputs. Four times the reads, on a projection whose
  // output is `rows * 50` floats - and the alternative is refusing a shape the
  // model actually has.
  const perComponent = (component) => `
  {
    int element = base + ${component};
    if (element < ROWS * OUT) {
      int row = element / OUT;
      int o = element - row * OUT;
      int xBase = row * IN4;
      float sum = ${useBias ? "fetch1(biasTex, o)" : "0.0"};
      for (int i = 0; i < IN4; ++i) {
        vec4 x = fetch4(xTex, xBase + i);
        int w = ${weightLayout === "in-major" ? "(i * 4) * OUT + o" : "o * IN + i * 4"};
        ${weightLayout === "in-major"
    ? `sum += x.x * fetch1(weightTex, w);
        sum += x.y * fetch1(weightTex, w + OUT);
        sum += x.z * fetch1(weightTex, w + 2 * OUT);
        sum += x.w * fetch1(weightTex, w + 3 * OUT);`
    : "sum += dot(x, fetch4(weightTex, (o * IN4) + i));"}
      }
      values[${component}] = sum;
    }
  }`;
  const ragged = outChannels % 4 !== 0;
  // 🔴 AND A RAGGED **INPUT** WIDTH NEEDS ITS OWN SWEEP, WHICH AF3 FORCED.
  // The aligned kernel reads whole texels of the input row, which only works
  // when the row starts on a texel boundary - and AF3's diffusion conditioning
  // projects a 267-channel concatenation (the trunk pair plus the RAW 139-column
  // relative encoding) down to the pair width. Refusing it, as this kernel first
  // did, would have meant that stage could not run. Element-by-element reads:
  // four times the input fetches on a path nothing else here takes.
  const raggedInput = inChannels % 4 !== 0;
  if (raggedInput && ragged) {
    throw new RangeError(
      `linear: ${inChannels} in and ${outChannels} out are both ragged, which no model`
      + " in this tree has - the two sweeps would need a third kernel");
  }
  const inMajorRagged = `  for (int i = 0; i < IN; ++i) {
    sum += fetch1(xTex, xBase + i) * fetch4(weightTex, i * OUT4 + group);
  }`;
  const outMajorRagged = `  for (int i = 0; i < IN; ++i) {
    float x = fetch1(xTex, xBase + i);
    sum.x += x * fetch1(weightTex, (group * 4) * IN + i);
    sum.y += x * fetch1(weightTex, (group * 4 + 1) * IN + i);
    sum.z += x * fetch1(weightTex, (group * 4 + 2) * IN + i);
    sum.w += x * fetch1(weightTex, (group * 4 + 3) * IN + i);
  }`;

  return `${PRELUDE}${ONE_TARGET}
const int ROWS = ${rows};
const int IN = ${inChannels};
const int IN4 = ${Math.ceil(inChannels / 4)};
const int OUT = ${outChannels};
const int OUT4 = ${Math.ceil(outChannels / 4)};
uniform sampler2D xTex;
uniform sampler2D weightTex;
${useBias ? "uniform sampler2D biasTex;" : ""}

void main() {
${ragged ? `  int base = linearTexel() * 4;
  if (base >= ROWS * OUT) { result = vec4(0.0); return; }
  vec4 values = vec4(0.0);
${[0, 1, 2, 3].map(perComponent).join("\n")}
  vec4 value = values;` : `  int texel = linearTexel();
  if (texel >= ROWS * OUT4) { result = vec4(0.0); return; }
  int row = texel / OUT4;
  int group = texel - row * OUT4;
  int xBase = row * ${raggedInput ? "IN" : "IN4"};

  vec4 sum = vec4(0.0);
${raggedInput
    ? (weightLayout === "in-major" ? inMajorRagged : outMajorRagged)
    : (weightLayout === "in-major" ? inMajor : outMajor)}
  vec4 value = sum${useBias ? " + fetch4(biasTex, group)" : ""};`}
  result = ${roundHalf ? ROUND_HALF(ACTIVATIONS[activation]) : ACTIVATIONS[activation]};
}`;
};

/**
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{x: Tensor, weight: Tensor, bias?: Tensor, rows: number, inChannels: number,
 *   outChannels: number, activation?: "none" | "relu" | "sigmoid",
 *   weightLayout?: "out-major" | "in-major", roundHalf?: boolean}} call
 * @param {Tensor} [into]
 * @returns {Tensor}
 */
export function linear(execution, call, into) {
  const {
    x, weight, bias, rows, inChannels, outChannels,
    activation = "none", weightLayout = "out-major", roundHalf = false,
  } = call;
  if (!(activation in ACTIVATIONS)) {
    throw new RangeError(`linear: no activation \`${activation}\``);
  }
  if (weightLayout !== "out-major" && weightLayout !== "in-major") {
    throw new RangeError(
      `linear: weightLayout is "out-major" ([out][in]) or "in-major" ([in][out]), `
      + `not \`${weightLayout}\``);
  }
  // Neither width has to be a multiple of four: AF2's 50 pLDDT bins made the
  // OUTPUT ragged and AF3's 267-channel diffusion conditioning the INPUT. Both
  // at once is refused inside `linearSource`, because no model here needs it.
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
    source: linearSource(
      rows, inChannels, outChannels, activation, bias !== undefined, weightLayout, roundHalf),
    output,
    inputs: bias === undefined
      ? { xTex: x, weightTex: weight }
      : { xTex: x, weightTex: weight, biasTex: bias },
    label: `linear:${rows}x${inChannels}x${outChannels}:${activation}:${weightLayout}`
      + `${roundHalf ? ":half" : ""}`,
  });
}
