/**
 * Row-wise LayerNorm, the operation an Evoformer block runs about ten times.
 *
 * Semantics are `normalizeRows` in `webgpu/kernels/triangle/cpu-reference.js`,
 * which is this repository's statement of what AlphaFold's LayerNorm is:
 *
 *     mean       = sum(x) / channels
 *     variance   = sum((x - mean)^2) / channels          (biased)
 *     out[c]     = (x[c] - mean) * inversesqrt(variance + epsilon) * w[c] + b[c]
 *
 * 🔴 TWO PASSES OVER THE ROW FOR AF2, AND `E[x^2] - E[x]^2` FOR AF3 - BECAUSE
 * THE TWO MODELS GENUINELY DIFFER HERE. The one-pass form is one loop instead
 * of two and it loses the top bits where the mean is large next to the spread:
 * a pair row whose values sit around 10 with a spread of 1e-3 has `E[x^2]` and
 * `E[x]^2` agreeing to six digits, and f32 keeps seven. AF2's reference centres
 * first, so `variance: "two-pass"` is the default and the two are bit-comparable
 * rather than merely close.
 *
 * AF3 does NOT. `cpu/af3/trunk/pairformer.js` states it: "`use_fast_variance` IS
 * THE DEFAULT AND IT IS NOT THE TWO-PASS FORMULA: AF3 takes E[x^2] - E[x]^2,
 * which is algebraically the same and numerically is not. The trunk's single
 * representation reaches 1.7e5, where the two expectations agree to about six
 * digits and their difference does not." So `variance: "fast"` is not a cheaper
 * approximation to be avoided - on that path it is the right answer, and the
 * centred form is the wrong one.
 *
 * 🔴 AND A RAGGED WIDTH NEEDS A DIFFERENT KERNEL, BECAUSE A TEXEL THEN
 * STRADDLES TWO ROWS. The aligned kernel below has a fragment own four
 * consecutive channels of ONE row and share that row's mean; where the width is
 * not a multiple of four the four can belong to two rows with two different
 * means, so each element resolves its own row and sweeps it.
 *
 * That is not a hypothetical shape: AF3's diffusion conditioning normalises a
 * concatenation of the trunk pair with the RAW 139-column relative encoding,
 * which is **267** channels. Refusing it outright - as this kernel first did -
 * would have meant that stage could not run at all. Four row sweeps a fragment
 * is the cost, on a path nothing else here takes; `kernels/linear.js` documents
 * the same split for AF2's 50 pLDDT bins.
 */
import { ONE_TARGET, PRELUDE } from "../runtime/execution.js";

/** @typedef {import("../runtime/storage.js").Tensor} Tensor */


/**
 * 🔴 AND THE SCALE AND THE OFFSET ARE BOTH OPTIONAL, BECAUSE AF3'S ADAPTIVE
 * LAYERNORM HAS NEITHER. `layerNormSlow(x, rows, channels, null, null)` is what
 * `cpu/af3/diffusion/atom-encoder.js` calls before the conditioning supplies a
 * scale and a shift of its own - and the conditioning's own norm takes a scale
 * with NO offset. Passing ones and zeros would compute the same thing and bind
 * two textures to say nothing; a missing sampler is refused by name, so the
 * source is generated without them instead.
 */
export const layerNormSource = (
  rows, channels, roundHalf = false, variance = "two-pass",
  hasWeight = true, hasBias = true,
) => (channels % 4 === 0
  ? alignedLayerNormSource(rows, channels, roundHalf, variance, hasWeight, hasBias)
  : raggedLayerNormSource(rows, channels, roundHalf, variance, hasWeight, hasBias));

/** The ragged form: every element resolves its own row and sweeps it. */
export const raggedLayerNormSource = (
  rows, channels, roundHalf, variance, hasWeight, hasBias,
) => `${PRELUDE}${ONE_TARGET}
const int CHANNELS = ${channels};
const int ELEMENTS = ${rows * channels};
uniform sampler2D xTex;
${hasWeight ? "uniform sampler2D weightTex;" : ""}
${hasBias ? "uniform sampler2D biasTex;" : ""}
uniform float epsilon;

float normalised(int element) {
  int row = element / CHANNELS;
  int channel = element - row * CHANNELS;
  int base = row * CHANNELS;
  float total = 0.0;
${variance === "fast" ? `  float totalSquares = 0.0;
  for (int c = 0; c < CHANNELS; ++c) {
    float v = fetch1(xTex, base + c);
    total += v;
    totalSquares += v * v;
  }
  float mean = total / float(CHANNELS);
  float inverseStd = inversesqrt(totalSquares / float(CHANNELS) - mean * mean + epsilon);`
    : `  for (int c = 0; c < CHANNELS; ++c) total += fetch1(xTex, base + c);
  float mean = total / float(CHANNELS);
  float spread = 0.0;
  for (int c = 0; c < CHANNELS; ++c) {
    float centred = fetch1(xTex, base + c) - mean;
    spread += centred * centred;
  }
  float inverseStd = inversesqrt(spread / float(CHANNELS) + epsilon);`}
  float value = (fetch1(xTex, element) - mean) * inverseStd;
${hasWeight ? "  value *= fetch1(weightTex, channel);" : ""}
${hasBias ? "  value += fetch1(biasTex, channel);" : ""}
  return value;
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  vec4 value = vec4(
    normalised(base),
    base + 1 < ELEMENTS ? normalised(base + 1) : 0.0,
    base + 2 < ELEMENTS ? normalised(base + 2) : 0.0,
    base + 3 < ELEMENTS ? normalised(base + 3) : 0.0);
  result = ${roundHalf ? "half4(value)" : "value"};
}`;

/**
 * 🔴 SPLITTING THE ROW STATISTICS INTO THEIR OWN PASS IS A MEASURED NULL, AND
 * THE ARITHMETIC THAT PREDICTED IT WAS COUNTING THE WRONG BYTES.
 *
 * Every fragment of the aligned kernel sweeps its whole row, and there are
 * `CH4` fragments a row - so the row is read 192 times over at 768 channels,
 * 384 two-pass. At 576 x 768 that is 42.5 M texel reads for an output of
 * 110,592, which read as **2.4 GB/s** against the 290 GB/s
 * `probe-webgl-draw-floor.js` clocks for a copy, and looked like a 40x
 * inefficiency with an obvious fix: statistics once a row, then applied.
 *
 * Built, measured on the fold: **21063-21848 ms against the fused kernel's
 * 21769-21815**, which is nothing, over two rounds. And the reason is that a
 * 768-float row is **3 KiB**: it is cache-resident for the whole column of
 * fragments that share it, so those 191 extra sweeps were already free and the
 * 42.5 M figure was LOGICAL reads, not traffic. The same mistake as reading the
 * contraction's "53 GB/s looks like DRAM" as a bandwidth bound - an amplified
 * read count is not a bandwidth measurement, and only an arm says which.
 *
 * 🔴 AND IT WAS NOT BIT-IDENTICAL EITHER, WHICH THE FIRST VERSION OF THIS NOTE
 * CLAIMED IT WOULD BE. The reduction runs over the same elements in the same
 * order, so the statistics ought to be the same floats - but this compiler
 * contracts the sums differently in a shader that then applies them and one
 * that only writes them, and a 20-step sampler amplifies that: `caca`
 * 3.8533516926432423 against 3.8531561311027773, with rmsd 1.595 and 201
 * contacts unmoved either way. A change that costs arithmetic and buys nothing
 * is a revert, so the split is gone and only this note is kept.
 *
 * What remains unexplained is the kernel's absolute cost: `layer-norm:576x768`
 * is about 0.45 ms a call once the profile's wait is discounted, against the
 * ~25 us a 110,592-texel draw costs, and neither its traffic nor its redundant
 * reduction accounts for the difference.
 */
export const alignedLayerNormSource = (
  rows, channels, roundHalf, variance, hasWeight, hasBias,
) => `${PRELUDE}${ONE_TARGET}
const int ROWS = ${rows};
const int CHANNELS = ${channels};
const int CH4 = ${channels / 4};
uniform sampler2D xTex;
${hasWeight ? "uniform sampler2D weightTex;" : ""}
${hasBias ? "uniform sampler2D biasTex;" : ""}
uniform float epsilon;

void main() {
  int texel = linearTexel();
  if (texel >= ROWS * CH4) { result = vec4(0.0); return; }
  int row = texel / CH4;
  int group = texel - row * CH4;
  int base = row * CH4;

${variance === "fast"
    ? `  // AF3's own: one pass, E[x^2] - E[x]^2. See the note at the top for why
  // this is the right answer on that path rather than a cheaper one.
  float total = 0.0;
  float totalSquares = 0.0;
  for (int i = 0; i < CH4; ++i) {
    vec4 v = fetch4(xTex, base + i);
    total += v.x + v.y + v.z + v.w;
    totalSquares += dot(v, v);
  }
  float mean = total / float(CHANNELS);
  float inverseStd = inversesqrt(totalSquares / float(CHANNELS) - mean * mean + epsilon);`
    : `  float total = 0.0;
  for (int i = 0; i < CH4; ++i) {
    vec4 v = fetch4(xTex, base + i);
    total += v.x + v.y + v.z + v.w;
  }
  float mean = total / float(CHANNELS);

  float spread = 0.0;
  for (int i = 0; i < CH4; ++i) {
    vec4 centred = fetch4(xTex, base + i) - mean;
    spread += dot(centred, centred);
  }
  float inverseStd = inversesqrt(spread / float(CHANNELS) + epsilon);`}

  vec4 value = fetch4(xTex, texel);
  vec4 normalised = (value - mean) * inverseStd${hasWeight ? " * fetch4(weightTex, group)" : ""}${hasBias ? "\n    + fetch4(biasTex, group)" : ""};
  // The shipped AF2 attention stores this tensor in f16 (normalizedStorage);
  // see half4 in webgl/runtime/execution.js. No backticks in a GLSL comment -
  // it lives inside a JS template literal and one would end it.
  result = ${roundHalf ? "half4(normalised)" : "normalised"};
}`;

/**
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{x: Tensor, weight?: Tensor, bias?: Tensor, rows: number, channels: number,
 *   epsilon?: number, roundHalf?: boolean,
 *   variance?: "two-pass" | "fast"}} call
 * @param {Tensor} [into]
 * @returns {Tensor}
 */
export function layerNorm(
  execution,
  {
    x, weight, bias, rows, channels, epsilon = 1e-5, roundHalf = false,
    variance = "two-pass",
  },
  into,
) {
  if (variance !== "two-pass" && variance !== "fast") {
    throw new RangeError(
      `layerNorm: variance is "two-pass" (AF2) or "fast" (AF3), not \`${variance}\``);
  }
  if (x.elements !== rows * channels) {
    throw new RangeError(
      `layerNorm: x holds ${x.elements} floats, not ${rows} x ${channels}`);
  }
  /** @type {[string, Tensor][]} */
  const parameters = /** @type {[string, Tensor][]} */ (
    [["weight", weight], ["bias", bias]].filter(([, t]) => t !== undefined));
  for (const [name, tensor] of parameters) {
    if (tensor.elements !== channels) {
      throw new RangeError(`layerNorm: ${name} holds ${tensor.elements}, not ${channels}`);
    }
  }
  const output = into ?? execution.allocate(rows * channels, "layerNorm");
  if (output === x) {
    throw new Error("layerNorm: a fragment shader cannot read the texture it draws into");
  }
  return execution.dispatch({
    source: layerNormSource(
      rows, channels, roundHalf, variance, weight !== undefined, bias !== undefined),
    output,
    inputs: {
      xTex: x,
      ...(weight === undefined ? {} : { weightTex: weight }),
      ...(bias === undefined ? {} : { biasTex: bias }),
    },
    floats: { epsilon },
    label: `layer-norm:${rows}x${channels}${roundHalf ? ":half" : ""}`
      + `${variance === "fast" ? ":fast" : ""}`
      + `${weight === undefined ? ":no-scale" : ""}${bias === undefined ? ":no-offset" : ""}`,
  });
}
