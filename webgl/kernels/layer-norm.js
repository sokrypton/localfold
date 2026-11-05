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
 * 🔴 TWO PASSES OVER THE ROW, NOT `E[x^2] - E[x]^2`. The one-pass form is one
 * loop instead of two and it loses the top bits where the mean is large next to
 * the spread: a pair row whose values sit around 10 with a spread of 1e-3 has
 * `E[x^2]` and `E[x]^2` agreeing to six digits, and f32 keeps seven. The
 * reference centres first and so does this, which also makes the two
 * bit-comparable rather than merely close.
 *
 * 🔴 AND A TEXEL MAY NOT STRADDLE TWO ROWS, which is why `channels % 4` is
 * refused up front. A fragment owns four consecutive channels; if a row's
 * length were not a multiple of four, the four would belong to two different
 * rows with two different means, and every shape this port actually runs
 * (cZ 128, cM 256, cHidden 32, 384) is a multiple of four. Refusing is better
 * than a kernel that is quietly wrong for cZ 7 - which is exactly the shape
 * that broke the WGSL triangle kernel, found by a sweep over widths.
 */
import { ONE_TARGET, PRELUDE } from "../runtime/execution.js";
/** @typedef {import("../runtime/storage.js").Tensor} Tensor */


export const layerNormSource = (rows, channels) => `${PRELUDE}${ONE_TARGET}
const int ROWS = ${rows};
const int CHANNELS = ${channels};
const int CH4 = ${channels / 4};
uniform sampler2D xTex;
uniform sampler2D weightTex;
uniform sampler2D biasTex;
uniform float epsilon;

void main() {
  int texel = linearTexel();
  if (texel >= ROWS * CH4) { result = vec4(0.0); return; }
  int row = texel / CH4;
  int group = texel - row * CH4;
  int base = row * CH4;

  float total = 0.0;
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
  float inverseStd = inversesqrt(spread / float(CHANNELS) + epsilon);

  vec4 value = fetch4(xTex, texel);
  result = (value - mean) * inverseStd * fetch4(weightTex, group) + fetch4(biasTex, group);
}`;

/**
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{x: Tensor, weight: Tensor, bias: Tensor, rows: number, channels: number,
 *   epsilon?: number}} call
 * @param {Tensor} [into]
 * @returns {Tensor}
 */
export function layerNorm(execution, { x, weight, bias, rows, channels, epsilon = 1e-5 }, into) {
  if (channels % 4 !== 0) {
    throw new RangeError(`layerNorm: channels must be a multiple of 4, got ${channels}`);
  }
  if (x.elements !== rows * channels) {
    throw new RangeError(
      `layerNorm: x holds ${x.elements} floats, not ${rows} x ${channels}`);
  }
  /** @type {[string, Tensor][]} */
  const parameters = [["weight", weight], ["bias", bias]];
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
    source: layerNormSource(rows, channels),
    output,
    inputs: { xTex: x, weightTex: weight, biasTex: bias },
    floats: { epsilon },
    label: `layer-norm:${rows}x${channels}`,
  });
}
