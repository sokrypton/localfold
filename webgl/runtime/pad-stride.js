/**
 * A tensor's rows re-laid at a row stride that DIVIDES THE TEXTURE WIDTH, which
 * is worth four times on every projection whose input width does not.
 *
 * 🔴 THIS IS THE ONE THAT WAS HIDING, AND IT IS THE ADDRESSING RATHER THAN THE
 * ARITHMETIC. A logical row of `IN4` texels begins at `row * IN4`, and `fetch4`
 * puts that at `(t % W, t / W)` - so unless `IN4` divides `W` the row WRAPS onto
 * the next texture row at a different offset for every row. A texture is stored
 * swizzled for two-dimensional locality, so a row that wraps is a row whose
 * neighbours are not neighbours, and twelve streams of them defeat the cache.
 *
 * Measured at bioemu2's dominant shape, 576 x 896 -> 1792, with the answer
 * identical (`relRms` 0) in every arm:
 *
 * | `in` | `IN4` | GFLOP/s |
 * |---|---|---|
 * | 768 | 192 | 31.5 |
 * | 896 | 224 | **25.0** |
 * | **1024** | **256** | **95.2** |
 * | 1152 | 288 | 25.6 |
 * | 1792 | 448 | 21.1 |
 * | **2048** | **512** | **113.3** |
 *
 * Only the powers of two are fast, and `W` is 2048, so "divides W" and "is a
 * power of two" are the same condition here. `in = 1024` does **more work** than
 * 896 and is four times faster.
 *
 * 🔴 SO THE FIX IS A STRIDE AND NOT A WIDTH: the row is padded to 256 texels and
 * the loop still runs 224, so not one extra multiply-add is done - 14% more
 * memory for **4.08x**. `tools/gpu/probe-webgl-2d-weight.js` is where that is
 * measured, and it also says which operand matters: padding **x** alone is
 * 4.08x and padding the WEIGHT alone is 0.94x, which is why this module exists
 * for activations and the weights are left as they are.
 *
 * 🔴 AND IT IS WHAT `sokrypton/rfd4-proteina-webgpu` SAW FROM THE OTHER SIDE.
 * Its `TW` is 384 "because the row widths the fast paths walk (24, 48, 64, 96,
 * 128, 192, 384 texels) divide it, so no such row wraps onto the next texture
 * row". It chose the WIDTH to suit its rows; this chooses the row to suit the
 * width, which is the same property. Narrowing `W` here was measured as a 3.8x
 * LOSS (webgl/README.md) because it makes more rows wrap, not fewer - the
 * condition was never the width's size.
 */
import { ONE_TARGET, PRELUDE, sized, memoSource } from "./execution.js";

/** The next power of two at or above `n`, which is what divides `TEXTURE_WIDTH`. */
export const strideFor = (texelsPerRow) => 2 ** Math.ceil(Math.log2(texelsPerRow));

/** Whether a row of `texelsPerRow` texels already divides the texture width. */
export const strideDividesWidth = (texelsPerRow) => strideFor(texelsPerRow) === texelsPerRow;

/**
 * One pass: `[rows][inChannels]` into `[rows][stride4 * 4]`, the tail zeroed.
 *
 * The zeros are read by the matmul's `k` loop only if it runs past the true
 * width, which it does not - they exist so the texture has no undefined texels,
 * which `Storage.fillFresh` would otherwise catch.
 */
const padStrideSourceUncached = (inChannels, stride4) => sized({ IN4: inChannels / 4, STRIDE4: stride4 }, `${PRELUDE}${ONE_TARGET}
uniform int ROWS;
uniform sampler2D xTex;

void main() {
  int texel = linearTexel();
  if (texel >= ROWS * STRIDE4) { result = vec4(0.0); return; }
  int row = texel / STRIDE4;
  int k = texel - row * STRIDE4;
  result = k < IN4 ? fetch4(xTex, row * IN4 + k) : vec4(0.0);
}`);

/**
 * @param {import("./execution.js").Execution} execution
 * @param {{x: import("./storage.js").Tensor, rows: number, inChannels: number,
 *   label?: string}} call
 * @returns {{tensor: import("./storage.js").Tensor, stride4: number}}
 */
export function padStride(execution, { x, rows, inChannels, label = "x-stride" }) {
  const in4 = inChannels / 4;
  const stride4 = strideFor(in4);
  // 🔴 AND THIS IS THE ONE PLACE x CAN BE NARROWED, which is worth 1.31x on its
  // own and 2.56x beside the tile-aware order - measured by
  // probe-webgl-linear-operand.js, where x is the stream that binds the
  // projection and the weight codes cost nothing. `Storage.uploadWeight`'s
  // comment says why it is only offered for weights - "an activation is written
  // by a kernel drawing into it, which would need an RGBA16F render target and
  // the rounding discipline that goes with one" - and this pass IS that kernel:
  // it already rewrites every element of x, and `linear.int5-decoded` is
  // already an RGBA16F target this backend draws into, so the render target is
  // not the obstacle and never was.
  //
  // 🔴 IT IS A PRECISION CHANGE AND THE ONLY ONE ON THE ACTIVATION PATH, so it
  // is OFF in `Execution` and ON in the fold drivers, exactly as `halfWeights`
  // is: the stage gates hold these kernels to f32 references at bars down to
  // 1e-7 and an f16 x would re-bar every one of them. It is also what the
  // PRIMARY path does - webgpu/af2/evoformer/block.js sets `normalizedStorage`
  // to f16 outright - so this is matching the reference rather than departing
  // from it.
  const tensor = execution.halfActivations
    ? execution.allocateHalf(rows * stride4 * 4, label)
    : execution.allocate(rows * stride4 * 4, label);
  execution.dispatch({
    source: padStrideSource(inChannels, stride4),
    output: tensor,
    inputs: { xTex: x },
    ints: { ROWS: rows },
    label: `pad-stride:${rows}x${inChannels}->${stride4 * 4}`
      + `${execution.halfActivations ? ":half" : ""}`,
  });
  return { tensor, stride4 };
}

// Every builder above, memoised on its arguments (`memoSource` in webgl/runtime/execution.js).
export const padStrideSource = memoSource(padStrideSourceUncached);
