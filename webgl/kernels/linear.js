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
import { ONE_TARGET, PRELUDE, targetDeclarations } from "../runtime/execution.js";
import { TEXTURE_WIDTH } from "../runtime/context.js";
import {
  INT5_DECODE, INT5_DECODE8, INT5_GROUP, int5DecodeSource,
} from "../runtime/int5.js";
import { padStride, strideDividesWidth } from "../runtime/pad-stride.js";
/** @typedef {import("../runtime/storage.js").Tensor} Tensor */
/** @typedef {import("../runtime/storage.js").Int5Weight} Int5Weight */


const ACTIVATIONS = {
  none: "value",
  relu: "max(value, vec4(0.0))",
  sigmoid: "vec4(1.0) / (vec4(1.0) + exp(-value))",
};

/** `half4` is in the prelude; see `webgl/runtime/execution.js` for why. */
const ROUND_HALF = (expression) => `half4(${expression})`;

/**
 * The shapes, as UNIFORMS rather than baked constants.
 *
 * 🔴 A BAKED SHAPE IS A SHADER, AND A SHADER IS A COMPILE. Every kernel here is
 * generated with its sizes as `const int`, so `linear` at 576x896x1792 and
 * `linear` at 68x768x3072 were two source texts and two programs - and a
 * bioemu2 fold has ELEVEN distinct projection shapes, so it compiled **168**
 * programs where it now compiles 109, and `programMs` went 3927 ms to 478.
 *
 * 🔴 AND THAT BOUGHT NOTHING ON THAT FOLD, WHICH IS WHY IT IS RECORDED AS A
 * SMALL RESULT. Compilation was the suspect for bioemu2's 28 s because the
 * draw-doubling instrument said the GPU was 2% - and that instrument was wrong
 * (`Execution.doubleLabels`); the draws are 97%, so removing 59 compiles moved
 * the fold by less than noise. It is still worth having, and it is worth
 * everything on the small models: **1122 ms of an AF2 fold's 1400 is its 120
 * programs**, so a first fold there is a compile queue and nothing else.
 *
 * With the sizes in uniforms one program serves every size, and only the
 * STRUCTURE - the weight layout, the activation, whether there is a bias,
 * whether a width is ragged - still forks the source. The cost is that the k
 * loop's bound is no longer a constant, so the compiler cannot unroll it;
 * tools/gpu/bench-webgl-linear.js says that costs nothing measurable, because
 * these kernels are fetch-bound rather than issue-bound.
 *
 * `IN4` and `OUT4` are passed rather than computed in the shader: they are
 * needed per fragment and an integer division is not free.
 */
const SHAPE_UNIFORMS = `uniform int ROWS;
uniform int IN;
uniform int IN4;
uniform int OUT;
uniform int OUT4;`;

/** The shape uniforms' values, as `optionalInts` (a branch may not read one). */
export const shapeInts = (rows, inChannels, outChannels) => ({
  ROWS: rows,
  IN: inChannels,
  IN4: Math.ceil(inChannels / 4),
  OUT: outChannels,
  OUT4: Math.ceil(outChannels / 4),
});

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
${SHAPE_UNIFORMS}
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
 * The BLOCKED projection: one fragment computes `R` rows x four output
 * channels, writing one render target per row.
 *
 * 🔴 THIS IS THE GEOMETRY THE HEADER ABOVE SAYS TO REVISIT, AND IT IS WORTH
 * FIVE TIMES THE ARITHMETIC INTENSITY. A fragment that owns four outputs of ONE
 * row reads five texels for sixteen multiply-adds - 3.2 flops a fetch, where
 * docs/A100.md measures the fragment path saturating at 21.3. A fragment that
 * owns four rows of the same four outputs reads the SAME four weight texels
 * once and four input texels against them: **eight fetches for sixty-four
 * multiply-adds**, 16 flops a fetch, because the weight read is amortised over
 * the row block instead of being repeated for every row.
 *
 * The four results cannot share a texel - they belong to four different rows of
 * the output - so they go to four render targets, and a second pass interleaves
 * them back into `[rows][outChannels]`. `MAX_DRAW_BUFFERS` is at least four
 * everywhere WebGL2 is, and eight here.
 *
 * 🔴 AND THE REPACK IS NOT FREE BUT IT IS CHEAP. It is one texel read and one
 * texel write per output texel, against the matmul's `IN4 * 2` reads for the
 * same texel, so at the narrowest width any model here uses (in = 32) it is 6%
 * of the fetches and at in = 128 it is 1.5%. A consumer that could read the
 * four planes directly would save it; none here can, and a plane layout that
 * leaks into every caller is how the reference port's `linear` grew six return
 * shapes.
 */
/**
 * Four, which is the floor `MAX_DRAW_BUFFERS` is guaranteed to reach wherever
 * WebGL2 is, and the default for a source generated without an `Execution` to
 * ask. `blockRowsFor` below is what a dispatch actually uses.
 */
export const BLOCK_ROWS = 4;

/**
 * Rows a fragment owns, which is a WEIGHT TRAFFIC divisor and nothing else.
 *
 * 🔴 A BIG PROJECTION HERE IS BOUND BY STREAMING ITS WEIGHT FROM DRAM, AND THE
 * ARITHMETIC MODEL IS EXACT. bioemu2's atom projection is 576 rows x 896 in x
 * 1792 out: one output texel a fragment reads the whole 6.4 MiB weight once per
 * ROW, which is 3.7 GB, and this context streams about 10 GB/s - 438 ms
 * measured against 370 predicted. A block of R rows reads it once per R rows,
 * so the traffic, and the time, divide by R: at R = 4 the same projection is
 * 92.3 ms where the model says 92. Nothing else about the kernel matters at
 * this size.
 *
 * So R is as large as the render targets allow. `MAX_DRAW_BUFFERS` is at least
 * four wherever WebGL2 is and eight on this context, and a fragment writes one
 * texel of each - the planes are interleaved back by `blockRepackSource`.
 */
export const blockRowsFor = (execution) =>
  (execution.linearBlockRows || Math.min(8, execution.maxDrawBuffers));

/**
 * Order fragments ROW BLOCK FASTEST when x is no bigger than this, and by
 * output group otherwise.
 *
 * 🔴 THE ORDER DECIDES WHICH OPERAND STREAMS AND WHICH ONE STAYS IN CACHE, AND
 * GETTING IT THE WRONG WAY ROUND IS WORTH FIVE TIMES. Ordered by output group,
 * neighbouring fragments share x and walk different weight rows, so the WEIGHT
 * streams - once per row block, which for a 576-row projection is 72 times.
 * Ordered row block fastest they share the weight rows and x streams instead.
 * So the right order is whichever operand is smaller:
 *
 *   - bioemu2's atom projection is 576 x 896: x is 2 MiB, the weight 1.4 to
 *     3.2 MiB, and row-block-fastest is **1.23x** (6414 ms to 5212 on the cheap
 *     fold, with the fold's answer identical).
 *   - the trunk's pair projection is 4624 x 256: x is **4.7 MiB** and the
 *     weight 262 KiB, and row-block-fastest is **5.1x WORSE** (the trunk stage
 *     5190 ms to 26456).
 *
 * A million floats is the threshold, which is the same number
 * `sokrypton/rfd4-proteina-webgpu` picks for the same decision (`cmLimit`); its
 * own note is that ordered by columns "each row block streamed the whole weight
 * matrix again... about 0.2 TFLOP/s, 350 ms of a trunk pass".
 *
 * 🔴 AND THIS IS WHAT THE INT5 KERNEL WAS REALLY WAITING FOR. Its diagnostic:
 * make every chunk read the SAME four code texels and a bioemu2 fold goes 6664
 * ms to 3476, so the kernel was bound by code-fetch MISSES - not by bandwidth
 * (float16 at 2.3x the bytes took the same time), not by the x reads (14%), and
 * not by occupancy (fewer live registers measured 2% worse). Three wrong guesses
 * before the cache.
 */
export const ROW_FASTEST_X_FLOATS = 1 << 20;

/**
 * Whether the row block is worth its repack, measured by
 * tools/gpu/bench-webgl-linear.js at the shapes four models fold at.
 *
 * `rows >= 64` because a block of R rows costs a 1/R of the fragments, and
 * below that there are too few left to fill the device; the widths must both be
 * multiples of four or a texel straddles two rows and the shared weight read is
 * not shared.
 *
 * 🔴 AND FOR `[in][out]` THE RULE IS EMPIRICAL AND NON-MONOTONIC, WHICH AN
 * ATTEMPT TO SIMPLIFY IT PROVED. The gain tracks how much weight DRAM traffic
 * there is to divide, so it runs from **6.2x to 8.4x** on bioemu2's atom and
 * token projections (1.5 to 3.5 million weight floats, streamed once per row
 * block) down to a loss where the weight sits in cache and the repack is pure
 * cost. For `[in][out]` the three sizes these models use do not line up:
 *
 *   - 128 x 128, 64 KiB: **0.44x to 0.65x**, a loss, repeatably;
 *   - 128 x 512, 256 KiB: a WIN - 1.18x to 1.38x on the bench, and removing it
 *     made an ESMFold2 fold **9.5% slower** (63.6 s to 69.7);
 *   - 256 x 512, 512 KiB: a loss - removing it took a bioemu2 trunk **5245 ms
 *     to 4152** and `linear.repack` from 9.2% of that fold to nothing.
 *
 * Loss, win, loss. 🔴 **SO "IN-MAJOR NEVER BLOCKS" WAS TRIED AND IS A NET LOSS
 * ACROSS THE MODELS**: it bought bioemu2 2.5% of a fold and cost ESMFold2 9.5%
 * of one, which is the whole argument for running four models' folds on every
 * verification pass rather than the one being optimised. The 65,536-float
 * threshold below is fitted to those three points and is not a model of
 * anything; a fourth shape could fall either way and would need measuring.
 * `[out][in]` wins at every size these models use.
 */
export const CACHE_RESIDENT_WEIGHT_FLOATS = 65536;

export const blockedApplies = (rows, inChannels, outChannels, weightLayout = "out-major",
  aspect = true) => {
  if (!(rows >= 64 && inChannels % 4 === 0 && outChannels % 4 === 0)) return false;
  if (weightLayout === "out-major") return true;
  if (!aspect) return inChannels * outChannels >= CACHE_RESIDENT_WEIGHT_FLOATS;
  // 🔴 AND FOR `[in][out]` THE OLD `in * out` THRESHOLD IS WRONG, BECAUSE IT WAS
  // FITTED WITH THE WRONG DTYPE. `bench-webgl-linear.js` uploaded its weights
  // with `upload`, which keeps f32, where every fold uses `uploadWeight` and
  // gets FLOAT16 above 65536 elements - and f16 halves the one-texel path's
  // weight traffic, which helps it far more than it helps the blocked one. So
  // the threshold was fitted on a kernel pair that no fold runs. With the bench
  // on `uploadWeight` and four interleaved rounds, the two largest in-major
  // labels in a bioemu2 fold come out on the ONE-TEXEL side:
  //
  //   4624 x 256 -> 512    one-texel 226-240 GFLOP/s   blocked 157-159   ONE 1.43-1.53x
  //   4624 x 256 -> 256    one-texel 239-244           blocked 186-188   ONE 1.29-1.30x
  //   4624 x 1024 -> 256   one-texel 153               blocked 179       blocked 1.17x
  //   4624 x 256 -> 2048   one-texel 231-234           blocked 475-481   blocked 2.03-2.09x
  //
  // 🔴 AND `in * out` IS NOT THE DISCRIMINATOR AT ALL: 384x512 and 256x768 are
  // the same 196,608 and disagree, as do 512x512 and 256x1024 at 262,144. What
  // fits ten of eleven measured shapes is the ASPECT RATIO - blocking wins when
  // one dimension is at least four times the other, which is what an 8-by-4
  // register tile should prefer, and the one-texel kernel's 1-by-4 suits a
  // balanced shape. The eleventh is 256x768, where the one-texel arm collapses
  // to 68 GFLOP/s because its weight stride is 192 texels and 192 does not
  // divide the texture width - the row-wrap fault of
  // webgl/runtime/pad-stride.js, in the weight this time - so that case takes
  // the block too.
  //
  // `--no-linear-aspect` restores the `in * out` threshold.
  const wide = Math.max(inChannels, outChannels) >= 4 * Math.min(inChannels, outChannels);
  return wide || !strideDividesWidth(outChannels / 4);
};

export const blockedSource = (
  rows, inChannels, outChannels, activation, useBias, weightLayout, roundHalf,
  blockRows = BLOCK_ROWS, orderRowFastest = false, xStride4 = 0,
) => {
  const each = (template) => Array.from(
    { length: blockRows }, (_, i) => template(i)).join("\n");
  // 🔴 UNROLLED BY HAND, BECAUSE AN INDEXED LOCAL ARRAY COST 7x. The first
  // version of this kernel held its accumulators in `vec4 acc[4]` and its row
  // offsets in `int xBase[4]`, written against `for (int i = 0; i < 4; ++i)` -
  // the shape the arithmetic is naturally in, and the shape the reference port
  // this geometry comes from uses. Measured through
  // tools/gpu/bench-webgl-linear.js it ran at **7.3 G fetches a second against
  // the device's 110**: this compiler puts an array that any loop indexes into
  // scratch memory rather than registers, so every accumulate became a
  // round trip to memory and the kernel was 5-7x SLOWER than the one-texel path
  // it was meant to beat. Named variables are the same arithmetic and
  // measure 2.4x FASTER than that path. A loop whose bound is a constant is not
  // a promise that the compiler will unroll it.
  const accumulate = weightLayout === "in-major"
    // `[in][out]`: four weight texels, one per input channel of this k group,
    // each already holding the four outputs this fragment owns - and all four
    // ROWS want the same four, which is the whole saving.
    ? `  for (int k = 0; k < IN4; ++k) {
    int wRow = (k * 4) * OUT4 + group;
    vec4 wa = fetch4(weightTex, wRow);
    vec4 wb = fetch4(weightTex, wRow + OUT4);
    vec4 wc = fetch4(weightTex, wRow + 2 * OUT4);
    vec4 wd = fetch4(weightTex, wRow + 3 * OUT4);
${each((i) => `    vec4 x${i} = fetch4(xTex, xBase${i} + k);
    acc${i} += x${i}.x * wa + x${i}.y * wb + x${i}.z * wc + x${i}.w * wd;`)}
  }`
    // `[out][in]`: four weight rows of this output group, one texel each per k
    // group, dotted against each row's input texel.
    : `  int wBase = group * 4 * IN4;
  for (int k = 0; k < IN4; ++k) {
    vec4 w0 = fetch4(weightTex, wBase + k);
    vec4 w1 = fetch4(weightTex, wBase + IN4 + k);
    vec4 w2 = fetch4(weightTex, wBase + 2 * IN4 + k);
    vec4 w3 = fetch4(weightTex, wBase + 3 * IN4 + k);
${each((i) => `    vec4 x${i} = fetch4(xTex, xBase${i} + k);
    acc${i} += vec4(dot(x${i}, w0), dot(x${i}, w1), dot(x${i}, w2), dot(x${i}, w3));`)}
  }`;
  const value = (i) => {
    const sum = useBias ? `acc${i} + fetch4(biasTex, group)` : `acc${i}`;
    const activated = ACTIVATIONS[activation].replace(/\bvalue\b/g, `(${sum})`);
    return roundHalf ? `half4(${activated})` : activated;
  };
  return `${PRELUDE}${targetDeclarations(blockRows)}
${SHAPE_UNIFORMS}
uniform int RQ;
// 🔴 x's ROW STRIDE, which must divide the texture width or every logical row
// wraps at a different offset and the cache cannot follow twelve such streams.
// See webgl/runtime/pad-stride.js: padding it is 4.08x and costs no arithmetic.
const int XSTRIDE4 = ${xStride4 || inChannels / 4};
uniform sampler2D xTex;
uniform sampler2D weightTex;
${useBias ? "uniform sampler2D biasTex;" : ""}

void main() {
  int texel = linearTexel();
${orderRowFastest
    ? `  int group = texel / RQ;
  int q = texel - group * RQ;`
    : `  int q = texel / OUT4;
  int group = texel - q * OUT4;`}
  // 🔴 THE TAIL IS CLAMPED, NOT SKIPPED. A row block that runs past the row
  // count still writes every target, because a target left unwritten is a texel
  // nothing wrote - the fault Storage.fillFresh exists to catch. Clamping to
  // the last real row keeps the value FINITE and the repack never reads it.
  int base = texel < RQ * OUT4 ? q * ${blockRows} : 0;
${each((i) => `  int xBase${i} = min(base + ${i}, ROWS - 1) * XSTRIDE4;`)}
${each((i) => `  vec4 acc${i} = vec4(0.0);`)}
${accumulate}
${each((i) => `  result${i} = ${value(i)};`)}
}`;
};

/**
 * The blocked projection with its weight DECODED FROM FIVE-BIT CODES.
 *
 * 🔴 THE SAME GEOMETRY AS `blockedSource`, AND THE ONLY DIFFERENCE IS WHERE THE
 * WEIGHT COMES FROM. A big projection is bound by streaming its weight, and the
 * codes are **0.875 bytes a weight against float16's two** while also being
 * fewer texels (an RGBA32UI texel holds 25.6 codes) - and the value decoded is
 * the float32 the host would have produced, so this is faster AND more accurate
 * than the f16 path it replaces. See webgl/runtime/int5.js for the format.
 *
 * 🔴 THE LOOP IS NESTED BY GROUP BECAUSE THE SCALE AND ZERO ARE PER GROUP. A
 * group is 32 elements, so one fetch of (scale, zero) per output row serves
 * eight sub-steps of four - written flat it would fetch the table once per four
 * weights and the table would cost more than the codes. `[out][in]` only: the
 * codes run along the tensor's contiguous axis, which is `in` for that layout
 * and is the axis a fragment sweeps.
 */
export const blockedInt5Source = (
  rows, inChannels, outChannels, activation, useBias, roundHalf, blockRows = BLOCK_ROWS,
  orderRowFastest = true, xStride4 = 0, tileWidth = 0,
) => {
  const each = (template) => Array.from(
    { length: blockRows }, (_, i) => template(i)).join("\n");
  const value = (i) => {
    const sum = useBias ? `acc${i} + fetch4(biasTex, group)` : `acc${i}`;
    const activated = ACTIVATIONS[activation].replace(/\bvalue\b/g, `(${sum})`);
    return roundHalf ? `half4(${activated})` : activated;
  };
  return `${PRELUDE}${targetDeclarations(blockRows)}
${SHAPE_UNIFORMS}
uniform int RQ;
// 🔴 x's ROW STRIDE, which must divide the texture width or every logical row
// wraps at a different offset and the cache cannot follow twelve such streams.
// See webgl/runtime/pad-stride.js: padding it is 4.08x and costs no arithmetic.
const int XSTRIDE4 = ${xStride4 || inChannels / 4};
uniform int GPR;
uniform sampler2D xTex;
uniform highp usampler2D codeTex;
uniform sampler2D tableTex;
${useBias ? "uniform sampler2D biasTex;" : ""}
${INT5_DECODE}
${INT5_DECODE8}

void main() {
  int texel = linearTexel();
  // 🔴 ROW BLOCK FASTEST, WHICH IS WORTH MORE THAN THE BYTES ARE. Ordered by
  // output group (q = texel / OUT4) neighbouring fragments walk DIFFERENT
  // weight rows and the weight is streamed again for every one of the 72 row
  // blocks a 576-row projection has. Ordered row block fastest they share the
  // same four rows, which for int5 codes is 2.2 KiB and stays in cache for the
  // whole column. The diagnostic that found it: make every chunk read the SAME
  // four code texels and the fold goes 6664 ms to 3476 - so this kernel was
  // bound by code-fetch MISSES, not by bandwidth (float16 at 2.3x the bytes
  // took the same time) and not by the x reads (worth 14%) or by occupancy
  // (fewer live registers measured 2% worse).
${tileWidth > 0
    ? `  // 🔴 TILE-AWARE, BECAUSE A RASTERISATION TILE IS 2D AND linearTexel() IS 1D.
  // Ordered row block fastest a TILEW-wide run of texels covers TILEW
  // consecutive row blocks at ONE output group, where the 1D order gave a tile
  // every row block there is - 918 KiB of x under one tile at bioemu2's
  // 576x896x1792 against ${tileWidth} * ${blockRows} rows. The same mapping, and the same
  // reason, as blockedContractSource; measured by
  // probe-webgl-linear-operand.js, where x binds at 6.5x and the codes at 1.0.
  const int TILEW = ${tileWidth};
  int lane = texel % TILEW;
  int group = (texel % W) / TILEW;
  int q = (texel / W) * TILEW + lane;
  // 🔴 AND THIS GRID IS PADDED, so the tail must write rather than wrap: q
  // advances by TILEW a texel row, so RQ row blocks take ceil(RQ/TILEW) rows of
  // W and the leftover fragments are real draws. Wrapping them with a modulo
  // would make them compute a second copy of real work.
  if (q >= RQ || group >= OUT4) {
${each((i) => `    result${i} = vec4(0.0);`)}
    return;
  }`
    : orderRowFastest
      ? `  int group = texel / RQ;
  int q = texel - group * RQ;`
      : `  int q = texel / OUT4;
  int group = texel - q * OUT4;`}
  int o0 = group * 4;
  int base = ${tileWidth > 0 ? `q * ${blockRows}` : `texel < RQ * OUT4 ? q * ${blockRows} : 0`};
${each((i) => `  int xBase${i} = min(base + ${i}, ROWS - 1) * XSTRIDE4;`)}
${each((i) => `  vec4 acc${i} = vec4(0.0);`)}
  for (int g = 0; g < GPR; ++g) {
    // The group index of output row o at its group g is o * GPR + g, because
    // every row is a whole number of groups (int5WeightSource refuses otherwise).
    vec2 sz0 = int5ScaleZero(tableTex, (o0 + 0) * GPR + g);
    vec2 sz1 = int5ScaleZero(tableTex, (o0 + 1) * GPR + g);
    vec2 sz2 = int5ScaleZero(tableTex, (o0 + 2) * GPR + g);
    vec2 sz3 = int5ScaleZero(tableTex, (o0 + 3) * GPR + g);
    // Four chunks of eight codes a group: forty bits each, two words each, and
    // the address arithmetic paid once per eight weights instead of per four.
    for (int chunk = 0; chunk < ${INT5_GROUP / 8}; ++chunk) {
      int k = g * ${INT5_GROUP} + chunk * 8;
      vec4 w0a, w0b, w1a, w1b, w2a, w2b, w3a, w3b;
      decodeInt5x8(codeTex, (o0 + 0) * IN + k, sz0.x, sz0.y, w0a, w0b);
      decodeInt5x8(codeTex, (o0 + 1) * IN + k, sz1.x, sz1.y, w1a, w1b);
      decodeInt5x8(codeTex, (o0 + 2) * IN + k, sz2.x, sz2.y, w2a, w2b);
      decodeInt5x8(codeTex, (o0 + 3) * IN + k, sz3.x, sz3.y, w3a, w3b);
      int kq = k >> 2;
${each((i) => `      {
        vec4 xa = fetch4(xTex, xBase${i} + kq);
        vec4 xb = fetch4(xTex, xBase${i} + kq + 1);
        acc${i} += vec4(
          dot(xa, w0a) + dot(xb, w0b),
          dot(xa, w1a) + dot(xb, w1b),
          dot(xa, w2a) + dot(xb, w2b),
          dot(xa, w3a) + dot(xb, w3b));
      }`)}
    }
  }
${each((i) => `  result${i} = ${value(i)};`)}
}`;
};

/** Four row planes back into `[rows][outChannels]`: plane `row % 4`, texel `(row / 4) * OUT4 + group`. */
export const blockRepackSource = (blockRows = BLOCK_ROWS, orderRowFastest = false,
  tileWidth = 0) => `${PRELUDE}${ONE_TARGET}
uniform int ROWS;
uniform int OUT4;
uniform int RQ;
${Array.from({ length: blockRows }, (_, i) => `uniform sampler2D plane${i};`).join("\n")}

void main() {
  int texel = linearTexel();
  if (texel >= ROWS * OUT4) { result = vec4(0.0); return; }
  int row = texel / OUT4;
  int group = texel - row * OUT4;
  int source = ${tileWidth > 0
    ? "((row / " + blockRows + ") / " + tileWidth + ") * W + group * " + tileWidth
      + " + ((row / " + blockRows + ") % " + tileWidth + ")"
    : orderRowFastest
      ? "group * RQ + (row / " + blockRows + ")"
      : "(row / " + blockRows + ") * OUT4 + group"};
  int plane = row - (row / ${blockRows}) * ${blockRows};
${Array.from({ length: blockRows - 1 }, (_, i) =>
  `  if (plane == ${i}) { result = fetch4(plane${i}, source); return; }`).join("\n")}
  result = fetch4(plane${blockRows - 1}, source);
}`;

/**
 * An int5 weight as an ordinary float16 tensor, decoded once and memoised on
 * the weight object so a stacked layer pays for it only on its first block.
 *
 * @param {import("../runtime/execution.js").Execution} execution
 */
/**
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {Int5Weight} weight
 * @param {string} shape
 * @returns {Tensor}
 */
function decodedInt5(execution, weight, shape) {
  if (weight.decoded !== undefined) return weight.decoded;
  const decoded = execution.allocateHalf(weight.elements, "linear.int5-decoded");
  execution.dispatch({
    source: int5DecodeSource(PRELUDE, ONE_TARGET, true),
    output: decoded,
    inputs: { codeTex: weight.codes, tableTex: weight.table },
    ints: { ELEMENTS: weight.elements },
    label: `linear.int5-decode:${weight.elements}:${shape}`,
  });
  weight.decoded = decoded;
  // The codes are dead once it is decoded, and a stacked weight holds both
  // otherwise; `free` on the weight object releases whichever exist.
  return decoded;
}

/**
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{x: Tensor, weight: Tensor | Int5Weight, bias?: Tensor, rows: number, inChannels: number,
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
  // An int5 weight decodes in the kernel; it has no single texture to sample.
  // A shape the inner loop cannot address is decoded once instead - see
  // `int5DecodeSource` for which shapes those are and why the uploader cannot
  // tell them apart.
  let int5 = /** @type {Int5Weight} */ (weight).int5 === true;
  /** @type {any} */
  let weightTensor = weight;
  const codeWeight = /** @type {Int5Weight} */ (weight);
  if (int5 && (weightLayout !== "out-major" || inChannels % INT5_GROUP !== 0)) {
    weightTensor = decodedInt5(execution, codeWeight, `${rows}x${inChannels}x${outChannels}`);
    int5 = false;
  }
  if (!execution.noLinearBlock
    && blockedApplies(rows, inChannels, outChannels, weightLayout,
      !execution.noLinearAspect)) {
    const blockRows = blockRowsFor(execution);
    const rowQuads = Math.ceil(rows / blockRows);
    // Row block fastest only where it is measured: the int5 path, whose codes
    // are small enough per output group to stay in cache across a column.
    // 🔴 THE INT5 PATH ONLY, AND THAT IS A MEASUREMENT AND NOT CAUTION. Applied
    // to the float paths as well this cost AF2 **3.7x** - 1264 ms to 4698 - and
    // the reason is the REPACK, not the matmul: ordered row block fastest, the
    // repack's source index strides by the row-block count, which is 72 for
    // bioemu2's atom projection and **436** for AF2's pair track, so every
    // output texel lands on its own cache line. The order is only worth having
    // where the block count is small enough that the repack survives it.
    const rowFastest = int5 && !execution.noRowFastest
      && rows * inChannels <= ROW_FASTEST_X_FLOATS;
    // 🔴 x's ROW STRIDE PADDED TO DIVIDE THE TEXTURE WIDTH, which is 4.08x on
    // every shape whose input width does not - see webgl/runtime/pad-stride.js.
    // One pass over x against sixty milliseconds of cache misses; measured, the
    // WEIGHT does not want this (0.94x) and only x does.
    // 🔴 THE TILE-AWARE FRAGMENT ORDER, AND ITS WIDTH IS A MEMORY DECISION AS
    // MUCH AS A LOCALITY ONE. The mapping gives a texel ROW every output group,
    // so the planes are padded to `W / TILEW` groups where only `OUT4` are
    // real: at bioemu2's 1792 outputs TILEW 2 pads 2.29x and is what
    // probe-webgl-linear-operand.js measures at **1.74-1.87x** over the shipped
    // 1D order, but the same TILEW at AF2's 256 outputs would pad SIXTEENFOLD -
    // 4.7 MiB of planes to 75.8 - which is not a trade this backend should make
    // silently. So the width is the NARROWEST that stays inside a 2.5x padding
    // budget, which prefers the measured 2 and widens for a narrow output
    // rather than declining to tile at all.
    //
    // 🔴 AND ONLY WHERE `rowFastest` ALREADY APPLIES, because that is the order
    // the 1.87x is measured AGAINST. Pointed at the group-fastest float paths
    // it would be an unmeasured change to the arm that cost AF2 3.7x.
    const out4 = Math.ceil(outChannels / 4);
    const tilePadding = (width) => Math.floor(TEXTURE_WIDTH / width) / out4;
    const wanted = Number(execution.linearTileWidth) || 0;
    const tileWidth = (!execution.noLinearTile && rowFastest)
      ? (wanted > 0
        ? (tilePadding(wanted) >= 1 ? wanted : 0)
        : ([2, 4, 8, 16, 32].find((w) => tilePadding(w) >= 1 && tilePadding(w) <= 2.5) ?? 0))
      : 0;
    const tileHeight = tileWidth > 0 ? Math.ceil(rowQuads / tileWidth) : 0;
    const padX = !execution.noStridePad && !strideDividesWidth(inChannels / 4);
    const padded = padX
      ? padStride(execution, { x, rows, inChannels, label: "linear.x-stride" })
      : null;
    const xUse = padded?.tensor ?? x;
    const xStride4 = padded?.stride4 ?? 0;
    const planeFloats = tileWidth > 0
      ? tileHeight * TEXTURE_WIDTH * 4
      : rowQuads * outChannels;
    const planes = Array.from({ length: blockRows },
      (_, i) => execution.allocate(planeFloats, `linear.plane${i}`));
    execution.dispatch({
      source: int5
        ? blockedInt5Source(
          rows, inChannels, outChannels, activation, bias !== undefined, roundHalf, blockRows,
          rowFastest, xStride4, tileWidth)
        : blockedSource(
          rows, inChannels, outChannels, activation, bias !== undefined, weightLayout, roundHalf,
          blockRows, rowFastest, xStride4),
      outputs: planes,
      inputs: int5
        ? (bias === undefined
          ? { xTex: xUse, codeTex: codeWeight.codes, tableTex: codeWeight.table }
          : { xTex: xUse, codeTex: codeWeight.codes, tableTex: codeWeight.table, biasTex: bias })
        : (bias === undefined
          ? { xTex: xUse, weightTex: weightTensor }
          : { xTex: xUse, weightTex: weightTensor, biasTex: bias }),
      optionalInts: {
        ...shapeInts(rows, inChannels, outChannels),
        RQ: rowQuads,
        ...(int5 ? { GPR: inChannels / INT5_GROUP } : {}),
      },
      label: `linear.block${int5 ? ".int5" : ""}:${rows}x${inChannels}x${outChannels}`
        + `:${activation}:${weightLayout}${roundHalf ? ":half" : ""}`
        + `${tileWidth > 0 ? `:tiled${tileWidth}` : ""}`,
    });
    execution.dispatch({
      source: blockRepackSource(blockRows, rowFastest, tileWidth),
      output,
      inputs: Object.fromEntries(planes.map((plane, i) => [`plane${i}`, plane])),
      optionalInts: { ROWS: rows, OUT4: Math.ceil(outChannels / 4), RQ: rowQuads },
      label: `linear.repack:${rows}x${outChannels}`,
    });
    for (const plane of planes) execution.release(plane);
    if (padded !== null) execution.release(padded.tensor);
    return output;
  }
  if (int5) {
    throw new Error(
      `linear: an int5 weight only has the blocked path, and ${rows}x${inChannels}x`
      + `${outChannels} does not take it (blockedApplies says no)`);
  }
  return execution.dispatch({
    source: linearSource(
      rows, inChannels, outChannels, activation, bias !== undefined, weightLayout, roundHalf),
    output,
    inputs: bias === undefined
      ? { xTex: x, weightTex: weightTensor }
      : { xTex: x, weightTex: weightTensor, biasTex: bias },
    optionalInts: shapeInts(rows, inChannels, outChannels),
    label: `linear:${rows}x${inChannels}x${outChannels}:${activation}:${weightLayout}`
      + `${roundHalf ? ":half" : ""}`,
  });
}
