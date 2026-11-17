/**
 * The triangle contraction - `sum_k a[i,k,h] * b[j,k,h]` - as one shader source
 * with two drivers.
 *
 * This is the arithmetic that was ported first, because it is 22% of an
 * Evoformer block at 221 residues and the one operation in the model with a CPU
 * reference already written. It lives here rather than inside the benchmark
 * class that used to own it so that the STAGE path (`./triangle-multiplication.js`,
 * through `webgl/runtime/execution.js` and its one allocator) and the BENCHMARK
 * path (`./triangle.js`, which splits prepare from dispatch so a timing excludes
 * the uploads) compile the same text. Two copies of a contraction is the drift
 * this repository keeps paying for.
 *
 * 🔴 L AND CH ARE COMPILE-TIME CONSTANTS, NOT UNIFORMS, so the index arithmetic
 * folds: `h = index % CH` becomes a mask when CH is a power of two and the loop
 * bound is known when the shader is optimised. That is also why the program
 * cache in execution.js is keyed by SOURCE - the shapes ARE the source.
 */
import { ONE_TARGET, PRELUDE, memoSource, sized, targetDeclarations } from "../runtime/execution.js";
import { CHANNELS_PER_TEXEL, TEXTURE_WIDTH } from "../runtime/context.js";
import { transposeLeading } from "../runtime/transpose.js";
/** @typedef {import("../runtime/storage.js").Tensor} Tensor */


/**
 * Channel-minor [pair][channel] to channel-major [channel/4][pair], four to a
 * texel. Done on the host, for the layout comparison below; a pipeline would
 * have the pass that produced a and b write this layout directly, exactly as
 * `projectAB` now does on the WGSL side.
 */
export function toChannelMajor(values, pairs, channels) {
  const groups = channels / CHANNELS_PER_TEXEL;
  const out = new Float32Array(values.length);
  for (let pair = 0; pair < pairs; pair += 1) {
    for (let group = 0; group < groups; group += 1) {
      const source = pair * channels + group * CHANNELS_PER_TEXEL;
      const target = (group * pairs + pair) * CHANNELS_PER_TEXEL;
      for (let m = 0; m < CHANNELS_PER_TEXEL; m += 1) out[target + m] = values[source + m];
    }
  }
  return out;
}

export function contractSource(length, channels, direction, layout = "channel-minor") {
  // 🔴 INCOMING SWAPS THE OPERANDS AS WELL AS THE INDICES, AND GETTING THAT
  // WRONG SURVIVED EVERY CHECK UNTIL THE TWO BACKENDS WERE RUN TOGETHER. The
  // WGSL kernel - gated against OpenFold and AlphaFold fixtures, and the
  // authority here - loads `b[k * L + i]` against `a[k * L + j]` for incoming
  // where outgoing loads `a[i * L + k]` against `b[j * L + k]`
  // (`loadATile`/`loadBTile` in webgpu/kernels/triangle/shaders.js). This
  // kernel's first version kept `a` on the i side in both directions, which
  // produces the PAIR-TRANSPOSE of the right answer: `sum_k a[k,i] b[k,j]`
  // against `sum_k b[k,i] a[k,j]`. Measured at relRMS 1.3 - no signal at all -
  // against the WGSL kernel on four shapes.
  //
  // 🔴 AND THE REASON NOTHING CAUGHT IT IS THE TRAP THIS REPOSITORY RECORDS
  // REPEATEDLY: the CPU reference it was first checked against was written FROM
  // THIS SHADER'S OWN INDEX ARITHMETIC, so it encoded the same convention and
  // agreed to 1e-7 while both were wrong. A reference derived from the thing
  // under test cannot see a convention; only a second implementation can, and
  // the operation's own CPU reference covers the OUTGOING direction only.
  const product = direction === "outgoing"
    ? "fetch4(aTex, texel(i, k, h4)) * fetch4(bTex, texel(j, k, h4))"
    : "fetch4(bTex, texel(k, i, h4)) * fetch4(aTex, texel(k, j, h4))";
  // 🔴 CHANNEL-MAJOR IS WORTH 14x ON THE WGSL SIDE AND HURTS HERE. It was added
  // to compare the two APIs on equal terms and the answer was that they are not
  // equal in this respect: WebGL2 never had the problem it fixes. A fragment
  // computes a vec4, so four consecutive channels already arrive in one texel,
  // and neighbouring fragments already cover consecutive channel groups - the
  // reads are coalesced by construction. Rearranging into per-channel planes
  // takes that away and puts a whole L*L plane between neighbouring fragments.
  // Measured on an M2, interleaved, bitwise-identical output:
  //   L=128  0.98 -> 1.25 ms     L=192  6.22 -> 8.32 ms
  //   L=256  16.62 -> 41.72 ms   (3.5x WORSE)
  // Kept because the negative result is the point: it is why the WGSL port
  // looked slow against this one before the WGSL layout was fixed.
  const index = layout === "channel-major"
    ? "int texel(int x, int y, int c) { return c * (L * L) + x * L + y; }"
    : "int texel(int x, int y, int c) { return (x * L + y) * CH4 + c; }";

  return sized({ L: length }, `${PRELUDE}${ONE_TARGET}
const int CH4 = ${channels / CHANNELS_PER_TEXEL};
uniform sampler2D aTex;
uniform sampler2D bTex;
// Added to every element, zero in use. A driver may drop a draw whose state and
// target match the previous one, which makes a repeated dispatch unmeasurable;
// varying this makes each draw observably distinct. Unset uniforms are zero.
uniform float bias;

${index}

void main() {
  int linear = linearTexel();
  if (linear >= L * L * CH4) { result = vec4(0.0); return; }
  int h4 = linear % CH4;
  int pair = linear / CH4;
  int j = pair % L;
  int i = pair / L;
  vec4 sum = vec4(0.0);
  // 🔴 NOT UNROLLED, AND THAT WAS MEASURED. Reading four steps of k before
  // multiplying any of them - so the fetches overlap instead of forming a
  // dependent chain - is the obvious latency hiding for a shader with no
  // workgroup to swap to. It made this kernel SLOWER at four of five lengths
  // (2.5x at 96 residues, 1.3x at 128), presumably because the extra live
  // registers cost more occupancy than the overlap bought. The straight loop
  // stays until something measures better.
  for (int k = 0; k < L; ++k) {
    sum += ${product};
  }
  result = sum + bias;
}`);
}

/**
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{a: Tensor, b: Tensor, length: number, channels: number,
 *   direction?: "outgoing" | "incoming", layout?: "channel-minor" | "channel-major"}} call
 * @param {Tensor} [into]
 * @returns {Tensor}
 */
export function triangleContract(execution, call, into) {
  const { a, b, length, channels, direction = "outgoing", layout = "channel-minor" } = call;
  const elements = length * length * channels;
  if (channels % CHANNELS_PER_TEXEL !== 0) {
    throw new RangeError(`triangleContract: channels must be a multiple of 4, got ${channels}`);
  }
  /** @type {[string, Tensor][]} */
  const operands = [["a", a], ["b", b]];
  for (const [name, tensor] of operands) {
    if (tensor.elements !== elements) {
      throw new RangeError(
        `triangleContract: ${name} holds ${tensor.elements} floats, not ${length}^2 x ${channels}`);
    }
  }
  const output = into ?? execution.allocate(elements, "triangle.contract");
  if (output === a || output === b) {
    throw new Error("triangleContract: a fragment shader cannot read the texture it draws into");
  }
  // 🔴 THE INCOMING DIRECTION READS A COLUMN AND THAT IS A 69 KiB STRIDE A STEP.
  // Both directions do the same arithmetic over the same number of texels, and
  // one of them is four and a half times slower: `outgoing` reads `a[i][k]` as
  // k advances, a stride of CH4 texels - one kilobyte at 256 channels - while
  // `incoming` reads `a[k][i]`, a stride of L*CH4, which at 68 tokens and 256
  // channels is **69 KiB**. Nothing is reused and every fetch is its own cache
  // line. Measured inside a bioemu2 fold's own profile, at 68 tokens: outgoing
  // 236 ms a call, incoming **1055**.
  //
  // Transposing both operands first turns the column into a row, so the
  // incoming contraction becomes the outgoing one on swapped pair axes - the
  // same products in the same k order, which is why the result is BIT-IDENTICAL
  // and `--no-triangle-transpose` is a control arm rather than a precision
  // trade. Two transposes are two passes over L^2 x CH floats, which is a tenth
  // of what the stride costs.
  if (direction === "incoming" && layout === "channel-minor"
    && !execution.noTriangleTranspose) {
    const aT = transposeLeading(execution, { x: a, outer: length, inner: length, channels });
    const bT = transposeLeading(execution, { x: b, outer: length, inner: length, channels });
    execution.dispatch({
      // `outgoing` on the transposed pair reads bT[i][k] * aT[j][k], where
      // incoming reads b[k][i] * a[k][j] - the same two elements.
      source: contractSource(length, channels, "outgoing", layout),
      output,
      inputs: { aTex: bT, bTex: aT },
      label: `triangle.contract:incoming-transposed:${length}x${channels}`,
    });
    execution.release(aT);
    execution.release(bT);
    return output;
  }
  return execution.dispatch({
    source: contractSource(length, channels, direction, layout),
    output,
    inputs: { aTex: a, bTex: b },
    label: `triangle.contract:${direction}:${length}x${channels}`,
  });
}

// ---------------------------------------------------------------------------------------------------------------------
// The pair-blocked contraction over CHANNEL-MAJOR planes, written for AF3's trunk (webgl/af3/trunk/triangle.js, which
// re-exports it) and shared with AF2's triangle multiplication (./triangle-multiplication.js), ESMFold2 and bioemu2.
// Every measurement in its comments is the AF3 lineage's; webgl/README.md has the AF2 numbers.
// ---------------------------------------------------------------------------------------------------------------------

/**
 * The contraction alone: a fragment owns four channels of one pair and
 * contracts exactly those four.
 *
 * 🔴 FOUR CONTRACTIONS A FRAGMENT RATHER THAN `channels` OF THEM, which is the
 * whole saving - 544 texel reads where the fused kernel does 34,816, at 68
 * tokens and 256 channels. The sum over `k` is in the same order and the same
 * products, so the intermediate is exactly what the fused kernel computed and
 * then threw away sixty-three sixty-fourths of.
 */
const splitContractSourceUncached = (length, channels, direction, divideByLength,
  coalesce = true, paddedLength = length) => {
  // 🔴 THE OUTGOING DIRECTION WALKS k ALONG THE CONTIGUOUS AXIS, SO IT CAN READ
  // WHOLE TEXELS - AND `fetch1` WAS ASKING FOR THE SAME TEXEL FOUR TIMES. Four
  // consecutive k of `a[c][i][k]` are four consecutive floats, which is one
  // RGBA texel; `fetch1` fetches that texel and selects one component, so the
  // loop issued four fetches where one would do. The texture cache absorbs the
  // BYTES, but this kernel is bound by the fetch RATE - about 10 G a second -
  // not by bandwidth, so the instruction count is the cost. One `dot` per four
  // k instead of four multiply-adds.
  //
  // `plane` is `c * L * L` and the row offset is `i * L`, so both are multiples
  // of four whenever L is, which is the only case this takes; the general loop
  // stays for an odd length.
  //
  // 🔴 THE INCOMING DIRECTION CANNOT: it walks `a[c][k][j]`, a stride of L
  // floats, so no two of its reads share a texel. Turning it into an outgoing
  // contraction over transposed planes would, and is not done here.
  const coalesced = coalesce && direction === "outgoing" && paddedLength % 4 === 0;
  const body = coalesced
    ? `  for (int k4 = 0; k4 < LP / 4; ++k4) {
    total += dot(fetch4(aTex, (plane + i * LP) / 4 + k4), fetch4(bTex, (plane + j * LP) / 4 + k4));
  }`
    : `  for (int k = 0; k < L; ++k) {
    ${direction === "outgoing"
    ? "total += fetch1(aTex, plane + i * LP + k) * fetch1(bTex, plane + j * LP + k);"
    : "total += fetch1(aTex, plane + k * LP + j) * fetch1(bTex, plane + k * LP + i);"}
  }`;
  return sized({ L: length, LP: paddedLength, PLANE: paddedLength * paddedLength, ELEMENTS: length * length * channels }, `${PRELUDE}${ONE_TARGET}
// The operands' pair axes, padded to a multiple of four so the coalesced loop
// is legal; equal to L whenever nothing needed padding.
const int C4 = ${channels / 4};
uniform sampler2D aTex;
uniform sampler2D bTex;

float contract(int c, int i, int j) {
  float total = 0.0;
  int plane = c * PLANE;
${body}
  return ${divideByLength ? "total / float(L)" : "total"};
}

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int pair = texel / C4;
  int i = pair / L;
  int j = pair - i * L;
  int c = group * 4;
  result = vec4(contract(c, i, j), contract(c + 1, i, j),
    contract(c + 2, i, j), contract(c + 3, i, j));
}`);
};

/**
 * The contraction with a PAIR BLOCK: two i and four j a fragment, into eight
 * render targets.
 *
 * 🔴 THE SPLIT CONTRACTION RUNS AT FOUR FLOPS A FETCH WHERE THE PROJECTION
 * REACHES TWENTY-THREE, AND SHARING THE OPERANDS IS THE ONLY WAY UP. A fragment
 * owning four channels of one pair reads `a[i][*][c]` and `b[j][*][c]` for each
 * of those four channels - and the four channels live in four different planes,
 * so that is 136 texel reads for 272 multiply-adds. Measured inside a 48-block
 * trunk the contraction is about half of it, at roughly **1.8 GFLOP/s** where
 * the same device's projections reach 70.
 *
 * Nothing about the sweep can be made cheaper; the operands can be REUSED. A
 * block of two i and four j computes eight pairs from two `a` rows and four `b`
 * rows, so a k-step costs six fetches instead of eight for four times the
 * arithmetic: **10.7 flops a fetch, 2.7x**. Eight pairs of four channels is
 * exactly `MAX_DRAW_BUFFERS` targets, and `blockRepackSource` interleaves them.
 *
 * The channel loop is unrolled so only eight accumulators are live at a time
 * rather than thirty-two, and so that a `vec4` is never indexed by a variable.
 */
/**
 * @param residentOperands a DIAGNOSTIC ONLY, and never a kernel: it forces every
 *   fragment onto pair block ZERO, so all of them read the same two rows and
 *   four columns and the operand working set collapses from 9.4 MiB to a few
 *   KiB. It computes the wrong answer and `bench-webgl-contract.js` never
 *   compares it for one.
 *
 *   🔴 IT EXISTS BECAUSE THE `aliased` ARM DOES NOT ASK WHAT IT CLAIMS TO.
 *   Handing the kernel one texture as both operands halves the DISTINCT
 *   footprint and leaves every fragment reading its own pair rows, so the
 *   working set is still 4.7 MiB and still past the cache - which is why that
 *   arm reads 1.05-1.14x and rules out far less than it appears to. The
 *   projection's equivalent (`x-aliased` in probe-webgl-linear-operand.js)
 *   makes every fragment read the SAME rows, and there it is 3.44x. This is
 *   that arm for the contraction.
 */
const blockedContractSourceUncached = (length, channels, direction, divideByLength,
  channelSlowest = true, paddedLength = length, tileMapped = false,
  residentOperands = false) => {
  const planes = 8;
  const component = ["x", "y", "z", "w"];
  // For the outgoing direction a row is `a[i][k]`, contiguous in k; for the
  // incoming one the caller has already transposed both operands, so it is the
  // same expression.
  const row = (tex, pair) => `(plane + ${pair} * LP) / 4 + k4`;
  const channelBlock = (c4) => `  {
    float s00 = 0.0, s01 = 0.0, s02 = 0.0, s03 = 0.0;
    float s10 = 0.0, s11 = 0.0, s12 = 0.0, s13 = 0.0;
    int plane = (cgroup * 4 + ${c4}) * PLANE;
    for (int k4 = 0; k4 < LP / 4; ++k4) {
      vec4 a0 = fetch4(aTex, ${row("aTex", "i0")});
      vec4 a1 = fetch4(aTex, ${row("aTex", "i1")});
      vec4 b0 = fetch4(bTex, ${row("bTex", "j0")});
      vec4 b1 = fetch4(bTex, ${row("bTex", "j1")});
      vec4 b2 = fetch4(bTex, ${row("bTex", "j2")});
      vec4 b3 = fetch4(bTex, ${row("bTex", "j3")});
      s00 += dot(a0, b0); s01 += dot(a0, b1); s02 += dot(a0, b2); s03 += dot(a0, b3);
      s10 += dot(a1, b0); s11 += dot(a1, b1); s12 += dot(a1, b2); s13 += dot(a1, b3);
    }
${[0, 1].map((di) => [0, 1, 2, 3].map((dj) =>
  `    result${di * 4 + dj}.${component[c4]} = s${di}${dj}${divideByLength ? " / float(L)" : ""};`)
  .join("\n")).join("\n")}
  }`;
  return sized({ L: length, LP: paddedLength, PLANE: paddedLength * paddedLength, JBLOCKS: paddedLength / 4, BLOCKS: (paddedLength / 2) * (paddedLength / 4), TILEW: Math.max(1, Math.floor(2048 / (channels / 4))) }, `${PRELUDE}${targetDeclarations(planes)}
// The padded pair axes: the block sweeps the PADDED grid, whose tail rows and
// columns are zeros, and the repack reads only the true L of them.
const int C4 = ${channels / 4};
const int IBLOCKS = ${paddedLength / 2};
uniform sampler2D aTex;
uniform sampler2D bTex;

void main() {
  int texel = linearTexel();
  if (${tileMapped ? "(texel / W) * TILEW + (texel % TILEW) >= BLOCKS"
    : "texel >= IBLOCKS * JBLOCKS * C4"}) {
${Array.from({ length: planes }, (_, p) => `    result${p} = vec4(0.0);`).join("\n")}
    return;
  }
${tileMapped
  ? `  // 🔴 A TILE-AWARE MAPPING, BECAUSE A RASTERISATION TILE IS TWO-DIMENSIONAL
  // AND THE TEXEL INDEX IS NOT. A tile covers a square of (x, y) and the texel
  // index is y * W + x, so consecutive ROWS of a tile are W apart in the index -
  // and with BLOCKS of them per channel a 32-row tile spans 32 * W / BLOCKS
  // channel groups, about a hundred at 68 tokens. Ordering the channel slowest
  // fixes the one-dimensional neighbour and leaves that untouched.
  //
  // Here the channel group is chosen by x / TILEW with TILEW = W / C4, so every
  // fragment sharing an x RANGE shares a channel - and a tile is a range of x
  // crossed with a range of y, so a whole tile shares one channel's operands.
  // It needs C4 to divide W, which every power-of-two channel count does.
  int lane = texel % TILEW;
  int cgroup = (texel % W) / TILEW;
  int block = (texel / W) * TILEW + lane;`
  : channelSlowest
    ? `  int block = texel % BLOCKS;
  int cgroup = texel / BLOCKS;`
    : `  int cgroup = texel % C4;
  int block = texel / C4;`}
  int jblock = ${residentOperands ? "0" : "block % JBLOCKS"};
  int iblock = ${residentOperands ? "0" : "block / JBLOCKS"};
  int i0 = iblock * 2;
  int i1 = i0 + 1;
  int j0 = jblock * 4;
  int j1 = j0 + 1;
  int j2 = j0 + 2;
  int j3 = j0 + 3;
${[0, 1, 2, 3].map(channelBlock).join("\n")}
}`);
};

/**
 * The contraction with a 4 i x 8 j x ONE channel block, against the 2 x 4 x 4
 * one above.
 *
 * 🔴 CHANNELS A FRAGMENT COST LOCALITY AND BUY NO INTENSITY, which is the whole
 * reason this shape exists. `bench-webgl-contract.js`'s `resident-tiled` arm
 * puts the contraction at **3.91x off its own operand-resident rate** - 1.26 ms
 * against 4.94, 127 GFLOP/s against 32 - and the working set a rasterisation
 * tile touches is `rows * channels * LP`: 136 pair rows by 4 channels by LP
 * values is **148 KiB** at 68 tokens and 256 channels. One channel a fragment
 * makes it **37 KiB**. And the same 32-float output budget then buys a wider
 * pair block, whose intensity is `8 * I * J / (I + J)`: **21.3 flops a fetch4**
 * at 4 x 8 against the 2 x 4 block's 10.7. Two factors from one change.
 *
 * Measured in the bench, block AND repack against block and repack:
 *
 * | L | 2x4x4 | 4x8x1 | factor |
 * |---|---|---|---|
 * | 64 | 3.49 ms | **2.61** | **1.33x** |
 * | 72 | 6.63 | **3.30** | **2.01x** |
 * | 128 | 25.84 | **14.64** | **1.77x** |
 *
 * 🔴 ITS x AXIS IS THE CHANNEL AND NOT THE CHANNEL GROUP, so `TILEW` is
 * `W / channels` where the other kernel's is `W / (channels / 4)` - 8 rather
 * than 32 at 256 channels. A TILEW-wide run of texels is one channel and TILEW
 * pair blocks, and a tile then holds one channel's operands.
 *
 * 🔴 AND IT NEEDS THE PAIR AXES PADDED TO A MULTIPLE OF EIGHT, where
 * `paddedPairLength` pads to four: at 68 tokens `JBLOCKS` would be 8.5 and the
 * shader does not compile - "cannot convert from const float to const highp
 * int", which is what a non-integer in a `const int` looks like. So this kernel
 * asks for its own padding and 68 tokens become 72, which is 1.19x the padded
 * arithmetic and is already inside the figures above.
 */
const widePairBlockSourceUncached = (length, channels, divideByLength,
  paddedLength = length, tileWidth = 8) => {
  const I = 4;
  const J = 8;
  const slots = [];
  for (let i = 0; i < I; i += 1) {
    for (let j = 0; j < J; j += 1) slots.push(`s${i}_${j}`);
  }
  const value = (name) => (divideByLength ? `${name} / float(L)` : name);
  return sized({ L: length, LP: paddedLength, PLANE: paddedLength * paddedLength, JBLOCKS: paddedLength / J, BLOCKS: (paddedLength / I) * (paddedLength / J), TILEW: tileWidth }, `${PRELUDE}${targetDeclarations(8)}
const int C = ${channels};
uniform sampler2D aTex;
uniform sampler2D bTex;

void main() {
  int texel = linearTexel();
  // 🔴 THE LANE IS TAKEN WITHIN THE ROW, as the repack reads it: texel % TILEW is the same number only when TILEW
  // divides W - 128, 256 and 448 channels (TILEW 16, 8, 4) - and at 192, 288, 320 or 384 (10, 7, 6, 5) every row
  // after the first wrote its blocks into a neighbour's slot: relRMS 1.15 at 24 tokens, from Mol-1's 320-wide trunk.
  int column = texel % W;
  int lane = column % TILEW;
  int channel = column / TILEW;
  int block = (texel / W) * TILEW + lane;
  // The grid is whole texture rows, so its tail is real draws and must write
  // rather than wrap - a target left unwritten is a texel nothing wrote.
  if (block >= BLOCKS || channel >= C) {
${Array.from({ length: 8 }, (_, p) => `    result${p} = vec4(0.0);`).join("\n")}
    return;
  }
  int jblock = block % JBLOCKS;
  int iblock = block / JBLOCKS;
  int plane = channel * PLANE;
${slots.map((a) => `  float ${a} = 0.0;`).join("\n")}
  for (int k4 = 0; k4 < LP / 4; ++k4) {
${Array.from({ length: I }, (_, i) =>
  `    vec4 a${i} = fetch4(aTex, (plane + (iblock * ${I} + ${i}) * LP) / 4 + k4);`).join("\n")}
${Array.from({ length: J }, (_, j) =>
  `    vec4 b${j} = fetch4(bTex, (plane + (jblock * ${J} + ${j}) * LP) / 4 + k4);`).join("\n")}
${Array.from({ length: I }, (_, i) => Array.from({ length: J }, (_, j) =>
    `    s${i}_${j} += dot(a${i}, b${j});`).join("\n")).join("\n")}
  }
${Array.from({ length: 8 }, (_, p) =>
  `  result${p} = vec4(${[0, 1, 2, 3].map((c) => value(slots[p * 4 + c])).join(", ")});`).join("\n")}
}`);
};

/**
 * The wide block's planes back into `[pair][channel group]`.
 *
 * 🔴 IT IS FOUR `fetch4` WHERE THE 2 x 4 x 4 REPACK IS ONE, AND THAT IS THE
 * SHAPE'S REAL PRICE. An output texel is four CHANNELS of one pair, and a wide
 * fragment holds one channel of thirty-two pairs - so the four channels come
 * from four different fragments, which are four source texels of the SAME
 * plane, and (i, j) picks the plane and the component. That cost is inside the
 * bench figures above, which time block and repack together; timing the block
 * alone read 2.76x at L = 72 where the pair reads 2.01x.
 */
const wideRepackSourceUncached = (length, channels, paddedLength = length,
  tileWidth = 8) => sized({ L: length, JBLOCKS: paddedLength / 8, TILEW: tileWidth, ELEMENTS: length * length * channels }, `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
${Array.from({ length: 8 }, (_, p) => `uniform sampler2D plane${p};`).join("\n")}

float pick(vec4 v, int c) {
  return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w;
}

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int cgroup = texel % C4;
  int pair = texel / C4;
  int i = pair / L;
  int j = pair - i * L;
  int block = (i / 4) * JBLOCKS + (j / 8);
  // Not named flat: that is a GLSL interpolation qualifier and a reserved word.
  int slot = (i % 4) * 8 + (j % 8);
  int plane = slot / 4;
  int component = slot - plane * 4;
  int lane = block % TILEW;
  int y = block / TILEW;
  vec4 gathered = vec4(0.0);
  for (int c = 0; c < 4; ++c) {
    int source = y * W + (cgroup * 4 + c) * TILEW + lane;
    vec4 v =
${Array.from({ length: 7 }, (_, p) =>
    `      plane == ${p} ? fetch4(plane${p}, source) :`).join("\n")}
      fetch4(plane7, source);
    float picked = pick(v, component);
    if (c == 0) gathered.x = picked;
    else if (c == 1) gathered.y = picked;
    else if (c == 2) gathered.z = picked;
    else gathered.w = picked;
  }
  result = gathered;
}`);

/**
 * The largest pair tensor, in FLOATS, for which the pair block beats the
 * one-pair-a-fragment kernel.
 *
 * 🔴 THE BLOCK HAD NO CEILING AND IS THE WRONG KERNEL ABOVE ONE, by up to 4.8x.
 * `tools/gpu/bench-webgl-contract.js` over a grid of length and channels, with
 * the two kernels' answers identical, gives the blocked kernel's time divided
 * by the one-pair kernel's - under one means the block LOSES:
 *
 * | `L * L * channels` | 296K | 590K | 592K | 1.05M | 1.18M | 2.10M | 2.36M | 4.19M |
 * |---|---|---|---|---|---|---|---|---|
 * | blocked / one-pair | 5.10 | 1.71 | 4.62 | 1.12 | 3.95 | **0.38** | **0.59** | **0.21** |
 *
 * Nine points over three lengths and three channel widths, and `L * L *
 * channels` SEPARATES THEM COMPLETELY: every one at or below 1,183,744 wins and
 * every one at or above 2,097,152 loses, with no overlap on either side. So the
 * threshold is the one quantity and not a pair of them, and it is set between
 * the two, where nothing is measured.
 *
 * 🔴 AND THE MECHANISM IS THE REPACK ROUND TRIP, WHICH IS WHY IT IS THE TENSOR
 * AND NOT THE LENGTH. The block's saving is fetches - 10.7 flops a fetch where
 * the one-pair kernel asks for the same texel four times - and its cost is that
 * it writes `L * L * channels` floats across eight planes and then reads all of
 * them back to interleave. That cost is proportional to the tensor while the
 * saving is not, so a big enough pair track spends more on the round trip than
 * the fetches were worth.
 *
 * bioemu2 at 68 tokens and 256 channels is **1,183,744** - the largest
 * measured win, and the top of the range - so this costs that fold nothing and
 * is worth 1.7x to 2.6x on the lengths above it. AF3's 128-channel pair track
 * crosses at 128 tokens and bioemu2's 256-channel one at about 86.
 */
export const CONTRACT_BLOCK_MAX_PAIR_FLOATS = 1572864;

/**
 * The eight pair planes back into `[pair][channel]`.
 *
 * Plane `(i & 1) * 4 + (j & 3)` holds the pair at
 * `((i >> 1) * JBLOCKS + (j >> 2)) * C4 + cgroup`.
 */
const contractRepackSourceUncached = (length, channels, channelSlowest = true,
  paddedLength = length, tileMapped = false) => sized({ L: length, JBLOCKS: paddedLength / 4, BLOCKS: (paddedLength / 2) * (paddedLength / 4), TILEW: Math.max(1, Math.floor(2048 / (channels / 4))), ELEMENTS: length * length * channels }, `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
// Its OUTPUT is the true L * L pairs and its SOURCE is the padded grid, which
// is the one place both lengths appear.
${Array.from({ length: 8 }, (_, p) => `uniform sampler2D plane${p};`).join("\n")}

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int cgroup = texel % C4;
  int pair = texel / C4;
  int i = pair / L;
  int j = pair - i * L;
  int block = (i >> 1) * JBLOCKS + (j >> 2);
  int source = ${tileMapped
    ? "(block / TILEW) * W + cgroup * TILEW + (block % TILEW)"
    : channelSlowest ? "cgroup * BLOCKS + block" : "block * C4 + cgroup"};
  int plane = (i & 1) * 4 + (j & 3);
${Array.from({ length: 7 }, (_, p) =>
  `  if (plane == ${p}) { result = fetch4(plane${p}, source); return; }`).join("\n")}
  result = fetch4(plane7, source);
}`);

/**
 * One contraction into `output`: the pair-blocked kernel where the shape takes
 * it, the one-pair-a-fragment kernel otherwise.
 *
 * `direction` is only a label here - the caller has already transposed the
 * incoming case into an outgoing one.
 *
 * @param {import("../runtime/execution.js").Execution} execution
 */
export function contractInto(execution, output, a, b, length, channels, divideByLength,
  direction, padded = length, prefix = "af3.triangle") {
  // The pair block needs the PADDED length divisible by four - two i and four j
  // a fragment - and eight render targets; `MAX_DRAW_BUFFERS` is at least four
  // everywhere, so the one-pair kernel stays for a context that has only four.
  // The size ceiling is on the TRUE tensor, which is what the repack moves.
  // 🔴 THE CEILING IS THE OLD 2 x 4 x 4 BLOCK'S, NOT THE WIDE ONE'S. It was fitted before `widePairBlockSource`
  // existed, and the wide block wins far past it: IntelliFold-2's 59 x 59 x 512 pair (1.78 M floats, just over) went
  // 16.7 -> 15.3 s a fold, and AF3's trunk at 150 tokens 8.4 -> 5.0 s and at 250 tokens 68.1 -> 18.7 s, bit-identical
  // (webgl/README.md). So a shape the wide block takes is never held back by it.
  const wideLegal = !execution.noContractWide && padded % 8 === 0 && Math.floor(TEXTURE_WIDTH / channels) >= 1
    && !execution.noWideUnbounded;
  const blocked = padded % 4 === 0 && execution.maxDrawBuffers >= 8
    && !execution.noContractBlock
    && (execution.noContractCeiling || wideLegal
      || length * length * channels <= CONTRACT_BLOCK_MAX_PAIR_FLOATS);
  const tag = padded === length ? "" : `:padded${padded}`;
  if (!blocked) {
    execution.dispatch({
      source: splitContractSource(length, channels, direction === "incoming" ? "incoming" : "outgoing",
        divideByLength, !execution.noTriangleCoalesce, padded),
      output,
      inputs: { aTex: a, bTex: b },
      label: `${prefix}.contract-only:${direction}:${length}x${channels}${tag}`
        + `${divideByLength ? ":over-length" : ""}`,
    });
    return output;
  }
  // 🔴 THE WIDE PAIR BLOCK FIRST, WHERE IT IS LEGAL: 4 i x 8 j x ONE channel
  // instead of 2 x 4 x 4, which is 1.33-2.01x in the bench because channels a
  // fragment cost locality and buy no intensity. See `widePairBlockSource`. It
  // needs the padded length divisible by EIGHT and `W / channels` to be at
  // least one TILEW, so a channel width past 2048 falls back.
  const wideTile = Math.floor(TEXTURE_WIDTH / channels);
  const wide = !execution.noContractWide && padded % 8 === 0 && wideTile >= 1
    && execution.maxDrawBuffers >= 8;
  if (wide) {
    const wideBlocks = (padded / 4) * (padded / 8);
    const wideTexels = Math.ceil(wideBlocks / wideTile) * TEXTURE_WIDTH;
    const widePlanes = Array.from({ length: 8 },
      (_, p) => execution.allocate(wideTexels * 4, `af3.triangle.wide-plane${p}`));
    execution.dispatch({
      source: widePairBlockSource(length, channels, divideByLength, padded, wideTile),
      outputs: widePlanes,
      inputs: { aTex: a, bTex: b },
      label: `${prefix}.contract-wide:${direction}:${length}x${channels}${tag}`
        + `${divideByLength ? ":over-length" : ""}`,
    });
    execution.dispatch({
      source: wideRepackSource(length, channels, padded, wideTile),
      output,
      inputs: Object.fromEntries(widePlanes.map((plane, p) => [`plane${p}`, plane])),
      label: `${prefix}.wide-repack:${length}x${channels}${tag}`,
    });
    for (const plane of widePlanes) execution.release(plane);
    return output;
  }
  // 🔴 THE TILE-AWARE MAPPING WHERE THE CHANNEL GROUPS DIVIDE THE TEXTURE
  // WIDTH, which is 3.29x and is what the fragment ORDER was a one-dimensional
  // approximation of. It needs `channels / 4` to divide W, which every
  // power-of-two channel width does; `TILEW` is then `W / C4` and a whole
  // rasterisation tile shares one channel's operands.
  const groups = channels / 4;
  const tileMapped = !execution.noContractTile && TEXTURE_WIDTH % groups === 0;
  const tileWidth = TEXTURE_WIDTH / groups;
  const blocks = (padded / 2) * (padded / 4);
  // The tiled grid is whole texture rows, which is 5% more fragments at 68
  // tokens and the price of the mapping.
  const planeTexels = tileMapped
    ? Math.ceil(blocks / tileWidth) * TEXTURE_WIDTH
    : blocks * groups;
  const planes = Array.from({ length: 8 },
    (_, p) => execution.allocate(planeTexels * 4, `af3.triangle.pair-plane${p}`));
  execution.dispatch({
    source: blockedContractSource(length, channels, direction, divideByLength,
      !execution.noContractChannelSlowest, padded, tileMapped),
    outputs: planes,
    inputs: { aTex: a, bTex: b },
    label: `${prefix}.contract-block:${direction}:${length}x${channels}${tag}`
      + `${tileMapped ? ":tiled" : ""}${divideByLength ? ":over-length" : ""}`,
  });
  execution.dispatch({
    source: contractRepackSource(length, channels,
      !execution.noContractChannelSlowest, padded, tileMapped),
    output,
    inputs: Object.fromEntries(planes.map((plane, p) => [`plane${p}`, plane])),
    label: `${prefix}.contract-repack:${length}x${channels}${tag}`,
  });
  for (const plane of planes) execution.release(plane);
  return output;
}

/**
 * The pair axes rounded up to a multiple of four - or EIGHT where the wide pair
 * block will run, since its 8 j a fragment needs it.
 *
 * 🔴 PADDING TO EIGHT IS 1.19x THE PADDED ARITHMETIC AT 68 TOKENS (72 cubed
 * against 68 cubed) AND THE WIDE BLOCK PAYS FOR IT TWICE OVER - 3.30 ms at 72
 * against the 2 x 4 x 4 kernel's 4.85 at 68. Asked for only when that kernel
 * can actually run, so a context without eight draw buffers or with
 * `--no-contract-wide` keeps the four-padding and the smaller tensors.
 */
export const paddedPairLength = (execution, length, channels = 0) => {
  if (execution.noContractPad) return length;
  const wideWanted = !execution.noContractWide && channels > 0
    && Math.floor(TEXTURE_WIDTH / channels) >= 1 && execution.maxDrawBuffers >= 8;
  const multiple = wideWanted ? 8 : 4;
  return length % multiple === 0 ? length : Math.ceil(length / multiple) * multiple;
};


// Every builder above, memoised on its arguments (`memoSource` in webgl/runtime/execution.js).
export const splitContractSource = memoSource(splitContractSourceUncached);
export const blockedContractSource = memoSource(blockedContractSourceUncached);
export const widePairBlockSource = memoSource(widePairBlockSourceUncached);
export const wideRepackSource = memoSource(wideRepackSourceUncached);
export const contractRepackSource = memoSource(contractRepackSourceUncached);
