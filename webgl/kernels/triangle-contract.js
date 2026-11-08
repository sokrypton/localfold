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
import { ONE_TARGET, PRELUDE, sized } from "../runtime/execution.js";
import { CHANNELS_PER_TEXEL } from "../runtime/context.js";
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
