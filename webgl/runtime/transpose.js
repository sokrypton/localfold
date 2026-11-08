/**
 * Swapping the two leading axes of a tensor: `[outer][inner][channels]` to
 * `[inner][outer][channels]`.
 *
 * 🔴 THIS IS WHAT MSA COLUMN ATTENTION IS. AF2's row attention attends over the
 * residues of each sequence and its column attention attends over the sequences
 * at each residue - the same gated attention, over the other axis. The WGSL
 * side reads its normalised tensor as
 * `normalized[(column * p.sequences + sequence) * CHANNELS + c]`, which is the
 * transposed layout, so the transpose is a real step and not a notational one.
 * With it, column attention is `attentionBlock` with `batch` and `queryLength`
 * exchanged - and the flash core's `transpose` flag is what lets the MASK stay
 * in its original `[sequence][residue]` order while the activations move.
 *
 * One texel in, one texel out: the channel axis is untouched, so a fragment's
 * four channels come from one source texel and nothing is gathered. The cost is
 * the round trip through memory, which is why the WGSL side avoids it by
 * indexing instead - an option a fragment shader writing one texel does not
 * have.
 */
import { ONE_TARGET, PRELUDE, sized, memoSource } from "./execution.js";

/** @typedef {import("./storage.js").Tensor} Tensor */

const transposeLeadingSourceUncached = (outer, inner, channels) => sized({ OUTER: outer, INNER: inner, C4: channels / 4 }, `${PRELUDE}${ONE_TARGET}
uniform sampler2D xTex;

void main() {
  int texel = linearTexel();
  if (texel >= OUTER * INNER * C4) { result = vec4(0.0); return; }
  int c4 = texel % C4;
  int row = texel / C4;
  int o = row % OUTER;
  int i = row / OUTER;
  result = fetch4(xTex, (o * INNER + i) * C4 + c4);
}`);

/**
 * @param {import("./execution.js").Execution} execution
 * @param {{x: Tensor, outer: number, inner: number, channels: number}} call
 * @param {Tensor} [into]
 * @returns {Tensor} `[inner][outer][channels]`
 */
export function transposeLeading(execution, { x, outer, inner, channels }, into) {
  if (channels % 4 !== 0) {
    throw new RangeError(`transposeLeading: channels must be a multiple of 4, got ${channels}`);
  }
  if (x.elements !== outer * inner * channels) {
    throw new RangeError(
      `transposeLeading: x holds ${x.elements} floats, not ${outer} x ${inner} x ${channels}`);
  }
  const output = into ?? execution.allocate(x.elements, "transposed");
  if (output === x) {
    throw new Error("transposeLeading: a fragment shader cannot read the texture it draws into");
  }
  return execution.dispatch({
    source: transposeLeadingSource(outer, inner, channels),
    output,
    inputs: { xTex: x },
    label: `transpose:${outer}x${inner}x${channels}`,
  });
}

// Every builder above, memoised on its arguments (`memoSource` in webgl/runtime/execution.js).
export const transposeLeadingSource = memoSource(transposeLeadingSourceUncached);
