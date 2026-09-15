/**
 * One weight buffer from several named tensors, laid out in a fixed order.
 *
 * 🔴 THIS EXISTS BECAUSE THE SAME TWELVE LINES WERE WRITTEN SIX TIMES AND TWO
 * OF THEM HAD THE SAME BUG. Every pack in this port is: walk a list, reserve an
 * offset, sum a length, then walk it again and copy. `packOuterProductMeanWeights`
 * reserved its offsets over `ORDER + OPTIONAL` and WROTE over `ORDER` alone, so
 * rosettafold3's projection bias was present in the generated WGSL, present in
 * the offsets, and absent from the buffer - the shader read a region of zeros,
 * added zero, and the fold came out BIT-IDENTICAL to one with no bias at all.
 * Every oracle seam matched the previous run to the last digit, which reads
 * exactly like "this convention does not matter for this model". It cost an
 * hour and it nearly became a published conclusion. `packGridAttentionWeights`
 * was one optional tensor away from the same thing.
 *
 * The fix is structural rather than careful: **there is one `packing` array, it
 * is a local variable, and both loops read it.** A caller cannot pass two
 * lists because the signature does not have two. The other four packs did not
 * have the bug only because they have no optional tensors YET, which is not a
 * property anyone should rely on.
 *
 * 🔴 AND AN ABSENT OPTIONAL TENSOR IS `null`, NOT `undefined`. The loader
 * writes null for a tensor a bundle does not carry and that null reaches the
 * SOURCES map, so `=== undefined` falls through and the caller reads `.length`
 * off it. That killed boltz2 once, in a path added for a model boltz2 does not
 * share. `!= null` covers both, and it is the test here so no caller has to
 * remember it.
 *
 * Everything variable is a hook, because the six call sites genuinely differ:
 * grid attention interleaves four projections into one `qkvgProjection` slot,
 * and the transition and diffusion blocks write f16.
 */
import { concatenateAs, writeInto } from "./float16.js";

/**
 * @param {object} weights            the named tensors
 * @param {object} options
 * @param {string} options.label      what to call this in an error
 * @param {string[]} options.order    the required tensors, in layout order
 * @param {string[]} [options.optional]  tensors packed only when the bundle has them
 * @param {(name: string) => number} [options.sizeOf]  elements a name occupies
 * @param {(target, name: string, offset: number) => void} [options.write]
 * @param {"f32"|"f16"} [options.precision]
 * @param {Set<string>} [options.composed]  layout names with no tensor of their own
 * @returns {{ data: Float32Array|Uint16Array, offsets: object, packing: string[] }}
 */
export function packNamedWeights(weights, options) {
  const { label, order, optional = [], sizeOf, write, precision = "f32",
          composed = new Set() } = options;
  // 🔴 ONE LIST. See the note above; this line is the whole point of the file.
  const packing = [...order, ...optional.filter((name) => weights[name] != null)];
  for (const name of packing) {
    // 🔴 A COMPOSED SLOT HAS NO TENSOR OF ITS OWN. Grid attention's
    // `qkvgProjection` is four projections interleaved into one region, so it
    // is a NAME in the layout and not a key in `weights` - the caller's `write`
    // hook owns it and validates its parts. Without this the presence check
    // rejects a perfectly good pack, which is how it first failed here.
    if (composed.has(name)) continue;
    if (weights[name] === undefined) throw new Error(`${label} missing ${name}`);
  }
  const offsets = {};
  let total = 0;
  for (const name of packing) {
    offsets[name] = total;
    total += sizeOf === undefined ? weights[name].length : sizeOf(name);
  }
  const data = concatenateAs(precision, total, (target) => {
    for (const name of packing) {
      if (write === undefined) writeInto(target, weights[name], offsets[name]);
      else write(target, name, offsets[name]);
    }
  });
  return { data, offsets, packing };
}
