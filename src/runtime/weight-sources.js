/**
 * The one symbol that means "this descriptor can hand over its CODES".
 *
 * 🔴 THREE MODULES DECLARED THEIR OWN AND THAT IS WHY A DEVICE PATH SILENTLY
 * DID NOTHING. `src/af3/pair-track-device-weights.js` looks a block's tensors
 * up by AF3's symbol; ESMFold2's trunk shares `packPairTrackWeights` with three
 * AF3 stacks and its blocks are the same shape, so wiring it to the same device
 * decode should have been one line - and it returned `undefined` for every
 * tensor, because `weights[af3.SOURCES]` on an object carrying
 * `esmfold2.SOURCES` is not a refusal, it is a miss. A symbol is unique by
 * IDENTITY and not by description, which is the whole point of one, and it is
 * exactly what makes two of them indistinguishable in a lookup.
 *
 * So there is one, here, and the three modules that used to declare it
 * re-export this. A new loader should import it rather than minting another.
 */
export const SOURCES = Symbol("localfold weight sources");

/**
 * Can this SOURCES entry be handed to `planBlockUpload`?
 *
 * 🔴 THE ENTRY IS A FUNCTION IN ONE LOADER AND AN OBJECT IN ANOTHER, and the
 * device decoder does not care: it reads `store`, `tensorName`, `first` and
 * `count` and never calls anything. AF3's loader returns a thunk with those
 * hung off it, ESMFold2's returns the source record itself with the same four
 * added, and a `typeof === "function"` test refused the second silently - which
 * is how the ESMFold2 trunk stayed on the host packer for a commit after it was
 * wired to the device one.
 */
export function bindable(entry) {
  return entry !== undefined && entry !== null
    && (typeof entry === "function" || typeof entry === "object")
    && Number.isInteger(entry.count)
    && typeof entry.store?.tensorSource === "function";
}
