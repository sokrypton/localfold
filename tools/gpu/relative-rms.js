/**
 * One relative-RMS, guarded, because fifty-five tools each wrote their own.
 *
 * 🔴 THE UNGUARDED VERSION CANNOT FAIL, WHICH IS WORSE THAN NOT CHECKING. Every
 * copy of this helper in the repository is some form of
 *
 *     for (let i = 0; i < a.length; i += 1) { num += (a[i] - b[i]) ** 2; ... }
 *     return Math.sqrt(num / Math.max(den, 1e-30));
 *
 * Hand it something that is not an array - a GPU stage's `run()` resolves to
 * `{output, elapsedMilliseconds, memory}`, not a Float32Array - and `a.length`
 * is undefined, the loop never runs, `num` and `den` stay 0, and it returns
 * `Math.sqrt(0 / 1e-30)`: **exactly zero**. A perfect score, from comparing
 * nothing.
 *
 * That happened here three times in one session, in
 * `check-difftx-splits.js`, `bench-difftx-splits.js` and a `fold.js` hook. 106
 * checker arms "passed" on it, the diffusion transformer was declared
 * exonerated on their strength, and seven other causes were chased for hours
 * before the real one - a doubled bias in the norm split, which the first
 * working comparison found immediately.
 *
 * 🔴 AND THE TELL WAS THERE AND WAS ARGUED AWAY: a K split regroups its
 * additions, so a correct one reads ~1e-7 and **never 0.0 exactly**. A
 * suspiciously perfect number is a reason to check the instrument.
 *
 * 🔴 EXACT EQUALITY IS ALSO THE WRONG BAR for anything that reorders a sum.
 * Measured on the diffusion transformer: `kSplits` under 1e-5, `outKSplits`
 * 2.5e-5, `attnKSplits` 3.5e-5, `normSubgroups` 9.9e-5 - all correct, all
 * float non-associativity. The one real failure was 0.813. Pick a tolerance
 * with that gap in mind; 1e-3 separates them with room to spare.
 */

/**
 * The array behind a value, or a throw naming what it got instead.
 *
 * Accepts a bare typed array, a plain array, or anything with an `output`
 * holding one - which is the shape every GPU stage in `src/af3` returns.
 *
 * @param {unknown} value
 * @param {string} [what] names the side, so a failure says which one
 * @returns {ArrayLike<number>}
 */
export function samples(value, what = "value") {
  const out = value !== null && typeof value === "object" && "output" in value
    ? /** @type {{output: unknown}} */ (value).output : value;
  if (ArrayBuffer.isView(out) && !(out instanceof DataView)) return /** @type {any} */ (out);
  if (Array.isArray(out)) return out;
  throw new TypeError(`${what} is not an array of numbers: `
    + `${Object.prototype.toString.call(out)}`
    + (out !== value ? " (unwrapped from .output)" : ""));
}

/**
 * ||a - b|| / ||b||, over two things that must both really be arrays.
 *
 * @param {unknown} actual
 * @param {unknown} expected  the denominator, so pass the reference here
 * @returns {number}
 */
export function relativeRms(actual, expected) {
  const a = samples(actual, "actual");
  const b = samples(expected, "expected");
  if (a.length !== b.length) {
    throw new Error(`length mismatch: actual ${a.length}, expected ${b.length}`);
  }
  if (a.length === 0) throw new Error("nothing to compare: both sides are empty");
  let num = 0;
  let den = 0;
  for (let i = 0; i < a.length; i += 1) {
    num += (a[i] - b[i]) ** 2;
    den += b[i] ** 2;
  }
  return Math.sqrt(num / Math.max(den, 1e-30));
}
