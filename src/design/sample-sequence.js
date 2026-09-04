/**
 * The starting sequence and the per-cycle schedules Protein Hunter runs on.
 *
 * All of this is arithmetic over strings and small arrays: no GPU, no DOM, no
 * model. It is a module of its own so `test/sample-sequence.test.js` can hold
 * every schedule to the reference's numbers without a device, and so
 * `hunter-loop.js` reads as the loop rather than as the loop plus its
 * bookkeeping.
 *
 * Transcribed from `boltz_ph/model_utils.py:sample_seq` and the bias block at
 * the top of `boltz_ph/pipeline.py:_run_design_cycle` in
 * github.com/yehlincho/Protein-Hunter.
 */
import { ALPHABET } from "./mpnn/constants.js";

/**
 * The pool the reference draws a starting residue from.
 *
 * 🔴 `excludeP` READS BACKWARDS AND THAT IS THE REFERENCE'S BEHAVIOUR. Its
 * pool is `"ACDEFGHIKLMNQRSTVWY" + ("" if exclude_P else "P")` - nineteen
 * letters with proline ALREADY absent, and the flag ADDS it back when false.
 * So the default (`--exclude_P` not passed, i.e. false) is the pool WITH
 * proline, and passing the flag removes a letter that the string literal never
 * had. Transcribed rather than corrected: this is what produced the paper's
 * starting sequences, and a "fix" here would silently be a different method.
 */
const POOL_WITHOUT_PROLINE = "ACDEFGHIKLMNQRSTVWY";

/**
 * A starting sequence: `percentX` of the positions unknown, the rest drawn
 * uniformly, then shuffled.
 *
 * @param {number} length
 * @param {{percentX?: number, excludeP?: boolean,
 *          random?: () => number}} [options] `random` returns [0, 1); pass
 *   `uniformFrom(seed)` from src/af3/fold.js to make a run reproducible.
 * @returns {string}
 */
export function sampleSequence(length, options = {}) {
  const random = options.random ?? Math.random;
  const percentX = options.percentX ?? 90;
  const pool = options.excludeP === true
    ? POOL_WITHOUT_PROLINE : `${POOL_WITHOUT_PROLINE}P`;
  // ...`round`, not `floor`: the reference's `round(length * frac_X)`, so 90%
  // of 15 is 14 rather than 13.
  const unknown = Math.round(length * (percentX / 100));
  const letters = [];
  for (let index = 0; index < unknown; index += 1) letters.push("X");
  for (let index = unknown; index < length; index += 1) {
    letters.push(pool[Math.floor(random() * pool.length)]);
  }
  // Fisher-Yates, so the X positions are not a prefix. `random.shuffle` there.
  for (let index = letters.length - 1; index > 0; index -= 1) {
    const other = Math.floor(random() * (index + 1));
    const held = letters[index];
    letters[index] = letters[other];
    letters[other] = held;
  }
  return letters.join("");
}

/**
 * The alanine bias for one cycle, ramped linearly across the run.
 *
 * The reference's `alpha = start - (cycle / (cycles - 1)) * (start - end)`,
 * with both defaults negative (-0.5 -> -0.1): early cycles discourage alanine
 * hard and later ones let it back, because MPNN's answer to a backbone it does
 * not believe in is a poly-alanine helix and the run has to be pushed past
 * that before it can be trusted to choose.
 *
 * @param {number} cycle zero-based, over `cycles` design cycles
 * @param {number} cycles
 * @param {{start?: number, end?: number}} [options]
 * @returns {number}
 */
export function alanineBias(cycle, cycles, options = {}) {
  const start = options.start ?? -0.5;
  const end = options.end ?? -0.1;
  // One cycle has no ramp to be at a point on, and `cycles - 1` would divide
  // by zero. The reference takes the start value there.
  const fraction = cycles > 1 ? cycle / (cycles - 1) : 0;
  return start - fraction * (start - end);
}

/**
 * The letters no cycle may choose.
 *
 * 🔴 PROLINE IS OMITTED ON THE FIRST DESIGN CYCLE ONLY. Cycle 0's structure
 * came from a mostly-`X` sequence, so its backbone is the model's guess at a
 * fold rather than a fold - and proline placed against a guessed backbone
 * breaks the helix the next cycle would otherwise have built on. After one
 * cycle there is a real backbone to read and proline goes back in the pool.
 *
 * @param {number} cycle zero-based
 * @param {string} [omit] the standing omissions; the reference's `"C"`
 * @returns {string}
 */
export function omittedLetters(cycle, omit = "C") {
  const letters = new Set(omit.toUpperCase().replace(/[^A-Z]/g, ""));
  if (cycle === 0) letters.add("P");
  return [...letters].join("");
}

/** How far below any real logit an omitted letter is pushed. */
const OMIT = -1e8;

/**
 * How many letters of an MPNN alphabet are protein: the twenty amino acids and
 * X. Both alphabets put them first - in different ORDERS, see designBias - so
 * this one boundary separates protein from nucleic under either.
 */
const PROTEIN_LETTERS = 21;

/**
 * The `[L, V]` bias `Model.sample` takes.
 *
 * 🔴 OMISSION AND BIAS SHARE ONE TENSOR, which is the reference's
 * `-1e8 * omit_AA + bias_AA` and not two mechanisms. `Model.sample` falls back
 * to the model's own default omissions ONLY when no bias is supplied, so a
 * caller that passes a bias owns every omission - including `X`, which is a
 * legal AF3 input letter but never something MPNN should be allowed to design.
 *
 * 🔴 THE ALPHABET IS AN ARGUMENT BECAUSE NA-MPNN'S IS 33 LETTERS, NOT 21, AND
 * A BIAS OF THE WRONG WIDTH IS READ AS THE RIGHT ONE. `Model.sample` indexes
 * it as `bias[position * V + letter]` with V from the CHECKPOINT, so a
 * 21-wide row handed to a 33-letter model is not a length error - it is every
 * position reading twelve letters into the next position's row, silently, and
 * every omission landing on a letter nobody named. Measured on the Top7
 * fixture: the correct bias designs `SKKITVTIKSKDKTKTITYEVESEKELENVKKELKLL`
 * and a 21-wide one designs `SARVTVTITEADTTRTLTAEVESAAAAAAAAAAAAAA` - a
 * plausible protein with a poly-alanine tail, which no assertion about "is it
 * made of amino acids" can tell from a real answer.
 *
 * 🔴 AND THE TWO ALPHABETS ARE NOT THE SAME 21 LETTERS IN THE SAME ORDER.
 * `ALPHABET` is `ACDEFGHIKLMNPQRSTVWYX` and `NA_ALPHABET` starts
 * `ARNDCQEGHILKMFPSTWYVX` - the same set, AlphaFold's ordering against
 * alphabetical, so token 11 is `N` in one and `K` in the other. Every letter
 * below is therefore looked up in the alphabet actually in use rather than
 * assumed; the same trap cost ../mpnn a bug where switching family turned
 * "omit cysteine" into "omit arginine".
 *
 * What the extra width needs is the nucleic tokens omitted, which is what
 * `proteinOnly` does - without it NA-MPNN could answer a protein position
 * with a nucleotide. Both alphabets put their 21 protein letters first, so
 * one boundary serves.
 *
 * @param {number} length residues in the structure, not in the designed chain
 * @param {{omit?: string, alanineBias?: number, alphabet?: string,
 *          proteinOnly?: boolean}} [options]
 * @returns {Float32Array} length * alphabet.length
 */
export function designBias(length, options = {}) {
  const alphabet = options.alphabet ?? ALPHABET;
  const bias = new Float32Array(length * alphabet.length);
  const row = new Float32Array(alphabet.length);
  // X is position 20 and is never a design choice: the loop puts X into the
  // sequence AF3 folds, and reads letters back out of MPNN.
  row[alphabet.indexOf("X")] = OMIT;
  // ...and under a wider alphabet, everything past the twenty amino acids and
  // X. `proteinOnly` defaults on because a designed chain here is a protein.
  for (let index = PROTEIN_LETTERS; options.proteinOnly !== false
    && index < alphabet.length; index += 1) {
    row[index] = OMIT;
  }
  for (const letter of options.omit ?? "") {
    const index = alphabet.indexOf(letter);
    if (index >= 0) row[index] = OMIT;
  }
  if (options.alanineBias !== undefined && options.alanineBias !== 0) {
    const alanine = alphabet.indexOf("A");
    // ...added, not assigned: "omit alanine" and "discourage alanine" have to
    // compose, and an omitted A stays omitted whatever the ramp says.
    row[alanine] += options.alanineBias;
  }
  for (let position = 0; position < length; position += 1) {
    bias.set(row, position * alphabet.length);
  }
  return bias;
}

/**
 * The share of a sequence that is alanine.
 *
 * The loop's guard: MPNN's failure mode on a backbone it cannot read is
 * poly-alanine, which scores well and means nothing, so a cycle over 20%
 * alanine cannot become the run's best however good its ipTM is.
 *
 * @param {string} sequence
 * @returns {number} 0 for the empty sequence
 */
export function alanineFraction(sequence) {
  if (sequence.length === 0) return 0;
  let count = 0;
  for (const letter of sequence) if (letter === "A") count += 1;
  return count / sequence.length;
}
