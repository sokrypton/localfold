/**
 * runspec.js
 * ----------
 * A RunSpec — the design sampler's knobs — translated into what
 * `cpu/design/mpnn/`'s `Model.sample` already takes.
 *
 * 🔴 THE FIELD NAMES AND SEMANTICS ARE `maraxen/aminx`'s, DELIBERATELY. They are
 * specified in docs/DESIGN-RUNSPEC.md, read from that repository at commit
 * `0729c458` (MIT, Copyright (c) 2025 Marielle Russo, used with her permission,
 * 2026-10-24). Adopting her names rather than inventing our own means a spec, a
 * script and a page mean the same thing by `fixed_positions`, and that a design
 * written against either implementation is legible to the other. No code is
 * copied here; this is her interface over our engine.
 *
 * 🔴 AND THE ENGINE ALREADY HAD EVERY PRIMITIVE, which is why this file is a
 * translation and not a feature. `Model.sample` takes `temperature`, `rng`,
 * `S`, a PER-POSITION `chainMask`, a `[L, V]` `bias`, `symmetry` (tie groups,
 * with per-position weights it does not even need) and `order`. So:
 *
 *   seed                 -> rng: uniformFrom(seed)
 *   temperature          -> temperature
 *   bias_AA, bias_AA_per_residue, omit_AA, omit_AA_per_residue  -> bias
 *   chains_to_design     -> chainMask
 *   fixed_positions      -> chainMask[pos] = 0, and S[pos] = that letter
 *   tied_positions       -> symmetry
 *   decoding_order       -> order
 *
 * WHERE THIS DELIBERATELY DIVERGES FROM HERS, and why. Each of these is a case
 * her implementation leaves silent (docs/DESIGN-RUNSPEC.md section 7.3) and
 * this repository refuses loudly instead, because a design that quietly ignored
 * a constraint is worse than one that did not run:
 *
 *   - a position outside `0..L-1` THROWS. Hers writes past a Float32Array,
 *     which is a silent no-op, so a 1-based index - the obvious mistake, since
 *     PDB numbering is 1-based and these are 0-based - would drop the last
 *     constraint and keep the rest.
 *   - a non-finite or non-positive `temperature` THROWS. Hers passes it into
 *     the graph unchecked; `null` coerces to 0 there.
 *   - an explicit `decoding_order` must be a permutation OF `0..L-1`. Hers
 *     checks the length and the distinctness but not the range.
 *
 * 🔴 SEEDS ARE NOT PORTABLE TO HER SAMPLER, and the reason is worth knowing:
 * she pads the structure to a length bucket (128 or 256) BEFORE building the
 * inputs, so her PRNG is consumed over the padded length and the same structure
 * at the same seed samples differently depending on which bucket it lands in.
 * We do not pad. Equal field names do not mean equal draws, and nothing here
 * claims they do.
 */
import { uniformFrom } from "../../shared/af3/random.js";
import { ALPHABET } from "./mpnn/constants.js";
import { designBias } from "./sample-sequence.js";

/** How far below any real logit an omitted letter is pushed. Hers is the same value. */
export const OMIT_BIAS = -1e8;

const letterIndex = (letter, alphabet) => {
  if (typeof letter !== "string" || letter.length !== 1) {
    throw new Error(`a bias or fixed position wants ONE letter, got ${JSON.stringify(letter)}`);
  }
  const index = alphabet.indexOf(letter);
  if (index < 0) throw new Error(`letter ${letter} is not in alphabet ${alphabet}`);
  return index;
};

/**
 * `{ 0: "M", "4": "G" }` as pairs, with the key read as a 0-BASED position.
 *
 * 🔴 0-BASED, AND THIS IS THE FIELD MOST LIKELY TO BE PASSED 1-BASED. PDB
 * numbering is 1-based and `residue_index` in a structure usually is too, so a
 * caller reaching for "residue 54" means something different from what this
 * reads. It is an index into the structure's arrays. Out of range throws rather
 * than being dropped.
 */
const positionEntries = (value, length, field) => {
  if (value === undefined || value === null) return [];
  return Object.entries(value).map(([key, inner]) => {
    const position = Number(key);
    if (!Number.isInteger(position) || position < 0 || position >= length) {
      throw new Error(`${field} names position ${key}, outside 0..${length - 1}`
        + " (these are 0-based indices into the structure, not PDB residue numbers)");
    }
    return [position, inner];
  });
};

/**
 * Translate a RunSpec into `Model.sample` options.
 *
 * @param {object} runspec the fields above; every one optional
 * @param {object} context what the structure says
 * @param {number} context.length L, the number of model positions
 * @param {string[]} [context.chainIds] the chain of each position, needed by
 *   `chains_to_design`
 * @param {ArrayLike<number>} [context.nativeTokens] the sequence the structure
 *   already has, which fixed positions are conditioned on
 * @param {string} [context.alphabet] defaults to the 21-letter protein one
 * @param {boolean} [context.proteinOnly] see `designBias`
 * @param {number} [context.alanineBias] see `designBias`
 * @returns {{temperature: number, rng: () => number, S: Int32Array,
 *            chainMask: Float32Array, bias: Float32Array,
 *            symmetry: number[][], order: (Int32Array|undefined)}}
 */
export function sampleOptionsFrom(runspec = {}, context = {}) {
  const spec = runspec ?? {};
  const length = context.length;
  if (!Number.isInteger(length) || length <= 0) {
    throw new Error(`a RunSpec needs the structure's length, got ${JSON.stringify(length)}`);
  }
  const alphabet = context.alphabet ?? ALPHABET;
  const width = alphabet.length;

  // 🔴 THE BASE ROW IS THIS REPOSITORY'S, NOT HERS. `designBias` already omits
  // X at every position - the loop puts X into what is folded and must never
  // read one back - and omits everything past the protein letters under a wider
  // alphabet. Those are invariants of our loop, not knobs, so they are applied
  // first and the RunSpec's own bias fields compose on top.
  const bias = designBias(length, {
    alphabet,
    proteinOnly: context.proteinOnly,
    alanineBias: context.alanineBias,
  });

  // The four bias fields, in HER order, because the later ones OVERWRITE rather
  // than accumulate and the order is therefore observable:
  //   bias_AA -> bias_AA_per_residue -> omit_AA -> omit_AA_per_residue
  for (const [letter, value] of Object.entries(spec.bias_AA ?? {})) {
    const column = letterIndex(letter, alphabet);
    const delta = Number(value);
    if (!Number.isFinite(delta)) throw new Error(`bias_AA.${letter} is ${value}, not a finite number`);
    for (let position = 0; position < length; position += 1) bias[position * width + column] += delta;
  }
  for (const [position, letters] of positionEntries(spec.bias_AA_per_residue, length, "bias_AA_per_residue")) {
    for (const [letter, value] of Object.entries(letters ?? {})) {
      const delta = Number(value);
      if (!Number.isFinite(delta)) {
        throw new Error(`bias_AA_per_residue[${position}].${letter} is ${value}, not a finite number`);
      }
      bias[position * width + letterIndex(letter, alphabet)] += delta;
    }
  }
  // 🔴 OMIT OVERWRITES, IT DOES NOT ADD - hers too. "Discourage C by 2" and
  // "never choose C" have to compose as the second one winning, or an omitted
  // letter could be biased back into reach.
  for (const letter of spec.omit_AA ?? "") {
    const column = letterIndex(letter, alphabet);
    for (let position = 0; position < length; position += 1) bias[position * width + column] = OMIT_BIAS;
  }
  for (const [position, letters] of positionEntries(spec.omit_AA_per_residue, length, "omit_AA_per_residue")) {
    for (const letter of letters ?? "") {
      bias[position * width + letterIndex(letter, alphabet)] = OMIT_BIAS;
    }
  }

  // Which positions the designer may write. 1 = designed, which is our engine's
  // sense; hers carries the complement (`fixedMask`) and the two are inverses.
  const chainMask = new Float32Array(length).fill(1);
  const native = context.nativeTokens;
  const S = new Int32Array(length);
  if (native !== undefined) {
    if (native.length !== length) {
      throw new Error(`nativeTokens is ${native.length} long, the structure is ${length}`);
    }
    for (let position = 0; position < length; position += 1) S[position] = native[position];
  }

  if (spec.chains_to_design !== undefined && spec.chains_to_design !== null) {
    const chainIds = context.chainIds;
    if (chainIds === undefined || chainIds.length !== length) {
      throw new Error("chains_to_design needs context.chainIds, one per position");
    }
    const design = new Set(spec.chains_to_design);
    for (const wanted of design) {
      if (!chainIds.includes(wanted)) {
        throw new Error(`chains_to_design names chain ${wanted},`
          + ` which the structure does not have (it has ${[...new Set(chainIds)].join(", ")})`);
      }
    }
    for (let position = 0; position < length; position += 1) {
      if (!design.has(chainIds[position])) chainMask[position] = 0;
    }
  }

  // ...and then individual positions, which override the chain rule: a fixed
  // position inside a designed chain is held, and its letter is what the
  // designer conditions its neighbours on.
  for (const [position, letter] of positionEntries(spec.fixed_positions, length, "fixed_positions")) {
    chainMask[position] = 0;
    S[position] = letterIndex(letter, alphabet);
  }

  // Tie groups, as `normaliseSymmetry` wants them: arrays of positions. It
  // takes the first occurrence of a position and ignores later ones, so a
  // position in two groups is not an error there - it is here, because it means
  // the caller believes something the sampler will not do.
  const symmetry = [];
  const tied = new Map();
  for (const group of spec.tied_positions ?? []) {
    const members = Array.from(group ?? []);
    if (members.length === 0) continue;
    for (const member of members) {
      if (!Number.isInteger(member) || member < 0 || member >= length) {
        throw new Error(`tied_positions names ${member}, outside 0..${length - 1}`);
      }
      if (tied.has(member)) {
        throw new Error(`tied_positions puts position ${member} in two groups`
          + ` (${JSON.stringify(tied.get(member))} and ${JSON.stringify(members)});`
          + " the sampler would silently keep only the first");
      }
      tied.set(member, members);
    }
    symmetry.push(members);
  }

  let order;
  const requested = spec.decoding_order ?? "random";
  if (requested !== "random") {
    order = Int32Array.from(requested);
    if (order.length !== length) {
      throw new Error(`decoding_order is ${order.length} long, the structure is ${length}`);
    }
    const seen = new Set(order);
    if (seen.size !== length) throw new Error("decoding_order repeats a position");
    for (const position of order) {
      if (position < 0 || position >= length) {
        throw new Error(`decoding_order names ${position}, outside 0..${length - 1}`);
      }
    }
  }

  const temperature = spec.temperature === undefined ? 0.1 : Number(spec.temperature);
  if (!Number.isFinite(temperature) || temperature <= 0) {
    throw new Error(`temperature is ${JSON.stringify(spec.temperature)};`
      + " it has to be a finite number above zero");
  }

  return {
    temperature,
    rng: uniformFrom(spec.seed ?? 0),
    S,
    chainMask,
    bias,
    symmetry,
    order,
  };
}
