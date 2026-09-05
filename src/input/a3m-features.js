import { parseA3m } from "./a3m.js";
import { makeQueryOnlyFeatures } from "./query-only-features.js";

/**
 * @typedef {object} A3mFeatureOptions
 * @property {number} [recycles]           extra passes after the first; default 0
 * @property {number} [randomSeed]         seeds the application PRNG; default 0
 * @property {number} [maxMsaSequences]    clustered rows kept; default 508
 * @property {number} [maxExtraSequences]  extra-MSA rows kept; default 1024
 * @property {readonly number[]} [chainLengths] physical-chain lengths whose sum is the query length
 * @property {number} [tolerance] recycle early-stop threshold in angstroms; consumed by the model
 *   Both defaults are explained at MAX_MSA_CLUSTERS below - the second is a
 *   deliberate reduction from AlphaFold's own model_1 value, not a copy of it.
 */

const RESTYPES = "ARNDCQEGHILKMFPSTWYV";
/**
 * The residue alphabet as a table over character codes: residue index, 20 for
 * anything unknown, 21 for a gap.
 *
 * 🔴 THIS WAS A Map AND A ONE-CHARACTER STRING PER RESIDUE.
 * `alignment.sequences[row][residue]` allocates and `Map.get` hashes it, and a
 * 200-residue query at 10,000 rows is two million of each before the
 * clustering has started.
 */
const CODE_OF_CHARACTER = (() => {
  const table = new Uint8Array(128).fill(20);
  for (let index = 0; index < RESTYPES.length; index += 1) {
    table[RESTYPES.charCodeAt(index)] = index;
  }
  table["-".charCodeAt(0)] = 21;
  return table;
})();

function generator(seed) {
  let state = seed >>> 0;
  return () => { state = (state + 0x6d2b79f5) >>> 0; let value = state;
    value = Math.imul(value ^ value >>> 15, value | 1); value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    return ((value ^ value >>> 14) >>> 0) / 4294967296; };
}

function shuffle(values, random) {
  for (let index = values.length - 1; index > 0; index -= 1) {
    const other = Math.floor(random() * (index + 1)); [values[index], values[other]] = [values[other], values[index]];
  }
}

function deletionValue(value) { return Math.atan(value / 3) * 2 / Math.PI; }

/**
 * HOW DEEP AN ALIGNMENT THE MODEL ACTUALLY SEES.
 *
 * 🔴 508, NOT 512. AlphaFold's monomer config asks for 512 MSA clusters and
 * then gives four of those rows to templates, so a templated model_1 - which is
 * what this repository runs, with mock templates - reads 508. ColabFold's
 * `--max-msa 512:1024` names the same pair of knobs.
 *
 * 🔴 1024 EXTRA ROWS IS A DELIBERATE REDUCTION, not AlphaFold's own default.
 * AlphaFold's model_1 sets `max_extra_msa: 5120`; 1024 is the value its
 * template-free models use, and the one ColabFold's 512:1024 preset selects to
 * make a run cheaper. It is kept here because the extra-MSA stack is the most
 * expensive thing in an A3M fold - profiling puts `extra.msa-row-attention.flash`
 * at about 1.2 s per block - so 5120 would cost roughly five times that stack.
 *
 * WHAT THE REDUCTION COSTS, measured: on the 8,076-row `tools/fixtures/test.a3m` the JS
 * clustering at 508/1024 reaches 96.8 pLDDT against the captured AlphaFold
 * reference's 96.625 at the same recycle. That is a shallow alignment of a
 * 59-residue protein, which is where a smaller `max_extra_msa` is least likely
 * to bite; a deep alignment of a large protein is where it would.
 */
const MAX_MSA_CLUSTERS = 508;
const MAX_EXTRA_SEQUENCES = 1024;

/**
 * Which cluster centre each extra row is nearest to, by agreeing residues.
 *
 * 🔴 THIS IS THE WHOLE COST OF PREPARING AN AF2 ALIGNMENT. It is
 * extras x centres x residues - 1024 x 508 x 59 is thirty million comparisons
 * for a 59-residue query, and it runs once per RECYCLE. Measured on
 * tools/fixtures/test.a3m's 8,076 rows, `makeA3mFeatures` was 179 ms at one
 * pass and 536 at four, and this loop was 81 ms of each 134. It grows linearly
 * in the query's length, so a 200-residue protein pays three times that,
 * before anything reaches the GPU.
 *
 * Four residues at a time, as one 32-bit compare:
 *
 * 🔴 A CENTRE CODE ABOVE 20 CAN NEVER MATCH, so it becomes 255 once, here,
 * rather than being tested per comparison. Extra codes are 0..21, so 255 is
 * unreachable and the `code <= 20` guard disappears into the data.
 *
 * 🔴 AND THE ROWS ARE PADDED TO A WHOLE NUMBER OF WORDS. A row is `length`
 * bytes and `length` is not a multiple of four, so a Uint32Array over the
 * unpadded buffer would straddle row boundaries. The padding is 255 on one
 * side and 254 on the other, which cannot agree with each other or with any
 * code.
 *
 * 🔴 AND THE ZERO-BYTE DETECT IS THE EXACT FORM, NOT THE SUBTRACTION ONE.
 * `(x - 0x01010101) & ~x & 0x80808080` is the trick everyone reaches for and
 * it is only exact for "is there a zero byte anywhere": a borrow out of a zero
 * byte marks its neighbour too, so COUNTING the marks overcounts. Measured, it
 * changed 1024 assignments' checksum from 195329 to 199057.
 * `~(((x & 0x7f7f7f7f) + 0x7f7f7f7f) | x) & 0x80808080` has no borrow between
 * bytes and gives the identical answer to the scalar loop, 3.8x faster - and
 * 6.8x against the loop it replaced.
 */
function nearestCentres(centerCodes, encoded, extras, centreCount, length) {
  const stride = Math.ceil(length / 4) * 4;
  const words = stride / 4;
  const centrePadded = new Uint8Array(centreCount * stride).fill(255);
  for (let centre = 0; centre < centreCount; centre += 1) {
    for (let residue = 0; residue < length; residue += 1) {
      const code = centerCodes[centre * length + residue];
      if (code <= 20) centrePadded[centre * stride + residue] = code;
    }
  }
  const rows = extras.length;
  const extraPadded = new Uint8Array(rows * stride).fill(254);
  for (let index = 0; index < rows; index += 1) {
    const from = extras[index] * length;
    for (let residue = 0; residue < length; residue += 1) {
      extraPadded[index * stride + residue] = encoded[from + residue];
    }
  }
  const centreWords = new Uint32Array(centrePadded.buffer);
  const extraWords = new Uint32Array(extraPadded.buffer);

  const assignments = new Uint16Array(rows);
  for (let index = 0; index < rows; index += 1) {
    const extraBase = index * words;
    let best = 0;
    let bestScore = -1;
    for (let centre = 0; centre < centreCount; centre += 1) {
      const centreBase = centre * words;
      let score = 0;
      for (let word = 0; word < words; word += 1) {
        const difference = centreWords[centreBase + word] ^ extraWords[extraBase + word];
        const zeros = ~(((difference & 0x7f7f7f7f) + 0x7f7f7f7f) | difference) & 0x80808080;
        score += ((zeros >>> 7) & 1) + ((zeros >>> 15) & 1)
          + ((zeros >>> 23) & 1) + ((zeros >>> 31) & 1);
      }
      // ...strictly greater, so a tie keeps the FIRST centre, as the scalar
      // loop did. The assignments feed the model.
      if (score > bestScore) { bestScore = score; best = centre; }
    }
    assignments[index] = best;
  }
  return assignments;
}

/** CPU feature preprocessing for A3M text. Neural inference remains entirely on WebGPU. */
export function makeA3mFeatures(a3mText, tables,
  options = {}) {
  const alignment = parseA3m(a3mText);
  const length = alignment.length; const depth = alignment.depth;
  const encoded = new Uint8Array(depth * length);
  for (let row = 0; row < depth; row += 1) {
    const sequence = alignment.sequences[row];
    const base = row * length;
    for (let residue = 0; residue < length; residue += 1) {
      const code = sequence.charCodeAt(residue);
      encoded[base + residue] = code < 128 ? CODE_OF_CHARACTER[code] : 20;
    }
  }
  const base = makeQueryOnlyFeatures(alignment.query, tables, {
    recycles: 0, chainLengths: options.chainLengths,
    chainAware: options.chainAware, chainSequences: options.chainSequences,
    legacyBreaks: options.legacyBreaks, forcePerChainNumbering: options.forcePerChainNumbering,
    maskedMsaCodes: [Float32Array.from(encoded.subarray(0, length))],
  })[0];
  // 🔴 THE PROFILE OVER THE WHOLE ALIGNMENT, which BERT masking draws from.
  // AF2 replaces a masked position from 0.1 uniform + 0.1 * msa_profile +
  // 0.1 same + 0.7 mask, and the profile is taken over EVERY sequence, before
  // any subsampling - so it is computed once here rather than per recycle.
  const msaProfile = new Float32Array(length * 23);
  for (let row = 0; row < depth; row += 1) {
    for (let residue = 0; residue < length; residue += 1) {
      msaProfile[residue * 23 + encoded[row * length + residue]] += 1;
    }
  }
  for (let residue = 0; residue < length; residue += 1) {
    for (let code = 0; code < 23; code += 1) msaProfile[residue * 23 + code] /= depth;
  }
  /** Draw a residue code from the alignment's profile at this position. */
  const sampleProfile = (residue, uniform) => {
    let cumulative = 0;
    for (let code = 0; code < 23; code += 1) {
      cumulative += msaProfile[residue * 23 + code];
      if (uniform < cumulative) return code;
    }
    return 20;
  };

  const recycles = options.recycles ?? 3;
  const maxMsa = Math.min(options.maxMsaSequences ?? MAX_MSA_CLUSTERS, depth);
  const maxExtra = options.maxExtraSequences ?? MAX_EXTRA_SEQUENCES;
  const results = [];
  for (let recycle = 0; recycle <= recycles; recycle += 1) {
    const random = generator(((options.randomSeed ?? 0) ^ Math.imul(recycle + 1, 0x9e3779b9)) >>> 0);
    const remainder = Array.from({ length: depth - 1 }, (_, index) => index + 1); shuffle(remainder, random);
    const centers = [0, ...remainder.slice(0, Math.max(0, maxMsa - 1))];
    const extraPool = remainder.slice(Math.max(0, maxMsa - 1)); shuffle(extraPool, random);
    const extras = extraPool.slice(0, maxExtra);
    const centerCodes = new Uint8Array(centers.length * length);
    for (let center = 0; center < centers.length; center += 1) {
      centerCodes.set(encoded.subarray(centers[center] * length, (centers[center] + 1) * length), center * length);
    }
    for (let index = 0; index < centerCodes.length; index += 1) {
      if (random() >= 0.15) continue;
      // 🔴 FOUR OUTCOMES, NOT THREE. This used to be 70% mask, 20% keep, 10%
      // uniform - the profile draw was missing and its share had been folded
      // into "keep", so a masked position was twice as likely to be left alone
      // as AlphaFold leaves it, and never took a residue the alignment thought
      // likely. AF2's own weights: 0.7 mask, 0.1 profile, 0.1 same, 0.1 uniform.
      const original = centerCodes[index]; const draw = random();
      if (draw < 0.7) centerCodes[index] = 22;
      else if (draw < 0.8) centerCodes[index] = sampleProfile(index % length, random());
      else if (draw < 0.9) centerCodes[index] = original;
      else centerCodes[index] = Math.floor(random() * 20);
    }
    const assignments = nearestCentres(centerCodes, encoded, extras, centers.length, length);
    const profile = new Float32Array(centers.length * length * 23);
    const deletionSums = new Float32Array(centers.length * length);
    const counts = new Float32Array(centers.length * length).fill(1 + 1e-6);
    for (let center = 0; center < centers.length; center += 1) for (let residue = 0; residue < length; residue += 1) {
      profile[(center * length + residue) * 23 + centerCodes[center * length + residue]] = 1;
      deletionSums[center * length + residue] = alignment.deletionMatrix[centers[center]] [residue];
    }
    for (let extraIndex = 0; extraIndex < extras.length; extraIndex += 1) {
      const row = extras[extraIndex]; const center = assignments[extraIndex];
      for (let residue = 0; residue < length; residue += 1) {
        const slot = center * length + residue; counts[slot] = counts[slot] + 1;
        const profileSlot = slot * 23 + encoded[row * length + residue];
        profile[profileSlot] = profile[profileSlot] + 1;
        deletionSums[slot] = deletionSums[slot] + alignment.deletionMatrix[row] [residue];
      }
    }
    // 🔴 THE PROFILE STAYS ON FOR AN ALIGNMENT. ColabDesign2's make_msa_feats
    // defaults use_cluster_profile=False, and create_msa_feat then falls back to
    // `c_msa = batch.get("cluster_profile", msa)` - channels 25..47 become a
    // copy of the one-hot. That is a DESIGN default, for single-sequence work
    // where a one-hot is what you want; AF-Multimer's own create_msa_feat reads
    // cluster_profile, exactly as the monomer does. So this switch exists for
    // that case and is not something multimer should be run with.
    const useClusterProfile = options.clusterProfile !== false;
    const msaFeatures = new Float32Array(centers.length * length * 49);
    for (let center = 0; center < centers.length; center += 1) for (let residue = 0; residue < length; residue += 1) {
      const slot = center * length + residue; const output = slot * 49;
      msaFeatures[output + centerCodes[slot]] = 1;
      const deletion = alignment.deletionMatrix[centers[center]] [residue];
      msaFeatures[output + 23] = Math.min(deletion, 1); msaFeatures[output + 24] = deletionValue(deletion);
      if (useClusterProfile) {
        for (let code = 0; code < 23; code += 1) msaFeatures[output + 25 + code] = profile[slot * 23 + code] / counts[slot];
        msaFeatures[output + 48] = deletionValue(deletionSums[slot] / counts[slot]);
      } else {
        msaFeatures[output + 25 + centerCodes[slot]] = 1;
        msaFeatures[output + 48] = deletionValue(deletion);
      }
    }
    const extraSequences = Math.max(1, extras.length);
    const extraMsa = new Float32Array(extraSequences * length);
    const extraHasDeletion = new Float32Array(extraSequences * length);
    const extraDeletionValue = new Float32Array(extraSequences * length);
    const extraMsaMask = new Float32Array(extraSequences * length);
    for (let extraIndex = 0; extraIndex < extras.length; extraIndex += 1) for (let residue = 0; residue < length; residue += 1) {
      const slot = extraIndex * length + residue; const row = extras[extraIndex];
      const deletion = alignment.deletionMatrix[row] [residue];
      extraMsa[slot] = encoded[row * length + residue]; extraHasDeletion[slot] = Math.min(deletion, 1);
      extraDeletionValue[slot] = deletionValue(deletion); extraMsaMask[slot] = 1;
    }
    results.push({
      targetFeatures: base.targetFeatures.slice(), msaFeatures, msaMask: new Float32Array(centers.length * length).fill(1),
      extraMsa, extraHasDeletion, extraDeletionValue, extraMsaMask,
      residueIndex: base.residueIndex.slice(), aatype: base.aatype.slice(), seqMask: base.seqMask.slice(),
      atom37ToAtom14: base.atom37ToAtom14.slice(), atom37Mask: base.atom37Mask.slice(),
      // ...present only for a chain-aware complex, and carried through rather
      // than rebuilt, so there is one place that decides what the chains are.
      ...(base.asymId === undefined ? {} : {
        asymId: base.asymId.slice(), entityId: base.entityId.slice(), symId: base.symId.slice(),
      }),
      msaSequences: centers.length, extraSequences, targetChannels: 22, msaFeatureChannels: 49,
    });
  }
  return results;
}

const GAP_CODE = 21;
const MSA_CHANNELS = 49;

/**
 * Write one all-gap MSA row segment, the shape a chain gets in a row that ran
 * out of homologs. Matches what the block-diagonal alignment produced for a
 * chain another chain's row did not cover: gap one-hot, gap profile, no
 * deletions. `deletionValue(0)` is 0, so only the two one-hot slots are set.
 */
function writeGapSegment(msaFeatures, row, offset, span, width) {
  for (let residue = 0; residue < span; residue += 1) {
    const output = ((row * width) + offset + residue) * MSA_CHANNELS;
    msaFeatures[output + GAP_CODE] = 1;
    msaFeatures[output + 25 + GAP_CODE] = 1;
  }
}

