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
 * @property {boolean} [multimerClustering] AF-Multimer's cluster profile (tied rows split, the centre weighed 1/L)
 *   for multimer WEIGHTS; the monomer's otherwise. webgpu/af2/multimer/model.js sets it from the regime.
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
/**
 * Where preparing an alignment spends its time, by phase.
 *
 * 🔴 THE FOLD'S OWN CLOCK CALLS ALL OF THIS "features" AND STOPS THERE. It is
 * 1.08 s of a 23.9 s fold at 825 residues and about 7% of a 59-residue one, it
 * is SERIAL with the stack rather than overlapped, and nothing said which of
 * the four loops it is. Upstream moved theirs to the GPU on the strength of
 * exactly this measurement and found the nearest-centre search was 3.2 s of
 * 3.4; ours has been word-parallel since the alignment-prep work, so the split
 * here had to be measured rather than assumed.
 */
export const featureStats = {
  calls: 0, encodeMs: 0, profileMs: 0, maskMs: 0,
  nearestMs: 0, clusterProfileMs: 0, msaFeatureMs: 0, extraMs: 0,
  // Rows dropped because the model had already been given that sequence.
  duplicateRows: 0,
};

/** Zero every phase, so a caller can measure one fold rather than a process. */
export function resetFeatureStats() {
  for (const key of Object.keys(featureStats)) featureStats[key] = 0;
}

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
/**
 * The two padded, word-aligned code blocks the comparison reads.
 *
 * 🔴 SHARED WITH THE GPU PATH ON PURPOSE. webgpu/input/nearest-centres.js
 * has to compare the SAME bytes for its answer to be the same answer, and a
 * second copy of this padding is a second chance to get 255-against-254 wrong.
 */
export function paddedCodeWords(centerCodes, encoded, extras, centreCount, length) {
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
  return {
    centreWords: new Uint32Array(centrePadded.buffer),
    extraWords: new Uint32Array(extraPadded.buffer),
    words, rows,
  };
}

/**
 * The answer of a nearest-centre search: `best[row]` is the first centre at the highest score, and with
 * `splitTies`, `ties` holds every centre at that score, `tieWords` 32-bit words a row, bit `c % 32` of word
 * `c >> 5` for centre c.
 *
 * 🔴 AF-MULTIMER SPLITS A TIE AND THE MONOMER DOES NOT. The monomer's pipeline assigns by argmax, which keeps
 * the first centre; modules_multimer.nearest_neighbor_clusters assigns by `softmax(1e3 * agreement)`, and an
 * agreement is a whole number of residues, so a row one residue nearer one centre goes to it alone and a row
 * EXACTLY as near k centres goes 1/k to each. That is no corner case: on 1AY7's alignment 17% of the rows
 * tie, and keeping the first centre for them put the cluster profile up to 0.997 from stock's
 * (cuda/af2/parity/sampling_parity.py).
 */
function nearestCentres(centerCodes, encoded, extras, centreCount, length, splitTies = false) {
  const { centreWords, extraWords, words, rows } =
    paddedCodeWords(centerCodes, encoded, extras, centreCount, length);

  const assignments = new Uint16Array(rows);
  const tieWords = splitTies ? Math.ceil(centreCount / 32) : 0;
  const ties = splitTies ? new Uint32Array(rows * tieWords) : undefined;
  const scores = splitTies ? new Int32Array(centreCount) : undefined;
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
      if (splitTies) scores[centre] = score;
    }
    assignments[index] = best;
    if (splitTies) {
      for (let centre = 0; centre < centreCount; centre += 1) {
        if (scores[centre] === bestScore) ties[index * tieWords + (centre >>> 5)] |= 1 << (centre & 31);
      }
    }
  }
  return { best: assignments, ties, tieWords };
}

/** CPU feature preprocessing for A3M text. Neural inference remains entirely on WebGPU. */
/**
 * A recycle's shuffling, its BERT masking, and everything that does not depend
 * on which centre an extra row joins.
 *
 * 🔴 PLANNED FIRST, ASSIGNED SECOND, FINISHED THIRD. Nothing in a plan depends
 * on an assignment and no recycle depends on another, so every recycle can be
 * planned, then every nearest-centre search run, then every recycle finished.
 * The host path still runs one search per plan; the device path sends all of
 * them in ONE submit, which is where the round trips go. Splitting the loop is
 * the whole reason webgpu/input/nearest-centres.js can batch.
 */
function planA3mFeatures(a3mText, tables, options) {
  featureStats.calls += 1;
  let mark = performance.now();
  const parsed = parseA3m(a3mText);
  // 🔴 EVERY ROW THE MODEL HAS ALREADY SEEN IS A ROW IT DOES NOT GET, which is
  // the argument shared/input/chains.js already makes for the multimer's paired
  // and unpaired blocks and which nothing applied here. AlphaFold's own
  // `make_msa_features` keeps a `seen_sequences` set across all MSAs and skips
  // a repeat; this is that.
  //
  // 🔴 IT MATTERS MOST FOR THE QUERY. `extractMmseqs2A3m` joins `uniref.a3m`
  // and the environmental A3M and returns each block WHOLE, so both start with
  // their own `>101` - a search that finds nothing but the query comes back as
  // depth 2, distinct 1, and every monomer fold carries the query twice.
  //
  // 🔴 AND ON THE ALIGNED COLUMNS, NOT THE RAW ROW. Featurisation has already
  // dropped the lowercase insertions into the deletion matrix, so two rows
  // differing only in their insertions are the same row to the model - which is
  // what AlphaFold hashes. Comparing the raw A3M text keeps duplicates it
  // removes; see the note on deduplicateUnpairedAgainstPaired.
  //
  // The FIRST occurrence is kept, so the query stays row 0.
  const alignment = options.deduplicateMsa === false ? parsed : (() => {
    const seen = new Set();
    const keep = [];
    for (let row = 0; row < parsed.depth; row += 1) {
      if (seen.has(parsed.sequences[row])) continue;
      seen.add(parsed.sequences[row]);
      keep.push(row);
    }
    featureStats.duplicateRows += parsed.depth - keep.length;
    if (keep.length === parsed.depth) return parsed;
    return {
      ...parsed,
      sequences: keep.map((row) => parsed.sequences[row]),
      deletionMatrix: keep.map((row) => parsed.deletionMatrix[row]),
      descriptions: keep.map((row) => parsed.descriptions[row]),
      rawSequences: keep.map((row) => parsed.rawSequences[row]),
      depth: keep.length,
    };
  })();
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
  // the deletions flat, as the codes are: what finishing reads, and what a worker is handed - nested arrays of
  // numbers were most of the bytes a structured clone copied, and a row's slice allocates
  const deletions = new Float32Array(depth * length);
  for (let row = 0; row < depth; row += 1) deletions.set(alignment.deletionMatrix[row], row * length);
  featureStats.encodeMs += performance.now() - mark; mark = performance.now();
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
  featureStats.profileMs += performance.now() - mark;
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
  const plans = [];
  for (let recycle = 0; recycle <= recycles; recycle += 1) {
    mark = performance.now();
    const random = generator(((options.randomSeed ?? 0) ^ Math.imul(recycle + 1, 0x9e3779b9)) >>> 0);
    const remainder = Array.from({ length: depth - 1 }, (_, index) => index + 1); shuffle(remainder, random);
    const centers = [0, ...remainder.slice(0, Math.max(0, maxMsa - 1))];
    const extraPool = remainder.slice(Math.max(0, maxMsa - 1)); shuffle(extraPool, random);
    const extras = extraPool.slice(0, maxExtra);
    // 🔴 THE CLUSTERS ARE BUILT OVER EVERY ROW THAT IS NOT A CENTRE, NOT OVER THE EXTRA STACK'S SHARE. Both of
    // stock AF2's pipelines assign the whole remainder before cropping it: the monomer clusters
    // (nearest_neighbor_clusters, summarize_clusters) and only then runs crop_extra_msa; the multimer's
    // sample_msa puts the entire remainder in extra_msa and create_extra_msa_feature crops it last. Clustering
    // over the kept 1024 built each centre's profile from a twelfth of a 15,000-row alignment.
    const pool = extraPool;
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
    featureStats.maskMs += performance.now() - mark; mark = performance.now();
    plans.push({ centers, extras, pool, centerCodes });
  }
  return { plans, context: { encoded, deletions, base, length, options } };
}

/**
 * The half of a recycle that follows its assignment: the cluster profile, the
 * 49-channel MSA block, and the extra rows.
 */
function finishRecycle(plan, assignments, context) {
  const { encoded, deletions, base, length, options } = context;
  const { centers, extras, pool, centerCodes } = plan;
  const multimer = options.multimerClustering === true;
  let mark = performance.now();
    const profile = new Float32Array(centers.length * length * 23);
    const deletionSums = new Float32Array(centers.length * length);
    const counts = new Float32Array(centers.length * length).fill(1 + 1e-6);
    // 🔴 TWO FORMULAS, ONE A MODEL. The monomer's summarize_clusters counts per POSITION:
    // (sum of the members' one-hots + the centre's) / (1e-6 + 1 + members). AF-Multimer's
    // nearest_neighbor_clusters multiplies each assignment by einsum('mr,nr->mn', msa_mask, extra_mask) - the
    // number of positions the centre and the row are both unmasked, which is L for every unpadded row - so its
    // profile is (L * sum of members + centre) / (L * members + 1): the centre's own row weighs about 1/L where
    // the monomer gives it a full share. That is what the multimer's weights were trained on, and what it is
    // given here; with members split over tied centres (nearestCentres) the sums are fractional.
    // The fractional sums are float32 with one rounding a step, as the C++ twin (cuda/featurise/af2_export.h)
    // keeps them, so the two stay byte-identical; the final ratio is taken in double on both sides.
    const members = multimer ? new Float32Array(centers.length) : undefined;
    if (!multimer) {
      for (let center = 0; center < centers.length; center += 1) for (let residue = 0; residue < length; residue += 1) {
        profile[(center * length + residue) * 23 + centerCodes[center * length + residue]] = 1;
        deletionSums[center * length + residue] = deletions[centers[center] * length + residue];
      }
    }
    const { best, ties, tieWords } = assignments;
    for (let poolIndex = 0; poolIndex < pool.length; poolIndex += 1) {
      const row = pool[poolIndex];
      if (!multimer) {
        const center = best[poolIndex];
        for (let residue = 0; residue < length; residue += 1) {
          const slot = center * length + residue; counts[slot] = counts[slot] + 1;
          const profileSlot = slot * 23 + encoded[row * length + residue];
          profile[profileSlot] = profile[profileSlot] + 1;
          deletionSums[slot] = deletionSums[slot] + deletions[row * length + residue];
        }
        continue;
      }
      let tied = 0;
      for (let word = 0; word < tieWords; word += 1) {
        let bits = ties[poolIndex * tieWords + word];
        while (bits !== 0) { tied += 1; bits &= bits - 1; }
      }
      const weight = Math.fround(1 / tied);
      for (let word = 0; word < tieWords; word += 1) {
        let bits = ties[poolIndex * tieWords + word];
        while (bits !== 0) {
          const bit = 31 - Math.clz32(bits & -bits); bits &= bits - 1;
          const center = word * 32 + bit;
          members[center] = members[center] + weight;
          for (let residue = 0; residue < length; residue += 1) {
            const slot = center * length + residue;
            const profileSlot = slot * 23 + encoded[row * length + residue];
            profile[profileSlot] = profile[profileSlot] + weight;
            deletionSums[slot] = deletionSums[slot] + Math.fround(weight * deletions[row * length + residue]);
          }
        }
      }
    }
    // 🔴 THE PROFILE STAYS ON FOR AN ALIGNMENT. ColabDesign2's make_msa_feats
    // defaults use_cluster_profile=False, and create_msa_feat then falls back to
    // `c_msa = batch.get("cluster_profile", msa)` - channels 25..47 become a
    // copy of the one-hot. That is a DESIGN default, for single-sequence work
    // where a one-hot is what you want; AF-Multimer's own create_msa_feat reads
    // cluster_profile, exactly as the monomer does. So this switch exists for
    // that case and is not something multimer should be run with.
    featureStats.clusterProfileMs += performance.now() - mark; mark = performance.now();
    const useClusterProfile = options.clusterProfile !== false;
    const msaFeatures = new Float32Array(centers.length * length * 49);
    for (let center = 0; center < centers.length; center += 1) for (let residue = 0; residue < length; residue += 1) {
      const slot = center * length + residue; const output = slot * 49;
      msaFeatures[output + centerCodes[slot]] = 1;
      const deletion = deletions[centers[center] * length + residue];
      msaFeatures[output + 23] = Math.min(deletion, 1); msaFeatures[output + 24] = deletionValue(deletion);
      if (useClusterProfile && multimer) {
        const denominator = length * members[center] + 1;
        for (let code = 0; code < 23; code += 1) {
          msaFeatures[output + 25 + code] = (length * profile[slot * 23 + code]
            + (code === centerCodes[slot] ? 1 : 0)) / denominator;
        }
        msaFeatures[output + 48] = deletionValue((length * deletionSums[slot] + deletion) / denominator);
      } else if (useClusterProfile) {
        for (let code = 0; code < 23; code += 1) msaFeatures[output + 25 + code] = profile[slot * 23 + code] / counts[slot];
        msaFeatures[output + 48] = deletionValue(deletionSums[slot] / counts[slot]);
      } else {
        msaFeatures[output + 25 + centerCodes[slot]] = 1;
        msaFeatures[output + 48] = deletionValue(deletion);
      }
    }
    featureStats.msaFeatureMs += performance.now() - mark; mark = performance.now();
    const extraSequences = Math.max(1, extras.length);
    const extraMsa = new Float32Array(extraSequences * length);
    const extraHasDeletion = new Float32Array(extraSequences * length);
    const extraDeletionValue = new Float32Array(extraSequences * length);
    const extraMsaMask = new Float32Array(extraSequences * length);
    for (let extraIndex = 0; extraIndex < extras.length; extraIndex += 1) for (let residue = 0; residue < length; residue += 1) {
      const slot = extraIndex * length + residue; const row = extras[extraIndex];
      const deletion = deletions[row * length + residue];
      extraMsa[slot] = encoded[row * length + residue]; extraHasDeletion[slot] = Math.min(deletion, 1);
      extraDeletionValue[slot] = deletionValue(deletion); extraMsaMask[slot] = 1;
    }
    featureStats.extraMs += performance.now() - mark;
    return {
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
    };
}

/** CPU feature preprocessing for A3M text, with the search on the host. */
export function makeA3mFeatures(a3mText, tables, options = {}) {
  const { plans, context } = planA3mFeatures(a3mText, tables, options);
  return plans.map((plan) => hostRecycle(plan, context));
}

/**
 * ONE recycle's features, exactly as `makeA3mFeatures(...)[index]`: every
 * recycle is planned (planning is cheap and seeded in sequence, so recycle k's
 * plan needs the k before it) and only `index` is searched and finished - so
 * independent workers can each take a recycle (cuda/af2/export_input.mjs).
 */
export function makeA3mFeatureRecycle(a3mText, tables, options, index) {
  const { plans, context } = planA3mFeatures(a3mText, tables, options);
  if (!(index >= 0 && index < plans.length)) throw new Error(`recycle ${index} of ${plans.length}`);
  return hostRecycle(plans[index], context);
}

/**
 * The three steps apart, for a caller that runs them in different places (cuda/af2/export_input.mjs: the
 * plan once - parsing the A3M is the largest single cost of an export, and every recycle worker used to repeat
 * it - then each recycle's search and finish in a worker, or the search on the CUDA port's device). The
 * context holds only typed arrays and plain values, so it crosses to a worker as it is.
 */
export function planA3mRecycles(a3mText, tables, options = {}) { return planA3mFeatures(a3mText, tables, options); }
export function searchA3mRecycle(plan, context) {
  return nearestCentres(plan.centerCodes, context.encoded, plan.pool, plan.centers.length, context.length,
    context.options.multimerClustering === true);
}
export function finishA3mRecycle(plan, assignments, context) { return finishRecycle(plan, assignments, context); }

function hostRecycle(plan, context) {
  const mark = performance.now();
  const assignments = nearestCentres(plan.centerCodes, context.encoded, plan.pool,
    plan.centers.length, context.length, context.options.multimerClustering === true);
  featureStats.nearestMs += performance.now() - mark;
  return finishRecycle(plan, assignments, context);
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
