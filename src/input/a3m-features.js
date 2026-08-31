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
const INDEX = new Map([...RESTYPES].map((residue, index) => [residue, index]));

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
 * WHAT THE REDUCTION COSTS, measured: on the 8,076-row `test.a3m` the JS
 * clustering at 508/1024 reaches 96.8 pLDDT against the captured AlphaFold
 * reference's 96.625 at the same recycle. That is a shallow alignment of a
 * 59-residue protein, which is where a smaller `max_extra_msa` is least likely
 * to bite; a deep alignment of a large protein is where it would.
 */
const MAX_MSA_CLUSTERS = 508;
const MAX_EXTRA_SEQUENCES = 1024;

/** CPU feature preprocessing for A3M text. Neural inference remains entirely on WebGPU. */
export function makeA3mFeatures(a3mText, tables,
  options = {}) {
  // ...an ARRAY IS A COMPLEX, built per chain and merged afterwards so the
  // copies do not come out identical. See makeComplexA3mFeatures.
  if (Array.isArray(a3mText)) return makeComplexA3mFeatures(a3mText, tables, options);
  const alignment = parseA3m(a3mText);
  const length = alignment.length; const depth = alignment.depth;
  const encoded = new Uint8Array(depth * length);
  for (let row = 0; row < depth; row += 1) for (let residue = 0; residue < length; residue += 1) {
    const symbol = alignment.sequences[row] [residue];
    encoded[row * length + residue] = symbol === "-" ? 21 : (INDEX.get(symbol) ?? 20);
  }
  const base = makeQueryOnlyFeatures(alignment.query, tables, {
    recycles: 0, chainLengths: options.chainLengths,
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
    const assignments = new Uint16Array(extras.length);
    for (let extraIndex = 0; extraIndex < extras.length; extraIndex += 1) {
      const extraRow = extras[extraIndex]; let best = 0; let bestScore = -1;
      for (let center = 0; center < centers.length; center += 1) {
        let score = 0;
        for (let residue = 0; residue < length; residue += 1) {
          const code = centerCodes[center * length + residue];
          if (code <= 20 && code === encoded[extraRow * length + residue]) score += 1;
        }
        if (score > bestScore) { bestScore = score; best = center; }
      }
      assignments[extraIndex] = best;
    }
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

/**
 * Features for a complex, built independently per chain and merged afterwards.
 *
 * 🔴 MERGING THE ALIGNMENTS FIRST MAKES THE COPIES IDENTICAL. Clustering,
 * subsampling and BERT masking all run over whatever alignment they are handed.
 * Give them one merged A3M whose repeated chains are paired and every copy of a
 * protein receives the same homolog in the same row, masked at the same
 * positions - the copies become substitutable, and the only thing left telling
 * them apart is the residue-index offset. The block-diagonal form got that
 * asymmetry for free, because each chain's rows were drawn and masked
 * separately; pairing the alignments threw it away.
 *
 * So the sampling runs per chain, on its own seed, and the RESULTING TENSORS
 * are concatenated along the residue axis. Every copy then draws a different
 * subset of the same homologs and is masked at different positions, while each
 * still gets the full cluster budget rather than its share of one.
 *
 * Row s of the merged MSA therefore holds unrelated homologs side by side. That
 * is deliberate and harmless HERE ONLY BECAUSE the outer product mean is
 * cov-masked across chains - see interChainCovarianceMask - so no covariance is
 * ever read between them. Without that mask this construction would invent
 * coevolution between organisms that never met.
 *
 * @param {readonly string[]} a3mTexts one A3M per physical chain
 * @param {{atom37ToAtom14: Float32Array, atom37Mask: Float32Array}} tables
 * @param {A3mFeatureOptions} [options]
 */
export function makeComplexA3mFeatures(a3mTexts, tables, options = {}) {
  if (!Array.isArray(a3mTexts) || a3mTexts.length === 0) {
    throw new RangeError("at least one chain A3M is required");
  }
  const recycles = options.recycles ?? 3;
  const seed = options.randomSeed ?? 0;
  // ...ONE SEED PER CHAIN, so identical copies diverge. Without this the whole
  // point of building per chain is lost: the same seed on the same alignment
  // reproduces the same centres and the same mask positions.
  const perChain = a3mTexts.map((text, chain) => makeA3mFeatures(text, tables, {
    ...options,
    chainLengths: undefined,
    randomSeed: (seed ^ Math.imul(chain + 1, 0x85ebca6b)) >>> 0,
  }));
  const chainLengths = a3mTexts.map((text) => parseA3m(text).length);
  const width = chainLengths.reduce((sum, length) => sum + length, 0);
  const offsets = [];
  let running = 0;
  for (const length of chainLengths) { offsets.push(running); running += length; }

  const query = a3mTexts.map((text) => parseA3m(text).query).join("");
  const base = makeQueryOnlyFeatures(query, tables, { recycles: 0, chainLengths })[0];

  const results = [];
  for (let recycle = 0; recycle <= recycles; recycle += 1) {
    const parts = perChain.map((chain) => chain[recycle]);
    const msaSequences = Math.max(...parts.map((part) => part.msaSequences));
    const extraSequences = Math.max(...parts.map((part) => part.extraSequences));

    const msaFeatures = new Float32Array(msaSequences * width * MSA_CHANNELS);
    const msaMask = new Float32Array(msaSequences * width).fill(1);
    const extraMsa = new Float32Array(extraSequences * width);
    const extraHasDeletion = new Float32Array(extraSequences * width);
    const extraDeletionValue = new Float32Array(extraSequences * width);
    const extraMsaMask = new Float32Array(extraSequences * width).fill(1);

    parts.forEach((part, chain) => {
      const span = chainLengths[chain];
      const offset = offsets[chain];
      for (let row = 0; row < msaSequences; row += 1) {
        if (row >= part.msaSequences) { writeGapSegment(msaFeatures, row, offset, span, width); continue; }
        const from = row * span * MSA_CHANNELS;
        const to = (row * width + offset) * MSA_CHANNELS;
        msaFeatures.set(part.msaFeatures.subarray(from, from + span * MSA_CHANNELS), to);
      }
      for (let row = 0; row < extraSequences; row += 1) {
        const to = row * width + offset;
        if (row >= part.extraSequences) { extraMsa.fill(GAP_CODE, to, to + span); continue; }
        const from = row * span;
        extraMsa.set(part.extraMsa.subarray(from, from + span), to);
        extraHasDeletion.set(part.extraHasDeletion.subarray(from, from + span), to);
        extraDeletionValue.set(part.extraDeletionValue.subarray(from, from + span), to);
      }
    });

    results.push({
      targetFeatures: base.targetFeatures.slice(), msaFeatures, msaMask,
      extraMsa, extraHasDeletion, extraDeletionValue, extraMsaMask,
      residueIndex: base.residueIndex.slice(), aatype: base.aatype.slice(), seqMask: base.seqMask.slice(),
      atom37ToAtom14: base.atom37ToAtom14.slice(), atom37Mask: base.atom37Mask.slice(),
      msaSequences, extraSequences, targetChannels: 22, msaFeatureChannels: 49,
    });
  }
  return results;
}
