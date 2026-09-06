// Can the distogram score the structure the sampler drew, per residue?
//
//     node tools/gpu-chrome.mjs tools/gpu/probe-esmfold2-confidence.js \
//       --crystal=/tools/fixtures/1qys-crystal.pdb
//
// 🔴 THE QUESTION IS WHETHER IT DISCRIMINATES, NOT WHETHER IT EXISTS. This
// checkpoint has no confidence head - verified at the artefact: 820 tensors and
// none named confidence, plddt, pae or pde, and `model.confidence_head` is None
// on the loaded model - so any per-residue number here is derived. The proposal
// under test: cross-entropy of the DISTOGRAM against the distances the sampler
// actually produced, `exp(-CCE)` (which is exactly the predicted probability of
// the observed bin), meaned over the best N partners at sequence separation
// above some minimum.
//
// 🔴 AND IT IS SCORED AGAINST lDDT-Ca, BECAUSE THAT IS WHAT pLDDT PREDICTS.
// Not against RMSD after superposition, which a single hinge dominates and
// which is not a per-residue quantity at all. lDDT is superposition-free and
// local, which is the whole reason AlphaFold predicts it.
//
// 🔴 AND AGAINST BASELINES, or a correlation says nothing. A score that merely
// counts a residue's neighbours would correlate with lDDT too - the core of a
// protein is both well-packed and well-predicted - so the interesting question
// is whether this beats "how buried is it". Both are reported.
//
// 🔴 FOUR DISTOGRAM-DERIVED CONFIDENCE ESTIMATES WERE REMOVED FROM THIS TREE
// ONCE ALREADY (commit 588b528), and the reason was that almost all of any fit
// was a two-number calibration that did not cross models. This one is different
// in kind - `exp(-CCE)` is a likelihood in [0, 1] and needs no affine map - but
// that is a reason to test it, not a reason to assume it.
import { GpuBufferAllocator } from "../../src/runtime/allocator.js";
import { EsmcTowerGpu } from "../../src/esmc/tower-webgpu.js";
import { readTensor, readTensorAsFloat16 } from "../../src/reference/dtype.js";
import { foldEsmfold2 } from "../../src/esmfold2/fold.js";
import { CONTACT_ANGSTROMS, CONTACT_EDGES } from "../../src/esmfold2/distogram-webgpu.js";
import {
  atomDecoderWeights, atomEncoderWeights, denoiserWeights, featuriserWeights,
  trunkBlockWeights,
} from "../../src/esmfold2/weights.js";
import { SHIM_PAIR_TENSORS } from "../../src/esmfold2/language-pair-webgpu.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};
const BLOCK_LEAVES = ["attn_norm/scale", "attn_norm/offset", "qkv/weights",
  "q_norm/scale", "k_norm/scale", "attn_out/weights", "ffn_norm/scale",
  "ffn_norm/offset", "fc1/weights", "fc2/weights"];
const NARROW = new Set(["qkv/weights", "attn_out/weights", "fc1/weights", "fc2/weights"]);
const TOWER_SHARED = ["embed/weights", "final_norm/scale", "lm/combine", "lm/norm/scale",
  "lm/norm/offset", "lm/projection/weights", "lm/downproject/weights",
  "lm/downproject/bias"];
const THREE = { ALA: "A", ARG: "R", ASN: "N", ASP: "D", CYS: "C", GLN: "Q", GLU: "E",
  GLY: "G", HIS: "H", ILE: "I", LEU: "L", LYS: "K", MET: "M", PHE: "F", PRO: "P",
  SER: "S", THR: "T", TRP: "W", TYR: "Y", VAL: "V", MSE: "M" };

function reader(bundle) {
  const shards = new Map();
  let table;
  const manifest = async () => {
    if (table === undefined) table = await (await fetch(`${bundle}/manifest.json`)).json();
    return table;
  };
  const read = async (name, half = false) => {
    const loaded = await manifest();
    const record = loaded.tensors[name];
    if (record === undefined) throw new Error(`${bundle} has no tensor ${name}`);
    if (!shards.has(record.file)) {
      shards.set(record.file, await (await fetch(`${bundle}/${record.file}`)).arrayBuffer());
    }
    return half
      ? readTensorAsFloat16(record, shards.get(record.file), record.byteOffset ?? 0)
      : readTensor(record, shards.get(record.file), record.byteOffset ?? 0, true);
  };
  return { manifest, read };
}

/**
 * A crystal chain's observed CA trace and its sequence.
 *
 * 🔴 ALTERNATE LOCATIONS ARE ONE RESIDUE, NOT TWO. 6MRR has three residues with
 * two CA records each; keeping both makes a 68-residue chain 71 residues long
 * and shifts every index after the first one. The first altloc wins, which is
 * the convention and is also the higher occupancy in these files.
 *
 * 🔴 AND A GAP IS A GAP. 1QYS has no residue 37, so its 92-residue construct is
 * 91 observed ones. What is folded is the OBSERVED sequence, so the model is
 * told 36 and 38 are adjacent - which is a real difference from the crystal,
 * confined to that junction, and worth knowing rather than hiding.
 */
function crystalChain(text, wanted) {
  const seen = new Map();
  for (const line of text.split("\n")) {
    if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) continue;
    if (line.slice(12, 16).trim() !== "CA") continue;
    const chain = line[21];
    if (wanted !== "" && chain !== wanted) continue;
    const number = Number(line.slice(22, 26));
    if (seen.has(number)) continue;
    const code = THREE[line.slice(17, 20).trim()];
    if (code === undefined) continue;
    seen.set(number, { code,
      x: Number(line.slice(30, 38)), y: Number(line.slice(38, 46)),
      z: Number(line.slice(46, 54)) });
  }
  const numbers = [...seen.keys()].sort((a, b) => a - b);
  const coordinates = new Float32Array(numbers.length * 3);
  numbers.forEach((number, index) => {
    const atom = seen.get(number);
    coordinates[index * 3] = atom.x;
    coordinates[index * 3 + 1] = atom.y;
    coordinates[index * 3 + 2] = atom.z;
  });
  return {
    sequence: numbers.map((number) => seen.get(number).code).join(""),
    coordinates,
    gaps: numbers.filter((number, index) => index > 0 && number !== numbers[index - 1] + 1).length,
  };
}

/**
 * A sequence corrupted at a rate, deterministically.
 *
 * 🔴 CORRUPTION IS A DEVICE FOR MAKING FOLDS THE MODEL IS UNSURE ABOUT, which
 * is the thing 46 real targets did not supply: 43 of them fold at lDDT-Ca above
 * 0.9, so the label the sweep was fitted against barely varies. Replacing a
 * fraction of the residues at random walks a target down the confidence scale
 * on demand.
 *
 * 🔴 AND THE LABEL DEGRADES WITH IT, WHICH HAS TO BE SAID. A mutant's true
 * structure is not the crystal, so at a high rate a low lDDT may mean "this
 * sequence really does fold differently" rather than "the model is wrong". The
 * per-residue result is therefore reported PER RATE rather than pooled, and the
 * cleanest reading is the low rates plus the GLOBAL question - does the
 * estimate know that a corrupted fold is worse - which does not depend on the
 * mutant's true structure at all.
 */
function corrupt(sequence, rate, seed) {
  if (rate <= 0) return sequence;
  const alphabet = "ACDEFGHIKLMNPQRSTVWY";
  let state = (seed >>> 0) ^ 0x9e3779b9;
  const next = () => {
    state ^= state << 13; state >>>= 0;
    state ^= state >> 17;
    state ^= state << 5; state >>>= 0;
    return state / 4294967296;
  };
  let out = "";
  for (const code of sequence) {
    out += next() < rate ? alphabet[Math.floor(next() * alphabet.length)] : code;
  }
  return out;
}

const distance = (a, i, b, j) => Math.hypot(
  a[i * 3] - b[j * 3], a[i * 3 + 1] - b[j * 3 + 1], a[i * 3 + 2] - b[j * 3 + 2]);

/**
 * Per-residue lDDT-Ca, which is the quantity pLDDT predicts.
 *
 * Superposition-free: every pair within `radius` in the REFERENCE is checked
 * for whether the model preserves its distance to within each of four
 * thresholds, and the residue's score is the mean of those four fractions.
 */
function perResidueLddt(model, reference, count, radius = 15) {
  const thresholds = [0.5, 1, 2, 4];
  const out = new Float32Array(count);
  for (let i = 0; i < count; i += 1) {
    let considered = 0, preserved = 0;
    for (let j = 0; j < count; j += 1) {
      if (i === j) continue;
      const reference_ij = distance(reference, i, reference, j);
      if (reference_ij >= radius) continue;
      considered += 1;
      const model_ij = distance(model, i, model, j);
      for (const threshold of thresholds) {
        if (Math.abs(model_ij - reference_ij) < threshold) preserved += 1;
      }
    }
    out[i] = considered === 0 ? 0 : preserved / (considered * thresholds.length);
  }
  return out;
}

const pearson = (a, b) => {
  const n = a.length;
  let sa = 0, sb = 0;
  for (let i = 0; i < n; i += 1) { sa += a[i]; sb += b[i]; }
  const ma = sa / n, mb = sb / n;
  let top = 0, va = 0, vb = 0;
  for (let i = 0; i < n; i += 1) {
    const da = a[i] - ma, db = b[i] - mb;
    top += da * db; va += da * da; vb += db * db;
  }
  return va === 0 || vb === 0 ? 0 : top / Math.sqrt(va * vb);
};
const ranks = (values) => {
  const order = [...values.keys()].sort((i, j) => values[i] - values[j]);
  const out = new Float64Array(values.length);
  order.forEach((index, rank) => { out[index] = rank; });
  return out;
};
const spearman = (a, b) => pearson(ranks(a), ranks(b));

/**
 * The proposal: `exp(-CCE)` of the observed distance under the distogram,
 * meaned over the best `top` partners at separation above `separation`.
 *
 * 🔴 `exp(-CCE)` IS THE PROBABILITY OF THE OBSERVED BIN, not a rescaled
 * distance error - which is why it needs no calibration to be read as a
 * confidence. It is bounded by how peaked a 0.39 A bin can be, so its absolute
 * value is not a probability of correctness; what it can be is a per-residue
 * ORDERING, which is what a colour needs.
 */
function distogramLikelihood(logits, bias, positions, tokens, bins,
                             { top = 10, separation = 6, edges = CONTACT_EDGES } = {}) {
  const width = (edges.maximum - edges.minimum) / bins;
  const out = new Float32Array(tokens);
  const scratch = [];
  for (let i = 0; i < tokens; i += 1) {
    scratch.length = 0;
    for (let j = 0; j < tokens; j += 1) {
      if (Math.abs(i - j) <= separation) continue;
      const d = distance(positions, i, positions, j);
      const bin = Math.max(0, Math.min(bins - 1, Math.floor((d - edges.minimum) / width)));
      const base = (i * tokens + j) * bins;
      let largest = -Infinity;
      for (let b = 0; b < bins; b += 1) {
        largest = Math.max(largest, logits[base + b] + bias[b]);
      }
      let total = 0;
      for (let b = 0; b < bins; b += 1) total += Math.exp(logits[base + b] + bias[b] - largest);
      scratch.push(Math.exp(logits[base + bin] + bias[bin] - largest) / total);
    }
    if (scratch.length === 0) { out[i] = 0; continue; }
    scratch.sort((a, b) => b - a);
    const take = Math.min(top, scratch.length);
    let sum = 0;
    for (let k = 0; k < take; k += 1) sum += scratch[k];
    out[i] = sum / take;
  }
  return out;
}

/**
 * ColabDesign's contact losses, as scores.
 *
 * 🔴 THE POINT IS THAT MOST PAIRS ARE NOT IN CONTACT AND SAYING SO IS FREE.
 * Scoring every pair's whole distribution spends nearly all of its evidence on
 * "these two residues are far apart, and they are" - which a distogram gets
 * right everywhere and which therefore separates nothing. ColabDesign's
 * `_get_con_loss` restricts to the bins below a cutoff, and `min_k` then keeps
 * only a residue's most confident partners, so the uninformative mass is
 * excluded twice over.
 *
 * The two upstream forms are different quantities and both are here:
 *
 *   binary       -log( sum of px over the contact bins )
 *                "how much does the model believe these two touch"
 *   categorical  -( px_ * log_softmax(dgram) ).sum(), where px_ is the
 *                distogram RENORMALISED inside the contact bins
 *                "...and how sharply, within the contact region"
 *
 * The categorical form simplifies: with `px_` summing to one it is
 * `logsumexp(dgram) - sum(px_ * dgram)`, so neither needs a second softmax.
 *
 * 🔴 AND `made` RANKS BY THE PREDICTION AND SCORES BY THE OUTCOME, which is the
 * only arm here that is an AGREEMENT rather than a belief. Ranking by the thing
 * being scored would select for its own answer.
 *
 * @param mode "binary" | "categorical" | "kept" | "made"
 */
function contactScore(logits, bias, positions, tokens, bins, mode,
                      { top = 10, separation = 6, cutoff = CONTACT_ANGSTROMS,
                        edges = CONTACT_EDGES } = {}) {
  const width = (edges.maximum - edges.minimum) / bins;
  let contactBins = 0;
  for (let b = 0; b < bins; b += 1) {
    if (edges.minimum + (b + 0.5) * width < cutoff) contactBins += 1;
  }
  const out = new Float32Array(tokens);
  const scratch = [];
  for (let i = 0; i < tokens; i += 1) {
    scratch.length = 0;
    for (let j = 0; j < tokens; j += 1) {
      if (Math.abs(i - j) <= separation) continue;
      const base = (i * tokens + j) * bins;
      let largest = -Infinity;
      for (let b = 0; b < bins; b += 1) {
        largest = Math.max(largest, logits[base + b] + bias[b]);
      }
      let total = 0, contact = 0;
      let contactLargest = -Infinity;
      for (let b = 0; b < bins; b += 1) {
        const weight = Math.exp(logits[base + b] + bias[b] - largest);
        total += weight;
        if (b < contactBins) {
          contact += weight;
          contactLargest = Math.max(contactLargest, logits[base + b] + bias[b]);
        }
      }
      const pContact = contact / total;
      let value = pContact;
      if (mode === "categorical") {
        // px_ over the contact bins alone, then its cross-entropy with the full
        // log-softmax. logsumexp(dgram) is `largest + log(total)`.
        let restricted = 0;
        for (let b = 0; b < contactBins; b += 1) {
          restricted += Math.exp(logits[base + b] + bias[b] - contactLargest);
        }
        let expectation = 0;
        for (let b = 0; b < contactBins; b += 1) {
          const share = Math.exp(logits[base + b] + bias[b] - contactLargest) / restricted;
          expectation += share * (logits[base + b] + bias[b]);
        }
        value = Math.exp(expectation - (largest + Math.log(total)));
      }
      if (mode === "kept" || mode === "made") {
        const made = distance(positions, i, positions, j) < cutoff ? 1 : 0;
        // ...ranked by the PREDICTION, scored by the outcome.
        scratch.push({ rank: pContact, value: mode === "made" ? made : pContact * made });
        continue;
      }
      scratch.push({ rank: value, value });
    }
    if (scratch.length === 0) { out[i] = 0; continue; }
    scratch.sort((a, b) => b.rank - a.rank);
    const take = Math.min(top, scratch.length);
    let sum = 0;
    for (let k = 0; k < take; k += 1) sum += scratch[k].value;
    out[i] = sum / take;
  }
  return out;
}

/**
 * Every per-pair quantity anyone has proposed, computed in one pass.
 *
 * 🔴 THEY MUST SHARE AN AGGREGATION OR THE COMPARISON IS RIGGED. The first pass
 * at this scored the contact arms with a top-N ranked by their own prediction
 * and the sharpness arms with a plain mean, then reported that sharpness won -
 * which is partly a statement about two aggregations. Everything here is a
 * per-pair number and the caller means all of them the same way.
 *
 * The families:
 *
 *   mode r     mass within r A of the distribution's MODE. r = 0 is the mode's
 *              own height, which is what "peakedness" meant.
 *   obs r      mass within r A of the distance the SAMPLER PRODUCED. r = 0 is
 *              exactly `exp(-CCE)`, the probability of the observed bin.
 *   negent     `exp(-H)` over the whole distribution.
 *   conBin     ColabDesign's binary contact loss as a probability: the mass
 *              below 8 A. `exp(-con_loss_bin_ent)`.
 *   conCat     its categorical form: the distribution renormalised inside the
 *              contact bins, cross-entropied against the full one.
 *
 * 🔴 AND THE RADIUS IS SWEPT IN ANGSTROMS, NOT IN BINS. The bin edges are
 * BORROWED from the disabled confidence head - 2 to 52 A over 128 bins, 0.39 A
 * each - so a mode's height is partly an artefact of a grid nobody chose. A
 * radius in angstroms is not.
 */
function pairMeasures(logits, bias, positions, tokens, bins, radii,
                      { cutoff = CONTACT_ANGSTROMS, edges = CONTACT_EDGES } = {}) {
  const width = (edges.maximum - edges.minimum) / bins;
  const spans = radii.map((radius) => Math.round(radius / width));
  let contactBins = 0;
  for (let b = 0; b < bins; b += 1) {
    if (edges.minimum + (b + 0.5) * width < cutoff) contactBins += 1;
  }
  const pairs = tokens * tokens;
  const named = new Map();
  const make = (label) => { const array = new Float32Array(pairs); named.set(label, array); return array; };
  const modeMass = radii.map((radius) => make(`mode ${radius}`));
  const observedMass = radii.map((radius) => make(`obs ${radius}`));
  const negent = make("negent");
  const conBin = make("conBin");
  const conCat = make("conCat");
  const mode = new Float32Array(pairs);
  const probability = new Float64Array(bins);

  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      const cell = i * tokens + j;
      const base = cell * bins;
      let largest = -Infinity, argmax = 0;
      for (let b = 0; b < bins; b += 1) {
        const value = logits[base + b] + bias[b];
        if (value > largest) { largest = value; argmax = b; }
      }
      let total = 0;
      for (let b = 0; b < bins; b += 1) {
        probability[b] = Math.exp(logits[base + b] + bias[b] - largest);
        total += probability[b];
      }
      const observed = Math.max(0, Math.min(bins - 1,
        Math.floor((distance(positions, i, positions, j) - edges.minimum) / width)));
      let entropy = 0, contact = 0, contactLargest = -Infinity;
      for (let r = 0; r < spans.length; r += 1) { modeMass[r][cell] = 0; observedMass[r][cell] = 0; }
      for (let b = 0; b < bins; b += 1) {
        const share = probability[b] / total;
        if (share > 0) entropy -= share * Math.log(share);
        if (b < contactBins) {
          contact += share;
          contactLargest = Math.max(contactLargest, logits[base + b] + bias[b]);
        }
        const fromMode = Math.abs(b - argmax);
        const fromObserved = Math.abs(b - observed);
        for (let r = 0; r < spans.length; r += 1) {
          if (fromMode <= spans[r]) modeMass[r][cell] += share;
          if (fromObserved <= spans[r]) observedMass[r][cell] += share;
        }
      }
      negent[cell] = Math.exp(-entropy);
      conBin[cell] = contact;
      // ...px_ renormalised inside the contact bins, cross-entropied against the
      // full log-softmax. logsumexp(dgram) is `largest + log(total)`.
      let restricted = 0;
      for (let b = 0; b < contactBins; b += 1) {
        restricted += Math.exp(logits[base + b] + bias[b] - contactLargest);
      }
      let expectation = 0;
      for (let b = 0; b < contactBins; b += 1) {
        const share = Math.exp(logits[base + b] + bias[b] - contactLargest) / restricted;
        expectation += share * (logits[base + b] + bias[b]);
      }
      conCat[cell] = Math.exp(expectation - (largest + Math.log(total)));
      mode[cell] = edges.minimum + (argmax + 0.5) * width;
    }
  }
  return { named, mode };
}

/**
 * Every (separation, cutoff) arm at once, in one pass over the pairs.
 *
 * 🔴 BOTH AXES ARE THRESHOLDS, SO A 2D CUMULATIVE TABLE ANSWERS ALL OF THEM.
 * A pair is kept when `|i - j| > separation` AND `mode <= cutoff`, so bucketing
 * each pair once by those two numbers and then taking a suffix sum over one
 * axis and a prefix sum over the other gives every combination in constant time
 * per residue. Without it a fine grid is a nested loop over the pairs per arm,
 * and the sweep costs more than the sixteen folds it is sweeping.
 *
 * 🔴 AND `top` IS FIXED AT ALL, WHICH THE COARSE SWEEP EARNED. Truncating to a
 * residue's best partners cost 0.08 of Spearman and got worse the harder it
 * truncated; keeping every pair is also what makes this table possible, since a
 * top-N needs a sort per arm.
 *
 * @returns {(separation: number, cutoff: number) => Float32Array}
 */
function cumulativeScorer(measure, mode, tokens, separations, cutoffs) {
  const S = separations.length, C = cutoffs.length;
  const sums = new Float64Array(tokens * S * C);
  const counts = new Float64Array(tokens * S * C);
  // ...bucketed by the SMALLEST separation and cutoff it satisfies, so the
  // cumulative pass below can widen it to every larger one.
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      const gap = Math.abs(i - j);
      let sBucket = -1;
      for (let s = 0; s < S; s += 1) if (gap > separations[s]) sBucket = s;
      if (sBucket < 0) continue;
      const distance = mode[i * tokens + j];
      let cBucket = -1;
      for (let c = 0; c < C; c += 1) { if (distance <= cutoffs[c]) { cBucket = c; break; } }
      if (cBucket < 0) continue;
      const at = (i * S + sBucket) * C + cBucket;
      sums[at] += measure[i * tokens + j];
      counts[at] += 1;
    }
  }
  for (let i = 0; i < tokens; i += 1) {
    // separations: a pair kept at separation s is kept at every SMALLER one.
    for (let s = S - 2; s >= 0; s -= 1) {
      for (let c = 0; c < C; c += 1) {
        const at = (i * S + s) * C + c, above = (i * S + s + 1) * C + c;
        sums[at] += sums[above]; counts[at] += counts[above];
      }
    }
    // cutoffs: a pair kept at cutoff c is kept at every LARGER one.
    for (let s = 0; s < S; s += 1) {
      for (let c = 1; c < C; c += 1) {
        const at = (i * S + s) * C + c, below = (i * S + s) * C + c - 1;
        sums[at] += sums[below]; counts[at] += counts[below];
      }
    }
  }
  return (sIndex, cIndex) => {
    const out = new Float32Array(tokens);
    for (let i = 0; i < tokens; i += 1) {
      const at = (i * S + sIndex) * C + cIndex;
      out[i] = counts[at] === 0 ? 0 : sums[at] / counts[at];
    }
    return out;
  };
}

/** The baseline that must be beaten: how many partners a residue has at all. */
function neighbourCount(positions, tokens, { separation = 6, radius = 10 } = {}) {
  const out = new Float32Array(tokens);
  for (let i = 0; i < tokens; i += 1) {
    let count = 0;
    for (let j = 0; j < tokens; j += 1) {
      if (Math.abs(i - j) <= separation) continue;
      if (distance(positions, i, positions, j) < radius) count += 1;
    }
    out[i] = count;
  }
  return out;
}

/** The 108 single-sequence targets plddt-data/ was collected over, small first. */
const DEFAULT_TARGETS = ["1r69", "1enh", "1i27", "1bk2", "1ctf", "1igd", "1poh",
  "1mjc", "1cc8", "1opd", "1tig", "1ubi", "2igd", "1lis", "1fna", "1pgx"];

export async function main(device, args = []) {
  const targets = option(args, "targets", DEFAULT_TARGETS.join(",")).split(",")
    .filter((code) => code !== "");
  const seed = Number(option(args, "seed", "0"));
  const minResidues = Number(option(args, "min-residues", "40"));
  const rates = option(args, "corrupt", "0").split(",").map(Number);
  const maxResidues = Number(option(args, "max-residues", "180"));

  const fold = reader(option(args, "bundle", "/model-esmfold2-int5"));
  const tower = reader(option(args, "esmc", "/model-esmc-600m-int3"));
  const manifest = await fold.manifest();
  const towerTable = await tower.manifest();
  const language = towerTable.languageModel;
  const M = manifest.trunk;

  const shim = {};
  for (const name of [...SHIM_PAIR_TENSORS, "lm/norm/offset", "lm/projection/weights",
                      "lm/downproject/weights", "lm/downproject/bias"]) {
    shim[name] = await tower.read(name);
  }
  const [featuriser, inputsEmbedder, denoiser, encoder, decoder] = await Promise.all([
    featuriserWeights(fold.read),
    atomEncoderWeights(fold.read, "atom", M.atomBlocks),
    denoiserWeights(fold.read, { tokenBlocks: M.tokenBlocks }),
    atomEncoderWeights(fold.read, "diffusionAtomEncoder", M.atomBlocks,
                       { withCoordinates: true }),
    atomDecoderWeights(fold.read, "diffusionAtomDecoder", M.atomBlocks),
  ]);
  denoiser.encoder = encoder;
  denoiser.decoder = decoder;
  const trunkBlocks = [];
  for (let layer = 0; layer < M.blocks; layer += 1) {
    trunkBlocks.push(await trunkBlockWeights(fold.read, layer));
  }
  const towerShared = {};
  for (const name of TOWER_SHARED) towerShared[name] = await tower.read(name);
  const allocator = new GpuBufferAllocator(device);
  const runTower = async (ids, sequenceId) =>
    (await new EsmcTowerGpu(device, allocator).run(ids, {
      rows: ids.length, model: language.width,
      heads: language.heads ?? language.width / 64,
      ffn: towerTable.tensors["blocks/0/fc2/weights"].shape[0],
      layers: language.layers, pair: M.pairChannels,
      residualScale: language.residualScale ?? 1,
    }, async (layer) => {
      const block = {};
      for (const leaf of BLOCK_LEAVES) {
        block[leaf] = await tower.read(`blocks/${layer}/${leaf}`, NARROW.has(leaf));
      }
      return block;
    }, towerShared, { sequenceId })).single;

  // 🔴 THE GRID IS SWEPT ON MANY TARGETS OR IT IS CHERRY-PICKING. On three
  // targets the coarse sweep's winner was `sep 24, top 10, no cutoff`; on
  // sixteen, two of those three axes reverse. What is reported per arm is the
  // MEDIAN across targets and its WORST target, never its best single score.
  //
  // 🔴 AND IT IS FINE, BECAUSE THE FOLD IS THE EXPENSIVE PART AND IT IS ALREADY
  // PAID FOR. Once the distogram exists the aggregation is host arithmetic over
  // a cumulative table, so a 1000-arm grid costs no more than a 96-arm one.
  const RADII = [0, 0.5, 1, 1.5, 2, 3, 4, 6];
  const SEPARATIONS = [];
  for (let value = 0; value <= 24; value += 1) SEPARATIONS.push(value);
  const CUTOFFS = [];
  for (let value = 8; value <= 52; value += 2) CUTOFFS.push(value);
  CUTOFFS.push(Infinity);
  const MEASURE_NAMES = [...RADII.map((r) => `mode ${r}`), ...RADII.map((r) => `obs ${r}`),
                         "negent", "conBin", "conCat"];
  const arms = [];
  for (const measure of MEASURE_NAMES) {
    for (let s = 0; s < SEPARATIONS.length; s += 1) {
      for (let c = 0; c < CUTOFFS.length; c += 1) {
        // ...one bucket of correlations PER corruption rate, because pooling
        // them would let the easy folds carry the hard ones.
        arms.push({ measure, s, c, pearson: [], spearman: [],
                    byRate: new Map(rates.map((rate) => [rate, []])) });
      }
    }
  }
  console.log(`  ${MEASURE_NAMES.length} measures x ${SEPARATIONS.length} separations`
    + ` x ${CUTOFFS.length} cutoffs = ${arms.length} arms,`
    + ` over ${targets.length} candidate targets\n`);

  const perTarget = [];
  for (const code of targets) {
    let text;
    try {
      const response = await fetch(`https://files.rcsb.org/download/${code.slice(0, 4)}.pdb`);
      if (!response.ok) { perTarget.push({ code, skipped: `HTTP ${response.status}` }); continue; }
      text = await response.text();
    } catch (error) {
      perTarget.push({ code, skipped: `unreachable: ${error.message}` });
      continue;
    }
    const crystal = crystalChain(text, "A");
    if (crystal.sequence.length < minResidues || crystal.sequence.length > maxResidues) {
      perTarget.push({ code, skipped: `${crystal.sequence.length} residues` });
      continue;
    }
    for (const rate of rates) {
      const sequence = corrupt(crystal.sequence, rate / 100, seed + code.charCodeAt(0));
      const result = await foldEsmfold2(device, {
        sequence, allocator, seed,
        sampler: option(args, "sampler", "diffusion-15"), distogramLogits: true,
        shape: { ...M, loops: (M.loops ?? 3) + 1 },
        weights: { featuriser, inputsEmbedder, trunkBlocks, denoiser, shim },
        tower: runTower,
      });
      const { features, coordinates } = result;
      const named = (atom, want) => {
        for (let i = 0; i < 4; i += 1) {
          const wanted = i < want.length ? want.charCodeAt(i) - 32 : 0;
          if (features.refAtomNameChars[atom * 4 + i] !== wanted) return false;
        }
        return true;
      };
      const alpha = new Int32Array(result.tokens).fill(-1);
      const representative = new Int32Array(result.tokens).fill(-1);
      for (let atom = 0; atom < result.atoms; atom += 1) {
        if (features.mask[atom] === 0) continue;
        const token = features.atomToToken[atom];
        if (named(atom, "CA")) alpha[token] = atom;
        if (named(atom, "CB")) representative[token] = atom;
      }
      for (let token = 0; token < result.tokens; token += 1) {
        if (representative[token] < 0) representative[token] = alpha[token];
      }
      const gather = (slots) => {
        const out = new Float32Array(result.tokens * 3);
        for (let token = 0; token < result.tokens; token += 1) {
          for (let axis = 0; axis < 3; axis += 1) {
            out[token * 3 + axis] = coordinates[slots[token] * 3 + axis];
          }
        }
        return out;
      };
      const modelAlpha = gather(alpha);
      const modelRepresentative = gather(representative);
      const lddt = perResidueLddt(modelAlpha, crystal.coordinates, result.tokens);
      let mean = 0;
      for (const value of lddt) mean += value;
      mean /= lddt.length;

      const measures = pairMeasures(result.distogram.logits, result.distogram.bias,
        modelRepresentative, result.tokens, M.distogramBins, RADII);
      for (const measure of MEASURE_NAMES) {
        const scorer = cumulativeScorer(measures.named.get(measure), measures.mode,
                                        result.tokens, SEPARATIONS, CUTOFFS);
        for (const arm of arms) {
          if (arm.measure !== measure) continue;
          const score = scorer(arm.s, arm.c);
          const rho = spearman(score, lddt);
          arm.pearson.push(pearson(score, lddt));
          arm.spearman.push(rho);
          arm.byRate.get(rate).push(rho);
          // 🔴 THE GLOBAL SIGNAL IS A DIFFERENT QUESTION FROM THE PER-RESIDUE
          // ONE, AND A MORE USEFUL ONE. "Which residue is least reliable" needs
          // an ordering inside a fold; "is this fold worth anything" needs one
          // ACROSS folds, and only the second survives a label that stops
          // meaning what it did. One point per fold: the mean of the estimate
          // against the mean lDDT.
          let total = 0;
          for (const value of score) total += value;
          (arm.global ??= []).push([total / score.length, mean]);
        }
      }
      const neighbours = neighbourCount(modelAlpha, result.tokens);
      const sorted = [...lddt].sort((a, b) => a - b);
      perTarget.push({ code, rate, residues: crystal.sequence.length, gaps: crystal.gaps,
        meanLddt: mean, lddt10: sorted[Math.floor(0.1 * sorted.length)],
        baselinePearson: pearson(neighbours, lddt),
        baselineSpearman: spearman(neighbours, lddt) });
      console.log(`  ${code} ${String(rate).padStart(3)}%  `
        + `${String(crystal.sequence.length).padStart(3)} res  lDDT ${mean.toFixed(3)}`
        + `  (10th ${sorted[Math.floor(0.1 * sorted.length)].toFixed(3)})`);
    }
  }

  const folded = perTarget.filter((row) => row.skipped === undefined);
  if (folded.length === 0) throw new Error("no target folded; nothing to sweep");
  const median = (values) => {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = sorted.length >> 1;
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  };
  for (const arm of arms) {
    arm.medianPearson = median(arm.pearson);
    arm.medianSpearman = median(arm.spearman);
    arm.worstSpearman = Math.min(...arm.spearman);
  }
  // ...ranked by the MEDIAN Spearman, because a colour is an ordering and a
  // median is what a single bad target cannot buy.
  const ranked = [...arms].sort((a, b) => b.medianSpearman - a.medianSpearman);
  const name = (arm) => arm.measure.padEnd(9)
    + `sep ${String(SEPARATIONS[arm.s]).padStart(2)}`
    + `  cut ${CUTOFFS[arm.c] === Infinity ? "none" : String(CUTOFFS[arm.c]).padStart(4)}`;

  console.log(`\n  ${folded.length} targets folded, `
    + `median lDDT-Ca ${median(folded.map((r) => r.meanLddt)).toFixed(3)}`);
  console.log(`  baseline (neighbour count)   median Spearman `
    + `${median(folded.map((r) => r.baselineSpearman)).toFixed(3)}\n`);
  console.log("  best ten arms, by median Spearman across targets:");
  console.log("  measure   sep    cutoff   medPearson  medSpearman   worst");
  for (const arm of ranked.slice(0, 10)) {
    console.log(`  ${name(arm)}    ${arm.medianPearson.toFixed(3).padStart(7)}`
      + `      ${arm.medianSpearman.toFixed(3).padStart(7)}  ${arm.worstSpearman.toFixed(3).padStart(7)}`);
  }
  // 🔴 AND THE ARM TO SHIP IS THE BEST WORST CASE, NOT THE BEST MEDIAN. A
  // colour that is right on fifteen targets and inverted on the sixteenth is
  // worse than one that is merely good everywhere, because the one it inverts
  // on is the one somebody is looking at when they wonder whether to trust it.
  const robust = [...arms].sort((a, b) => b.worstSpearman - a.worstSpearman);
  console.log("\n  best ten by WORST target, which is what a colour needs:");
  console.log("  measure   sep    cutoff   medPearson  medSpearman   worst");
  for (const arm of robust.slice(0, 10)) {
    console.log(`  ${name(arm)}    ${arm.medianPearson.toFixed(3).padStart(7)}`
      + `      ${arm.medianSpearman.toFixed(3).padStart(7)}  ${arm.worstSpearman.toFixed(3).padStart(7)}`);
  }

  // 🔴 THE SAME ARM, RATE BY RATE. Pooling the corruptions would let the clean
  // folds carry the corrupted ones and report a number true of neither.
  if (rates.length > 1) {
    // 🔴 THE WORST CASE PER RATE, NOT POOLED, BECAUSE THAT IS WHERE THE
    // OBJECTION LIVES. A negative worst case over all 80 folds is only a reason
    // not to colour a structure if the inversions happen on folds a reader
    // would actually look at. At 80% corruption the fold is garbage AND the
    // label is meaningless - the mutant's true structure is not the crystal -
    // so an inversion there says nothing about colouring a real prediction.
    //
    // 🔴 AND THE ARM IS CHOSEN ON THE REALISTIC RATES, for the same reason.
    // Ranking on all 80 optimises partly for folds nobody will make.
    const realistic = rates.filter((rate) => rate <= 15);
    const inBucket = (arm, chosen) => chosen.flatMap((rate) => arm.byRate.get(rate));
    for (const arm of arms) arm.realistic = inBucket(arm, realistic);
    const forColour = [...arms].sort((a, b) => {
      const worst = Math.min(...b.realistic) - Math.min(...a.realistic);
      return worst !== 0 ? worst : median(b.realistic) - median(a.realistic);
    });
    console.log("\n  the leading arm, per corruption rate:");
    console.log("  rate   folds   mean lDDT   median Spearman   worst   25th");
    for (const rate of rates) {
      const bucket = ranked[0].byRate.get(rate);
      const lddts = perTarget.filter((row) => row.rate === rate && row.skipped === undefined)
        .map((row) => row.meanLddt);
      if (bucket.length === 0) continue;
      const sorted = [...bucket].sort((a, b) => a - b);
      console.log(`  ${String(rate).padStart(3)}%   ${String(bucket.length).padStart(5)}`
        + `   ${median(lddts).toFixed(3).padStart(9)}   ${median(bucket).toFixed(3).padStart(15)}`
        + `  ${sorted[0].toFixed(3).padStart(6)}  ${sorted[Math.floor(0.25 * sorted.length)].toFixed(3).padStart(6)}`);
    }
    console.log(`\n  best ten for COLOURING - ranked on rates <= 15% by WORST fold:`);
    console.log("  measure   sep    cutoff    median   worst    25th");
    const show = (arm) => {
      const sorted = [...arm.realistic].sort((a, b) => a - b);
      console.log(`  ${name(arm)}    ${median(arm.realistic).toFixed(3).padStart(6)}`
        + `  ${sorted[0].toFixed(3).padStart(6)}  ${sorted[Math.floor(0.25 * sorted.length)].toFixed(3).padStart(6)}`);
    };
    for (const arm of forColour.slice(0, 10)) show(arm);
    // 🔴 AND THE BEST OF EACH FAMILY, BECAUSE A TOP TEN THAT IS ALL ONE FAMILY
    // does not say by how much the other lost. `mode` uses the distogram alone
    // and `obs` scores the distance the SAMPLER produced; the difference
    // between them is what "does the structure help" means, and the top ten
    // being all `mode` is only an answer if the margin is shown.
    console.log("  ...the best arm of each family, on the same ranking:");
    for (const family of ["mode", "obs", "negent", "conBin", "conCat"]) {
      const best = forColour.find((arm) => arm.measure.split(" ")[0] === family);
      if (best !== undefined) show(best);
    }
    // 🔴 AND THE GLOBAL QUESTION, WHICH THE CORRUPTIONS ARE ACTUALLY FOR. Does
    // a fold's MEAN estimate know that the fold is bad? One point per fold,
    // across every target and rate - an ordering across folds rather than
    // inside one, and the question a reader really asks of a colour.
    const globalRanked = [...arms].map((arm) => {
      const estimate = Float64Array.from(arm.global.map((row) => row[0]));
      const truth = Float64Array.from(arm.global.map((row) => row[1]));
      return { arm, pearson: pearson(estimate, truth), spearman: spearman(estimate, truth) };
    }).sort((a, b) => b.spearman - a.spearman);
    console.log(`\n  across folds - does the mean estimate know the fold is bad?`
      + `  (${globalRanked[0].arm.global.length} folds)`);
    console.log("  measure   sep    cutoff    Pearson   Spearman");
    for (const row of globalRanked.slice(0, 8)) {
      console.log(`  ${name(row.arm)}    ${row.pearson.toFixed(3).padStart(7)}`
        + `   ${row.spearman.toFixed(3).padStart(8)}`);
    }
    const chosen = globalRanked.find((row) => row.arm === ranked[0]);
    console.log(`  ...the per-residue winner, for comparison: `
      + `${chosen.pearson.toFixed(3)} / ${chosen.spearman.toFixed(3)}`);
  }

  const describe = (arm) => ({
    measure: arm.measure,
    separation: SEPARATIONS[arm.s],
    cutoff: CUTOFFS[arm.c] === Infinity ? "none" : CUTOFFS[arm.c],
    medianPearson: arm.medianPearson, medianSpearman: arm.medianSpearman,
    worstSpearman: arm.worstSpearman,
  });
  return { targets: perTarget, folded: folded.length, arms: arms.length,
           byMedian: ranked.slice(0, 10).map(describe),
           byWorst: robust.slice(0, 10).map(describe) };
}
