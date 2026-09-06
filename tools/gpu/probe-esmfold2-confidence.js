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
 * Every per-pair way of asking "how sure is the distogram about this distance",
 * computed once so the hyperparameter sweep is pure aggregation.
 *
 * 🔴 IT IS THE WHOLE DISTRIBUTION, NOT THE CONTACT BINS. Restricting to contact
 * was measured and collapses into a buriedness baseline - see CLAUDE.md. A
 * distance confidently predicted to be LARGE is evidence of confidence too.
 *
 * 🔴 AND FOUR MEASURES, BECAUSE `max` IS BIN-WIDTH SENSITIVE. The bin edges
 * here are BORROWED from the disabled confidence head (2 to 52 A over 128
 * bins, 0.39 A each), so the height of a mode is partly an artefact of how fine
 * that grid is. `w1` and `w2` ask instead how much mass lies within 1 A or 2 A
 * of the mode, which is what "how precisely does it know the distance" means
 * and is robust to a grid nobody chose. `negent` is `exp(-H)`, the whole
 * distribution's sharpness rather than its peak's.
 */
function pairMeasures(logits, bias, tokens, bins, edges = CONTACT_EDGES) {
  const width = (edges.maximum - edges.minimum) / bins;
  const near1 = Math.max(1, Math.round(1 / width));
  const near2 = Math.max(1, Math.round(2 / width));
  const pairs = tokens * tokens;
  const out = {
    max: new Float32Array(pairs), w1: new Float32Array(pairs),
    w2: new Float32Array(pairs), negent: new Float32Array(pairs),
    mode: new Float32Array(pairs),
  };
  const probability = new Float64Array(bins);
  for (let cell = 0; cell < pairs; cell += 1) {
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
    let entropy = 0, within1 = 0, within2 = 0;
    for (let b = 0; b < bins; b += 1) {
      const share = probability[b] / total;
      if (share > 0) entropy -= share * Math.log(share);
      if (Math.abs(b - argmax) <= near1) within1 += share;
      if (Math.abs(b - argmax) <= near2) within2 += share;
    }
    out.max[cell] = 1 / total;
    out.w1[cell] = within1;
    out.w2[cell] = within2;
    out.negent[cell] = Math.exp(-entropy);
    out.mode[cell] = edges.minimum + (argmax + 0.5) * width;
  }
  return out;
}

/**
 * One arm: aggregate a per-pair measure into a per-residue score.
 *
 * 🔴 THE PAIR CUTOFF IS ON THE PREDICTED DISTANCE, NOT ON THE BINS. It asks
 * whether a confidently-predicted 45 A separation is informative about a
 * residue's reliability or merely easy - which is the honest version of the
 * "we do not care about far pairs" instinct, and the one that keeps the whole
 * distribution's sharpness rather than throwing four fifths of it away.
 */
function aggregate(measure, mode, tokens, { top, separation, cutoff }) {
  const out = new Float32Array(tokens);
  const scratch = [];
  for (let i = 0; i < tokens; i += 1) {
    scratch.length = 0;
    for (let j = 0; j < tokens; j += 1) {
      if (Math.abs(i - j) <= separation) continue;
      if (Number.isFinite(cutoff) && mode[i * tokens + j] > cutoff) continue;
      scratch.push(measure[i * tokens + j]);
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

  // 🔴 THE GRID IS SWEPT ON MANY TARGETS OR IT IS CHERRY-PICKING. Ninety-six
  // arms against two proteins finds an arm that suits two proteins. What is
  // reported per arm is the MEDIAN across targets and how many targets it is
  // best on, not its best single score.
  const MEASURES = ["max", "w1", "w2", "negent"];
  const SEPARATIONS = [0, 6, 12, 24];
  const TOPS = [1, 5, 10, 20, 40, Infinity];
  const CUTOFFS = [Infinity, 20, 12];
  const arms = [];
  for (const measure of MEASURES) {
    for (const separation of SEPARATIONS) {
      for (const top of TOPS) {
        for (const cutoff of CUTOFFS) arms.push({ measure, separation, top, cutoff });
      }
    }
  }
  for (const arm of arms) { arm.pearson = []; arm.spearman = []; }

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
    const result = await foldEsmfold2(device, {
      sequence: crystal.sequence, allocator, seed,
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
    for (let atom = 0; atom < result.atoms; atom += 1) {
      if (features.mask[atom] !== 0 && named(atom, "CA")) alpha[features.atomToToken[atom]] = atom;
    }
    const modelAlpha = new Float32Array(result.tokens * 3);
    for (let token = 0; token < result.tokens; token += 1) {
      for (let axis = 0; axis < 3; axis += 1) {
        modelAlpha[token * 3 + axis] = coordinates[alpha[token] * 3 + axis];
      }
    }
    const lddt = perResidueLddt(modelAlpha, crystal.coordinates, result.tokens);
    let mean = 0;
    for (const value of lddt) mean += value;
    mean /= lddt.length;
    const sorted = [...lddt].sort((a, b) => a - b);
    const measures = pairMeasures(result.distogram.logits, result.distogram.bias,
                                  result.tokens, M.distogramBins);
    for (const arm of arms) {
      const score = aggregate(measures[arm.measure], measures.mode, result.tokens, arm);
      arm.pearson.push(pearson(score, lddt));
      arm.spearman.push(spearman(score, lddt));
    }
    const neighbours = neighbourCount(modelAlpha, result.tokens);
    perTarget.push({ code, residues: crystal.sequence.length, gaps: crystal.gaps,
      meanLddt: mean, lddt10: sorted[Math.floor(0.1 * sorted.length)],
      baselinePearson: pearson(neighbours, lddt),
      baselineSpearman: spearman(neighbours, lddt) });
    console.log(`  ${code}  ${String(crystal.sequence.length).padStart(3)} res`
      + `  lDDT ${mean.toFixed(3)}  (10th ${sorted[Math.floor(0.1 * sorted.length)].toFixed(3)})`);
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
  const label = (arm) => `${arm.measure.padEnd(6)} sep ${String(arm.separation).padStart(2)}`
    + `  top ${String(arm.top === Infinity ? "all" : arm.top).padStart(3)}`
    + `  cut ${arm.cutoff === Infinity ? "none" : String(arm.cutoff).padStart(4)}`;

  console.log(`\n  ${folded.length} targets folded, `
    + `median lDDT-Ca ${median(folded.map((r) => r.meanLddt)).toFixed(3)}`);
  console.log(`  baseline (neighbour count)   median Spearman `
    + `${median(folded.map((r) => r.baselineSpearman)).toFixed(3)}\n`);
  console.log("  best twelve arms, by median Spearman across targets:");
  console.log("  measure  sep   topN   cutoff   medPearson  medSpearman  worst");
  for (const arm of ranked.slice(0, 12)) {
    console.log(`  ${label(arm)}    ${arm.medianPearson.toFixed(3).padStart(7)}`
      + `      ${arm.medianSpearman.toFixed(3).padStart(7)}   ${arm.worstSpearman.toFixed(3).padStart(6)}`);
  }
  console.log("\n  each axis at its best, holding the rest at the winner:");
  const best = ranked[0];
  for (const [axis, values] of [["measure", MEASURES], ["separation", SEPARATIONS],
                                ["top", TOPS], ["cutoff", CUTOFFS]]) {
    const line = values.map((value) => {
      const found = arms.find((arm) => MEASURES.concat().length
        && arm.measure === (axis === "measure" ? value : best.measure)
        && arm.separation === (axis === "separation" ? value : best.separation)
        && arm.top === (axis === "top" ? value : best.top)
        && arm.cutoff === (axis === "cutoff" ? value : best.cutoff));
      const shown = value === Infinity ? "all" : value;
      return `${shown}:${found.medianSpearman.toFixed(3)}`;
    }).join("  ");
    console.log(`  ${axis.padEnd(11)} ${line}`);
  }

  return { targets: perTarget, folded: folded.length,
           best: { ...best, pearson: undefined, spearman: undefined },
           ranked: ranked.slice(0, 12).map((arm) => ({
             measure: arm.measure, separation: arm.separation,
             top: arm.top === Infinity ? "all" : arm.top,
             cutoff: arm.cutoff === Infinity ? "none" : arm.cutoff,
             medianPearson: arm.medianPearson, medianSpearman: arm.medianSpearman,
             worstSpearman: arm.worstSpearman })) };
}
