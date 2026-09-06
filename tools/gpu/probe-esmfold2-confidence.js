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
 * The control that decides whether the STRUCTURE is doing any work.
 *
 * 🔴 IF THE DISTOGRAM'S OWN PEAKEDNESS SCORES AS WELL, THE AGREEMENT IS
 * DECORATION. `exp(-CCE)` is the probability the distogram assigns to the
 * distance the sampler produced - but if the sampler simply realises the
 * distogram's mode, that probability IS the mode's height, and the structure
 * has told us nothing the trunk did not already know. This takes the maximum of
 * each pair's distribution instead of the observed bin's, which is the same
 * aggregate over the same pairs with the structure removed. It is also
 * available BEFORE the sampler runs, so if it wins it is strictly better.
 */
function distogramPeak(logits, bias, tokens, bins, { top = 10, separation = 6 } = {}) {
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
      let total = 0;
      for (let b = 0; b < bins; b += 1) total += Math.exp(logits[base + b] + bias[b] - largest);
      scratch.push(1 / total);
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

export async function main(device, args = []) {
  const crystalPath = option(args, "crystal", "/tools/fixtures/1qys-crystal.pdb");
  const wantedChain = option(args, "chain", "A");
  const seed = Number(option(args, "seed", "0"));
  const crystal = crystalChain(await (await fetch(crystalPath)).text(), wantedChain);

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

  const result = await foldEsmfold2(device, {
    sequence: crystal.sequence, allocator, seed,
    sampler: option(args, "sampler", "diffusion-15"),
    distogramLogits: true,
    shape: { ...M, loops: (M.loops ?? 3) + 1 },
    weights: { featuriser, inputsEmbedder, trunkBlocks, denoiser, shim },
    tower: async (ids, sequenceId) => (await new EsmcTowerGpu(device, allocator).run(ids, {
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
    }, towerShared, { sequenceId })).single,
  });

  // 🔴 THE DISTOGRAM IS OVER THE REPRESENTATIVE ATOM, NOT THE ALPHA CARBON.
  // Upstream's `compute_representative_atoms` takes CB, or CA for glycine - so
  // scoring CA-CA distances against a CB-CB distribution is scoring the wrong
  // distances, which would look like a weak estimator rather than a wrong one.
  const { features, coordinates } = result;
  const named = (atom, text) => {
    for (let i = 0; i < 4; i += 1) {
      const wanted = i < text.length ? text.charCodeAt(i) - 32 : 0;
      if (features.refAtomNameChars[atom * 4 + i] !== wanted) return false;
    }
    return true;
  };
  const representative = new Int32Array(result.tokens).fill(-1);
  const alpha = new Int32Array(result.tokens).fill(-1);
  for (let atom = 0; atom < result.atoms; atom += 1) {
    if (features.mask[atom] === 0) continue;
    const token = features.atomToToken[atom];
    if (named(atom, "CA")) alpha[token] = atom;
    if (representative[token] < 0 || named(atom, "CB")) {
      if (representative[token] < 0 || named(atom, "CB")) representative[token] = atom;
    }
  }
  for (let token = 0; token < result.tokens; token += 1) {
    if (representative[token] < 0) representative[token] = alpha[token];
  }
  const gather = (slots) => {
    const out = new Float32Array(slots.length * 3);
    slots.forEach((atom, index) => {
      for (let axis = 0; axis < 3; axis += 1) out[index * 3 + axis] = coordinates[atom * 3 + axis];
    });
    return out;
  };
  const modelAlpha = gather([...alpha]);
  const modelRepresentative = gather([...representative]);

  const lddt = perResidueLddt(modelAlpha, crystal.coordinates, result.tokens);
  let meanLddt = 0;
  for (const value of lddt) meanLddt += value;
  meanLddt /= lddt.length;

  const rows = [];
  for (const separation of [6, 12]) {
    for (const top of [1, 3, 5, 10, 20, 1e9]) {
      const score = distogramLikelihood(result.distogram.logits, result.distogram.bias,
        modelRepresentative, result.tokens, M.distogramBins, { top, separation });
      rows.push({ separation, top: top > 1e8 ? "all" : top,
                  pearson: pearson(score, lddt), spearman: spearman(score, lddt) });
    }
  }
  const neighbours = neighbourCount(modelRepresentative, result.tokens);
  const baseline = { pearson: pearson(neighbours, lddt), spearman: spearman(neighbours, lddt) };
  const controls = [];
  for (const top of [10, 20]) {
    const peak = distogramPeak(result.distogram.logits, result.distogram.bias,
      result.tokens, M.distogramBins, { top, separation: 6 });
    controls.push({ label: `peakedness, top ${top}`,
                    pearson: pearson(peak, lddt), spearman: spearman(peak, lddt) });
  }
  const contact = [];
  for (const mode of ["binary", "categorical", "kept", "made"]) {
    for (const top of [5, 10, 20]) {
      const score = contactScore(result.distogram.logits, result.distogram.bias,
        modelRepresentative, result.tokens, M.distogramBins, mode,
        { top, separation: 6 });
      contact.push({ mode, top,
                     pearson: pearson(score, lddt), spearman: spearman(score, lddt) });
    }
  }

  console.log(`  ${crystalPath}  chain ${wantedChain}: ${crystal.sequence.length} observed`
    + ` residues, ${crystal.gaps} gap(s)`);
  // 🔴 THE LABEL'S SPREAD IS PART OF THE RESULT. A correlation against a label
  // that barely varies is a weak test however high it reads, and a target the
  // model folds at 0.92 has little to be uncertain about. Both of these are
  // well-folded; a target the model FAILS is where a confidence estimate earns
  // its place, and this probe has not seen one.
  const sorted = [...lddt].sort((a, b) => a - b);
  const at = (q) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
  console.log(`  mean lDDT-Ca against the crystal   ${meanLddt.toFixed(3)}`
    + `   (10th ${at(0.1).toFixed(3)}, 90th ${at(0.9).toFixed(3)})\n`);
  console.log("  sep  topN     Pearson   Spearman");
  for (const row of rows) {
    console.log(`  ${String(row.separation).padStart(3)}  ${String(row.top).padStart(4)}`
      + `    ${row.pearson.toFixed(3).padStart(7)}   ${row.spearman.toFixed(3).padStart(8)}`);
  }
  console.log("\n  ColabDesign-style, restricted to the contact bins:");
  console.log("  mode         topN     Pearson   Spearman");
  for (const row of contact) {
    console.log(`  ${row.mode.padEnd(12)} ${String(row.top).padStart(4)}`
      + `    ${row.pearson.toFixed(3).padStart(7)}   ${row.spearman.toFixed(3).padStart(8)}`);
  }
  console.log();
  for (const control of controls) {
    console.log(`  ${control.label.padEnd(22)} (control)`
      + `  ${control.pearson.toFixed(3).padStart(7)}   ${control.spearman.toFixed(3).padStart(8)}`);
  }
  console.log(`  baseline: neighbour count          `
    + `${baseline.pearson.toFixed(3).padStart(7)}   ${baseline.spearman.toFixed(3).padStart(8)}`);

  return { crystal: crystalPath, chain: wantedChain, seed,
           residues: crystal.sequence.length, gaps: crystal.gaps,
           meanLddt, lddtSpread: [at(0.1), at(0.9)], rows, contact, controls, baseline };
}
