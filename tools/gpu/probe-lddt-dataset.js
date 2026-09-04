/**
 * Fit data for the distogram lDDT estimator, against NATIVE structures.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-lddt-dataset.js --to=20 > a.json
 *     node tools/gpu-chrome.mjs tools/gpu/probe-lddt-dataset.js --from=20 --to=40
 *
 * WHY THE NATIVE AND NOT pLDDT. Everything before this fitted the estimator to
 * the confidence head's own answer, which makes the head the ceiling: at best
 * the estimator learns to reproduce another estimate. lDDT against a
 * crystal structure is the quantity pLDDT is a prediction OF, so fitting to it
 * measures the thing itself.
 *
 * 🔴 lDDT NEEDS NO SUPERPOSITION, which is what makes this cheap and exact. It
 * compares two distance matrices, so a predicted structure is scored against
 * the native without fitting one onto the other and without a single arbitrary
 * choice of frame.
 *
 * 🔴 AND THE CORRESPONDENCE IS BY CONSTRUCTION. The sequence folded is the one
 * read off the native's own CA records, so residue k of the prediction is
 * residue k of the native. Residues the crystal did not resolve are simply
 * absent from both - the distances across such a gap are whatever the native
 * says, which is the truth being scored against.
 *
 * WHAT IT COLLECTS. One row per residue per structure: the estimator's value
 * under several settings, and the true lDDT. Structures are the trunk preview
 * after every recycle plus the finished sample, so the fit sees the range from
 * a two-step guess to a converged fold rather than only good ones.
 *
 * 🔴 AND THE COORDINATES, BOTH SIDES, SO THE TARGET CAN BE RE-DEFINED WITHOUT
 * RE-FOLDING. lDDT's own radius and thresholds are choices; with the model's
 * alpha carbons and the native's kept, true lDDT can be recomputed under any
 * of them offline, in seconds, for every structure at once. Only the
 * ESTIMATOR side then needs a re-fold - the sweeps here are the ones that read
 * the distogram.
 *
 * 🔴 THE DISTOGRAM IS SAVED, BECAUSE IT IS THE POINT. A set that holds only
 * the values of estimators someone already wrote can fit nothing new; what a
 * predictor of pLDDT per position needs is the raw features it will be built
 * from. So every pass's distogram travels with its structure.
 *
 * 🔴 QUANTISED AND HALVED, OR IT DOES NOT TRAVEL. Full float32 is L^2 * 64 * 4
 * bytes per pass - 2.4 GB over the set, and this harness returns its result as
 * JSON on stdout, so it would be base64 and half again as large. The
 * probabilities are stored as bytes after the softmax, and only the upper
 * triangle: AF3's distogram head is symmetrised by construction
 * (logits + logits^T), so the lower half is not data. That is 390 MB for the
 * set, about 39 MB a chunk, and a byte is 1/255 of a probability - renormalise
 * each pair on decode and the error is a fraction of a percent on the sums any
 * feature takes.
 *
 * 🔴 pLDDT IS COMPUTED FOR EVERY STRUCTURE, WHICH A FOLD DOES NOT DO. The head
 * runs once in a fold, on the finished sample, so a trunk preview has no
 * label - and a label is exactly what a training set needs. It is cheap to
 * supply: the head reads the trunk's pair and single and the structure's
 * pseudo-beta, all of which every pass already has, and costs about 150 ms at
 * these sizes. So each structure carries the pLDDT the head WOULD have given
 * it, which is the quantity a live estimate is trying to anticipate.
 */
import { featuriseProtein } from "../../src/af3/featurise.js";
import { foldBatch } from "../../src/af3/fold.js";
import { confidenceWeights, openAf3Store, trunkWeights }
  from "../../src/af3/weights.js";
import { diffusionWeights, atomReference, targetFeatureWeights }
  from "../../src/af3/diffusion-weights.js";
import { distogramLddt, distogramLddtTable } from "../../src/af3/distogram-lddt.js";
import { Af3ConfidenceHeadGpu } from "../../src/af3/confidence-webgpu.js";
import { DIALECT } from "../../src/af3/fold.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const AA3 = {
  ALA: "A", ARG: "R", ASN: "N", ASP: "D", CYS: "C", GLN: "Q", GLU: "E", GLY: "G",
  HIS: "H", ILE: "I", LEU: "L", LYS: "K", MET: "M", PHE: "F", PRO: "P", SER: "S",
  THR: "T", TRP: "W", TYR: "Y", VAL: "V",
};

/** The first protein chain's sequence and alpha carbons. */
function parseNative(text) {
  const sequence = [];
  const coords = [];
  let chain;
  for (const line of text.split("\n")) {
    if (!line.startsWith("ATOM")) continue;
    if (line.slice(12, 16).trim() !== "CA") continue;
    const alt = line[16];
    if (alt !== " " && alt !== "A") continue;
    const code = AA3[line.slice(17, 20)];
    if (code === undefined) continue;
    if (chain === undefined) chain = line[21];
    if (line[21] !== chain) break;
    sequence.push(code);
    coords.push([
      Number(line.slice(30, 38)), Number(line.slice(38, 46)), Number(line.slice(46, 54)),
    ]);
  }
  return { sequence: sequence.join(""), coords };
}

const THRESHOLDS = [0.5, 1, 2, 4];

/**
 * True lDDT per residue: the model's distances against the native's, over the
 * pairs the NATIVE puts inside the radius. No superposition anywhere.
 */
function trueLddt(model, native, radius) {
  const n = native.length;
  const out = new Float64Array(n);
  const distance = (a, b) => Math.sqrt(
    (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2);
  for (let i = 0; i < n; i += 1) {
    let preserved = 0;
    let included = 0;
    for (let j = 0; j < n; j += 1) {
      if (i === j) continue;
      const reference = distance(native[i], native[j]);
      if (reference >= radius) continue;
      included += 1;
      const error = Math.abs(distance(model[i], model[j]) - reference);
      for (const threshold of THRESHOLDS) if (error < threshold) preserved += 1;
    }
    out[i] = included === 0 ? 0 : (100 * preserved) / (THRESHOLDS.length * included);
  }
  return out;
}

/** Token alpha carbons from a dense atom array. */
function alphaCarbons(batch, positions) {
  const out = [];
  for (let token = 0; token < batch.tokens; token += 1) {
    const slot = batch.tokenAtomsToPseudoBeta;
    const index = Number(slot.indices[token]) * 3;
    out.push([positions[index], positions[index + 1], positions[index + 2]]);
  }
  return out;
}

export async function main(device, args) {
  const from = Number(option(args, "from", "0"));
  const to = Number(option(args, "to", "1000"));
  const recycles = Number(option(args, "recycles", "4"));
  const maxLength = Number(option(args, "max-length", "1000"));
  // 🔴 AN INDEX FILE, BECAUSE THE DEV SERVER DOES NOT LIST DIRECTORIES.
  // tools/gpu-chrome.mjs serves files, not listings, so fetching the folder
  // returns nothing and the run silently does no work at all.
  let names = await (await fetch("/natives/index.json")).json();
  // 🔴 A RANDOM SUBSET, SEEDED, so a small run is representative rather than
  // alphabetical - and so the same small run can be repeated exactly.
  const sample = Number(option(args, "sample", "0"));
  if (sample > 0) {
    let state = Number(option(args, "sample-seed", "1")) >>> 0;
    const random = () => {
      state = (state * 1664525 + 1013904223) >>> 0;
      return state / 4294967296;
    };
    const shuffled = [...names];
    for (let i = shuffled.length - 1; i > 0; i -= 1) {
      const j = Math.floor(random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    names = shuffled.slice(0, sample);
  }

  const store = await openAf3Store(option(args, "model", "/model-af3-int5/manifest.json"));
  const [trunk, diffusion, confidence, reference, targetFeat] = await Promise.all([
    trunkWeights(store), diffusionWeights(store), confidenceWeights(store),
    atomReference(store), targetFeatureWeights(store),
  ]);
  const weights = { trunk, diffusion, confidence, atomReference: reference, targetFeat };

  // 🔴 COMPACT ROWS, because this is 150 targets by 100 residues by 5
  // structures and an object per row is megabytes of punctuation. Columns
  // first, values as arrays.
  // `plddt` is the head's answer for THAT structure and is the fitting target;
  // `true` is lDDT against the crystal, kept beside it because it costs
  // nothing and says which of the two a predictor is really tracking.
  const columns = ["target", "pass", "residue", "r12", "r15", "r18", "r22",
    "sep3", "sep6", "k16", "k32", "plddt", "true"];
  const head = new Af3ConfidenceHeadGpu(device);
  const rows = [];
  const failures = [];
  // Per target: the native's alpha carbons, the head's per-residue pLDDT, and
  // every structure's alpha carbons. Small beside the rows, and enough to
  // recompute any distance-based target offline.
  const structures = {};
  const withDistogram = option(args, "distograms", "1") !== "0";

  /**
   * The distogram as bytes: softmax, upper triangle, one byte a bin.
   *
   * 🔴 THE SOFTMAX HAPPENS HERE, NOT ON DECODE. Logits are unbounded and their
   * scale is arbitrary; probabilities are what every feature is a sum over, so
   * they are what is stored - and they quantise into a byte without a scale
   * factor anyone has to remember.
   */
  const packDistogram = (logits, tokens, bins) => {
    const bytes = new Uint8Array((tokens * (tokens + 1) / 2) * bins);
    const probability = new Float64Array(bins);
    let out = 0;
    for (let i = 0; i < tokens; i += 1) {
      for (let j = i; j < tokens; j += 1) {
        const base = (i * tokens + j) * bins;
        let largest = -Infinity;
        for (let b = 0; b < bins; b += 1) {
          if (logits[base + b] > largest) largest = logits[base + b];
        }
        let total = 0;
        for (let b = 0; b < bins; b += 1) {
          probability[b] = Math.exp(logits[base + b] - largest);
          total += probability[b];
        }
        for (let b = 0; b < bins; b += 1) {
          bytes[out + b] = Math.round((probability[b] / total) * 255);
        }
        out += bins;
      }
    }
    let binary = "";
    const step = 0x8000;
    for (let at = 0; at < bytes.length; at += step) {
      binary += String.fromCharCode(...bytes.subarray(at, at + step));
    }
    return btoa(binary);
  };
  for (const name of names.slice(from, to)) {
    try {
      const native = parseNative(await (await fetch(`/natives/${name}.pdb`)).text());
      if (native.sequence.length < 30 || native.sequence.length > maxLength) continue;
      const batch = featuriseProtein(native.sequence, {});
      const captured = [];
      let latestTrunk;
      const result = await foldBatch(device, batch, weights, {
        mode: "flow", steps: 16, recycles, seed: 3,
        // 🔴 THE TRUNK IS HELD, NOT ATTACHED TO THE LAST STRUCTURE SEEN.
        // fold.js fires `recycle-done` BEFORE that pass's preview, so reading
        // backwards from the captured list pairs each distogram with the
        // PREVIOUS pass's structure - an off-by-one that nothing downstream
        // could detect, because every row would still look well formed.
        onStage: (stage, detail) => {
          if (stage === "recycle-done") latestTrunk = detail.trunk;
        },
        onPreview: ({ positions, pass }) => {
          captured.push({ pass, positions: Float32Array.from(positions),
            trunk: latestTrunk });
        },
      });
      captured.push({ pass: recycles, positions: result.positions, trunk: result.trunk });

      /** Per-residue pLDDT from a run of the head over one structure. */
      const perResiduePlddt = (scores) => {
        const out = [];
        for (let token = 0; token < batch.tokens; token += 1) {
          let total = 0;
          let count = 0;
          for (let atom = 0; atom < batch.dense; atom += 1) {
            const slot = token * batch.dense + atom;
            if (!batch.predDenseAtomMask[slot]) continue;
            total += scores.plddt[slot];
            count += 1;
          }
          out.push(count === 0 ? 0 : Number((total / count).toFixed(2)));
        }
        return out;
      };
      const plddt = perResiduePlddt(result.scores);
      structures[name] = {
        sequence: native.sequence,
        native: native.coords.map((p) => p.map((v) => Number(v.toFixed(2)))),
        plddt,
        ptm: Number(result.ptm.toFixed(3)),
        models: {},
      };
      structures[name].binEdges = Array.from(result.trunk.binEdges);
      structures[name].distograms = {};

      for (const structure of captured) {
        const source = structure.trunk ?? result.trunk;
        const model = alphaCarbons(batch, structure.positions);
        const truth = trueLddt(model, native.coords, 15);
        // ...and what the head says about THIS structure, using the trunk of
        // the pass that produced it.
        const pseudo = new Float32Array(batch.tokens * 3);
        for (let t = 0; t < batch.tokens; t += 1) {
          for (let a = 0; a < 3; a += 1) pseudo[t * 3 + a] = model[t][a];
        }
        const scored = structure.pass === recycles ? result.scores
          : await head.run({
            tokens: batch.tokens, dense: batch.dense, seqMask: batch.seqMask,
            pair: source.pair, single: source.single,
            targetFeat: result.targetFeat, pseudoBeta: pseudo,
          }, weights.confidence, DIALECT);
        const passPlddt = perResiduePlddt(scored);
        structures[name].plddtByPass = structures[name].plddtByPass ?? {};
        structures[name].plddtByPass[structure.pass] = passPlddt;
        structures[name].models[structure.pass] =
          model.map((p) => p.map((v) => Number(v.toFixed(2))));
        if (withDistogram) {
          structures[name].distograms[structure.pass] = packDistogram(
            source.logits, batch.tokens, source.binEdges.length + 1);
        }
        const beta = new Float32Array(batch.tokens * 3);
        for (let t = 0; t < batch.tokens; t += 1) {
          for (let a = 0; a < 3; a += 1) beta[t * 3 + a] = model[t][a];
        }
        const variants = {};
        for (const radius of [12, 15, 18, 22]) {
          variants[`r${radius}`] = distogramLddt(distogramLddtTable(
            source.logits, source.binEdges, batch.tokens, batch.seqMask, { radius }), beta);
        }
        for (const separation of [3, 6]) {
          variants[`sep${separation}`] = distogramLddt(distogramLddtTable(
            source.logits, source.binEdges, batch.tokens, batch.seqMask,
            { radius: 18, separation }), beta);
        }
        for (const maxContacts of [16, 32]) {
          variants[`k${maxContacts}`] = distogramLddt(distogramLddtTable(
            source.logits, source.binEdges, batch.tokens, batch.seqMask,
            { radius: 18, maxContacts }), beta);
        }
        for (let i = 0; i < batch.tokens; i += 1) {
          rows.push([name, structure.pass, i,
            +variants.r12[i].toFixed(2), +variants.r15[i].toFixed(2),
            +variants.r18[i].toFixed(2), +variants.r22[i].toFixed(2),
            +variants.sep3[i].toFixed(2), +variants.sep6[i].toFixed(2),
            +variants.k16[i].toFixed(2), +variants.k32[i].toFixed(2),
            passPlddt[i], +truth[i].toFixed(2)]);
        }
      }
    } catch (cause) {
      failures.push({ name, error: String(cause?.message ?? cause) });
    }
  }
  return { columns, targets: names.slice(from, to).length, rows, structures,
    failures };
}
