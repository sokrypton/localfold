/**
 * Per-token rows for fitting a pLDDT predictor from the trunk's distogram.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-plddt-features.js > rows.json
 *
 * WHY. The colour a frame is drawn with during a fold is the raw distogram
 * agreement, and its bias is per TARGET: on Top7 it reads 86.5 against a real
 * pLDDT of 85.3, on a nonsense fusion 83.6 against 54.0. The page fixes this
 * afterwards by matching the estimate's mean and spread to the finished
 * structure's real pLDDT - which is exact, and only exists once the confidence
 * head has run. A GLOBAL fit does not help: over seventeen targets, pLDDT =
 * 21.1 + 0.670 * agreement cuts the mean error from 9.6 to 7.8 and makes the
 * worst case worse.
 *
 * So this collects what a fit would need instead of guessing at it: one row
 * per TOKEN, with everything knowable before the head runs beside the answer
 * the head gives.
 *
 * 🔴 THE FEATURES ARE TAKEN AT PREVIEW TIME, NOT ON THE FINISHED STRUCTURE,
 * because that is when the number is needed. Two flow cycles against the
 * trunk, exactly as fold.js draws its previews.
 *
 * 🔴 AND THE PANEL HAS TO CONTAIN FAILURES. A predictor fitted only on
 * targets that fold learns that everything is 85. Half of these are designed
 * or natural proteins that resolve from a single sequence; the rest are a
 * scramble, a linker, a homopolymer and two fusions of unrelated chains, which
 * do not.
 */
import { featuriseProtein } from "../../src/af3/featurise.js";
import { foldBatch } from "../../src/af3/fold.js";
import { confidenceWeights, openAf3Store, trunkWeights }
  from "../../src/af3/weights.js";
import { diffusionWeights, atomReference, targetFeatureWeights }
  from "../../src/af3/diffusion-weights.js";
import { distogramAgreementTable, distogramConfidence }
  from "../../src/af3/distogram-confidence.js";
import { distogramLddt, distogramLddtTable } from "../../src/af3/distogram-lddt.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const MRR = "GWSTELEKHREELKEFLKKEGITLGFTNAEKQEQAQKLGLGKKVSPELLIKAFAILKK";
const TOP7 = "DIQVQVNIDDNGKNFDYTYTVTTESELQKVLNELMDYIKKQGAKRVRISITARTKKEAEKFAAILIKVFAELGYNDINVTFDGDTVTVEGQLE";
const DES58 = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";
const GCN4 = "MKQLEDKVEELLSKNYHLENEVARLKKLVGER";
const VILLIN = "LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF";
const TRPCAGE = "NLYIQWLKDGGPSSGRPPPS";
const SCRAMBLE = "EKLGKFLKSLEHTKEEGRWLNFAKQGKKGLEAIVELPLIKELSTKQFAEKAIGLRLTEKS";
const LINKER = "GSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGS";
const POLYALA = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
const BBA = "EQYTAKYKGRTFRNEKELRDFIEKFKGR";
const WW = "GSKLPPGWEKRMSRSSGRVYYFNHITNASQWERPSGNSS";

const PANEL = [
  ["6MRR", MRR], ["Top7", TOP7], ["designed-58", DES58], ["GCN4", GCN4],
  ["villin", VILLIN], ["trp-cage", TRPCAGE], ["BBA", BBA], ["WW", WW],
  ["scrambled", SCRAMBLE], ["GS-linker", LINKER], ["poly-alanine", POLYALA],
  ["GCN4+6MRR", `${GCN4}${MRR}`], ["Top7+linker", `${TOP7}${LINKER}`],
  ["DES58+polyA", `${DES58}${POLYALA}`],
];

/** Token pseudo-beta from a dense atom array. */
function pseudoBetaOf(batch, positions) {
  const beta = batch.tokenAtomsToPseudoBeta;
  const out = new Float32Array(batch.tokens * 3);
  for (let token = 0; token < batch.tokens; token += 1) {
    if (!beta.mask[token]) continue;
    const from = Number(beta.indices[token]) * 3;
    for (let axis = 0; axis < 3; axis += 1) out[token * 3 + axis] = positions[from + axis];
  }
  return out;
}

export async function main(device, args) {
  const recycles = Number(option(args, "recycles", "1"));
  const store = await openAf3Store(option(args, "model", "/model-af3-int5/manifest.json"));
  const [trunk, diffusion, confidence, reference, targetFeat] = await Promise.all([
    trunkWeights(store), diffusionWeights(store), confidenceWeights(store),
    atomReference(store), targetFeatureWeights(store),
  ]);
  const weights = { trunk, diffusion, confidence, atomReference: reference, targetFeat };

  const rows = [];
  const perTarget = [];
  for (const [name, sequence] of PANEL) {
    const batch = featuriseProtein(sequence, {});
    let preview;
    const result = await foldBatch(device, batch, weights, {
      mode: "flow", steps: 16, recycles, seed: 3,
      // ...the same two-cycle preview fold.js draws, captured for its features.
      onPreview: ({ positions }) => { if (preview === undefined) preview = Float32Array.from(positions); },
    });
    if (preview === undefined) preview = result.positions;

    const table = distogramAgreementTable(
      result.trunk.logits, result.trunk.binEdges, batch.tokens, batch.seqMask);
    const beta = pseudoBetaOf(batch, preview);
    const raw = distogramConfidence(table, beta);
    // 🔴 THE DERIVED ESTIMATOR, SCORED ON THE SAME FRAME. lDDT's own
    // arithmetic with the distogram standing in for the reference - four
    // thresholds, probabilistic inclusion inside 15 A, no separation floor.
    // 🔴 SEVERAL SETTINGS IN ONE PASS, because the fold is the expensive part
    // and the estimator is CPU arithmetic over the same distogram. The
    // inclusion radius and the threshold set are lDDT's own constants; whether
    // they are the right ones for a DISTOGRAM standing in for the reference is
    // a question only the panel can answer.
    const variants = {};
    for (const radius of [10, 12, 15, 18, 22]) {
      const built = distogramLddtTable(result.trunk.logits, result.trunk.binEdges,
        batch.tokens, batch.seqMask, { radius });
      variants[`r${radius}`] = distogramLddt(built, beta);
      if (radius === 18) {
        variants.r18wide = distogramLddt(built, beta, { thresholds: [1, 2, 4, 8] });
      }
      if (radius === 15) {
        variants.t1 = distogramLddt(built, beta, { thresholds: [1] });
        variants.t2 = distogramLddt(built, beta, { thresholds: [2] });
        variants.wide = distogramLddt(built, beta, { thresholds: [1, 2, 4, 8] });
        variants.tight = distogramLddt(built, beta, { thresholds: [0.25, 0.5, 1, 2] });
      }
    }
    // ...and the two parameters lDDT itself does not have: a separation floor
    // and a cap on how many contacts a token is scored on.
    for (const separation of [1, 2, 3, 6, 12]) {
      const built = distogramLddtTable(result.trunk.logits, result.trunk.binEdges,
        batch.tokens, batch.seqMask, { radius: 18, separation });
      variants[`sep${separation}`] = distogramLddt(built, beta);
    }
    for (const maxContacts of [8, 16, 32, 64]) {
      const built = distogramLddtTable(result.trunk.logits, result.trunk.binEdges,
        batch.tokens, batch.seqMask, { radius: 18, maxContacts });
      variants[`k${maxContacts}`] = distogramLddt(built, beta);
    }
    const lddt = variants.r18;

    // The head's answer, per token: the mean over the atoms a token has.
    const real = new Float64Array(batch.tokens);
    for (let token = 0; token < batch.tokens; token += 1) {
      let total = 0;
      let count = 0;
      for (let atom = 0; atom < batch.dense; atom += 1) {
        const slot = token * batch.dense + atom;
        if (!batch.predDenseAtomMask[slot]) continue;
        total += result.scores.plddt[slot];
        count += 1;
      }
      real[token] = count === 0 ? 0 : total / count;
    }

    // 🔴 THE CEILING IS WHAT ANY STRUCTURE COULD SCORE on this token's
    // contacts - the best agreement available given how sharp the trunk's
    // distogram is - so `raw / ceiling` separates "placed badly" from "the
    // model never knew". Read straight off the table: the max over the
    // observed bin of the mass within tolerance.
    const { agreement, neighbours, counts, contacts, bins, spread, tokens } = table;
    for (let i = 0; i < tokens; i += 1) {
      if (batch.seqMask[i] <= 0 || counts[i] === 0) continue;
      let ceiling = 0;
      let sharp = 0;
      for (let k = 0; k < counts[i]; k += 1) {
        const slot = (i * contacts + k) * bins;
        let best = 0;
        for (let b = 0; b < bins; b += 1) best = Math.max(best, agreement[slot + b]);
        ceiling += best;
        sharp += spread[i * tokens + neighbours[i * contacts + k]];
      }
      rows.push({
        target: name,
        raw: Number(raw[i].toFixed(2)),
        lddt: Number(lddt[i].toFixed(2)),
        ...Object.fromEntries(Object.entries(variants)
          .map(([k, v]) => [k, Number(v[i].toFixed(2))])),
        ceiling: Number((100 * ceiling / counts[i]).toFixed(2)),
        spread: Number((sharp / counts[i]).toFixed(3)),
        contacts: counts[i],
        real: Number(real[i].toFixed(2)),
      });
    }
    const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
    perTarget.push({ name, tokens: batch.tokens,
      raw: Number(mean([...raw]).toFixed(1)),
      lddt: Number(mean([...lddt]).toFixed(1)),
      real: Number(mean([...real]).toFixed(1)),
      ptm: Number(result.ptm.toFixed(3)) });
  }
  return { recycles, targets: perTarget.length, perTarget, rows };
}
