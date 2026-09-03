/**
 * Do p(intra) and p(inter) track the confidence head's pTM and ipTM?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-contact-confidence.js
 *     node tools/gpu-chrome.mjs tools/gpu/probe-contact-confidence.js --steps=4
 *
 * WHY IT EXISTS. The page shows two numbers off the trunk's distogram, named
 * after what they are - the mean of the strongest contact probabilities within
 * a chain and across chains - rather than after pTM and ipTM, which they are
 * not. That naming is only worth anything if the numbers are worth showing,
 * and what makes them worth showing is that they RANK folds the same way the
 * real head does, minutes earlier. This measures that.
 *
 * 🔴 IT ALSO SWEEPS p(intra)'s SEQUENCE SEPARATION, which is the one free
 * parameter it has. Adjacent residues are in contact in any chain, folded or
 * not, so the separation floor is what decides whether p(intra) asks "is this
 * a chain?" or "does this chain come back on itself?".
 *
 * 🔴 CORRELATION ACROSS TARGETS IS THE QUESTION, so the panel has to span the
 * range: things that fold, a miniprotein, and things that do not - a scramble,
 * a linker, a homopolymer. A correlation measured only on good targets says
 * nothing about a page that has to be honest on bad ones.
 */
import { featuriseProtein } from "../../src/af3/featurise.js";
import { foldBatch } from "../../src/af3/fold.js";
import { confidenceWeights, openAf3Store, trunkWeights }
  from "../../src/af3/weights.js";
import { diffusionWeights, atomReference, targetFeatureWeights }
  from "../../src/af3/diffusion-weights.js";
import { distogramContactConfidence } from "../../src/af3/distogram-confidence.js";
import { bestAlignmentTmScore, tmPerBinFor, tmScoreD0 }
  from "../../src/heads/tm-score.js";

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

const PANEL = [
  ["6MRR", MRR],
  ["Top7", TOP7],
  ["designed-58", DES58],
  ["GCN4", GCN4],
  ["villin-HP36", VILLIN],
  ["trp-cage", TRPCAGE],
  ["6MRR-scrambled", SCRAMBLE],
  ["GS-linker", LINKER],
  ["poly-alanine", POLYALA],
  // 🔴 THE COMPLEXES, BECAUSE ipTM DOES NOT EXIST FOR ONE CHAIN. A panel of
  // monomers measures p(intra) and says nothing at all about p(inter).
  ["GCN4-homodimer", `${GCN4}:${GCN4}`],
  ["6MRR-homodimer", `${MRR}:${MRR}`],
  ["Top7-homodimer", `${TOP7}:${TOP7}`],
  ["GCN4+6MRR", `${GCN4}:${MRR}`],
  ["6MRR+designed-58", `${MRR}:${DES58}`],
  ["Top7+trp-cage", `${TOP7}:${TRPCAGE}`],
  ["linker+GCN4", `${LINKER}:${GCN4}`],
  ["polyA-dimer", `${POLYALA}:${POLYALA}`],
];

const SEPARATIONS = [1, 6, 12, 24];

function pearson(a, b) {
  const n = a.length;
  if (n < 2) return 0;
  let ma = 0; let mb = 0;
  for (let i = 0; i < n; i += 1) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let num = 0; let da = 0; let db = 0;
  for (let i = 0; i < n; i += 1) {
    num += (a[i] - ma) * (b[i] - mb);
    da += (a[i] - ma) ** 2;
    db += (b[i] - mb) ** 2;
  }
  return da === 0 || db === 0 ? 0 : num / Math.sqrt(da * db);
}

/** Ranks with ties averaged, so Spearman is Pearson on them. */
function ranks(values) {
  const order = [...values.keys()].sort((x, y) => values[x] - values[y]);
  const out = new Float64Array(values.length);
  let index = 0;
  while (index < order.length) {
    let end = index;
    while (end + 1 < order.length && values[order[end + 1]] === values[order[index]]) end += 1;
    const mean = (index + end) / 2;
    for (let k = index; k <= end; k += 1) out[order[k]] = mean;
    index = end + 1;
  }
  return out;
}

/** pTM's own reduction over the distogram's expected TM term - no frame. */
function distogramPtm(trunk, batch) {
  const bins = trunk.binEdges.length + 1;
  const centres = new Float64Array(bins);
  for (let bin = 0; bin < bins; bin += 1) {
    const low = bin === 0 ? 0 : trunk.binEdges[bin - 1];
    const high = bin === bins - 1 ? trunk.binEdges[bins - 2] : trunk.binEdges[bin];
    centres[bin] = (low + high) / 2;
  }
  const perBin = tmPerBinFor(centres, tmScoreD0(batch.tokens));
  return bestAlignmentTmScore(trunk.logits, batch.tokens, perBin,
    (a, b) => batch.seqMask[a] > 0 && batch.seqMask[b] > 0);
}

const both = (truth, guess) => ({
  targets: truth.length,
  pearson: Number(pearson(truth, guess).toFixed(3)),
  spearman: Number(pearson(ranks(truth), ranks(guess)).toFixed(3)),
});

export async function main(device, args) {
  const steps = Number(option(args, "steps", "8"));
  const store = await openAf3Store(option(args, "model", "/model-af3-int5/manifest.json"));
  const [trunk, diffusion, confidence, reference, targetFeat] = await Promise.all([
    trunkWeights(store), diffusionWeights(store), confidenceWeights(store),
    atomReference(store), targetFeatureWeights(store),
  ]);
  const weights = { trunk, diffusion, confidence, atomReference: reference, targetFeat };
  const rows = [];
  for (const [name, sequence] of PANEL) {
    const batch = featuriseProtein(sequence, {});
    const result = await foldBatch(device, batch, weights, {
      mode: "flow", steps, recycles: 0, seed: 3,
    });
    const { contactProbs } = result.trunk;
    const row = {
      name,
      tokens: batch.tokens,
      ptm: Number(result.ptm.toFixed(3)),
      meanPlddt: Number(result.meanPlddt.toFixed(1)),
      // 🔴 THE OTHER WAY TO ASK A DISTOGRAM FOR pTM, kept here as the control
      // p(intra) has to beat: run the REAL pTM reduction over the distogram's
      // own expected TM term, with no coordinates in it at all. If this tracks
      // pTM and p(intra) does not, the failure is p(intra)'s and not the
      // distogram's.
      distogramPtm: Number(distogramPtm(result.trunk, batch).toFixed(3)),
      iptm: Number.isNaN(result.iptm) ? null : Number(result.iptm.toFixed(3)),
    };
    for (const separation of SEPARATIONS) {
      const { intra, inter } = distogramContactConfidence(
        contactProbs, batch.asymId, batch.seqMask, batch.tokens, { separation });
      row[`intra@${separation}`] = Number(intra.toFixed(3));
      if (separation === SEPARATIONS[0]) {
        row.inter = Number.isNaN(inter) ? null : Number(inter.toFixed(3));
      }
    }
    rows.push(row);
  }

  const complexes = rows.filter((row) => row.iptm !== null);
  const monomers = rows.filter((row) => row.iptm === null);
  // 🔴 SPLIT ON THE HOMOPOLYMERS, because they are where p(intra) and pTM
  // disagree hardest and the split says WHICH is disagreeing. A helical bundle
  // with no tertiary contacts at all scores well on pTM.
  const sensible = rows.filter((row) => !row.name.includes("poly")
    && !row.name.includes("linker") && !row.name.includes("GS-"));
  const separationSweep = {};
  for (const separation of SEPARATIONS) {
    const key = `intra@${separation}`;
    separationSweep[`${key} vs pTM`] = {
      all: both(rows.map((r) => r.ptm), rows.map((r) => r[key])),
      monomers: both(monomers.map((r) => r.ptm), monomers.map((r) => r[key])),
      complexes: both(complexes.map((r) => r.ptm), complexes.map((r) => r[key])),
      excludingNonsense: both(sensible.map((r) => r.ptm), sensible.map((r) => r[key])),
    };
  }
  return {
    steps,
    rows,
    separationSweep,
    // ...and the one the page ships, against both real scores.
    interVsIptm: both(
      complexes.map((row) => row.iptm), complexes.map((row) => row.inter)),
    interVsPtm: both(
      complexes.map((row) => row.ptm), complexes.map((row) => row.inter)),
    // ...the other two pairings p(intra) could plausibly have had, and the
    // control that shares its input but not its reduction.
    intraVsPlddt: both(
      rows.map((row) => row.meanPlddt), rows.map((row) => row["intra@12"])),
    distogramPtmVsPtm: both(
      rows.map((row) => row.ptm), rows.map((row) => row.distogramPtm)),
  };
}
