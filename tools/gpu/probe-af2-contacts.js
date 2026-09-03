/**
 * Does AF2's distogram head predict the structure AF2 actually built?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-af2-contacts.js
 *     node tools/gpu-chrome.mjs tools/gpu/probe-af2-contacts.js --recycles=2
 *
 * WHY IT EXISTS. src/heads/distogram.js is covered by unit tests, but those
 * are differential: they say the head computes a projection, a symmetrisation
 * and a softmax. They cannot catch the two things most likely to be wrong
 * about a head converted from someone else's checkpoint - a transposed weight
 * matrix and a mistaken bin convention - because both produce a perfectly
 * well-formed contact map about nothing.
 *
 * 🔴 SO IT IS SCORED AGAINST THE STRUCTURE, WHICH IS A REAL ANSWER. The
 * distogram and the structure module read the SAME pair representation, so a
 * correctly converted head must agree with the coordinates AF2 went on to
 * build: pairs it calls contacts should be close in space. Reported as the
 * area under the ROC curve of P(contact) against `d(CB, CB) < 8 A`, which is
 * the standard way a contact predictor is scored and needs no crystal.
 *
 * A transposed weight matrix scores about 0.5 here. So does a head reading the
 * wrong bins. A correct one on a well-folded chain is well above 0.9.
 *
 * WHAT IT FOUND: 0.999, 1.000 and 1.000 over three passes of a 58-mer, with
 * the map moving 0.084 and 0.114 at its largest between them - so the head is
 * converted correctly AND each recycle really is a different picture, which is
 * the premise for putting one on every frame.
 *
 * 🔴 THE POSITIVE COUNT IS SMALL AND THAT LIMITS THE CLAIM. Only 6 to 8 of the
 * 1378 scored pairs are true contacts here, because the alignment is synthetic
 * - rows built by gap-masking the query - so the fold is mediocre (pLDDT 70 to
 * 78) and loosely packed. An AUC over so few positives says the head puts the
 * real contacts at the very top, which a transposed matrix cannot do; it does
 * not say the head is calibrated. Run it with a real alignment for that.
 *
 * 🔴 AND THE DIAGONAL IS EXCLUDED. Neighbours are in contact in any chain,
 * folded or not, so scoring them inflates every number and would hide exactly
 * the failure this is looking for. |i - j| >= 6, the usual short-range cut.
 */
import { AlphaFoldFixture } from "../../src/reference/alphafold-fixture.js";
import { HttpTensorStore } from "../../src/reference/http-tensor-store.js";
import { AlphaFoldMonomerGpu } from "../../src/model/monomer.js";
import { distogramContactProbabilities } from "../../src/heads/distogram.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const DEFAULT_SEQUENCE =
  "GWSTELEKHREELKEFLKKEGITLGFTNAEKQEQAQKLGLGKKVSPELLIKAFAILKK";

/** Area under the ROC curve, by rank. No thresholds to choose. */
function rocAuc(scores, labels) {
  const order = [...scores.keys()].sort((a, b) => scores[a] - scores[b]);
  let positives = 0;
  let negatives = 0;
  for (const label of labels) { if (label) positives += 1; else negatives += 1; }
  if (positives === 0 || negatives === 0) return Number.NaN;
  // Ranks with ties averaged, so a flat predictor scores exactly 0.5.
  const rank = new Float64Array(scores.length);
  let index = 0;
  while (index < order.length) {
    let end = index;
    while (end + 1 < order.length && scores[order[end + 1]] === scores[order[index]]) end += 1;
    const mean = (index + end) / 2 + 1;
    for (let k = index; k <= end; k += 1) rank[order[k]] = mean;
    index = end + 1;
  }
  let sum = 0;
  for (let i = 0; i < labels.length; i += 1) if (labels[i]) sum += rank[i];
  return (sum - (positives * (positives + 1)) / 2) / (positives * negatives);
}

export async function main(device, args) {
  const sequence = option(args, "sequence", DEFAULT_SEQUENCE);
  const recycles = Number(option(args, "recycles", "1"));
  const rows = Number(option(args, "rows", "128"));

  const { MODEL_BUNDLES, loadManifest } = await import("../../src/reference/manifests/index.js");
  const store = await HttpTensorStore.fromManifest(
    MODEL_BUNDLES.monomer.directory, await loadManifest("monomer"));
  const fixture = AlphaFoldFixture.fromStore(store);
  const [embedding, template, extraStack, mainStack, structure, confidence,
         geometry, featureTables, paeBreaks, distogram] = await Promise.all([
    fixture.embeddingWeights(), fixture.templateWeights(), fixture.extraStackWeights(),
    fixture.mainStackWeights(), fixture.structureWeights(), fixture.confidenceWeights(),
    fixture.geometryTables(), fixture.queryOnlyFeatureTables(),
    fixture.tensor("confidencePaeBreaks"), fixture.distogramHeadWeights(),
  ]);
  if (distogram === undefined) {
    return { ok: false, error: "this bundle has no distogram head;"
      + " run tools/add_distogram_head.py" };
  }
  const weights = {
    embedding, template, extraStack, mainStack, structure,
    lddt: confidence.lddt, pae: confidence.pae, geometry,
  };

  const lines = [">query", sequence];
  for (let row = 1; row < rows; row += 1) {
    lines.push(`>synthetic${row}`);
    lines.push([...sequence].map((code, column) =>
      (column % (row % 11 + 3) === 0 ? "-" : code)).join(""));
  }
  const result = await new AlphaFoldMonomerGpu(device).predictA3m(
    `${lines.join("\n")}\n`, weights, featureTables,
    { recycles, randomSeed: 0, maxMsaSequences: rows, maxExtraSequences: rows,
      chainLengths: [sequence.length] },
    paeBreaks,
  );

  const length = sequence.length;
  const passes = [];
  for (const [index, recycle] of result.recycles.entries()) {
    const contacts = distogramContactProbabilities(
      recycle.pair, distogram.halfLogitsWeights, distogram.halfLogitsBias, length,
      { bins: distogram.bins, first: distogram.firstBreak, last: distogram.lastBreak });

    // 🔴 CB, NOT CA, because that is the atom AlphaFold's own distogram is
    // defined on - and glycine has none, so it falls back to CA exactly as
    // the reference featuriser does.
    const atom = recycle.structure.atom37;
    const point = (residue) => {
      const cb = (residue * 37 + 3) * 3;
      const ca = (residue * 37 + 1) * 3;
      const useCb = atom[cb] !== 0 || atom[cb + 1] !== 0 || atom[cb + 2] !== 0;
      const base = useCb ? cb : ca;
      return [atom[base], atom[base + 1], atom[base + 2]];
    };
    const scores = [];
    const labels = [];
    let predicted = 0;
    for (let i = 0; i < length; i += 1) {
      for (let j = i + 6; j < length; j += 1) {
        const a = point(i);
        const b = point(j);
        const d = Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2);
        scores.push(contacts[i * length + j]);
        labels.push(d < 8 ? 1 : 0);
        if (contacts[i * length + j] > 0.5) predicted += 1;
      }
    }
    passes.push({
      recycle: index,
      auc: Number(rocAuc(scores, labels).toFixed(3)),
      trueContacts: labels.reduce((a, b) => a + b, 0),
      predictedOverHalf: predicted,
      pairsScored: scores.length,
      meanPlddt: Number(recycle.confidence.meanPlddt.toFixed(1)),
      // 🔴 AND WHETHER IT MOVES. The whole reason each recycle carries its own
      // map is that the model changes its mind; if every pass produced the
      // same contacts there would be nothing to show per frame.
      maxAbsChangeFromPrevious: index === 0 ? null : Number(Math.max(...
        contacts.map((value, k) => Math.abs(value - passes[index - 1].contacts[k]))).toFixed(3)),
      contacts,
    });
  }
  return {
    ok: true,
    sequence: length,
    recycles: result.recycles.length,
    passes: passes.map(({ contacts, ...rest }) => rest),
  };
}
