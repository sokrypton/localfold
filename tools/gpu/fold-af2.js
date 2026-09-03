/**
 * An AlphaFold 2 monomer fold, end to end, so a kernel change can be compared
 * against the tree before it.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/fold-af2.js
 *     node tools/gpu-chrome.mjs tools/gpu/fold-af2.js --rows=512 --recycles=1
 *     node tools/gpu-chrome.mjs tools/gpu/fold-af2.js --family=multimer --chains=30,29
 *
 * 🔴 AF2 HAD NO END-TO-END GATE THAT RUNS ON THIS MACHINE, and its kernels have
 * now been rewritten three times. `npm run test:gpu` cannot load Dawn here;
 * tools/gpu/check-evoformer-stack.js is the official-value check and wants
 * test/fixtures/evoformer/model1-query-59-stack, which this checkout does not
 * carry - it has the features and not `stackInputMsa`. So every AF2 kernel
 * change has been gated on per-kernel differential checkers, each of which can
 * only say that ONE kernel still computes its own operation. Nothing said the
 * assembled model still folds.
 *
 * This is not an oracle either - it does not know what AlphaFold would say. It
 * folds deterministically and prints enough to compare two trees: mean pLDDT,
 * pTM, the backbone CA-CA geometry, and a checksum over every coordinate.
 * Run it, stash the change, run it again.
 *
 * 🔴 THE ALIGNMENT IS SYNTHESISED FROM THE QUERY, WHICH IS FINE HERE AND ONLY
 * HERE. Rows are the query with every (i+3)th column dropped to a gap, so the
 * MSA path and its 512-row kernels actually run without fetching anything -
 * this machine is on a metered connection and an MMseqs2 search is 88 s and a
 * download besides. It makes the numbers meaningless as biology and perfectly
 * good as a fingerprint, which is what a regression needs.
 *
 * 🔴 MULTIMER SHARES EVERY KERNEL AND HAD NO GATE AT ALL. src/multimer/block.js
 * builds its blocks from the same attention, transition and outer-product-mean
 * shaders src/evoformer/block.js does, with its own dispatches - so a tile
 * changed in one and not threaded through the other is a bug that only a
 * multimer fold can see. `--family=multimer` runs that path, through
 * AlphaFoldUnifiedGpu and the multimer regime the page passes (outer product
 * mean first, chain-aware, position scale 20), with `--chains` splitting the
 * sequence.
 *
 * 🔴 AND pLDDT IS NOT THE CHECK. AF3.md records a batch with one broken gather
 * folding 17 A of spaghetti at pLDDT 55. Consecutive CA are 3.80 A apart in a
 * real protein and nothing else; `caca` is the number that a wrong kernel
 * cannot fake, so it is printed with its worst outlier.
 */
import { AlphaFoldFixture } from "../../src/reference/alphafold-fixture.js";
import { HttpTensorStore } from "../../src/reference/http-tensor-store.js";
import { AlphaFoldMonomerGpu } from "../../src/model/monomer.js";
import { AlphaFoldUnifiedGpu } from "../../src/multimer/model.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/** 59 residues with side chains of every length; the shape the benches use. */
const DEFAULT_SEQUENCE = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";

export async function main(device, args) {
  const sequence = option(args, "sequence", DEFAULT_SEQUENCE);
  const family = option(args, "family", "monomer");
  if (family !== "monomer" && family !== "multimer") {
    throw new RangeError(`unknown family ${family}: expected "monomer" or "multimer"`);
  }
  const chainLengths = option(args, "chains", "").split(",").filter(Boolean).map(Number);
  const rows = Number(option(args, "rows", "128"));
  const recycles = Number(option(args, "recycles", "0"));
  const seed = Number(option(args, "seed", "0"));
  if (!Number.isSafeInteger(rows) || rows < 1) throw new RangeError("rows must be a positive integer");

  // ...the LOCAL bundle, by directory rather than through web/model.js's
  // loadModel: that resolves the monomer family to its remote base, and this
  // machine should not pull 227 MB to run a regression.
  const { MODEL_BUNDLES, loadManifest } = await import("../../src/reference/manifests/index.js");
  const store = await HttpTensorStore.fromManifest(
    MODEL_BUNDLES[family].directory, await loadManifest(family));
  const fixture = AlphaFoldFixture.fromStore(store);
  const loadStart = performance.now();
  // ...the same split web/model.js makes: multimer's embedder runs its template
  // track every recycle, the monomer's is the query-only residual.
  const multimer = family === "multimer";
  const [embedding, template, templateEmbedding, extraStack, mainStack, structure, confidence,
         geometry, featureTables, paeBreaks] = await Promise.all([
    fixture.embeddingWeights(),
    multimer ? Promise.resolve(undefined) : fixture.templateWeights(),
    multimer ? fixture.templateEmbeddingWeights() : Promise.resolve(undefined),
    fixture.extraStackWeights(),
    fixture.mainStackWeights(), fixture.structureWeights(), fixture.confidenceWeights(),
    fixture.geometryTables(), fixture.queryOnlyFeatureTables(),
    fixture.tensor("confidencePaeBreaks"),
  ]);
  const weights = {
    embedding, template, templateEmbedding, extraStack, mainStack, structure,
    lddt: confidence.lddt, pae: confidence.pae, geometry,
  };
  const loadMs = Math.round(performance.now() - loadStart);

  const lines = [">query", sequence];
  for (let row = 1; row < rows; row += 1) {
    lines.push(`>synthetic${row}`);
    lines.push([...sequence].map((code, column) =>
      (column % (row % 11 + 3) === 0 ? "-" : code)).join(""));
  }
  const a3m = `${lines.join("\n")}\n`;

  const chains = chainLengths.length > 0 ? chainLengths : [sequence.length];
  if (chains.reduce((sum, value) => sum + value, 0) !== sequence.length) {
    throw new RangeError(`--chains sums to ${chains.reduce((a, b) => a + b, 0)}, not ${sequence.length}`);
  }
  // The regime web/app.js passes for multimer, verbatim - it is not defaults,
  // and dropping it once ran multimer WEIGHTS on the monomer graph.
  const regime = multimer
    ? { outerProductMeanFirst: true, positionScale: 20, chainAware: true, chainSequences: chains }
    : {};
  const started = performance.now();
  const prediction = await new (multimer ? AlphaFoldUnifiedGpu : AlphaFoldMonomerGpu)(device)
    .predictA3m(
      a3m, weights, featureTables,
      { recycles, randomSeed: seed, maxMsaSequences: rows, maxExtraSequences: rows,
        chainLengths: chains, ...regime },
      paeBreaks,
    );
  const elapsed = Math.round(performance.now() - started);
  const final = prediction.final;
  const length = sequence.length;
  const atom37 = final.structure.atom37;

  // Consecutive alpha carbons, which is atom 1 of the 37.
  //
  // 🔴 THE CHAIN JUNCTIONS ARE SKIPPED, OR THE METRIC MEASURES NOTHING. Two
  // chains are not bonded to each other, so the step across the boundary is
  // whatever the fold placed them at - 18.4 A on the first run of this - and it
  // would sit in `worst` for ever looking like a broken backbone.
  const breaks = new Set();
  let boundary = 0;
  for (const chain of chains.slice(0, -1)) { boundary += chain; breaks.add(boundary - 1); }
  const distances = [];
  for (let residue = 0; residue + 1 < length; residue += 1) {
    if (breaks.has(residue)) continue;
    const a = (residue * 37 + 1) * 3;
    const b = ((residue + 1) * 37 + 1) * 3;
    distances.push(Math.hypot(
      atom37[a] - atom37[b], atom37[a + 1] - atom37[b + 1], atom37[a + 2] - atom37[b + 2]));
  }
  const sorted = [...distances].sort((x, y) => x - y);
  const median = sorted[Math.floor(sorted.length / 2)];
  const worst = distances.reduce((far, value) =>
    Math.abs(value - 3.8) > Math.abs(far - 3.8) ? value : far, 3.8);

  // A checksum over every coordinate, so two trees can be compared in one
  // number before anyone looks at the geometry. Scaled and summed as integers,
  // because a float sum of a million terms is not reproducible in itself.
  let checksum = 0;
  for (let index = 0; index < atom37.length; index += 1) {
    checksum = (checksum + Math.round(atom37[index] * 1000)) | 0;
  }

  const round = (value, places = 4) => Number(value.toFixed(places));
  return {
    sequence: sequence.length > 24 ? `${sequence.slice(0, 24)}...(${length})` : sequence,
    family, chains, length, rows, recycles, seed,
    weightLoadMs: loadMs, elapsedMilliseconds: elapsed,
    meanPlddt: round(final.confidence.meanPlddt, 3),
    ptm: round(final.confidence.ptm, 4),
    ...(final.confidence.iptm === undefined ? {} : { iptm: round(final.confidence.iptm, 4) }),
    caca: { median: round(median, 3), worst: round(worst, 3) },
    checksum,
    // The first and last CA, so a difference has somewhere to be looked at.
    firstCa: [0, 1, 2].map((axis) => round(atom37[1 * 3 + axis], 3)),
    lastCa: [0, 1, 2].map((axis) => round(atom37[((length - 1) * 37 + 1) * 3 + axis], 3)),
  };
}
