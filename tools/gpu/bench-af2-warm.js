/**
 * One AF2 trunk pass, COLD AND WARM, at the shape another port is measured on.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-af2-warm.js --length=59 --rows=128 --extra=256
 *
 * 🔴 THIS EXISTS BECAUSE THE COMPARISON IN docs/AF2.md WAS NOT LIKE FOR LIKE,
 * AND THE DIRECTION OF THE ERROR HID IT. `fold-af2.js` reports ONE pass with
 * every fixed cost in it - featurisation, pipeline compilation, the first
 * touch of every buffer - and martin-steinegger/alphafold2-webgpu's harness
 * reports the MINIMUM OF TWO WARM PASSES after a cold one. Their own numbers
 * show what that is worth: 16.39 cold against 14.21 warm at 825 residues, and
 * 3.82 against 1.26 at 59. So every "LocalFold vs the other port" figure in
 * this repository has been a cold number against a warm one, which flattered
 * THEM - and at short lengths, where a fold is mostly fixed cost, by more than
 * the whole difference between the two ports.
 *
 * This runs three passes and reports both, in their format, so the two can be
 * read off against each other at either temperature. Everything else is
 * `fold-af2.js`: the same synthetic alignment, which is character-for-character
 * the one their harness builds.
 *
 * 🔴 `--passes` MEANS SOMETHING ELSE IN THEIR HARNESS, AND THE MISTAKE IS A
 * SILENT 3x AGAINST THEM. Here it is the number of TIMED REPETITIONS: three,
 * one cold and two warm. In `bench/bench825.js` the repetition count is
 * hardcoded at three and `--passes` is read as `recycles: passes - 1` - the
 * number of TRUNK PASSES INSIDE ONE PREDICTION. So `--passes=3` on both sides
 * times the same three repetitions here and three RECYCLES there, and their
 * 59-residue figure goes 1.24 s to 3.65 while ours does not move. Run theirs
 * with no `--passes` at all, which is 0 recycles and the three repetitions its
 * loop always does; that is what this tool's default matches.
 */
import { AlphaFoldMonomerGpu } from "../../src/model/monomer.js";
import { makeA3mFeatures } from "../../src/input/a3m-features.js";
import { AlphaFoldFixture } from "../../src/reference/alphafold-fixture.js";
import { HttpTensorStore } from "../../src/reference/http-tensor-store.js";
import { DEFAULT_TUNING, setDeviceTuning } from "../../src/runtime/device-profile.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const QUERY = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";

export async function main(device, args) {
  const length = Number(option(args, "length", "825"));
  const rows = Number(option(args, "rows", "512"));
  const extraRows = Number(option(args, "extra", "1024"));
  const passes = Number(option(args, "passes", "3"));
  const tune = option(args, "tune", "");
  for (const pair of tune.split(",").filter(Boolean)) {
    const at = pair.indexOf("=");
    if (at < 0) throw new Error(`--tune wants key=value, got ${pair}`);
    const key = pair.slice(0, at);
    if (!(key in DEFAULT_TUNING)) throw new Error(`--tune names ${key}, not a tuning knob`);
    const raw = pair.slice(at + 1);
    let value;
    try { value = JSON.parse(raw); } catch { value = raw; }
    setDeviceTuning(device, { [key]: value });
  }

  const sequence = QUERY.repeat(Math.ceil(length / QUERY.length)).slice(0, length);
  const { MODEL_BUNDLES, loadManifest } = await import("../../src/reference/manifests/index.js");
  const store = await HttpTensorStore.fromManifest(
    MODEL_BUNDLES.monomer.directory, await loadManifest("monomer"));
  const fixture = AlphaFoldFixture.fromStore(store);
  const [embedding, template, extraStack, mainStack, structure, confidence,
         geometry, featureTables, paeBreaks] = await Promise.all([
    fixture.embeddingWeights(), fixture.templateWeights(), fixture.extraStackWeights(),
    fixture.mainStackWeights(), fixture.structureWeights(), fixture.confidenceWeights(),
    fixture.geometryTables(), fixture.queryOnlyFeatureTables(),
    fixture.tensor("confidencePaeBreaks"),
  ]);
  const weights = {
    embedding, template, extraStack, mainStack, structure,
    lddt: confidence.lddt, pae: confidence.pae, geometry,
  };

  // Their alignment, verbatim - the gap pattern below is the one their bench
  // writes, so the two ports see the same rows and not merely the same count.
  const lines = [">query", sequence];
  for (let row = 1; row < rows + extraRows; row += 1) {
    lines.push(`>synthetic${row}`);
    lines.push([...sequence].map((code, column) =>
      (column % (row % 11 + 3) === 0 ? "-" : code)).join(""));
  }
  const a3m = `${lines.join("\n")}\n`;

  // 🔴 THE FEATURES ARE BUILT ONCE, OUTSIDE THE CLOCK, BECAUSE THEIRS ARE.
  // `predictA3m` deliberately times the alignment prep - see the note on it,
  // and docs/AF2.md has had to rewrite that prep once already - but their
  // harness calls `makeA3mFeatures` before the loop and times `predict` alone.
  // Leaving ours inside would put 0.56 s of main-thread JavaScript at 825
  // residues on one side of the comparison and not the other. It is reported
  // separately instead, so neither number goes missing.
  const featureStart = performance.now();
  const features = makeA3mFeatures(a3m, featureTables,
    { recycles: 0, randomSeed: 0, maxMsaSequences: rows, maxExtraSequences: extraRows });
  const featureMs = performance.now() - featureStart;

  const runner = new AlphaFoldMonomerGpu(device);
  const times = [];
  let prediction;
  for (let pass = 0; pass < passes; pass += 1) {
    const started = performance.now();
    prediction = await runner.predict(features, weights, paeBreaks);
    times.push(performance.now() - started);
  }
  const round = (v) => Number((v / 1000).toFixed(3));
  const atom37 = prediction.final.structure.atom37;
  let checksum = 0;
  for (let i = 0; i < atom37.length; i += 1) checksum = (checksum + Math.round(atom37[i] * 1000)) | 0;
  return {
    length, clustered: rows, extra: extraRows, passes,
    featureSeconds: round(featureMs),
    coldSeconds: round(times[0]),
    warmSeconds: round(Math.min(...times.slice(1))),
    allSeconds: times.map(round),
    meanPlddt: Number(prediction.final.confidence.meanPlddt.toFixed(2)),
    checksum,
  };
}
