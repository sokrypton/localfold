/**
 * WHICH COLUMNS of OpenDDE's `target_feat` disagree with af3-any-model?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-opendde-target-feat.js \
 *       --model=/model-opendde-full-f32/manifest.json \
 *       --oracle=/oracle-dumps/af3-oracle-trunk-opendde.json
 *
 * 🔴 THE FIRST OPENDDE TRUNK ORACLE EVER RUN DISAGREES AT THE FIRST SEAM -
 * `target_feat` 9.65e-2 at f32, and every stage after it inherits that. A whole
 * -tensor relRMS names no cause: `target_feat` is 447 columns from several
 * different sources, and a wrong SLICE and a uniformly wrong tensor read the
 * same. This reports the residual per column so the slice can be named.
 */
import { loadTrunkOracle } from "./trunk-oracle.js";
import { af3BatchFromA3m } from "../../src/af3/featurise/batch.js";
import { openAf3Store, trunkWeights } from "../../src/af3/weights/weights.js";
import { targetFeatureWeights, atomReference } from "../../src/af3/weights/diffusion-weights.js";
import { foldBatch } from "../../src/af3/fold.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const sequence = option(args, "sequence",
    "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE");
  const dump = await loadTrunkOracle(option(args, "oracle",
    "/oracle-dumps/af3-oracle-trunk-opendde.json"));
  const store = await openAf3Store(option(args, "model",
    "/model-opendde-full-f32/manifest.json"));
  store.prefetch();
  const weights = {
    trunk: await trunkWeights(store),
    atomReference: await atomReference(store),
    targetFeat: await targetFeatureWeights(store),
  };
  const { batch } = af3BatchFromA3m(sequence, null, {});

  let ours;
  await foldBatch(device, batch, weights, {
    mode: "diffusion", steps: 0, recycles: 0, seed: 0,
    onStage: (name, detail) => {
      if (name === "target-feat") ours = detail?.targetFeat;
    },
  }).catch((error) => {
    // Stopping after target_feat is the point; a head that never runs is not a
    // failure here, and re-folding the whole model to read one stage is 6 s.
    if (ours === undefined) throw error;
  });
  if (ours === undefined) throw new Error("the fold never announced target-feat");

  if (args.includes("--raw")) {
    // 30k floats: small enough to hand back whole, and analysing it offline
    // beats another six-second round trip through the browser per question.
    return { raw: Array.from(ours, (v) => Number(v.toFixed(6))) };
  }
  const entry = dump.stages.target_feat;
  const [tokens, width] = entry.shape;
  const expected = Float32Array.from(entry.data);
  if (ours.length !== expected.length) {
    return { tokens, width, oursLength: ours.length, expectedLength: expected.length,
             verdict: "LENGTH MISMATCH - the widths differ, which is the finding" };
  }
  // Per COLUMN, because target_feat is a concatenation and a wrong slice is the
  // shape this bug most likely has.
  const columns = [];
  for (let c = 0; c < width; c += 1) {
    let error = 0, scale = 0;
    for (let t = 0; t < tokens; t += 1) {
      const d = ours[t * width + c] - expected[t * width + c];
      error += d * d; scale += expected[t * width + c] ** 2;
    }
    columns.push({ column: c, relRms: Math.sqrt(error / Math.max(scale, 1e-30)),
                   nativeRms: Math.sqrt(scale / tokens) });
  }
  const bad = columns.filter((c) => c.relRms > 1e-3);
  // Contiguous runs of bad columns, which is what names a slice.
  const runs = [];
  for (const c of bad) {
    const last = runs[runs.length - 1];
    if (last !== undefined && c.column === last.to + 1) last.to = c.column;
    else runs.push({ from: c.column, to: c.column });
  }
  return {
    tokens, width,
    badColumns: bad.length,
    goodColumns: width - bad.length,
    runsOfDisagreement: runs.map((r) => ({ ...r, count: r.to - r.from + 1 })),
    worst: columns.reduce((w, c) => (c.relRms > w.relRms ? c : w), columns[0]),
    // A sample either side of each boundary, because "columns 384..446 are
    // wrong" and "every column is a little wrong" are different bugs.
    // Every disagreeing column, with the first token's two values, because a
    // column that is ours=0 native=1 is a MISSING term and one that is scaled
    // is a wrong one - and the whole-column relRMS cannot tell them apart.
    bad: bad.map((c) => ({
      column: c.column,
      relRms: Number(c.relRms.toExponential(2)),
      nativeRms: Number(c.nativeRms.toFixed(5)),
      firstTokens: [0, 1, 2, 3].map((t) => ({
        ours: Number(ours[t * width + c.column].toFixed(5)),
        native: Number(expected[t * width + c.column].toFixed(5)),
      })),
    })),
    // ...and per TOKEN, because a residue-type-dependent bug and a
    // column-slice bug are the two shapes this can have and they need
    // different reading.
    tokens_: (() => {
      const rows = [];
      for (let t = 0; t < tokens; t += 1) {
        let error = 0, scale = 0;
        for (let c = 0; c < width; c += 1) {
          const d = ours[t * width + c] - expected[t * width + c];
          error += d * d; scale += expected[t * width + c] ** 2;
        }
        rows.push({ token: t, letter: sequence[t] ?? "?",
                    relRms: Number(Math.sqrt(error / Math.max(scale, 1e-30)).toExponential(2)) });
      }
      return rows.filter((r) => Number(r.relRms) > 1e-4);
    })(),
    sample: [0, 1, 100, 200, 300, 380, 384, 400, 446]
      .filter((c) => c < width)
      .map((c) => ({ column: c, relRms: Number(columns[c].relRms.toExponential(2)),
                     nativeRms: Number(columns[c].nativeRms.toFixed(5)) })),
  };
}
