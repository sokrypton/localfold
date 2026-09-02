/**
 * AlphaFold 2's evoformer block against AlphaFold 3's pairformer block.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-blocks.js
 *
 * This is tools/benchmark-a3m-stack.js's measurement moved onto the Chrome
 * lane - the Dawn one does not load on this OS, see tools/gpu-chrome.mjs - and
 * put beside the AF3 stack it is meant to be compared with.
 *
 * 🔴 THE SAME LENGTH OR IT MEANS NOTHING. Both stacks are at least quadratic in
 * the token count, so a comparison at different n is a comparison of two
 * lengths. AF2's stack fixture is 59 residues, so AF3 is run at 59 tokens.
 *
 * 🔴 AND THEY ARE NOT THE SAME BLOCK, which is the whole reason to measure.
 * AF2's carries the MSA - row attention with a pair bias, column attention, an
 * MSA transition and the outer product mean - on top of the pair track, and its
 * cost rises with MSA depth. AF3's pairformer has no MSA in it at all: two
 * triangle multiplications, two triangle ("grid") attentions, a pair
 * transition, and the single track. So AF3's block SHOULD be the cheaper one.
 *
 * WHAT IT SAYS, at 59 tokens: AF2 11.91 ms a block, AF3 15.53 ms - so AF3's is
 * 1.30x AF2's for strictly less work. Measured separately, AF3's single track
 * is free (747 ms against 751 with it skipped, which is noise), so essentially
 * all of that 15.53 ms is the pair track, while AF2's 11.91 ms is a pair track
 * PLUS the whole MSA stack. The gap in the pair track alone is therefore wider
 * than 1.30x, and it is where any further work belongs.
 */
import { EvoformerStackGpu } from "../../src/evoformer/stack.js";
import { AlphaFoldFixture } from "../../src/reference/alphafold-fixture.js";
import { HttpTensorStore } from "../../src/reference/http-tensor-store.js";
import { featuriseProtein } from "../../src/af3/featurise.js";
import { Af3TrunkGpu } from "../../src/af3/trunk-webgpu.js";
import { buildTargetFeat, DIALECT } from "../../src/af3/fold.js";
import { openAf3Store, trunkWeights } from "../../src/af3/weights.js";
import { targetFeatureWeights } from "../../src/af3/diffusion-weights.js";

function option(args, name, fallback) {
  const prefix = `--${name}=`;
  return args.find((argument) => argument.startsWith(prefix))?.slice(prefix.length) ?? fallback;
}

const LENGTH = 59;

import { profileDevice } from "./profile.js";

export async function main(device, args) {
  // --profile times every labelled compute pass; see tools/gpu/profile.js. The
  // AF2 and AF3 labels are distinct, so one run splits both stacks.
  const profile = args.includes("--profile") ? profileDevice(device) : null;
  for (const flag of ["TRICONTRACT","TRIPROJECT","TRIOUT","TRINORM","GNORM","GBIAS","GPROJ","GATT","GOUT","TRANS","ADD","SINGLE"]) {
    globalThis["__SKIP_" + flag] = args.includes("--skip-" + flag.toLowerCase());
  }
  const repeats = Number(option(args, "repeats", "2"));
  // 🔴 REAL WEIGHTS, SYNTHETIC ACTIVATIONS, AND THAT IS SOUND FOR A TIMING
  // MEASUREMENT. The stack fixture's own activations are not checked in - only
  // its manifest is - and the run time of these kernels depends on the SHAPES,
  // not the values: every dispatch is the same size whatever the numbers are,
  // and nothing here branches on data. The weights are the exported monomer
  // bundle, the same ones the page folds with, because their layout and
  // precision do affect the memory traffic.
  //
  // It is not a correctness check and cannot be read as one; the checkers under
  // tools/gpu do that.
  const { MODEL_BUNDLES, loadManifest } = await import("../../src/reference/manifests/index.js");
  const fixture = AlphaFoldFixture.fromStore(await HttpTensorStore.fromManifest(
    MODEL_BUNDLES.monomer.directory, await loadManifest("monomer")));
  // 🔴 THE QUERY-ONLY STACK, which is the fair one to compare. Both sides are
  // then folding a single sequence: this fixture's MSA is five rows of the
  // query's own features rather than an alignment, and AF3's pairformer never
  // sees an MSA at all.
  const blockWeights = await fixture.mainStackWeights();
  // The shapes a query-only 59-residue fold actually has - five MSA rows, from
  // the fixture's manifest.
  const sequences = 5;
  const cM = 256;
  const noise = (count, seed) => {
    const values = new Float32Array(count);
    let state = seed >>> 0;
    for (let index = 0; index < count; index += 1) {
      state = (state * 1664525 + 1013904223) >>> 0;
      values[index] = (state / 4294967296) - 0.5;
    }
    return values;
  };
  const stackInput = {
    msa: noise(sequences * LENGTH * cM, 1),
    pair: noise(LENGTH * LENGTH * 128, 2),
    msaMask: new Float32Array(sequences * LENGTH).fill(1),
    pairMask: new Float32Array(LENGTH * LENGTH).fill(1),
    sequences, length: LENGTH, cM, cZ: 128, cOuter: 32, triangleHidden: 128, blockWeights,
  };

  const af3Store = await openAf3Store(option(args, "model", "/model-af3-int5/manifest.json"));
  const af3Weights = await trunkWeights(af3Store, 48, 4);
  const batch = featuriseProtein("A".repeat(LENGTH));
  const targetFeat = await buildTargetFeat(batch, await targetFeatureWeights(af3Store), device);
  const af3Input = {
    tokens: LENGTH, sequences: 1, templates: 4, targetFeat, features: batch.features,
    msaRows: batch.msa.subarray(0, LENGTH),
    deletionMatrix: batch.deletionMatrix.subarray(0, LENGTH),
    msaMask: batch.msaMask.subarray(0, LENGTH),
    pairMask: new Float32Array(LENGTH * LENGTH).fill(1), seqMask: batch.seqMask,
    previousPair: new Float32Array(LENGTH * LENGTH * 128),
    previousSingle: new Float32Array(LENGTH * 384),
  };

  // 🔴 INTERLEAVED, IN ONE PROCESS, AND REPORTED AS A MEDIAN. This machine
  // drifts by up to 3.2x between runs, so two numbers from two invocations
  // cannot be compared at all - a difference smaller than the drift is noise
  // wearing a result's clothes. A and B alternate inside one process and the
  // medians are compared.
  //
  // The first pass also compiles every pipeline, so it is discarded.
  const af2 = [];
  const af3 = [];
  const msa = [];
  for (let attempt = 0; attempt <= repeats; attempt += 1) {
    const started = performance.now();
    await new EvoformerStackGpu(device).run(stackInput);
    const timings = {};
    await new Af3TrunkGpu(device).run(af3Input, af3Weights, DIALECT,
      { onStage: (name, ms) => { timings[name] = ms; } });
    if (attempt === 0) continue;
    af2.push(performance.now() - started - timings.pairformer - timings["msa-stack"]
      - timings.embedder - timings.template - timings.distogram);
    af3.push(timings.pairformer);
    msa.push(timings["msa-stack"]);
  }
  const median = (values) => {
    const sorted = [...values].sort((x, y) => x - y);
    return sorted[Math.floor(sorted.length / 2)];
  };
  const best = median;

  const af2Block = best(af2) / blockWeights.length;
  const af3Block = best(af3) / 48;
  console.log(`${LENGTH} tokens, best of ${repeats}`);
  console.log(`AF2 evoformer   ${best(af2).toFixed(0)} ms / ${blockWeights.length} blocks`
    + `   ${af2Block.toFixed(2)} ms per block   (MSA ${sequences} rows)`);
  console.log(`AF3 pairformer  ${best(af3).toFixed(0)} ms / 48 blocks`
    + `   ${af3Block.toFixed(2)} ms per block   (no MSA)`);
  console.log(`AF3 msa-stack   ${best(msa).toFixed(0)} ms / 4 blocks`
    + `   ${(best(msa) / 4).toFixed(2)} ms per block   (MSA 1 row)`);
  console.log(`AF3 pairformer block is ${(af3Block / af2Block).toFixed(2)}x AF2's evoformer block`);
  const gpuPasses = profile === null ? undefined : await profile.report();
  profile?.restore();
  return {
    ...(gpuPasses === undefined ? {} : { gpuPasses }), tokens: LENGTH, af2Block, af3Block, msaRows: sequences };
}
