/**
 * How long a fold takes, as a function of what was asked for.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-runtime.js
 *
 * 🔴 THIS EXISTS TO CALIBRATE THE PROGRESS BAR, WHICH WAS GUESSING. web/af3-model.js
 * split the bar between featurisation, trunk and sampler using constants
 * measured once on a 59-residue chain - a trunk pass was "3.7 s" and a denoiser
 * call "0.85", with no dependence on LENGTH at all. The trunk grows faster than
 * the sampler does, so the split is wrong at both ends: on a long chain the bar
 * races through the trunk band and then sits, and on a short one it crawls.
 *
 * 🔴 WHAT IS FITTED IS A SHAPE, NOT A SPEED. Absolute milliseconds here are
 * this machine's; a phone is five to ten times slower and would make every
 * constant a lie. What travels is the RATIO between the pieces of work, which
 * is a property of the arithmetic rather than of the device, and
 * src/runtime/cost-model.js turns the fit below into relative units that the
 * page scales by what it actually observes.
 *
 * The sweeps are a cross rather than a grid, which is what separability buys:
 * the pairformer does not read the alignment, so its cost is a function of
 * tokens alone; the MSA stack is linear in rows at fixed tokens. Both are
 * checked here rather than assumed.
 *
 * 🔴 ONE RUN OF THIS CANNOT FIT THE MODEL, AND THE REASON IS THE DRIFT THIS
 * REPO WARNS ABOUT EVERYWHERE ELSE. Running everything in one process was meant
 * to defeat it - and does, for two arms measured back to back. It does not for
 * a sweep that takes two minutes, because the shapes run in SEQUENCE and the
 * drift accumulates across them. Two runs of this file, same tree, same
 * machine, on the identical shapes:
 *
 *     AF3 trunk    59 tokens  +0.7%   128 -22.9%   192 -33.2%   256 -37.7%
 *     AF3 denoiser 59         -5.7%   128 -14.4%   192 -21.8%   256 -30.3%
 *     AF2 stack    59        -39.6%   128 -33.3%   192 -20.6%   256 +10.5%
 *
 * AF3 got uniformly faster and AF2 uniformly slower between the two, which is
 * not a property of either model. A fit over one of these columns puts a 3x
 * error into the exponent - the L^3 term came out 3x apart between the runs.
 *
 * So: to refit, INTERLEAVE the shapes the way tools/gpu/bench-ab.js interleaves
 * its arms, and take medians over several rounds. Until then the constants in
 * src/runtime/cost-model.js stay where they are, and the check that they are
 * still good enough is tools/gpu/probe-progress-bar.js, which measures the bar
 * against the clock rather than the model against a bench.
 */
import { EvoformerStackGpu } from "../../src/evoformer/stack.js";
import { AlphaFoldFixture } from "../../src/reference/alphafold-fixture.js";
import { HttpTensorStore } from "../../src/reference/http-tensor-store.js";
import { MODEL_BUNDLES, loadManifest } from "../../src/reference/manifests/index.js";
import { featuriseProtein } from "../../src/af3/featurise.js";
import { buildTargetFeat, DIALECT } from "../../src/af3/fold.js";
import { Af3TrunkGpu } from "../../src/af3/trunk-webgpu.js";
import { Af3DiffusionHeadGpu } from "../../src/af3/diffusion-head-webgpu.js";
import { perAtomConditioning } from "../../src/af3/atom-conditioning-reference.js";
import { openAf3Store, trunkWeights } from "../../src/af3/weights.js";
import { targetFeatureWeights, diffusionWeights, atomReference }
  from "../../src/af3/diffusion-weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};
const numbers = (text) => text.split(",").map(Number);

const ALPHABET = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";
const sequenceOf = (tokens) => Array.from({ length: tokens },
  (_, index) => ALPHABET[index % ALPHABET.length]).join("");

/** A deterministic filler, so shapes are comparable between runs. */
function noiseFrom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return (state / 4294967296) * 2 - 1;
  };
}

function synthAlignment(batch, tokens, rows) {
  const msa = new Int32Array(rows * tokens);
  for (let row = 0; row < rows; row += 1) {
    for (let token = 0; token < tokens; token += 1) {
      msa[row * tokens + token] = (batch.msa[token] + row) % 20;
    }
  }
  return {
    msaRows: msa,
    deletionMatrix: new Float32Array(rows * tokens),
    msaMask: new Float32Array(rows * tokens).fill(1),
  };
}

export async function main(device, args) {
  const tokenSweep = numbers(option(args, "tokens", "59,128,192,256"));
  const rowSweep = numbers(option(args, "msa", "1,128,512"));
  const rowsForTokenSweep = Number(option(args, "rows", "32"));
  const tokensForRowSweep = Number(option(args, "row-tokens", "128"));
  const blocks = Number(option(args, "blocks", "48"));

  const store = await openAf3Store(option(args, "model", "/model-af3-int5/manifest.json"));
  const weights = {
    trunk: await trunkWeights(store, blocks, 4),
    targetFeat: await targetFeatureWeights(store),
    diffusion: await diffusionWeights(store),
    reference: await atomReference(store),
  };

  const trunkGpu = new Af3TrunkGpu(device);
  const head = new Af3DiffusionHeadGpu(device);

  /**
   * One trunk pass at a shape, twice, reporting the second.
   *
   * 🔴 THE FIRST PASS AT ANY SHAPE COMPILES ITS PIPELINES, and the shaders are
   * generated per token count, so every point on the sweep pays it again. It is
   * seconds at the larger shapes and would swamp what is being measured.
   */
  const trunkAt = async (tokens, rows) => {
    const sequence = sequenceOf(tokens);
    const batch = featuriseProtein(sequence, {});
    const targetFeat = await buildTargetFeat(batch, weights.targetFeat, device);
    const alignment = synthAlignment(batch, tokens, rows);
    const seqMask = batch.seqMask;
    const pairMask = new Float32Array(tokens * tokens);
    for (let i = 0; i < tokens; i += 1) {
      for (let j = 0; j < tokens; j += 1) pairMask[i * tokens + j] = seqMask[i] * seqMask[j];
    }
    const input = {
      tokens, sequences: rows, templates: 4, targetFeat, features: batch.features,
      ...alignment, bondMatrix: batch.bondMatrix, pairMask, seqMask,
      previousPair: new Float32Array(tokens * tokens * 128),
      previousSingle: new Float32Array(tokens * 384),
    };
    let last = null;
    for (let pass = 0; pass < 2; pass += 1) {
      const timings = {};
      const started = performance.now();
      await trunkGpu.run(input, weights.trunk, DIALECT,
        { onStage: (name, ms) => { timings[name] = ms; } });
      last = { whole: performance.now() - started, ...timings };
    }
    return {
      tokens, rows,
      whole: Math.round(last.whole),
      ...Object.fromEntries(Object.entries(last).filter(([k]) => k !== "whole")
        .map(([k, v]) => [k, Math.round(v)])),
    };
  };

  /** One denoiser call at a token count, median of three after a warm-up. */
  const denoiserAt = async (tokens) => {
    const batch = featuriseProtein(sequenceOf(tokens), {});
    const { dense } = batch;
    const noise = noiseFrom(11 + tokens);
    const fill = (length) => {
      const out = new Float32Array(length);
      for (let index = 0; index < length; index += 1) out[index] = noise();
      return out;
    };
    const input = {
      shape: batch.shape,
      conditioning: perAtomConditioning({
        positions: batch.refPos, mask: batch.refMask, element: batch.refElement,
        charge: batch.refCharge, atomNameChars: batch.refAtomNameChars,
      }, tokens, dense, weights.reference),
      atomMask: batch.predDenseAtomMask, seqMask: batch.seqMask, features: batch.features,
      targetFeat: fill(tokens * 447),
      refPos: batch.refPos, refSpaceUid: batch.refSpaceUid,
      tokenAtomsToQueries: batch.tokenAtomsToQueries, queriesToKeys: batch.queriesToKeys,
      queriesToTokenAtoms: batch.queriesToTokenAtoms,
      tokensToQueries: batch.tokensToQueries, tokensToKeys: batch.tokensToKeys,
      trunkSingle: fill(tokens * 384), trunkPair: fill(tokens * tokens * 128),
      positionsNoisy: fill(tokens * dense * 3),
      noiseLevel: 16,
    };
    const samples = [];
    for (let call = 0; call < 4; call += 1) {
      const started = performance.now();
      await head.run(input, weights.diffusion);
      if (call > 0) samples.push(performance.now() - started);
    }
    samples.sort((a, b) => a - b);
    return { tokens, atoms: tokens * dense, whole: Math.round(samples[1]) };
  };

  /**
   * AF2's whole evoformer stack at a shape, timed as a wall clock.
   *
   * 🔴 NOT THROUGH profile-af2-block.js, WHICH COSTS 20% AND MEASURES ONE
   * BLOCK. The timestamp profiler is the right tool for asking where a block's
   * time goes; what a progress bar needs is what a STACK costs, and running
   * every shape in this one process is what makes the shapes comparable - this
   * machine drifts by up to 3.2x between processes.
   *
   * Synthetic activations, real weights: every dispatch is the same size
   * whatever the numbers are, and nothing branches on data, but the weights'
   * layout and precision do affect memory traffic.
   */
  const af2At = async (blockWeights, length, sequences) => {
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
    const input = {
      msa: noise(sequences * length * cM, 1),
      pair: noise(length * length * 128, 2),
      msaMask: new Float32Array(sequences * length).fill(1),
      pairMask: new Float32Array(length * length).fill(1),
      sequences, length, cM, cZ: 128, cOuter: 32, triangleHidden: 128, blockWeights,
    };
    const stack = new EvoformerStackGpu(device);
    let last = 0;
    // ...twice, because the first run at a shape compiles its pipelines.
    for (let pass = 0; pass < 2; pass += 1) {
      const started = performance.now();
      await stack.run(input);
      last = performance.now() - started;
    }
    return { length, sequences, blocks: blockWeights.length, whole: Math.round(last) };
  };

  const trunkByTokens = [];
  const trunkByRows = [];
  const denoiser = [];
  if (!args.includes("--no-af3")) {
    for (const tokens of tokenSweep) trunkByTokens.push(await trunkAt(tokens, rowsForTokenSweep));
    for (const rows of rowSweep) trunkByRows.push(await trunkAt(tokensForRowSweep, rows));
    for (const tokens of tokenSweep) denoiser.push(await denoiserAt(tokens));
  }

  // AF2's monomer stack, the other model the page offers.
  const af2ByTokens = [];
  const af2ByRows = [];
  if (!args.includes("--no-af2")) {
    const fixture = AlphaFoldFixture.fromStore(await HttpTensorStore.fromManifest(
      MODEL_BUNDLES.monomer.directory, await loadManifest("monomer")));
    const blockWeights = await fixture.mainStackWeights();
    for (const length of tokenSweep) {
      af2ByTokens.push(await af2At(blockWeights, length, rowsForTokenSweep));
    }
    for (const sequences of rowSweep) {
      af2ByRows.push(await af2At(blockWeights, tokensForRowSweep, sequences));
    }
  }

  return {
    machine: "whatever ran this; the SHAPE is what transfers, not the milliseconds",
    rowsForTokenSweep, tokensForRowSweep, blocks,
    trunkByTokens, trunkByRows, denoiser, af2ByTokens, af2ByRows,
  };
}
