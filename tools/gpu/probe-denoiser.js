/**
 * Ask the diffusion head what it predicts from NOTHING.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-denoiser.js \
 *       --sequence=GWSTELEK... --model=/model-af3-int5/manifest.json
 *
 * THE QUESTION. A 200-step fold's prediction is already 1.17 A from the crystal
 * structure at step 4, and the animation looks like a protein from its first
 * frame. So how much of AF3's diffusion head is DIFFUSION, and how much of it is
 * a structure module wearing a denoiser's clothes? If the answer barely depends
 * on the noisy input, then the trunk decides the fold and the sampler only
 * polishes it.
 *
 * THE EXPERIMENT. Run the head ONCE, with positionsNoisy set to exactly zero -
 * every atom at the origin, a black hole, no noise and no information about
 * where anything is - and score what comes out. `--start` also allows `noise`
 * (the usual Gaussian at that sigma) as a control, because a result from zeros
 * means nothing without knowing what the same single call gives normally.
 *
 * 🔴 THE INPUT BARELY REACHES THE OUTPUT AT HIGH SIGMA, WHICH IS THE POINT. AF3
 * returns skip * positionsNoisy + out * update, with
 * skip = sigma_d^2 / (sigma^2 + sigma_d^2). At sigma = 2560 that is 3.9e-5 and
 * `out` is 16, so the returned coordinates are essentially 16 x update whatever
 * went in. Zeros in is therefore not a trick question - it is asking the network
 * to place 574 atoms knowing only the trunk, the sequence and the chemistry.
 *
 * 🔴 AND THE SIGMA STILL MATTERS EVEN WITH A ZERO INPUT. It is embedded through
 * the Fourier noise embedding into the conditioning, so the network is told
 * which rung of the schedule it is on. --sigma sets it; the default is the top
 * of the schedule, which is the rung a sampler starts from.
 *
 * --iterations RUNS IT AS A STRUCTURE MODULE. Feed the head's own output back in
 * as its input, repeatedly, at a fixed sigma: no noise, no schedule, no
 * sampling - just a fixed-point iteration, which is what AF2's structure module
 * does with its recycles.
 *
 * 🔴 ITERATING AT HIGH SIGMA IS A NO-OP, AND THAT IS ARITHMETIC RATHER THAN A
 * BUG. skip is 3.9e-5 at sigma 2560, so the input is discarded and every
 * iteration returns the same coordinates - the network is deterministic. The
 * feedback only means anything where skip is large: at sigma 16 it is 0.5, at
 * sigma 4 it is 0.94, and there the call IS a residual update on the structure
 * it was handed.
 *
 * 🔴 AND IT IS MUCH WORSE THAN THE SINGLE HIGH-SIGMA CALL. Measured from zeros:
 *
 *     sigma 2560, one call      1.39 A, gyration 11.1   (the crystal is 11.1)
 *     sigma 16, eight rounds    9.12 -> 9.72 A, getting WORSE
 *     sigma 4,  eight rounds    8.67 -> 6.88 A, gyration only 7.3
 *
 * SIGMA IS NOT A FREE PARAMETER: it is a CLAIM ABOUT THE INPUT, told to the
 * network through the Fourier noise embedding. At sigma 4 the claim is "this
 * structure is nearly right, correct it slightly" - and a black hole is wildly
 * out of distribution for that claim, so the network makes small corrections to
 * garbage forever. At sigma 2560 the claim is "this input is noise, ignore it",
 * which zeros satisfies perfectly, and the network predicts from the trunk
 * instead. That single call is the structure module; the loop is not.
 */
import { featuriseProtein } from "../../src/af3/featurise.js";
import { buildTargetFeat, backboneGeometry, toPdb, normalFrom, DIALECT }
  from "../../src/af3/fold.js";
import { perAtomConditioning } from "../../src/af3/atom-conditioning-reference.js";
import { Af3TrunkGpu } from "../../src/af3/trunk-webgpu.js";
import { Af3DiffusionHeadGpu, scalings } from "../../src/af3/diffusion-head-webgpu.js";
import { noiseLevels } from "../../src/af3/diffusion-sampler-reference.js";
import { openAf3Store, trunkWeights } from "../../src/af3/weights.js";
import { diffusionWeights, atomReference, targetFeatureWeights }
  from "../../src/af3/diffusion-weights.js";

function option(args, name, fallback) {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
}

export async function main(device, args) {
  const sequence = option(args, "sequence",
    "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE");
  const start = option(args, "start", "zeros");
  const levels = noiseLevels(200, {});
  const sigma = Number(option(args, "sigma", String(levels[0])));

  const batch = featuriseProtein(sequence);
  const { tokens, dense } = batch;
  const store = await openAf3Store(option(args, "model", "/model-af3-int5/manifest.json"));
  const weights = {
    trunk: await trunkWeights(store, 48, 4),
    diffusion: await diffusionWeights(store),
    atomReference: await atomReference(store),
    targetFeat: await targetFeatureWeights(store),
  };

  const targetFeat = await buildTargetFeat(batch, weights.targetFeat, device);
  const seqMask = batch.seqMask;
  const pairMask = new Float32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) pairMask[i * tokens + j] = seqMask[i] * seqMask[j];
  }
  const trunk = await new Af3TrunkGpu(device).run({
    tokens, sequences: 1, templates: 4, targetFeat, features: batch.features,
    msaRows: batch.msa.subarray(0, tokens),
    deletionMatrix: batch.deletionMatrix.subarray(0, tokens),
    msaMask: batch.msaMask.subarray(0, tokens),
    pairMask, seqMask,
    previousPair: new Float32Array(tokens * tokens * 128),
    previousSingle: new Float32Array(tokens * 384),
  }, weights.trunk, DIALECT);

  const conditioning = perAtomConditioning({
    positions: batch.refPos, mask: batch.refMask,
    element: batch.refElement, charge: batch.refCharge,
    atomNameChars: batch.refAtomNameChars,
  }, tokens, dense, weights.atomReference);

  const atoms = tokens * dense;
  const positionsNoisy = new Float32Array(atoms * 3);
  if (start === "noise") {
    const normal = normalFrom(Number(option(args, "seed", "20260831")));
    for (let index = 0; index < positionsNoisy.length; index += 1) {
      positionsNoisy[index] = normal() * sigma;
    }
  }

  const scale = scalings(sigma);
  console.log(`${sequence.length} residues, one denoiser call, sigma ${sigma.toFixed(1)}`);
  console.log(`start: ${start === "zeros" ? "every atom at the origin" : "Gaussian noise"}`);
  console.log(`the blend keeps ${(scale.skip * 100).toExponential(2)}% of the input`
    + ` and ${scale.out.toFixed(2)}x the network's update`);

  const iterations = Number(option(args, "iterations", "1"));
  // 🔴 A RAMP IS NOT `iterations` AT ANOTHER SIGMA. Holding sigma fixed tells
  // the network the same thing every round; walking it down tells the truth
  // about an input that is getting better, which is what it was trained on.
  const ramp = Number(option(args, "ramp", "0"));
  const schedule = ramp > 0
    ? Array.from(noiseLevels(ramp, {}).slice(0, ramp))
    : new Array(iterations).fill(sigma);
  const head = new Af3DiffusionHeadGpu(device);
  const headInput = {
    shape: batch.shape, conditioning, atomMask: batch.predDenseAtomMask, seqMask,
    features: batch.features, targetFeat,
    refPos: batch.refPos, refSpaceUid: batch.refSpaceUid,
    tokenAtomsToQueries: batch.tokenAtomsToQueries,
    queriesToKeys: batch.queriesToKeys,
    queriesToTokenAtoms: batch.queriesToTokenAtoms,
    tokensToQueries: batch.tokensToQueries,
    tokensToKeys: batch.tokensToKeys,
    trunkSingle: trunk.single, trunkPair: trunk.pair,
    noiseLevel: sigma, positionsNoisy,
  };

  const pdbs = [];
  let geometry = null;
  let previous = null;
  for (let round = 1; round <= schedule.length; round += 1) {
    headInput.noiseLevel = schedule[round - 1];
    const denoised = await head.run(headInput, weights.diffusion);
    // How far the structure moved, in a frame-invariant way. There is no
    // randomAugmentation here so the frames DO line up, but a distance-matrix
    // metric is the honest one for "has it converged" either way.
    let moved = NaN;
    if (previous !== null) {
      let sum = 0;
      for (let i = 0; i < previous.length; i += 1) {
        sum += (denoised.positions[i] - previous[i]) ** 2;
      }
      moved = Math.sqrt(sum / (previous.length / 3));
    }
    previous = Float32Array.from(denoised.positions);
    headInput.positionsNoisy = denoised.positions;
    geometry = backboneGeometry(batch, denoised.positions);
    pdbs.push(toPdb(batch, denoised.positions, null));
    console.log(`round ${String(round).padStart(3)}`
      + ` sigma ${schedule[round - 1].toFixed(1).padStart(8)}`
      + `   N-CA ${geometry.nca.toFixed(2)}   CA-C ${geometry.cac.toFixed(2)}`
      + `   CA-CA ${geometry.caca.toFixed(2)}   gyration ${geometry.gyration.toFixed(1)}`
      + `   moved ${Number.isNaN(moved) ? "-" : moved.toFixed(3)} A`);
  }

  return {
    sequence, tokens, steps: schedule.length, start, sigma, geometry,
    gyration: geometry.gyration, meanPlddt: 0, seconds: 0,
    pdb: pdbs[pdbs.length - 1], iterationPdbs: pdbs,
  };
}
