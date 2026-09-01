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

  const targetFeat = buildTargetFeat(batch, weights.targetFeat);
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

  const denoised = await new Af3DiffusionHeadGpu(device).run({
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
  }, weights.diffusion);

  const geometry = backboneGeometry(batch, denoised.positions);
  console.log(`backbone  N-CA ${geometry.nca.toFixed(2)} A (ideal 1.46)`
    + `   CA-C ${geometry.cac.toFixed(2)} A (ideal 1.52)`
    + `   CA-CA ${geometry.caca.toFixed(2)} A (ideal 3.80)`);
  console.log(`radius of gyration ${geometry.gyration.toFixed(1)} A over`
    + ` ${geometry.residues} CA   (a compact 68-mer is about 11-12 A)`);

  return {
    sequence, tokens, steps: 1, start, sigma, geometry,
    gyration: geometry.gyration, meanPlddt: 0, seconds: 0,
    pdb: toPdb(batch, denoised.positions, null),
  };
}
