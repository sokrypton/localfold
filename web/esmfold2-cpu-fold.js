/**
 * A whole ESMFold2 fold with no GPU: a sequence in, a structure out.
 *
 * The same stages in the same order as webgpu/esmfold2/fold.js, composed from the
 * reference kernels in cpu/, so this is no second implementation of the model: each
 * kernel is the one the GPU port is held to. What makes it usable is that their
 * matrix multiplies run through cpu/runtime/backend.js when one is set, because
 * without it a 40-residue fold is 6.5 minutes (docs/EF2FAST.md).
 *
 * Unlike the GPU fold this blocks its thread, so the page runs it in a worker
 * (web/esmfold2-cpu-worker.js).
 *
 * A separate draw from the GPU's, like any two samples: the sampler's noise comes
 * from the same stream and seed, but float32 rounding differs between the two, so
 * compare structures after superposition and not coordinate by coordinate.
 */
import { assembleSingleInputs, SAMPLER_DEFAULTS, SAMPLER_PRESETS, scatterLanguageSingle }
  from "../webgpu/esmfold2/fold.js";
import { featuriseForEsmfold2, languageModelInput } from "../shared/esmfold2/featurise.js";
import { contactBinCountsByPair } from "../shared/esmfold2/contacts.js";
import { representativeAtoms } from "../shared/esmfold2/representative-atoms.js";
import { transition, triangleMultiplication } from "../cpu/af3/trunk/pairformer.js";
import { esmcBlock, layerMix, layerNorm as towerNorm, linear as towerLinear, shimPair }
  from "../cpu/esmc/tower.js";
import { atomDecoder, inputsEmbedder } from "../cpu/esmfold2/atom-transformer.js";
import { esmfold2Confidence } from "../cpu/esmfold2/confidence.js";
import { denoiseStep } from "../cpu/esmfold2/diffusion.js";
import { distogramLogits } from "../cpu/esmfold2/distogram.js";
import { recycleProjection, relativePositionEncoding, tokenBondEncoding, zInitFromInputs }
  from "../cpu/esmfold2/pair-features.js";
import { centreRandomAugmentation, churnFactors, gaussians, noiseLevels, noiseSchedule, samplerStep }
  from "../cpu/esmfold2/sampler.js";

/**
 * The language model's single, one block in memory at a time.
 *
 * `towerStates` and `shimSingle` hold every hidden state because a reference should
 * be obvious. The mix is a running sum, so this keeps none of them, which at 300M is
 * 31 x rows x 960 floats, and asks for each block's weights only when it is reached.
 */
async function languageSingle(lm, language, onBlock) {
  const { manifest, shared, block, ffn } = language;
  const model = manifest.width;
  const layers = manifest.layers;
  const pair = language.pairChannels;
  const dims = { model, heads: manifest.heads ?? model / 64, ffn, layers, pair,
                 sequenceId: lm.sequenceId };
  const rows = lm.ids.length;
  const mix = layerMix(shared["lm/combine"]);
  const accumulator = new Float32Array(rows * pair);
  const accumulate = (state, weight) => {
    const normed = towerNorm(state.slice(), rows, model, shared["lm/norm/scale"], shared["lm/norm/offset"]);
    const projected = towerLinear(normed, rows, model, shared["lm/projection/weights"], pair);
    for (let i = 0; i < accumulator.length; i += 1) accumulator[i] += weight * projected[i];
  };
  let x = new Float32Array(rows * model);
  const table = shared["embed/weights"];
  for (let row = 0; row < rows; row += 1) {
    x.set(table.subarray(lm.ids[row] * model, (lm.ids[row] + 1) * model), row * model);
  }
  accumulate(x, mix[0]);
  for (let layer = 0; layer < layers; layer += 1) {
    const weights = await block(layer);
    x = esmcBlock(x, rows, weights, dims, layer);
    // ...the last state is read through the tower's final norm.
    const state = layer === layers - 1
      ? towerNorm(x.slice(), rows, model, shared["final_norm/scale"], null) : x;
    accumulate(state, mix[layer + 1]);
    onBlock?.(layer + 1, layers);
  }
  const down = towerLinear(accumulator, rows, pair, shared["lm/downproject/weights"], pair);
  const bias = shared["lm/downproject/bias"];
  for (let row = 0; row < rows; row += 1) {
    for (let c = 0; c < pair; c += 1) down[row * pair + c] += bias[c];
  }
  return down;
}

/** The pair term from the single, on the tokens the rows landed on. */
function languagePair(single, tokens, channels, shim) {
  const pair = shimPair(single, tokens, shim, { pair: channels });
  return pair;
}

/** Contact probability per pair: the mass under each pair's own distance threshold. */
function contactProbabilities(logits, pairs, bins, nearBins) {
  const contacts = new Float32Array(pairs);
  for (let cell = 0; cell < pairs; cell += 1) {
    const base = cell * bins;
    let largest = -Infinity;
    for (let bin = 0; bin < bins; bin += 1) largest = Math.max(largest, logits[base + bin]);
    let total = 0;
    let near = 0;
    for (let bin = 0; bin < bins; bin += 1) {
      const weight = Math.exp(logits[base + bin] - largest);
      total += weight;
      if (bin < nearBins[cell]) near += weight;
    }
    contacts[cell] = near / Math.max(total, 1e-30);
  }
  return contacts;
}

/**
 * Fold.
 *
 * @param {object} options the GPU fold's, where they mean the same:
 *   `entities`, `sequence`, `shape`, `weights`, `confidenceWeights`, `language`,
 *   `languageModel`, `sampler`, `seed`, and the same callbacks `onBatch`, `onStatus`,
 *   `onProgress`, `onContacts` and `onStep`. `shouldStop` throws to stop.
 */
export async function foldEsmfold2Cpu(options) {
  const { shape, weights } = options;
  const started = performance.now();
  const timings = {};
  const features = featuriseForEsmfold2(options.entities ?? options.sequence, options.features);
  options.onBatch?.(features.batch ?? features);
  const { tokens, atoms } = features;
  const pairs = tokens * tokens;
  const channels = shape.pairChannels;
  const onStatus = options.onStatus ?? (() => {});
  const onProgress = options.onProgress ?? (() => {});
  const loops = shape.loops ?? 4;
  const settings = { ...SAMPLER_DEFAULTS, ...(SAMPLER_PRESETS[options.sampler ?? "diffusion-15"] ?? {}) };
  const schedule = noiseSchedule({ steps: settings.steps, sMax: settings.sMax, sMin: settings.sMin,
                                   p: settings.p, sigmaData: settings.sigmaData,
                                   maxSigma: settings.maxSigma, schedule: settings.schedule });
  const gammas = churnFactors(schedule, settings.gammaMin, settings.churn === false ? 0 : settings.gamma0);
  const levels = noiseLevels(schedule, gammas);

  // A bar over the stages by their multiply-adds, counted rather than timed: the
  // ratios hold from one machine to the next where a measured table would not.
  const trunkUnits = loops * shape.blocks * pairs * channels * channels * 22;
  const towerUnits = options.languageModel === false ? 0
    : options.language.manifest.layers * tokens * options.language.manifest.width ** 2 * 8;
  const stepUnits = tokens * shape.tokenChannels2 ** 2 * shape.tokenBlocks * 8
    + atoms * shape.atomChannels ** 2 * shape.atomBlocks * 24;
  const confidenceUnits = (options.confidenceWeights?.blocks.length ?? 0) * pairs * channels * channels * 22;
  const total = towerUnits + trunkUnits + stepUnits * levels.length + confidenceUnits + 1;
  let done = 0;
  const advance = (units) => { done += units; onProgress(Math.min(1, done / total)); };
  const stage = async (label, work) => {
    options.shouldStop?.();
    onStatus(`${label} · ${Math.round(100 * done / total)}%`);
    const at = performance.now();
    const value = await work();
    timings[label] = (timings[label] ?? 0) + performance.now() - at;
    return value;
  };

  const lm = options.languageModel === false
    ? { ids: new Int32Array(0), sequenceId: new Int32Array(0),
        tokenToRow: new Int32Array(tokens).fill(-1), rows: 0, chains: 0 }
    : languageModelInput(features);
  const languageSingleOnTokens = await stage("Language model", async () => {
    const rows = lm.ids.length === 0 ? new Float32Array(0)
      : await languageSingle(lm, { ...options.language, pairChannels: channels }, (_, layers) => {
        options.shouldStop?.();
        advance(towerUnits / layers);
      });
    return scatterLanguageSingle(rows, lm, tokens, channels, weights.shim);
  });
  const lmPair = languagePair(languageSingleOnTokens, tokens, channels, weights.shim);

  const atomShape = {
    atoms, tokens, channels: shape.atomChannels, heads: shape.atomHeads, blocks: shape.atomBlocks,
    tokenChannels: shape.tokenChannels, hidden: shape.atomChannels * 2,
    attentionPrecision: "bf16",
  };
  const tokenAct = await stage("Inputs", () => inputsEmbedder(features, atomShape, weights.inputsEmbedder).tokenAct);
  const sInputs = assembleSingleInputs(tokenAct, features, tokens, shape.tokenChannels,
                                       features.aatype.length / tokens, shape.singleInputs);

  const featuriser = weights.featuriser;
  const relPos = relativePositionEncoding(features, tokens, channels, featuriser.relPos);
  const zInit = zInitFromInputs(sInputs, tokens, shape.singleInputs, channels,
                                featuriser.zInit1, featuriser.zInit2);
  const bonds = tokenBondEncoding(features.tokenBonds, tokens, channels, featuriser.tokenBonds);
  for (let i = 0; i < zInit.length; i += 1) zInit[i] += relPos[i] + bonds[i] + lmPair[i];

  const recycle = { scale: featuriser.recycleScale, offset: featuriser.recycleOffset,
                    projection: featuriser.recycleProjection };
  const mask2d = new Float32Array(pairs).fill(1);
  let pair = new Float32Array(pairs * channels);
  const add = (delta) => { for (let i = 0; i < pair.length; i += 1) pair[i] += delta[i]; };
  const runBlocks = (blocks, units, label) => stage(label, async () => {
    for (const block of blocks) {
      options.shouldStop?.();
      add(triangleMultiplication(pair, mask2d, tokens, channels, "outgoing", block.triangleMultiplicationOutgoing));
      add(triangleMultiplication(pair, mask2d, tokens, channels, "incoming", block.triangleMultiplicationIncoming));
      add(transition(pair, pairs, channels, block.pairTransition));
      advance(units / blocks.length);
      onStatus(`${label} · ${Math.round(100 * done / total)}%`);
      // ...a macrotask, so a stop message or a progress post can land between blocks.
      await new Promise((resolve) => { setTimeout(resolve, 0); });
    }
  });
  for (let loop = 0; loop < loops; loop += 1) {
    const projected = recycleProjection(pair, pairs, channels, recycle);
    pair = new Float32Array(pairs * channels);
    for (let i = 0; i < pair.length; i += 1) pair[i] = zInit[i] + projected[i];
    await runBlocks(weights.trunkBlocks, trunkUnits / loops, "Trunk");
  }

  const logits = distogramLogits(pair, tokens, channels, featuriser.distogramWeights,
                                 featuriser.distogramBias, shape.distogramBins);
  const nearBins = contactBinCountsByPair(features.molType, features.residueType, tokens, shape.distogramBins);
  const contacts = contactProbabilities(logits, pairs, shape.distogramBins, nearBins);
  await options.onContacts?.(contacts);

  // ---- the sampler.
  const diffusionAtoms = { ...atomShape, tokenChannels: shape.tokenChannels2, windowSize: shape.atomWindow };
  const denoiseShape = { tokens, pairChannels: channels, singleInputs: shape.singleInputs,
                         tokenChannels: shape.tokenChannels2, multiplier: shape.transitionMultiplier,
                         tokenHeads: shape.tokenHeads, sigmaData: settings.sigmaData };
  const encoder = {
    embed: (extra) => inputsEmbedder({ ...features, ...extra }, diffusionAtoms, weights.denoiser.encoder),
    decode: (act, embedded) => atomDecoder(act, embedded.atomState, embedded.conditioning, embedded.rope,
                                           features, diffusionAtoms, weights.denoiser.decoder),
  };
  const draw = gaussians(options.seed ?? 0);
  let x = new Float32Array(atoms * 3);
  for (let i = 0; i < x.length; i += 1) x[i] = schedule[0] * draw();
  let cachedPair;
  for (let step = 0; step < levels.length; step += 1) {
    await stage("Diffusion", async () => {
      x = centreRandomAugmentation(x, features.mask, atoms, draw);
      const tHat = levels[step];
      const epsilon = settings.noiseScale
        * Math.sqrt(Math.max(tHat * tHat - schedule[step] * schedule[step], 0));
      const noisy = new Float32Array(x.length);
      for (let i = 0; i < noisy.length; i += 1) noisy[i] = x[i] + epsilon * draw();
      const answer = denoiseStep(noisy, tHat, features, sInputs, pair, relPos, denoiseShape,
                                 weights.denoiser, encoder, cachedPair);
      cachedPair = answer.conditioning.pair;
      x = samplerStep(noisy, answer.coordinates, features.mask, atoms, tHat, schedule[step + 1],
                      settings.stepScale, settings.update);
      advance(stepUnits);
      await options.onStep?.({ step, total: levels.length, coordinates: x,
                               denoised: answer.coordinates, features });
    });
    await new Promise((resolve) => { setTimeout(resolve, 0); });
  }
  cachedPair = undefined;

  // ---- the head.
  let confidence;
  if (options.confidenceWeights != null) {
    confidence = await stage("Confidence", () => {
      const out = esmfold2Confidence({
        tokens, atoms, sInputs, pair, coordinates: x, repAtom: representativeAtoms(features, tokens),
        atomToToken: features.atomToToken, atomMask: features.mask,
        tokenMask: new Float32Array(tokens).fill(1), pairBias: relPos, asymId: features.asymId,
      }, options.confidenceWeights);
      advance(confidenceUnits);
      return { plddt: out.plddt ?? out.plddtPerToken, plddtPerAtom: out.plddtPerAtom, plddtCa: out.plddtCa,
               complexPlddt: out.complexPlddt, pae: out.pae, ptm: out.ptm, iptm: out.iptm };
    });
  }
  onProgress(1);
  return { features, coordinates: x, contacts, confidence, steps: levels.length,
           scheduleLength: schedule.length, settings, timings,
           elapsedMilliseconds: performance.now() - started };
}
