/**
 * A whole ESMFold2 fold on the GPU: a sequence in, a structure out.
 *
 *     ids      -> ESM-C's 36 blocks -> a mixed single -> lm_z
 *     z_init    = z_init_1[i] + z_init_2[j] + rel_pos + token_bonds + lm_z
 *     z         = 0;  repeat loops:  z = trunk(z_init + pair_loop_proj(z))
 *     structure = sampler(denoiser(z, s_inputs))
 *
 * 🔴 THE PAIR NEVER LEAVES THE DEVICE BETWEEN THE LOOPS. The trunk's own driver
 * will upload and read back a pair for a caller that wants one, and four loops
 * of that is 736 MB of traffic at 300 tokens for a tensor the recycle
 * projection - itself a GPU pass - is the only thing between. `state.buffer`
 * is the borrowed path.
 *
 * 🔴 AND THE LANGUAGE MODEL IS THE FIRST THING, NOT THE LAST. ESM-C streams one
 * block's weights at a time and holds 2190 MiB if it does not, so it runs and
 * is released before the trunk allocates anything pair-sized. Its 37 hidden
 * states are never materialised at all - see src/esmc/tower-webgpu.js.
 *
 * 🔴 AND `lm_shim(0)` IS NOT ZERO, so "fold without the language model" is not
 * a mode this offers. The shim's biases make the term non-zero for a zero
 * input, so substituting zeros is a different model that folds; a caller
 * without ESM-C has no ESMFold2 to run.
 */
import { GRID_WIDTH, LANES, createLayerNormShader, createLinearShader, linearGrid }
  from "../esmc/block-webgpu.js";
import { featuriseForEsmfold2, languageModelInput } from "./featurise.js";
import { EsmcTowerGpu } from "../esmc/tower-webgpu.js";
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";
import { Esmfold2TrunkGpu } from "./trunk-webgpu.js";
import { Esmfold2DenoiserGpu, atomConditioning, createAddShader } from "./diffusion-webgpu.js";
import { buildRope } from "./atom-encoder-reference.js";
import { runInputsEmbedder } from "./atom-transformer-webgpu.js";
import { encodeLanguagePair } from "./language-pair-webgpu.js";
import { encodeContactMap } from "./distogram-webgpu.js";
import { linear } from "./featuriser-reference.js";
import {
  createBondShader, createRelativePositionShader, createZInitShader,
  relativeLayout, relativeRows,
} from "./pair-features-webgpu.js";
import {
  centreRandomAugmentation, churnFactors, gaussians, noiseLevels, noiseSchedule,
  samplerStep,
} from "./sampler-reference.js";

/**
 * The sampler settings a caller can name, the way AF3's page offers
 * `diffusion-200` and `flow-16`.
 *
 * 🔴 THE STEP COUNT IS AN OUTPUT OF THE SCHEDULE, NOT AN INPUT TO IT.
 * `max_inference_sigma` is a DEFAULT ARGUMENT of `sample`, not a config field,
 * and it drops every schedule entry above the cap and prepends the cap itself -
 * so upstream's `inference_num_steps: 15` runs ELEVEN steps. Reading the config
 * number as the step count gives a sampler that visits four noise levels the
 * model never sees. `steps` here is the schedule's length before truncation,
 * and `run` reports how many actually ran.
 */
/**
 * 🔴 AND SIX STEPS IS NOT "FASTER", IT IS BROKEN, WHICH IS WHY THE TABLE IS
 * HERE. Measured on ubiquitin's first 40 residues against ESMFold2's own fold,
 * one seed each - a direction rather than a margin, but the top row is not a
 * margin:
 *
 * | preset | steps run | CA-CA | RMSD |
 * |---|---|---|---|
 * | diffusion-8 | 6 | **58.0 A** | **48.1 A** |
 * | flow-8 | 6 | 4.27 | 1.72 |
 * | diffusion-15 (shipped) | 11 | 3.806 | 1.05 |
 * | flow-16 | 12 | 3.804 | 1.01 |
 * | diffusion-32 | 23 | 3.802 | 1.14 |
 * | diffusion-200 | 138 | 3.808 | 1.48 |
 *
 * A peptide bond is 3.8 A, so `diffusion-8` is not a structure. The churn is
 * what breaks: `gamma0` re-noises to `sigma * 1.605` and `step_scale` 1.638
 * then overshoots, and at six steps the gap between noise levels is too wide
 * for either to be corrected. The flow arm at the same six steps is merely
 * poor, because it re-noises not at all. More steps than the schedule buy
 * nothing here - 138 of them is no better than 11 and is eight times the time.
 */
export const SAMPLER_PRESETS = {
  /** What the checkpoint ships: 15 scheduled, 11 after the 256 cap. */
  "diffusion-15": { steps: 15, maxSigma: 256 },
  /** Fewer steps for a quick look. See the table: six of them is not a fold. */
  "diffusion-8": { steps: 8, maxSigma: 256 },
  "diffusion-32": { steps: 32, maxSigma: 256 },
  "diffusion-64": { steps: 64, maxSigma: 256 },
  "diffusion-200": { steps: 200, maxSigma: 256 },
  /**
   * 🔴 A FLOW ARM IS THE SAME SCHEDULE WITH THE CHURN AND THE NOISE TURNED OFF.
   * `gamma0 = 0` makes `t_hat` equal `sigma` at every step, so no noise is
   * re-injected and the update is a plain Euler step down the probability-flow
   * ODE.
   *
   * 🔴 AND IT BUYS NOTHING HERE, WHICH IS WHY THE PAGE DOES NOT OFFER IT. For
   * AF3 the switch is a twelve-fold saving - that model's diffusion default is
   * 200 steps and flow-16 is sixteen. ESMFold2's own sampler is ELEVEN steps
   * (`inference_num_steps: 15` truncated by `max_inference_sigma`), so there is
   * nothing to escape from, and a step costs the same either way: `flow-16`
   * runs TWELVE steps, one MORE than the shipped `diffusion-15`, and at 32 the
   * two run 23 each. These stay because the measurement is worth keeping and a
   * tool may ask for one; see ESMFOLD2_COUNTS in web/esmfold2-model.js.
   */
  "flow-16": { steps: 16, maxSigma: 256, gamma0: 0, noiseScale: 0 },
  "flow-8": { steps: 8, maxSigma: 256, gamma0: 0, noiseScale: 0 },
  "flow-32": { steps: 32, maxSigma: 256, gamma0: 0, noiseScale: 0 },
};

/** ESMFold2's own sampler constants, from the checkpoint's config. */
export const SAMPLER_DEFAULTS = {
  sigmaData: 16, gamma0: 0.605, gammaMin: 1.107, noiseScale: 0.901,
  stepScale: 1.638, sMax: 160, sMin: 4e-4, p: 8, steps: 15, maxSigma: 256,
};

/**
 * The dense-layout atom each token is represented by: CB, or CA where there is
 * none, or the token's first atom for a ligand.
 */
export function representativeAtoms(features, tokens) {
  const named = (atom, want) => {
    for (let i = 0; i < 4; i += 1) {
      const wanted = i < want.length ? want.charCodeAt(i) - 32 : 0;
      if (features.refAtomNameChars[atom * 4 + i] !== wanted) return false;
    }
    return true;
  };
  const alpha = new Int32Array(tokens).fill(-1);
  const beta = new Int32Array(tokens).fill(-1);
  const first = new Int32Array(tokens).fill(-1);
  for (let atom = 0; atom < features.atoms; atom += 1) {
    if (features.mask[atom] === 0) continue;
    const token = features.atomToToken[atom];
    if (first[token] < 0) first[token] = atom;
    if (named(atom, "CA")) alpha[token] = atom;
    if (named(atom, "CB")) beta[token] = atom;
  }
  const out = new Int32Array(tokens);
  for (let token = 0; token < tokens; token += 1) {
    out[token] = beta[token] >= 0 ? beta[token]
      : (alpha[token] >= 0 ? alpha[token] : Math.max(0, first[token]));
  }
  return out;
}

/** Those atoms' coordinates out of a full atom array. */
export function gatherPositions(coordinates, slots, tokens) {
  const out = new Float32Array(tokens * 3);
  for (let token = 0; token < tokens; token += 1) {
    for (let axis = 0; axis < 3; axis += 1) {
      out[token * 3 + axis] = coordinates[slots[token] * 3 + axis];
    }
  }
  return out;
}

const perRow = (rows) => [Math.min(GRID_WIDTH, rows), Math.ceil(rows / GRID_WIDTH)];
const elementwise = (elements) => {
  const groups = Math.ceil(elements / LANES);
  return [Math.min(GRID_WIDTH, groups), Math.ceil(groups / GRID_WIDTH)];
};

/** How many pair rows the recycle projection walks at once. */
export const RECYCLE_CHUNK = 8192;

/**
 * `s_inputs`: the pooled atom representation, the residue one-hot, the MSA
 * profile and the deletion mean, in that order.
 *
 * 🔴 THE PROFILE AND THE DELETION MEAN ARE ZERO FOR A SINGLE SEQUENCE, and that
 * is a fact about this model rather than a placeholder. ESMFold2 folds without
 * an alignment; the two blocks exist because the featuriser is shared, and the
 * dumps record them as zeros. Filling the profile with the sequence one-hot
 * instead - the plausible-looking thing - is a different input.
 */
export function assembleSingleInputs(tokenAct, features, tokens, tokenChannels,
                                     classes, singleInputs) {
  const out = new Float32Array(tokens * singleInputs);
  for (let token = 0; token < tokens; token += 1) {
    const to = token * singleInputs;
    for (let c = 0; c < tokenChannels; c += 1) out[to + c] = tokenAct[token * tokenChannels + c];
    for (let c = 0; c < classes; c += 1) {
      out[to + tokenChannels + c] = features.aatype[token * classes + c];
      out[to + tokenChannels + classes + c] = features.profile[token * classes + c];
    }
    out[to + singleInputs - 1] = features.deletionMean[token];
  }
  return out;
}

/**
 * What the shim's single half returns for a hidden state of all zeros.
 *
 * 🔴 A NON-PROTEIN TOKEN'S HIDDEN STATE IS ZERO, WHICH IS NOT THE SAME AS
 * ABSENT. `compute_lm_hidden_states` fills a zero tensor and writes only the
 * protein positions, so a ligand atom or a nucleotide reaches the shim as
 * zeros - and the shim's LayerNorm has an OFFSET and its downprojection a
 * BIAS, so its answer there is a fixed non-zero vector rather than nothing.
 * Dropping those tokens from the pair term instead is a different model.
 *
 * LayerNorm(0) is the offset exactly, the mix weights sum to one, and the
 * projection is shared across the 37 states - so the whole thing collapses to
 * one row of arithmetic, done once.
 */
export function shimSingleForZeroState(weights, model, pair) {
  const offset = weights["lm/norm/offset"];
  const projection = weights["lm/projection/weights"];
  const accumulated = new Float32Array(pair);
  for (let i = 0; i < model; i += 1) {
    const value = offset[i];
    if (value === 0) continue;
    const base = i * pair;
    for (let c = 0; c < pair; c += 1) accumulated[c] += value * projection[base + c];
  }
  const down = weights["lm/downproject/weights"];
  const out = Float32Array.from(weights["lm/downproject/bias"]);
  for (let i = 0; i < pair; i += 1) {
    const value = accumulated[i];
    if (value === 0) continue;
    const base = i * pair;
    for (let c = 0; c < pair; c += 1) out[c] += value * down[base + c];
  }
  return out;
}

/**
 * The tower's rows, placed back on the tokens that asked for them.
 *
 * 🔴 THE TOWER'S ROWS ARE NOT THE MODEL'S TOKENS, in three ways at once. A
 * non-protein token was never sent; an atom-tokenised residue sent ONE row for
 * all of its tokens; and every chain contributes a BOS and an EOS that are rows
 * of the tower's output and not tokens of anything. `tokenToRow` is the map,
 * and -1 is what it says for a token the tower did not see.
 */
export function scatterLanguageSingle(rows, lm, tokens, pair, shim) {
  const zero = shimSingleForZeroState(shim, shim["lm/norm/offset"].length, pair);
  const out = new Float32Array(tokens * pair);
  for (let token = 0; token < tokens; token += 1) {
    const row = lm.tokenToRow[token];
    const base = token * pair;
    if (row < 0) { out.set(zero, base); continue; }
    if ((row + 1) * pair > rows.length) {
      throw new Error(`the tower returned ${rows.length / pair} rows; token ${token} `
        + `wants row ${row}`);
    }
    out.set(rows.subarray(row * pair, (row + 1) * pair), base);
  }
  return out;
}

/**
 * @param device   a WebGPU device
 * @param options  { sequence, tower, weights, sampler, seed, onProgress }
 *   `tower` is { ids -> single }: the ESM-C half, given separately because it
 *   is a different bundle with a different licence and a caller may already
 *   have run it.
 */
export async function foldEsmfold2(device, options) {
  const { sequence, weights, shape } = options;
  const allocator = options.allocator ?? new GpuBufferAllocator(device);
  const cache = pipelineCacheForDevice(device);
  const storage = GPUBufferUsage.STORAGE;
  const report = options.onProgress ?? (() => {});
  const started = performance.now();
  const timings = {};
  const mark = async (label, work) => {
    await report(label);
    const at = performance.now();
    const value = await work();
    timings[label] = performance.now() - at;
    return value;
  };

  const features = options.featuresOverride
    ?? featuriseForEsmfold2(options.entities ?? sequence, options.features);
  const tokens = features.tokens;
  const atoms = features.atoms;
  const pairs = tokens * tokens;
  const channels = shape.pairChannels;
  const held = [];
  const keep = (allocation) => { held.push(allocation); return allocation; };

  const submit = async (label, passes) => {
    const encoder = device.createCommandEncoder({ label });
    for (const [name, pipeline, buffers, x, y] of passes) {
      const pass = encoder.beginComputePass({ label: name });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: buffers.map((entry, binding) => ({
          binding,
          resource: entry.byteOffset === undefined
            ? { buffer: entry.buffer }
            : { buffer: entry.buffer, offset: entry.byteOffset, size: entry.byteSize },
        })),
      }));
      pass.dispatchWorkgroups(x, y ?? 1, 1);
      pass.end();
    }
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  };

  try {
    // ---- the language model, first and released before anything pair-sized.
    const lmPair = keep(allocator.allocate("esmfold2.lm-pair", pairs * channels * 4, storage));
    const lm = languageModelInput(features);
    const single = await mark("language model", async () => {
      const rows = lm.ids.length === 0 ? new Float32Array(0)
        : await options.tower(lm.ids, lm.sequenceId);
      return scatterLanguageSingle(rows, lm, tokens, channels, weights.shim);
    });
    await mark("language pair", () => encodeLanguagePair(
      { device, allocator, cache, submit },
      { tokens, channels, single, weights: weights.shim, destination: lmPair }));

    // ---- the inputs embedder, whose pooled output is most of `s_inputs`.
    const atomShape = {
      atoms, tokens, channels: shape.atomChannels, heads: shape.atomHeads,
      blocks: shape.atomBlocks, hidden: shape.atomChannels * 2,
      tokenChannels: shape.tokenChannels, window: shape.atomWindow,
      precision: options.attentionPrecision ?? "bf16",
    };
    const rope = buildRope(features.refPos, features.refSpaceUid, atoms,
                           shape.atomChannels / shape.atomHeads);
    const sInputs = await mark("inputs embedder", async () => {
      const tokenAct = await runInputsEmbedder(
        { device, allocator, cache, rope,
          atomConditioning: atomConditioning(features, atoms, shape.atomChannels,
                                             weights.inputsEmbedder) },
        { features, shape: atomShape, weights: weights.inputsEmbedder });
      return assembleSingleInputs(tokenAct, features, tokens, shape.tokenChannels,
                                  features.aatype.length / tokens, shape.singleInputs);
    });

    // ---- z_init's other four terms.
    const layout = relativeLayout();
    const [relative, bond, zInitShader, recycleNorm, recycleProject, addPair] =
      await Promise.all([
        cache.get(`esmfold2-rel:${pairs}:${channels}`,
          createRelativePositionShader({ pairs, channels, entityBase: layout.entityBase })),
        cache.get(`esmfold2-bond:${pairs}:${channels}`,
          createBondShader({ pairs, channels })),
        cache.get(`esmfold2-zinit:${tokens}:${channels}`,
          createZInitShader({ tokens, channels, hasBonds: true, hasLanguageModel: true })),
        cache.get(`esmfold2-recycle-norm:${Math.min(RECYCLE_CHUNK, pairs)}:${channels}`,
          createLayerNormShader(
            { rows: Math.min(RECYCLE_CHUNK, pairs), channels }, true, 1e-5)),
        cache.get(`esmfold2-recycle-project:${Math.min(RECYCLE_CHUNK, pairs)}:${channels}`,
          createLinearShader(
            { rows: Math.min(RECYCLE_CHUNK, pairs), inner: channels, outer: channels }, true)),
        cache.get(`esmfold2-pair-add:${pairs * channels}`,
          createAddShader(pairs * channels)),
      ]);

    const relPos = keep(allocator.allocate("esmfold2.rel-pos", pairs * channels * 4, storage));
    const zInit = keep(allocator.allocate("esmfold2.z-init", pairs * channels * 4, storage));
    const pair = keep(allocator.allocate("esmfold2.pair", pairs * channels * 4,
      storage | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST));
    await mark("z_init", async () => {
      const bins = keep(allocator.upload("esmfold2.rel-bins",
        relativeRows(features, tokens), storage));
      const relWeights = keep(allocator.upload("w.esmfold2.rel-pos",
        weights.featuriser.relPos, storage));
      const bondValues = keep(allocator.upload("esmfold2.bonds", features.tokenBonds, storage));
      const bondWeights = keep(allocator.upload("w.esmfold2.token-bonds",
        weights.featuriser.tokenBonds, storage));
      const bonds = keep(allocator.allocate("esmfold2.bond-term",
        pairs * channels * 4, storage));
      // 🔴 PROJECT THEN BROADCAST. The two per-token projections are 2n rows of
      // 451 channels on the host; broadcasting first would make them 2n^2.
      const rows = keep(allocator.upload("esmfold2.z-rows",
        linear(sInputs, tokens, shape.singleInputs, channels,
               weights.featuriser.zInit1), storage));
      const columns = keep(allocator.upload("esmfold2.z-columns",
        linear(sInputs, tokens, shape.singleInputs, channels,
               weights.featuriser.zInit2), storage));
      await submit("esmfold2.z-init", [
        ["rel-pos", relative, [bins, relWeights, relPos], ...elementwise(pairs * channels)],
        ["bonds", bond, [bondValues, bondWeights, bonds], ...elementwise(pairs * channels)],
        ["z-init", zInitShader, [rows, columns, relPos, bonds, lmPair, zInit],
         ...elementwise(pairs * channels)],
      ]);
      for (const allocation of [bins, relWeights, bondValues, bondWeights, bonds,
                                rows, columns, lmPair]) {
        allocation.release();
        held.splice(held.indexOf(allocation), 1);
      }
    });

    // ---- the trunk, `loops` times over a pair that stays on the device.
    const pairMask = keep(allocator.upload("esmfold2.pair-mask",
      new Float32Array(pairs).fill(1), storage));
    const recycleScale = keep(allocator.upload("w.esmfold2.recycle-scale",
      weights.featuriser.recycleScale, storage));
    const recycleOffset = keep(allocator.upload("w.esmfold2.recycle-offset",
      weights.featuriser.recycleOffset, storage));
    const recycleWeights = keep(allocator.upload("w.esmfold2.recycle-projection",
      weights.featuriser.recycleProjection, storage));
    const recycleScratch = keep(allocator.allocate("esmfold2.recycle-scratch",
      Math.min(RECYCLE_CHUNK, pairs) * channels * 4, storage));
    // 🔴 `z` STARTS AT ZERO AND `pair_loop_proj(0)` IS NOT ZERO. Its Linear is
    // zero-INITIALISED upstream and then trained, and the LayerNorm in front of
    // it has an offset - so the first loop's input is z_init plus a real
    // vector. Skipping the projection on the first loop is the natural
    // shortcut and a different model.
    device.queue.writeBuffer(pair.buffer, 0, new Float32Array(pairs * channels));
    const trunk = new Esmfold2TrunkGpu(device, { allocator, ...options.trunk });
    const slice = (allocation, row, rows) => ({
      buffer: allocation.buffer, byteOffset: row * channels * 4,
      byteSize: rows * channels * 4,
    });
    const loops = shape.loops ?? 4;
    for (let loop = 0; loop < loops; loop += 1) {
      await mark(`recycle ${loop}`, async () => {
        const chunk = Math.min(RECYCLE_CHUNK, pairs);
        for (let start = 0; start < pairs; start += chunk) {
          const rows = Math.min(chunk, pairs - start);
          if (rows !== chunk) {
            // The tail chunk needs its own pipelines; at these sizes it is one
            // extra compile, and a wrong-sized dispatch leaves rows unwritten -
            // which reads as a speedup, not as an error.
            const [norm, project] = await Promise.all([
              cache.get(`esmfold2-recycle-norm:${rows}:${channels}`,
                createLayerNormShader({ rows, channels }, true, 1e-5)),
              cache.get(`esmfold2-recycle-project:${rows}:${channels}`,
                createLinearShader({ rows, inner: channels, outer: channels }, true)),
            ]);
            await submit("esmfold2.recycle", [
              ["norm", norm, [slice(pair, start, rows), recycleScale, recycleOffset,
                              recycleScratch], ...perRow(rows)],
              ["project", project, [recycleScratch, recycleWeights,
                                    slice(zInit, start, rows), slice(pair, start, rows)],
               ...linearGrid(rows, channels)],
            ]);
            continue;
          }
          await submit("esmfold2.recycle", [
            ["norm", recycleNorm, [slice(pair, start, rows), recycleScale, recycleOffset,
                                   recycleScratch], ...perRow(rows)],
            ["project", recycleProject, [recycleScratch, recycleWeights,
                                         slice(zInit, start, rows), slice(pair, start, rows)],
             ...linearGrid(rows, channels)],
          ]);
        }
      });
      await mark(`trunk ${loop}`, () => trunk.run(
        { buffer: pair, maskBuffer: pairMask },
        weights.trunkBlocks,
        { n: tokens, channels, readback: false,
          onBlock: options.onBlock && ((index) => options.onBlock(loop, index)) }));
    }
    recycleScratch.release();
    held.splice(held.indexOf(recycleScratch), 1);

    // ---- the distogram, which is the trunk's one output besides the pair.
    // 🔴 IT RUNS BEFORE THE DIFFUSION MODULE ALLOCATES, not after the fold.
    // Its symmetrised copy of the pair is another 92 MiB at 300 tokens, and the
    // denoiser is holding its own pair conditioning and twelve bias tensors by
    // then - so running it here costs nothing and running it at the end raises
    // the peak by a whole pair representation.
    const distogram = options.contacts === false ? undefined
      : await mark("distogram", () => encodeContactMap(
        { device, allocator, cache, submit },
        { tokens, channels, bins: shape.distogramBins, pair,
          weights: weights.featuriser.distogramWeights,
          bias: weights.featuriser.distogramBias,
          wantLogits: options.distogramLogits === true,
          retainForFrames: options.frameCertainty === true }));
    const contacts = distogram?.contacts;
    const certainty = distogram?.certainty;
    const frames = distogram?.frames;
    // 🔴 THE CERTAINTY GOES OUT WITH THE CONTACTS, BEFORE THE SAMPLER RUNS.
    // Both come off the trunk's distogram, so a caller colouring its live
    // frames has them from the first one - and the first version assigned it
    // only from the RESULT, which left every frame but the last at a zero
    // B-factor and therefore the colour of no confidence at all.
    await options.onContacts?.(contacts, certainty);

    // ---- the sampler.
    const settings = { ...SAMPLER_DEFAULTS,
                       ...(SAMPLER_PRESETS[options.sampler ?? "diffusion-15"] ?? {}),
                       ...(options.samplerOverrides ?? {}) };
    const denoiser = new Esmfold2DenoiserGpu(device, allocator, cache);
    await mark("conditioning", () => denoiser.prepare({
      shape: {
        tokens, atoms,
        pairChannels: channels, singleInputs: shape.singleInputs,
        tokenChannels: shape.tokenChannels2, tokenHeads: shape.tokenHeads,
        multiplier: shape.transitionMultiplier, sigmaData: settings.sigmaData,
        atomChannels: shape.atomChannels, atomHeads: shape.atomHeads,
        atomBlocks: shape.atomBlocks, atomHidden: shape.atomChannels * 2,
        window: shape.atomWindow, attentionPrecision: options.attentionPrecision ?? "bf16",
      },
      weights: weights.denoiser, features, sInputs, pair, relPos,
    }));

    // 🔴 THE DISTOGRAM IS OVER THE REPRESENTATIVE ATOM - CB, or CA for glycine,
    // or a ligand token's only atom - so a frame is scored on those and not on
    // alpha carbons. Gathered once: the slots do not move, only the coordinates.
    const representative = frames === undefined ? undefined
      : representativeAtoms(features, tokens);

    const schedule = noiseSchedule({ steps: settings.steps, sMax: settings.sMax,
                                     sMin: settings.sMin, p: settings.p,
                                     sigmaData: settings.sigmaData,
                                     maxSigma: settings.maxSigma });
    const gammas = churnFactors(schedule, settings.gammaMin, settings.gamma0);
    const levels = noiseLevels(schedule, gammas);
    const draw = gaussians(options.seed ?? 0);
    let x = new Float32Array(atoms * 3);
    for (let i = 0; i < x.length; i += 1) x[i] = schedule[0] * draw();
    for (let step = 0; step < levels.length; step += 1) {
      await report(`sampler ${step + 1} of ${levels.length}`);
      const at = performance.now();
      x = centreRandomAugmentation(x, features.mask, atoms, draw);
      const tHat = levels[step];
      const epsilon = settings.noiseScale
        * Math.sqrt(Math.max(tHat * tHat - schedule[step] * schedule[step], 0));
      const noisy = new Float32Array(x.length);
      for (let i = 0; i < noisy.length; i += 1) noisy[i] = x[i] + epsilon * draw();
      const denoised = await denoiser.denoise(noisy, tHat);
      x = samplerStep(noisy, denoised, features.mask, atoms, tHat,
                      schedule[step + 1], settings.stepScale);
      // 🔴 SCORED ON THE DENOISED PREDICTION, WHICH IS WHAT IS DRAWN. The
      // sampler's own state is Gaussian noise at the top of the schedule and is
      // not what a viewer shows; colouring the state while drawing the
      // prediction would put one frame's colour on another frame's structure.
      const frameCertainty = frames === undefined ? undefined
        : await frames.score(gatherPositions(denoised, representative, tokens));
      timings[`sampler ${step}`] = performance.now() - at;
      // 🔴 BOTH, BECAUSE A VIEWER WANTS THE ONE THE SAMPLER DOES NOT KEEP.
      // `coordinates` is the trajectory state at the NEXT noise level - what the
      // sampler carries forward - and at the top of the schedule that is
      // Gaussian noise at sigma 411, which is not a picture of anything.
      // `denoised` is the model's predicted structure at this call, EDM
      // preconditioning included, and is protein-sized in every frame.
      await options.onStep?.({ step, total: levels.length, coordinates: x,
                               denoised, features, certainty: frameCertainty });
    }
    const memory = allocator.snapshot();
    denoiser.release();
    // ...the retained distogram is 46 MiB at 300 tokens and nothing reads it
    // after the last frame.
    frames?.release();

    return {
      coordinates: x, features, sequence, tokens, atoms, sInputs, contacts, certainty,
      distogram: options.distogramLogits === true ? distogram : undefined,
      steps: levels.length, scheduleLength: schedule.length, settings,
      elapsedMilliseconds: performance.now() - started, timings, memory,
    };
  } finally {
    for (let at = held.length - 1; at >= 0; at -= 1) {
      try { held[at].release(); } catch { /* already released */ }
    }
  }
}
