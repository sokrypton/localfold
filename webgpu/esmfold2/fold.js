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
 * states are never materialised at all - see webgpu/esmc/tower.js.
 *
 * 🔴 AND `lm_shim(0)` IS NOT ZERO, so "fold without the language model" is not
 * a mode this offers. The shim's biases make the term non-zero for a zero
 * input, so substituting zeros is a different model that folds; a caller
 * without ESM-C has no ESMFold2 to run.
 */
import { GRID_WIDTH, LANES, createLayerNormShader, createLinearShader, linearGrid }
  from "../esmc/block.js";
import {
  featuriseForEsmfold2, languageModelInput, maskLanguageModelInput,
  MOL_DNA, MOL_RNA,
} from "../../shared/esmfold2/featurise.js";
import { EsmcTowerGpu } from "../esmc/tower.js";
import { createAddShader } from "../runtime/execution.js";
import { deviceTuning } from "../runtime/device-profile.js";
import { memoryTotals } from "../runtime/device-memory.js";
import { GpuBufferAllocator, settleReleasedMemory } from "../runtime/allocator.js";
import { spillToHost } from "../runtime/host-spill.js";
import { throwIfAborted } from "../runtime/abort.js";
import { yieldToBrowser } from "../runtime/yield.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";
import { Esmfold2TrunkGpu } from "./trunk.js";
import { esmfold2ConfidenceFold } from "./confidence.js";
import { Esmfold2DenoiserGpu, atomConditioning } from "./diffusion.js";
import { buildRope } from "../../cpu/esmfold2/atom-transformer.js";
import { runInputsEmbedder } from "./atom-transformer.js";
import { encodeLanguagePair } from "./language-pair.js";
import { encodeContactMap, partnerKeys } from "./distogram.js";
import { linear } from "../../cpu/esmfold2/pair-features.js";
import { alignedErrorFromDistogram } from "../../shared/esmfold2/aligned-error.js";
import {
  createBondShader, createRelativePositionShader, createZInitShader,
  relativeLayout, relativeRows,
} from "./pair-features.js";
import {
  centreRandomAugmentation, churnFactors, gaussians, noiseLevels, noiseSchedule,
  samplerStep, uniforms,
} from "../../cpu/esmfold2/sampler.js";
import { ESMFOLD2_PHASES, esmfold2Plan, trunkPhase } from "./cost.js";
import { esmfold2Spills } from "./memory-plan.js";

/** The largest pair copied to the host; past it the browser refuses the array. */
const HOST_PAIR_MAX_BYTES = 1024 * 1024 * 1024;

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


import { representativeAtoms, gatherPositions } from "../../shared/esmfold2/representative-atoms.js";
export { representativeAtoms, gatherPositions } from "../../shared/esmfold2/representative-atoms.js";


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
  const started = performance.now();
  const timings = {};

  const features = options.featuresOverride
    ?? featuriseForEsmfold2(options.entities ?? sequence, options.features);
  // 🔴 THE BATCH, BEFORE THE FOLD RATHER THAN AFTER IT - the same hook the AF3
  // path has, and for the same reason: the contact map arrives from the trunk
  // in TOKEN space, and only the batch says which tokens the viewer draws as
  // one residue. See web/prediction-results.js `viewerTokens`.
  options.onBatch?.(features.batch ?? features);
  const tokens = features.tokens;
  const atoms = features.atoms;
  const pairs = tokens * tokens;
  const channels = shape.pairChannels;

  // 🔴 THE DENOISER IS BUILT HERE AND ITS PIPELINES ARE COMPILED HERE, not
  // beside the `prepare` that needs them. Everything `#compile` reads is a
  // shape and the shape is known from the FEATURES, which exist before the
  // language model has run; `createComputePipelineAsync` compiles off the main
  // thread and the trunk leaves the host waiting for seconds. Measured on a
  // 40-residue fold: `conditioning` 966 ms on a first fold and 55 on a second,
  // and all of that difference is the compile.
  //
  // 🔴 THE WARM-UP IS NOT AWAITED AND ITS SHAPE IS CHECKED. `prepare` takes the
  // memoised promise only when the key matches what it resolved for itself; a
  // warm-up that guessed wrong compiles again rather than handing back
  // pipelines for the wrong molecule.
  const denoiser = new Esmfold2DenoiserGpu(device, allocator, cache,
    options.denoiserWeightPrecision === undefined ? {}
      : { weightPrecision: options.denoiserWeightPrecision });

  // 🔴 THE SAMPLER'S SETTINGS ARE RESOLVED FIRST, BECAUSE THE PLAN NEEDS THE
  // STEP COUNT. They depend on nothing the fold computes - the schedule is a
  // function of six constants - and the bar cannot be laid out until it knows
  // how many denoiser calls are coming.
  const settings = { ...SAMPLER_DEFAULTS,
                     ...(SAMPLER_PRESETS[options.sampler ?? "diffusion-15"] ?? {}),
                     ...(options.samplerOverrides ?? {}) };
  // (schedule, update and churn: cpu/esmfold2/sampler.js's options; biohub's own sampler without them)
  const schedule = noiseSchedule({ steps: settings.steps, sMax: settings.sMax,
                                   sMin: settings.sMin, p: settings.p,
                                   sigmaData: settings.sigmaData,
                                   maxSigma: settings.maxSigma, schedule: settings.schedule });
  const gammas = churnFactors(schedule, settings.gammaMin, settings.churn === false ? 0 : settings.gamma0);
  const levels = noiseLevels(schedule, gammas);

  // ...and the denoiser's shape, which needs `settings.sigmaData` and nothing
  // the fold computes.
  const denoiserShape = {
    tokens, atoms,
    pairChannels: channels, singleInputs: shape.singleInputs,
    tokenChannels: shape.tokenChannels2, tokenHeads: shape.tokenHeads,
    multiplier: shape.transitionMultiplier, sigmaData: settings.sigmaData,
    atomChannels: shape.atomChannels, atomHeads: shape.atomHeads,
    atomBlocks: shape.atomBlocks, atomHidden: shape.atomChannels * 2,
    // 🔴 THE DIFFUSION ATOM STACKS ARE WINDOWED AND THE INPUTS EMBEDDER IS NOT,
    // which is not a guess: `check-esmfold2-diffusion-gpu.js` holds the
    // denoiser to ESMFold2's own dump and reads **1.43e-4 windowed against
    // 3.46e-3 dense** - 24x worse, over the bf16 bound. Reasoning from
    // fastplms, where every `SWA3DRoPEAttention` defaults to
    // ATOM_ATTENTION_DENSE, said both should be dense; the oracle for this
    // stage says otherwise and the oracle wins. See docs/EF2FAST.md.
    window: shape.atomWindow,
    attentionPrecision: options.attentionPrecision ?? "bf16",
  };
  // The catch is only so a compile that fails before anything awaits it is not
  // an unhandled rejection; the real failure still arrives at `prepare`.
  denoiser.warm(denoiserShape, features).catch(() => {});

  // 🔴 A COARSE PHASE AND A PERCENTAGE, NOT A STAGE NAME PER STAGE. Reporting
  // each stage gave "recycle 0", "trunk 0", "recycle 1", "trunk 1" - and a
  // recycle is two milliseconds against a trunk loop's several seconds, so the
  // line flickered between two stages whose costs differ by a thousand. The
  // recycle is not a phase; it is the seam between two trunk passes.
  const onStatus = options.onStatus ?? (() => {});
  const onProgress = options.onProgress ?? (() => {});
  const loops = shape.loops ?? 4;
  // ...and how big the tower is, since the band it gets is its bytes. Zero
  // when it is not running at all, which is a ligand-only input or a reader who
  // turned it off.
  // 🔴 THE TRUNK IS THE FOLD, SO CHANGING ONLY THE SAMPLER SHOULD COST ONLY THE
  // SAMPLER. AlphaFold 3's path has cached its trunk since it had one - "the
  // last fold's trunk, so changing only the sampler costs only the sampler" -
  // and this one re-ran ESM-C and all 96 trunk blocks to change a step count,
  // which at 300 tokens is 27 s of a 32 s fold thrown away.
  //
  // 🔴 AND IT IS VALIDATED ON SHAPE, NOT TRUSTED. A pair of the wrong width or
  // token count is the shape of every cache bug this repository has recorded -
  // two models' representations agree in everything but their parameters - so
  // the KEY is the caller's business and this refuses anything that does not
  // fit the batch in front of it.
  const reuse = (() => {
    const given = options.reuse;
    if (given === undefined) return undefined;
    if (given.tokens !== tokens || given.channels !== channels) return undefined;
    if (given.pair?.length !== tokens * tokens * channels) return undefined;
    if (given.sInputs?.length !== tokens * shape.singleInputs) return undefined;
    return given;
  })();

  const plan = esmfold2Plan({
    tokens, steps: levels.length, loops: reuse === undefined ? loops : 0,
    languageModelMiB: reuse !== undefined || options.languageModel === false
      ? 0 : options.languageModelMiB,
  });
  let completed = 0;
  let phase = ESMFOLD2_PHASES.languageModel;
  const say = () => onStatus(`${phase} · ${Math.round(100 * completed / plan.total)}%`);
  const enter = (name) => { phase = name; say(); };
  const advance = (units) => { completed = Math.min(plan.total, completed + units); say();
                               onProgress(completed / plan.total); };
  // Stop was only read after the whole fold had finished, so a large fold kept the
  // GPU for minutes after being told to stop. Checked at each stage, at each trunk
  // block and at each sampler step, none of which has work in flight it still owns.
  const stopIfAsked = () => throwIfAborted(options.signal);
  const mark = async (label, work) => {
    stopIfAsked();
    const at = performance.now();
    const value = await work();
    timings[label] = performance.now() - at;
    return value;
  };

  const held = [];
  const keep = (allocation) => { held.push(allocation); return allocation; };

  // 🔴 `wait` DEFAULTS TO A DRAIN, AND THE RECYCLE'S ROW CHUNKS DO NOT WANT
  // ONE: they write disjoint rows of the pair and the trunk after them is
  // queue-ordered behind them, so waiting on each was 32 drains a fold at 255
  // tokens around small passes.
  const submit = async (label, passes, wait = true) => {
    const encoder = device.createCommandEncoder({ label });
    const batch = deviceTuning(device).batchComputePasses !== false;
    let openPass = null;
    for (const [name, pipeline, buffers, x, y] of passes) {
      const pass = batch ? (openPass ??= encoder.beginComputePass({ label }))
        : encoder.beginComputePass({ label: `${label}.${name}` });
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
      if (!batch) pass.end();
    }
    if (openPass !== null) openPass.end();
    device.queue.submit([encoder.finish()]);
    if (wait) await device.queue.onSubmittedWorkDone();
  };

  try {
    // ---- the language model, first and released before anything pair-sized.
    // 🔴 THE LANGUAGE MODEL CAN BE TURNED OFF, WHICH IS THIS MODEL'S "SINGLE
    // SEQUENCE". AF2 and AF3 can be run without their alignment; the analogue
    // here is running without the protein language model, since ESM-C is where
    // this model's evolutionary information comes from. It is not a matter of
    // zeroing anything: `tokenToRow` of -1 already means "no row", which is the
    // path every ligand and nucleotide token takes, and
    // `shimSingleForZeroState` gives it the fixed NON-ZERO vector the shim's
    // own offset and bias produce. Substituting zeros instead would be a
    // different model.
    const lm = options.languageModel === false
      ? { ids: new Int32Array(0), sequenceId: new Int32Array(0),
          tokenToRow: new Int32Array(tokens).fill(-1), rows: 0, chains: 0 }
      : languageModelInput(features);
    // 🔴 `lm_mask_pct`, ON A STREAM OF ITS OWN. Zero for this checkpoint - see
    // maskLanguageModelInput for why the config class's docstring says
    // otherwise - so nothing is drawn and a fold is bit-identical to one
    // before this existed. Where a caller does ask for it, the draws must not
    // come from the sampler's stream: sharing one would make the STRUCTURE
    // change at mask fraction zero, since a conditional draw shifts every
    // later value.
    const maskFraction = options.lmMaskFraction ?? shape.lmMaskPct ?? 0;
    const maskedTokens = maskLanguageModelInput(
      lm.ids, maskFraction, uniforms((options.seed ?? 0) ^ 0x5bf03635));
    enter(ESMFOLD2_PHASES.languageModel);
    const single = reuse !== undefined ? new Float32Array(0)
      : await mark("language model", async () => {
      const rows = lm.ids.length === 0 ? new Float32Array(0)
        : await options.tower(lm.ids, lm.sequenceId,
            (layer, layers) => advance(plan.languageModel / layers));
      return scatterLanguageSingle(rows, lm, tokens, channels, weights.shim);
    });
    // ...whatever the per-block reporting did not account for, so the band
    // ends exactly where the plan says however many blocks the tower ran.
    completed = Math.max(completed, plan.languageModelEnd);
    enter(ESMFOLD2_PHASES.embedder);
    await settleReleasedMemory(device);
    const lmPair = reuse !== undefined ? undefined
      : keep(allocator.allocate("esmfold2.lm-pair", pairs * channels * 4,
        storage | (options.onLanguagePair !== undefined ? GPUBufferUsage.COPY_SRC : 0)));
    if (reuse === undefined) await mark("language pair", () => encodeLanguagePair(
      { device, allocator, cache, submit },
      { tokens, channels, single, weights: weights.shim, destination: lmPair }));
    // `onLanguagePair(pair)`: the language model's pair term [tokens, tokens, channels] read back - a comparison
    // aid (the native ports write theirs with EF2_SAVE_LMZ), only when asked
    if (reuse === undefined && options.onLanguagePair !== undefined) {
      const back = allocator.allocate("esmfold2.lm-pair-rb", pairs * channels * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      const encoder = device.createCommandEncoder({ label: "esmfold2.lm-pair-rb" });
      encoder.copyBufferToBuffer(lmPair.buffer, 0, back.buffer, 0, pairs * channels * 4);
      device.queue.submit([encoder.finish()]);
      await back.buffer.mapAsync(GPUMapMode.READ);
      options.onLanguagePair(new Float32Array(back.buffer.getMappedRange().slice(0)), single);
      back.buffer.unmap();
      back.release();
    }

    // ---- the inputs embedder, whose pooled output is most of `s_inputs`.
    const atomShape = {
      atoms, tokens, channels: shape.atomChannels, heads: shape.atomHeads,
      // 🔴 OVERRIDABLE FOR THE ORACLE ARM ONLY. Their atom transformer can be
      // truncated to k blocks from python; this is the same knob on this side,
      // so the two can be compared one block at a time. Never set in a fold.
      blocks: options.atomBlocks ?? shape.atomBlocks,
      hidden: shape.atomChannels * 2,
      tokenChannels: shape.tokenChannels,
      // 🔴 THE REFERENCE'S ATOM ATTENTION IS DENSE BY DEFAULT, NOT WINDOWED.
      // `SWA3DRoPEAttention.__init__` sets `_atom_attention = ATOM_ATTENTION_DENSE`
      // and its forward takes the windowed branch only when something calls
      // `set_atom_attention("windowed")` - which nothing on the fold path does.
      // `swa_window_size` is in the config and unused by the shipped model. A
      // window covering every atom IS dense, so this needs no second kernel -
      // only an `attend` that does not hold the whole window in workgroup
      // memory, which is why the chunked online softmax landed first.
      // `--atom-windowed=1` restores the old behaviour for comparison.
      window: options.atomWindowed === true ? shape.atomWindow : atoms * 2,
      precision: options.attentionPrecision ?? "bf16",
    };
    let embedderConditioning;
    let embedderCapture;
    let embedderZInit;
    const rope = buildRope(features.refPos, features.refSpaceUid, atoms,
                           shape.atomChannels / shape.atomHeads);
    const sInputs = reuse !== undefined ? reuse.sInputs
      : await mark("inputs embedder", async () => {
      const conditioning = atomConditioning(features, atoms, shape.atomChannels,
                                            weights.inputsEmbedder);
      // ...kept for the oracle arm: this is their `c_base`, the atom encoder's
      // first stage, and comparing it separates the projection from the blocks.
      embedderConditioning = conditioning;
      // 🔴 TRUNCATING THE STACK MEANS TRUNCATING THE WEIGHTS, NOT THE SHAPE.
      // `encodeAtomStack` loops `for (const block of weights)`, so a smaller
      // `shape.blocks` changes what is COMPILED and not what is RUN - which is
      // how an oracle arm can look like it is bisecting while every run is the
      // full stack. Found by asking for 0 blocks and getting the same answer.
      const embedderWeights = options.atomBlocks === undefined
        ? weights.inputsEmbedder
        : { ...weights.inputsEmbedder,
            blocks: weights.inputsEmbedder.blocks.slice(0, options.atomBlocks) };
      const tokenAct = await runInputsEmbedder(
        { device, allocator, cache, rope, atomConditioning: conditioning,
          capture: options.returnConfidenceInputs === true ? (embedderCapture = {}) : undefined },
        { features, shape: atomShape, weights: embedderWeights });
      return assembleSingleInputs(tokenAct, features, tokens, shape.tokenChannels,
                                  features.aatype.length / tokens, shape.singleInputs);
    });

    // ---- z_init's other four terms.
    const layout = relativeLayout();
    // 🔴 A CHUNK OF ROWS AT A TIME, because every term is. z_init held the language pair, the relative encoding, the
    // bond term, z_init itself and the empty pair at once, five pair-sized tensors where its own two will do, and on a
    // small card that was where a fold was refused. The same additions in the same order, so the same bits.
    const chunkRows = Math.min(RECYCLE_CHUNK, pairs);
    const heights = [...new Set([chunkRows, pairs % chunkRows].filter((rows) => rows > 0))];
    const perHeight = async (key, source) => {
      const compiled = await Promise.all(heights.map((rows) => cache.get(`${key}:${rows}:${channels}`, source(rows))));
      return Object.fromEntries(heights.map((rows, index) => [rows, compiled[index]]));
    };
    const [relative, bond, zInitShader, addChunk, recycleNorm, recycleProject] = await Promise.all([
      perHeight("esmfold2-rel", (rows) => createRelativePositionShader(
        { pairs: rows, channels, entityBase: layout.entityBase })),
      perHeight("esmfold2-bond", (rows) => createBondShader({ pairs: rows, channels })),
      perHeight(`esmfold2-zinit:${tokens}`, (rows) => createZInitShader(
        { tokens, channels, chunkRows: rows, hasBonds: true, hasLanguageModel: true })),
      perHeight("esmfold2-add", (rows) => createAddShader(rows * channels)),
      perHeight("esmfold2-recycle-norm", (rows) => createLayerNormShader({ rows, channels }, true, 1e-5)),
      perHeight("esmfold2-recycle-project", (rows) => createLinearShader(
        { rows, inner: channels, outer: channels }, true)),
    ]);
    const bytesOf = (allocation, offset, size) => ({ buffer: allocation.buffer, byteOffset: offset, byteSize: size });
    const eachChunk = async (work) => {
      for (let start = 0; start < pairs; start += chunkRows) {
        // eslint-disable-next-line no-await-in-loop
        await work(start, Math.min(chunkRows, pairs - start));
      }
    };

    // 🔴 THE RELATIVE-POSITION ENCODING IS BUILT WHEN IT IS WANTED, NOT HELD. It is
    // pair-sized - 87.9 MiB at 300 tokens - and it has three readers: `z_init`, the
    // diffusion conditioning after 96 block evaluations during which it would sit
    // idle, and the confidence head after the sampler. Building it is one
    // elementwise pass over a small table, about four milliseconds. So the table,
    // its weights and the bonds' outlive `z_init` where its other inputs do not.
    const relBins = keep(allocator.upload("esmfold2.rel-bins", relativeRows(features, tokens), storage));
    const relWeights = keep(allocator.upload("w.esmfold2.rel-pos", weights.featuriser.relPos, storage));
    const bondValues = keep(allocator.upload("esmfold2.bonds", features.tokenBonds, storage));
    const bondWeights = keep(allocator.upload("w.esmfold2.token-bonds", weights.featuriser.tokenBonds, storage));
    const relativeInto = (target, start, rows) =>
      ["rel-pos", relative[rows], [bytesOf(relBins, start * 16, rows * 16), relWeights, target],
       ...elementwise(rows * channels)];
    const bondsInto = (target, start, rows) =>
      ["bonds", bond[rows], [bytesOf(bondValues, start * 4, rows * 4), bondWeights, target],
       ...elementwise(rows * channels)];
    const buildRelativePositions = async (target) => {
      await eachChunk((start, rows) => submit("esmfold2.rel-pos",
        [relativeInto(bytesOf(target, start * channels * 4, rows * channels * 4), start, rows)], false));
      await device.queue.onSubmittedWorkDone();
    };

    const pairBytes = pairs * channels * 4;
    // Whether what a stage only reads waits on the host, which costs seconds of traffic and gives back a whole
    // pair-sized tensor twice over. See webgpu/esmfold2/memory-plan.js.
    const spill = options.spillPairs ?? esmfold2Spills({ tokens, channels, ...memoryTotals(device) });
    let zInit = reuse !== undefined ? undefined
      : keep(allocator.allocate("esmfold2.z-init", pairBytes,
          storage | (options.returnConfidenceInputs === true || spill ? GPUBufferUsage.COPY_SRC : 0)));
    if (reuse === undefined) await mark("z_init", async () => {
      // 🔴 PROJECT THEN BROADCAST. The two per-token projections are 2n rows of
      // 451 channels on the host; broadcasting first would make them 2n^2.
      const rowTerm = keep(allocator.upload("esmfold2.z-rows",
        linear(sInputs, tokens, shape.singleInputs, channels, weights.featuriser.zInit1), storage));
      const columnTerm = keep(allocator.upload("esmfold2.z-columns",
        linear(sInputs, tokens, shape.singleInputs, channels, weights.featuriser.zInit2), storage));
      const relativeTerm = keep(allocator.allocate("esmfold2.rel-pos", chunkRows * channels * 4, storage));
      const bondTerm = keep(allocator.allocate("esmfold2.bond-term", chunkRows * channels * 4, storage));
      const firstRow = keep(allocator.allocate("esmfold2.z-start", 16, storage | GPUBufferUsage.COPY_DST));
      await eachChunk(async (start, rows) => {
        device.queue.writeBuffer(firstRow.buffer, 0, Uint32Array.of(start));
        await submit("esmfold2.z-init", [
          relativeInto(relativeTerm, start, rows),
          bondsInto(bondTerm, start, rows),
          ["z-init", zInitShader[rows],
           [rowTerm, columnTerm, relativeTerm, bondTerm,
            bytesOf(lmPair, start * channels * 4, rows * channels * 4),
            bytesOf(zInit, start * channels * 4, rows * channels * 4), firstRow],
           ...elementwise(rows * channels)],
        ], false);
      });
      await device.queue.onSubmittedWorkDone();
      for (const allocation of [rowTerm, columnTerm, relativeTerm, bondTerm, firstRow, lmPair]) {
        allocation.release();
        held.splice(held.indexOf(allocation), 1);
      }
    });
    await settleReleasedMemory(device);

    // ...and the trunk's INPUT, for the oracle arm: comparing z_init separates a
    // trunk that diverges from a trunk that is fed something different. Only
    // when asked; it is `tokens^2 x 256` floats.
    if (options.returnConfidenceInputs === true) {
      const back = allocator.allocate("esmfold2.z-init-rb", pairs * channels * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      const encoder = device.createCommandEncoder({ label: "esmfold2.z-init-rb" });
      encoder.copyBufferToBuffer(zInit.buffer, 0, back.buffer, 0, pairs * channels * 4);
      device.queue.submit([encoder.finish()]);
      await back.buffer.mapAsync(GPUMapMode.READ);
      embedderZInit = new Float32Array(back.buffer.getMappedRange().slice(0));
      options.onZInit?.(embedderZInit);
      back.buffer.unmap();
      back.release();
    }

    // z_init has one reader, once a loop, a chunk of rows at a time. It can wait on the host between them.
    const zInitHost = !spill || zInit === undefined ? undefined
      : await spillToHost(device, allocator, zInit.buffer, pairBytes, "esmfold2.z-init");
    if (zInitHost !== undefined) {
      zInit.release();
      held.splice(held.indexOf(zInit), 1);
      zInit = undefined;
      await settleReleasedMemory(device);
    }
    const pair = keep(allocator.allocate("esmfold2.pair", pairBytes,
      storage | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST));

    advance(plan.embedder);

    // ---- the trunk, `loops` times over a pair that stays on the device.
    // 🔴 THE BAR MOVES PER BLOCK AND THE LINE SAYS "Trunk", ONCE. Ninety-six
    // block evaluations is the one part of this fold with enough events to make
    // a bar move smoothly, and none of them is worth a line of its own.
    const trunkBlocks = weights.trunkBlocks.length;
    const perBlock = plan.trunk / (loops * Math.max(1, trunkBlocks));
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
    const zInitChunk = zInitHost === undefined ? undefined
      : keep(allocator.allocate("esmfold2.z-init-chunk", chunkRows * channels * 4, storage | GPUBufferUsage.COPY_DST));
    // 🔴 `z` STARTS AT ZERO AND `pair_loop_proj(0)` IS NOT ZERO. Its Linear is
    // zero-INITIALISED upstream and then trained, and the LayerNorm in front of
    // it has an offset - so the first loop's input is z_init plus a real
    // vector. Skipping the projection on the first loop is the natural
    // shortcut and a different model.
    // ...zeroed on the device: a host array of zeros is 2 GiB at 1448 tokens,
    // which the browser refused ("Array buffer allocation failed").
    {
      const encoder = device.createCommandEncoder({ label: "esmfold2.pair-zero" });
      encoder.clearBuffer(pair.buffer, 0, pairs * channels * 4);
      device.queue.submit([encoder.finish()]);
    }
    const trunk = new Esmfold2TrunkGpu(device, { allocator, sequentialTriangle: true, ...options.trunk });
    keep({ release: () => trunk.releaseHeldScratch() });
    const slice = (allocation, row, rows) => ({
      buffer: allocation.buffer, byteOffset: row * channels * 4,
      byteSize: rows * channels * 4,
    });
    // 🔴 THE LOOPS, OR THE PAIR THEY WOULD HAVE PRODUCED. On reuse the cached
    // pair is uploaded straight into the buffer the sampler reads, which is the
    // whole saving: 96 block evaluations at 256 channels, 85% of a 300-token
    // fold. `z_init` does not run at all then.
    if (reuse !== undefined) {
      device.queue.writeBuffer(pair.buffer, 0, reuse.pair);
    } else {
      for (let loop = 0; loop < loops; loop += 1) {
        // ...named per pass, as AF3's line is. The recycle is not announced: it
        // is two milliseconds and it is the seam between two passes, not a phase.
        enter(trunkPhase(loop, loops));
        await mark(`recycle ${loop}`, async () => {
          await eachChunk(async (start, rows) => {
            let term;
            if (zInitHost === undefined) {
              term = slice(zInit, start, rows);
            } else {
              zInitHost.writeTo(device, zInitChunk.buffer, 0, start * channels * 4, rows * channels * 4);
              term = bytesOf(zInitChunk, 0, rows * channels * 4);
            }
            await submit("esmfold2.recycle", [
              ["norm", recycleNorm[rows], [slice(pair, start, rows), recycleScale, recycleOffset,
                                           recycleScratch], ...perRow(rows)],
              ["project", recycleProject[rows], [recycleScratch, recycleWeights, term, slice(pair, start, rows)],
               ...linearGrid(rows, channels)],
            ], false);
            // Bounded, so the host's writes above do not pile up as staging memory.
            if (zInitHost !== undefined && (start / chunkRows) % 16 === 15) await device.queue.onSubmittedWorkDone();
          });
        });
        await mark(`trunk ${loop}`, () => trunk.run(
          { buffer: pair, maskBuffer: pairMask },
          weights.trunkBlocks,
          { n: tokens, channels, readback: false, holdScratch: true,
            submissionWindow: options.submissionWindow,
            // 🔴 THE BAR READS `onBlockDone`, NOT `onBlock`. The first fires at
            // ENCODE time, sixteen at a stride, so a bar driven by it sprints
            // through a submission window and then sits still. This one settles
            // when the device has actually finished the block.
            onBlockDone: () => advance(perBlock),
            // ...and the encode-time one is where the page gets a chance to
            // paint. A GPU promise resolves as a microtask, which never returns
            // control to the browser; `yieldToBrowser` posts a MessageChannel
            // message, which is a task and is not clamped in a background tab.
            onBlock: async (index) => {
              await yieldToBrowser();
              stopIfAsked();
              return options.onBlock?.(loop, index);
            } }));
      }
      recycleScratch.release();
    }
    held.splice(held.indexOf(recycleScratch), 1);
    trunk.releaseHeldScratch();
    if (zInitChunk !== undefined) {
      zInitChunk.release();
      held.splice(held.indexOf(zInitChunk), 1);
    }
    // `onTrunkPair(pair)`: the trunk's output pair read back - a comparison aid (EF2_SAVE_PAIR), only when asked
    if (options.onTrunkPair !== undefined) {
      const back = allocator.allocate("esmfold2.pair-rb", pairs * channels * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      const encoder = device.createCommandEncoder({ label: "esmfold2.pair-rb" });
      encoder.copyBufferToBuffer(pair.buffer, 0, back.buffer, 0, pairs * channels * 4);
      device.queue.submit([encoder.finish()]);
      await back.buffer.mapAsync(GPUMapMode.READ);
      options.onTrunkPair(new Float32Array(back.buffer.getMappedRange().slice(0)));
      back.buffer.unmap();
      back.release();
    }

    // 🔴 z_init IS DEAD WHEN THE LOOP ENDS AND THE FOLD'S PEAK IS AFTER IT.
    // `z = z_init + pair_loop_proj(z)` is its only reader, once a loop; it was
    // held to the end of the fold for nothing. It is PAIR-SIZED - 87.9 MiB at
    // 300 tokens. Same shape as AF3's releaseResidentWeights: give a stage's
    // memory back when the stage is over, before reaching for kernels.
    if (zInit !== undefined) {
      zInit.release();
      held.splice(held.indexOf(zInit), 1);
    }
    await settleReleasedMemory(device);

    // 🔴 READ BACK ONLY WHEN A CALLER ASKS, because this is the one thing the
    // fold otherwise never does: the pair stays on the device from z_init to
    // the sampler, which is why the trunk's four loops cost no traffic at all.
    // One readback at the end is a quarter of what looping through the host
    // would have cost, and it buys a fold that changes only its sampler.
    // 🔴 AND NOT PAST A GIGABYTE, where the browser will not hand out the array
    // (`HOST_PAIR_MAX_BYTES`; AF3's fold.js has the same rule and the same
    // failure). A trunk that cannot be cached is simply not offered for reuse.
    const hostPairFits = pairs * channels * 4 <= HOST_PAIR_MAX_BYTES;
    const reusable = options.wantReusable !== true || !hostPairFits ? undefined : {
      tokens, channels, sInputs,
      pair: await (async () => {
        const back = allocator.allocate("esmfold2.pair-readback", pairs * channels * 4,
          GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
        const encoder = device.createCommandEncoder({ label: "esmfold2.pair-readback" });
        encoder.copyBufferToBuffer(pair.buffer, 0, back.buffer, 0, pairs * channels * 4);
        device.queue.submit([encoder.finish()]);
        await back.buffer.mapAsync(GPUMapMode.READ);
        const copy = new Float32Array(back.buffer.getMappedRange().slice(0));
        back.buffer.unmap();
        back.release();
        return copy;
      })(),
    };

    // 🔴 THE CONFIDENCE HEAD READS THE TRUNK'S PAIR AND THE SAMPLER'S
    // COORDINATES, and those two do not exist at the same moment. The pair's
    // allocation is gone by the time the structure is - binding it at the end
    // submits against a destroyed buffer, which is the allocator doing its job
    // - so the head's copy is taken HERE, beside the `wantReusable` one above
    // and by the same route. It costs `tokens^2 x 256` of traffic once a fold,
    // and only when the bundle carries a head at all.
    // 🔴 A DEVICE COPY, NOT A HOST ONE, unless a memory ceiling cannot spare it or
    // the caller wants the head's inputs back as arrays. The head binds it; a host
    // copy was a 64 MiB readback at 255 tokens followed by the same upload - with
    // the relative encoding's, 3.96 -> 3.56 s a warm fold. Where the card is tight
    // it waits on the host in pieces, which a pair past a gigabyte can do where one
    // array cannot.
    const hostConfidence = options.confidenceWeights != null && options.returnConfidenceInputs === true;
    const deviceConfidence = options.confidenceWeights != null && !hostConfidence && !spill;
    const deviceCopy = (label, source) => {
      const copy = allocator.allocate(label, pairBytes, storage | GPUBufferUsage.COPY_DST);
      const encoder = device.createCommandEncoder({ label });
      encoder.copyBufferToBuffer(source.buffer, 0, copy.buffer, 0, pairBytes);
      device.queue.submit([encoder.finish()]);
      return copy;
    };
    let confidencePairOnDevice = deviceConfidence ? deviceCopy("esmfold2.confidence-pair", pair) : undefined;
    let confidencePairSpill = options.confidenceWeights != null && !hostConfidence && spill
      ? await spillToHost(device, allocator, pair.buffer, pairBytes, "esmfold2.confidence-pair") : undefined;
    const confidencePair = !hostConfidence ? undefined
      : await (async () => {
        const back = allocator.allocate("esmfold2.confidence-pair", pairBytes,
          GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
        const encoder = device.createCommandEncoder({ label: "esmfold2.confidence-pair" });
        encoder.copyBufferToBuffer(pair.buffer, 0, back.buffer, 0, pairBytes);
        device.queue.submit([encoder.finish()]);
        await back.buffer.mapAsync(GPUMapMode.READ);
        const copy = new Float32Array(back.buffer.getMappedRange().slice(0));
        back.buffer.unmap();
        back.release();
        return copy;
      })();

    // ---- the distogram, which is the trunk's one output besides the pair.
    // 🔴 IT RUNS BEFORE THE DIFFUSION MODULE ALLOCATES, not after the fold.
    // Its symmetrised copy of the pair is another 92 MiB at 300 tokens, and the
    // denoiser is holding its own pair conditioning and twelve bias tensors by
    // then - so running it here costs nothing and running it at the end raises
    // the peak by a whole pair representation.
    const distogram = options.contacts === false ? undefined
      : await mark("distogram", () => (enter(ESMFOLD2_PHASES.conditioning),
        encodeContactMap(
        { device, allocator, cache, submit },
        { tokens, channels, bins: shape.distogramBins, pair,
          weights: weights.featuriser.distogramWeights,
          bias: weights.featuriser.distogramBias,
          partners: partnerKeys(features, tokens),
          // ...so a covalently attached ligand is not scored on the bond it
          // was given. See the note in createCertaintyShader.
          bonds: features.tokenBonds,
          molType: features.molType, residueType: features.residueType,
          wantLogits: options.distogramLogits === true,
          // 🔴 THE pAE NEEDS THE STRUCTURE, SO ONLY ITS INPUTS COME OFF THE
          // TRUNK HERE. The moments are three floats a pair and are computed on
          // the device beside the contacts; the estimate itself is assembled
          // after the sampler, because half its features are distances.
          // 🔴 OPT-IN, because nothing draws it. See aligned-error.js: the
          // estimate inverts across folds, so the page does not show it, and a
          // pass plus three pair-sized buffers for a matrix nobody reads is
          // pure cost. `tools/pae-transfer.py` is what asks for it.
          wantMoments: options.alignedError === true,
          retainForFrames: options.frameCertainty === true })));
    const contacts = distogram?.contacts;
    const distogramMoments = distogram?.moments;
    const certainty = distogram?.certainty;
    // ...the same reading across the interface, -1 per token where there is
    // no other chain. A monomer's is all -1 and means nothing.
    const interfaceCertainty = distogram?.interface;
    const frames = distogram?.frames;
    // 🔴 THE CERTAINTY GOES OUT WITH THE CONTACTS, BEFORE THE SAMPLER RUNS.
    // Both come off the trunk's distogram, so a caller colouring its live
    // frames has them from the first one - and the first version assigned it
    // only from the RESULT, which left every frame but the last at a zero
    // B-factor and therefore the colour of no confidence at all.
    await options.onContacts?.(contacts, certainty);

    // ---- the sampler.
    // ...and the relative-position encoding comes back, into the buffer that
    // becomes the pair conditioning. The table and its weights are what were
    // kept through the trunk in its place; see where it is first built.
    // ...COPY_SRC, because the confidence head's copy is taken off this buffer
    // below and the conditioning overwrites it immediately after.
    const relPos = keep(allocator.allocate("esmfold2.rel-pos", pairBytes, storage | GPUBufferUsage.COPY_SRC));
    await buildRelativePositions(relPos);
    // Once, from whichever comes first: the head after its pair init, or the fold's end.
    const releaseConfidenceCopies = () => {
      confidencePairOnDevice?.release();
      confidencePairOnDevice = undefined;
      confidencePairSpill = undefined;
    };
    // 🔴 THE HEAD'S BIAS IS BUILT WHEN THE HEAD WANTS IT, a chunk at a time. Their
    // head adds the relative encoding to the NORMALISED pair, and it is the term
    // this port once omitted entirely - an optional argument defaulting to None on
    // their side, so nothing objected. Measured absent: 13.8 pLDDT and a PAE 2.4x
    // rougher. See docs/EF2FAST.md. It was copied here and held through the sampler,
    // beside three other pair-sized tensors; the table it comes from is 1/64 the size.
    //
    // 🔴 AND THE BOND ENCODING GOES INTO THE SAME TERM, because their head adds BOTH
    // to the normalised pair - `if relative_position_encoding is not None` and `if
    // token_bonds_encoding is not None`, one after the other - and a sum of two
    // addends is one addend, which keeps the pair-init kernel at eight storage
    // bindings, WebGPU's floor. It is zero on a plain chain and not on a ligand.
    // The product stored and then added, never fused: the host's f32 sum.
    let confidenceBondTerm;
    keep({ release: () => confidenceBondTerm?.release() });
    const fillConfidenceBias = async (start, rows, target) => {
      confidenceBondTerm ??= allocator.allocate("esmfold2.confidence-bonds", chunkRows * channels * 4, storage);
      await submit("esmfold2.confidence-bias", [
        relativeInto(target, start, rows),
        bondsInto(confidenceBondTerm, start, rows),
        ["bonds-add", addChunk[rows], [bytesOf(target, 0, rows * channels * 4),
                                       bytesOf(confidenceBondTerm, 0, rows * channels * 4)],
         ...elementwise(rows * channels)],
      ], false);
    };
    const confidenceRelPos = !hostConfidence ? undefined
      : await (async () => {
        const back = allocator.allocate("esmfold2.confidence-relpos", pairBytes,
          GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
        const encoder = device.createCommandEncoder({ label: "esmfold2.confidence-relpos" });
        encoder.copyBufferToBuffer(relPos.buffer, 0, back.buffer, 0, pairBytes);
        device.queue.submit([encoder.finish()]);
        await back.buffer.mapAsync(GPUMapMode.READ);
        const copy = new Float32Array(back.buffer.getMappedRange().slice(0));
        back.buffer.unmap();
        back.release();
        return copy;
      })();
    if (confidenceRelPos !== undefined) {
      const bondTerm = allocator.allocate("esmfold2.confidence-bonds", pairBytes, storage | GPUBufferUsage.COPY_SRC);
      await eachChunk((start, rows) => submit("esmfold2.confidence-bonds",
        [bondsInto(bytesOf(bondTerm, start * channels * 4, rows * channels * 4), start, rows)], false));
      const back = allocator.allocate("esmfold2.confidence-bonds-rb", pairBytes,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      const encoder = device.createCommandEncoder({ label: "esmfold2.confidence-bonds" });
      encoder.copyBufferToBuffer(bondTerm.buffer, 0, back.buffer, 0, pairBytes);
      device.queue.submit([encoder.finish()]);
      await back.buffer.mapAsync(GPUMapMode.READ);
      const added = new Float32Array(back.buffer.getMappedRange());
      for (let index = 0; index < confidenceRelPos.length; index += 1) {
        confidenceRelPos[index] += added[index];
      }
      back.buffer.unmap();
      for (const allocation of [back, bondTerm]) allocation.release();
    }

    // 🔴 THE ELEMENT THE TOKEN TRANSFORMER'S WEIGHTS ARE HELD IN. Twelve
    // blocks are 459 MiB of a 799 MiB fold; see the note in
    // webgpu/esmfold2/diffusion.js for why this stack takes f16 and the
    // atom stacks do not.
    await mark("conditioning", () => denoiser.prepare({
      shape: denoiserShape,
      // 🔴 AND THE CONDITIONING IS WRITTEN BACK INTO relPos. It is the last
      // pass that reads it, chunk by chunk, and the fold is at its fullest
      // exactly there - three pair-sized f32 tensors where two will do. See
      // prepare() in webgpu/esmfold2/diffusion.js for why the row chunking
      // makes that safe. And once the pair bias is built from it nothing reads
      // it again, so prepare() hands it back through releaseRelPos below.
      weights: weights.denoiser, features, sInputs, pair, relPos, reuseRelPos: true,
      // 🔴 AND THE TRUNK'S PAIR IS DEAD ONCE THE CONDITIONING HAS READ IT. The
      // distogram head ran before the denoiser and nothing after the sampler
      // touches it, so the fold holds ONE pair-sized tensor through the
      // sampler where it held three.
      releasePair: () => {
        pair.release();
        held.splice(held.indexOf(pair), 1);
      },
      releaseRelPos: () => {
        relPos.release();
        held.splice(held.indexOf(relPos), 1);
      },
    }));

    // 🔴 THE DISTOGRAM IS OVER THE REPRESENTATIVE ATOM - CB, or CA for glycine,
    // or a ligand token's only atom - so a frame is scored on those and not on
    // alpha carbons. Gathered once: the slots do not move, only the coordinates.
    const representative = frames === undefined ? undefined
      : representativeAtoms(features, tokens);

    advance(plan.conditioning);
    enter(ESMFOLD2_PHASES.sampler);
    const draw = gaussians(options.seed ?? 0);
    let x = new Float32Array(atoms * 3);
    for (let i = 0; i < x.length; i += 1) x[i] = schedule[0] * draw();
    for (let step = 0; step < levels.length; step += 1) {
      stopIfAsked();
      const at = performance.now();
      x = centreRandomAugmentation(x, features.mask, atoms, draw);
      const tHat = levels[step];
      const epsilon = settings.noiseScale
        * Math.sqrt(Math.max(tHat * tHat - schedule[step] * schedule[step], 0));
      const noisy = new Float32Array(x.length);
      for (let i = 0; i < noisy.length; i += 1) noisy[i] = x[i] + epsilon * draw();
      const denoised = await denoiser.denoise(noisy, tHat);
      options.onDenoise?.(step, noisy, denoised, tHat);     // (a comparison aid: one denoiser call's input and output)
      x = samplerStep(noisy, denoised, features.mask, atoms, tHat,
                      schedule[step + 1], settings.stepScale, settings.update);
      // 🔴 SCORED ON THE DENOISED PREDICTION, WHICH IS WHAT IS DRAWN. The
      // sampler's own state is Gaussian noise at the top of the schedule and is
      // not what a viewer shows; colouring the state while drawing the
      // prediction would put one frame's colour on another frame's structure.
      const frameCertainty = frames === undefined ? undefined
        : await frames.score(gatherPositions(denoised, representative, tokens));
      timings[`sampler ${step}`] = performance.now() - at;
      advance(plan.perStep);
      // 🔴 BOTH, BECAUSE A VIEWER WANTS THE ONE THE SAMPLER DOES NOT KEEP.
      // `coordinates` is the trajectory state at the NEXT noise level - what the
      // sampler carries forward - and at the top of the schedule that is
      // Gaussian noise at sigma 411, which is not a picture of anything.
      // `denoised` is the model's predicted structure at this call, EDM
      // preconditioning included, and is protein-sized in every frame.
      await options.onStep?.({ step, total: levels.length, coordinates: x,
                               denoised, features, certainty: frameCertainty });
    }
    // 🔴 THE HEAD RUNS BEFORE THE RELEASES, because it BINDS the trunk's own
    // pair rather than a copy of it. Placed after them it submitted against a
    // destroyed buffer - "[Buffer \"esmfold2.pair\"] used in submit while
    // destroyed" - which is the allocator doing exactly what it should.
    // 🔴 A REAL CONFIDENCE HEAD WHERE THE BUNDLE CARRIES ONE, and the certainty
    // estimate where it does not. biohub's ESMFold2 has `confidence_head.enabled:
    // false` and zero confidence tensors, which is what `alignedError` above
    // exists for; Synthyra froze that trunk and trained a head on it, and a
    // bundle exported with `--confidence` carries it. Both are returned, so a
    // caller can prefer the real one and nothing that read the estimate breaks.
    // 🔴 THE SAMPLER'S BUFFERS GO BEFORE THE HEAD RUNS, not after it: the
    // conditioning (written into relPos) and the twelve bias tensors are dead
    // once the last step is in, and the head reads its own copies taken before
    // the sampler. Held through it, they were 3.5 GB of ESMFold2's 16.9 GB
    // peak at 1448 tokens, inside the head's block stack.
    const memory = allocator.snapshot();
    denoiser.release();
    if (held.includes(relPos)) {
      relPos.release();
      held.splice(held.indexOf(relPos), 1);
    }
    await settleReleasedMemory(device);
    let confidence;
    if (options.confidenceWeights != null) {
      const rep = representativeAtoms(features, tokens);
      confidence = await esmfold2ConfidenceFold(device, {
        tokens, atoms, sInputs, pair: confidencePair, coordinates: x, repAtom: rep,
        pairBuffer: confidencePairOnDevice?.buffer,
        pairSpill: confidencePairSpill,
        spillInitial: spill,
        atomToToken: features.atomToToken,
        atomMask: features.mask,
        tokenMask: new Float32Array(tokens).fill(1),
        pairBias: confidenceRelPos, fillPairBias: hostConfidence ? undefined : fillConfidenceBias,
        releaseInputs: () => { releaseConfidenceCopies(); confidenceBondTerm?.release(); confidenceBondTerm = undefined; },
        // 🔴 WITHOUT THIS EVERY COMPLEX REPORTS ipTM 0.000. ipTM is the same
        // expectation as pTM taken over the pairs whose `asymId` DIFFER, and
        // the reference defaulted a missing one to `new Int32Array(tokens)` -
        // one chain - so no pair was ever inter-chain and the score was
        // structurally zero rather than unknown. Measured on two copies of a
        // 58-mer: 0.000 before, and the page showed it as the model's opinion
        // of its own interface. Single-chain folds cannot see this, which is
        // every fold this head was gated on.
        asymId: features.asymId,
      }, options.confidenceWeights, { allocator });
      releaseConfidenceCopies();
      for (const allocation of [relBins, relWeights, bondValues, bondWeights]) {
        allocation.release();
        held.splice(held.indexOf(allocation), 1);
      }
      // 🔴 THE HEAD'S OWN INPUTS, FOR THE ARM THAT RUNS SYNTHYRA'S MODULE ON A
      // REAL TRUNK. `dump_esmfold2_confidence.py` gates the head's ARITHMETIC
      // on seeded normals - its own header says a run on a real trunk's pair is
      // a second arm "that wants our own fold to dump one" - so this is that
      // dump. Off unless asked: the pair alone is tokens^2 x 256 floats.
      if (options.returnConfidenceInputs === true) {
        confidence = { ...confidence, inputs: {
          tokens, atoms, pair: confidencePair, sInputs, coordinates: x,
          repAtom: rep, atomToToken: features.atomToToken,
          atomMask: features.mask, asymId: features.asymId,
          pairBias: confidenceRelPos,
          // ...and the ATOM ENCODER's own inputs, so their InputsEmbedder can
          // be run on ours and the encoder separated from what it is fed. The
          // featuriser's own channels already match to the bit; `ref_pos` is
          // the one feature that cannot, because each side draws its own
          // conformer per residue instance.
          refPos: features.refPos, refCharge: features.refCharge,
          refElement: features.refElement,
          refAtomNameChars: features.refAtomNameChars,
          refSpaceUid: features.refSpaceUid,
          aatype: features.aatype, profile: features.profile,
          deletionMean: features.deletionMean,
          atomConditioning: embedderConditioning,
          zInit: embedderZInit,
          atomActivation: embedderCapture?.activation,
        } };
      }
    }

    // ...the retained distogram is 46 MiB at 300 tokens and nothing reads it
    // after the last frame.
    frames?.release();

    // 🔴 THE pAE IS ASSEMBLED AFTER THE SAMPLER, because half its features are
    // DISTANCES and those do not exist until there is a structure. The other
    // half came off the trunk with the contacts. See aligned-error.js for what
    // it is and what it is not: a predicted aligned error, fitted against
    // AlphaFold 3's, and a CONTRAST rather than a calibrated angstrom.
    let alignedError;
    if (distogramMoments !== undefined) {
      // ...between the REPRESENTATIVE atoms the distogram is defined on, not
      // alpha carbons: a distogram scores the pair it was trained on, and
      // measuring a different pair is a plausible matrix computed from the
      // wrong geometry.
      const rep = representativeAtoms(features, tokens);
      const separations = new Float32Array(tokens * tokens);
      for (let i = 0; i < tokens; i += 1) {
        for (let j = 0; j < tokens; j += 1) {
          const a = rep[i] * 3, b = rep[j] * 3;
          separations[i * tokens + j] = Math.hypot(
            x[a] - x[b], x[a + 1] - x[b + 1], x[a + 2] - x[b + 2]);
        }
      }
      alignedError = alignedErrorFromDistogram(distogramMoments, separations, tokens);
    }

    return {
      coordinates: x, features, sequence, tokens, atoms, sInputs, contacts, certainty,
      interfaceCertainty, alignedError, confidence, reusable,
      // ...so a caller can tell a fold that ran the trunk from one that did not.
      trunkReused: reuse !== undefined,
      lmMask: { fraction: maskFraction, masked: maskedTokens, of: lm.ids.length },
      languageModel: options.languageModel !== false,
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
