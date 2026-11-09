/**
 * An AlphaFold 3 fold in WebGL2: a featurised batch and a real checkpoint in,
 * coordinates and confidence out.
 *
 *     target_feat  = per-atom conditioning -> atom encoder -> aggregate
 *     per pass:      embed -> template -> 4 x msaBlock -> 48 x pairformerBlock
 *     sample:        the EDM walk, each step one diffusion head call
 *     confidence:    pLDDT, PAE, PDE, and pTM off the PAE logits
 *
 * 🔴 THIS IS THE FIRST TIME THE WebGL2 TREE FOLDS AF3 FROM A CHECKPOINT, AND
 * THAT IS A DIFFERENT TEST FROM EVERY STAGE GATE BEFORE IT. All of them drive
 * one stage with NOISE, and docs/AF2.md records what that misses: "a packing
 * that moved the real model 4e-4 read 6e-7 on noise". Real weights are not
 * uniform, so a layout or a transpose that noise forgives is visible here.
 *
 * 🔴 target_feat IS NOT A FEATURE, IT IS A FORWARD PASS. 384 of its 447 columns
 * come from the atom cross-attention ENCODER run with the noisy positions
 * ZEROED and no trunk conditioning - the same module the diffusion head runs
 * with both - so a fold that treats it as featuriser output silently drops a
 * third of the model's input. The three terms this encoder does not have
 * contribute nothing because their weights are zero, not because they are
 * skipped.
 *
 * 🔴 AND THE TRUNK'S SINGLE AND THE HEAD'S CONDITIONING ARE DIFFERENT TENSORS,
 * both of which AF3 uses - see the note in diffusion/head.js. This file passes
 * `trunk.single` to the encoder and the head computes its own.
 *
 * 🔴 AND A RECYCLE RUNS THE WHOLE TRUNK AGAIN, EMBEDDER AND ALL. That is what a
 * recycle IS: the embedder adds `prev_embedding(LayerNorm(previous pair))` and
 * the previous pass's pair and single are what fill those slots. Two passes cost
 * two trunks, and the state that crosses between them is exactly those two
 * tensors.
 *
 * 🔴 AND THE CONFIDENCE HEAD READS THE SAMPLED COORDINATES, through the batch's
 * own `token_atoms_to_pseudo_beta` gather - the representative atom a token -
 * which is a different gather from any the diffusion uses. Handing it the dense
 * atom array would put a ligand's geometry one place to the left of everything
 * after the first one.
 *
 * What this does NOT do, and says so rather than approximating: OpenDDE's
 * structural tokens (a second token space with its own expander, refiner and
 * confidence head), boltz2's nine-term confidence re-embedding, and the device
 * weight streaming `webgpu/`'s fold does - there is nothing here to stream,
 * because every tensor is a texture allocated up front.
 */
import {
  af3ContactBins, af3ContactClasses,
} from "../../../shared/af3/featurise/contact-classes.js";
import { targetFeatures } from "../../../cpu/af3/diffusion/atom-encoder.js";
// 🔴 THE PAGE'S REDUCTION, NOT cpu/af3/confidence/ptm.js's. That one anchors
// pTM on every live token and reports no chain scores at all, so this path -
// which tools/gpu/fold-af3-webgl.js and the four WebGL2 probes fold through -
// could not see the chain-pTM fix (reported by Yeqing Lin) even after the page's
// own WebGL2 fold had it. `af3TmScores` is the definition; the terms either side
// of it are the ones foldBatchWebGl2 and webgpu/af3/fold.js build.
import { af3TmScores, chainTmInfo, interfaceTmTerm, tmPerBinFor, tmScoreD0, tmTermFromLogits, tokenFrameMask }
  from "../../../shared/heads/tm-score.js";
// 🔴 THE SAME DRAWS THE WebGPU FOLD USES, so the two backends' coordinates can
// be compared at all: "a reproducible normal deviate, so a seed names a
// structure". A second Box-Muller here would make every fold a different sample
// from the same distribution and leave nothing to compare.
import { normalFrom } from "../../../shared/af3/random.js";
import { Execution, memoSource } from "../../runtime/execution.js";
import { atomCrossAttentionEncoder } from "../diffusion/atom-encoder.js";
import { perAtomConditioning } from "../diffusion/atom-conditioning.js";
import { confidenceHeadSteps, errorBinCentres, MAX_ERROR_BIN, NUM_BINS } from "../confidence/confidence.js";
import { embed, RELATIVE_WIDTH, tokenIdentity } from "../trunk/embedder.js";
import { binEdges, runTrunkSteps } from "../trunk/trunk.js";
import { sampleOnWebGl2Steps } from "../diffusion/sampler.js";
import { drain, drainAsync, gpuIdle, laggedPause } from "../../runtime/steps.js";
import {
  slotCodes, slotGeometry, templateEmbeddingSteps, uploadTemplateWeights,
  releaseTemplateWeights,
} from "../trunk/template.js";
import { uploadPairformerBlockWeights, releasePairformerBlockWeights }
  from "../trunk/pairformer-block.js";
import { uploadEmbedderWeights, releaseEmbedderWeights } from "../trunk/embedder.js";
import { COND_BATCH_KEYS, condBatchWeights, fuseQkvgWeights } from "../diffusion/transformer.js";
import { uploadMsaBlockWeights, releaseMsaBlockWeights } from "../trunk/msa-block.js";
import { uploadConditioningWeights, releaseConditioningWeights }
  from "../diffusion/atom-conditioning.js";
import { uploadConfidenceHeadWeights, releaseConfidenceHeadWeights }
  from "../confidence/confidence.js";
import { ONE_TARGET, PRELUDE, sized } from "../../runtime/execution.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */
/** @typedef {{indices: ArrayLike<number>, mask: Float32Array, count: number}} Gather */
/** @typedef {{codes: ArrayLike<number>, deletions: ArrayLike<number>, mask: ArrayLike<number>, sequences: number}} MsaRows */
/**
 * The fields of a featurised batch this fold reads, named rather than taken as
 * an `object`: "a key the batch carries and this literal does not name is a key
 * thrown away", and the same applies to reading one.
 *
 * @typedef {{tokens: number, dense: number, subsets: number,
 *   shape: {tokens: number, dense: number, subsets: number, queries: number,
 *     keys: number},
 *   aatype: Int32Array, profile: Float32Array, deletionMean: Float32Array,
 *   isDna: Float32Array, isRna: Float32Array, isLigand: Float32Array,
 *   isModified: Float32Array, cyclicPeriod?: Float32Array,
 *   refPos: Float32Array, refMask: Float32Array, refElement: Float32Array,
 *   refCharge: Float32Array, refAtomNameChars: Float32Array,
 *   refSpaceUid: Float32Array, seqMask: Float32Array, asymId: Int32Array,
 *   features: object, msa: ArrayLike<number>, deletionMatrix: ArrayLike<number>,
 *   msaMask: ArrayLike<number>, bondMatrix?: Float32Array,
 *   tokenAtomsToQueries: Gather, queriesToKeys: Gather,
 *   queriesToTokenAtoms: Gather, tokensToQueries: Gather, tokensToKeys: Gather,
 *   tokenAtomsToPseudoBeta: Gather}} Batch
 */

/**
 * The sampler settings the PAGE folds with, which are not the reference's
 * defaults.
 *
 * 🔴 THE SHIPPED WALK IS BLITZ'S EXPONENTIAL UPDATE ON A GEOMETRIC SCHEDULE,
 * and `cpu/af3/diffusion/diffusion-sampler.js`'s defaults are AF3's own euler
 * and karras - "whose coefficient on x goes NEGATIVE once a step is large (r <
 * 1/3 at S 1.5), so below ~25 steps the walk overshoots: at 8 steps AlphaFold
 * 3's 6MRR is 20.9 A and not a chain, and 0.55 A with this update". A fold here
 * that took the reference's defaults would be a different fold from the site's
 * at every step count anyone actually uses; `webgpu/af3/fold.js` names the same
 * two in `DIFFUSION_DEFAULTS` and the fold tool holds the two lists together.
 */
export const SAMPLER_DEFAULTS = Object.freeze({
  update: "exponential", schedule: "geometric",
});

/** `out[i][axis] = positions[indices[i]][axis] * mask[i]`, over a 3-wide row. */
const pseudoBetaSourceUncached = (tokens) => sized({ ELEMENTS: tokens * 3 }, `${PRELUDE}${ONE_TARGET}
uniform sampler2D positionsTex;
uniform sampler2D indicesTex;
uniform sampler2D maskTex;

float at(int element) {
  int token = element / 3;
  int axis = element - token * 3;
  if (!(fetch1(maskTex, token) > 0.0)) return 0.0;
  return fetch1(positionsTex, int(fetch1(indicesTex, token)) * 3 + axis);
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    at(base),
    base + 1 < ELEMENTS ? at(base + 1) : 0.0,
    base + 2 < ELEMENTS ? at(base + 2) : 0.0,
    base + 3 < ELEMENTS ? at(base + 3) : 0.0);
}`);

/** `pairMask[i][j] = seqMask[i] * seqMask[j]`, on the host: it is read by name. */
export function pairMaskOf(seqMask, tokens) {
  const out = new Float32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) out[i * tokens + j] = seqMask[i] * seqMask[j];
  }
  return out;
}

/**
 * Every weight a fold reads, uploaded.
 *
 * 🔴 A GENERIC TREE UPLOAD, AND THAT IS A DECISION RATHER THAN A SHORTCUT. The
 * loaders in `shared/af3/weights/` already name every tensor in the reference's
 * own spelling - which is why `cpu/af3/`'s modules take their output
 * unchanged - so a per-module uploader here would be a second list of those
 * names and a second chance to drop one. The three modules that DO have their
 * own uploader keep it, because each refuses a missing tensor by name.
 *
 * @param {Execution} execution
 * @param {{trunk: any, diffusion: any, confidence: any, atomReference: any,
 *   targetFeat: any}} weights as `foldWeights` returns them
 */
export function uploadAf3Weights(execution, weights) {
  return drain(uploadAf3WeightsSteps(execution, weights));
}

/**
 * `uploadAf3Weights` as a generator: yields after each module's weights (the embedder, each
 * trunk block, the heads), so a page can upload ~800 MB of them without holding the thread
 * for seconds (`drainAsync`).
 */
export function* uploadAf3WeightsSteps(execution, weights) {
  const seen = [];
  // 🔴 AN ABSENT WEIGHT MUST ARRIVE AS ABSENT, NOT AS null. The loaders write
  // `null` for a tensor a bundle does not carry - chai1's single-cond embedding
  // norm, for one - and every module here tests `=== undefined`, so a forwarded
  // null is "present" and reaches a kernel as a sampler with no texture. That
  // exact trap killed boltz2 upstream in a path it does not even share, so the
  // keys are DROPPED here rather than copied.
  const uploadTree = (value, label) => {
    if (value === undefined || value === null) return undefined;
    if (value instanceof Float32Array) {
      const tensor = execution.uploadWeight(value, label);
      seen.push(tensor);
      return tensor;
    }
    if (value instanceof Int32Array || value instanceof Uint8Array) {
      const tensor = execution.uploadWeight(Float32Array.from(value), label);
      seen.push(tensor);
      return tensor;
    }
    if (Array.isArray(value)) return value.map((v, i) => uploadTree(v, `${label}.${i}`));
    // 🔴 A STRING IS NOT A TREE: `Object.entries("abc")` is three
    // single-character strings, so a generic walk recurses on one for ever and
    // dies with "Maximum call stack size exceeded", naming nothing. A weight
    // tree that carries a `source` string is enough to do it.
    if (typeof value !== "object") return value;
    return Object.fromEntries(Object.entries(value)
      .map(([key, v]) => [key, uploadTree(v, `${label}.${key}`)])
      .filter(([, v]) => v !== undefined));
  };
  const trunk = weights.trunk;
  const piece = { stage: "weights" };
  const embedder = uploadEmbedderWeights(execution, trunk.embedder);
  yield piece;
  const template = trunk.template === undefined || trunk.template === null
    ? undefined : uploadTemplateWeights(execution, trunk.template);
  const msaBlocks = [];
  for (const block of trunk.msaBlocks) {
    msaBlocks.push(uploadMsaBlockWeights(execution, block));
    yield piece;
  }
  const pairformerBlocks = [];
  for (const block of trunk.pairformerBlocks) {
    pairformerBlocks.push(uploadPairformerBlockWeights(execution, block));
    yield piece;
  }
  const distogram = uploadTree(trunk.distogram, "w.af3.distogram");
  // The diffusion transformer is nearly all of the checkpoint (757 MB as floats): a block at a time.
  const diffusion = uploadTree({ ...weights.diffusion, transformer: undefined }, "w.af3.diffusion");
  const hostTransformer = weights.diffusion.transformer;
  if (hostTransformer !== undefined && hostTransformer !== null) {
    const label = "w.af3.diffusion.transformer";
    diffusion.transformer = { ...uploadTree({ ...hostTransformer, superBlocks: undefined }, label), superBlocks: [] };
    // Every block's conditioning projections batched into two weights (`condBatchWeights`), taken off the blocks -
    // only where every super-block is full, since a block's index in them is `superBlock * perSuper + inner`.
    const perSuper = hostTransformer.blocksPerSuperBlock;
    const full = hostTransformer.superBlocks.every((group) => group.blocks.length === perSuper);
    const batched = full
      ? condBatchWeights(hostTransformer.superBlocks.flatMap((group) => group.blocks), hostTransformer.channels)
      : null;
    if (batched !== null) {
      diffusion.transformer.condBatch = uploadTree({
        scaleShiftWeight: batched.scaleShiftWeight, scaleShiftBias: batched.scaleShiftBias,
        gateWeight: batched.gateWeight, gateBias: batched.gateBias,
      }, `${label}.condBatch`);
      yield piece;
    }
    for (const [index, group] of hostTransformer.superBlocks.entries()) {
      const uploadedGroup = { ...uploadTree({ ...group, blocks: undefined }, `${label}.superBlocks.${index}`), blocks: [] };
      for (const [inner, block] of group.blocks.entries()) {
        // ...its four attention projections as one wide weight (`fuseQkvgWeights`), in place of the four
        const fused = fuseQkvgWeights(block);
        let host = fused === null ? block : {
          ...block, qProjection: undefined, kProjection: undefined, vProjection: undefined, gatingQuery: undefined,
          qBias: undefined, qkvgProjection: fused.weight, qkvgBias: fused.bias,
        };
        if (batched !== null) host = { ...host, ...Object.fromEntries(COND_BATCH_KEYS.map((key) => [key, undefined])) };
        uploadedGroup.blocks.push(uploadTree(host, `${label}.superBlocks.${index}.blocks.${inner}`));
        yield piece;
      }
      diffusion.transformer.superBlocks.push(uploadedGroup);
    }
  }
  yield piece;
  const atomReference = uploadConditioningWeights(execution, weights.atomReference);
  const targetFeat = {
    ...uploadTree(
      { ...weights.targetFeat, reference: undefined }, "w.af3.target-feat"),
    reference: uploadConditioningWeights(execution, weights.targetFeat.reference),
    dialect: weights.targetFeat.dialect,
  };
  yield piece;
  let confidence;
  if (weights.confidence !== undefined) {
    const head = uploadConfidenceHeadWeights(execution, weights.confidence);
    const blocks = [];
    for (const block of weights.confidence.blocks) {
      blocks.push(uploadPairformerBlockWeights(execution, block));
      yield piece;
    }
    confidence = { head, blocks };
  }
  const uploaded = {
    dialect: trunk.dialect,
    // 🔴 THE MODULES THAT HAVE THEIR OWN UPLOADER KEEP IT, and the MSA block's
    // is not optional: it TRANSLATES AF3's outer-product-mean weight names into
    // the AF2 kernel's, hoists `outerChannels` to where the block reads it, and
    // fills in the two projection biases rosettafold3 has and AF3 does not.
    // Uploading that tree generically compiles and then refuses with "cOuter
    // must be a multiple of 4, got undefined".
    embedder, template, msaBlocks, pairformerBlocks, distogram,
    // The diffusion head's own tree, and the per-atom conditioning's reference
    // table, which `targetFeat` and the head both read.
    diffusion,
    // 🔴 THE DIFFUSION HAS ITS OWN REFERENCE TABLE, and it is not target_feat's.
    // `atomReference` is "the diffusion head's own copy of it" - same five
    // projections, different trained values, and a bias target_feat's does not
    // carry. Using one for both runs, folds, and conditions every atom on the
    // wrong table.
    atomReference, targetFeat, confidence,
    // The host keeps these: the Fourier embedding is one row and the absent
    // target_feat rows are read as numbers after the global norm.
    hostFourier: {
      weight: weights.diffusion?.conditioning?.fourierWeight,
      bias: weights.diffusion?.conditioning?.fourierBias,
    },
    tensors: seen,
  };
  return uploaded;
}

export function releaseAf3Weights(execution, uploaded) {
  for (const tensor of uploaded.tensors) execution.release(tensor);
  releaseEmbedderWeights(execution, uploaded.embedder);
  for (const block of uploaded.msaBlocks) releaseMsaBlockWeights(execution, block);
  releaseConditioningWeights(execution, uploaded.targetFeat.reference);
  releaseConditioningWeights(execution, uploaded.atomReference);
  if (uploaded.template !== undefined) releaseTemplateWeights(execution, uploaded.template);
  for (const block of uploaded.pairformerBlocks) {
    releasePairformerBlockWeights(execution, block);
  }
  if (uploaded.confidence !== undefined) {
    releaseConfidenceHeadWeights(execution, uploaded.confidence.head);
    for (const block of uploaded.confidence.blocks) {
      releasePairformerBlockWeights(execution, block);
    }
  }
}

/** The dialect flags each stage asks for, read once from the bundle's own. */
export function dialectFlags(dialect = {}) {
  return {
    // The trunk
    pairInitFromSingle: dialect.pairInitFromSingle === true,
    msaUpdateBeforeOuterProduct: dialect.msaUpdateBeforeOuterProduct === true,
    msaPairedQueryRow: dialect.msaPairedQueryRow === true,
    swapTransposedBias: dialect.swapTransposedBias === true,
    divideByLength: dialect.triangleMulDivideByLength === true,
    opmBiasAfterNorm: dialect.outerProductMeanBiasAfterNorm === true,
    msaDoubleAddPair: dialect.msaDoubleAddPair === true,
    templateMeanOverAllSlots: dialect.templateMeanOverAllSlots === true,
    // The atom stack and the diffusion
    maskPaddedKeys: dialect.maskPaddedKeys === true,
    keysUidFromQueriesLayout: dialect.keysUidFromQueriesLayout === true,
    keyMaskedAtomAttention: dialect.keyMaskedAtomAttention === true,
    chainedAtomLayerNorm: dialect.chainedAtomLayerNorm === true,
    diffusionNoResidual: dialect.diffusionNoResidual === true,
    pairNormPerBlock: dialect.pairNormPerBlock === true,
    maskAtomActPerBlock: dialect.maskAtomActPerBlock === true,
    preTrunkQuery: dialect.preTrunkQuery === true,
    rawRefCharge: dialect.rawRefCharge === true,
    padUnknownDna: dialect.padSingleCondUnknownDna === true,
    // The confidence head
    confidenceGlobalNorm: dialect.confidenceGlobalNorm === true,
    confidenceCaDgram: dialect.confidenceCaDgram === true,
    preSymmetrisedPde: dialect.preSymmetrisedPde === true,
    perAtomHeadsByRank: dialect.perAtomHeadsByRank === true,
    reembedConfidencePair: dialect.reembedConfidencePair === true,
  };
}

/**
 * `target_feat`: the atom encoder's token aggregate, plus the per-token
 * features, on the host where the reference computes them.
 *
 * @param {Execution} execution
 * @param {{batch: Batch, weights: any, flags: any,
 *   onStage?: (name: string, tensor: Tensor) => void}} call
 * @returns {{values: Float32Array, width: number}}
 */
export function buildTargetFeat(execution, { batch, weights, flags, onStage }) {
  const { tokens, dense } = batch;
  const w = weights.targetFeat;
  const conditioning = perAtomConditioning(execution, {
    positions: weights.batchTensors.refPos,
    mask: weights.batchTensors.refMask,
    element: weights.batchTensors.refElement,
    charge: weights.batchTensors.refCharge,
    atomNameChars: weights.batchTensors.refAtomNameChars,
    weights: w.reference, tokens, dense, rawRefCharge: flags.rawRefCharge,
  });
  onStage?.("target-feat.conditioning", conditioning);
  const encoded = atomCrossAttentionEncoder(execution, {
    conditioning,
    atomMask: weights.batchTensors.refMask,
    refPos: weights.batchTensors.refPos,
    refSpaceUid: weights.batchTensors.refSpaceUid,
    gathers: weights.batchTensors.gathers,
    shape: batch.shape,
    weights: w.encoder,
    maskPaddedKeys: flags.maskPaddedKeys,
    keysUidFromQueriesLayout: flags.keysUidFromQueriesLayout,
    keyMaskedAtomAttention: flags.keyMaskedAtomAttention,
    chainedAtomLayerNorm: flags.chainedAtomLayerNorm,
    diffusionNoResidual: flags.diffusionNoResidual,
    pairNormPerBlock: flags.pairNormPerBlock,
    maskAtomActPerBlock: flags.maskAtomActPerBlock,
    preTrunkQuery: flags.preTrunkQuery,
  });
  const tokenAct = Float32Array.from(execution.read(encoded.tokenAct));
  for (const tensor of Object.values(encoded)) execution.release(tensor);
  execution.release(conditioning);
  const values = targetFeatures(/** @type {any} */ ({
    aatype: batch.aatype, profile: batch.profile, deletionMean: batch.deletionMean,
    atomFeatures: tokenAct,
    sum: w.encoder?.targetFeatSum ?? null,
    isDna: batch.isDna, isRna: batch.isRna, isLigand: batch.isLigand,
    isModified: batch.isModified, cyclicPeriod: batch.cyclicPeriod,
  }), tokens, w.dialect ?? w.encoder?.dialect);
  return { values, width: values.length / tokens };
}

/**
 * @param {Execution} execution
 * @param {{batch: Batch, weights: ReturnType<typeof uploadAf3Weights>,
 *   hostWeights: any, msa: MsaRows | ((pass: number) => MsaRows),
 *   recycles?: number, steps?: number, seed?: number, templates?: number,
 *   templateSlots?: any[], sampler?: object, readTrunk?: boolean,
 *   hoistPairConditioning?: boolean, normal?: () => number,
 *   onStep?: (frame: object) => void,
 *   onStage?: (name: string, detail: object) => void}} call
 * @returns {{positions: Float32Array, plddt: Float32Array, meanPlddt: number,
 *   pae: Float32Array, pde: Float32Array, contactProbs: Float32Array,
 *   trunkSingle?: Float32Array, trunkPair?: Float32Array,
 *   ptm: number, iptm: number, trunkPasses: number, steps: number}}
 */
export function foldAf3(execution, call) {
  return drain(foldAf3Steps(execution, call));
}

/**
 * `foldAf3` without holding the thread: the same fold, awaiting the GPU (a
 * fence, not a read) after every trunk block and before every denoiser call, and
 * stopping at the next of those when `signal` aborts. What a page calls; the
 * tools and gates keep `foldAf3`, which runs the same steps straight through.
 *
 * @param {Execution} execution
 * @param {Parameters<typeof foldAf3>[1] & {signal?: AbortSignal}} call
 */
export async function foldAf3Async(execution, call) {
  const groups = await prepareFoldPrograms(execution, call);
  // The rest compile while the fold waits on the GPU, one a task, in the order it first needs
  // them; and before it goes on, the whole group its next slice first draws with (the earliest
  // group not yet compiled), so that slice does not link a dozen programs inside itself - which
  // the GPU's wait alone did not always cover at the trunk's start (417 ms on Safari 16.3).
  // ...each slice group's parts as one program (Execution.combine), the first's done by prepare.
  for (const group of groups) execution.combine(group);
  const later = groups.flat();
  const groupEnd = [];
  for (const group of groups) groupEnd.push((groupEnd.at(-1) ?? 0) + group.length);
  let next = 0;
  const compiled = (source) => execution.programs.has(execution.packaged(source));
  const compileNext = () => {
    while (next < later.length && compiled(later[next])) next += 1;
    if (next === later.length) return false;
    execution.program(execution.packaged(later[next]));
    next += 1;
    return true;
  };
  // ...one slice behind the GPU rather than at its drained end (`laggedPause`); `noLaggedPause` is the old arm
  const behind = execution.noLaggedPause ? null : laggedPause(execution.gl, compileNext);
  const pause = async () => {
    while (next < later.length && compiled(later[next])) next += 1;
    const needed = groupEnd.find((end) => end > next) ?? 0;
    if (behind === null) await gpuIdle(execution.gl, compileNext);
    else await behind();
    while (next < needed && compileNext()) {
      await new Promise((resolve) => { setTimeout(resolve, 0); });
    }
  };
  try {
    return await drainAsync(foldAf3Steps(execution, call), pause, call.signal);
  } finally {
    behind?.finish();
  }
}

/**
 * The programs this fold will draw with, found by a dry run of the same batch with no draws
 * (`Execution.collect`, one block of each stack and one denoiser call, which reach every kernel):
 * those its first slice draws with compiled now (`Execution.prepare`), the rest returned grouped
 * by the slice that first draws with them, for `foldAf3Async` to compile while it waits on the GPU. A
 * first fold links ~100 programs, ~20 ms each on Safari 16.3 on the page's thread; the kernels
 * take their lengths as uniforms (`sized`), so a later fold at another length finds them compiled.
 */
export async function prepareFoldPrograms(execution, call) {
  // ...on the first fold with these weights only: the programs serve every length, so a later
  // fold finds them compiled and the dry run (0.2-0.35 s) would find nothing; the odd kernel a
  // size threshold brings in is compiled at its draw.
  let prepared = preparedWeights.get(execution);
  if (prepared === undefined) preparedWeights.set(execution, prepared = new WeakSet());
  if (prepared.has(call.weights)) return [];
  const { groups } = await foldPrograms(execution, call);
  await execution.prepare(groups[0]);
  prepared.add(call.weights);
  return groups.slice(1);
}

/** Per execution, the uploaded weights a fold has already prepared programs for. */
const preparedWeights = new WeakMap();

/**
 * The programs a fold of this batch draws with, as source -> kernel label in the order the fold
 * first draws with them, and grouped by the slice (between two yields) that first draws with them:
 * a dry run with no draws (`Execution.collect`), one block of each stack and one denoiser call.
 *
 * @returns {Promise<{programs: Map<string, string>, groups: string[][]}>}
 */
export async function foldPrograms(execution, call) {
  const { weights } = call;
  const trimmed = {
    ...weights,
    msaBlocks: weights.msaBlocks.slice(0, 1),
    pairformerBlocks: weights.pairformerBlocks.slice(0, 1),
    ...(weights.confidence === undefined ? {}
      : { confidence: { ...weights.confidence, blocks: weights.confidence.blocks.slice(0, 1) } }),
  };
  execution.collect = new Map();
  execution.storage.virtual = true;
  const ends = [];
  // ...and what each slice of it cost the host, by the stage that ended it (probe-webgl-compile.js)
  const sliceMs = [];
  let sliceFrom = performance.now();
  let taskFrom = sliceFrom;
  try {
    await drainAsync(foldAf3Steps(execution, {
      ...call, weights: trimmed, recycles: 0, steps: 1, readTrunk: false,
      onStage: undefined, onStep: undefined,
    }), (yielded) => {
      const now = performance.now();
      sliceMs.push([yielded?.stage, Math.round((now - sliceFrom) * 10) / 10]);
      sliceFrom = now;
      if (execution.collect.size > (ends.at(-1) ?? 0)) ends.push(execution.collect.size);
      // 🔴 THE PAGE A TASK EVERY ~16 ms OF IT, NOT EVERY SLICE: its slices are a few ms of host
      // work each and a nested setTimeout(0) waits >= 4 ms, so yielding at all ~20 of them was
      // most of the dry run (107 ms in Chrome, of which ~10 work).
      if (now - taskFrom < 16) return undefined;
      return new Promise((resolve) => { setTimeout(resolve, 0); }).then(() => {
        taskFrom = performance.now();
        sliceFrom = taskFrom;
      });
    }, call.signal);
    const programs = execution.collect;
    const ordered = [...programs.keys()];
    if (ordered.length > (ends.at(-1) ?? 0)) ends.push(ordered.length);
    const groups = ends.map((end, index) => ordered.slice(ends[index - 1] ?? 0, end));
    return { programs, groups, sliceMs };
  } finally {
    execution.collect = null;
    execution.storage.virtual = false;
  }
}

/** `foldAf3` as a generator (see `foldAf3Async`): yields at each trunk block and denoiser call. */
export function* foldAf3Steps(execution, call) {
  const { batch, weights, hostWeights, msa } = call;
  const recycles = call.recycles ?? 0;
  const steps = call.steps ?? 200;
  const seed = call.seed ?? 42;
  const templates = call.templates ?? 4;
  const flags = dialectFlags(weights.dialect);
  const { tokens, dense } = batch;
  const atoms = tokens * dense;
  const pairChannels = hostWeights.trunk.embedder.pairChannels;
  const held = [];
  const upload = (values, label) => {
    const tensor = execution.upload(
      values instanceof Float32Array ? values : Float32Array.from(values), label);
    held.push(tensor);
    return tensor;
  };
  const uploadGather = (gather, name) => ({
    indices: upload(Float32Array.from(gather.indices), `${name}.indices`),
    mask: upload(gather.mask, `${name}.mask`),
  });

  // Everything the batch carries that a kernel reads, uploaded once: none of it
  // changes between passes or between steps.
  const seqMask = batch.seqMask;
  const pairMask = pairMaskOf(seqMask, tokens);
  const batchTensors = {
    refPos: upload(batch.refPos, "af3.batch.ref-pos"),
    refMask: upload(batch.refMask, "af3.batch.ref-mask"),
    refElement: upload(batch.refElement, "af3.batch.ref-element"),
    refCharge: upload(batch.refCharge, "af3.batch.ref-charge"),
    refAtomNameChars: upload(batch.refAtomNameChars, "af3.batch.ref-names"),
    refSpaceUid: upload(batch.refSpaceUid, "af3.batch.ref-uid"),
    seqMask: upload(seqMask, "af3.batch.seq-mask"),
    pairMask: upload(pairMask, "af3.batch.pair-mask"),
    asymId: upload(Float32Array.from(batch.asymId), "af3.batch.asym"),
    identity: upload(tokenIdentity(batch.features, tokens), "af3.batch.identity"),
    // 🔴 PER PAIR, NOT PER TOKEN. `af3ContactClasses` gives one CLASS a token -
    // what kind of thing it is - and the kernel reads "how many distogram bins
    // count as a contact for THIS PAIR", which is `af3ContactBins` over those
    // classes and the head's own edges. Handing over the classes compiles, runs,
    // and counts two or three bins of sixty-four: the whole contact map came out
    // at 0.0007 on the ADJACENT pairs, which are 3.8 A apart in every protein.
    contactBins: upload(
      Float32Array.from(af3ContactBins(
        af3ContactClasses(batch, tokens), tokens, binEdges())),
      "af3.batch.contact-bins"),
    gathers: {
      tokenAtomsToQueries: uploadGather(batch.tokenAtomsToQueries, "af3.batch.ta2q"),
      queriesToKeys: uploadGather(batch.queriesToKeys, "af3.batch.q2k"),
      queriesToTokenAtoms: uploadGather(batch.queriesToTokenAtoms, "af3.batch.q2ta"),
      tokensToQueries: uploadGather(batch.tokensToQueries, "af3.batch.t2q"),
      tokensToKeys: uploadGather(batch.tokensToKeys, "af3.batch.t2k"),
    },
    pseudoBetaGather: uploadGather(batch.tokenAtomsToPseudoBeta, "af3.batch.t2pb"),
  };

  const targetFeat = buildTargetFeat(execution, {
    batch, weights: { ...weights, batchTensors }, flags, onStage: call.onStage,
  });
  const targetFeatTensor = upload(targetFeat.values, "af3.target-feat");
  call.onStage?.("target-feat", { width: targetFeat.width });
  yield { stage: "target-feat" };

  // `msa` is one alignment for every pass, or - a deep one, which every pass draws its own rows from as
  // webgpu/af3/fold.js does (shared/af3/msa-draw.js) - a function of the pass. A pass's draw is uploaded
  // at its start and released at its end, so only one pass's rows are on the GPU at a time.
  const msaOfPass = typeof msa === "function" ? msa : null;
  let msaCodes, deletions, msaMask;
  let sequences = msaOfPass === null ? msa.sequences : 0;
  if (msaOfPass === null) {
    msaCodes = upload(msa.codes, "af3.msa.codes");
    deletions = upload(msa.deletions, "af3.msa.deletions");
    msaMask = upload(msa.mask, "af3.msa.mask");
  }

  // The template slots, built once: they do not depend on the pass.
  const slotInput = (slot) => ({
    template: call.templateSlots?.[slot] ?? null,
    tokens, slot, firstEmpty: call.templateSlots?.length ?? 0,
    asymId: batch.asymId,
    emptyTemplateAatype: weights.dialect?.emptyTemplateAatype ?? null,
    templateGapWhereNoAtoms: weights.dialect?.templateGapWhereNoAtoms === true,
  });
  const hostCodes = Array.from({ length: templates }, (_, slot) => slotCodes(slotInput(slot)));
  const codes = hostCodes.map((values, slot) => upload(values, `af3.template.codes.${slot}`));
  const geometry = Array.from({ length: templates }, (_, slot) => {
    const packed = slotGeometry(slotInput(slot));
    return packed === undefined
      ? undefined : upload(packed, `af3.template.geometry.${slot}`);
  });
  // 🔴 EMPTY SLOTS WITH THE SAME RESTYPE CODES ARE THE SAME COMPUTATION - AF3 pads to four and a template-free fold
  // ran the template stack four times over one answer. Each such slot is run once and weighted by how many it stands for
  // (`templateEmbeddingSteps`'s `repeats`). `--no-template-dedupe` is the arm.
  const repeatOf = hostCodes.map((values, slot) => {
    if (execution.noTemplateDedupe || geometry[slot] !== undefined) return -1;
    for (let earlier = 0; earlier < slot; earlier += 1) {
      if (geometry[earlier] === undefined && repeatOfSame(hostCodes[earlier], values)) return earlier;
    }
    return -1;
  });
  const repeats = repeatOf.map((of, slot) => (of >= 0 ? 0 : 1 + repeatOf.filter((other) => other === slot).length));

  // 🔴 THE RECYCLED STATE IS ZEROS ON PASS ONE AND IT STILL CONTRIBUTES, so it
  // is a real tensor rather than an absent one: the embedder LayerNorms it and
  // the norm's OFFSET is non-zero, so a pass-one fold that skipped the term is
  // not the same fold.
  //
  // 🔴 AND THIS COMMENT USED TO END "`Storage.allocate` zero-initialises every
  // texture, so this costs one texImage2D each and no kernel", WHICH WAS NOT
  // TRUE. `texImage2D` with a null pointer leaves the contents UNDEFINED;
  // Chrome on Metal happens to hand back zeros, so the belief held here and
  // would not hold on a driver that recycles. `Storage.zeros` writes them, and
  // `--nan-watch` is what found this - it named `layer-norm:3481x128:fast`
  // inside the embedder. Three of the four fold drivers had the same fault.
  const singleChannels = hostWeights.trunk.embedder.singleChannels;
  let previousPair = execution.zeros(
    tokens * tokens * pairChannels, "af3.recycle.pair");
  let previousSingle = execution.zeros(
    tokens * singleChannels, "af3.recycle.single");
  let trunk;
  for (let pass = 0; pass <= recycles; pass += 1) {
    call.onStage?.("recycle", { pass, passes: recycles + 1 });
    let passMsa = [];
    if (msaOfPass !== null) {
      const drawn = msaOfPass(pass);
      sequences = drawn.sequences;
      passMsa = [
        execution.upload(Float32Array.from(drawn.codes), "af3.msa.codes"),
        execution.upload(Float32Array.from(drawn.deletions), "af3.msa.deletions"),
        execution.upload(Float32Array.from(drawn.mask), "af3.msa.mask"),
      ];
      [msaCodes, deletions, msaMask] = passMsa;
    }
    const embedded = embed(execution, {
      targetFeat: targetFeatTensor,
      identity: batchTensors.identity,
      msaCodes, deletions,
      previousPair, previousSingle,
      weights: weights.embedder,
      tokens, sequences,
      pairInitFromSingle: flags.pairInitFromSingle,
      msaPairedQueryRow: flags.msaPairedQueryRow,
    });
    call.onStage?.("embed", { pass });
    yield { stage: "embed", pass };
    // 🔴 THE TEMPLATE STACK RUNS WHETHER OR NOT THERE IS A TEMPLATE, because
    // "an empty slot is not a no-op": the query term and the bias are outside
    // the slot mask, so a template-free fold still gets a pair update.
    let templateUpdate;
    if (weights.template !== undefined) {
      templateUpdate = yield* templateEmbeddingSteps(execution, {
        pair: embedded.pair, pairMask: batchTensors.pairMask,
        tokens, templates, weights: weights.template, codes, geometry, repeats, repeatOf,
        swapTransposedBias: flags.swapTransposedBias,
        divideByLength: flags.divideByLength,
      });
      call.onStage?.("template", { pass, templates });
      yield { stage: "template", pass };
    }
    const next = yield* runTrunkSteps(execution, {
      embedded, templateUpdate,
      msaMask, pairMask: batchTensors.pairMask, seqMask: batchTensors.seqMask,
      contactBins: batchTensors.contactBins,
      halfLogits: weights.distogram.halfLogits,
      msaBlocks: weights.msaBlocks,
      pairformerBlocks: weights.pairformerBlocks,
      tokens, sequences, pairChannels,
      msaUpdateBeforeOuterProduct: flags.msaUpdateBeforeOuterProduct,
      msaDoubleAddPair: flags.msaDoubleAddPair,
      swapTransposedBias: flags.swapTransposedBias,
      divideByLength: flags.divideByLength,
      opmBiasAfterNorm: flags.opmBiasAfterNorm,
    });
    if (templateUpdate !== undefined) execution.release(templateUpdate);
    for (const tensor of passMsa) execution.release(tensor);
    // ...the pass's pair and single ARE the next pass's recycled state.
    if (previousPair !== undefined) execution.release(previousPair);
    if (previousSingle !== undefined) execution.release(previousSingle);
    if (trunk !== undefined) {
      execution.release(trunk.logits);
      execution.release(trunk.contactProbs);
    }
    previousPair = next.pair;
    previousSingle = next.single;
    trunk = next;
    call.onStage?.("trunk", { pass });
  }
  yield { stage: "contacts" };
  const contactProbs = Float32Array.from(execution.read(trunk.contactProbs));
  execution.release(trunk.logits);
  execution.release(trunk.contactProbs);

  // The diffusion: one head call a step, the walk on the host.
  const headWeights = weights.diffusion;
  const headInput = {
    conditioning: undefined, // filled below, because the head needs its own copy
    atomMask: batchTensors.refMask,
    seqMask: batchTensors.seqMask,
    refPos: batchTensors.refPos,
    refSpaceUid: batchTensors.refSpaceUid,
    trunkSingle: trunk.single,
    trunkPair: trunk.pair,
    targetFeat: targetFeatTensor,
    identity: batchTensors.identity,
    gathers: batchTensors.gathers,
    shape: batch.shape,
    weights: headWeights,
    relativeWidth: RELATIVE_WIDTH,
    fourierWeight: weights.hostFourier.weight,
    fourierBias: weights.hostFourier.bias,
    padUnknownDna: flags.padUnknownDna,
    maskPaddedKeys: flags.maskPaddedKeys,
    keysUidFromQueriesLayout: flags.keysUidFromQueriesLayout,
    keyMaskedAtomAttention: flags.keyMaskedAtomAttention,
    chainedAtomLayerNorm: flags.chainedAtomLayerNorm,
    diffusionNoResidual: flags.diffusionNoResidual,
    pairNormPerBlock: flags.pairNormPerBlock,
    maskAtomActPerBlock: flags.maskAtomActPerBlock,
    preTrunkQuery: flags.preTrunkQuery,
    ...(call.hoistPairConditioning === false
      ? { hoistPairConditioning: false } : {}),
  };
  // 🔴 THE HEAD'S PER-ATOM CONDITIONING IS THE SAME TENSOR target_feat BUILT,
  // and it is rebuilt rather than kept because `buildTargetFeat` released it:
  // the encoder consumed the only copy and keeping it live across the whole
  // trunk would hold `atoms x channels` floats through the pass that peaks.
  const conditioning = perAtomConditioning(execution, {
    positions: batchTensors.refPos, mask: batchTensors.refMask,
    element: batchTensors.refElement, charge: batchTensors.refCharge,
    atomNameChars: batchTensors.refAtomNameChars,
    weights: weights.atomReference, tokens, dense,
    rawRefCharge: flags.rawRefCharge,
  });
  headInput.conditioning = conditioning;

  const positions = yield* sampleOnWebGl2Steps(execution, headInput, {
    steps, mask: batch.refMask, normal: call.normal ?? normalFrom(seed),
    ...SAMPLER_DEFAULTS,
    ...(call.sampler ?? {}),
    ...(weights.dialect?.sampler ?? {}),
    realign: weights.dialect?.realignSampler === true,
    onStep: call.onStep,
  });
  execution.release(conditioning);
  call.onStage?.("sample", { steps });

  // The confidence head, on the sampled coordinates through the batch's own
  // pseudo-beta gather.
  let plddt = new Float32Array(atoms);
  let pae = new Float32Array(tokens * tokens);
  // ...and the PAE's logits, which a caller needs for any per-chain or per-interface score
  let paeLogits;
  let pde = new Float32Array(tokens * tokens);
  let scores = { ptm: NaN, iptm: NaN };
  if (weights.confidence !== undefined) {
    const sampled = upload(positions, "af3.conf.positions");
    const pseudoBeta = execution.allocate(tokens * 3, "af3.conf.pseudo-beta");
    execution.dispatch({
      source: pseudoBetaSource(tokens),
      output: pseudoBeta,
      inputs: {
        positionsTex: sampled,
        indicesTex: batchTensors.pseudoBetaGather.indices,
        maskTex: batchTensors.pseudoBetaGather.mask,
      },
      label: `af3.conf.pseudo-beta:${tokens}`,
    });
    const out = yield* confidenceHeadSteps(execution, {
      pair: trunk.pair, single: trunk.single, targetFeat: targetFeatTensor,
      pseudoBeta, seqMask: batchTensors.seqMask,
      asymId: batchTensors.asymId,
      atomNameChars: batchTensors.refAtomNameChars,
      tokens, dense,
      weights: weights.confidence.head, blocks: weights.confidence.blocks,
      confidenceGlobalNorm: flags.confidenceGlobalNorm,
      confidenceCaDgram: flags.confidenceCaDgram,
      preSymmetrisedPde: flags.preSymmetrisedPde,
      perAtomHeadsByRank: flags.perAtomHeadsByRank,
      reembedConfidencePair: flags.reembedConfidencePair,
      swapTransposedBias: flags.swapTransposedBias,
      divideByLength: flags.divideByLength,
    });
    yield { stage: "confidence" };
    plddt = Float32Array.from(execution.read(out.plddt));
    pae = Float32Array.from(execution.read(out.pae));
    pde = Float32Array.from(execution.read(out.pde));
    paeLogits = Float32Array.from(execution.read(out.paeLogits));
    for (const tensor of Object.values(out)) execution.release(tensor);
    execution.release(pseudoBeta);
    const centres = errorBinCentres(NUM_BINS, MAX_ERROR_BIN);
    // The global term for pTM and ipTM - one d0 over every token, as AF3's
    // `_compute_ptm` reads for both - and the interface term for the chain
    // scores, d0 from the pair's own chains.
    const tmAdjusted = tmTermFromLogits(paeLogits, tokens,
      tmPerBinFor(centres, tmScoreD0(tokens)));
    const tmInterface = interfaceTmTerm(paeLogits, tokens,
      chainTmInfo(batch.asymId, seqMask, centres));
    // 🔴 AND THE ANCHORS ARE THE TOKENS WITH FRAMES, AF3's pae_single_mask: an
    // ion, a two-atom ligand and a colinear three-atom one align nothing. It
    // moves pTM and ipTM by almost nothing - a framed token wins the maximum
    // anyway - and moves a ligand chain's own scores a long way, which is why a
    // path missing it looks right on the two headline numbers.
    const frame = tokenFrameMask({ tokens, dense, isLigand: batch.isLigand,
      asymId: batch.asymId, residueIndex: batch.residueIndex,
      refPos: batch.refPos, refMask: batch.refMask });
    scores = { ...af3TmScores({ tmAdjusted, tmInterface, tokens,
      asymId: batch.asymId, seqMask, frame }), tmAdjusted, tmInterface, frame };
    call.onStage?.("confidence", {});
  }

  // Read back for a cross-backend comparison before they go: the two tracks
  // fail separately and a fold that compares only coordinates cannot say which.
  const trunkSingle = call.readTrunk === true
    ? Float32Array.from(execution.read(trunk.single)) : undefined;
  const trunkPair = call.readTrunk === true
    ? Float32Array.from(execution.read(trunk.pair)) : undefined;
  execution.release(trunk.pair);
  execution.release(trunk.single);
  for (const tensor of held) execution.release(tensor);

  let total = 0;
  let live = 0;
  for (let atom = 0; atom < atoms; atom += 1) {
    if (!(batch.refMask[atom] > 0)) continue;
    total += plddt[atom];
    live += 1;
  }
  return {
    positions, plddt, pae, pde, paeLogits, contactProbs, trunkSingle, trunkPair,
    meanPlddt: live === 0 ? NaN : total / live,
    ptm: scores.ptm, iptm: scores.iptm,
    // The chain-level scores beside them, which this path reported as nothing at
    // all: `chainPtm`/`chainIptm` keyed by asym id, `chainPairIptm` by "a|b".
    chainPtm: scores.chainPtm, chainIptm: scores.chainIptm,
    chainPairIptm: scores.chainPairIptm,
    // ...and the two terms they were reduced from, with the anchor mask, for a
    // caller that wants to reduce them another way (what the frame mask is
    // worth, an interface of its own) or to count the frameless tokens.
    tmAdjusted: scores.tmAdjusted, tmInterface: scores.tmInterface,
    frame: scores.frame,
    trunkPasses: recycles + 1, steps,
  };
}

// Every builder above, memoised on its arguments (`memoSource` in webgl/runtime/execution.js).
export const pseudoBetaSource = memoSource(pseudoBetaSourceUncached);

/** Whether two host arrays hold the same values. */
function repeatOfSame(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) if (a[i] !== b[i]) return false;
  return true;
}
