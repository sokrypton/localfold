/**
 * AlphaFold 2 end to end: the driver that makes the stages a fold, on either
 * graph.
 *
 *     once:        templateUpdate = templateEmbedding(geometry features)
 *     every pass:  msa, extra, pair = inputEmbedder(features, recycled state)
 *                  pair += templateUpdate
 *                  extra stack   (global column attention, 4 blocks)
 *                  main stack    (48 blocks)
 *                  structure module, confidence heads, contact map
 *                  carry the MSA's first row, the pair and the atom37 forward
 *
 * `webgpu/af2/model/monomer.js` is the order and the conventions and
 * `webgpu/af2/multimer/model.js` is the unified driver this follows - "AF2
 * monomer and multimer through one path... it differs in three imports and two
 * options". There is no command-buffer windowing here, because a draw is
 * submitted as it is issued.
 *
 * 🔴 AND THE TWO GRAPHS DIFFER IN SIX PLACES, EVERY ONE OF WHICH IS THE
 * CHECKPOINT'S AND NOT A TUNING CHOICE. `graph: "multimer"` sets all six:
 *
 *   - the outer product mean runs FIRST in both block stacks;
 *   - the relative encoding reads a `[4, length]` chain identity over 73 rows
 *     rather than the residue index over 65, so the monomer table is WIDENED at
 *     upload (the shipped bytes stay the bytes on disk);
 *   - `position_scale` is 20 rather than 10;
 *   - the transitions keep their hidden row in f32 where the monomer's round it
 *     to f16, and the attention targets are all f32 where the monomer's are f16;
 *   - the template embedder is the NINE-PROJECTION one, and it runs EVERY PASS
 *     rather than once, because it reads the pair through a LayerNorm and so is
 *     a function of the pair rather than a constant;
 *   - and its template needs the chain ids, which it refuses to guess.
 *
 * Getting any of them wrong produces a fold rather than an error. The reference
 * says what dropping the regime once cost: it "ran multimer WEIGHTS on the
 * monomer graph", and that reached a visitor.
 *
 * 🔴 THE RECYCLED STATE NEVER LEAVES THE DEVICE, WHICH THE WebGPU DRIVER HAD TO
 * BE TAUGHT. Its own comment records the cost of the other way: the pair track
 * is `L^2 * 128` floats, "348 MB at 825 residues, read back to a JavaScript
 * array and uploaded again". Here the three carried tensors are simply tensors,
 * so a pass hands the next one its state with no crossing at all, and the only
 * readbacks are the ones a caller asked for.
 *
 * 🔴 AND THE FIRST PASS'S STATE IS AN ALLOCATION, NOT AN UPLOAD OF ZEROS.
 * `Storage.allocate` creates the texture with `texImage2D(..., null)`, which
 * WebGL zero-initialises, so three zeroed tensors cost three `texImage2D` calls
 * and no bus traffic. The WebGPU driver makes the same point about its own
 * buffers - "at 825 residues the pair state alone is 348 MB of zeros".
 *
 * 🔴 A CHECKPOINT WITH NO TEMPLATE EMBEDDER IS SKIPPED, NOT HANDED A MASKED
 * TEMPLATE. model_3, model_4 and model_5 are AF2's template-free models and the
 * 67 tensors are simply absent, so `weights.template === null` means the term
 * does not exist. A masked template is NOT the same thing: it leaves the
 * embedder's own bias and layer norms in the pair, which is a term this graph
 * must not have when the reference never built the module.
 *
 * 🔴 AND THE STRUCTURE MODULE READS THE TRUNK MSA'S FIRST ROW, WHICH IS A COPY
 * AND NOT A VIEW. The WebGPU driver does `copyBufferToBuffer` of the leading
 * `length * 256` floats; a texture cannot be aliased with a smaller element
 * count, so `prefix` is a draw. Handing the structure module the whole MSA
 * would be a shape error, and handing it row zero of the EMBEDDER's MSA rather
 * than the trunk's would be 48 blocks of work thrown away - which is a fold
 * that looks plausible and is not the model.
 */
import { confidenceHeads, releaseConfidenceWeights, uploadConfidenceWeights } from "../../heads/confidence.js";
import { distogramContacts } from "../../heads/distogram.js";
import { evoformerStack, extraMsaStack } from "../evoformer/stack.js";
import {
  inputEmbedder, releaseInputEmbedderWeights, residueIdentity, uploadInputEmbedderWeights,
  widenRelativePositionWeight,
} from "../evoformer/input-embedder.js";
import { prefix } from "../../runtime/elementwise.js";
import { add } from "../../runtime/elementwise.js";
import {
  releaseGeometryTables, releaseStructureModuleWeights, structureModule,
  uploadGeometryTables, uploadStructureModuleWeights,
} from "../structure/module.js";
import {
  releaseTemplateWeights, templateEmbedding, templateFeatures, uploadTemplateWeights,
} from "../evoformer/template.js";
import {
  multimerTemplateEmbedding, multimerTemplateFeatures, releaseMultimerTemplateWeights,
  uploadMultimerTemplateWeights,
} from "../multimer/template.js";
import {
  releaseEvoformerBlockWeights, uploadEvoformerBlockWeights, uploadTemplatePairBlockWeights,
} from "../evoformer/block.js";
import {
  recycleConvergenceDistance, shouldStopAfterRecycle, validatedRecycleTolerance,
} from "../../../shared/af2/model/recycle-convergence.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */
/**
 * One pass's features, exactly as `shared/input/a3m-features.js` builds them -
 * `makeA3mFeatures(a3m, tables, options)[recycle]`. Typed here rather than taken
 * as `object` because every one of these names is a seam: a feature this driver
 * reads under a name the featuriser does not use is a tensor of `undefined`,
 * which fails at upload rather than folding something wrong, but only once.
 *
 * @typedef {{aatype: ArrayLike<number>, residueIndex: ArrayLike<number>,
 *   seqMask: ArrayLike<number>, msaMask: ArrayLike<number>,
 *   extraMsaMask: ArrayLike<number>, targetFeatures: ArrayLike<number>,
 *   msaFeatures: ArrayLike<number>, extraMsa: ArrayLike<number>,
 *   extraHasDeletion: ArrayLike<number>, extraDeletionValue: ArrayLike<number>,
 *   atom37ToAtom14: ArrayLike<number>, atom37Mask: ArrayLike<number>,
 *   msaSequences: number, extraSequences: number,
 *   targetChannels: number, msaFeatureChannels: number,
 *   asymId?: ArrayLike<number>, entityId?: ArrayLike<number>,
 *   symId?: ArrayLike<number>}} RecycleFeatures
 */

/**
 * AF2 monomer's widths, which are the checkpoint's and not ours to choose.
 * `webgpu/af2/model/monomer.js` passes the same numbers as literals.
 */
export const AF2_SHAPE = {
  msaChannels: 256,
  extraMsaChannels: 64,
  pairChannels: 128,
  templateChannels: 64,
  structureChannels: 384,
  distogramBins: 64,
  // Ten for every monomer checkpoint; the multimer graph trains against twenty.
  positionScale: 10,
};

/** ...and what `graph: "multimer"` changes about it. */
export const MULTIMER_SHAPE = { ...AF2_SHAPE, positionScale: 20 };

/**
 * Every tensor a monomer fold reads, uploaded once.
 *
 * The argument is `shared/bundles/alphafold-fixture.js`'s own weight set,
 * unchanged - `embedding`, `template`, `extraStack`, `mainStack`, `structure`,
 * `lddt`, `pae`, `distogramHead` and `geometry` - because that is what the
 * WebGPU driver takes and a second spelling would be a seam that goes stale.
 *
 * 🔴 TWO SECTIONS ARE LEGITIMATELY ABSENT AND A THIRD IS NOT. `template` is
 * null for model_3, model_4 and model_5, AF2's template-free checkpoints, and
 * `distogramHead` is absent from any bundle exported before the head was
 * appended (tools/add_distogram_head.py) - which costs the contact map and
 * nothing else. Both are passed through as null rather than refused. Everything
 * else missing is a fold that cannot run, and the per-section upload throws
 * naming the tensor.
 */
export function uploadAf2Weights(execution, weights, graph = "monomer") {
  const multimer = graph === "multimer";
  const hidden = (group) => ({
    triangleHidden: group.triangleMultiplicationOutgoing.linearAPBias.length,
    pairHidden: group.pairTransition.firstBias.length,
  });
  // 🔴 THE RELATIVE TABLE IS WIDENED HERE RATHER THAN AT EXPORT, so the shipped
  // monomer weights stay the bytes on disk: no re-quantisation, no new manifest,
  // no re-download. A monomer's eight trailing rows are zero and the chains
  // kernel then computes exactly what the 65-row lookup does - which
  // `check-webgl-multimer-embedder.js` holds bit-identical rather than assuming.
  const embedding = multimer
    ? {
      ...weights.embedding,
      relativePositionWeight: widenRelativePositionWeight(
        weights.embedding.relativePositionWeight, AF2_SHAPE.pairChannels),
    }
    : weights.embedding;
  return {
    embedding: uploadInputEmbedderWeights(execution, embedding),
    // The multimer's nine-projection embedder, which runs every pass.
    templateEmbedding: weights.templateEmbedding === undefined
      ? undefined
      : {
        stage: uploadMultimerTemplateWeights(execution, weights.templateEmbedding),
        blocks: weights.templateEmbedding.stack.map(
          (block) => uploadTemplatePairBlockWeights(execution, block)),
        triangleHidden:
          weights.templateEmbedding.stack[0].triangleMultiplicationOutgoing.linearAPBias.length,
        pairHidden: weights.templateEmbedding.stack[0].pairTransition.firstBias.length,
      },
    template: weights.template === null || weights.template === undefined ? null : {
      stage: uploadTemplateWeights(execution, weights.template),
      blocks: weights.template.blockWeights.map(
        (block) => uploadTemplatePairBlockWeights(execution, block)),
      ...hidden(weights.template.blockWeights[0]),
    },
    extraStack: weights.extraStack.map(
      (block) => uploadEvoformerBlockWeights(execution, block)),
    mainStack: weights.mainStack.map(
      (block) => uploadEvoformerBlockWeights(execution, block)),
    structure: uploadStructureModuleWeights(execution, weights.structure),
    tables: uploadGeometryTables(execution, weights.geometry),
    confidence: uploadConfidenceWeights(execution, weights.lddt, weights.pae),
    // 🔴 THE DISTOGRAM HEAD'S WEIGHT IS A **HALF** LOGITS MATRIX, AND ITS BINS
    // AND BREAKS COME WITH IT. `distogramHeadWeights` returns
    // `halfLogitsWeights`/`halfLogitsBias` because the head symmetrises the pair
    // and adds the bias once in each half, which `webgl/heads/distogram.js`
    // reproduces; and the bin count and the two edges are the SECTION's, not
    // this port's, since a bundle may declare its own.
    distogramHead: weights.distogramHead === undefined || weights.distogramHead === null
      ? null
      : {
        logitsWeight: execution.upload(
          weights.distogramHead.halfLogitsWeights, "w.distogram.halfLogitsWeights"),
        logitsBias: execution.upload(
          weights.distogramHead.halfLogitsBias, "w.distogram.halfLogitsBias"),
        bins: weights.distogramHead.bins,
        firstBreak: weights.distogramHead.firstBreak,
        lastBreak: weights.distogramHead.lastBreak,
      },
  };
}

export function releaseAf2Weights(execution, weights) {
  releaseInputEmbedderWeights(execution, weights.embedding);
  if (weights.templateEmbedding !== undefined) {
    releaseMultimerTemplateWeights(execution, weights.templateEmbedding.stage);
    for (const block of weights.templateEmbedding.blocks) {
      releaseEvoformerBlockWeights(execution, block);
    }
  }
  if (weights.template !== null) {
    releaseTemplateWeights(execution, weights.template.stage);
    for (const block of weights.template.blocks) {
      releaseEvoformerBlockWeights(execution, block);
    }
  }
  for (const block of [...weights.extraStack, ...weights.mainStack]) {
    releaseEvoformerBlockWeights(execution, block);
  }
  releaseStructureModuleWeights(execution, weights.structure);
  releaseGeometryTables(execution, weights.tables);
  releaseConfidenceWeights(execution, weights.confidence);
  if (weights.distogramHead !== null) {
    execution.release(weights.distogramHead.logitsWeight);
    execution.release(weights.distogramHead.logitsBias);
  }
}

/**
 * @param {{featuresByRecycle: readonly RecycleFeatures[], graph?: "monomer" | "multimer",
 *   weights: ReturnType<typeof uploadAf2Weights>,
 *   paeBreaks: ArrayLike<number>,
 *   template?: Parameters<typeof templateFeatures>[0]["template"],
 *   useTemplateUnitVector?: boolean, templates?: number,
 *   tolerance?: number, pairHost?: boolean, contactMap?: boolean,
 *   shape?: Partial<typeof AF2_SHAPE>,
 *   onRecycle?: (result: object, recycle: number) => void,
 *   onBlock?: (stack: "extra" | "main", block: number, blocks: number) => void,
 *   yieldTo?: () => Promise<void>}} call
 * @param {import("../../runtime/execution.js").Execution} execution
 * @returns {Promise<{recycles: object[], final: object}>}
 */
export async function foldAf2(execution, call) {
  const multimer = (call.graph ?? "monomer") === "multimer";
  if (multimer && call.weights.templateEmbedding === undefined) {
    throw new Error(
      "foldAf2: the multimer graph needs `templateEmbedding` weights - its embedder runs"
      + " whether or not there is a template, and skipping it leaves the pair track"
      + " about 30% out from the first Evoformer block");
  }
  const {
    featuresByRecycle, weights: w, paeBreaks,
    pairHost = false, contactMap = true, onRecycle, yieldTo,
  } = call;
  // 🔴 `yieldTo` IS THE ONLY REASON THIS FUNCTION IS ASYNC. Every draw here is
  // issued synchronously, so a 48-block fold would hold the main thread from
  // the first dispatch to the last readback; a caller on a page passes
  // `() => new Promise(requestAnimationFrame)` and gets its frames back. A
  // caller in a tool passes nothing and the awaits cost one microtask each.
  if (featuresByRecycle.length === 0) {
    throw new RangeError("foldAf2: at least one feature set is required");
  }
  if (w.extraStack.length === 0 || w.mainStack.length === 0) {
    throw new RangeError("foldAf2: the extra and main Evoformer stacks cannot be empty");
  }
  const {
    msaChannels, extraMsaChannels, pairChannels, templateChannels,
    structureChannels, distogramBins, positionScale,
  } = { ...(multimer ? MULTIMER_SHAPE : AF2_SHAPE), ...(call.shape ?? {}) };
  const length = featuresByRecycle[0].aatype.length;
  const tolerance = validatedRecycleTolerance(call.tolerance);
  const pairs = length * length;

  // The pair mask is the sequence mask outer-multiplied with itself, and it is
  // the same for every pass.
  const pairMaskValues = new Float32Array(pairs);
  const seqMask = featuresByRecycle[0].seqMask;
  for (let i = 0; i < length; i += 1) {
    for (let j = 0; j < length; j += 1) pairMaskValues[i * length + j] = seqMask[i] * seqMask[j];
  }
  const pairMask = execution.upload(pairMaskValues, "monomer.pair-mask");
  const aatype = execution.upload(
    Float32Array.from(featuresByRecycle[0].aatype), "monomer.aatype");
  const atom37ToAtom14 = execution.upload(
    Float32Array.from(featuresByRecycle[0].atom37ToAtom14), "monomer.atom37-to-14");
  const atom37MaskValues = Float32Array.from(featuresByRecycle[0].atom37Mask);
  const atom37Mask = execution.upload(atom37MaskValues, "monomer.atom37-mask");
  const residueIndex = execution.upload(
    Float32Array.from(featuresByRecycle[0].residueIndex), "af2.residue-index");
  // 🔴 THE CHAIN IDENTITY, AND A MONOMER'S IS ALL ZEROS RATHER THAN ABSENT. One
  // chain, one entity, one symmetry copy is what a monomer IS, so the chains
  // kernel lights none of its eight trailing rows and reproduces the monomer
  // lookup exactly. `makeA3mFeatures` emits the three lanes under `chainAware`.
  const identity = multimer
    ? execution.upload(residueIdentity({
      length,
      residueIndex: featuresByRecycle[0].residueIndex,
      asymId: featuresByRecycle[0].asymId,
      entityId: featuresByRecycle[0].entityId,
      symId: featuresByRecycle[0].symId,
    }), "af2.residue-identity")
    : undefined;

  // 🔴 THE MULTIMER TEMPLATE'S FEATURES ARE CONSTANT ACROSS PASSES EVEN THOUGH
  // ITS EMBEDDING IS NOT. The geometry and the template's aatype depend on the
  // template alone; only the query term reads the pair. So these two are
  // uploaded once and the embedder runs every pass over them.
  let templateGeometry;
  let templateAatype;
  if (multimer) {
    const features = multimerTemplateFeatures({
      template: call.template, length, asymId: featuresByRecycle[0].asymId,
    });
    templateGeometry = execution.upload(features.geometry, "af2.mt-geometry");
    templateAatype = execution.upload(features.aatype, "af2.mt-aatype");
  }

  // The monomer template update is computed from the sequence alone, so it is
  // computed ONCE and added to every pass's pair.
  let templateUpdate = null;
  if (w.template !== null && w.template !== undefined) {
    const features = templateFeatures({ template: call.template, length });
    const geometry = execution.upload(features.geometry, "monomer.template-geometry");
    const templateAatype = execution.upload(features.aatype, "monomer.template-aatype");
    templateUpdate = templateEmbedding(execution, {
      geometry, aatype: templateAatype, pairMask,
      weights: w.template.stage, blockWeights: w.template.blocks,
      length, templateChannels, pairChannels,
      triangleHidden: w.template.triangleHidden, pairHidden: w.template.pairHidden,
      // The template stack's pair track is 64 channels, not the trunk's 128.
      useTemplateUnitVector: call.useTemplateUnitVector,
    });
    execution.release(geometry);
    execution.release(templateAatype);
  }

  // 🔴 ZEROS, NOT AN ALLOCATION. The first pass has no previous activations and
  // the embedder reads all three, so a fresh texture's contents ARE the answer
  // here - and `allocate` does not promise any. See `Storage.zeros`.
  let previousMsa = execution.zeros(length * msaChannels, "monomer.recycle-msa");
  let previousPair = execution.zeros(pairs * pairChannels, "monomer.recycle-pair");
  let previousPositions = execution.zeros(length * 37 * 3, "monomer.recycle-positions");
  let previousAtom37 = new Float32Array(length * 37 * 3);

  // The regime the page passes for multimer, verbatim - "it is not defaults,
  // and dropping it once ran multimer WEIGHTS on the monomer graph".
  /** @type {{outerProductMeanFirst?: boolean, transitionPrecision?: "f32",
   *   msaAttentionPrecision?: "f32", triangleAttentionPrecision?: "f32"}} */
  const multimerRegime = multimer
    ? {
      outerProductMeanFirst: true,
      transitionPrecision: "f32",
      msaAttentionPrecision: "f32",
      triangleAttentionPrecision: "f32",
    }
    : {};

  const results = [];
  for (let recycle = 0; recycle < featuresByRecycle.length; recycle += 1) {
    const features = featuresByRecycle[recycle];
    if (features.aatype.length !== length) {
      throw new RangeError("foldAf2: all recycle feature lengths must match");
    }
    const msaMask = execution.upload(
      Float32Array.from(features.msaMask), `monomer.msa-mask-${recycle}`);
    const extraMsaMask = execution.upload(
      Float32Array.from(features.extraMsaMask), `monomer.extra-msa-mask-${recycle}`);
    const uploaded = [msaMask, extraMsaMask];
    const upload = (values, label) => {
      const tensor = execution.upload(Float32Array.from(values), label);
      uploaded.push(tensor);
      return tensor;
    };

    const embedding = inputEmbedder(execution, {
      targetFeatures: upload(features.targetFeatures, "monomer.target"),
      msaFeatures: upload(features.msaFeatures, "monomer.msa-features"),
      extraMsa: upload(features.extraMsa, "monomer.extra-codes"),
      extraHasDeletion: upload(features.extraHasDeletion, "monomer.extra-deletion-flag"),
      extraDeletionValue: upload(features.extraDeletionValue, "monomer.extra-deletion-value"),
      previousMsaFirstRow: previousMsa,
      previousPair,
      previousPositions,
      aatype, residueIndex,
      ...(identity === undefined ? {} : { residueIdentity: identity }),
      weights: w.embedding,
      length,
      msaSequences: features.msaSequences,
      extraSequences: features.extraSequences,
      targetChannels: features.targetChannels,
      msaFeatureChannels: features.msaFeatureChannels,
      msaChannels, pairChannels, extraMsaChannels,
    });
    for (const tensor of uploaded.slice(2)) execution.release(tensor);
    execution.release(previousMsa);
    execution.release(previousPair);
    execution.release(previousPositions);

    let pair = embedding.pairWithoutTemplates;
    // 🔴 AF2-MULTIMER'S TEMPLATE EMBEDDER RUNS EVERY PASS, templates or not: it
    // reads the pair through a LayerNorm, so it is a function of the pair rather
    // than a constant that could be folded in once the way the monomer's is.
    // 🔴 THE GRAPH DECIDES, NOT THE PRESENCE OF THE WEIGHTS. Reading only
    // `w.templateEmbedding` let a monomer-graph call run the multimer embedder
    // over geometry this function never built - which is how the tool's dither
    // fold, which had forgotten to pass `graph`, failed with "cannot read
    // properties of undefined". A graph is a property of the call, and a weight
    // set that carries more than the graph needs is normal.
    if (multimer && w.templateEmbedding !== undefined) {
      const update = multimerTemplateEmbedding(execution, {
        pair, geometry: templateGeometry, aatype: templateAatype, pairMask,
        weights: w.templateEmbedding.stage, blockWeights: w.templateEmbedding.blocks,
        length, pairChannels, templateChannels,
        triangleHidden: w.templateEmbedding.triangleHidden,
        pairHidden: w.templateEmbedding.pairHidden,
        templates: call.templates ?? 1,
      });
      const withTemplates = add(execution, pair, update);
      execution.release(update);
      execution.release(pair);
      pair = withTemplates;
    }
    if (templateUpdate !== null) {
      const withTemplate = add(execution, pair, templateUpdate);
      execution.release(pair);
      pair = withTemplate;
    }
    await yieldTo?.();

    const extra = extraMsaStack(execution, {
      msa: embedding.extraMsa, pair, msaMask: extraMsaMask, pairMask,
      sequences: features.extraSequences, length,
      cM: extraMsaChannels, cZ: pairChannels,
      cOuter: w.extraStack[0].outerProductMean.leftBias.elements,
      triangleHidden: w.extraStack[0].triangleMultiplicationOutgoing.linearAPBias.elements,
      msaHidden: w.extraStack[0].msaTransition.firstBias.elements,
      pairHidden: w.extraStack[0].pairTransition.firstBias.elements,
      blockWeights: w.extraStack,
      ...multimerRegime,
      onBlock: call.onBlock === undefined
        ? undefined : (block, blocks) => call.onBlock("extra", block, blocks),
    });
    execution.release(embedding.extraMsa);
    execution.release(pair);
    await yieldTo?.();

    const trunk = evoformerStack(execution, {
      msa: embedding.msa, pair: extra.pair, msaMask, pairMask,
      sequences: features.msaSequences, length,
      cM: msaChannels, cZ: pairChannels,
      cOuter: w.mainStack[0].outerProductMean.leftBias.elements,
      triangleHidden: w.mainStack[0].triangleMultiplicationOutgoing.linearAPBias.elements,
      msaHidden: w.mainStack[0].msaTransition.firstBias.elements,
      pairHidden: w.mainStack[0].pairTransition.firstBias.elements,
      blockWeights: w.mainStack,
      ...multimerRegime,
      onBlock: call.onBlock === undefined
        ? undefined : (block, blocks) => call.onBlock("main", block, blocks),
    });
    execution.release(embedding.msa);
    execution.release(extra.msa);
    execution.release(extra.pair);
    await yieldTo?.();

    // The structure module's single representation: row zero of the TRUNK's
    // MSA, copied out of a tensor holding every alignment row.
    const msaFirstRowTensor = prefix(execution, trunk.msa, length * msaChannels);
    execution.release(trunk.msa);
    const maskTensor = execution.upload(
      Float32Array.from(features.seqMask), `monomer.seq-mask-${recycle}`);
    const structure = structureModule(execution, {
      msaFirstRow: msaFirstRowTensor, pair: trunk.pair, mask: maskTensor,
      aatype, atom37ToAtom14, atom37Mask, length,
      weights: w.structure, tables: w.tables,
      shape: { msaChannels, pairChannels, structureChannels, positionScale },
    });
    execution.release(maskTensor);
    await yieldTo?.();

    const confidence = confidenceHeads(execution, {
      structure: structure.finalRepresentation, pair: trunk.pair, length,
      structureChannels, pairChannels,
      weights: w.confidence,
      hiddenChannels: w.confidence.lddt.act0Bias.elements,
      lddtBins: w.confidence.lddt.logitsBias.elements,
      breaks: paeBreaks,
    });

    let contactProbs;
    if (contactMap && w.distogramHead !== undefined && w.distogramHead !== null) {
      const contacts = distogramContacts(execution, {
        pair: trunk.pair, weights: w.distogramHead.logitsWeight,
        bias: w.distogramHead.logitsBias,
        length, channels: pairChannels, bins: w.distogramHead.bins ?? distogramBins,
        firstBreak: w.distogramHead.firstBreak, lastBreak: w.distogramHead.lastBreak,
      });
      contactProbs = Float32Array.from(execution.read(contacts));
      execution.release(contacts);
    }

    const msaFirstRow = Float32Array.from(execution.read(msaFirstRowTensor));
    const atom37 = Float32Array.from(execution.read(structure.atom37));
    const pairValues = pairHost ? Float32Array.from(execution.read(trunk.pair)) : undefined;
    const recycleDistance = recycleConvergenceDistance(
      previousAtom37, atom37, Float32Array.from(features.seqMask));
    results.push({
      msaFirstRow,
      pair: pairValues,
      contactProbs,
      recycleDistance,
      atom37,
      atom37Mask: atom37MaskValues,
      atom14: Float32Array.from(execution.read(structure.atom14)),
      affine: Float32Array.from(execution.read(structure.affine)),
      angles: Float32Array.from(execution.read(structure.angles)),
      unnormalizedAngles: Float32Array.from(execution.read(structure.unnormalizedAngles)),
      confidence,
    });
    onRecycle?.(results[results.length - 1], recycle);

    // What the next pass reads. The frames, the angles and the two coordinate
    // sets have been read back, so only the three carried tensors stay live.
    previousMsa = msaFirstRowTensor;
    previousPair = trunk.pair;
    previousPositions = structure.atom37;
    previousAtom37 = atom37;
    for (const tensor of [structure.atom14, structure.affine, structure.angles,
      structure.unnormalizedAngles, structure.finalRepresentation, msaMask, extraMsaMask]) {
      execution.release(tensor);
    }
    if (shouldStopAfterRecycle(recycle, recycleDistance, tolerance)) break;
    await yieldTo?.();
  }

  for (const tensor of [previousMsa, previousPair, previousPositions, pairMask,
    aatype, atom37ToAtom14, atom37Mask, residueIndex]) {
    execution.release(tensor);
  }
  for (const tensor of [identity, templateGeometry, templateAatype]) {
    if (tensor !== undefined) execution.release(tensor);
  }
  if (templateUpdate !== null) execution.release(templateUpdate);
  return { recycles: results, final: results[results.length - 1] };
}
