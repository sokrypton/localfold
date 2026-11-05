/**
 * AlphaFold 2 monomer, end to end: the driver that makes the stages a fold.
 *
 *     once:        templateUpdate = templateEmbedding(geometry features)
 *     every pass:  msa, extra, pair = inputEmbedder(features, recycled state)
 *                  pair += templateUpdate
 *                  extra stack   (global column attention, 4 blocks)
 *                  main stack    (48 blocks)
 *                  structure module, confidence heads, contact map
 *                  carry the MSA's first row, the pair and the atom37 forward
 *
 * `webgpu/af2/model/monomer.js` is the order and the conventions; this is the
 * same graph without the command-buffer windowing, which has no counterpart
 * here because a draw is submitted as it is issued.
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
  inputEmbedder, releaseInputEmbedderWeights, uploadInputEmbedderWeights,
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
 *   targetChannels: number, msaFeatureChannels: number}} RecycleFeatures
 */

/**
 * AF2 monomer's widths, which are the checkpoint's and not ours to choose.
 * `webgpu/af2/model/monomer.js` passes the same numbers as literals.
 */
export const MONOMER_SHAPE = {
  msaChannels: 256,
  extraMsaChannels: 64,
  pairChannels: 128,
  templateChannels: 64,
  structureChannels: 384,
  distogramBins: 64,
};

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
export function uploadMonomerWeights(execution, weights) {
  const hidden = (group) => ({
    triangleHidden: group.triangleMultiplicationOutgoing.linearAPBias.length,
    pairHidden: group.pairTransition.firstBias.length,
  });
  return {
    embedding: uploadInputEmbedderWeights(execution, weights.embedding),
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

export function releaseMonomerWeights(execution, weights) {
  releaseInputEmbedderWeights(execution, weights.embedding);
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
 * @param {{featuresByRecycle: readonly RecycleFeatures[],
 *   weights: ReturnType<typeof uploadMonomerWeights>,
 *   paeBreaks: ArrayLike<number>,
 *   template?: Parameters<typeof templateFeatures>[0]["template"],
 *   useTemplateUnitVector?: boolean,
 *   tolerance?: number, pairHost?: boolean, contactMap?: boolean,
 *   shape?: Partial<typeof MONOMER_SHAPE>,
 *   onRecycle?: (result: object, recycle: number) => void,
 *   onBlock?: (stack: "extra" | "main", block: number, blocks: number) => void,
 *   yieldTo?: () => Promise<void>}} call
 * @param {import("../../runtime/execution.js").Execution} execution
 * @returns {Promise<{recycles: object[], final: object}>}
 */
export async function foldMonomer(execution, call) {
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
    throw new RangeError("foldMonomer: at least one feature set is required");
  }
  if (w.extraStack.length === 0 || w.mainStack.length === 0) {
    throw new RangeError("foldMonomer: the extra and main Evoformer stacks cannot be empty");
  }
  const {
    msaChannels, extraMsaChannels, pairChannels, templateChannels,
    structureChannels, distogramBins,
  } = { ...MONOMER_SHAPE, ...(call.shape ?? {}) };
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
    Float32Array.from(featuresByRecycle[0].residueIndex), "monomer.residue-index");

  // The template update is computed from the sequence alone, so it is computed
  // ONCE and added to every pass's pair.
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

  let previousMsa = execution.allocate(length * msaChannels, "monomer.recycle-msa");
  let previousPair = execution.allocate(pairs * pairChannels, "monomer.recycle-pair");
  let previousPositions = execution.allocate(length * 37 * 3, "monomer.recycle-positions");
  let previousAtom37 = new Float32Array(length * 37 * 3);

  const results = [];
  for (let recycle = 0; recycle < featuresByRecycle.length; recycle += 1) {
    const features = featuresByRecycle[recycle];
    if (features.aatype.length !== length) {
      throw new RangeError("foldMonomer: all recycle feature lengths must match");
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
      shape: { msaChannels, pairChannels, structureChannels },
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
  if (templateUpdate !== null) execution.release(templateUpdate);
  return { recycles: results, final: results[results.length - 1] };
}
