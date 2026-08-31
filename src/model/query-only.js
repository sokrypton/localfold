import { ConfidenceHeadsGpu } from "../heads/confidence.js";
import { InputEmbedderGpu } from "../evoformer/input-embedder.js";
import { EvoformerStackGpu, ExtraMsaPairStackGpu } from "../evoformer/stack.js";

import { QueryOnlyTemplateGpu } from "../evoformer/template.js";
import { StructureModuleGpu } from "../structure/module.js";
import { WebGpuExecution } from "../runtime/execution.js";
import { isAbortError, predictionAbortError, throwIfAborted, withAbort } from "../runtime/abort.js";
import {
  recycleConvergenceDistance, shouldStopAfterRecycle, validatedRecycleTolerance,
} from "./recycle-convergence.js";

import { makeQueryOnlyFeatures } from "../input/query-only-features.js";

/**
 * @typedef {import("../structure/module.js").StructureModuleResult} StructureModuleResult
 * @typedef {import("../heads/confidence.js").ConfidenceResult} ConfidenceResult
 */

/**
 * @typedef {object} QueryOnlyRecycleResult
 * @property {Float32Array} msaFirstRow
 * @property {Float32Array} pair
 * @property {StructureModuleResult} structure
 * @property {ConfidenceResult} confidence
 * @property {number} recycleDistance ColabFold C-alpha distance convergence metric, in angstroms
 * @property {number} elapsedMilliseconds
 */

/** @typedef {(result: QueryOnlyRecycleResult, recycle: number) => void} QueryOnlyRecycleCallback */

/**
 * @typedef {object} QueryOnlyPrediction
 * @property {readonly QueryOnlyRecycleResult[]} recycles
 * @property {QueryOnlyRecycleResult} final
 * @property {number} elapsedMilliseconds
 */

/** Model-1 query-only path, including templates, 54 pair blocks, structure, confidence, and recycling. */
export class AlphaFoldQueryOnlyGpu {
  device;
  constructor(device) { this.device = device; }

  async predictSequence(
    sequence,
    weights,
    featureTables,
    options = {},
    paeBreaks,
    onRecycle,
    onProgress,
  ) {
    return this.predict(makeQueryOnlyFeatures(sequence, featureTables, options), weights,
      paeBreaks, onRecycle, onProgress, { tolerance: options.tolerance, signal: options.signal, chainLengths: options.chainLengths });
  }

  /**
   * @param {QueryOnlyRecycleCallback} [onRecycle] called as each pass finishes,
   *   before the next one starts, so a caller can draw the structure while the
   *   remaining recycles are still running. Same contract as the monomer path.
   */
  async predict(
    recycleFeatures,
    weights,
    paeBreaks,
    onRecycle,
    onProgress,
    recycleOptions = {},
  ) {
    if (recycleFeatures.length === 0) throw new RangeError("at least one recycle feature set is required");
    const length = recycleFeatures[0] .aatype.length;
    const tolerance = validatedRecycleTolerance(recycleOptions.tolerance);
    const signal = recycleOptions.signal;
    throwIfAborted(signal);
    const pairMask = new Float32Array(length * length);
    for (let i = 0; i < length; i += 1) for (let j = 0; j < length; j += 1) {
      pairMask[i * length + j] = recycleFeatures[0] .seqMask[i] * recycleFeatures[0] .seqMask[j];
    }
    const template = await withAbort(new QueryOnlyTemplateGpu(this.device).run({
      length, templateChannels: 64, pairChannels: 128, pairMask, weights: weights.template,
    }), signal);
    throwIfAborted(signal);
    // 🔴 ONE PAIR BUFFER FOR THE WHOLE TRUNK.
    //
    // The pair representation is the largest tensor the model touches - L*L*128
    // floats, 1.7 MiB at 59 residues but 25 MiB at 221 and 44 MiB at 300. The
    // input embedder wrote it, read it back, and handed it to an elementwise add
    // that uploaded it again along with the template update, read the sum back,
    // gave it to the extra-MSA stack which uploaded it and read it back, and
    // then to the main stack which did the same: nine crossings of the bus per
    // pass to move a tensor between four stages that all run on the same device.
    //
    // Every one of them writes the pair in place, so they can share a single
    // resident buffer and the template residual becomes one add-in-place on it.
    // What is left is the ONE readback the main stack owes the structure module,
    // the confidence heads and the next recycle, all of which want it on the
    // host. The execution outlives every stage, which is what lets them share.
    const execution = new WebGpuExecution(this.device);
    try {
      const pairElements = length * length * 128;
      const pairTensor = execution.allocate(
        "trunk.pair", pairElements, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      );
      // ...THE TEMPLATE UPDATE DOES NOT CHANGE BETWEEN RECYCLES. It is computed
      // once above from the sequence alone, so it is uploaded once here rather
      // than pushed up again with every pass.
      const templateTensor = execution.upload("trunk.template-update", template.pairUpdate);
      let previousMsa = new Float32Array(length * 256);
      let previousPair = new Float32Array(length * length * 128);
      let previousPositions = new Float32Array(length * 37 * 3);
      // HOW MUCH WORK A FOLD IS, counted in the unit the run actually advances
      // in: an Evoformer block. Every pass is the extra-MSA pair stack plus the
      // main trunk, and then two coarser stages - the structure module and the
      // confidence heads - which are counted as one step each so the bar does not
      // sit still through them. The template runs once, before any of it.
      // WHAT A PASS IS MADE OF, in the units it actually advances through: the two
      // block stacks, then the structure module - which is four stages with eight
      // IPA iterations inside the second - and the confidence heads, which report
      // twice. As one step each the structure module and the heads left the bar
      // sitting at 89% for the longest part of a long fold.
      const STRUCTURE_STEPS = 11;    // initialize, 8 iterations, sidechains, geometry
      const CONFIDENCE_STEPS = 2;    // reading back, then scoring
      const blocksPerPass = weights.extraStack.length + weights.mainStack.length;
      const totalSteps = recycleFeatures.length * (blocksPerPass + STRUCTURE_STEPS + CONFIDENCE_STEPS);
      let completed = 0;
      const step = (count = 1) => {
        completed += count;
        onProgress?.({ completed, total: totalSteps, waiting: false });
      };
      // ...AND A PHASE, which is the honest report for work whose length is not
      // known in steps: the trunk queues 48 blocks and then waits for the device
      // to work through them, which is most of a long fold and has no internal
      // milestones to count.
      const waiting = (on) => onProgress?.({ completed, total: totalSteps, waiting: on });
      const results = [];
      const start = performance.now();
      for (let recycle = 0; recycle < recycleFeatures.length; recycle += 1) {
        throwIfAborted(signal);
        const recycleStart = performance.now();
        const features = recycleFeatures[recycle];
        if (features.aatype.length !== length) throw new RangeError("recycle lengths differ");
        const embedding = await withAbort(new InputEmbedderGpu(this.device).run({
          targetFeatures: features.targetFeatures,
          msaFeatures: features.msaFeatures,
          extraMsa: features.extraMsa,
          extraHasDeletion: features.extraHasDeletion,
          extraDeletionValue: features.extraDeletionValue,
          residueIndex: features.residueIndex,
          aatype: features.aatype,
          previousMsaFirstRow: previousMsa,
          previousPair,
          previousPositions,
          pairBuffer: pairTensor.allocation,
          keepPair: true,
          length,
          msaSequences: 1,
          extraSequences: features.extraSequences,
          targetChannels: features.targetChannels,
          msaFeatureChannels: features.msaFeatureChannels,
          msaChannels: 256,
          pairChannels: 128,
          extraMsaChannels: 64,
          weights: weights.embedding,
        }), signal);
        throwIfAborted(signal);
        // The template contribution, added where the pair already is.
        const residualEncoder = this.device.createCommandEncoder({ label: "trunk.template-residual" });
        this.device.pushErrorScope("validation");
        await execution.addInPlace(residualEncoder, pairTensor, templateTensor, "trunk.template-residual");
        execution.endComputePass(residualEncoder);
        this.device.queue.submit([residualEncoder.finish()]);
        const residualError = await this.device.popErrorScope();
        if (residualError !== null) {
          throw new Error(`WebGPU template residual failed: ${residualError.message}`);
        }
        throwIfAborted(signal);
        const extra = await withAbort(new ExtraMsaPairStackGpu(this.device).run({
          execution,
          msa: embedding.extraMsa,
          pairTensor,
          keepPair: true,
          msaMask: features.extraMsaMask,
          pairMask,
          sequences: features.extraSequences,
          length,
          cM: 64,
          cZ: 128,
          cOuter: weights.extraStack[0] .outerProductMean.leftBias.length,
          triangleHidden: weights.extraStack[0] .triangleMultiplicationOutgoing.linearAPBias.length,
          blockWeights: weights.extraStack,
          signal,
          // ...WHEN THE DEVICE FINISHES A BLOCK, not when one is queued. The stack
          // queues all 48 ahead, so anything counted at encode time arrives in
          // the first moment and tells the reader nothing.
          onBlockDone: () => step(),
          onStage: (stage) => waiting(stage === "gpu"),
        }), signal);
        const trunk = await withAbort(new EvoformerStackGpu(this.device).run({
          execution,
          msa: embedding.msa,
          pairTensor: extra.pairTensor,
          msaMask: features.msaMask,
          pairMask,
          sequences: 1,
          length,
          cM: 256,
          cZ: 128,
          cOuter: weights.mainStack[0] .outerProductMean.leftBias.length,
          triangleHidden: weights.mainStack[0] .triangleMultiplicationOutgoing.linearAPBias.length,
          blockWeights: weights.mainStack,
          signal,
          profileBlock: recycle === 0 && this.device.features.has("timestamp-query") ? 0 : undefined,
          // ...WHEN THE DEVICE FINISHES A BLOCK, not when one is queued. The stack
          // queues all 48 ahead, so anything counted at encode time arrives in
          // the first moment and tells the reader nothing.
          onBlockDone: () => step(),
          onStage: (stage) => waiting(stage === "gpu"),
        }), signal);
        throwIfAborted(signal);
        const msaFirstRow = trunk.msa.subarray(0, length * 256).slice();
        const structure = await withAbort(new StructureModuleGpu(this.device).run({
          msaFirstRow,
          pair: trunk.pair,
          mask: features.seqMask,
          aatype: features.aatype,
          atom37ToAtom14: features.atom37ToAtom14,
          atom37Mask: features.atom37Mask,
          length,
          weights: weights.structure,
          geometry: weights.geometry,
          signal,
          onStep: () => step(),
        }), signal);
        throwIfAborted(signal);
        const confidence = await withAbort(new ConfidenceHeadsGpu(this.device).run(
          structure.finalRepresentation, trunk.pair, length, weights.lddt, weights.pae, paeBreaks,
          () => step(),
          signal,
          recycleOptions.chainLengths,
        ), signal);
        throwIfAborted(signal);
        const recycleDistance = recycleConvergenceDistance(
          previousPositions, structure.atom37, features.seqMask,
        );
        const recycleResult = {
          msaFirstRow, pair: trunk.pair, structure, confidence, recycleDistance,
          elapsedMilliseconds: performance.now() - recycleStart,
        };
        results.push(recycleResult);
        onRecycle?.(recycleResult, recycle);
        throwIfAborted(signal);
        if (shouldStopAfterRecycle(recycle, recycleDistance, tolerance)) break;
        previousMsa = msaFirstRow;
        previousPair = trunk.pair;
        previousPositions = structure.atom37;
      }
      // ...AND THE BUDGET IS CHECKED AGAINST WHAT WAS ACTUALLY REPORTED. The two
      // are written in different files - a stage added to the structure module
      // without a matching number here would silently leave the bar short, which
      // is the bug this whole block exists to fix.
      // ...OVERFLOW ONLY. Block completions arrive as promises resolve, so the
      // last few may land after this line - falling short here is timing, not a
      // miscount. Going OVER the budget is always a bug, and that is worth saying.
      if (onProgress !== undefined && completed > totalSteps) {
        console.warn(`progress reported ${completed} steps against a budget of ${totalSteps}`);
      }
      return { recycles: results, final: results[results.length - 1], elapsedMilliseconds: performance.now() - start };
    } finally {
      execution.release();
    }
  }
}
