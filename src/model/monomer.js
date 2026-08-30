import { ConfidenceHeadsGpu } from "../heads/confidence.js";
import { encodeInputEmbedder } from "../evoformer/input-embedder.js";
import {
  encodeEvoformerBlock, encodeExtraMsaBlock,
} from "../evoformer/block.js";
import { QueryOnlyTemplateGpu } from "../evoformer/template.js";
import { WebGpuExecution } from "../runtime/execution.js";
import { StructureModuleGpu } from "../structure/module.js";

import { makeA3mFeatures } from "../input/a3m-features.js";

/**
 * @typedef {import("../structure/module.js").StructureModuleResult} StructureModuleResult
 * @typedef {import("../heads/confidence.js").ConfidenceResult} ConfidenceResult
 */

/**
 * One pass of the trunk, the structure module and the confidence heads.
 * @typedef {object} MonomerRecycleResult
 * @property {Float32Array} msaFirstRow
 * @property {Float32Array} pair
 * @property {StructureModuleResult} structure
 * @property {ConfidenceResult} confidence
 * @property {number} elapsedMilliseconds
 */

/**
 * @typedef {object} MonomerPrediction
 * @property {readonly MonomerRecycleResult[]} recycles  every pass, in order
 * @property {MonomerRecycleResult} final                the last one
 * @property {number} elapsedMilliseconds
 */

/** @typedef {(result: MonomerRecycleResult, recycle: number) => void} MonomerRecycleCallback */

/** Full monomer model for clustered MSA/A3M inputs, with all learned operations dispatched through WebGPU. */
export class AlphaFoldMonomerGpu {
  device;
  constructor(device) { this.device = device; }
  async predictA3m(a3mText, weights, featureTables,
    options = {}, paeBreaks,
    onRecycle, onProgress) {
    return this.predict(makeA3mFeatures(a3mText, featureTables, options), weights, paeBreaks,
      onRecycle, onProgress);
  }
  /**
   * @param {(p: {completed: number, total: number, waiting: boolean}) => void} [onProgress]
   *   called as the run advances, in Evoformer blocks. The A3M path is the slow
   *   one - a minute or more - and without this its status line said "Folding"
   *   and nothing else for the whole of it, which reads as a hang.
   */
  async predict(featuresByRecycle, weights,
    paeBreaks, onRecycle, onProgress) {
    if (featuresByRecycle.length === 0) throw new RangeError("at least one feature set is required");
    const length = featuresByRecycle[0] .aatype.length;
    const pairMask = new Float32Array(length * length);
    for (let i = 0; i < length; i += 1) for (let j = 0; j < length; j += 1) {
      pairMask[i * length + j] = featuresByRecycle[0] .seqMask[i] * featuresByRecycle[0] .seqMask[j];
    }
    const template = await new QueryOnlyTemplateGpu(this.device).run({
      length, templateChannels: 64, pairChannels: 128, pairMask, weights: weights.template,
    });
    if (weights.extraStack.length === 0 || weights.mainStack.length === 0) {
      throw new RangeError("AlphaFold monomer requires non-empty extra and main Evoformer stacks");
    }
    const execution = new WebGpuExecution(this.device);
    const results = [];
    const start = performance.now();
    // WHAT A PASS IS MADE OF, in the units it advances through: both block
    // stacks, then the structure module - four stages with eight IPA iterations
    // inside the second - and the confidence heads, which report twice. The
    // same budget the single-sequence path uses, so the two bars mean the same.
    const STRUCTURE_STEPS = 11;    // initialize, 8 iterations, sidechains, geometry
    const CONFIDENCE_STEPS = 2;    // reading back, then scoring
    const blocksPerPass = weights.extraStack.length + weights.mainStack.length;
    const totalSteps = featuresByRecycle.length
      * (blocksPerPass + STRUCTURE_STEPS + CONFIDENCE_STEPS);
    let completed = 0;
    const step = () => {
      completed += 1;
      onProgress?.({ completed, total: totalSteps, waiting: false });
    };
    const submit = async(encoder, label) => {
      execution.endComputePass(encoder);
      this.device.queue.submit([encoder.finish()]);
      const error = await this.device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU ${label} failed: ${error.message}`);
    };
    const releaseTensor = (tensor) => tensor.allocation.release();
    try {
      const templateUpdate = execution.upload("monomer.template-update", template.pairUpdate);
      const pairMaskTensor = execution.upload("monomer.pair-mask", pairMask);
      let previousMsa = execution.upload("monomer.recycle-msa-zero", new Float32Array(length * 256));
      let previousPair = execution.upload("monomer.recycle-pair-zero", new Float32Array(length * length * 128));
      let previousPositions = execution.upload(
        "monomer.recycle-positions-zero", new Float32Array(length * 37 * 3),
      );

      for (let recycle = 0; recycle < featuresByRecycle.length; recycle += 1) {
        const features = featuresByRecycle[recycle];
        if (features.aatype.length !== length) throw new RangeError("all recycle feature lengths must match");
        const recycleStart = performance.now();
        const msaMask = execution.upload(`monomer.msa-mask-${recycle}`, features.msaMask);
        const extraMsaMask = execution.upload(`monomer.extra-msa-mask-${recycle}`, features.extraMsaMask);
        const embeddingEncoder = this.device.createCommandEncoder({ label: `monomer.embedding-${recycle}` });
        this.device.pushErrorScope("validation");
        const embedding = await encodeInputEmbedder(execution, embeddingEncoder, {
          ...features,
          previousMsaFirstRow: new Float32Array(0), previousPair: new Float32Array(0),
          previousPositions: new Float32Array(0), length,
          msaChannels: 256, pairChannels: 128, extraMsaChannels: 64, weights: weights.embedding,
        }, previousMsa, previousPair, previousPositions);
        await execution.addInPlace(
          embeddingEncoder, embedding.pairWithoutTemplates, templateUpdate, `monomer.template-residual-${recycle}`,
        );
        await submit(embeddingEncoder, `embedding recycle ${recycle}`);
        for (const temporary of embedding.temporaries) releaseTensor(temporary);
        releaseTensor(previousMsa); releaseTensor(previousPair); releaseTensor(previousPositions);

        const extraShape = {
          sequences: features.extraSequences, length, cM: 64, cZ: 128,
          cOuter: weights.extraStack[0] .outerProductMean.leftBias.length,
          triangleHidden: weights.extraStack[0] .triangleMultiplicationOutgoing.linearAPBias.length,
        };
        for (let block = 0; block < weights.extraStack.length; block += 1) {
          const checkpoint = execution.checkpoint();
          const encoder = this.device.createCommandEncoder({ label: `monomer.extra-${recycle}-${block}` });
          this.device.pushErrorScope("validation");
          await encodeExtraMsaBlock(execution, encoder, extraShape, weights.extraStack[block],
            embedding.extraMsa, embedding.pairWithoutTemplates, extraMsaMask, pairMaskTensor);
          await submit(encoder, `extra-MSA recycle ${recycle} block ${block}`);
          execution.releaseSince(checkpoint);
          // ...WHEN THE DEVICE REACHES THIS BLOCK, not when one is queued.
          // Unawaited, so the loop carries straight on encoding and the
          // pipelining that makes this fast is untouched.
          void this.device.queue.onSubmittedWorkDone().then(step);
        }
        releaseTensor(embedding.extraMsa); releaseTensor(extraMsaMask);

        const mainDescriptor = {
          msa: new Float32Array(0), pair: new Float32Array(0), msaMask: new Float32Array(0),
          pairMask: new Float32Array(0), sequences: features.msaSequences, length, cM: 256, cZ: 128,
          cOuter: weights.mainStack[0] .outerProductMean.leftBias.length,
          triangleHidden: weights.mainStack[0] .triangleMultiplicationOutgoing.linearAPBias.length,
        };
        for (let block = 0; block < weights.mainStack.length; block += 1) {
          const checkpoint = execution.checkpoint();
          const encoder = this.device.createCommandEncoder({ label: `monomer.main-${recycle}-${block}` });
          this.device.pushErrorScope("validation");
          await encodeEvoformerBlock(execution, encoder, {
            ...mainDescriptor, weights: weights.mainStack[block],
          }, embedding.msa, embedding.pairWithoutTemplates, msaMask, pairMaskTensor);
          await submit(encoder, `main Evoformer recycle ${recycle} block ${block}`);
          execution.releaseSince(checkpoint);
          void this.device.queue.onSubmittedWorkDone().then(step);
        }

        const readbackEncoder = this.device.createCommandEncoder({ label: `monomer.readback-${recycle}` });
        const msaFirstRowTensor = execution.allocate(
          `monomer.msa-first-row-readback-${recycle}`, length * 256,
          GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        );
        execution.endComputePass(readbackEncoder);
        readbackEncoder.copyBufferToBuffer(
          embedding.msa.allocation.buffer, 0, msaFirstRowTensor.allocation.buffer, 0, length * 256 * 4,
        );
        const pairReadback = execution.createReadback(
          `monomer.pair-readback-${recycle}`, embedding.pairWithoutTemplates, readbackEncoder,
        );
        this.device.pushErrorScope("validation");
        await submit(readbackEncoder, `readback recycle ${recycle}`);
        const [msaFirstRow, pair] = await Promise.all([
          execution.mapFloat32(msaFirstRowTensor), execution.mapFloat32(pairReadback),
        ]);
        releaseTensor(msaFirstRowTensor); releaseTensor(pairReadback); releaseTensor(msaMask);

        const structure = await new StructureModuleGpu(this.device).run({
          msaFirstRow, pair, mask: features.seqMask, aatype: features.aatype,
          atom37ToAtom14: features.atom37ToAtom14, atom37Mask: features.atom37Mask,
          length, weights: weights.structure, geometry: weights.geometry,
          onStep: step,
        });
        const confidence = await new ConfidenceHeadsGpu(this.device).run(
          structure.finalRepresentation, pair, length, weights.lddt, weights.pae, paeBreaks,
          step,
        );
        const recycleResult = { msaFirstRow, pair, structure, confidence,
          elapsedMilliseconds: performance.now() - recycleStart };
        results.push(recycleResult);
        onRecycle?.(recycleResult, recycle);
        previousMsa = embedding.msa;
        previousPair = embedding.pairWithoutTemplates;
        previousPositions = execution.upload(`monomer.recycle-positions-${recycle}`, structure.atom37);
      }
      return {
        recycles: results, final: results[results.length - 1], elapsedMilliseconds: performance.now() - start,
      };
    } finally {
      execution.release();
    }
  }
}
