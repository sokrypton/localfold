import { ConfidenceHeadsGpu } from "../heads/confidence.js";
import { encodeInputEmbedder } from "../evoformer/input-embedder.js";
import {
  encodeEvoformerBlock, encodeExtraMsaBlock,
} from "../evoformer/block.js";
import { QueryOnlyTemplateGpu } from "../evoformer/template.js";
import { WebGpuExecution } from "../runtime/execution.js";
import { af2Plan, planTotal } from "../runtime/cost-model.js";
import { isAbortError, predictionAbortError, throwIfAborted, withAbort } from "../runtime/abort.js";
import { DeferredValidation } from "../runtime/validation.js";
import { StructureModuleGpu } from "../structure/module.js";
import {
  recycleConvergenceDistance, shouldStopAfterRecycle, validatedRecycleTolerance,
} from "./recycle-convergence.js";

import { makeA3mFeatures, makeA3mFeaturesOnDevice } from "../input/a3m-features.js";

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
 * @property {number} recycleDistance ColabFold C-alpha distance convergence metric, in angstroms
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
    // 🔴 THE ALIGNMENT PREP IS TIMED TOO, because it is main-thread JavaScript
    // outside every GPU clock and docs/AF2.md has already had to rewrite it once.
    const featureStart = performance.now();
    // 🔴 THE NEAREST-CENTRE SEARCH GOES TO THE DEVICE. It is 59% of preparing
    // an alignment - 640 ms of 1072 at 825 residues with two recycles - and it
    // is serial with the fold, so it is main-thread time the GPU sits out.
    // `options.hostFeaturisation` is the control arm, not a fallback: a device
    // that cannot run the kernel raises rather than quietly reverting.
    const features = options.hostFeaturisation === true
      ? makeA3mFeatures(a3mText, featureTables, options)
      : await makeA3mFeaturesOnDevice(this.device, a3mText, featureTables, options);
    const featureMilliseconds = performance.now() - featureStart;
    // 🔴 FORWARD THE WHOLE OPTIONS OBJECT, for the reason src/multimer/model.js
    // gives at the same seam and this one had to learn separately. The
    // hand-copied allow-list here went stale the moment `pairHost` was added:
    // web/app.js asks for it through this method for the distogram contact
    // overlay, the list dropped it, `recycle.pair` came back undefined, and
    // `attachContactMap` returns early on exactly that - so the overlay
    // silently disappeared from the shipped page and nothing raised.
    // tools/gpu/probe-af2-contacts.js, the one gate that would have caught it,
    // was broken by the same change in the same way.
    // An allow-list of options is a list that goes stale every time one is
    // added; the feature-building keys predict() does not read are harmless.
    const prediction = await this.predict(features, weights, paeBreaks,
      onRecycle, onProgress, options);
    if (prediction.stageMilliseconds !== undefined) {
      prediction.stageMilliseconds.features = featureMilliseconds;
    }
    return prediction;
  }
  /**
   * @param {(p: {completed: number, total: number, waiting: boolean}) => void} [onProgress]
   *   called as the run advances, in Evoformer blocks. The A3M path is the slow
   *   one - a minute or more - and without this its status line said "Folding"
   *   and nothing else for the whole of it, which reads as a hang.
   */
  async predict(featuresByRecycle, weights,
    paeBreaks, onRecycle, onProgress, recycleOptions = {}) {
    if (featuresByRecycle.length === 0) throw new RangeError("at least one feature set is required");
    const length = featuresByRecycle[0] .aatype.length;
    const tolerance = validatedRecycleTolerance(recycleOptions.tolerance);
    const signal = recycleOptions.signal;
    throwIfAborted(signal);
    // 🔴 THE CLOCK STARTS HERE, BEFORE THE PAIR MASK AND THE TEMPLATE. Those two
    // ran outside every phase and outside the progress stream - a 680,625-element
    // JavaScript loop and a whole template pair track - and showed up only as the
    // gap between the phases and the wall.
    const stageMilliseconds = {
      pairMask: 0, template: 0, setup: 0,
      embedder: 0, warm: 0, extraStack: 0, mainStack: 0, trunkReadback: 0,
      structure: 0, confidence: 0, convergence: 0, resumable: 0,
    };
    let phaseStart = performance.now();
    const pairMask = new Float32Array(length * length);
    for (let i = 0; i < length; i += 1) for (let j = 0; j < length; j += 1) {
      pairMask[i * length + j] = featuresByRecycle[0] .seqMask[i] * featuresByRecycle[0] .seqMask[j];
    }
    stageMilliseconds.pairMask = performance.now() - phaseStart;
    phaseStart = performance.now();
    if (weights.extraStack.length === 0 || weights.mainStack.length === 0) {
      throw new RangeError("AlphaFold monomer requires non-empty extra and main Evoformer stacks");
    }
    const execution = new WebGpuExecution(this.device);
    const results = [];
    const start = performance.now();
    // WHAT A PASS IS MADE OF, in the units it advances through: both block
    // stacks, then the structure module - four stages with eight IPA iterations
    // inside the second - and the confidence heads, which report twice.
    //
    // 🔴 WEIGHTED BY WHAT EACH ONE COSTS, WHICH IT DID NOT USED TO BE. Every
    // one of these counted as one step, so eleven IPA iterations and two
    // confidence reads were a quarter of the bar at 59 residues - and at 512
    // alignment rows an evoformer block is a hundred times an IPA iteration.
    // The bar therefore crawled through the stacks and then jumped to the end.
    // The two stacks are not equal either: they run at DIFFERENT DEPTHS, and
    // AF2's cost is mostly its depth. src/runtime/cost-model.js has the fits.
    const STRUCTURE_STEPS = 11;    // initialize, 8 iterations, sidechains, geometry
    const CONFIDENCE_STEPS = 2;    // reading back, then scoring
    const extraRows = featuresByRecycle[0] .extraSequences ?? 1;
    const mainRows = featuresByRecycle[0] .msaSequences ?? 1;
    const plan = af2Plan({
      length, extraRows, mainRows,
      extraBlocks: weights.extraStack.length,
      mainBlocks: weights.mainStack.length,
      passes: featuresByRecycle.length,
      structureSteps: STRUCTURE_STEPS, confidenceSteps: CONFIDENCE_STEPS,
    });
    const unitsOf = (name) => plan.stages.find((stage) => stage.name === name).units;
    const EXTRA_BLOCK = unitsOf("extra-stack");
    const MAIN_BLOCK = unitsOf("main-stack");
    const STRUCTURE_STEP = unitsOf("structure");
    const CONFIDENCE_STEP = unitsOf("confidence");
    const totalSteps = planTotal(plan);
    let completed = 0;
    // `units` defaults to a structure step because that is what the callers
    // which cannot pass one - StructureModuleGpu's onStep - are.
    // 🔴 THE CALLERS DO NOT ALL PASS A NUMBER. StructureModuleGpu reports its
    // stages by NAME - step("initialize"), step("sidechains") - which the old
    // counter ignored because it added one whatever it was handed. Adding a
    // string instead made `completed` NaN, which reached the progress element
    // as "the provided double value is non-finite" and failed the fold. A label
    // means "one step of whatever I am", which is the structure module's step.
    stageMilliseconds.setup = performance.now() - phaseStart;
    phaseStart = performance.now();
    const step = (units) => {
      completed += Number.isFinite(units) ? units : STRUCTURE_STEP;
      onProgress?.({ completed, total: totalSteps, waiting: false });
    };
    // 🔴 THE SCOPE OPENS AT THE ENCODER, NOT HERE. Validation errors are raised
    // as commands are encoded far more often than when a buffer is submitted,
    // and a scope pushed at submit time covers only the rarer half.
    const encode = (label) => {
      this.device.pushErrorScope("validation");
      return this.device.createCommandEncoder({ label });
    };
    const submit = async(encoder, label) => {
      execution.endComputePass(encoder);
      this.device.queue.submit([encoder.finish()]);
      const error = await this.device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU ${label} failed: ${error.message}`);
    };
    const releaseTensor = (tensor) => tensor.allocation.release();
    try {
      // 🔴 THE TEMPLATE WRITES STRAIGHT INTO THIS, so its pair update never
      // leaves the device. It used to come back as a host Float32Array -
      // `L^2 * 128` floats, **348 MB at 825 residues** - and go up again on the
      // next line. The GPU work in the whole template embedder is 130 ms at
      // that size; the call took 1040.
      const templateUpdate = execution.allocate(
        "monomer.template-update", length * length * 128);
      await withAbort(new QueryOnlyTemplateGpu(this.device).run({
        length, templateChannels: 64, pairChannels: 128, pairMask, weights: weights.template,
        outputTensor: templateUpdate,
      }), signal);
      stageMilliseconds.template = performance.now() - phaseStart;
      phaseStart = performance.now();
      throwIfAborted(signal);
      const pairMaskTensor = execution.upload("monomer.pair-mask", pairMask);
      // 🔴 A RECYCLE'S STATE IS FOUR THINGS, and `resume` is all four from a
      // previous run - so asking for more recycles runs the difference rather
      // than starting again. It is sound because the per-recycle features do
      // not depend on how many were asked for: a3m-features.js seeds each pass
      // from `randomSeed ^ hash(recycle + 1)`, so features[k] is the same in a
      // three-recycle run and a five-recycle one, and a continuation lands on
      // the structure the longer run would have produced.
      const resume = recycleOptions.resume;
      // 🔴 A ZERO BUFFER IS ALLOCATED, NOT UPLOADED. Without a continuation the
      // recycle state is all zeros, and this built them in JavaScript and pushed
      // them across the bus: at 825 residues the pair state alone is
      // **348 MB of zeros**, a Float32Array of 87 million elements and a
      // writeBuffer of the same. WebGPU zero-initialises a new buffer, so the
      // whole thing is one allocation the driver already had to do.
      const zeros = (label, elements) => execution.allocate(label, elements);
      let previousMsa = resume?.msa === undefined
        ? zeros("monomer.recycle-msa-zero", length * 256)
        : execution.upload("monomer.recycle-msa", resume.msa);
      let previousPair = resume?.pair === undefined
        ? zeros("monomer.recycle-pair-zero", length * length * 128)
        : execution.upload("monomer.recycle-pair", resume.pair);
      let previousPositions = resume?.atom37 === undefined
        ? zeros("monomer.recycle-positions-zero", length * 37 * 3)
        : execution.upload("monomer.recycle-positions", resume.atom37);
      let previousAtom37 = resume?.atom37 ?? new Float32Array(length * 37 * 3);

      // Features are built for every pass and only the outstanding ones run;
      // indexing by absolute recycle is what keeps a continuation on the same
      // random stream as the run it continues.
      const firstRecycle = resume === undefined ? 0 : resume.recycles + 1;
      for (let recycle = firstRecycle; recycle < featuresByRecycle.length; recycle += 1) {
        throwIfAborted(signal);
        const features = featuresByRecycle[recycle];
        if (features.aatype.length !== length) throw new RangeError("all recycle feature lengths must match");
        const recycleStart = performance.now();
        const msaMask = execution.upload(`monomer.msa-mask-${recycle}`, features.msaMask);
        const extraMsaMask = execution.upload(`monomer.extra-msa-mask-${recycle}`, features.extraMsaMask);
        const embeddingEncoder = encode(`monomer.embedding-${recycle}`);
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
        throwIfAborted(signal);
        for (const temporary of embedding.temporaries) releaseTensor(temporary);
        releaseTensor(previousMsa); releaseTensor(previousPair); releaseTensor(previousPositions);

        const extraShape = {
          sequences: features.extraSequences, length, cM: 64, cZ: 128,
          cOuter: weights.extraStack[0] .outerProductMean.leftBias.length,
          triangleHidden: weights.extraStack[0] .triangleMultiplicationOutgoing.linearAPBias.length,
        };
        stageMilliseconds.embedder += performance.now() - phaseStart;
        phaseStart = performance.now();
        const windowSize = signal !== undefined ? 8 : weights.mainStack.length;
        const validation = new DeferredValidation(this.device, `recycle ${recycle}`);
        const mainDescriptor = {
          msa: new Float32Array(0), pair: new Float32Array(0), msaMask: new Float32Array(0),
          pairMask: new Float32Array(0), sequences: features.msaSequences, length, cM: 256, cZ: 128,
          cOuter: weights.mainStack[0] .outerProductMean.leftBias.length,
          triangleHidden: weights.mainStack[0] .triangleMultiplicationOutgoing.linearAPBias.length,
        };
        // 🔴 THE TWO STACKS' PIPELINES, ASKED FOR ALL AT ONCE, BEFORE EITHER
        // RUNS. A first fold used to be its own compile queue: 95 pipelines
        // requested at the moment each kernel was first encoded, an average of
        // 1.47 ever in flight, and a span of 1133 ms inside a 1163 ms fold.
        // This drives the same two block functions in warm mode - no encoder,
        // no dispatch, just their pipelines and their resident weights - so
        // this browser's compiler gets all of them together, which it does 5.4x
        // faster. Recycle 0 only: by the second the cache is full.
        // 🔴 AND IT WARMS BLOCK 0 OF EACH, WHICH IS EVERY KEY BOTH STACKS USE.
        // The blocks differ in their WEIGHTS and not in their shapes, and a
        // pipeline key is shapes; 48 main blocks share one set.
        if (recycle === 0) {
          await execution.warm(async() => {
            await Promise.all([
              encodeExtraMsaBlock(execution, undefined, extraShape, weights.extraStack[0],
                embedding.extraMsa, embedding.pairWithoutTemplates, extraMsaMask, pairMaskTensor),
              encodeEvoformerBlock(execution, undefined, {
                ...mainDescriptor, weights: weights.mainStack[0],
              }, embedding.msa, embedding.pairWithoutTemplates, msaMask, pairMaskTensor),
            ]);
          });
          stageMilliseconds.warm += performance.now() - phaseStart;
          phaseStart = performance.now();
        }
        for (let block = 0; block < weights.extraStack.length; block += 1) {
          throwIfAborted(signal);
          const checkpoint = execution.checkpoint();
          const encoder = this.device.createCommandEncoder({ label: `monomer.extra-${recycle}-${block}` });
          validation.begin();
          await encodeExtraMsaBlock(execution, encoder, extraShape, weights.extraStack[block],
            embedding.extraMsa, embedding.pairWithoutTemplates, extraMsaMask, pairMaskTensor);
          execution.endComputePass(encoder);
          this.device.queue.submit([encoder.finish()]);
          validation.end(`extra-MSA block ${block}`);
          execution.releaseSince(checkpoint);
          const endOfWindow = (block + 1) % windowSize === 0 || block + 1 === weights.extraStack.length;
          if (endOfWindow) await withAbort(this.device.queue.onSubmittedWorkDone(), signal);
          void this.device.queue.onSubmittedWorkDone().then(() => step(EXTRA_BLOCK));
        }
        stageMilliseconds.extraStack += performance.now() - phaseStart;
        phaseStart = performance.now();
        releaseTensor(embedding.extraMsa); releaseTensor(extraMsaMask);

        for (let block = 0; block < weights.mainStack.length; block += 1) {
          throwIfAborted(signal);
          const checkpoint = execution.checkpoint();
          const encoder = this.device.createCommandEncoder({ label: `monomer.main-${recycle}-${block}` });
          validation.begin();
          await encodeEvoformerBlock(execution, encoder, {
            ...mainDescriptor, weights: weights.mainStack[block],
          }, embedding.msa, embedding.pairWithoutTemplates, msaMask, pairMaskTensor);
          execution.endComputePass(encoder);
          this.device.queue.submit([encoder.finish()]);
          validation.end(`main Evoformer block ${block}`);
          execution.releaseSince(checkpoint);
          const endOfWindow = (block + 1) % windowSize === 0 || block + 1 === weights.mainStack.length;
          if (endOfWindow) await withAbort(this.device.queue.onSubmittedWorkDone(), signal);
          void this.device.queue.onSubmittedWorkDone().then(() => step(MAIN_BLOCK));
        }

        await validation.settle();
        stageMilliseconds.mainStack += performance.now() - phaseStart;
        phaseStart = performance.now();
        const readbackEncoder = encode(`monomer.readback-${recycle}`);
        const msaFirstRowTensor = execution.allocate(
          `monomer.msa-first-row-readback-${recycle}`, length * 256,
          GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        );
        execution.endComputePass(readbackEncoder);
        readbackEncoder.copyBufferToBuffer(
          embedding.msa.allocation.buffer, 0, msaFirstRowTensor.allocation.buffer, 0, length * 256 * 4,
        );
        // 🔴 THE PAIR REPRESENTATION STAYS ON THE DEVICE UNLESS SOMEBODY ASKS
        // FOR IT. `L^2 * 128` floats is 348 MB at 825 residues, read back to a
        // JavaScript array and uploaded again by the structure module and the
        // confidence heads that take it - 0.82 s of a 19.7 s fold for a copy of
        // something already on the GPU. Both take a device tensor now.
        //
        // The one caller that needs the host copy is `web/app.js`, for the
        // distogram contact overlay, and it asks: `pairHost: true`. The same
        // shape as `resumable`.
        const wantsPairHost = recycleOptions.pairHost === true;
        const pairReadback = wantsPairHost
          ? execution.createReadback(
            `monomer.pair-readback-${recycle}`, embedding.pairWithoutTemplates, readbackEncoder)
          : undefined;
        await submit(readbackEncoder, `readback recycle ${recycle}`);
        const [msaFirstRow, pair] = await withAbort(Promise.all([
          execution.mapFloat32(msaFirstRowTensor),
          pairReadback === undefined ? undefined : execution.mapFloat32(pairReadback),
        ]), signal);
        throwIfAborted(signal);
        releaseTensor(msaFirstRowTensor); releaseTensor(msaMask);
        if (pairReadback !== undefined) releaseTensor(pairReadback);

        stageMilliseconds.trunkReadback += performance.now() - phaseStart;
        phaseStart = performance.now();
        const structure = await withAbort(new StructureModuleGpu(this.device).run({
          msaFirstRow, pair: embedding.pairWithoutTemplates,
          mask: features.seqMask, aatype: features.aatype,
          atom37ToAtom14: features.atom37ToAtom14, atom37Mask: features.atom37Mask,
          length, weights: weights.structure, geometry: weights.geometry,
          signal,
          onStep: step,
        }), signal);
        throwIfAborted(signal);
        stageMilliseconds.structure += performance.now() - phaseStart;
        phaseStart = performance.now();
        const confidence = await withAbort(new ConfidenceHeadsGpu(this.device).run(
          structure.finalRepresentation, embedding.pairWithoutTemplates,
          length, weights.lddt, weights.pae, paeBreaks,
          () => step(CONFIDENCE_STEP), signal, recycleOptions.chainLengths,
        ), signal);
        throwIfAborted(signal);
        stageMilliseconds.confidence += performance.now() - phaseStart;
        phaseStart = performance.now();
        const recycleDistance = recycleConvergenceDistance(
          previousAtom37, structure.atom37, features.seqMask,
        );
        const recycleResult = { msaFirstRow, pair, structure, confidence,
          recycleDistance,
          elapsedMilliseconds: performance.now() - recycleStart };
        results.push(recycleResult);
        onRecycle?.(recycleResult, recycle);
        throwIfAborted(signal);
        if (shouldStopAfterRecycle(recycle, recycleDistance, tolerance)) break;
        previousMsa = embedding.msa;
        previousPair = embedding.pairWithoutTemplates;
        previousPositions = execution.upload(`monomer.recycle-positions-${recycle}`, structure.atom37);
        previousAtom37 = structure.atom37;
      }
      // The state the next continuation needs. Read back BEFORE the finally
      // releases the allocator, and only these two: atom37 is already on the
      // CPU, and previousPositions is re-uploaded from it.
      //
      // 🔴 AND IT IS 781 MB AT 825 RESIDUES THAT MOST FOLDS NEVER READ. The MSA
      // and the pair representation come back to the host so `web/app.js` can
      // CONTINUE a fold at more recycles - one cache, one caller - and every
      // other fold pays a device-to-host copy of both plus the JavaScript arrays
      // to hold them. At 0 recycles that is the whole of the cost and none of
      // the benefit. So it is a thunk: the buffers stay alive until the
      // execution is released, and `resume()` is what copies them.
      const stateStart = performance.now();
      let resumable = { atom37: previousAtom37, recycles: firstRecycle + results.length - 1 };
      if (recycleOptions.resumable === true) {
        const stateEncoder = encode("recycle-state");
        const msaReadback = execution.createReadback("state.msa", previousMsa, stateEncoder);
        const pairReadback = execution.createReadback("state.pair", previousPair, stateEncoder);
        await submit(stateEncoder, "recycle state readback");
        resumable = {
          ...resumable,
          msa: await execution.mapFloat32(msaReadback),
          pair: await execution.mapFloat32(pairReadback),
        };
      }
      stageMilliseconds.resumable = performance.now() - stateStart;
      return {
        recycles: results, final: results[results.length - 1], resumable,
        stageMilliseconds,
        elapsedMilliseconds: performance.now() - start,
      };
    } finally {
      execution.release();
    }
  }
}
