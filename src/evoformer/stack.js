import {
  encodeEvoformerPairBlock,
  encodeExtraMsaBlock,
  encodeEvoformerBlock,
} from "./block.js";
import { WebGpuExecution } from "../runtime/execution.js";
import { isAbortError, predictionAbortError, throwIfAborted, withAbort } from "../runtime/abort.js";
import { DeferredValidation } from "../runtime/validation.js";

/**
 * The pair track, wherever it already is.
 *
 * A STACK MAY BE HANDED THE DEVICE BUFFER instead of a Float32Array. The pair
 * representation is the biggest thing in the model - L*L*128 floats, 25 MiB at
 * 221 residues - and the trunk passes it from the input embedder through the
 * extra-MSA stack into the main stack, every one of which used to read it back
 * to the host and push it up again. Both stacks mutate it in place, so when the
 * caller keeps it resident they are all writing to one buffer and the tensor
 * never moves at all.
 *
 * The buffer is the CALLER'S in that case: it was allocated before this stack
 * was entered and it is not released with the stack's own scratch.
 */
function residentPair(execution, label, input, pairElements) {
  // 🔴 KEEPING THE PAIR ONLY MEANS ANYTHING IF SOMETHING ELSE OWNS IT. A stack
  // that made its own execution releases everything in it on the way out, so a
  // tensor "kept" from one would be handed back already freed - and pooling
  // would give that same memory to the next caller. Refusing the combination is
  // the difference between an error here and silent corruption downstream.
  if (input.keepPair === true && input.execution === undefined) {
    throw new RangeError("keepPair requires a shared execution: this stack releases its own on exit");
  }
  if (input.pairTensor !== undefined) {
    if (input.pairTensor.elements !== pairElements) {
      throw new RangeError(`resident pair tensor has ${input.pairTensor.elements} elements,`
        + ` and this stack wants ${pairElements}`);
    }
    return input.pairTensor;
  }
  if (input.pair.length !== pairElements) throw new RangeError("stack pair shape mismatch");
  return execution.upload(label, input.pair, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
}

/**
 * Turn the chain masks from Float32Arrays into tensors, once per stack.
 *
 * The two masks are the SAME [L, L] buffer wherever both are on - one gates the
 * outer product mean's covariance, the other gates row attention across chains -
 * so this uploads it once and hands the same tensor to both. A stack that wants
 * neither uploads nothing.
 */
function uploadChainMasks(execution, input, label) {
  const wanted = [input.covMask, input.rowAttentionChainMask].filter((mask) => mask !== undefined);
  if (wanted.length === 0) return { covMask: undefined, rowAttentionChainMask: undefined };
  const shared = wanted.every((mask) => mask === wanted[0]);
  const upload = (name, values) => execution.upload(`${label}.${name}`, values);
  if (shared) {
    const tensor = upload("chain-mask", wanted[0]);
    return {
      covMask: input.covMask === undefined ? undefined : tensor,
      rowAttentionChainMask: input.rowAttentionChainMask === undefined ? undefined : tensor,
    };
  }
  return {
    covMask: input.covMask === undefined ? undefined : upload("cov-mask", input.covMask),
    rowAttentionChainMask: input.rowAttentionChainMask === undefined
      ? undefined : upload("row-chain-mask", input.rowAttentionChainMask),
  };
}

export class EvoformerStackGpu {
  device;

  constructor(device) { this.device = device; }

  async run(input) {
    throwIfAborted(input.signal);
    if (input.blockWeights.length === 0) throw new RangeError("Evoformer stack requires at least one block");
    const execution = input.execution ?? new WebGpuExecution(this.device);
    const ownsExecution = input.execution === undefined;
    const entry = execution.checkpoint();
    try {
      const msaElements = input.sequences * input.length * input.cM;
      const pairElements = input.length * input.length * input.cZ;
      if (input.msa.length !== msaElements) throw new RangeError("Evoformer stack activation shape mismatch");
      const pair = residentPair(execution, "stack.pair", input, pairElements);
      const msa = execution.upload("stack.msa", input.msa, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const msaMask = execution.upload("stack.msa-mask", input.msaMask);
      const pairMask = execution.upload("stack.pair-mask", input.pairMask);
      const persistentCheckpoint = execution.checkpoint();
      const start = performance.now();
      let timestampProfile;
      const requestedWindow = input.submissionWindow ?? (input.signal !== undefined ? 8 : input.blockWeights.length);
      if (!Number.isSafeInteger(requestedWindow) || requestedWindow < 1) {
        throw new RangeError("submissionWindow must be a positive safe integer");
      }
      const submissionWindow = input.profileBlock === undefined ? requestedWindow : 1;
      const validation = new DeferredValidation(this.device, "Evoformer stack");
      // ...a Float32Array on the way in, a tensor from here on, uploaded once
      // rather than per block. Absent for a monomer, which keeps the plain path.
      // 🔴 BOTH MASKS HAVE TO BECOME TENSORS, not just the one. They arrive as
      // Float32Arrays and are bound as buffers, so a lane left un-uploaded
      // reaches the bind group as a plain array and fails on `.buffer` - which
      // is exactly how this shipped: the shaders compiled, and the fold died at
      // runtime with "Cannot read properties of undefined".
      const uploadedChainMask = uploadChainMasks(execution, input, "stack");

      for (let block = 0; block < input.blockWeights.length; block += 1) {
        throwIfAborted(input.signal);
        const encoder = this.device.createCommandEncoder({ label: `evoformer-stack.block-${block}` });
        const profiling = input.profileBlock === block;
        if (profiling) execution.beginTimestampProfile();
        validation.begin();
        await encodeEvoformerBlock(execution, encoder, {
          ...input,
          ...uploadedChainMask,
          weights: input.blockWeights[block],
        }, msa, pair, msaMask, pairMask);
        execution.endComputePass(encoder);
        const pendingProfile = profiling ? execution.finishTimestampProfile(encoder) : undefined;
        this.device.queue.submit([encoder.finish()]);
        validation.end(`block ${block}`);
        const endOfWindow = (block + 1) % submissionWindow === 0 || block + 1 === input.blockWeights.length;
        if (pendingProfile !== undefined) {
          await this.device.queue.onSubmittedWorkDone();
          timestampProfile = await execution.readTimestampProfile(pendingProfile);
          execution.releaseSince(persistentCheckpoint);
        } else {
          // Pooling makes these buffers available to the next encoded block.
          // Queue ordering ensures its commands execute only after this block.
          execution.releaseSince(persistentCheckpoint);
          if (endOfWindow) await withAbort(this.device.queue.onSubmittedWorkDone(), input.signal);
        }
        throwIfAborted(input.signal);
        // 🔴 WHEN THE DEVICE REACHES THIS BLOCK, reported without waiting for it.
        //
        // queue.onSubmittedWorkDone() resolves once everything submitted so far
        // has finished, so one taken here - and NOT awaited - settles exactly
        // when the GPU has got through this block. The loop carries straight on
        // encoding, so nothing is serialised and the pipelining that makes this
        // stack fast is untouched. What arrives is a stream of real completion
        // events instead of a stream of "queued" events that all fire at once.
        //
        // They resolve in submission order, so the count cannot go backwards.
        const submitted = block + 1;
        const total = input.blockWeights.length;
        void this.device.queue.onSubmittedWorkDone()
          .then(() => input.onBlockDone?.(submitted, total));
      }

      // 🔴 THE BLOCKS ABOVE WERE ENCODED, NOT COMPUTED. Submission runs ahead of
      // the device on purpose - it is what takes this stack from ~21.7s to
      // ~2.3s - so onBlock fires once a block is queued and validated, and the
      // GPU is still working through them. THIS is where that work is waited
      // on, and at any real length it is the bulk of the wall time. Reporting
      // it as a phase rather than a step is the honest shape: the caller can
      // show that something long is happening without pretending to know how
      // far through it is.
      input.onStage?.("gpu");
      const encoder = this.device.createCommandEncoder({ label: "evoformer-stack.readback" });
      const msaReadback = execution.createReadback("stack.msa-readback", msa, encoder);
      const pairReadback = input.keepPair === true
        ? undefined : execution.createReadback("stack.pair-readback", pair, encoder);
      this.device.queue.submit([encoder.finish()]);
      await validation.settle();
      const msaOutput = await withAbort(execution.mapFloat32(msaReadback), input.signal);
      const pairOutput = pairReadback === undefined ? undefined : await withAbort(execution.mapFloat32(pairReadback), input.signal);
      throwIfAborted(input.signal);
      input.onStage?.("done");
      return {
        msa: msaOutput,
        pair: pairOutput,
        pairTensor: pair,
        elapsedMilliseconds: performance.now() - start,
        memory: execution.snapshot(),
        blocks: input.blockWeights.length,
        ...(timestampProfile === undefined ? {} : { timestampProfile }),
      };
    } finally {
      // A SHARED EXECUTION OUTLIVES THIS STACK, and so does whatever the caller
      // allocated in it before we were entered - the pair track, above all.
      // releaseSince only reaches what this call added.
      if (ownsExecution) execution.release(); else execution.releaseSince(entry);
    }
  }
}

export class ExtraMsaPairStackGpu {
  device;

  constructor(device) { this.device = device; }

  async run(input)

  {
    throwIfAborted(input.signal);
    const execution = input.execution ?? new WebGpuExecution(this.device);
    const ownsExecution = input.execution === undefined;
    const entry = execution.checkpoint();
    try {
      const pair = residentPair(
        execution, "extra-stack.pair", input, input.length * input.length * input.cZ,
      );
      const msa = execution.upload("extra-stack.msa", input.msa);
      const msaMask = execution.upload("extra-stack.msa-mask", input.msaMask);
      const pairMask = execution.upload("extra-stack.pair-mask", input.pairMask);
      const persistentCheckpoint = execution.checkpoint();
      const start = performance.now();
      const validation = new DeferredValidation(this.device, "extra-MSA pair stack");
      const uploadedChainMask = uploadChainMasks(execution, input, "extra-pair-stack");
      for (let block = 0; block < input.blockWeights.length; block += 1) {
        throwIfAborted(input.signal);
        const encoder = this.device.createCommandEncoder({ label: `extra-msa-pair-stack.block-${block}` });
        validation.begin();
        await encodeEvoformerPairBlock(
          execution, encoder, { ...input, ...uploadedChainMask }, input.blockWeights[block], msa, pair, msaMask, pairMask,
        );
        execution.endComputePass(encoder);
        this.device.queue.submit([encoder.finish()]);
        validation.end(`block ${block}`);
        throwIfAborted(input.signal);
        // Commands are queue ordered, so the next block may alias these
        // pooled scratch buffers without a host-side wait.
        execution.releaseSince(persistentCheckpoint);
        const endOfWindow = block + 1 === input.blockWeights.length;
        if (endOfWindow) await withAbort(this.device.queue.onSubmittedWorkDone(), input.signal);
        // 🔴 WHEN THE DEVICE REACHES THIS BLOCK, reported without waiting for it.
        //
        // queue.onSubmittedWorkDone() resolves once everything submitted so far
        // has finished, so one taken here - and NOT awaited - settles exactly
        // when the GPU has got through this block. The loop carries straight on
        // encoding, so nothing is serialised and the pipelining that makes this
        // stack fast is untouched. What arrives is a stream of real completion
        // events instead of a stream of "queued" events that all fire at once.
        //
        // They resolve in submission order, so the count cannot go backwards.
        const submitted = block + 1;
        const total = input.blockWeights.length;
        void this.device.queue.onSubmittedWorkDone()
          .then(() => input.onBlockDone?.(submitted, total));
      }
      // 🔴 NOTHING IS WAITED FOR WHEN THE PAIR STAYS PUT. The readback below is
      // the only reason this stack ever drained the pipeline, so keeping the
      // pair on the device does not merely save 25 MiB of traffic at 221
      // residues - it removes a full host-device round trip from the middle of
      // the trunk. The main stack's commands queue straight in behind these.
      if (input.keepPair === true) {
        await validation.settle();
        throwIfAborted(input.signal);
        return {
          pair: undefined, pairTensor: pair,
          elapsedMilliseconds: performance.now() - start, memory: execution.snapshot(),
        };
      }
      input.onStage?.("gpu");
      const encoder = this.device.createCommandEncoder({ label: "extra-msa-pair-stack.readback" });
      const readback = execution.createReadback("extra-stack.pair-readback", pair, encoder);
      this.device.queue.submit([encoder.finish()]);
      await validation.settle();
      const output = await withAbort(execution.mapFloat32(readback), input.signal);
      throwIfAborted(input.signal);
      input.onStage?.("done");
      return {
        pair: output, pairTensor: pair,
        elapsedMilliseconds: performance.now() - start, memory: execution.snapshot(),
      };
    } finally {
      if (ownsExecution) execution.release(); else execution.releaseSince(entry);
    }
  }
}

export class ExtraMsaStackGpu {
  device;
  constructor(device) { this.device = device; }
  async run(input)

  {
    throwIfAborted(input.signal);
    const execution = new WebGpuExecution(this.device);
    try {
      const msa = execution.upload("extra-full-stack.msa", input.msa, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const pair = execution.upload("extra-full-stack.pair", input.pair, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const msaMask = execution.upload("extra-full-stack.msa-mask", input.msaMask);
      const pairMask = execution.upload("extra-full-stack.pair-mask", input.pairMask);
      const persistentCheckpoint = execution.checkpoint();
      const start = performance.now();
      const validation = new DeferredValidation(this.device, "extra-MSA stack");
      for (let block = 0; block < input.blockWeights.length; block += 1) {
        throwIfAborted(input.signal);
        const encoder = this.device.createCommandEncoder({ label: `extra-msa-stack.block-${block}` });
        validation.begin();
        await encodeExtraMsaBlock(execution, encoder, input, input.blockWeights[block], msa, pair, msaMask, pairMask);
        execution.endComputePass(encoder);
        this.device.queue.submit([encoder.finish()]);
        validation.end(`block ${block}`);
        throwIfAborted(input.signal);
        execution.releaseSince(persistentCheckpoint);
      }
      await withAbort(this.device.queue.onSubmittedWorkDone(), input.signal);
      await validation.settle();
      const encoder = this.device.createCommandEncoder({ label: "extra-full-stack.readback" });
      const msaReadback = execution.createReadback("extra-full-stack.msa-readback", msa, encoder);
      const pairReadback = execution.createReadback("extra-full-stack.pair-readback", pair, encoder);
      this.device.queue.submit([encoder.finish()]);
      const [msaOutput, pairOutput] = await withAbort(Promise.all([
        execution.mapFloat32(msaReadback), execution.mapFloat32(pairReadback),
      ]), input.signal);
      throwIfAborted(input.signal);
      return { msa: msaOutput, pair: pairOutput, elapsedMilliseconds: performance.now() - start,
        memory: execution.snapshot() };
    } finally { execution.release(); }
  }
}
