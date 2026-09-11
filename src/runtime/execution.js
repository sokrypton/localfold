import { GpuBufferAllocator } from "./allocator.js";
import { DeviceWeightRefusal, packedBytesOf, residentPackFromSources }
  from "./device-pack.js";
import { residentPack } from "./resident.js";
import {
  GpuMemoryBudgetError, memoryBudgetBytes, memoryTotals, noteResidencyRefused,
  residencyAllowed,
} from "./device-memory.js";
import { deviceTuning } from "./device-profile.js";
import { pipelineCacheForDevice } from "./pipeline-cache.js";
import { storageBytes, storageWords } from "./storage.js";
import { shaderSource } from "./shader-source-cache.js";

const GRID_WIDTH = 32_768;
const MAX_WORKGROUPS_PER_DIMENSION = 65_535;
/**
 * 🔴 EXPORTED SO A PROBE CAN DRIVE THE SHIPPED TEXT AND NOT A COPY OF IT, which
 * is check-quantised-upload.js's rule. See tools/gpu/probe-grid-overdispatch.js:
 * this kernel indexes a FOLDED grid, `linearGrid` rounds the dispatch up to
 * whole workgroups and whole rows of y, and what the out-of-range invocations
 * then do is a property of the BACKEND rather than of this repository.
 */
export const ADD_IN_PLACE_SHADER = `
const GRID_WIDTH: u32 = 32768u;
@group(0) @binding(0) var<storage, read_write> base: array<f32>;
@group(0) @binding(1) var<storage, read> update: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  base[index] += update[index];
}`;

export class WebGpuExecution {
  device;
  allocator;
  pipelines;
  #allocations = [];
  // 🔴 WHICH OF THEM CAME FROM writeBuffer, WHICH IS THE ONE THING THAT CANNOT
  // BE RECYCLED INSIDE AN ENCODER. See releaseScratchSince.
  #uploaded = new WeakSet();
  #timestamps;
  #activeEncoder;
  #activePass;

  /**
   * @param {GPUDevice} device
   * @param {{transitionBufferLimit?: number}} [options] a smaller binding limit
   *   than the device's, which is how the chunked transition path is exercised
   *   on hardware whose real limit is large enough never to need it.
   */
  constructor(device, options = {}) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device, true);
    this.pipelines = pipelineCacheForDevice(device);
    /**
     * True while an encode path is being run for its PIPELINES only.
     *
     * 🔴 95 COMPILES, ASKED FOR ONE AT A TIME, ARE AF2's WHOLE FIRST FOLD. The
     * span from the first `createComputePipelineAsync` to the last settle is
     * 1133 ms of a 1163 ms first fold, and the sum of their individual waits is
     * 1662 - so an average of 1.47 were ever in flight, because an encode asks
     * for a pipeline at the moment it needs it and the block's ten operations
     * run one after another. This browser compiles 32 pipelines 5.4x faster
     * concurrently than serially, measured, so the fix is to ask for all of
     * them before encoding anything.
     *
     * Every encode function returns early once its pipelines are requested when
     * this is set. Nothing below that point in any of them touches the encoder,
     * and everything above is derivation plus the resident weight upload, which
     * is cached and wanted either way.
     */
    this.warming = false;
    this.pendingWarm = [];
    // 🔴 A TENSOR CAN FIT IN A BUFFER AND STILL NOT BE BINDABLE. maxBufferSize
    // and maxStorageBufferBindingSize are different limits, and the second is
    // the smaller one - so a transition over 508 MSA rows of a long sequence
    // allocates fine and then fails to bind. This is the number the transition
    // chunks against; it can be lowered but never raised past the device's.
    this.transitionBufferLimit = Math.min(
      device.limits.maxStorageBufferBindingSize,
      options.transitionBufferLimit ?? device.limits.maxStorageBufferBindingSize,
    );
    if (!Number.isSafeInteger(this.transitionBufferLimit) || this.transitionBufferLimit <= 0) {
      throw new RangeError("transitionBufferLimit must be a positive safe integer");
    }
  }

  /**
   * A window onto part of a tensor, bound as its own range.
   *
   * The chunked transition works on row windows of one big allocation rather
   * than on many small ones, so what changes per chunk is the BINDING, not the
   * buffer. Offsets compose, so a view of a view is measured from the original.
   */
  view(tensor, offsetElements, elements) {
    if (!Number.isSafeInteger(offsetElements) || !Number.isSafeInteger(elements)
      || offsetElements < 0 || elements <= 0 || offsetElements + elements > tensor.elements) {
      throw new RangeError(`invalid GPU tensor view ${offsetElements}:${elements} of ${tensor.elements}`);
    }
    const storage = tensor.storage ?? "f32";
    const offset = (tensor.offsetElements ?? 0) + offsetElements;
    // 🔴 A PACKED VIEW HAS TO START ON A WORD. Two elements share one, so an
    // odd offset would put the view's first element in the HIGH half of the
    // word the binding starts at, and every index inside it would be off by
    // one - silently, since the shapes still agree.
    if (storage === "f16" && offset % 2 !== 0) {
      throw new RangeError(`a packed tensor view must start on an even element; got ${offset}`);
    }
    return { allocation: tensor.allocation, elements, offsetElements: offset, storage };
  }

  upload(label, data, usage = GPUBufferUsage.STORAGE) {
    const allocation = this.allocator.upload(label, data, usage);
    this.#allocations.push(allocation);
    this.#uploaded.add(allocation);
    return { allocation, elements: data.byteLength / 4, storage: "f32" };
  }

  /**
   * @param {number} elements how many VALUES the tensor holds, whatever the
   *   storage. A packed tensor is half the bytes and the same shape, so every
   *   caller's arithmetic is unchanged and only the allocation shrinks.
   * @param {"f32"|"f16"} [storage] see src/runtime/storage.js
   */
  allocate(label, elements, usage = GPUBufferUsage.STORAGE, storage = "f32") {
    const allocation = this.allocator.allocate(label, storageBytes(elements, storage), usage);
    this.#allocations.push(allocation);
    return { allocation, elements, storage };
  }

  /**
   * The same upload, kept on the device for the model's lifetime.
   *
   * 🔴 THIS IS THE REVISIT THE NOTE BELOW ASKS FOR, and it is a different
   * machine's answer. That measurement was taken on an M2 - shared memory,
   * writeBuffer close to free, a GPU-bound fold - and it concluded "memory
   * spent for no time saved". On a discrete GPU the same traffic crosses PCIe,
   * and more to the point the PACK is host work that no amount of shared
   * memory makes free: 232 ms for 24 ESMFold2 blocks, measured, and that
   * trunk's wall went 701 -> 501 ms when it stopped paying it per pass.
   *
   * `pack` runs only on a MISS, so the packed array is transient on the first
   * pass and never allocated again. `elements` comes off the buffer rather
   * than off `pack`, because on a hit there is no array to ask - and the
   * OFFSETS are kept beside it, because every caller needs those on every pass
   * and fetching them would otherwise run the pack anyway.
   */
  /**
   * `uploadResident`, but decoding the tensors on the DEVICE when the caller
   * can say which they are.
   *
   * 🔴 THE ONLY REASON IT IS ASYNC. Filling a buffer from codes means a compute
   * pass, and `createComputePipelineAsync` is a promise; every AF2 caller of
   * this is already inside an `async` encode function, so the cost is a
   * microtask and not a redesign.
   *
   * @param {{sources: object, order: string[], precision: string}} plan
   */
  async uploadResidentPacked(label, key, pack, variant, plan) {
    // 🔴 A BUDGET DOES NOT DISQUALIFY THIS ONE, WHICH IS THE DIFFERENCE FROM
    // `uploadResident` BELOW. That one declines under a ceiling because the
    // POOLED allocator evicts to make room and can evict a buffer an in-flight
    // submit still names; this allocates through `noteAllocation`, which RAISES
    // before `createBuffer` and never evicts anything. The page ALWAYS sets a
    // budget, so gating on its absence gave the whole device path to the tools
    // and none of it to the shipped page - measured as a 2.5 s page fold
    // becoming 3.0.
    //
    // 🔴 AND A QUARTER OF THE CEILING IS THE LINE, the same rule
    // src/esmc/tower-webgpu.js takes. A device with room keeps everything; a
    // 200 MiB one keeps nothing and streams as it always did, because filling a
    // tight ceiling with resident weights is exactly what makes the pooled
    // allocator evict something in flight - `--budget=200` reproduced that in
    // one run without this line.
    const ceiling = memoryBudgetBytes(this.device);
    const wanted = plan?.order === undefined ? undefined
      : packedBytesOf(plan.order, plan.sources, plan.precision);
    const affordable = ceiling === undefined || (wanted !== undefined
      && memoryTotals(this.device).residentBytes + wanted <= ceiling / 4);
    if (plan?.order !== undefined && residencyAllowed(this.device) && affordable) {
      try {
        const built = await residentPackFromSources(this.device, {
          key, label, variant, sources: plan.sources, order: plan.order,
          precision: plan.precision,
        });
        return {
          weights: { allocation: { buffer: built.buffer },
                     elements: built.buffer.size / 4, storage: "f32" },
          // 🔴 THE OFFSET TABLE'S SHAPE IS THE HOST PACKER'S. AF2's transition
          // and attention packers return a positional array and the triangle's
          // returns an object keyed by name, because that is what its shader
          // factory indexes; a device path that returned the other one hands a
          // kernel `undefined` for every offset, which WGSL then compiles as
          // the string "undefined" and fails to parse. `named` says which.
          offsets: plan.named === true ? built.offsetsByName : built.offsets,
        };
      } catch (error) {
        if (error instanceof GpuMemoryBudgetError) {
          noteResidencyRefused(this.device);
        } else if (error instanceof DeviceWeightRefusal) {
          // 🔴 THE FALLBACK IS OPT-IN, because a silent one is a bug that looks
          // like a slow machine. A bundle that genuinely cannot be decoded on
          // the device - float32, or a fixture built over plain arrays - says
          // so once by setting `allowHostWeightPacking`; anything else is a
          // descriptor that lost its sources and should stop the fold.
          if (deviceTuning(this.device).allowHostWeightPacking !== true) throw error;
        } else {
          throw error;
        }
      }
    }
    return this.uploadResident(label, key, pack, variant);
  }

  /**
   * Run `body` for its pipelines and its resident weights, encoding nothing.
   *
   * 🔴 IT IS A MODE AND NOT A SEPARATE LIST OF KERNELS, which is the whole
   * point: a warm written as its own enumeration of shader keys is a list
   * written twice, and the copy that goes stale is the one nobody runs. This
   * drives the SAME encode functions the fold does, so a kernel that exists is
   * a kernel that gets warmed.
   *
   * Not re-entrant, and it must not overlap real encoding - the flag is on the
   * execution, so a fold that started warming halfway through would silently
   * skip the rest of its dispatches. Callers warm, then fold.
   */
  async warm(body) {
    if (this.warming) throw new Error("Execution.warm is already running");
    const checkpoint = this.checkpoint();
    this.warming = true;
    this.pendingWarm = [];
    try {
      await body();
      // ...drained in rounds, because a warmed operation may register more.
      while (this.pendingWarm.length > 0) {
        const batch = this.pendingWarm;
        this.pendingWarm = [];
        await Promise.all(batch);
      }
    } finally {
      this.warming = false;
      this.pendingWarm = [];
      this.releaseScratchSince(checkpoint);
    }
  }

  /**
   * Register a warm that is running, so `warm` can await it at the end.
   *
   * 🔴 THIS IS WHERE THE CONCURRENCY COMES FROM. `staged` awaits its body in a
   * fold because the encoder is ordered; in a warm there is no encoder, so it
   * hands the promise here and returns, and the block's ten operations request
   * their pipelines in the same tick instead of ten waits apart.
   */
  notePending(promise) {
    if (!this.warming) throw new Error("notePending outside a warm");
    const tracked = Promise.resolve(promise);
    // ...so a rejection that the drain has not reached yet is not an unhandled
    // one; the drain still sees it, because `tracked` is what is pushed.
    tracked.catch(() => {});
    this.pendingWarm.push(tracked);
  }

  uploadResident(label, key, pack, variant = "") {
    // 🔴 RESIDENCY IS A TRADE AND THE BUDGET ANSWERS IT. Keeping an evoformer
    // stack's weights costs 280 MiB of a 387 MiB fold at 59 residues; on this
    // card that is nothing and on a phone it is the difference between folding
    // and not. A device with a budget raises GpuMemoryBudgetError from the
    // first allocation that would cross it, and from then on this whole device
    // uploads per pass instead - which is the same policy AF3's pairformer
    // takes, expressed once here so every AF2 caller inherits it.
    // 🔴 AND A DEVICE WITH A CEILING DOES NOT TAKE IT AT ALL, which is stricter
    // than AF3's pairformer and has to be. That stack catches
    // GpuMemoryBudgetError and RESTARTS itself without residency; AF2's has no
    // restart, and under a 200 MiB budget the allocator instead evicts pooled
    // buffers to make room for the 280 MiB of resident weights - and evicts one
    // that an in-flight command buffer still names: "[Buffer
    // msa-row-attention.value] used in submit while destroyed". Measured. With
    // residency declined, the same fold under the same budget is 108.5 MiB and
    // the same answer to the checksum.
    if (residencyAllowed(this.device) && memoryBudgetBytes(this.device) === undefined) {
      try {
        const { buffer, offsets } = residentPack(this.device, key, label, pack, variant);
        return {
          weights: { allocation: { buffer }, elements: buffer.size / 4, storage: "f32" },
          offsets,
        };
      } catch (error) {
        if (!(error instanceof GpuMemoryBudgetError)) throw error;
        noteResidencyRefused(this.device);
      }
    }
    const packed = pack();
    return { weights: this.upload(label, packed.data), offsets: packed.offsets };
  }

  // 🔴 WEIGHTS ARE NOT CACHED ACROSS PASSES, and that was measured, not assumed.
  //
  // A block packs its parameters into one Float32Array and uploads them, and
  // the stack then releases the block's allocations so the next block can alias
  // the scratch - so a four-pass fold re-packs and re-uploads the 345 MiB of
  // Evoformer weights four times. Holding them instead is easy (key the upload
  // on the block, keep it outside #allocations so releaseSince cannot reach it)
  // and it demonstrably works: 456 uploads on the first pass, 1368 cache hits
  // over the next three, and the 221 ms of packing paid once instead of four
  // times.
  //
  // It bought NOTHING. A 59-residue four-pass fold measured 4.3 s with the
  // cache and 4.3 s without it. The fold is GPU-bound and the uploads were
  // already overlapping with compute - and this is Apple Silicon, where host
  // and device share memory, so writeBuffer is close to free. What the cache
  // did cost was 345 MiB of GPU memory resident for the length of a predict,
  // against roughly 7 MiB when each block's weights are transient.
  //
  // Worth revisiting on a discrete GPU, where the same traffic crosses PCIe.
  // Not worth carrying here: memory spent for no time saved.

  /**
   * One workgroup per ROW, folded across two dimensions.
   *
   * A dispatch may be 65535 workgroups wide at most, and a pair track has L*L
   * rows - which passes that at L=256. Shaders reached this way read their row
   * as `group.x + group.y * GRID_WIDTH`.
   */
  rowGrid(rows) {
    return [Math.min(rows, GRID_WIDTH), Math.ceil(rows / GRID_WIDTH)];
  }

  linearGrid(elements, workgroupSize = 64) {
    const groups = Math.ceil(elements / workgroupSize);
    return [Math.min(groups, GRID_WIDTH), Math.ceil(groups / GRID_WIDTH)];
  }

  dispatch(
    encoder,
    pipeline,
    tensors,
    x,
    y = 1,
    z = 1,
    label,
  ) {
    // A DISPATCH IS AT MOST 65535 WORKGROUPS PER DIMENSION, and going over is a
    // validation error naming a count and nothing else. Every grid here should
    // have been folded by linearGrid or rowGrid; saying which one was not, and
    // at what size, is the difference between a five-minute fix and a hunt.
    // 🔴 ALL THREE, NOT JUST x. This checked x alone, and the one that
    // overflowed in the field was Y: a 1566-residue AF3 fold refused with
    // "Dispatch workgroup count Y (76637) exceeds max compute workgroups per
    // dimension (65535)" from inside the template embedder, which is the raw
    // browser message this guard exists to replace. A limit worth naming is
    // worth naming on every axis that has it.
    for (const [axis, count] of [["x", x], ["y", y], ["z", z]]) {
      if (count > MAX_WORKGROUPS_PER_DIMENSION) {
        throw new RangeError(`${label ?? "dispatch"} needs ${count} workgroups in ${axis}, over`
          + ` the ${MAX_WORKGROUPS_PER_DIMENSION} limit - it wants folding through linearGrid`
          + " or rowGrid");
      }
    }
    const timestamp = this.#timestamps;
    let timestampWrites;
    if (timestamp !== undefined) {
      if (timestamp.nextQuery + 2 > timestamp.querySet.count) {
        throw new RangeError("GPU timestamp query capacity exceeded");
      }
      timestamp.labels.push(label ?? `dispatch-${timestamp.labels.length}`);
      timestampWrites = {
        querySet: timestamp.querySet,
        beginningOfPassWriteIndex: timestamp.nextQuery,
        endOfPassWriteIndex: timestamp.nextQuery + 1,
      };
      timestamp.nextQuery += 2;
    }
    let pass;
    const reusable = timestampWrites === undefined;
    if (reusable) {
      if (this.#activeEncoder !== encoder || this.#activePass === undefined) {
        this.endComputePass();
        this.#activeEncoder = encoder;
        this.#activePass = encoder.beginComputePass({ label: "localfold.compute" });
      }
      pass = this.#activePass;
      if (label !== undefined) pass.pushDebugGroup(label);
    } else {
      pass = encoder.beginComputePass({
        ...(label === undefined ? {} : { label }),
        timestampWrites: timestampWrites,
      });
    }
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      // ...THE TENSOR'S RANGE, not the whole buffer. Without an explicit offset
      // and size a view would bind everything behind it and the shader would
      // index from the wrong place.
      entries: tensors.map((tensor, binding) => {
        const offset = storageWords(tensor.offsetElements ?? 0, tensor.storage ?? "f32") * 4;
        // 🔴 A BOUND RANGE MUST START ON 256 BYTES, AND WEBGPU'S OWN MESSAGE
        // NAMES THE WRONG TENSOR. The allocator pools buffers by size and a
        // pooled buffer keeps the LABEL it was created with, so the validation
        // error reads "Offset (771968) of [Buffer opm.left]" for a binding that
        // has nothing to do with the outer product mean - which is most of an
        // afternoon. An odd MSA depth used to produce exactly that, from
        // block.js's OPM view; see the note on `residueMultiple` there.
        if (offset % 256 !== 0) {
          throw new RangeError(`${label ?? "a compute pass"} binding ${binding} starts at `
            + `byte ${offset}, which is not a multiple of 256: a view's element offset `
            + `must land on a 256-byte boundary. Pad the stride the view steps by.`);
        }
        return {
          binding,
          resource: {
            buffer: tensor.allocation.buffer,
            offset,
            size: storageBytes(tensor.elements, tensor.storage ?? "f32"),
          },
        };
      }),
    }));
    pass.dispatchWorkgroups(x, y, z);
    if (reusable) {
      if (label !== undefined) pass.popDebugGroup();
    } else {
      pass.end();
    }
  }

  endComputePass(encoder) {
    if (encoder !== undefined && this.#activeEncoder !== undefined && this.#activeEncoder !== encoder) {
      throw new Error("attempted to end a compute pass with a different command encoder");
    }
    this.#activePass?.end();
    this.#activePass = undefined;
    this.#activeEncoder = undefined;
  }

  async addInPlace(encoder, base, update, label) {
    if (base.elements !== update.elements) throw new RangeError("residual tensors must have equal sizes");
    const pipeline = await this.pipelines.get("runtime:add-in-place", ADD_IN_PLACE_SHADER);
    const grid = this.linearGrid(base.elements);
    this.dispatch(encoder, pipeline, [base, update], grid[0], grid[1], 1, label);
  }

  createReadback(label, tensor, encoder) {
    this.endComputePass(encoder);
    const readback = this.allocate(label, tensor.elements, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    encoder.copyBufferToBuffer(
      tensor.allocation.buffer, (tensor.offsetElements ?? 0) * 4,
      readback.allocation.buffer, 0, tensor.elements * 4,
    );
    return readback;
  }

  async mapFloat32(readback) {
    await readback.allocation.buffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readback.allocation.buffer.getMappedRange().slice(0));
    readback.allocation.buffer.unmap();
    return result;
  }

  snapshot() { return this.allocator.snapshot(); }

  beginTimestampProfile(maxDispatches = 256) {
    if (!this.device.features.has("timestamp-query")) {
      throw new Error("timestamp-query was not requested on this WebGPU device");
    }
    if (this.#timestamps !== undefined) throw new Error("a GPU timestamp profile is already active");
    this.#timestamps = {
      querySet: this.device.createQuerySet({ type: "timestamp", count: maxDispatches * 2 }),
      labels: [],
      nextQuery: 0,
    };
  }

  finishTimestampProfile(encoder) {
    this.endComputePass(encoder);
    const capture = this.#timestamps;
    if (capture === undefined) throw new Error("no GPU timestamp profile is active");
    this.#timestamps = undefined;
    const queryCount = capture.nextQuery;
    const elements = queryCount * 2;
    const resolve = this.allocate(
      "profile.timestamp-resolve", elements, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    );
    const readback = this.allocate(
      "profile.timestamp-readback", elements, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    );
    encoder.resolveQuerySet(capture.querySet, 0, queryCount, resolve.allocation.buffer, 0);
    encoder.copyBufferToBuffer(resolve.allocation.buffer, 0, readback.allocation.buffer, 0, queryCount * 8);
    return { querySet: capture.querySet, labels: capture.labels, readback };
  }

  async readTimestampProfile(pending) {
    try {
      await pending.readback.allocation.buffer.mapAsync(GPUMapMode.READ);
      const values = new BigUint64Array(pending.readback.allocation.buffer.getMappedRange().slice(0));
      pending.readback.allocation.buffer.unmap();
      return pending.labels.map((label, index) => ({
        label,
        nanoseconds: Number(values[index * 2 + 1] - values[index * 2]),
      }));
    } finally {
      pending.querySet.destroy();
    }
  }

  /**
   * A pipeline whose WGSL is generated at most once per device.
   *
   * 🔴 `pipelines.get` TAKES A FINISHED SOURCE AND DISCARDS IT ON A HIT, and a
   * block asks for the same pipelines on every block of every recycle. A
   * 59-residue AF2 fold generated 26.9 MiB of WGSL for 2,795 hits before this
   * existed - see src/runtime/shader-source-cache.js, which also carries the
   * verification that replaces the collision check the memo makes vacuous.
   *
   * The key must name everything the source depends on, which is the same rule
   * the pipeline key already lives under.
   */
  shaderPipeline(key, build, entryPoint = "main") {
    return this.pipelines.get(key, shaderSource(this.device, key, build), entryPoint);
  }

  checkpoint() { return this.#allocations.length; }

  /**
   * Release only the buffers this execution ALLOCATED, keeping the uploaded
   * ones, so a sub-layer's scratch can be recycled without waiting for the
   * encoder to be submitted.
   *
   * 🔴 `queue.writeBuffer` IS ORDERED AGAINST THE QUEUE, NOT THE ENCODER, AND
   * THAT DISTINCTION IS A WRONG ANSWER. A block encodes every sub-layer into
   * ONE command buffer and submits it at the end, but each sub-layer's weight
   * upload goes onto the queue as it is encoded - so ALL of a block's
   * writeBuffers run before ANY of its dispatches. Recycle an uploaded buffer
   * mid-block and the second sub-layer's weights land in it before the first
   * sub-layer's dispatch ever reads it: `fold-af2.js` went checksum -1805925 to
   * -1207195 and pLDDT 57.280 to 56.109, which is a plausible-looking structure
   * and a silently wrong one.
   *
   * Allocated scratch has no such hazard. Nothing writes it but a dispatch, and
   * dispatches in one pass are ordered - so the reader was encoded before the
   * writer that reuses the buffer, and WebGPU keeps them in that order.
   *
   * `releaseSince` stays what a stack calls between blocks, where the previous
   * block's submit already separates the two sets of writeBuffers.
   */
  releaseScratchSince(checkpoint) {
    if (!Number.isSafeInteger(checkpoint) || checkpoint < 0 || checkpoint > this.#allocations.length) {
      throw new RangeError(`invalid GPU allocation checkpoint ${checkpoint}`);
    }
    const kept = [];
    for (let index = this.#allocations.length - 1; index >= checkpoint; index -= 1) {
      const allocation = this.#allocations[index];
      if (this.#uploaded.has(allocation)) kept.push(allocation);
      else allocation.release();
    }
    this.#allocations.length = checkpoint;
    // ...in the order they were made, so a later checkpoint is still a suffix.
    for (let index = kept.length - 1; index >= 0; index -= 1) {
      this.#allocations.push(kept[index]);
    }
  }

  releaseSince(checkpoint) {
    if (!Number.isSafeInteger(checkpoint) || checkpoint < 0 || checkpoint > this.#allocations.length) {
      throw new RangeError(`invalid GPU allocation checkpoint ${checkpoint}`);
    }
    for (let index = this.#allocations.length - 1; index >= checkpoint; index -= 1) {
      this.#allocations[index] .release();
    }
    this.#allocations.length = checkpoint;
  }

  release() {
    this.releaseSince(0);
    this.allocator.destroyPooled();
  }
}
