import { GpuBufferAllocator } from "./allocator.js";
import { pipelineCacheForDevice } from "./pipeline-cache.js";
import { storageBytes, storageWords } from "./storage.js";

const GRID_WIDTH = 32_768;
const MAX_WORKGROUPS_PER_DIMENSION = 65_535;
const ADD_IN_PLACE_SHADER = `
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
      entries: tensors.map((tensor, binding) => ({
        binding,
        resource: {
          buffer: tensor.allocation.buffer,
          offset: storageWords(tensor.offsetElements ?? 0, tensor.storage ?? "f32") * 4,
          size: storageBytes(tensor.elements, tensor.storage ?? "f32"),
        },
      })),
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

  checkpoint() { return this.#allocations.length; }

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
