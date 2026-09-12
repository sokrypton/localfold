// ESM-C's 36 blocks on the GPU, and ESMFold2's layer mix accumulated as they go.
//
// 🔴 THE 37 HIDDEN STATES NEVER EXIST AT ONCE, AND THAT IS THE WHOLE REASON
// THIS FITS A BROWSER. ESMFold2 mixes them with a learned softmax whose weights
// are CONSTANT, and both the LayerNorm and the projection in front of that sum
// are shared across k - so
//
//     single = sum_k mix[k] * LN(h_k) @ W
//
// is a running accumulator of (tokens, 256), not a (37, tokens, 1152) tensor.
// At 300 tokens that is 0.3 MiB against 51. The reference materialises the
// states because a reference should be obvious; this is the shipping shape, and
// `tools/esmc/esmc_forward.py` measures the two agreeing to 1.1e-7.
//
// 🔴 AND THE WEIGHTS STREAM A BLOCK AT A TIME. A block is 15.9 M parameters -
// 64 MiB as float32, 32 MiB at half - against the tower's 2190 MiB, so what is
// resident is one block plus the residual stream plus the accumulator. Each
// block's buffers are released after its own submit completes, which is why
// there is a submit per block rather than one for the tower: a buffer still
// queued cannot be freed.
//
// 🔴 AND THE DOWNPROJECTION IS OUTSIDE THE SUM. It is affine, so folding it into
// the per-state accumulation would add its bias 37 times. The reference has the
// same note; it is the kind of thing that survives a shape check and moves a
// fold.
import { GpuBufferAllocator } from "../runtime/allocator.js";
import {
  GpuMemoryBudgetError, memoryBudgetBytes, noteAllocation, noteDestroy,
  noteResidencyRefused, residencyAllowed,
} from "../runtime/device-memory.js";
import { float32ToFloat16Array } from "../runtime/float16.js";
import { halfPrecisionAvailable } from "../runtime/device-profile.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";
import { planBlockUpload, runBlockUpload } from "../runtime/quantised-upload.js";
import {
  GRID_WIDTH, LANES, createAttentionShader, createLayerNormShader,
  createLinearShader, createPrepareShader, createSwigluShader, linearGrid,
  swigluGrid, QUERY_TILE,
} from "./block-webgpu.js";

/**
 * One state's contribution to the mix: LayerNorm, project, scale, accumulate.
 *
 * One dispatch rather than three. The state is read once, normalised in
 * workgroup memory and projected straight into the accumulator, so the
 * (rows, model) normalised tensor never reaches global memory.
 */
function createMixShader({ rows, model, pair }, epsilon) {
  return `
@group(0) @binding(0) var<storage, read> state: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> offset: array<f32>;
@group(0) @binding(3) var<storage, read> projection: array<f32>;
@group(0) @binding(4) var<storage, read> weight: array<f32>;
@group(0) @binding(5) var<storage, read_write> accumulator: array<f32>;

var<workgroup> row_values: array<f32, ${model}>;
var<workgroup> sums: array<f32, ${LANES}>;
var<workgroup> squares: array<f32, ${LANES}>;

@compute @workgroup_size(${LANES})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
  let row = group.x + group.y * ${GRID_WIDTH}u;
  if (row >= ${rows}u) { return; }
  let base = row * ${model}u;
  var total = 0.0;
  var square = 0.0;
  for (var c = local.x; c < ${model}u; c += ${LANES}u) {
    let v = state[base + c];
    total += v;
    square += v * v;
  }
  sums[local.x] = total;
  squares[local.x] = square;
  workgroupBarrier();
  for (var stride = ${LANES / 2}u; stride > 0u; stride >>= 1u) {
    if (local.x < stride) {
      sums[local.x] += sums[local.x + stride];
      squares[local.x] += squares[local.x + stride];
    }
    workgroupBarrier();
  }
  let mean = sums[0] / ${model}.0;
  let variance = max(squares[0] / ${model}.0 - mean * mean, 0.0);
  let inverse = inverseSqrt(variance + ${epsilon});
  for (var c = local.x; c < ${model}u; c += ${LANES}u) {
    row_values[c] = (state[base + c] - mean) * inverse * scale[c] + offset[c];
  }
  workgroupBarrier();

  let share = weight[0];
  for (var o = local.x; o < ${pair}u; o += ${LANES}u) {
    var sum = 0.0;
    for (var i = 0u; i < ${model}u; i += 1u) {
      sum += row_values[i] * projection[i * ${pair}u + o];
    }
    accumulator[row * ${pair}u + o] += share * sum;
  }
}`;
}

/** softmax over the mix logits. A constant, so it is computed once on the host. */
export function layerMix(combine) {
  let largest = -Infinity;
  for (const value of combine) if (value > largest) largest = value;
  const out = new Float32Array(combine.length);
  let total = 0;
  for (let i = 0; i < combine.length; i += 1) {
    out[i] = Math.exp(combine[i] - largest);
    total += out[i];
  }
  for (let i = 0; i < out.length; i += 1) out[i] /= total;
  return out;
}

/**
 * A block's weight buffers, kept on the device for the model's lifetime.
 *
 * 🔴 STREAMING IS THE RIGHT TRADE ONCE AND THE WRONG ONE AFTERWARDS. This tower
 * decodes each block out of int3 and narrows it to f16 on the host - 1121 ms
 * and 723 ms across 36 blocks, against 660 ms of GPU - and releases the buffers
 * as it goes, so 2190 MiB of float32 never exists at once. That is what makes
 * the FIRST fold possible on a small device. It also means every fold after it
 * pays the whole 1.8 seconds again, over weights that never change, which on a
 * card with room is pure loss - the same shape as `keepTrunkWeights` in AF3 and
 * the residency added to every other stack here.
 *
 * 🔴 AND A HIT SKIPS THE DECODE, NOT JUST THE UPLOAD, which is why this cannot
 * be `residentWeightBuffer`. That one takes a `pack` thunk and calls it on a
 * miss, and by then the caller has already awaited `blockWeights(layer)` - the
 * 1121 ms. This is asked BEFORE the weights are requested, and a hit means the
 * layer's promise is never created.
 *
 * Keyed on a caller-supplied object, which is the model: the tower instance is
 * built fresh per fold and cannot be it.
 */
const RESIDENT_BLOCKS = new WeakMap();

function residentBlocks(device, key) {
  let forDevice = RESIDENT_BLOCKS.get(device);
  if (forDevice === undefined) {
    forDevice = new WeakMap();
    RESIDENT_BLOCKS.set(device, forDevice);
  }
  let forKey = forDevice.get(key);
  if (forKey === undefined) {
    forKey = new Map();
    forDevice.set(key, forKey);
  }
  return forKey;
}

/**
 * One of the tower's four big matrices, decoded from its quantisation codes on
 * the DEVICE and written straight into `destination` as f16.
 *
 * 🔴 THIS IS 4.8 SECONDS OF AN ESMFold2 FIRST FOLD. `readTensorAsFloat16`
 * decodes int3-at-128 into halves on the main thread; measured inside a fold,
 * the tower's block reads were 5482 ms of a 7990 ms wall, and the four matrices
 * are 99.6% of a block. The GPU decoder has understood this codec since it
 * stopped being int5-at-32 only, and nothing was pointed at it.
 *
 * Returns false when the tensor cannot be decoded this way - an f32 bundle, a
 * store with no `tensorSource`, a shard not open yet - and the caller decodes
 * on the host as it always did.
 *
 * @param {{record: object, buffer: ArrayBuffer, byteOffset: number}} source
 */
async function decodeIntoOnDevice(device, source, elements, destination) {
  if (source === undefined || source === null) return false;
  // A thunk in the shape planBlockUpload reads: it never calls it, it only
  // wants the store, the name and the range. See src/af3/weights.js.
  const thunk = () => { throw new Error("the device decoder does not call the thunk"); };
  thunk.store = { tensorSource: () => source };
  thunk.tensorName = "tower";
  thunk.first = 0;
  thunk.count = elements;
  const planned = planBlockUpload([{ name: "tower", thunk, offset: 0, length: elements }]);
  if (planned === undefined || planned.gpu.params.length === 0) return false;
  const release = await runBlockUpload(device, planned.gpu, destination);
  // 🔴 NOT AWAITED. The staging goes when the queue says so; waiting here would
  // put a host-device synchronisation inside the block loop, which is the one
  // thing this loop is written to avoid. See src/af3/device-weights.js.
  void device.queue.onSubmittedWorkDone().then(release);
  return true;
}

/** How many elements a manifest record holds. */
function elementsOf(record) {
  return (record.shape ?? []).reduce((total, extent) => total * extent, 1);
}

export class EsmcTowerGpu {
  constructor(device, allocator = new GpuBufferAllocator(device)) {
    this.device = device;
    this.allocator = allocator;
    this.pipelines = pipelineCacheForDevice(device);
  }

  /**
   * @param ids          token ids, BOS and EOS attached
   * @param shape        { rows, model, heads, ffn, layers, pair, residualScale }
   * @param blockWeights (layer) => the ten tensors, or a promise of them
   * @param shared       embed/weights, final_norm/scale and the lm/* shim tensors
   * @param options      { capture: [stateIndex], epsilon, ropeBase }
   */
  async run(ids, shape, blockWeights, shared, options = {}) {
    const { rows, model, heads, ffn, layers, pair } = shape;
    const residualScale = shape.residualScale ?? 1;
    const epsilon = (options.epsilon ?? 1e-5).toExponential();
    const ropeBase = (options.ropeBase ?? 10000).toFixed(1);
    const capture = new Set(options.capture ?? []);
    // 🔴 ONE PACKED RUN WITH A MASK, NOT ONE RUN A CHAIN. See
    // createAttentionShader: the rotary positions are absolute over the packed
    // array, so per-chain runs are a different model however the attention is
    // masked. `sequenceId` is one entry a row, and PAD is -1.
    const sequenceId = options.sequenceId;
    if (sequenceId !== undefined && sequenceId.length !== ids.length) {
      throw new Error(`${sequenceId.length} sequence ids for ${ids.length} tokens`);
    }
    // 🔴 THE TOWER HAS ITS OWN UPLOAD PATH AND HAD ITS OWN DEFAULT. Setting f16
    // as the block's default changed nothing here and the tower's numbers came
    // back byte-identical - which looked like f16 costing nothing and was two
    // code paths having drifted. Same hazard as a checker that builds its own
    // kernel: what is measured has to be what runs.
    // 🔴 AND THE DEFAULT IS A CAPABILITY, NOT A PREFERENCE. It was a bare
    // "f16", and a stock Chrome has no `shader-f16` on ANY NVIDIA GPU - Dawn
    // gates it vendor-wide, see docs/A100.md - so `esmc-linear` emitted
    // `enable f16` onto a device that refuses the extension and **ESMFold2 did
    // not fold at all for those visitors**. It failed as a WGSL parse error
    // with no key attached, which is why it went unseen: the harness passes
    // the flag that hides it.
    const weightPrecision = options.weightPrecision
      ?? (halfPrecisionAvailable(this.device) ? "f16" : "f32");
    // 🔴 AN ALREADY-NARROW ARRAY PASSES THROUGH, WHICH IS HOW A CALLER SKIPS A
    // WHOLE PASS OVER THE WEIGHTS. Decoding int3 to float32 and narrowing
    // afterwards reads and writes 573 M elements twice; a caller that asks its
    // store for `tensorAsFloat16` has already done the narrowing as it
    // unpacked, and this must not do it again - `float32ToFloat16Array` of a
    // Float16Array would reinterpret its BITS as numbers.
    const isNarrow = (values) => values instanceof Uint16Array
      || (typeof Float16Array === "function" && values instanceof Float16Array);
    const narrow = (values) => {
      if (isNarrow(values)) {
        if (weightPrecision !== "f16") {
          throw new Error("this tower is compiled for float32 weights and was handed"
            + " half-precision ones; the shader would read them at twice the stride");
        }
        return values;
      }
      return weightPrecision === "f16" ? float32ToFloat16Array(values) : values;
    };
    if (ids.length !== rows) throw new Error(`${ids.length} ids for ${rows} rows`);

    const storage = GPUBufferUsage.STORAGE;
    const persistent = [];
    const keepPersistent = (allocation) => { persistent.push(allocation); return allocation; };

    const pipeline = (name, source) => this.pipelines.get(name, source);
    const [normPipeline, qkvPipeline, preparePipeline, attentionPipeline,
      outPipeline, swigluPipeline, downPipeline, mixPipeline,
      finalNormPipeline, singlePipeline] = await Promise.all([
      pipeline(`esmc-ln:${rows}:${model}:${epsilon}`,
        createLayerNormShader({ rows, channels: model }, true, epsilon)),
      pipeline(`esmc-linear:${rows}:${model}:${3 * model}:0:${weightPrecision}`,
        createLinearShader({ rows, inner: model, outer: 3 * model }, false,
          weightPrecision)),
      pipeline(`esmc-prepare:${rows}:${model}:${heads}:${epsilon}:${ropeBase}`,
        createPrepareShader({ rows, model, heads }, epsilon, ropeBase)),
      // 🔴 CHAIN-AWARE IS PART OF THE CACHE KEY, NOT JUST OF THE BINDINGS. The
      // two shaders differ by a binding and a test, and a page that folds a
      // monomer and then a complex at the same length would otherwise reuse the
      // monomer's pipeline and quietly let the chains attend to each other.
      pipeline(`esmc-attend:${rows}:${model}:${heads}:${sequenceId === undefined ? "flat" : "chains"}`,
        createAttentionShader({ rows, model, heads }, QUERY_TILE, sequenceId !== undefined)),
      pipeline(`esmc-linear:${rows}:${model}:${model}:1:${weightPrecision}`,
        createLinearShader({ rows, inner: model, outer: model }, true,
          weightPrecision)),
      pipeline(`esmc-swiglu:${rows}:${model}:${ffn}:${weightPrecision}`,
        createSwigluShader({ rows, model, ffn }, weightPrecision)),
      pipeline(`esmc-linear:${rows}:${ffn}:${model}:1:${weightPrecision}`,
        createLinearShader({ rows, inner: ffn, outer: model }, true,
          weightPrecision)),
      pipeline(`esmc-mix:${rows}:${model}:${pair}:${epsilon}`,
        createMixShader({ rows, model, pair }, epsilon)),
      pipeline(`esmc-ln-nooffset:${rows}:${model}:${epsilon}`,
        createLayerNormShader({ rows, channels: model }, false, epsilon)),
      pipeline(`esmc-linear:${rows}:${pair}:${pair}:0`,
        createLinearShader({ rows, inner: pair, outer: pair }, false)),
    ]);

    const captured = new Map();
    try {
      // The embedding lookup on the host: 61 rows of a 64-entry table is not
      // work worth a dispatch, and it keeps the vocabulary out of the shaders.
      const table = shared["embed/weights"];
      const embedded = new Float32Array(rows * model);
      for (let row = 0; row < rows; row += 1) {
        embedded.set(table.subarray(ids[row] * model, (ids[row] + 1) * model), row * model);
      }

      let current = keepPersistent(this.allocator.upload("esmc.x", embedded, storage
        | GPUBufferUsage.COPY_SRC));
      const accumulator = keepPersistent(this.allocator.allocate(
        "esmc.accumulator", rows * pair * 4, storage | GPUBufferUsage.COPY_SRC));
      const mixScale = keepPersistent(this.allocator.upload(
        "esmc.lm.scale", shared["lm/norm/scale"], storage));
      const mixOffset = keepPersistent(this.allocator.upload(
        "esmc.lm.offset", shared["lm/norm/offset"], storage));
      const mixProjection = keepPersistent(this.allocator.upload(
        "esmc.lm.projection", shared["lm/projection/weights"], storage));
      const sequenceBuffer = sequenceId === undefined ? undefined
        : keepPersistent(this.allocator.upload("esmc.sequence-id", sequenceId, storage));
      const mix = layerMix(shared["lm/combine"]);
      const shares = [];
      for (let k = 0; k < mix.length; k += 1) {
        shares.push(keepPersistent(this.allocator.upload(
          `esmc.lm.share.${k}`, Float32Array.of(mix[k]), storage)));
      }

      const bind = (built, bindings) => this.device.createBindGroup({
        layout: built.getBindGroupLayout(0),
        entries: bindings.map((allocation, binding) => ({
          binding, resource: { buffer: allocation.buffer },
        })),
      });
      const dispatchInto = (pass, built, bindings, count) => {
        pass.setPipeline(built);
        pass.setBindGroup(0, bind(built, bindings));
        pass.dispatchWorkgroups(Math.min(count, GRID_WIDTH), Math.ceil(count / GRID_WIDTH));
      };
      // The tiled linear pass means something different by each grid axis.
      const dispatchLinear = (pass, built, bindings, outer) => {
        pass.setPipeline(built);
        pass.setBindGroup(0, bind(built, bindings));
        const [x, y] = linearGrid(rows, outer);
        pass.dispatchWorkgroups(x, y);
      };

      const readBack = async (allocation, elements, label) => {
        const readback = this.allocator.allocate(`esmc.read.${label}`, elements * 4,
          GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(allocation.buffer, 0, readback.buffer, 0, elements * 4);
        this.device.queue.submit([encoder.finish()]);
        await readback.buffer.mapAsync(GPUMapMode.READ);
        const values = new Float32Array(readback.buffer.getMappedRange().slice(0));
        readback.buffer.unmap();
        readback.release();
        return values;
      };

      // State 0 is the embedding, and it is mixed like any other.
      {
        const encoder = this.device.createCommandEncoder({ label: "esmc-mix-0" });
        const pass = encoder.beginComputePass();
        dispatchInto(pass, mixPipeline,
          [current, mixScale, mixOffset, mixProjection, shares[0], accumulator], rows);
        pass.end();
        this.device.queue.submit([encoder.finish()]);
      }
      if (capture.has(0)) captured.set(0, await readBack(current, rows * model, "s0"));

      const started = performance.now();
      // 🔴 THE NEXT BLOCK'S WEIGHTS ARE ASKED FOR BEFORE THIS ONE'S GPU WORK IS
      // AWAITED, AND THAT IS WORTH A THIRD OF A FOLD'S LANGUAGE MODEL. Decoding
      // one block out of int3 and narrowing it to f16 is HOST work - measured
      // at 1121 ms and 723 ms respectively across 36 blocks, against 660 ms of
      // GPU - and the loop used to do it strictly between submits: decode,
      // upload, submit, wait, decode the next. JavaScript is single-threaded
      // but the GPU is not, so starting the decode before `onSubmittedWorkDone`
      // puts it inside the window the device is busy in.
      //
      // 🔴 AND ONE BLOCK AHEAD, NOT ALL OF THEM. The whole point of streaming
      // is that 2190 MiB of float32 never exists at once; prefetching the lot
      // would defeat it exactly. Two blocks live is 64 MiB.
      // 🔴 RESIDENT IF THE CALLER NAMES A MODEL AND THE DEVICE HAS NO CEILING.
      // See residentBlocks: a hit skips the DECODE and not just the upload, so
      // it is asked before the layer's promise is created. A device with a
      // budget streams as it always did, which is what makes the first fold
      // possible on a phone; a card with room pays 1.8 s once instead of once a
      // fold. The same rule uploadResident takes in src/runtime/execution.js.
      //
      // 🔴 AND A CEILING IS ANSWERED BY SIZE, NOT BY REFUSING OUTRIGHT. The
      // page is the caller that matters and it ALWAYS sets a budget - `null`
      // in web/model.js means "guess one from navigator.deviceMemory" - so a
      // rule of "no budget, no residency" would give this to a bench and to
      // nobody else. What the weights actually cost is known here: ten tensors
      // a block, the four big ones being qkv, attn_out, fc1 and fc2, in halves.
      // A quarter of the ceiling is the line: a 16 GiB desktop guesses about
      // 5.5 GiB and takes it, a 4 GiB phone guesses 1.3 and streams as before.
      const blockBytes = 2 * (4 * model * model + 2 * model * ffn) + 8 * model * 2;
      const residentBytes = blockBytes * layers;
      const budget = memoryBudgetBytes(this.device);
      const affordable = budget === undefined || budget >= residentBytes * 4;
      const resident = options.weightKey !== undefined
        && residencyAllowed(this.device) && affordable
        ? residentBlocks(this.device, options.weightKey) : null;
      const cached = (layer) => (resident === null ? undefined : resident.get(layer));
      const want = (layer) => (layer < layers && cached(layer) === undefined
        ? blockWeights(layer) : undefined);
      let pending = layers > 0 ? want(0) : undefined;
      for (let layer = 0; layer < layers; layer += 1) {
        // Awaited, so a caller may stream a block's weights from the network or
        // decode them from a quantised bundle rather than holding 2190 MiB of
        // float32 in the tab. A bench should pre-load and hand back a plain
        // object; a checker should not have to.
        const held = cached(layer);
        const weights = held === undefined ? await pending : undefined;
        pending = want(layer + 1);
        const perBlock = [];
        const store = held ?? (resident === null ? undefined : {});
        if (held === undefined && store !== undefined) resident.set(layer, store);
        // 🔴 THE VALUE IS A THUNK, because on a cache hit `weights` is undefined
        // and JavaScript evaluates an argument before the function can decide
        // it does not need it. That is the whole saving: `narrow(...)` and
        // `scaled(...)` are the 723 ms.
        const upload = (name, make) => {
          if (store !== undefined && store[name] !== undefined) return store[name];
          const values = make();
          // 🔴 NOT THROUGH THE POOLED ALLOCATOR WHEN IT IS KEPT, because the
          // pool recycles at the end of the run that made it and this has to
          // outlive every run - the same rule src/runtime/resident.js states.
          if (store !== undefined) {
            // 🔴 AND THE BUDGET STILL GETS THE LAST WORD. The estimate above is
            // an estimate; if an allocation crosses the ceiling anyway,
            // noteAllocation raises before createBuffer, this device gives up
            // on residency for good and every block from here streams. Partial
            // residency is safe because a block's buffers are independent.
            try {
              noteAllocation(this.device, `esmc.${name}`,
                             Math.ceil(values.byteLength / 4) * 4);
            } catch (error) {
              if (!(error instanceof GpuMemoryBudgetError)) throw error;
              noteResidencyRefused(this.device);
              resident.delete(layer);
              const fallback = this.allocator.upload(
                `esmc.b${layer}.${name}`, values, storage);
              perBlock.push(fallback);
              return fallback;
            }
            const buffer = this.device.createBuffer({
              label: `esmc.${name}`, size: Math.ceil(values.byteLength / 4) * 4,
              usage: storage | GPUBufferUsage.COPY_DST });
            this.device.queue.writeBuffer(buffer, 0, values.buffer,
                                          values.byteOffset, values.byteLength);
            store[name] = { buffer };
            return store[name];
          }
          const allocation = this.allocator.upload(`esmc.b${layer}.${name}`, values, storage);
          perBlock.push(allocation);
          return allocation;
        };
        const scratch = (name, elements) => {
          const allocation = this.allocator.allocate(`esmc.b${layer}.${name}`,
            elements * 4, storage | GPUBufferUsage.COPY_SRC);
          perBlock.push(allocation);
          return allocation;
        };
        const scaled = (values) => {
          if (residualScale === 1) return values;
          // 🔴 A SCALED WEIGHT CANNOT ARRIVE ALREADY NARROW. This divides
          // element by element and a Uint16Array holds BITS, so the arithmetic
          // would be on the encoding. No checkpoint here scales its residual,
          // which is exactly why this would go unnoticed.
          if (isNarrow(values)) {
            throw new Error(`this tower scales its residual by ${residualScale} and was`
              + " handed half-precision weights, which cannot be scaled elementwise");
          }
          const out = new Float32Array(values.length);
          for (let i = 0; i < values.length; i += 1) out[i] = values[i] / residualScale;
          return out;
        };

        const normed = scratch("normed", rows * model);
        const qkv = scratch("qkv", rows * 3 * model);
        const query = scratch("query", rows * model);
        const key = scratch("key", rows * model);
        const value = scratch("value", rows * model);
        const context = scratch("context", rows * model);
        const afterAttention = scratch("afterAttention", rows * model);
        const ffnNormed = scratch("ffnNormed", rows * model);
        const gated = scratch("gated", rows * ffn);
        const next = keepPersistent(this.allocator.allocate(`esmc.x.${layer}`,
          rows * model * 4, storage | GPUBufferUsage.COPY_SRC));

        // 🔴 THE FOUR BIG MATRICES GO THROUGH THE DEVICE DECODER WHEN THE
        // CALLER OFFERS THEIR CODES. `sources` is a leaf name to
        // `{record, buffer, byteOffset}`, exactly what HttpTensorStore's
        // `tensorSource` returns - and a caller that does not offer one gets
        // the host path unchanged, which is every checker and every bench.
        // Measured: the tower's block reads were 5482 ms of a 7990 ms first
        // ESMFold2 fold, and this is 99.6% of them.
        //
        // 🔴 AND A SCALED RESIDUAL CANNOT TAKE IT, for the reason `scaled`
        // already gives: the division is elementwise over floats and these
        // never become floats here. No checkpoint in this repository scales,
        // so the branch is a guard rather than a path.
        const sources = residualScale === 1 ? weights?.sources : undefined;
        const uploadNarrow = async (name, leaf, make) => {
          if (store !== undefined && store[name] !== undefined) return store[name];
          const source = sources?.[leaf];
          if (source !== undefined) {
            const elements = elementsOf(source.record);
            const bytes = Math.ceil(elements / 2) * 4;
            // The resident path owns its buffer for the model's lifetime; the
            // streaming one hands it to the pool, as `upload` does.
            if (store !== undefined) {
              try {
                noteAllocation(this.device, `esmc.${name}`, bytes);
              } catch (error) {
                if (!(error instanceof GpuMemoryBudgetError)) throw error;
                noteResidencyRefused(this.device);
                resident.delete(layer);
                return upload(name, make);
              }
              const buffer = this.device.createBuffer({
                label: `esmc.${name}`, size: bytes,
                usage: storage | GPUBufferUsage.COPY_DST });
              if (await decodeIntoOnDevice(this.device, source, elements, buffer)) {
                store[name] = { buffer };
                return store[name];
              }
              buffer.destroy();
              noteDestroy(this.device, bytes, `esmc.${name}`);
              return upload(name, make);
            }
            const allocation = this.allocator.allocate(
              `esmc.b${layer}.${name}`, bytes, storage | GPUBufferUsage.COPY_DST);
            if (await decodeIntoOnDevice(this.device, source, elements, allocation.buffer)) {
              perBlock.push(allocation);
              return allocation;
            }
            allocation.release();
          }
          return upload(name, make);
        };

        const attnScale = upload("attn_norm/scale", () => weights["attn_norm/scale"]);
        const attnOffset = upload("attn_norm/offset", () => weights["attn_norm/offset"]);
        const qkvWeights = await uploadNarrow("qkv", "qkv/weights",
          () => narrow(weights["qkv/weights"]));
        const qScale = upload("q_norm", () => weights["q_norm/scale"]);
        const kScale = upload("k_norm", () => weights["k_norm/scale"]);
        const attnOut = await uploadNarrow("attn_out", "attn_out/weights",
          () => narrow(scaled(weights["attn_out/weights"])));
        const ffnScale = upload("ffn_norm/scale", () => weights["ffn_norm/scale"]);
        const ffnOffset = upload("ffn_norm/offset", () => weights["ffn_norm/offset"]);
        const fc1 = await uploadNarrow("fc1", "fc1/weights",
          () => narrow(weights["fc1/weights"]));
        const fc2 = await uploadNarrow("fc2", "fc2/weights",
          () => narrow(scaled(weights["fc2/weights"])));

        const encoder = this.device.createCommandEncoder({ label: `esmc-block-${layer}` });
        const pass = encoder.beginComputePass({ label: `esmc-block-${layer}` });
        dispatchInto(pass, normPipeline, [current, attnScale, attnOffset, normed], rows);
        dispatchLinear(pass, qkvPipeline, [normed, qkvWeights, qkv], 3 * model);
        dispatchInto(pass, preparePipeline, [qkv, qScale, kScale, query, key, value], rows);
        pass.setPipeline(attentionPipeline);
        pass.setBindGroup(0, bind(attentionPipeline, sequenceId === undefined
          ? [query, key, value, context]
          : [query, key, value, context, sequenceBuffer]));
        pass.dispatchWorkgroups(Math.ceil(rows / QUERY_TILE), heads);
        dispatchLinear(pass, outPipeline, [context, attnOut, current, afterAttention], model);
        dispatchInto(pass, normPipeline,
          [afterAttention, ffnScale, ffnOffset, ffnNormed], rows);
        pass.setPipeline(swigluPipeline);
        pass.setBindGroup(0, bind(swigluPipeline, [ffnNormed, fc1, gated]));
        {
          const [gx, gy] = swigluGrid(rows, ffn);
          pass.dispatchWorkgroups(gx, gy);
        }
        dispatchLinear(pass, downPipeline, [gated, fc2, afterAttention, next], model);
        // 🔴 THE LAST STATE IS FINAL-NORMED AND THE OTHER 36 ARE NOT, so the
        // mix reads `next` directly here and a normed copy on the last layer.
        if (layer + 1 < layers) {
          dispatchInto(pass, mixPipeline,
            [next, mixScale, mixOffset, mixProjection, shares[layer + 1], accumulator], rows);
        }
        pass.end();
        this.device.queue.submit([encoder.finish()]);

        // A buffer still queued cannot be released, which is why the tower
        // submits per block rather than once.
        await this.device.queue.onSubmittedWorkDone();
        for (let index = perBlock.length - 1; index >= 0; index -= 1) perBlock[index].release();

        if (capture.has(layer + 1) && layer + 1 < layers) {
          captured.set(layer + 1, await readBack(next, rows * model, `s${layer + 1}`));
        }
        // AllocatedGpuBuffer has no label and release() is idempotent, so the
        // previous residual stream is simply released - including the initial
        // embedding upload after the first block.
        current.release();
        current = next;
        // 🔴 THE TOWER IS THE LARGEST BAND WITH NOTHING TO SAY, so it says this.
        // At 76 residues it is 53% of a fold's predicted time and it used to
        // complete in one step, which put the progress bar at zero and then
        // straight to a half. Thirty-six blocks is plenty to move a bar with.
        await options.onBlock?.(layer, layers);
      }

      // The final norm, then the last state's contribution to the mix.
      const finalScale = keepPersistent(this.allocator.upload(
        "esmc.final_norm", shared["final_norm/scale"], storage));
      const finalState = keepPersistent(this.allocator.allocate("esmc.final",
        rows * model * 4, storage | GPUBufferUsage.COPY_SRC));
      {
        const encoder = this.device.createCommandEncoder({ label: "esmc-final" });
        const pass = encoder.beginComputePass();
        // Three bindings, not four: the scale-only norm declares no offset.
        dispatchInto(pass, finalNormPipeline, [current, finalScale, finalState], rows);
        dispatchInto(pass, mixPipeline,
          [finalState, mixScale, mixOffset, mixProjection, shares[layers], accumulator], rows);
        pass.end();
        this.device.queue.submit([encoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
      }
      if (capture.has(layers)) {
        captured.set(layers, await readBack(finalState, rows * model, "final"));
      }

      // The downprojection, once, outside the sum - its bias must not be added
      // thirty-seven times.
      const downWeights = keepPersistent(this.allocator.upload(
        "esmc.lm.down", shared["lm/downproject/weights"], storage));
      const single = keepPersistent(this.allocator.allocate("esmc.single",
        rows * pair * 4, storage | GPUBufferUsage.COPY_SRC));
      {
        const encoder = this.device.createCommandEncoder({ label: "esmc-single" });
        const pass = encoder.beginComputePass();
        dispatchLinear(pass, singlePipeline, [accumulator, downWeights, single], pair);
        pass.end();
        this.device.queue.submit([encoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
      }
      const mixed = await readBack(single, rows * pair, "single");
      const bias = shared["lm/downproject/bias"];
      for (let row = 0; row < rows; row += 1) {
        for (let c = 0; c < pair; c += 1) mixed[row * pair + c] += bias[c];
      }

      return {
        single: mixed,
        states: captured,
        elapsedMilliseconds: performance.now() - started,
        memory: this.allocator.snapshot?.(),
      };
    } finally {
      for (let index = persistent.length - 1; index >= 0; index -= 1) {
        try { persistent[index].release(); } catch { /* already released */ }
      }
    }
  }
}
