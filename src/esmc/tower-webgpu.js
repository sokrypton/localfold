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
import { float32ToFloat16Array } from "../runtime/float16.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";
import {
  GRID_WIDTH, LANES, createAttentionShader, createLayerNormShader,
  createLinearShader, createPrepareShader, createSwigluShader, linearGrid,
  swigluGrid,
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
    // 🔴 THE TOWER HAS ITS OWN UPLOAD PATH AND HAD ITS OWN DEFAULT. Setting f16
    // as the block's default changed nothing here and the tower's numbers came
    // back byte-identical - which looked like f16 costing nothing and was two
    // code paths having drifted. Same hazard as a checker that builds its own
    // kernel: what is measured has to be what runs.
    const weightPrecision = options.weightPrecision ?? "f16";
    const narrow = (values) => (weightPrecision === "f16"
      ? float32ToFloat16Array(values) : values);
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
      pipeline(`esmc-attend:${rows}:${model}:${heads}`,
        createAttentionShader({ rows, model, heads })),
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
      for (let layer = 0; layer < layers; layer += 1) {
        // Awaited, so a caller may stream a block's weights from the network or
        // decode them from a quantised bundle rather than holding 2190 MiB of
        // float32 in the tab. A bench should pre-load and hand back a plain
        // object; a checker should not have to.
        const weights = await blockWeights(layer);
        const perBlock = [];
        const upload = (name, values) => {
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

        const attnScale = upload("attn_norm/scale", weights["attn_norm/scale"]);
        const attnOffset = upload("attn_norm/offset", weights["attn_norm/offset"]);
        const qkvWeights = upload("qkv", narrow(weights["qkv/weights"]));
        const qScale = upload("q_norm", weights["q_norm/scale"]);
        const kScale = upload("k_norm", weights["k_norm/scale"]);
        const attnOut = upload("attn_out", narrow(scaled(weights["attn_out/weights"])));
        const ffnScale = upload("ffn_norm/scale", weights["ffn_norm/scale"]);
        const ffnOffset = upload("ffn_norm/offset", weights["ffn_norm/offset"]);
        const fc1 = upload("fc1", narrow(weights["fc1/weights"]));
        const fc2 = upload("fc2", narrow(scaled(weights["fc2/weights"])));

        const encoder = this.device.createCommandEncoder({ label: `esmc-block-${layer}` });
        const pass = encoder.beginComputePass({ label: `esmc-block-${layer}` });
        dispatchInto(pass, normPipeline, [current, attnScale, attnOffset, normed], rows);
        dispatchLinear(pass, qkvPipeline, [normed, qkvWeights, qkv], 3 * model);
        dispatchInto(pass, preparePipeline, [qkv, qScale, kScale, query, key, value], rows);
        dispatchInto(pass, attentionPipeline, [query, key, value, context], rows * heads);
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
