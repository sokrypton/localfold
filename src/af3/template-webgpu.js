/**
 * AF3's template embedder on the GPU - the empty-template path.
 *
 * 🔴 THIS IS THE MODULE EVERYONE SKIPS AND NOBODY SHOULD. With FOUR EMPTY
 * template slots its output measures std 13.1 against a pair whose own std is
 * 55 - about a quarter of what enters the MSA stack. Nine features are summed
 * into the embedding and only six are template geometry; the other three are
 * the query's own aatype (once per axis) and the query pair representation
 * itself. With no template the geometry vanishes and those three do not, so the
 * module becomes a learned transform of the query - and then runs it through
 * two pairformer blocks. AF2-multimer had the identical trap and it cost this
 * project a week there.
 *
 * 🔴 ONLY THE EMPTY-TEMPLATE PATH IS HERE, AND REAL TEMPLATES RAISE. The six
 * geometry features are identically zero without a template, so nothing in this
 * repository can tell a correct implementation of them from a wrong one.
 * Writing them anyway would add code no measurement covers.
 *
 * 🔴 THE SUM IS DIVIDED BY THE SLOT COUNT, NOT BY HOW MANY SLOTS ARE REAL. Four
 * empty slots produce the same embedding four times, so the division puts it
 * back and the module behaves as though there were exactly one template.
 * Dividing by the number of REAL templates would be a division by zero here.
 *
 * The two template blocks are the shared pair track at 64 channels with a
 * factor-2 transition; see src/af3/pair-track-gpu.js.
 */
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";
import {
  GRID_WIDTH, compilePairTrack, encodePairTrack, packPairTrackWeights,
} from "./pair-track-gpu.js";

const CHANNELS = 64;
const RESTYPES = 31;

const ORDER = [
  "queryEmbeddingNormScale", "queryEmbeddingNormOffset", "templatePairEmbedding8",
  "templatePairEmbedding2", "templatePairEmbedding3",
  "outputLayerNormScale", "outputLayerNormOffset", "outputLinear",
];

export function packTemplateWeights(weights) {
  const offsets = {};
  let total = 0;
  for (const name of ORDER) {
    if (weights[name] === undefined) throw new Error(`template weights missing ${name}`);
    offsets[name] = total;
    total += weights[name].length;
  }
  const data = new Float32Array(total);
  for (const name of ORDER) data.set(weights[name], offsets[name]);
  return { data, offsets };
}

export function createTemplateShaders(shape, offsets, epsilon, variance) {
  const { tokens, queryChannels, templates } = shape;
  const pairs = tokens * tokens;

  const common = `
const TOKENS: u32 = ${tokens}u;
const PAIRS: u32 = ${pairs}u;
const QUERY_CHANNELS: u32 = ${queryChannels}u;
const CHANNELS: u32 = ${CHANNELS}u;
const RESTYPES: u32 = ${RESTYPES}u;
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;
const EPSILON: f32 = ${epsilon};
// The slot count, not the real-template count - see the note at the top.
const TEMPLATE_SCALE: f32 = ${templates / (1e-7 + templates)};
const W_QUERY_SCALE: u32 = ${offsets.queryEmbeddingNormScale}u;
const W_QUERY_OFFSET: u32 = ${offsets.queryEmbeddingNormOffset}u;
const W_EMBED8: u32 = ${offsets.templatePairEmbedding8}u;
const W_EMBED2: u32 = ${offsets.templatePairEmbedding2}u;
const W_EMBED3: u32 = ${offsets.templatePairEmbedding3}u;
const W_OUT_SCALE: u32 = ${offsets.outputLayerNormScale}u;
const W_OUT_OFFSET: u32 = ${offsets.outputLayerNormOffset}u;
const W_OUT: u32 = ${offsets.outputLinear}u;
`;

  const varianceCode = (count, read) => variance === "fast"
    ? `let variance = squares / f32(${count}) - mean * mean;`
    : `var variance = 0.0;
  for (var c = 0u; c < ${count}; c += 1u) {
    let d = ${read} - mean;
    variance += d * d;
  }
  variance /= f32(${count});`;

  // The query pair representation, normalised and projected to 64, plus the
  // template aatype along each axis. An empty slot carries type 0, so those two
  // contribute ROW 0 of each weight rather than nothing.
  const embed = `${common}
@group(0) @binding(0) var<storage, read> pair: array<f32>;
@group(0) @binding(1) var<storage, read> aatype: array<i32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> act: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= PAIRS) { return; }
  let i = row / TOKENS;
  let j = row % TOKENS;
  let base = row * QUERY_CHANNELS;

  var total = 0.0;
  var squares = 0.0;
  for (var c = 0u; c < QUERY_CHANNELS; c += 1u) {
    let value = pair[base + c];
    total += value;
    squares += value * value;
  }
  let mean = total / f32(QUERY_CHANNELS);
  ${varianceCode("QUERY_CHANNELS", "pair[base + c]")}
  let inverse_std = inverseSqrt(variance + EPSILON);

  // 🔴 FEATURE 2 VARIES ALONG j AND FEATURE 3 ALONG i. AF3 writes them as
  // aatype[None, :, :] and aatype[:, None, :]; swapping them transposes a term
  // that nothing downstream complains about.
  let code_row = aatype[j];
  let code_column = aatype[i];
  for (var e = 0u; e < CHANNELS; e += 1u) {
    var value = 0.0;
    for (var c = 0u; c < QUERY_CHANNELS; c += 1u) {
      let normalized = (pair[base + c] - mean) * inverse_std * weights[W_QUERY_SCALE + c]
        + weights[W_QUERY_OFFSET + c];
      value += normalized * weights[W_EMBED8 + c * CHANNELS + e];
    }
    if (code_row >= 0 && u32(code_row) < RESTYPES) {
      value += weights[W_EMBED2 + u32(code_row) * CHANNELS + e];
    }
    if (code_column >= 0 && u32(code_column) < RESTYPES) {
      value += weights[W_EMBED3 + u32(code_column) * CHANNELS + e];
    }
    act[row * CHANNELS + e] = value;
  }
}`;

  // LayerNorm, the slot-count scaling, a relu, and the projection back up.
  const output = `${common}
@group(0) @binding(0) var<storage, read> act: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= PAIRS) { return; }
  let base = row * CHANNELS;

  var total = 0.0;
  var squares = 0.0;
  for (var c = 0u; c < CHANNELS; c += 1u) {
    let value = act[base + c];
    total += value;
    squares += value * value;
  }
  let mean = total / f32(CHANNELS);
  ${varianceCode("CHANNELS", "act[base + c]")}
  let inverse_std = inverseSqrt(variance + EPSILON);

  for (var f = 0u; f < QUERY_CHANNELS; f += 1u) {
    var value = 0.0;
    for (var c = 0u; c < CHANNELS; c += 1u) {
      var normalized = (act[base + c] - mean) * inverse_std * weights[W_OUT_SCALE + c]
        + weights[W_OUT_OFFSET + c];
      normalized = normalized * TEMPLATE_SCALE;
      // ...relu BEFORE the projection, so the module can only add along a
      // non-negative combination of output_linear's directions.
      normalized = max(normalized, 0.0);
      value += normalized * weights[W_OUT + c * QUERY_CHANNELS + f];
    }
    output[row * QUERY_CHANNELS + f] = value;
  }
}`;

  return { embed, output };
}

export class Af3TemplateEmbedderGpu {
  constructor(device) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  /**
   * @param {{pair: Float32Array, pairMask: Float32Array, tokens: number,
   *          templates: number, templateOccupied?: boolean,
   *          templateAatype?: ArrayLike<number>}} input
   * @param {object} weights the eight tensors in ORDER, `blocks`, `queryChannels`
   * @param {{swapTransposedBias: boolean}} dialect
   */
  async run(input, weights, dialect, options = {}) {
    const { tokens, templates } = input;
    if (input.templateOccupied) {
      throw new Error("this template embedder only implements the empty-template path;"
        + " see src/af3/template-reference.js");
    }
    if (dialect?.swapTransposedBias === undefined) {
      throw new Error("dialect.swapTransposedBias has no default");
    }
    const queryChannels = weights.queryChannels;
    const pairs = tokens * tokens;
    const epsilon = options.epsilon ?? 1e-5;
    const variance = options.variance ?? "fast";

    const packed = packTemplateWeights(weights);
    const sources = createTemplateShaders(
      { tokens, queryChannels, templates }, packed.offsets, epsilon, variance);
    const base = `af3-template:${tokens}:${queryChannels}:${templates}:${epsilon}`
      + `:${variance}:${dialect.swapTransposedBias}`;
    const compiled = {};
    for (const [name, source] of Object.entries(sources)) {
      compiled[name] = await this.pipelines.get(`${base}:${name}`, source);
    }
    // The template stack: the shared pair track at 64 channels, factor 2.
    const trackPipelines = await compilePairTrack(this.pipelines, {
      n: tokens, channels: CHANNELS, transitionFactor: 2,
      sample: weights.blocks[0], epsilon, variance, dialect, base: `${base}:track`,
    });

    const gridHeads = weights.blocks[0].pairAttention1.heads;
    const storage = GPUBufferUsage.STORAGE;
    const allocations = [];
    const keep = (allocation) => { allocations.push(allocation); return allocation; };
    try {
      const pair = keep(this.allocator.upload("af3-template.pair", input.pair, storage));
      const pairMask = keep(this.allocator.upload("af3-template.mask", input.pairMask, storage));
      const weightBuffer = keep(this.allocator.upload("af3-template.weights", packed.data, storage));
      const aatypeData = new Int32Array(tokens);
      if (input.templateAatype !== undefined) {
        for (let t = 0; t < tokens; t += 1) aatypeData[t] = input.templateAatype[t];
      }
      const aatype = keep(this.allocator.upload("af3-template.aatype", aatypeData, storage));

      const act = keep(this.allocator.allocate("af3-template.act", pairs * CHANNELS * 4, storage));
      const scratch = [];
      for (let index = 0; index < 7; index += 1) {
        scratch.push(keep(this.allocator.allocate(
          `af3-template.scratch${index}`, pairs * CHANNELS * 4, storage)));
      }
      const biasBuffer = keep(this.allocator.allocate(
        "af3-template.bias", gridHeads * pairs * 4, storage));
      const output = keep(this.allocator.allocate(
        "af3-template.output", pairs * queryChannels * 4, storage | GPUBufferUsage.COPY_SRC));
      const readback = keep(this.allocator.allocate(
        "af3-template.readback", pairs * queryChannels * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));

      const blockAllocations = [];
      const upload = (label, data) => {
        const allocation = this.allocator.upload(label, data, storage);
        blockAllocations.push(allocation);
        return allocation;
      };

      this.device.pushErrorScope("validation");
      const encoder = this.device.createCommandEncoder({ label: "af3-template" });
      const run = (label, pipeline, buffers, x, y = 1, z = 1) => {
        const pass = encoder.beginComputePass({ label });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((allocation, binding) => ({
            binding, resource: { buffer: allocation.buffer },
          })),
        }));
        pass.dispatchWorkgroups(x, y, z);
        pass.end();
      };
      const spread = (groups) => [Math.min(groups, GRID_WIDTH), Math.ceil(groups / GRID_WIDTH)];
      const linear = spread(Math.ceil(pairs / 64));

      run("template.embed", compiled.embed, [pair, aatype, weightBuffer, act],
          linear[0], linear[1]);
      for (let index = 0; index < weights.blocks.length; index += 1) {
        const block = weights.blocks[index];
        const packedTrack = packPairTrackWeights(block, CHANNELS);
        encodePairTrack({
          run, pipelines: trackPipelines, n: tokens, channels: CHANNELS, gridHeads,
          pair: act, pairMask, scratch, biasBuffer,
          weights: {
            outgoing: upload(`w.tri.out.${index}`, packedTrack.outgoing),
            incoming: upload(`w.tri.in.${index}`, packedTrack.incoming),
            grid1: upload(`w.grid1.${index}`, packedTrack.grid1),
            grid2: upload(`w.grid2.${index}`, packedTrack.grid2),
            transition: upload(`w.transition.${index}`, packedTrack.transition),
          },
        });
      }
      run("template.output", compiled.output, [act, weightBuffer, output],
          linear[0], linear[1]);
      encoder.copyBufferToBuffer(output.buffer, 0, readback.buffer, 0, pairs * queryChannels * 4);

      const start = performance.now();
      this.device.queue.submit([encoder.finish()]);
      const error = await this.device.popErrorScope();
      await this.device.queue.onSubmittedWorkDone();
      for (let index = blockAllocations.length - 1; index >= 0; index -= 1) {
        blockAllocations[index].release();
      }
      if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
      await readback.buffer.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(readback.buffer.getMappedRange().slice(0));
      readback.buffer.unmap();
      return {
        output: result,
        elapsedMilliseconds: performance.now() - start,
        memory: this.allocator.snapshot(),
      };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index].release();
    }
  }
}
