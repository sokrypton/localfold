/**
 * AF3's atom decoder: token features back down to a position update per atom.
 *
 *     act = broadcast(project(tokenAct)) + encoder's skip connection, masked
 *     3 x cross-attention block, the same module the encoder runs
 *     -> LayerNorm -> project to 3 -> back to the token-atom layout
 *
 * 🔴 IT READS THE ENCODER'S PAIR CONDITIONING BUT HAS ITS OWN PAIR LAYERNORM
 * AND PROJECTION. The atom geometry did not change between encoder and decoder,
 * so the expensive part - the reference-conformer offsets, the trunk pair, the
 * three-layer MLP - is not recomputed; only the head biases are. Recomputing
 * the pair here would agree numerically and cost the most expensive pass in the
 * atom stack twice.
 *
 * 🔴 THE TOKEN FEATURES ARE BROADCAST TO EVERY ATOM SLOT OF A TOKEN, including
 * padded ones, and the mask is applied after the skip connection is added
 * rather than before the broadcast.
 *
 * The blocks are src/af3/atom-encoder-webgpu.js's, with the decoder's weights.
 */
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";
import { residentWeightBuffer } from "../runtime/resident.js";
import {
  createAtomBlockShaders, createAtomCommon, packAtomBlockWeights, packCached,
} from "./atom-encoder-webgpu.js";

const GRID_WIDTH = 32_768;

const PAIR_ORDER = [
  "pairInputLayerNormScale", "pairLogitsProjection",
  "projectTokenFeaturesForBroadcast", "atomFeaturesLayerNormScale",
  "atomFeaturesToPositionUpdate",
];

export function packDecoderPairWeights(weights) {
  const offsets = {};
  let total = 0;
  for (const name of PAIR_ORDER) {
    if (weights[name] === undefined) throw new Error(`atom decoder missing ${name}`);
    offsets[name] = total;
    total += weights[name].length;
  }
  const data = new Float32Array(total);
  for (const name of PAIR_ORDER) data.set(weights[name], offsets[name]);
  return { data, offsets };
}

export function createAtomDecoderShaders(shape, pairOffsets, blockOffsets) {
  const { channels } = shape;
  const common = createAtomCommon(shape, pairOffsets, blockOffsets);

  // The decoder's own head biases, from the encoder's pair representation.
  const pairLogits = `${common}
@group(0) @binding(0) var<storage, read> pair: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> logits: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= PAIR_ROWS) { return; }
  let base = row * C_PAIR;
  var total = 0.0;
  for (var c = 0u; c < C_PAIR; c += 1u) { total += pair[base + c]; }
  let mean = total / f32(C_PAIR);
  var variance = 0.0;
  for (var c = 0u; c < C_PAIR; c += 1u) {
    let d = pair[base + c] - mean;
    variance += d * d;
  }
  let inverse = inverseSqrt(variance / f32(C_PAIR) + EPSILON);

  let key = row % KEYS;
  let query_index = row / KEYS;
  let subset = query_index / QUERIES;
  let query = query_index % QUERIES;
  for (var block = 0u; block < BLOCKS; block += 1u) {
    for (var head = 0u; head < HEADS; head += 1u) {
      var value = 0.0;
      for (var c = 0u; c < C_PAIR; c += 1u) {
        value += (pair[base + c] - mean) * inverse
          * weights[P_pairInputLayerNormScale + c]
          * weights[P_pairLogitsProjection + c * BLOCKS * HEADS + block * HEADS + head];
      }
      logits[((block * SUBSETS + subset) * HEADS + head) * QUERIES * KEYS
        + query * KEYS + key] = value;
    }
  }
}`;

  // The starting activation: token features broadcast to atoms, gathered into
  // queries layout, plus the encoder's skip connection, then masked.
  const start = `${common}
@group(0) @binding(0) var<storage, read> token_act: array<f32>;
@group(0) @binding(1) var<storage, read> skip: array<f32>;
@group(0) @binding(2) var<storage, read> queries_mask: array<f32>;
@group(0) @binding(3) var<storage, read> gathers: array<i32>;
@group(0) @binding(4) var<storage, read> weights: array<f32>;
@group(0) @binding(5) var<storage, read_write> act: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= QUERY_ROWS) { return; }
  let live = gathers[G_TA_MASK + row] != 0;
  // The gather is into token-atom layout; the token is that slot / DENSE.
  let slot = u32(max(gathers[G_TA_IDX + row], 0));
  let token = slot / DENSE;

  for (var c = 0u; c < C; c += 1u) {
    var value = 0.0;
    if (live) {
      for (var d = 0u; d < C_TOKEN; d += 1u) {
        value += token_act[token * C_TOKEN + d]
          * weights[P_projectTokenFeaturesForBroadcast + d * C + c];
      }
    }
    act[row * C + c] = (value + skip[row * C + c]) * queries_mask[row];
  }
}`;

  // Mask, LayerNorm, project to three, and scatter back to token-atom layout.
  const finish = `${common}
@group(0) @binding(0) var<storage, read> act: array<f32>;
@group(0) @binding(1) var<storage, read> queries_mask: array<f32>;
@group(0) @binding(2) var<storage, read> gathers: array<i32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> update: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let slot = id.x + id.y * GRID_WIDTH * 64u;
  if (slot >= TOKENS * DENSE) { return; }
  // 🔴 THE OUTPUT IS IN TOKEN-ATOM LAYOUT, gathered THROUGH
  // queries_to_token_atoms - a slot with no query is left at zero.
  if (gathers[G_QTA_MASK + slot] == 0) {
    for (var axis = 0u; axis < 3u; axis += 1u) { update[slot * 3u + axis] = 0.0; }
    return;
  }
  let row = u32(max(gathers[G_QTA_IDX + slot], 0));

  var total = 0.0;
  for (var c = 0u; c < C; c += 1u) { total += act[row * C + c] * queries_mask[row]; }
  let mean = total / f32(C);
  var variance = 0.0;
  for (var c = 0u; c < C; c += 1u) {
    let d = act[row * C + c] * queries_mask[row] - mean;
    variance += d * d;
  }
  let inverse = inverseSqrt(variance / f32(C) + EPSILON);

  for (var axis = 0u; axis < 3u; axis += 1u) {
    var value = 0.0;
    for (var c = 0u; c < C; c += 1u) {
      value += ((act[row * C + c] * queries_mask[row] - mean) * inverse
        * weights[P_atomFeaturesLayerNormScale + c])
        * weights[P_atomFeaturesToPositionUpdate + c * 3u + axis];
    }
    update[slot * 3u + axis] = value;
  }
}`;

  // Only the four block passes; the encoder's masking and aggregation are its
  // own, and `aggregate` reads a weight this bundle does not carry.
  const { project, projectKeys, projectKeysAtoms, expandKeys, attendFor, output,
          outputRowTile } = createAtomBlockShaders(common, shape);
  return { pairLogits, start, finish, project, projectKeys, projectKeysAtoms, expandKeys,
           attendFor, output, outputRowTile };
}

export class Af3AtomDecoderGpu {
  constructor(device) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  /**
   * @param {Float32Array} tokenAct tokens * perTokenChannels
   * @param {{skipConnection, queriesMask, keysMask, queriesCond, keysCond,
   *          pairCond}} encoded the encoder's outputs
   * @param {object} input shape and the gathers
   * @param {object} weights the decoder's own, plus `blocks`
   */
  async run(tokenAct, encoded, input, weights) {
    const { tokens, dense, subsets, queries, keys } = input.shape;
    const channels = weights.channels;
    const pairChannels = weights.pairChannels;
    const heads = weights.heads;
    const dimension = weights.dimension;
    const width = heads * dimension;
    const queryRows = subsets * queries;
    const keyRows = subsets * keys;
    const pairRows = subsets * queries * keys;

    const pairPacked = packCached(weights, "dec.pair", () => packDecoderPairWeights(weights));
    const blockPacked = weights.blocks.map(
      (block) => packCached(block, "dec.block", () => packAtomBlockWeights(block)));
    const shape = {
      tokens, dense, subsets, queries, keys, channels, pairChannels, heads, dimension,
      perTokenChannels: weights.perTokenChannels,
      trunkSingleChannels: weights.trunkSingleChannels ?? 384,
      trunkPairChannels: weights.trunkPairChannels ?? 128,
      blocks: weights.blocks.length,
    };
    const sources = createAtomDecoderShaders(shape, pairPacked.offsets, blockPacked[0].offsets);
    const base = `af3-atom-dec:${tokens}:${dense}:${subsets}:${queries}:${keys}`
      + `:${channels}:${pairChannels}:${heads}:${dimension}:${weights.perTokenChannels}`;
    const compiled = {};
    for (const [name, source] of Object.entries(sources)) {
      // ...the factory also returns the row tile the dispatch needs, which is a
      // number rather than a shader.
      if (name === "attendFor" || typeof source !== "string") continue;
      compiled[name] = await this.pipelines.get(`${base}:${name}`, source);
    }
    compiled.attend = [];
    for (let index = 0; index < weights.blocks.length; index += 1) {
      compiled.attend.push(await this.pipelines.get(
        `${base}:attend:${index}`, sources.attendFor(index)));
    }

    const storage = GPUBufferUsage.STORAGE;
    const allocations = [];
    const keep = (a) => { allocations.push(a); return a; };
    const up = (label, data) => keep(this.allocator.upload(label, data, storage));
    const alloc = (label, bytes, extra = 0) =>
      keep(this.allocator.allocate(label, bytes, storage | extra));
    const ints = (source) => Int32Array.from(source, (v) => Number(v));

    try {
      const gathers = new Int32Array(5 * queryRows + 5 * keyRows + 2 * tokens * dense);
      let at = 0;
      const place = (source) => { gathers.set(ints(source), at); at += source.length; };
      place(input.tokenAtomsToQueries.indices);
      place(input.tokenAtomsToQueries.mask);
      place(input.tokensToQueries.indices);
      place(input.tokensToQueries.mask);
      place(input.queriesToKeys.indices);
      place(input.queriesToKeys.mask);
      place(input.tokensToKeys.indices);
      place(input.tokensToKeys.mask);
      place(input.queriesToTokenAtoms.indices);
      place(input.queriesToTokenAtoms.mask);
      const gatherBuffer = up("dec.gathers", gathers);

      const tokenActBuffer = up("dec.token-act", tokenAct);
      const skip = up("dec.skip", encoded.skipConnection);
      const queriesMask = up("dec.q-mask", encoded.queriesMask);
      const keysMask = up("dec.k-mask", encoded.keysMask);
      const queriesCond = up("dec.q-cond", encoded.queriesCond);
      const keysCond = up("dec.k-cond", encoded.keysCond);
      const pairCond = up("dec.pair", encoded.pairCond);
      const pairWeights = { buffer: residentWeightBuffer(this.device, weights,
        "dec.pair-weights", () => pairPacked.data) };
      const blockBuffers = weights.blocks.map((block, index) => ({
        buffer: residentWeightBuffer(this.device, block, "dec.block",
                                     () => blockPacked[index].data),
      }));

      const act = alloc("dec.act", queryRows * channels * 4);
      const logits = alloc("dec.logits",
        weights.blocks.length * subsets * heads * queries * keys * 4);
      const q = alloc("dec.q", queryRows * width * 4);
      const k = alloc("dec.k", keyRows * width * 4);
      const v = alloc("dec.v", keyRows * width * 4);
      const kAtoms = alloc("dec.k-atoms", queryRows * width * 4);
      const vAtoms = alloc("dec.v-atoms", queryRows * width * 4);
      const gate = alloc("dec.gate", queryRows * width * 4);
      const gathered = alloc("dec.gathered", queryRows * width * 4);
      const update = alloc("dec.update", tokens * dense * 3 * 4, GPUBufferUsage.COPY_SRC);
      const readback = keep(this.allocator.allocate("dec.rb", tokens * dense * 3 * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));

      this.device.pushErrorScope("validation");
      const encoder = this.device.createCommandEncoder({ label: "af3-atom-decoder" });
      const run = (label, pipeline, buffers, x, y = 1) => {
        const pass = encoder.beginComputePass({ label });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((a, binding) => ({ binding, resource: { buffer: a.buffer } })),
        }));
        pass.dispatchWorkgroups(x, y);
        pass.end();
      };
      const spread = (groups) => [Math.min(groups, GRID_WIDTH), Math.ceil(groups / GRID_WIDTH)];
      const lin = (count) => spread(Math.ceil(count / 64));

      const pr = lin(pairRows);
      run("pair-logits", compiled.pairLogits, [pairCond, pairWeights, logits], pr[0], pr[1]);
      const qr = lin(queryRows);
      run("start", compiled.start,
          [tokenActBuffer, skip, queriesMask, gatherBuffer, pairWeights, act], qr[0], qr[1]);

      for (let index = 0; index < weights.blocks.length; index += 1) {
        const w = blockBuffers[index];
        // ...one workgroup per TILE of query rows; see the note on `output`.
        const perQuery = spread(queryRows);
        const perOutput = spread(Math.ceil(queryRows / sources.outputRowTile));
        run(`project-${index}`, compiled.project, [act, queriesCond, w, q, gate],
            perQuery[0], perQuery[1]);
        run(`project-keys-${index}`, compiled.projectKeysAtoms,
            [act, queriesCond, w, kAtoms, vAtoms], perQuery[0], perQuery[1]);
        const expand = lin(keyRows * width);
        run(`expand-keys-${index}`, compiled.expandKeys,
            [kAtoms, vAtoms, gatherBuffer, k, v], expand[0], expand[1]);
        const slots = spread(queryRows * heads);
        run(`attend-${index}`, compiled.attend[index],
            [q, k, v, logits, queriesMask, keysMask, gathered], slots[0], slots[1]);
        run(`output-${index}`, compiled.output, [gathered, gate, queriesCond, w, act],
            perOutput[0], perOutput[1]);
      }

      const slotGroups = lin(tokens * dense);
      run("finish", compiled.finish, [act, queriesMask, gatherBuffer, pairWeights, update],
          slotGroups[0], slotGroups[1]);
      encoder.copyBufferToBuffer(update.buffer, 0, readback.buffer, 0, tokens * dense * 3 * 4);

      const start = performance.now();
      this.device.queue.submit([encoder.finish()]);
      const error = await this.device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
      await readback.buffer.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(readback.buffer.getMappedRange().slice(0));
      readback.buffer.unmap();
      return {
        update: result,
        elapsedMilliseconds: performance.now() - start,
        memory: this.allocator.snapshot(),
      };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index].release();
    }
  }
}
