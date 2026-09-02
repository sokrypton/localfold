/**
 * AF3's diffusion conditioning: what the denoiser knows besides the atoms.
 *
 *     pair   = proj(LayerNorm([trunk pair | relative encoding]))  then 2 transitions
 *     single = proj(LayerNorm([trunk single | target_feat]))
 *              + proj(LayerNorm(fourier(noise level)))            then 2 transitions
 *
 * 🔴 THE RELATIVE ENCODING IS CONCATENATED RAW HERE, NOT PROJECTED. In the
 * embedder its 139 columns are projected and can be gathered; here they are
 * glued to the trunk pair's 128 and the LayerNorm runs over all 267 together.
 * So the one-hot columns change the NORMALISATION STATISTICS of the trunk pair
 * columns beside them - the two uses of the same feature are not
 * interchangeable, and treating this one as a projection drops that coupling.
 *
 * It still does not have to be materialised. Only three or four of the 139 are
 * ever set, so the row's sum, its sum of squares and its projection all have
 * closed forms:
 *
 *     count      = 3 + sameEntity
 *     sum        = sum(pair) + count
 *     sum sq dev = sum((pair - mean)^2) + count*(1-mean)^2 + (139-count)*mean^2
 *     relative's contribution to output o
 *                = inv * (sum over SET bins of scale*W  -  mean * S[o])
 *
 * where S[o] = sum over ALL 139 of scale*W is a constant of the weights,
 * computed once at setup. That keeps a tokens^2 x 267 tensor (96 MB at 300
 * tokens) from ever existing, and unlike the embedder's gather it is exact
 * rather than an optimisation of a sparse matmul.
 *
 * 🔴 THE NOISE LEVEL IS DIVIDED BY SIGMA_DATA BEFORE THE LOG. Feeding raw
 * angstroms gives a Fourier embedding of a number three orders too large, which
 * aliases across the schedule instead of separating it.
 *
 * 🔴 THE TWO TRANSITIONS ARE UNCONDITIONED - `conditionedTransition(x, null)` -
 * so they are an ordinary LayerNorm-SwiGLU pair with a bias-carrying norm, not
 * AdaLN. They reuse the trunk's transition shader with two-pass variance.
 */
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";
import { createTransitionShader, packTransitionWeights } from "./transition-webgpu.js";

const GRID_WIDTH = 32_768;
const SIGMA_DATA = 16.0;
const MAX_RELATIVE_IDX = 32;
const MAX_RELATIVE_CHAIN = 2;
const POSITION_BINS = 2 * MAX_RELATIVE_IDX + 2;
const RELATIVE_WIDTH = POSITION_BINS * 2 + 1 + (2 * MAX_RELATIVE_CHAIN + 2);

/**
 * S[o] = sum over all 139 relative columns of scale[c] * W[c][o]. A constant of
 * the weights, so it is computed once here rather than per pair on the GPU.
 */
export function relativeColumnSums(scale, projection, pairChannels, outChannels) {
  const sums = new Float32Array(outChannels);
  for (let c = 0; c < RELATIVE_WIDTH; c += 1) {
    const row = pairChannels + c;
    const weight = scale[row];
    for (let out = 0; out < outChannels; out += 1) {
      sums[out] += weight * projection[row * outChannels + out];
    }
  }
  return sums;
}

export function createConditioningShaders(shape, offsets) {
  const { tokens, pairChannels, seqChannels, targetFeatWidth, noiseChannels } = shape;
  const pairs = tokens * tokens;
  const pairWidth = pairChannels + RELATIVE_WIDTH;
  const singleWidth = seqChannels + targetFeatWidth;

  const pairInitial = `
const TOKENS: u32 = ${tokens}u;
const PAIRS: u32 = ${pairs}u;
const C_PAIR: u32 = ${pairChannels}u;
const WIDTH: u32 = ${pairWidth}u;
const RELATIVE_WIDTH: u32 = ${RELATIVE_WIDTH}u;
const POSITION_BINS: u32 = ${POSITION_BINS}u;
const MAX_RELATIVE_IDX: i32 = ${MAX_RELATIVE_IDX};
const MAX_RELATIVE_CHAIN: i32 = ${MAX_RELATIVE_CHAIN};
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;
const EPSILON: f32 = 1.0e-5;

@group(0) @binding(0) var<storage, read> trunk_pair: array<f32>;
@group(0) @binding(1) var<storage, read> features: array<i32>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read> projection: array<f32>;
@group(0) @binding(4) var<storage, read> column_sums: array<f32>;
@group(0) @binding(5) var<storage, read_write> pair: array<f32>;

fn residue_index(t: u32) -> i32 { return features[t]; }
fn token_index(t: u32) -> i32 { return features[TOKENS + t]; }
fn asym_id(t: u32) -> i32 { return features[2u * TOKENS + t]; }
fn entity_id(t: u32) -> i32 { return features[3u * TOKENS + t]; }
fn sym_id(t: u32) -> i32 { return features[4u * TOKENS + t]; }
fn clamp_bin(value: i32, high: i32) -> i32 { return min(max(value, 0), high); }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= PAIRS) { return; }
  let i = row / TOKENS;
  let j = row % TOKENS;
  let base = row * C_PAIR;

  // Which relative columns are set.
  let same_chain = asym_id(i) == asym_id(j);
  let same_entity = entity_id(i) == entity_id(j);
  var bin_a = u32(2 * MAX_RELATIVE_IDX + 1);
  if (same_chain) {
    bin_a = u32(clamp_bin(residue_index(i) - residue_index(j) + MAX_RELATIVE_IDX,
                          2 * MAX_RELATIVE_IDX));
  }
  var bin_b = u32(2 * MAX_RELATIVE_IDX + 1);
  if (same_chain && residue_index(i) == residue_index(j)) {
    bin_b = u32(clamp_bin(token_index(i) - token_index(j) + MAX_RELATIVE_IDX,
                          2 * MAX_RELATIVE_IDX));
  }
  var bin_c = u32(2 * MAX_RELATIVE_CHAIN + 1);
  if (same_entity) {
    bin_c = u32(clamp_bin(sym_id(i) - sym_id(j) + MAX_RELATIVE_CHAIN,
                          2 * MAX_RELATIVE_CHAIN));
  }
  var count = 3.0;
  if (same_entity) { count = 4.0; }

  // The 267-wide statistics, in closed form.
  var total = count;
  for (var c = 0u; c < C_PAIR; c += 1u) { total += trunk_pair[base + c]; }
  let mean = total / f32(WIDTH);
  var variance = count * (1.0 - mean) * (1.0 - mean)
    + (f32(RELATIVE_WIDTH) - count) * mean * mean;
  for (var c = 0u; c < C_PAIR; c += 1u) {
    let d = trunk_pair[base + c] - mean;
    variance += d * d;
  }
  let inverse_std = inverseSqrt(variance / f32(WIDTH) + EPSILON);

  let row_a = C_PAIR + bin_a;
  let row_b = C_PAIR + POSITION_BINS + bin_b;
  let row_entity = C_PAIR + POSITION_BINS * 2u;
  let row_c = C_PAIR + POSITION_BINS * 2u + 1u + bin_c;

  for (var out = 0u; out < C_PAIR; out += 1u) {
    var value = 0.0;
    for (var c = 0u; c < C_PAIR; c += 1u) {
      value += (trunk_pair[base + c] - mean) * inverse_std * scale[c]
        * projection[c * C_PAIR + out];
    }
    // The set bins, minus the mean times every column's contribution.
    var gathered = scale[row_a] * projection[row_a * C_PAIR + out]
      + scale[row_b] * projection[row_b * C_PAIR + out]
      + scale[row_c] * projection[row_c * C_PAIR + out];
    if (same_entity) {
      gathered += scale[row_entity] * projection[row_entity * C_PAIR + out];
    }
    value += inverse_std * (gathered - mean * column_sums[out]);
    pair[row * C_PAIR + out] = value;
  }
}`;

  const singleInitial = `
const TOKENS: u32 = ${tokens}u;
const C_SEQ: u32 = ${seqChannels}u;
const TARGET_WIDTH: u32 = ${targetFeatWidth}u;
const WIDTH: u32 = ${singleWidth}u;
const NOISE_CHANNELS: u32 = ${noiseChannels}u;
const EPSILON: f32 = 1.0e-5;
const W_NOISE_SCALE: u32 = ${offsets.noiseEmbeddingInitialNormScale}u;
const W_NOISE_PROJECT: u32 = ${offsets.noiseEmbeddingInitialProjection}u;

@group(0) @binding(0) var<storage, read> trunk_single: array<f32>;
@group(0) @binding(1) var<storage, read> target_feat: array<f32>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read> projection: array<f32>;
@group(0) @binding(4) var<storage, read> noise: array<f32>;
@group(0) @binding(5) var<storage, read> noise_weights: array<f32>;
@group(0) @binding(6) var<storage, read_write> single: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let token = id.x;
  if (token >= TOKENS) { return; }

  var total = 0.0;
  for (var c = 0u; c < C_SEQ; c += 1u) { total += trunk_single[token * C_SEQ + c]; }
  for (var c = 0u; c < TARGET_WIDTH; c += 1u) { total += target_feat[token * TARGET_WIDTH + c]; }
  let mean = total / f32(WIDTH);
  var variance = 0.0;
  for (var c = 0u; c < C_SEQ; c += 1u) {
    let d = trunk_single[token * C_SEQ + c] - mean;
    variance += d * d;
  }
  for (var c = 0u; c < TARGET_WIDTH; c += 1u) {
    let d = target_feat[token * TARGET_WIDTH + c] - mean;
    variance += d * d;
  }
  let inverse_std = inverseSqrt(variance / f32(WIDTH) + EPSILON);

  // The noise embedding is one row, shared by every token: normalise and
  // project it here rather than in its own pass.
  var noise_total = 0.0;
  for (var c = 0u; c < NOISE_CHANNELS; c += 1u) { noise_total += noise[c]; }
  let noise_mean = noise_total / f32(NOISE_CHANNELS);
  var noise_variance = 0.0;
  for (var c = 0u; c < NOISE_CHANNELS; c += 1u) {
    let d = noise[c] - noise_mean;
    noise_variance += d * d;
  }
  let noise_inverse = inverseSqrt(noise_variance / f32(NOISE_CHANNELS) + EPSILON);

  for (var out = 0u; out < C_SEQ; out += 1u) {
    var value = 0.0;
    for (var c = 0u; c < C_SEQ; c += 1u) {
      value += (trunk_single[token * C_SEQ + c] - mean) * inverse_std * scale[c]
        * projection[c * C_SEQ + out];
    }
    for (var c = 0u; c < TARGET_WIDTH; c += 1u) {
      let index = C_SEQ + c;
      value += (target_feat[token * TARGET_WIDTH + c] - mean) * inverse_std * scale[index]
        * projection[index * C_SEQ + out];
    }
    for (var c = 0u; c < NOISE_CHANNELS; c += 1u) {
      value += (noise[c] - noise_mean) * noise_inverse * noise_weights[W_NOISE_SCALE + c]
        * noise_weights[W_NOISE_PROJECT + c * C_SEQ + out];
    }
    single[token * C_SEQ + out] = value;
  }
}`;

  return { pairInitial, singleInitial };
}

/**
 * The reference reads an unconditioned transition's weights as `ffwLayerNorm*`
 * and `ffwTransition*`, which is adaptiveLayerNorm's naming convention carried
 * over; the transition kernel names them for its own shader. Mapped here rather
 * than renamed in either, because both names are load-bearing where they are.
 */
function asTransitionWeights(block) {
  return {
    inputLayerNormScale: block.ffwLayerNormScale,
    inputLayerNormOffset: block.ffwLayerNormOffset,
    transition1: block.ffwTransition1,
    transition2: block.ffwTransition2,
  };
}

/** `residual += transition(residual)`, for the four unconditioned transitions. */
function createAddShader(elements) {
  return `
const ELEMENTS: u32 = ${elements}u;
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;
@group(0) @binding(0) var<storage, read_write> accumulator: array<f32>;
@group(0) @binding(1) var<storage, read> delta: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= ELEMENTS) { return; }
  accumulator[index] = accumulator[index] + delta[index];
}`;
}

/** AF3's Fourier noise embedding: cos(2 pi (w log(sigma)/4 + b)). */
export function noiseEmbedding(scaledNoiseLevel, weight, bias) {
  if (weight === undefined || bias === undefined) {
    throw new Error("the Fourier constants are required; see tools/export_af3_model.py");
  }
  const output = new Float32Array(weight.length);
  const logLevel = Math.log(scaledNoiseLevel) / 4;
  for (let index = 0; index < weight.length; index += 1) {
    output[index] = Math.cos(2 * Math.PI * (weight[index] * logLevel + bias[index]));
  }
  return output;
}

export class Af3DiffusionConditioningGpu {
  constructor(device) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  /**
   * @param {{tokens: number, trunkSingle, trunkPair, targetFeat, noiseLevel,
   *          features: {residueIndex, tokenIndex, asymId, entityId, symId}}} input
   * @param {object} weights
   */
  async run(input, weights, options = {}) {
    const tokens = input.tokens;
    const pairs = tokens * tokens;
    const pairChannels = weights.pairChannels;
    const seqChannels = weights.seqChannels;
    const targetFeatWidth = weights.targetFeatWidth;
    const noiseChannels = weights.fourierWeight.length;

    const embedded = noiseEmbedding(input.noiseLevel / SIGMA_DATA,
                                    weights.fourierWeight, weights.fourierBias);
    const columnSums = relativeColumnSums(weights.pairCondInitialNormScale,
                                          weights.pairCondInitialProjection,
                                          pairChannels, pairChannels);

    const noisePacked = (() => {
      const scale = weights.noiseEmbeddingInitialNormScale;
      const projection = weights.noiseEmbeddingInitialProjection;
      const data = new Float32Array(scale.length + projection.length);
      data.set(scale, 0);
      data.set(projection, scale.length);
      return { data, offsets: { noiseEmbeddingInitialNormScale: 0,
                                noiseEmbeddingInitialProjection: scale.length } };
    })();

    const shape = { tokens, pairChannels, seqChannels, targetFeatWidth, noiseChannels };
    const sources = createConditioningShaders(shape, noisePacked.offsets);
    const base = `af3-diffcond:${tokens}:${pairChannels}:${seqChannels}:${targetFeatWidth}`
      + `:${noiseChannels}`;
    const compiled = {
      pairInitial: await this.pipelines.get(`${base}:pair-initial`, sources.pairInitial),
      singleInitial: await this.pipelines.get(`${base}:single-initial`, sources.singleInitial),
      addPair: await this.pipelines.get(`${base}:add-pair`,
        createAddShader(pairs * pairChannels)),
      addSingle: await this.pipelines.get(`${base}:add-single`,
        createAddShader(tokens * seqChannels)),
    };
    // The four unconditioned transitions: the trunk's shader, two-pass variance.
    const transitionPipelines = { pair: [], single: [] };
    for (let index = 0; index < 2; index += 1) {
      const pairOffsets = packTransitionWeights(asTransitionWeights(weights.pairTransitions[index])).offsets;
      transitionPipelines.pair.push(await this.pipelines.get(`${base}:pair-transition:${index}`,
        createTransitionShader({ rows: pairs, channels: pairChannels, factor: 2 },
                               pairOffsets, 1e-5, "two-pass")));
      const singleOffsets = packTransitionWeights(
        asTransitionWeights(weights.singleTransitions[index])).offsets;
      transitionPipelines.single.push(await this.pipelines.get(
        `${base}:single-transition:${index}`,
        createTransitionShader({ rows: tokens, channels: seqChannels, factor: 2 },
                               singleOffsets, 1e-5, "two-pass")));
    }

    const featureData = new Int32Array(5 * tokens);
    ["residueIndex", "tokenIndex", "asymId", "entityId", "symId"].forEach((name, index) => {
      const source = input.features[name];
      if (source === undefined) throw new Error(`features.${name} is required`);
      for (let t = 0; t < tokens; t += 1) featureData[index * tokens + t] = source[t];
    });

    const storage = GPUBufferUsage.STORAGE;
    const allocations = [];
    const keep = (allocation) => { allocations.push(allocation); return allocation; };
    try {
      const up = (label, data) => keep(this.allocator.upload(label, data, storage));
      const trunkPair = up("cond.trunk-pair", input.trunkPair);
      const trunkSingle = up("cond.trunk-single", input.trunkSingle);
      const targetFeat = up("cond.target", input.targetFeat);
      const features = up("cond.features", featureData);
      const pairScale = up("cond.pair-scale", weights.pairCondInitialNormScale);
      const pairProjection = up("cond.pair-projection", weights.pairCondInitialProjection);
      const sums = up("cond.column-sums", columnSums);
      const singleScale = up("cond.single-scale", weights.singleCondInitialNormScale);
      const singleProjection = up("cond.single-projection", weights.singleCondInitialProjection);
      const noise = up("cond.noise", embedded);
      const noiseWeights = up("cond.noise-weights", noisePacked.data);

      const pair = keep(this.allocator.allocate("cond.pair", pairs * pairChannels * 4,
        storage | GPUBufferUsage.COPY_SRC));
      const single = keep(this.allocator.allocate("cond.single", tokens * seqChannels * 4,
        storage | GPUBufferUsage.COPY_SRC));
      const pairScratch = keep(this.allocator.allocate("cond.pair-scratch",
        pairs * pairChannels * 4, storage));
      const singleScratch = keep(this.allocator.allocate("cond.single-scratch",
        tokens * seqChannels * 4, storage));
      const readPair = keep(this.allocator.allocate("cond.rb-pair", pairs * pairChannels * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));
      const readSingle = keep(this.allocator.allocate("cond.rb-single", tokens * seqChannels * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));

      const transitionWeights = {
        pair: weights.pairTransitions.map((w, index) =>
          up(`cond.pair-transition-w${index}`, packTransitionWeights(asTransitionWeights(w)).data)),
        single: weights.singleTransitions.map((w, index) =>
          up(`cond.single-transition-w${index}`, packTransitionWeights(asTransitionWeights(w)).data)),
      };

      this.device.pushErrorScope("validation");
      const encoder = this.device.createCommandEncoder({ label: "af3-diffusion-conditioning" });
      const run = (label, pipeline, buffers, x, y = 1) => {
        const pass = encoder.beginComputePass({ label });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((allocation, binding) => ({
            binding, resource: { buffer: allocation.buffer },
          })),
        }));
        pass.dispatchWorkgroups(x, y);
        pass.end();
      };
      const spread = (groups) => [Math.min(groups, GRID_WIDTH), Math.ceil(groups / GRID_WIDTH)];
      // 🔴 THE PAIR CONDITIONING DOES NOT DEPEND ON THE NOISE LEVEL. It is the
      // trunk's pair and the relative encoding, projected and twice
      // transitioned - and none of that reads sigma, which enters only through
      // the Fourier embedding added to the SINGLE. A sampler calls this two
      // hundred times down one schedule and got the identical pair every time.
      // `reusePair` hands back the one a previous call already computed and
      // skips three of the five pipelines here; the head owns the caching,
      // because only the head knows the trunk has not changed underneath it.
      const reusePair = options.reusePair;
      const pairLinear = spread(Math.ceil(pairs / 64));
      if (reusePair === undefined) {
        run("pair-initial", compiled.pairInitial,
            [trunkPair, features, pairScale, pairProjection, sums, pair],
            pairLinear[0], pairLinear[1]);
      }
      run("single-initial", compiled.singleInitial,
          [trunkSingle, targetFeat, singleScale, singleProjection, noise, noiseWeights, single],
          Math.ceil(tokens / 64));

      const pairAdd = spread(Math.ceil(pairs * pairChannels / 64));
      const singleAdd = spread(Math.ceil(tokens * seqChannels / 64));
      // `transitions: 0` stops after the initial projections, which is how the
      // closed-form relative-encoding path is checked on its own.
      const transitionCount = options.transitions ?? 2;
      for (let index = 0; index < transitionCount; index += 1) {
        if (reusePair === undefined) {
          const perPair = spread(pairs);
          run(`pair-transition-${index}`, transitionPipelines.pair[index],
              [pair, transitionWeights.pair[index], pairScratch], perPair[0], perPair[1]);
          run(`pair-add-${index}`, compiled.addPair, [pair, pairScratch], pairAdd[0], pairAdd[1]);
        }
        run(`single-transition-${index}`, transitionPipelines.single[index],
            [single, transitionWeights.single[index], singleScratch], tokens);
        run(`single-add-${index}`, compiled.addSingle, [single, singleScratch],
            singleAdd[0], singleAdd[1]);
      }
      if (reusePair === undefined) {
        encoder.copyBufferToBuffer(pair.buffer, 0, readPair.buffer, 0, pairs * pairChannels * 4);
      }
      encoder.copyBufferToBuffer(single.buffer, 0, readSingle.buffer, 0, tokens * seqChannels * 4);

      const start = performance.now();
      this.device.queue.submit([encoder.finish()]);
      const error = await this.device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
      const read = async (allocation) => {
        await allocation.buffer.mapAsync(GPUMapMode.READ);
        const copy = new Float32Array(allocation.buffer.getMappedRange().slice(0));
        allocation.buffer.unmap();
        return copy;
      };
      return {
        pair: reusePair ?? await read(readPair), single: await read(readSingle),
        elapsedMilliseconds: performance.now() - start,
        memory: this.allocator.snapshot(),
      };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index].release();
    }
  }
}
