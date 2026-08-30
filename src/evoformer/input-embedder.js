import { ATTENTION_NORMALIZE_SHADER, createAttentionNormParameters } from "./attention.js";
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";
import { WebGpuExecution } from "../runtime/execution.js";

const GRID_WIDTH = 32_768;
const ceilDivide = (value, divisor) => Math.ceil(value / divisor);

function packWeights(input) {
  const w = input.weights;
  const tensors = [
    w.preprocess1dWeight, w.preprocess1dBias, w.preprocessMsaWeight, w.preprocessMsaBias,
    w.leftSingleWeight, w.leftSingleBias, w.rightSingleWeight, w.rightSingleBias,
    w.previousPositionWeight, w.previousPositionBias,
    w.previousMsaNormScale, w.previousMsaNormOffset,
    w.previousPairNormScale, w.previousPairNormOffset,
    w.relativePositionWeight, w.relativePositionBias,
    w.extraMsaWeight, w.extraMsaBias,
  ];
  const offsets = [];
  let size = 0;
  for (const tensor of tensors) { offsets.push(size); size += tensor.length; }
  const data = new Float32Array(size);
  tensors.forEach((tensor, index) => data.set(tensor, offsets[index]));
  return { data, offsets };
}

function parameters(input, offsets) {
  const buffer = new ArrayBuffer(128);
  const view = new DataView(buffer);
  const dimensions = [
    input.length, input.msaSequences, input.extraSequences, input.targetChannels,
    input.msaFeatureChannels, input.msaChannels, input.pairChannels, input.extraMsaChannels,
    25, 15, 32,
  ];
  [...dimensions, ...offsets].forEach((value, index) => view.setUint32(index * 4, value, true));
  view.setFloat32(116, 3.25, true);
  view.setFloat32(120, 20.75, true);
  return new Uint8Array(buffer);
}

const COMMON = `
struct Parameters {
  length: u32, msa_sequences: u32, extra_sequences: u32, target_channels: u32,
  msa_feature_channels: u32, msa_channels: u32, pair_channels: u32, extra_channels: u32,
  extra_feature_channels: u32, dgram_bins: u32, max_relative: u32,
  preprocess_1d_weight: u32, preprocess_1d_bias: u32,
  preprocess_msa_weight: u32, preprocess_msa_bias: u32,
  left_weight: u32, left_bias: u32, right_weight: u32, right_bias: u32,
  previous_position_weight: u32, previous_position_bias: u32,
  previous_msa_scale: u32, previous_msa_offset: u32,
  previous_pair_scale: u32, previous_pair_offset: u32,
  relative_weight: u32, relative_bias: u32,
  extra_weight: u32, extra_bias: u32,
  min_bin: f32, max_bin: f32, padding: u32,
};
const GRID_WIDTH: u32 = 32768u;
`;

const MSA_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> target_features: array<f32>;
@group(0) @binding(1) var<storage, read> msa_features: array<f32>;
@group(0) @binding(2) var<storage, read> previous_msa: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<uniform> p: Parameters;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.msa_sequences * p.length * p.msa_channels) { return; }
  let channel = index % p.msa_channels;
  let row = index / p.msa_channels;
  let residue = row % p.length;
  let sequence = row / p.length;
  var result = weights[p.preprocess_1d_bias + channel] + weights[p.preprocess_msa_bias + channel];
  for (var c = 0u; c < p.target_channels; c += 1u) {
    result += target_features[residue * p.target_channels + c]
      * weights[p.preprocess_1d_weight + c * p.msa_channels + channel];
  }
  for (var c = 0u; c < p.msa_feature_channels; c += 1u) {
    result += msa_features[row * p.msa_feature_channels + c]
      * weights[p.preprocess_msa_weight + c * p.msa_channels + channel];
  }
  if (sequence == 0u) { result += previous_msa[residue * p.msa_channels + channel]; }
  output[index] = result;
}`;

const EXTRA_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> extra_msa: array<f32>;
@group(0) @binding(1) var<storage, read> has_deletion: array<f32>;
@group(0) @binding(2) var<storage, read> deletion_value: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<uniform> p: Parameters;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.extra_sequences * p.length * p.extra_channels) { return; }
  let channel = index % p.extra_channels;
  let row = index / p.extra_channels;
  let code = u32(extra_msa[row]);
  var result = weights[p.extra_bias + channel];
  result += weights[p.extra_weight + code * p.extra_channels + channel];
  result += has_deletion[row] * weights[p.extra_weight + 23u * p.extra_channels + channel];
  result += deletion_value[row] * weights[p.extra_weight + 24u * p.extra_channels + channel];
  output[index] = result;
}`;

const PAIR_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> target_features: array<f32>;
@group(0) @binding(1) var<storage, read> previous_pair: array<f32>;
@group(0) @binding(2) var<storage, read> previous_positions: array<f32>;
@group(0) @binding(3) var<storage, read> aatype: array<f32>;
@group(0) @binding(4) var<storage, read> residue_index: array<f32>;
@group(0) @binding(5) var<storage, read> weights: array<f32>;
@group(0) @binding(6) var<uniform> p: Parameters;
@group(0) @binding(7) var<storage, read_write> output: array<f32>;

fn pseudo_beta_coordinate(residue: u32, coordinate: u32) -> f32 {
  let atom = select(3u, 1u, u32(aatype[residue]) == 7u);
  return previous_positions[(residue * 37u + atom) * 3u + coordinate];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.length * p.length * p.pair_channels) { return; }
  let channel = index % p.pair_channels;
  let pair = index / p.pair_channels;
  let i = pair / p.length;
  let j = pair % p.length;
  var result = weights[p.left_bias + channel] + weights[p.right_bias + channel];
  for (var c = 0u; c < p.target_channels; c += 1u) {
    result += target_features[i * p.target_channels + c] * weights[p.left_weight + c * p.pair_channels + channel];
    result += target_features[j * p.target_channels + c] * weights[p.right_weight + c * p.pair_channels + channel];
  }
  result += weights[p.previous_position_bias + channel];
  var distance_squared = 0.0;
  for (var coordinate = 0u; coordinate < 3u; coordinate += 1u) {
    let delta = pseudo_beta_coordinate(i, coordinate) - pseudo_beta_coordinate(j, coordinate);
    distance_squared += delta * delta;
  }
  let bin_width = (p.max_bin - p.min_bin) / f32(p.dgram_bins - 1u);
  for (var bin = 0u; bin < p.dgram_bins; bin += 1u) {
    let lower = p.min_bin + f32(bin) * bin_width;
    let upper = select(p.max_bin + bin_width, p.min_bin + f32(bin + 1u) * bin_width, bin + 1u < p.dgram_bins);
    if (distance_squared > lower * lower && (bin + 1u == p.dgram_bins || distance_squared < upper * upper)) {
      result += weights[p.previous_position_weight + bin * p.pair_channels + channel];
    }
  }
  result += previous_pair[index];
  let raw_offset = i32(residue_index[i]) - i32(residue_index[j]) + i32(p.max_relative);
  let relative = u32(clamp(raw_offset, 0, i32(2u * p.max_relative)));
  result += weights[p.relative_bias + channel]
    + weights[p.relative_weight + relative * p.pair_channels + channel];
  output[index] = result;
}`;

/** Encode the input embedding into an existing execution without crossing the CPU boundary. */
export async function encodeInputEmbedder(
  execution,
  encoder,
  input,
  previousMsa,
  previousPair,
  previousPositions,
) {
  const expectedPreviousMsa = input.length * input.msaChannels;
  const expectedPreviousPair = input.length * input.length * input.pairChannels;
  const expectedPreviousPositions = input.length * 37 * 3;
  if (previousMsa.elements < expectedPreviousMsa || previousPair.elements !== expectedPreviousPair
      || previousPositions.elements !== expectedPreviousPositions) {
    throw new RangeError("resident recycle tensor shape mismatch");
  }
  const packed = packWeights(input);
  const [normalize, msaPipeline, pairPipeline, extraPipeline] = await Promise.all([
    execution.pipelines.get("embed:normalize", ATTENTION_NORMALIZE_SHADER),
    execution.pipelines.get("embed:msa", MSA_SHADER),
    execution.pipelines.get("embed:pair", PAIR_SHADER),
    execution.pipelines.get("embed:extra", EXTRA_SHADER),
  ]);
  const temporaries = [];
  const temporaryUpload = (label, data, usage = GPUBufferUsage.STORAGE) => {
    const tensor = execution.upload(label, data, usage); temporaries.push(tensor); return tensor;
  };
  const temporaryAllocate = (label, elements) => {
    const tensor = execution.allocate(label, elements); temporaries.push(tensor); return tensor;
  };
  const target = temporaryUpload("embed.target", input.targetFeatures);
  const msaFeatures = temporaryUpload("embed.msa-features", input.msaFeatures);
  const extraMsaInput = temporaryUpload("embed.extra-codes", input.extraMsa);
  const hasDeletion = temporaryUpload("embed.extra-has-deletion", input.extraHasDeletion);
  const deletionValue = temporaryUpload("embed.extra-deletion-value", input.extraDeletionValue);
  const residueIndex = temporaryUpload("embed.residue-index", input.residueIndex);
  const aatype = temporaryUpload("embed.aatype", input.aatype);
  const weights = temporaryUpload("embed.weights", packed.data);
  const params = temporaryUpload("embed.parameters", parameters(input, packed.offsets), GPUBufferUsage.UNIFORM);
  const previousMsaNormParams = temporaryUpload("embed.previous-msa-norm-params", createAttentionNormParameters(
    input.length, input.msaChannels, packed.offsets[10], packed.offsets[11],
    false, 1, input.length, 1e-5,
  ), GPUBufferUsage.UNIFORM);
  const previousPairNormParams = temporaryUpload("embed.previous-pair-norm-params", createAttentionNormParameters(
    input.length * input.length, input.pairChannels, packed.offsets[12], packed.offsets[13],
    false, 1, input.length * input.length, 1e-5,
  ), GPUBufferUsage.UNIFORM);
  const previousMsaNormalized = temporaryAllocate("embed.previous-msa-normalized", expectedPreviousMsa);
  const previousPairNormalized = temporaryAllocate("embed.previous-pair-normalized", expectedPreviousPair);
  const msaElements = input.msaSequences * input.length * input.msaChannels;
  const pairElements = expectedPreviousPair;
  const extraElements = input.extraSequences * input.length * input.extraMsaChannels;
  const msa = execution.allocate("embed.msa", msaElements, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const pair = execution.allocate("embed.pair", pairElements, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const extra = execution.allocate("embed.extra", extraElements);
  const msaNormGrid = execution.rowGrid(input.length);
  execution.dispatch(encoder, normalize, [previousMsa, weights, previousMsaNormParams, previousMsaNormalized],
    msaNormGrid[0], msaNormGrid[1]);
  // ...L*L rows, which passes the 65535 limit at L=256
  const pairNormGrid = execution.rowGrid(input.length * input.length);
  execution.dispatch(encoder, normalize, [previousPair, weights, previousPairNormParams, previousPairNormalized],
    pairNormGrid[0], pairNormGrid[1]);
  let grid = execution.linearGrid(msaElements);
  execution.dispatch(encoder, msaPipeline, [target, msaFeatures, previousMsaNormalized, weights, params, msa],
    grid[0], grid[1]);
  grid = execution.linearGrid(pairElements);
  execution.dispatch(encoder, pairPipeline,
    [target, previousPairNormalized, previousPositions, aatype, residueIndex, weights, params, pair], grid[0], grid[1]);
  grid = execution.linearGrid(extraElements);
  execution.dispatch(encoder, extraPipeline, [extraMsaInput, hasDeletion, deletionValue, weights, params, extra],
    grid[0], grid[1]);
  return { msa, pairWithoutTemplates: pair, extraMsa: extra, temporaries };
}

export class InputEmbedderGpu {
  device;
  allocator;
  pipelines;
  constructor(device) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  async run(input) {
    const packed = packWeights(input);
    const [normalize, msaPipeline, pairPipeline, extraPipeline] = await Promise.all([
      this.pipelines.get("embed:normalize", ATTENTION_NORMALIZE_SHADER),
      this.pipelines.get("embed:msa", MSA_SHADER),
      this.pipelines.get("embed:pair", PAIR_SHADER),
      this.pipelines.get("embed:extra", EXTRA_SHADER),
    ]);
    const allocations = [];
    const keep = (value) => { allocations.push(value); return value; };
    const upload = (label, value, usage = GPUBufferUsage.STORAGE) =>
      keep(this.allocator.upload(label, value, usage));
    const allocate = (label, elements, usage = GPUBufferUsage.STORAGE) =>
      keep(this.allocator.allocate(label, elements * 4, usage));
    const grid = (elements) => {
      const groups = ceilDivide(elements, 64);
      return [Math.min(groups, GRID_WIDTH), ceilDivide(groups, GRID_WIDTH)];
    };
    const rowGrid = (rows) => [Math.min(rows, GRID_WIDTH), ceilDivide(rows, GRID_WIDTH)];
    try {
      const target = upload("embed.target", input.targetFeatures);
      const msaFeatures = upload("embed.msa-features", input.msaFeatures);
      const extraMsaInput = upload("embed.extra-codes", input.extraMsa);
      const hasDeletion = upload("embed.extra-has-deletion", input.extraHasDeletion);
      const deletionValue = upload("embed.extra-deletion-value", input.extraDeletionValue);
      const residueIndex = upload("embed.residue-index", input.residueIndex);
      const aatype = upload("embed.aatype", input.aatype);
      const previousMsa = upload("embed.previous-msa", input.previousMsaFirstRow);
      const previousPair = upload("embed.previous-pair", input.previousPair);
      const previousPositions = upload("embed.previous-positions", input.previousPositions);
      const weights = upload("embed.weights", packed.data);
      const params = upload("embed.parameters", parameters(input, packed.offsets), GPUBufferUsage.UNIFORM);
      const previousMsaNormParams = upload("embed.previous-msa-norm-params", createAttentionNormParameters(
        input.length, input.msaChannels, packed.offsets[10], packed.offsets[11],
        false, 1, input.length, 1e-5,
      ), GPUBufferUsage.UNIFORM);
      const previousPairNormParams = upload("embed.previous-pair-norm-params", createAttentionNormParameters(
        input.length * input.length, input.pairChannels, packed.offsets[12], packed.offsets[13],
        false, 1, input.length * input.length, 1e-5,
      ), GPUBufferUsage.UNIFORM);
      const previousMsaNormalized = allocate("embed.previous-msa-normalized", input.length * input.msaChannels);
      const previousPairNormalized = allocate(
        "embed.previous-pair-normalized", input.length * input.length * input.pairChannels,
      );
      const msaElements = input.msaSequences * input.length * input.msaChannels;
      const pairElements = input.length * input.length * input.pairChannels;
      const extraElements = input.extraSequences * input.length * input.extraMsaChannels;
      const msa = allocate("embed.msa", msaElements, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      // 🔴 THE PAIR TRACK MAY BELONG TO THE CALLER. It is the largest tensor in
      // the model - L*L*128 floats, 25 MiB at 221 residues - and the trunk that
      // consumes it next can hold it on the device for the whole pass rather
      // than take it back off a bus it is about to put it on again. When a
      // buffer is handed in, the pair is written straight into it and NOT
      // released here: the caller allocated it and the caller owns it.
      const pair = input.pairBuffer ?? allocate(
        "embed.pair", pairElements, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      );
      const extra = allocate("embed.extra", extraElements, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const msaReadback = allocate("embed.msa-readback", msaElements, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      const pairReadback = input.keepPair === true ? undefined
        : allocate("embed.pair-readback", pairElements, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      const extraReadback = allocate("embed.extra-readback", extraElements, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      const encoder = this.device.createCommandEncoder({ label: "input-embedder" });
      this.device.pushErrorScope("validation");
      const pass = (pipeline, buffers, x, y = 1) => {
        const compute = encoder.beginComputePass();
        compute.setPipeline(pipeline);
        compute.setBindGroup(0, this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer: buffer.buffer } })),
        }));
        compute.dispatchWorkgroups(x, y);
        compute.end();
      };
      pass(normalize, [previousMsa, weights, previousMsaNormParams, previousMsaNormalized], input.length);
      // 🔴 ONE WORKGROUP PER PAIR ROW IS L*L OF THEM, and a dispatch is 65535
      // wide at most - so this ran off the end at 256 residues. The shader has
      // always read its row as group.x + group.y * GRID_WIDTH; the caller was
      // what lacked the fold. The sibling encoder in this file folds it.
      const pairRows = rowGrid(input.length * input.length);
      pass(normalize, [previousPair, weights, previousPairNormParams, previousPairNormalized],
        pairRows[0], pairRows[1]);
      let dispatch = grid(msaElements);
      pass(msaPipeline, [target, msaFeatures, previousMsaNormalized, weights, params, msa], dispatch[0], dispatch[1]);
      dispatch = grid(pairElements);
      pass(pairPipeline, [target, previousPairNormalized, previousPositions, aatype, residueIndex, weights, params, pair],
        dispatch[0], dispatch[1]);
      dispatch = grid(extraElements);
      pass(extraPipeline, [extraMsaInput, hasDeletion, deletionValue, weights, params, extra], dispatch[0], dispatch[1]);
      encoder.copyBufferToBuffer(msa.buffer, 0, msaReadback.buffer, 0, msaElements * 4);
      if (pairReadback !== undefined) {
        encoder.copyBufferToBuffer(pair.buffer, 0, pairReadback.buffer, 0, pairElements * 4);
      }
      encoder.copyBufferToBuffer(extra.buffer, 0, extraReadback.buffer, 0, extraElements * 4);
      const start = performance.now();
      this.device.queue.submit([encoder.finish()]);
      const validationError = await this.device.popErrorScope();
      if (validationError !== null) throw new Error(`WebGPU validation failed: ${validationError.message}`);
      const mapped = [msaReadback, extraReadback, ...(pairReadback === undefined ? [] : [pairReadback])];
      await Promise.all(mapped.map((buffer) => buffer.buffer.mapAsync(GPUMapMode.READ)));
      const msaOutput = new Float32Array(msaReadback.buffer.getMappedRange().slice(0));
      const extraOutput = new Float32Array(extraReadback.buffer.getMappedRange().slice(0));
      const pairOutput = pairReadback === undefined ? undefined
        : new Float32Array(pairReadback.buffer.getMappedRange().slice(0));
      for (const buffer of mapped) buffer.buffer.unmap();
      return {
        msa: msaOutput, pairWithoutTemplates: pairOutput, extraMsa: extraOutput,
        elapsedMilliseconds: performance.now() - start, memory: this.allocator.snapshot(),
      };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index] .release();
    }
  }
}
