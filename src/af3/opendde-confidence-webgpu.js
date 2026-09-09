/**
 * OpenDDE's per-pair confidence readouts on the GPU: PAE and PDE.
 *
 * Each is the same three steps over `tokens^2` rows - a LayerNorm across the
 * pair channels, a projection to 64 distance-error bins, and a softmax against
 * those bins' centres - and PDE runs them on the pair SYMMETRISED, because a
 * distance error is symmetric and an aligned error is not.
 *
 * 🔴 THEY WERE 1.6 SECONDS OF A 13.5-SECOND OpenDDE FOLD, ON THE HOST. At 130
 * structural tokens the two projections are 16,900 x 128 x 64 multiply-
 * accumulates each - 138 million, twice - written as `linear()` in JavaScript
 * beside a comment calling the head's host half "the cheap half". That was true
 * of the per-token projections; it was never true of these. Measured inside a
 * 6MRR fold: PAE 777 ms, PDE 821 ms.
 *
 * 🔴 AND THE LOGITS NEVER LEAVE THE DEVICE. The softmax expectation is in the
 * same dispatch, so what comes back is one float a pair rather than 64 - 68 KiB
 * instead of 4.3 MB, and no host loop over the bins at all.
 *
 * A workgroup a row, a lane a bin: the pair row is read once per lane and the
 * 128 reads are a broadcast, the projection reads are coalesced across the
 * bins, and the softmax is two workgroup reductions.
 */
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { residentWeightBuffer } from "../runtime/resident.js";

const GRID_WIDTH = 32_768;
const LANES = 64;

export function createReadoutShader({ tokens, channels, bins, minBin, maxBin, symmetrise }) {
  const rows = tokens * tokens;
  return `
const TOKENS: u32 = ${tokens}u;
const ROWS: u32 = ${rows}u;
const CHANNELS: u32 = ${channels}u;
const BINS: u32 = ${bins}u;
const LANES: u32 = ${LANES}u;
const MIN_BIN: f32 = ${minBin};
const BIN_WIDTH: f32 = ${(maxBin - minBin) / bins};
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;
const EPSILON: f32 = 1.0e-5;

@group(0) @binding(0) var<storage, read> pair: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> offset: array<f32>;
@group(0) @binding(3) var<storage, read> projection: array<f32>;
@group(0) @binding(4) var<storage, read_write> expected: array<f32>;

var<workgroup> reduce: array<f32, LANES>;

fn source(row: u32, channel: u32) -> f32 {
${symmetrise
  ? `  let i = row / TOKENS;
  let j = row % TOKENS;
  return pair[row * CHANNELS + channel] + pair[(j * TOKENS + i) * CHANNELS + channel];`
  : `  return pair[row * CHANNELS + channel];`}
}

fn total_of(lane: u32, value: f32) -> f32 {
  reduce[lane] = value;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride >>= 1u) {
    if (lane < stride) { reduce[lane] += reduce[lane + stride]; }
    workgroupBarrier();
  }
  return reduce[0];
}

fn largest_of(lane: u32, value: f32) -> f32 {
  reduce[lane] = value;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride >>= 1u) {
    if (lane < stride) { reduce[lane] = max(reduce[lane], reduce[lane + stride]); }
    workgroupBarrier();
  }
  return reduce[0];
}

@compute @workgroup_size(LANES)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
  let row = group.x + group.y * GRID_WIDTH;
  if (row >= ROWS) { return; }
  let lane = local.x;

  var partial = 0.0;
  for (var c = lane; c < CHANNELS; c += LANES) { partial += source(row, c); }
  let mean = total_of(lane, partial) / f32(CHANNELS);
  workgroupBarrier();
  var squares = 0.0;
  for (var c = lane; c < CHANNELS; c += LANES) {
    let d = source(row, c) - mean;
    squares += d * d;
  }
  let inverse_std = inverseSqrt(total_of(lane, squares) / f32(CHANNELS) + EPSILON);
  workgroupBarrier();

  // The logits this lane owns, and the running softmax over them. A lane holds
  // ceil(BINS/LANES) of them; at OpenDDE's 64 bins that is exactly one, so the
  // two passes below read the same value twice rather than storing an array
  // this shader would have to index dynamically.
  var lane_largest = -3.4e38;
  for (var b = lane; b < BINS; b += LANES) {
    var logit = 0.0;
    for (var c = 0u; c < CHANNELS; c += 1u) {
      logit += ((source(row, c) - mean) * inverse_std * scale[c] + offset[c])
        * projection[c * BINS + b];
    }
    lane_largest = max(lane_largest, logit);
  }
  let peak = largest_of(lane, lane_largest);
  workgroupBarrier();

  var lane_total = 0.0;
  var lane_weighted = 0.0;
  for (var b = lane; b < BINS; b += LANES) {
    var logit = 0.0;
    for (var c = 0u; c < CHANNELS; c += 1u) {
      logit += ((source(row, c) - mean) * inverse_std * scale[c] + offset[c])
        * projection[c * BINS + b];
    }
    let value = exp(logit - peak);
    lane_total += value;
    lane_weighted += value * (MIN_BIN + BIN_WIDTH * (f32(b) + 0.5));
  }
  let denominator = total_of(lane, lane_total);
  workgroupBarrier();
  let numerator = total_of(lane, lane_weighted);
  if (lane == 0u) { expected[row] = numerator / denominator; }
}`;
}

/**
 * The pair this head starts from, built on the device.
 *
 * 🔴 THE PER-TOKEN HALF STAYS ON THE HOST AND THE PER-PAIR HALF DOES NOT. `s1`
 * and `s2` are `tokens x inputWidth x c` - 22 million multiply-accumulates at
 * 130 structural tokens, which really is cheap - but adding them into every
 * pair with the distance embedding is `tokens^2 x c` WRITES, 6.5 million of
 * them, and that measured 148 ms inside a fold.
 *
 * 🔴 AND ITS OUTPUT NEVER LEAVES THE DEVICE. The only reader is the four-block
 * pairformer stack, which took a host array and uploaded it; it takes the
 * buffer now, so a 26 MB readback and a 26 MB upload both stop happening.
 */
export function createPairInitShader({ tokens, channels, bins, binStart, binStep }) {
  return `
const TOKENS: u32 = ${tokens}u;
const PAIRS: u32 = ${tokens * tokens}u;
const CHANNELS: u32 = ${channels}u;
const BINS: i32 = ${bins};
const BIN_START: f32 = ${binStart};
const BIN_STEP: f32 = ${binStep};
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;

@group(0) @binding(0) var<storage, read> pair: array<f32>;
@group(0) @binding(1) var<storage, read> s1: array<f32>;
@group(0) @binding(2) var<storage, read> s2: array<f32>;
@group(0) @binding(3) var<storage, read> coordinates: array<f32>;
@group(0) @binding(4) var<storage, read> distance: array<f32>;
@group(0) @binding(5) var<storage, read> distanceRaw: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let slot = id.x + id.y * GRID_WIDTH * 64u;
  if (slot >= PAIRS * CHANNELS) { return; }
  let row = slot / CHANNELS;
  let d = slot % CHANNELS;
  let i = row / TOKENS;
  let j = row % TOKENS;
  var squared = 0.0;
  for (var axis = 0u; axis < 3u; axis += 1u) {
    let delta = coordinates[i * 3u + axis] - coordinates[j * 3u + axis];
    squared += delta * delta;
  }
  let length = sqrt(max(1.0e-10, squared));
  // The one-hot bin, which is a single row of the embedding - so this is a
  // gather rather than the 39-wide matrix multiply it is written as.
  var bin = i32(floor((length - BIN_START) / BIN_STEP));
  if (length < BIN_START) { bin = -1; }
  if (bin >= BINS) { bin = BINS - 1; }
  var value = pair[slot] + s1[j * CHANNELS + d] + s2[i * CHANNELS + d]
    + length * distanceRaw[d];
  if (bin >= 0) { value += distance[u32(bin) * CHANNELS + d]; }
  out[slot] = value;
}`;
}

/**
 * OpenDDE's two PER-ATOM readouts: pLDDT and experimentally-resolved.
 *
 * 🔴 THE WEIGHT IS CHOSEN BY THE ATOM'S SLOT, NOT SHARED. `plddt_weight` is
 * [24, c_s, 50]: an atom takes its TOKEN's single representation and the matrix
 * belonging to its dense slot within that token, which is how one head gives
 * the 24 atoms of a residue different answers from one vector. See
 * opendde-confidence.js, where reading it as [c_s, 50] is recorded as looking
 * plausible on a backbone and being wrong everywhere.
 *
 * A workgroup an atom, a lane a bin. The token's row is gathered inside the
 * kernel rather than materialised, the LayerNorm is two workgroup reductions,
 * and the softmax expectation is in the same dispatch - so what comes back is
 * one float an atom rather than fifty.
 */
export function createAtomReadoutShader({ atoms, channels, bins, slots, scoreBins }) {
  return `
const ATOMS: u32 = ${atoms}u;
const CHANNELS: u32 = ${channels}u;
const BINS: u32 = ${bins}u;
const SLOTS: u32 = ${slots}u;
const LANES: u32 = ${LANES}u;
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;
const EPSILON: f32 = 1.0e-5;

@group(0) @binding(0) var<storage, read> single: array<f32>;
@group(0) @binding(1) var<storage, read> placement: array<i32>;   // token, then slot
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read> offset: array<f32>;
@group(0) @binding(4) var<storage, read> table: array<f32>;
@group(0) @binding(5) var<storage, read_write> expected: array<f32>;

var<workgroup> reduce: array<f32, LANES>;

fn total_of(lane: u32, value: f32) -> f32 {
  reduce[lane] = value;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride >>= 1u) {
    if (lane < stride) { reduce[lane] += reduce[lane + stride]; }
    workgroupBarrier();
  }
  return reduce[0];
}

fn largest_of(lane: u32, value: f32) -> f32 {
  reduce[lane] = value;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride >>= 1u) {
    if (lane < stride) { reduce[lane] = max(reduce[lane], reduce[lane + stride]); }
    workgroupBarrier();
  }
  return reduce[0];
}

@compute @workgroup_size(LANES)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
  let atom = group.x + group.y * GRID_WIDTH;
  if (atom >= ATOMS) { return; }
  let lane = local.x;
  let base = u32(placement[atom]) * CHANNELS;
  let matrix = u32(placement[ATOMS + atom]) * CHANNELS * BINS;

  var partial = 0.0;
  for (var c = lane; c < CHANNELS; c += LANES) { partial += single[base + c]; }
  let mean = total_of(lane, partial) / f32(CHANNELS);
  workgroupBarrier();
  var squares = 0.0;
  for (var c = lane; c < CHANNELS; c += LANES) {
    let d = single[base + c] - mean;
    squares += d * d;
  }
  let inverse_std = inverseSqrt(total_of(lane, squares) / f32(CHANNELS) + EPSILON);
  workgroupBarrier();

  var lane_largest = -3.4e38;
  for (var b = lane; b < BINS; b += LANES) {
    var logit = 0.0;
    for (var c = 0u; c < CHANNELS; c += 1u) {
      logit += ((single[base + c] - mean) * inverse_std * scale[c] + offset[c])
        * table[matrix + c * BINS + b];
    }
    lane_largest = max(lane_largest, logit);
  }
  let peak = largest_of(lane, lane_largest);
  workgroupBarrier();

  var lane_total = 0.0;
  var lane_weighted = 0.0;
  for (var b = lane; b < BINS; b += LANES) {
    var logit = 0.0;
    for (var c = 0u; c < CHANNELS; c += 1u) {
      logit += ((single[base + c] - mean) * inverse_std * scale[c] + offset[c])
        * table[matrix + c * BINS + b];
    }
    let value = exp(logit - peak);
    lane_total += value;
    // 🔴 TWO REDUCTIONS, TWO MEANINGS. pLDDT is a softmax against BIN CENTRES
    // over [0, 1], scaled by 100 on the host; "resolved" is the probability of
    // the second of two bins, which is the same sum with the bin's INDEX as its
    // value. Reading either on the other's grid gives a number in the right
    // range and the wrong place.
    lane_weighted += value * ${scoreBins === "index"
      ? "f32(b)" : `(${(1 / bins).toExponential(9)} * (f32(b) + 0.5))`};
  }
  let denominator = total_of(lane, lane_total);
  workgroupBarrier();
  let numerator = total_of(lane, lane_weighted);
  if (lane == 0u) { expected[atom] = numerator / denominator; }
}`;
}

/**
 * pLDDT and "resolved" for every atom, in two dispatches and one submit.
 *
 * @param {{single: Float32Array, atoms: number, channels: number,
 *          atomToToken: Int32Array, atomToSlot: Int32Array}} input
 * @returns {Promise<{plddt: Float32Array, resolved: Float32Array}>}
 */
export async function openddeAtomReadouts(device, input, weights) {
  const { atoms, channels } = input;
  const pipelines = pipelineCacheForDevice(device);
  const allocator = new GpuBufferAllocator(device);
  const arms = [
    { name: "plddt", bins: weights.plddtBins, scoreBins: "centre",
      scale: weights.plddtLnScale, offset: weights.plddtLnOffset,
      table: weights.plddtWeight, factor: 100 },
    { name: "resolved", bins: weights.resolvedBins, scoreBins: "index",
      scale: weights.resolvedLnScale, offset: weights.resolvedLnOffset,
      table: weights.resolvedWeight, factor: 1 },
  ];
  const slots = weights.denseSlots;
  const compiled = [];
  for (const arm of arms) {
    compiled.push(await pipelines.get(
      `opendde-conf-atom:${atoms}:${channels}:${arm.bins}:${slots}:${arm.scoreBins}`,
      createAtomReadoutShader({ atoms, channels, bins: arm.bins, slots,
                                scoreBins: arm.scoreBins })));
  }

  // One buffer, token indices then slot indices, so the kernel takes one
  // binding for what is two per-atom tables.
  const layout = new Int32Array(2 * atoms);
  layout.set(input.atomToToken, 0);
  layout.set(input.atomToSlot, atoms);

  const storage = GPUBufferUsage.STORAGE;
  const allocations = [];
  const keep = (allocation) => { allocations.push(allocation); return allocation; };
  try {
    const single = keep(allocator.upload("opendde-conf.single", input.single, storage));
    const layoutBuffer = keep(allocator.upload("opendde-conf.layout", layout, storage));
    const resident = (label, build) =>
      ({ buffer: residentWeightBuffer(device, weights, label, build) });
    const outputs = arms.map((arm) => keep(allocator.allocate(
      `opendde-conf.${arm.name}`, atoms * 4, storage | GPUBufferUsage.COPY_SRC)));
    const readbacks = arms.map((arm) => keep(allocator.allocate(
      `opendde-conf.rb-${arm.name}`, atoms * 4,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST)));

    device.pushErrorScope("validation");
    const encoder = device.createCommandEncoder({ label: "opendde-confidence-atoms" });
    arms.forEach((arm, index) => {
      const buffers = [single, layoutBuffer,
        resident(`opendde-conf.${arm.name}-scale`, () => arm.scale),
        resident(`opendde-conf.${arm.name}-offset`, () => arm.offset),
        resident(`opendde-conf.${arm.name}-table`, () => arm.table),
        outputs[index]];
      const pass = encoder.beginComputePass({ label: `opendde-conf.${arm.name}` });
      pass.setPipeline(compiled[index]);
      pass.setBindGroup(0, device.createBindGroup({
        layout: compiled[index].getBindGroupLayout(0),
        entries: buffers.map((allocation, binding) => ({
          binding, resource: { buffer: allocation.buffer },
        })),
      }));
      pass.dispatchWorkgroups(Math.min(atoms, GRID_WIDTH), Math.ceil(atoms / GRID_WIDTH));
      pass.end();
      encoder.copyBufferToBuffer(outputs[index].buffer, 0, readbacks[index].buffer, 0, atoms * 4);
    });
    device.queue.submit([encoder.finish()]);
    const error = await device.popErrorScope();
    if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
    const read = async (allocation) => {
      await allocation.buffer.mapAsync(GPUMapMode.READ);
      const copy = new Float32Array(allocation.buffer.getMappedRange().slice(0));
      allocation.buffer.unmap();
      return copy;
    };
    const plddt = await read(readbacks[0]);
    for (let index = 0; index < plddt.length; index += 1) plddt[index] *= 100;
    return { plddt, resolved: await read(readbacks[1]) };
  } finally {
    for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index].release();
  }
}

/**
 * PAE and PDE from one refined pair, in two dispatches and one submit.
 *
 * @param {GPUDevice} device
 * @param {{pair: Float32Array, tokens: number, channels: number}} input
 * @param {object} weights the head's, for the names see openddeConfidenceWeights
 * @returns {Promise<{pae: Float32Array, pde: Float32Array}>}
 */
export async function openddePairReadouts(device, input, weights) {
  const { tokens, channels } = input;
  const rows = tokens * tokens;
  const pipelines = pipelineCacheForDevice(device);
  const allocator = new GpuBufferAllocator(device);
  const arms = [
    { name: "pae", symmetrise: false, bins: weights.paeBins,
      scale: weights.paeLnScale, offset: weights.paeLnOffset, projection: weights.pae },
    { name: "pde", symmetrise: true, bins: weights.pdeBins,
      scale: weights.pdeLnScale, offset: weights.pdeLnOffset, projection: weights.pde },
  ];
  const compiled = [];
  for (const arm of arms) {
    compiled.push(await pipelines.get(
      `opendde-conf-readout:${tokens}:${channels}:${arm.bins}:${arm.symmetrise}`,
      createReadoutShader({ tokens, channels, bins: arm.bins,
                            minBin: 0, maxBin: 32, symmetrise: arm.symmetrise })));
  }

  const storage = GPUBufferUsage.STORAGE;
  const allocations = [];
  const keep = (allocation) => { allocations.push(allocation); return allocation; };
  try {
    // 🔴 THE CALLER'S BUFFER WHEN IT HAS ONE; see openddeConfidence.
    const pair = input.pairBuffer !== undefined
      ? { buffer: input.pairBuffer }
      : keep(allocator.upload("opendde-conf.pair", input.pair, storage));
    const resident = (label, build) =>
      ({ buffer: residentWeightBuffer(device, weights, label, build) });
    const outputs = arms.map((arm) => keep(allocator.allocate(
      `opendde-conf.${arm.name}`, rows * 4, storage | GPUBufferUsage.COPY_SRC)));
    const readbacks = arms.map((arm) => keep(allocator.allocate(
      `opendde-conf.rb-${arm.name}`, rows * 4,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST)));

    device.pushErrorScope("validation");
    const encoder = device.createCommandEncoder({ label: "opendde-confidence-readouts" });
    arms.forEach((arm, index) => {
      const buffers = [pair,
        resident(`opendde-conf.${arm.name}-scale`, () => arm.scale),
        resident(`opendde-conf.${arm.name}-offset`, () => arm.offset),
        resident(`opendde-conf.${arm.name}-projection`, () => arm.projection),
        outputs[index]];
      const pass = encoder.beginComputePass({ label: `opendde-conf.${arm.name}` });
      pass.setPipeline(compiled[index]);
      pass.setBindGroup(0, device.createBindGroup({
        layout: compiled[index].getBindGroupLayout(0),
        entries: buffers.map((allocation, binding) => ({
          binding, resource: { buffer: allocation.buffer },
        })),
      }));
      pass.dispatchWorkgroups(Math.min(rows, GRID_WIDTH), Math.ceil(rows / GRID_WIDTH));
      pass.end();
      encoder.copyBufferToBuffer(outputs[index].buffer, 0, readbacks[index].buffer, 0, rows * 4);
    });
    device.queue.submit([encoder.finish()]);
    const error = await device.popErrorScope();
    if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
    const read = async (allocation) => {
      await allocation.buffer.mapAsync(GPUMapMode.READ);
      const copy = new Float32Array(allocation.buffer.getMappedRange().slice(0));
      allocation.buffer.unmap();
      return copy;
    };
    return { pae: await read(readbacks[0]), pde: await read(readbacks[1]) };
  } finally {
    for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index].release();
  }
}

/**
 * `confidencePairInit` on the device, leaving its result there.
 *
 * @returns {Promise<{allocation: object, release: () => void}>} the caller owns
 *   the buffer and must release it once the stack that reads it has run.
 */
export async function openddePairInit(device, input, weights, allocator) {
  const { tokens, channels } = input;
  const pairs = tokens * tokens;
  const pipelines = pipelineCacheForDevice(device);
  const pipeline = await pipelines.get(
    `opendde-conf-pairinit:${tokens}:${channels}:${weights.distanceBins}`,
    createPairInitShader({ tokens, channels, bins: weights.distanceBins,
                           binStart: input.binStart, binStep: input.binStep }));

  const storage = GPUBufferUsage.STORAGE;
  const scratch = [];
  const up = (label, data) => {
    const allocation = allocator.upload(label, data, storage);
    scratch.push(allocation);
    return allocation;
  };
  const out = allocator.allocate("opendde-conf.pair-init",
    pairs * channels * 4, storage | GPUBufferUsage.COPY_SRC);
  const resident = (label, build) =>
    ({ buffer: residentWeightBuffer(device, weights, label, build) });
  const buffers = [
    input.pairBuffer !== undefined
      ? { buffer: input.pairBuffer } : up("opendde-conf.trunk-pair", input.pair),
    up("opendde-conf.s1", input.s1),
    up("opendde-conf.s2", input.s2),
    up("opendde-conf.coordinates", input.coordinates),
    resident("opendde-conf.distance", () => weights.distance),
    resident("opendde-conf.distance-raw", () => weights.distanceRaw),
    out,
  ];
  device.pushErrorScope("validation");
  const encoder = device.createCommandEncoder({ label: "opendde-confidence-pair-init" });
  const pass = encoder.beginComputePass({ label: "opendde-conf.pair-init" });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((allocation, binding) => ({
      binding, resource: { buffer: allocation.buffer },
    })),
  }));
  const slots = Math.ceil(pairs * channels / 64);
  pass.dispatchWorkgroups(Math.min(slots, GRID_WIDTH), Math.ceil(slots / GRID_WIDTH));
  pass.end();
  device.queue.submit([encoder.finish()]);
  const error = await device.popErrorScope();
  if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
  // 🔴 THE UPLOADS GO NOW AND THE OUTPUT DOES NOT. Queue ordering means the
  // stack's reads happen after this dispatch, so the sources can be recycled;
  // the destination is what the stack binds.
  for (const allocation of scratch) allocation.release();
  return { allocation: out, release: () => out.release() };
}
