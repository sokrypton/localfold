import { ATTENTION_NORMALIZE_SHADER, createAttentionNormParameters } from "../evoformer/attention.js";
import {
  createTransitionShaders, TRANSITION_TILE_COLUMNS, TRANSITION_TILE_ROWS,
} from "../evoformer/transition.js";
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";
import { throwIfAborted, withAbort } from "../runtime/abort.js";

/**
 * @typedef {object} ConfidenceResult
 * @property {Float32Array} lddtLogits             per-residue pLDDT bin logits
 * @property {Float32Array} plddt                  per-residue confidence, 0-100
 * @property {number} meanPlddt                    the single number people quote
 * @property {Float32Array} paeLogits              per-pair PAE bin logits
 * @property {Float32Array} predictedAlignedError  length*length, in angstroms
 * @property {number} maxPredictedAlignedError
 * @property {number} ptm                          predicted TM-score
 */

const LINEAR_SHADER = createTransitionShaders({}, [])[1];
const RELU_SHADER = `
const GRID_WIDTH: u32 = 32768u;
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index < arrayLength(&source)) { output[index] = max(source[index], 0.0); }
}`;

function softmaxExpected(logits, rows, bins, centers) {
  const result = new Float32Array(rows);
  for (let row = 0; row < rows; row += 1) {
    const base = row * bins;
    let maximum = -Infinity;
    for (let bin = 0; bin < bins; bin += 1) maximum = Math.max(maximum, logits[base + bin]);
    let denominator = 0;
    let numerator = 0;
    for (let bin = 0; bin < bins; bin += 1) {
      const probability = Math.exp(logits[base + bin] - maximum);
      denominator += probability;
      numerator += probability * centers[bin];
    }
    result[row] = numerator / denominator;
  }
  return result;
}

function paeCenters(breaks) {
  const bins = breaks.length + 1;
  const centers = new Float32Array(bins);
  const step = breaks[1] - breaks[0];
  for (let bin = 0; bin < bins - 1; bin += 1) centers[bin] = breaks[bin] + step / 2;
  centers[bins - 1] = breaks[breaks.length - 1] + step / 2;
  return centers;
}

export function predictedTmScore(logits, length, breaks) {
  const bins = breaks.length + 1;
  if (logits.length !== length * length * bins) throw new RangeError("invalid PAE logits shape");
  const centers = paeCenters(breaks);
  const effectiveLength = Math.max(length, 19);
  const d0 = 1.24 * Math.cbrt(effectiveLength - 15) - 1.8;
  const tmPerBin = centers.map((center) => 1 / (1 + center * center / (d0 * d0)));
  let score = 0;
  for (let anchor = 0; anchor < length; anchor += 1) {
    let alignment = 0;
    for (let residue = 0; residue < length; residue += 1) {
      const base = (anchor * length + residue) * bins;
      let maximum = -Infinity;
      for (let bin = 0; bin < bins; bin += 1) maximum = Math.max(maximum, logits[base + bin]);
      let denominator = 0;
      let numerator = 0;
      for (let bin = 0; bin < bins; bin += 1) {
        const probability = Math.exp(logits[base + bin] - maximum);
        denominator += probability;
        numerator += probability * tmPerBin[bin];
      }
      alignment += numerator / denominator;
    }
    score = Math.max(score, alignment / length);
  }
  return score;
}

export class ConfidenceHeadsGpu {
  device;
  allocator;
  pipelines;

  constructor(device) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  async run(
    structureRepresentation,
    pairRepresentation,
    length,
    lddtWeights,
    paeWeights,
    breaks = Float32Array.from({ length: 63 }, (_, index) => index * 0.5),
    onStage,
    signal = undefined,
  ) {
    throwIfAborted(signal);
    const structureChannels = structureRepresentation.length / length;
    const pairChannels = pairRepresentation.length / (length * length);
    const hiddenChannels = lddtWeights.act0Bias.length;
    const lddtBins = lddtWeights.logitsBias.length;
    const paeBins = paeWeights.logitsBias.length;
    const tensors = [lddtWeights.normScale, lddtWeights.normOffset, lddtWeights.act0Weight,
      lddtWeights.act0Bias, lddtWeights.act1Weight, lddtWeights.act1Bias, lddtWeights.logitsWeight,
      lddtWeights.logitsBias, paeWeights.logitsWeight, paeWeights.logitsBias];
    const offsets = [];
    let packedSize = 0;
    for (const tensor of tensors) { offsets.push(packedSize); packedSize += tensor.length; }
    const packed = new Float32Array(packedSize);
    tensors.forEach((tensor, index) => packed.set(tensor, offsets[index]));
    const allocations = [];
    const keep = (value) => { allocations.push(value); return value; };
    const upload = (label, data, usage = GPUBufferUsage.STORAGE) =>
      keep(this.allocator.upload(label, data, usage));
    const allocate = (label, elements, usage = GPUBufferUsage.STORAGE) =>
      keep(this.allocator.allocate(label, elements * 4, usage));
    try {
      const [linear, normalize, relu] = await Promise.all([
        this.pipelines.get("confidence:linear", LINEAR_SHADER),
        this.pipelines.get("confidence:normalize", ATTENTION_NORMALIZE_SHADER),
        this.pipelines.get("confidence:relu", RELU_SHADER),
      ]);
      const structure = upload("confidence.structure", structureRepresentation);
      const pair = upload("confidence.pair", pairRepresentation);
      const weights = upload("confidence.weights", packed);
      const normParams = upload("confidence.norm-params", createAttentionNormParameters(
        length, structureChannels, offsets[0], offsets[1], false, 1, length, 1e-5,
      ), GPUBufferUsage.UNIFORM);
      const linearParams = (label, rows, inner, columns, weight, bias) =>
        upload(label, new Uint32Array([rows, inner, columns, weight, bias, 0, 0, 0]), GPUBufferUsage.UNIFORM);
      const params = [
        linearParams("confidence.act0-params", length, structureChannels, hiddenChannels, offsets[2], offsets[3]),
        linearParams("confidence.act1-params", length, hiddenChannels, hiddenChannels, offsets[4], offsets[5]),
        linearParams("confidence.lddt-params", length, hiddenChannels, lddtBins, offsets[6], offsets[7]),
        linearParams("confidence.pae-params", length * length, pairChannels, paeBins, offsets[8], offsets[9]),
      ];
      const normalized = allocate("confidence.normalized", length * structureChannels);
      const act0Raw = allocate("confidence.act0-raw", length * hiddenChannels);
      const act0 = allocate("confidence.act0", length * hiddenChannels);
      const act1Raw = allocate("confidence.act1-raw", length * hiddenChannels);
      const act1 = allocate("confidence.act1", length * hiddenChannels);
      const lddtLogits = allocate("confidence.lddt-logits", length * lddtBins,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const paeLogits = allocate("confidence.pae-logits", length * length * paeBins,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const encoder = this.device.createCommandEncoder({ label: "confidence-heads" });
      const dispatch = (pipeline, buffers, x, y = 1) => {
        const pass = encoder.beginComputePass(); pass.setPipeline(pipeline);
        pass.setBindGroup(0, this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer: buffer.buffer } })),
        }));
        pass.dispatchWorkgroups(x, y); pass.end();
      };
      const linearDispatch = (source, parameter,
        output, rows, columns) =>
        dispatch(linear, [source, weights, parameter, output],
          Math.ceil(columns / TRANSITION_TILE_COLUMNS), Math.ceil(rows / TRANSITION_TILE_ROWS));
      dispatch(normalize, [structure, weights, normParams, normalized], length);
      linearDispatch(normalized, params[0], act0Raw, length, hiddenChannels);
      dispatch(relu, [act0Raw, act0], Math.ceil(act0Raw.byteLength / 4 / 64));
      linearDispatch(act0, params[1], act1Raw, length, hiddenChannels);
      dispatch(relu, [act1Raw, act1], Math.ceil(act1Raw.byteLength / 4 / 64));
      linearDispatch(act1, params[2], lddtLogits, length, lddtBins);
      linearDispatch(pair, params[3], paeLogits, length * length, paeBins);
      const readbacks = [lddtLogits, paeLogits].map((source, index) => {
        const target = allocate(`confidence.readback-${index}`, source.byteLength / 4,
          GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
        encoder.copyBufferToBuffer(source.buffer, 0, target.buffer, 0, source.byteLength); return target;
      });
      this.device.queue.submit([encoder.finish()]);
      await withAbort(Promise.all(readbacks.map((buffer) => buffer.buffer.mapAsync(GPUMapMode.READ))), signal);
      throwIfAborted(signal);
      const values = readbacks.map((buffer) => {
        const value = new Float32Array(buffer.buffer.getMappedRange().slice(0)); buffer.buffer.unmap(); return value;
      });
      const lddtLogitValues = values[0];
      const paeLogitValues = values[1];
      onStage?.("reading confidence");
      const lddtCenters = Float32Array.from({ length: lddtBins }, (_, index) => (index + 0.5) / lddtBins * 100);
      const plddt = softmaxExpected(lddtLogitValues, length, lddtBins, lddtCenters);
      const centers = paeCenters(breaks);
      const predictedAlignedError = softmaxExpected(paeLogitValues, length * length, paeBins, centers);
      // 🔴 A YIELD BEFORE THE ONE PIECE OF REAL CPU WORK IN THE MODEL.
      // predictedTmScore is O(L^2 * bins) on the main thread - at L=221 that is
      // some millions of iterations with no await in them, so the page cannot
      // paint and the progress bar appears frozen at whatever it last said.
      // One macrotask costs about a millisecond and lets the bar show where it
      // actually is before the thread is taken.
      onStage?.("scoring");
      await new Promise((resolve) => setTimeout(resolve, 0));
      return {
        lddtLogits: lddtLogitValues, plddt,
        meanPlddt: plddt.reduce((sum, value) => sum + value, 0) / length,
        paeLogits: paeLogitValues, predictedAlignedError,
        maxPredictedAlignedError: centers[paeBins - 1],
        ptm: predictedTmScore(paeLogitValues, length, breaks),
      };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index] .release();
    }
  }
}
