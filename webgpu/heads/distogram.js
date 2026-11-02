/**
 * P(d < threshold) for every pair, on the device: distogramContactProbabilities
 * (shared/heads/distogram.js) without the pair ever leaving the GPU.
 *
 * 🔴 THE HOST VERSION WAS HALF OF AN AF2 FOLD ON THE PAGE. The page asked for
 * the pair representation back after every pass (`L^2 * 128` floats, 35 MB at
 * 261 residues) and ran `L^2 * 128 * 64` multiply-adds of JavaScript over it on
 * the main thread, between the fold's own steps: at 261 residues the page's
 * fold was 6.0 s with it and 2.6 s without, and on a Colab A100's 2.2 GHz Xeon
 * it was the difference between that runtime and this one. Here it is one
 * dispatch and `L^2` floats come back.
 *
 * One workgroup per pair (i, j), one lane per bin: the lane sums
 * `(pair[i][j] + pair[j][i]) . W[:, bin]` plus twice the bias - the host's
 * `half(i, j) + half(j, i)` - and the workgroup takes the softmax and the mass
 * of the bins whose upper edge is at or below the threshold, the host's rule.
 * The host keeps f64 accumulators and this keeps f32, so the two agree to
 * about 1e-6; the map is quantised to a byte for the heatmap either way.
 */
import { distogramBreaks, FIRST_BREAK, LAST_BREAK } from "../../shared/heads/distogram.js";

const WEIGHTS = new WeakMap();

function contactShader(length, channels, bins, counted) {
  // A tree over the bins, generated rather than looped so every barrier sits
  // in straight-line code the uniformity analysis accepts.
  const steps = [];
  for (let stride = bins / 2; stride >= 1; stride /= 2) {
    steps.push(`  if (lane < ${stride}u) {
    peak[lane] = max(peak[lane], peak[lane + ${stride}u]);
  }
  workgroupBarrier();`);
  }
  const sums = [];
  for (let stride = bins / 2; stride >= 1; stride /= 2) {
    sums.push(`  if (lane < ${stride}u) {
    total[lane] += total[lane + ${stride}u];
    under[lane] += under[lane + ${stride}u];
  }
  workgroupBarrier();`);
  }
  return `
const LENGTH: u32 = ${length}u;
const CHANNELS: u32 = ${channels}u;
const BINS: u32 = ${bins}u;
const COUNTED: u32 = ${counted}u;
const GRID_WIDTH: u32 = 32768u;

@group(0) @binding(0) var<storage, read> pair: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> contacts: array<f32>;

var<workgroup> summed: array<f32, CHANNELS>;
var<workgroup> peak: array<f32, BINS>;
var<workgroup> total: array<f32, BINS>;
var<workgroup> under: array<f32, BINS>;

@compute @workgroup_size(${bins})
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let row = group.x + group.y * GRID_WIDTH;
  if (row >= LENGTH * LENGTH) { return; }
  let i = row / LENGTH;
  let j = row % LENGTH;
  let forward = row * CHANNELS;
  let backward = (j * LENGTH + i) * CHANNELS;
  for (var c = lane; c < CHANNELS; c += BINS) {
    summed[c] = pair[forward + c] + pair[backward + c];
  }
  workgroupBarrier();
  var logit = 2.0 * bias[lane];
  for (var c = 0u; c < CHANNELS; c += 1u) {
    logit += summed[c] * weights[c * BINS + lane];
  }
  peak[lane] = logit;
  workgroupBarrier();
${steps.join("\n")}
  let weight = exp(logit - peak[0]);
  total[lane] = weight;
  under[lane] = select(0.0, weight, lane < COUNTED);
  workgroupBarrier();
${sums.join("\n")}
  if (lane == 0u) {
    contacts[row] = under[0] / total[0];
  }
}
`;
}

/**
 * Encode the contact map of `pair` into `encoder` and return the `L * L`
 * tensor it writes. The caller reads it back.
 *
 * @param {object} execution  a WebGpuExecution
 * @param {GPUCommandEncoder} encoder
 * @param {object} pair  the pair tensor, `L * L * channels` f32
 * @param {{halfLogitsWeights: ArrayLike<number>, halfLogitsBias: ArrayLike<number>,
 *          bins?: number, firstBreak?: number, lastBreak?: number}} head  the distogram head
 * @param {number} length  L
 * @param {{threshold?: number}} [options]
 */
export async function encodeContactProbabilities(execution, encoder, pair, head, length, options = {}) {
  const bins = head.bins ?? 64;
  const channels = head.halfLogitsWeights.length / bins;
  const threshold = options.threshold ?? 8;
  if (!Number.isInteger(channels) || pair.elements !== length * length * channels) {
    throw new RangeError(`pair is ${pair.elements} elements; expected ${length * length}`
      + ` x ${channels} channels`);
  }
  if ((pair.storage ?? "f32") !== "f32") throw new RangeError(`contact map takes an f32 pair; got ${pair.storage}`);
  if (bins & (bins - 1) || bins > 256) throw new RangeError(`contact map wants a power-of-two bin count; got ${bins}`);
  const breaks = distogramBreaks(head.firstBreak ?? FIRST_BREAK, head.lastBreak ?? LAST_BREAK, bins);
  let counted = 0;
  while (counted < breaks.length && breaks[counted] <= threshold) counted += 1;

  // The head's two tensors, uploaded once per head and kept: they are 32 KiB
  // and the same for every pass of every fold with these weights.
  let uploaded = WEIGHTS.get(head);
  if (uploaded === undefined || uploaded.device !== execution.device) {
    const make = (data) => {
      const buffer = execution.device.createBuffer({
        label: "distogram.contact-head", size: data.length * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      execution.device.queue.writeBuffer(buffer, 0, Float32Array.from(data));
      return { allocation: { buffer }, elements: data.length, storage: "f32" };
    };
    uploaded = { device: execution.device, weights: make(head.halfLogitsWeights), bias: make(head.halfLogitsBias) };
    WEIGHTS.set(head, uploaded);
  }
  const key = `distogram:contacts:${length}:${channels}:${bins}:${counted}`;
  const pipeline = await execution.shaderPipeline(key, () => contactShader(length, channels, bins, counted));
  const contacts = execution.allocate("distogram.contacts", length * length,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const [x, y] = execution.rowGrid(length * length);
  execution.dispatch(encoder, pipeline, [pair, uploaded.weights, uploaded.bias, contacts], x, y, 1,
    "distogram.contacts");
  return contacts;
}
