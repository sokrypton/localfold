/**
 * ESMFold2's distogram head, and the contact map a page draws from it.
 *
 *     logits = distogram_head(z + z.transpose(-2, -3))
 *
 * 🔴 THE HEAD SYMMETRISES, AND THAT IS PART OF THE HEAD. A distance is
 * symmetric and the trunk's pair is not, so `z` alone conforms in shape and
 * returns a plausible distogram - which is why tools/check-esmfold2-fold.js
 * runs the unsymmetrised form as a control and measures it at 5.29e-1 against
 * the head's own 4.82e-5.
 *
 * 🔴 AND THE BIN EDGES ARE NOT IN THIS CHECKPOINT'S CONFIG FOR THIS HEAD.
 * `distogram_bins: 128` is stated and no range is; the CONFIDENCE head - which
 * is disabled in this checkpoint and has its own 128 bins - carries
 * `min_dist: 2.0, max_dist: 52.0`. So `CONTACT_EDGES` is that range, borrowed,
 * and it is a borrowing rather than a fact: `distogramLogits` returns LOGITS
 * and every distance in this file is downstream of an assumption a caller can
 * replace. A contact map is a picture and is the right place to take it; a
 * reported distance would not be.
 */
import { GRID_WIDTH, LANES, createLinearShader, linearGrid } from "../esmc/block-webgpu.js";

/** Borrowed from the disabled confidence head's own 128 bins. See above. */
export const CONTACT_EDGES = { minimum: 2, maximum: 52 };
/** What counts as a contact, which is the usual 8 A between pseudo-betas. */
export const CONTACT_ANGSTROMS = 8;

/** `out[i, j] = pair[i, j] + pair[j, i]`, which is what the head is handed. */
export function createSymmetriseShader({ tokens, channels }) {
  const pairs = tokens * tokens;
  return `
@group(0) @binding(0) var<storage, read> pair: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(${LANES})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * ${GRID_WIDTH * LANES}u;
  if (i >= ${pairs * channels}u) { return; }
  let cell = i / ${channels}u;
  let c = i % ${channels}u;
  let row = cell / ${tokens}u;
  let column = cell % ${tokens}u;
  output[i] = pair[i] + pair[(column * ${tokens}u + row) * ${channels}u + c];
}`;
}

/**
 * softmax over the bins, then the mass below `CONTACT_ANGSTROMS`.
 *
 * 🔴 THE BIAS IS ADDED HERE RATHER THAN BY THE PROJECTION, because
 * `createLinearShader` has none and a whole extra pass to add 128 numbers to
 * every row would cost more than the branch. It is the same arithmetic either
 * way; what it must not be is forgotten, and a distogram missing its bias is
 * still a distribution because the softmax renormalises - it just puts its mass
 * in the wrong bins.
 */
export function createContactShader({ pairs, bins }, contactBins) {
  return `
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> contacts: array<f32>;

@compute @workgroup_size(${LANES})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let cell = id.x + id.y * ${GRID_WIDTH * LANES}u;
  if (cell >= ${pairs}u) { return; }
  let base = cell * ${bins}u;
  var largest = -3.0e38;
  for (var b = 0u; b < ${bins}u; b += 1u) {
    largest = max(largest, logits[base + b] + bias[b]);
  }
  var total = 0.0;
  var near = 0.0;
  for (var b = 0u; b < ${bins}u; b += 1u) {
    let weight = exp(logits[base + b] + bias[b] - largest);
    total += weight;
    if (b < ${contactBins}u) { near += weight; }
  }
  contacts[cell] = near / max(total, 1.0e-30);
}`;
}

/** How many of `bins` have their centre inside `CONTACT_ANGSTROMS`. */
export function contactBinCount(bins, edges = CONTACT_EDGES,
                                threshold = CONTACT_ANGSTROMS) {
  const width = (edges.maximum - edges.minimum) / bins;
  let count = 0;
  for (let bin = 0; bin < bins; bin += 1) {
    if (edges.minimum + (bin + 0.5) * width < threshold) count += 1;
  }
  return count;
}

/**
 * Build the contact map for a trunk's final pair, in row chunks.
 *
 * @param context { device, allocator, cache, submit }
 * @returns {Float32Array} one probability per token pair
 */
export async function encodeContactMap(context, { tokens, channels, bins, pair,
                                                  weights, bias, chunk = 8192 }) {
  const { allocator, cache, submit, device } = context;
  const pairs = tokens * tokens;
  const storage = GPUBufferUsage.STORAGE;
  const height = Math.min(chunk, pairs);
  const heights = [...new Set([height, pairs % height].filter((h) => h > 0))];
  const key = `esmfold2-disto:${tokens}:${channels}:${bins}`;
  const symmetrise = await cache.get(`${key}:sym`,
    createSymmetriseShader({ tokens, channels }));
  const project = {};
  const contact = {};
  const near = contactBinCount(bins);
  for (const rows of heights) {
    project[rows] = await cache.get(`${key}:project:${rows}`,
      createLinearShader({ rows, inner: channels, outer: bins }, false));
    contact[rows] = await cache.get(`${key}:contact:${rows}`,
      createContactShader({ pairs: rows, bins }, near));
  }

  const held = [];
  const keep = (allocation) => { held.push(allocation); return allocation; };
  const elementwise = (elements) => {
    const groups = Math.ceil(elements / LANES);
    return [Math.min(GRID_WIDTH, groups), Math.ceil(groups / GRID_WIDTH)];
  };
  try {
    const symmetric = keep(allocator.allocate("esmfold2.disto.sym",
      pairs * channels * 4, storage));
    const projection = keep(allocator.upload("w.esmfold2.disto", weights, storage));
    const biasBuffer = keep(allocator.upload("w.esmfold2.disto-bias", bias, storage));
    const logits = keep(allocator.allocate("esmfold2.disto.logits",
      height * bins * 4, storage));
    const contacts = keep(allocator.allocate("esmfold2.disto.contacts",
      pairs * 4, storage | GPUBufferUsage.COPY_SRC));
    await submit("esmfold2.distogram", [
      ["symmetrise", symmetrise, [pair, symmetric], ...elementwise(pairs * channels)],
    ]);
    for (let start = 0; start < pairs; start += height) {
      const rows = Math.min(height, pairs - start);
      await submit("esmfold2.distogram", [
        ["project", project[rows],
         [{ buffer: symmetric.buffer, byteOffset: start * channels * 4,
            byteSize: rows * channels * 4 }, projection, logits],
         ...linearGrid(rows, bins)],
        ["contacts", contact[rows],
         [logits, biasBuffer,
          { buffer: contacts.buffer, byteOffset: start * 4, byteSize: rows * 4 }],
         ...elementwise(rows)],
      ]);
    }
    const readback = keep(allocator.allocate("esmfold2.disto.readback", pairs * 4,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));
    const encoder = device.createCommandEncoder({ label: "esmfold2.disto.readback" });
    encoder.copyBufferToBuffer(contacts.buffer, 0, readback.buffer, 0, pairs * 4);
    device.queue.submit([encoder.finish()]);
    await readback.buffer.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(readback.buffer.getMappedRange().slice(0));
    readback.buffer.unmap();
    return out;
  } finally {
    for (let at = held.length - 1; at >= 0; at -= 1) held[at].release();
  }
}
