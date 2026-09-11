/**
 * The alignment's nearest-centre search, on the GPU.
 *
 * 🔴 IT IS 59% OF PREPARING AN ALIGNMENT AND IT IS SERIAL WITH THE FOLD.
 * `featureStats` at 825 residues, 512 clusters, 1024 extras and two recycles:
 * 640 ms of a 1072 ms featurisation, itself 1.08 s of a 23.9 s fold that the
 * GPU sits out. The next term is a fifth of it. The host loop is already
 * word-parallel - four residues to a 32-bit compare, 6.8x the scalar loop it
 * replaced - and it is still the largest single piece of main-thread work in
 * an AF2 fold.
 *
 * It is a reduction over residues and an argmax over centres, which is a
 * kernel: one workgroup an extra row, the centres split across 64 lanes, then
 * a tree join.
 *
 * 🔴 AND AGREEING EXACTLY IS THE POINT, NOT AGREEING CLOSELY. The assignment
 * decides the cluster profile and so the prediction, so this is held to the
 * host loop's answer element for element, ties included:
 *
 *   - The host keeps the FIRST centre at an equal score. The join packs
 *     `(score << 16) | (0xffff - centre)` into one u32 and takes the MAX, so a
 *     tie is broken towards the lower index by construction rather than by a
 *     comparison somebody has to keep correct. `score` is at most the query's
 *     length and `centre` at most the cluster count, so the pack is exact for
 *     any alignment this repository can hold.
 *   - The padding and the "a code above 20 never agrees" rule come from
 *     `paddedCodeWords` in a3m-features.js, which the host path calls too. A
 *     second copy of that padding is a second chance to get 255-against-254
 *     wrong.
 *   - The zero-byte detect is the exact form, `~(((x & 0x7f7f7f7f) +
 *     0x7f7f7f7f) | x) & 0x80808080`, and NOT the subtraction one - which
 *     marks a zero byte's neighbour as well, so counting the marks overcounts.
 *     See the note on the host loop; it moved 1024 assignments' checksum.
 *
 * 🔴 EVERY RECYCLE IN ONE SUBMIT. A fold plans all of its recycles before it
 * needs any assignment, so the four searches go up as one dispatch and come
 * back in one map. At 825 residues that is four round trips saved out of four.
 */

const WORKGROUP = 64;

const SHADER = `
struct Parameters {
  rows: u32,
  centres: u32,
  words: u32,
  row_offset: u32,
  centre_offset: u32,
  output_offset: u32,
  pad_0: u32,
  pad_1: u32,
};
@group(0) @binding(0) var<storage, read> centre_words: array<u32>;
@group(0) @binding(1) var<storage, read> extra_words: array<u32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<storage, read_write> assignments: array<u32>;

var<workgroup> best: array<u32, ${WORKGROUP}>;

// The count of agreeing bytes in one 32-bit lane of the difference.
fn agreeing(difference: u32) -> u32 {
  let zeros = ~(((difference & 0x7f7f7f7fu) + 0x7f7f7f7fu) | difference) & 0x80808080u;
  return ((zeros >> 7u) & 1u) + ((zeros >> 15u) & 1u)
    + ((zeros >> 23u) & 1u) + ((zeros >> 31u) & 1u);
}

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
  let row = group.x;
  if (row >= p.rows) { return; }
  let extra_base = p.row_offset + row * p.words;
  // 🔴 THE EMPTY LANE'S CANDIDATE IS ZERO, WHICH LOSES TO EVERY REAL ONE. A
  // real candidate carries 0xffff - centre in its low half, so its packed key
  // is at least 0xffff - centre + 1 > 0 even at a score of zero.
  var mine = 0u;
  for (var centre = local.x; centre < p.centres; centre += ${WORKGROUP}u) {
    let centre_base = p.centre_offset + centre * p.words;
    var score = 0u;
    for (var word = 0u; word < p.words; word += 1u) {
      score += agreeing(centre_words[centre_base + word] ^ extra_words[extra_base + word]);
    }
    let key = (score << 16u) | (0xffffu - centre);
    mine = max(mine, key);
  }
  best[local.x] = mine;
  workgroupBarrier();
  for (var stride = ${WORKGROUP / 2}u; stride > 0u; stride /= 2u) {
    if (local.x < stride) { best[local.x] = max(best[local.x], best[local.x + stride]); }
    workgroupBarrier();
  }
  if (local.x == 0u) {
    assignments[p.output_offset + row] = 0xffffu - (best[0] & 0xffffu);
  }
}`;


// 🔴 ONE COMPILE PER DEVICE, NOT ONE PER FOLD. The shader is a constant, and a
// pipeline built per call puts a compile on the critical path of every fold -
// the thing this kernel exists to take OFF it.
const PIPELINES = new WeakMap();

async function pipelineFor(device) {
  const existing = PIPELINES.get(device);
  if (existing !== undefined) return existing;
  // 🔴 AN EXPLICIT LAYOUT, BECAUSE THE UNIFORM IS BOUND AT A DYNAMIC OFFSET.
  // `layout: "auto"` never sets hasDynamicOffset, and setBindGroup with an
  // offset against an automatic layout is a validation error rather than a
  // slow path.
  const bindLayout = device.createBindGroupLayout({
    label: "features.nearest-centres.layout",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 32 } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });
  const pipeline = await device.createComputePipelineAsync({
    label: "features.nearest-centres",
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindLayout] }),
    compute: { module: device.createShaderModule({ code: SHADER }), entryPoint: "main" },
  });

  const built = { pipeline, bindLayout };
  PIPELINES.set(device, built);
  return built;
}

/**
 * One search. `centreWords` and `extraWords` come from `paddedCodeWords`.
 *
 * @typedef {{centreWords: Uint32Array, extraWords: Uint32Array,
 *            words: number, rows: number, centres: number}} Search
 */

/**
 * Run every search in one submit and return one Uint16Array of assignments per
 * search, in the order given.
 *
 * @param {GPUDevice} device
 * @param {Search[]} searches
 * @returns {Promise<Uint16Array[]>}
 */
export async function assignNearestCentres(device, searches) {
  if (searches.length === 0) return [];
  for (const search of searches) {
    if (search.centres > 0xffff) {
      throw new RangeError(`nearest-centre search has ${search.centres} centres, `
        + "past what the tie-breaking pack holds");
    }
    if (search.words * 4 > 0xffff) {
      throw new RangeError(`nearest-centre search has rows of ${search.words * 4} bytes, `
        + "past what the tie-breaking pack holds");
    }
  }

  const { pipeline, bindLayout } = await pipelineFor(device);

  // One buffer for every search's centres, one for every search's extras, and
  // one for every search's answers, so the whole set is a single submit.
  const centreLengths = searches.map((s) => s.centreWords.length);
  const extraLengths = searches.map((s) => s.extraWords.length);
  const offsets = (lengths) => {
    let at = 0;
    return lengths.map((length) => { const start = at; at += length; return start; });
  };
  const centreAt = offsets(centreLengths);
  const extraAt = offsets(extraLengths);
  const outputAt = offsets(searches.map((s) => s.rows));
  const centreTotal = centreLengths.reduce((sum, n) => sum + n, 0);
  const extraTotal = extraLengths.reduce((sum, n) => sum + n, 0);
  const outputTotal = searches.reduce((sum, s) => sum + s.rows, 0);

  const centres = device.createBuffer({
    label: "features.centres", size: Math.max(4, centreTotal * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const extras = device.createBuffer({
    label: "features.extras", size: Math.max(4, extraTotal * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    label: "features.assignments", size: Math.max(4, outputTotal * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    label: "features.assignments.read", size: Math.max(4, outputTotal * 4),
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  // 🔴 ONE UNIFORM BUFFER, BOUND AT A DYNAMIC OFFSET PER SEARCH, so a set of
  // recycles is still one submit. The offset has to be a multiple of the
  // device's alignment, which is 256 on every adapter measured.
  const stride = Math.max(32,
    device.limits?.minUniformBufferOffsetAlignment ?? 256);
  const parameters = device.createBuffer({
    label: "features.nearest-centres.parameters", size: stride * searches.length,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  for (const [index, search] of searches.entries()) {
    device.queue.writeBuffer(centres, centreAt[index] * 4, search.centreWords);
    device.queue.writeBuffer(extras, extraAt[index] * 4, search.extraWords);
    const block = new Uint32Array(8);
    block.set([search.rows, search.centres, search.words,
      extraAt[index], centreAt[index], outputAt[index]]);
    device.queue.writeBuffer(parameters, stride * index, block);
  }

  const bind = device.createBindGroup({
    layout: bindLayout,
    entries: [
      { binding: 0, resource: { buffer: centres } },
      { binding: 1, resource: { buffer: extras } },
      { binding: 2, resource: { buffer: parameters, size: 32 } },
      { binding: 3, resource: { buffer: output } },
    ],
  });

  const encoder = device.createCommandEncoder({ label: "features.nearest-centres" });
  const pass = encoder.beginComputePass({ label: "features.nearest-centres" });
  pass.setPipeline(pipeline);
  for (const [index, search] of searches.entries()) {
    if (search.rows === 0) continue;
    pass.setBindGroup(0, bind, [stride * index]);
    pass.dispatchWorkgroups(search.rows);
  }
  pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, Math.max(4, outputTotal * 4));
  device.queue.submit([encoder.finish()]);

  await readback.mapAsync(GPUMapMode.READ);
  const all = new Uint32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  for (const buffer of [centres, extras, output, readback, parameters]) buffer.destroy();

  return searches.map((search, index) => {
    const result = new Uint16Array(search.rows);
    for (let row = 0; row < search.rows; row += 1) result[row] = all[outputAt[index] + row];
    return result;
  });
}
