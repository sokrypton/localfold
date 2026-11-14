/**
 * Does a lane see what ANOTHER lane wrote to workgroup memory before a barrier? Pattern by pattern.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-barrier-visibility.js
 *     LOCALFOLD_GPU_FIREFOX=1 node tools/gpu-chrome.mjs tools/gpu/probe-barrier-visibility.js
 *
 * 🔴 FIREFOX ON APPLE SILICON GETS ONE OF THESE WRONG, AND IT TURNED EVERY AF3-LINEAGE FOLD INTO NaN (issue #10).
 * The atom attention's softmax finished its maximum in lane 0 alone - `if (local == 0u) { reduce[0] = max(...); }`,
 * a barrier, `let largest = reduce[0];` - and the other 63 lanes read reduce[0] as it stood BEFORE lane 0's store.
 * Each arm here is one shape of "few lanes write, every lane reads after a barrier", over 64 lanes that each hold
 * a distinct value, so the right answer is known exactly; the probe counts the lanes that got another one.
 * A kernel must not use a shape this reports wrong on any browser.
 */
const ARMS = {
  // the shape that failed: lane 0 folds eight partials into a constant slot, every lane loads that slot
  "lane0-combine8-const": `
  if (local < 32u) { reduce[local] = max(reduce[local], reduce[local + 32u]); } workgroupBarrier();
  if (local < 16u) { reduce[local] = max(reduce[local], reduce[local + 16u]); } workgroupBarrier();
  if (local < 8u) { reduce[local] = max(reduce[local], reduce[local + 8u]); } workgroupBarrier();
  if (local == 0u) {
    let m0 = max(max(reduce[0], reduce[4]), max(reduce[2], reduce[6]));
    let m1 = max(max(reduce[1], reduce[5]), max(reduce[3], reduce[7]));
    reduce[0] = max(m0, m1);
  }
  workgroupBarrier();
  let answer = reduce[0];
  workgroupBarrier();`,
  // the same without the trailing barrier
  "lane0-combine8-const-nobarrier": `
  if (local < 32u) { reduce[local] = max(reduce[local], reduce[local + 32u]); } workgroupBarrier();
  if (local < 16u) { reduce[local] = max(reduce[local], reduce[local + 16u]); } workgroupBarrier();
  if (local < 8u) { reduce[local] = max(reduce[local], reduce[local + 8u]); } workgroupBarrier();
  if (local == 0u) {
    let m0 = max(max(reduce[0], reduce[4]), max(reduce[2], reduce[6]));
    let m1 = max(max(reduce[1], reduce[5]), max(reduce[3], reduce[7]));
    reduce[0] = max(m0, m1);
  }
  workgroupBarrier();
  let answer = reduce[0];`,
  // the full tree, its last step written with constant indices
  "tree-last-const": `
  if (local < 32u) { reduce[local] = max(reduce[local], reduce[local + 32u]); } workgroupBarrier();
  if (local < 16u) { reduce[local] = max(reduce[local], reduce[local + 16u]); } workgroupBarrier();
  if (local < 8u) { reduce[local] = max(reduce[local], reduce[local + 8u]); } workgroupBarrier();
  if (local < 4u) { reduce[local] = max(reduce[local], reduce[local + 4u]); } workgroupBarrier();
  if (local < 2u) { reduce[local] = max(reduce[local], reduce[local + 2u]); } workgroupBarrier();
  if (local == 0u) { reduce[0] = max(reduce[0], reduce[1]); } workgroupBarrier();
  let answer = reduce[0];
  workgroupBarrier();`,
  // the full tree, every step indexed by the lane
  "tree-last-lane": `
  if (local < 32u) { reduce[local] = max(reduce[local], reduce[local + 32u]); } workgroupBarrier();
  if (local < 16u) { reduce[local] = max(reduce[local], reduce[local + 16u]); } workgroupBarrier();
  if (local < 8u) { reduce[local] = max(reduce[local], reduce[local + 8u]); } workgroupBarrier();
  if (local < 4u) { reduce[local] = max(reduce[local], reduce[local + 4u]); } workgroupBarrier();
  if (local < 2u) { reduce[local] = max(reduce[local], reduce[local + 2u]); } workgroupBarrier();
  if (local < 1u) { reduce[local] = max(reduce[local], reduce[local + 1u]); } workgroupBarrier();
  let answer = reduce[0];`,
  // lane 0 writes a DIFFERENT workgroup variable, every lane reads it
  "lane0-other-variable": `
  if (local < 32u) { reduce[local] = max(reduce[local], reduce[local + 32u]); } workgroupBarrier();
  if (local < 16u) { reduce[local] = max(reduce[local], reduce[local + 16u]); } workgroupBarrier();
  if (local < 8u) { reduce[local] = max(reduce[local], reduce[local + 8u]); } workgroupBarrier();
  if (local == 0u) {
    let m0 = max(max(reduce[0], reduce[4]), max(reduce[2], reduce[6]));
    let m1 = max(max(reduce[1], reduce[5]), max(reduce[3], reduce[7]));
    shared_value = max(m0, m1);
  }
  workgroupBarrier();
  let answer = shared_value;`,
  // lane 0 alone does serial work over what every lane staged, into another array every lane reads
  "lane0-serial-array": `
  workgroupBarrier();
  if (local == 0u) {
    var largest = -3.0e38;
    for (var i = 0u; i < 64u; i += 1u) { largest = max(largest, reduce[i]); }
    for (var i = 0u; i < 64u; i += 1u) { table[i] = largest; }
  }
  workgroupBarrier();
  let answer = table[local];`,
  // lane 0 alone, but the slot named by the lane
  "lane0-lane-indexed": `
  if (local < 32u) { reduce[local] = max(reduce[local], reduce[local + 32u]); } workgroupBarrier();
  if (local < 16u) { reduce[local] = max(reduce[local], reduce[local + 16u]); } workgroupBarrier();
  if (local < 8u) { reduce[local] = max(reduce[local], reduce[local + 8u]); } workgroupBarrier();
  if (local < 4u) { reduce[local] = max(reduce[local], reduce[local + 4u]); } workgroupBarrier();
  if (local < 2u) { reduce[local] = max(reduce[local], reduce[local + 2u]); } workgroupBarrier();
  if (local == 0u) { reduce[local] = max(reduce[local], reduce[local + 1u]); } workgroupBarrier();
  let answer = reduce[0];`,
  // lane 0 alone into a constant slot, the store inside a loop that runs once for lane 0 and never for the rest
  "lane0-const-in-loop": `
  if (local < 32u) { reduce[local] = max(reduce[local], reduce[local + 32u]); } workgroupBarrier();
  if (local < 16u) { reduce[local] = max(reduce[local], reduce[local + 16u]); } workgroupBarrier();
  if (local < 8u) { reduce[local] = max(reduce[local], reduce[local + 8u]); } workgroupBarrier();
  if (local < 4u) { reduce[local] = max(reduce[local], reduce[local + 4u]); } workgroupBarrier();
  if (local < 2u) { reduce[local] = max(reduce[local], reduce[local + 2u]); } workgroupBarrier();
  for (var once = local; once < 1u; once += 1u) { reduce[0] = max(reduce[0], reduce[1]); }
  workgroupBarrier();
  let answer = reduce[0];`,
  // lane 0 alone into another workgroup array, at an index every lane shares but no compiler can see
  "lane0-uniform-dynamic-slot": `
  if (local < 32u) { reduce[local] = max(reduce[local], reduce[local + 32u]); } workgroupBarrier();
  if (local < 16u) { reduce[local] = max(reduce[local], reduce[local + 16u]); } workgroupBarrier();
  if (local < 8u) { reduce[local] = max(reduce[local], reduce[local + 8u]); } workgroupBarrier();
  if (local < 4u) { reduce[local] = max(reduce[local], reduce[local + 4u]); } workgroupBarrier();
  if (local < 2u) { reduce[local] = max(reduce[local], reduce[local + 2u]); } workgroupBarrier();
  let slot = group.x % 64u;
  if (local == 0u) { table[slot] = max(reduce[0], reduce[1]); }
  workgroupBarrier();
  let answer = table[slot];`,
  // lane 0 alone into a STORAGE buffer, every lane reads it back
  "lane0-storage-buffer": `
  if (local < 32u) { reduce[local] = max(reduce[local], reduce[local + 32u]); } workgroupBarrier();
  if (local < 16u) { reduce[local] = max(reduce[local], reduce[local + 16u]); } workgroupBarrier();
  if (local < 8u) { reduce[local] = max(reduce[local], reduce[local + 8u]); } workgroupBarrier();
  if (local < 4u) { reduce[local] = max(reduce[local], reduce[local + 4u]); } workgroupBarrier();
  if (local < 2u) { reduce[local] = max(reduce[local], reduce[local + 2u]); } workgroupBarrier();
  if (local == 0u) { scratch[group.x] = max(reduce[0], reduce[1]); }
  workgroupBarrier();
  let answer = scratch[group.x];`,
  // several lanes of one channel group: the lanes with c == 0 write a storage slot the group shares
  "few-lanes-storage-shared-slot": `
  if (local < 32u) { reduce[local] = max(reduce[local], reduce[local + 32u]); } workgroupBarrier();
  if (local < 16u) { reduce[local] = max(reduce[local], reduce[local + 16u]); } workgroupBarrier();
  if (local < 8u) { reduce[local] = max(reduce[local], reduce[local + 8u]); } workgroupBarrier();
  if (local < 4u) { reduce[local] = max(reduce[local], reduce[local + 4u]); } workgroupBarrier();
  if (local < 2u) { reduce[local] = max(reduce[local], reduce[local + 2u]); } workgroupBarrier();
  if (local == 0u) { reduce[local] = max(reduce[local], reduce[local + 1u]); } workgroupBarrier();
  if (local % 16u == 0u) { scratch[group.x] = reduce[0]; }
  workgroupBarrier();
  let answer = scratch[group.x];`,
  // lane 0 alone, two slots named by the lane plus an offset, read back at constant indices (a kernel's stats pair)
  "lane0-lane-plus-offset": `
  if (local < 32u) { reduce[local] = max(reduce[local], reduce[local + 32u]); } workgroupBarrier();
  if (local < 16u) { reduce[local] = max(reduce[local], reduce[local + 16u]); } workgroupBarrier();
  if (local < 8u) { reduce[local] = max(reduce[local], reduce[local + 8u]); } workgroupBarrier();
  if (local == 0u) {
    let m0 = max(max(reduce[0], reduce[4]), max(reduce[2], reduce[6]));
    let m1 = max(max(reduce[1], reduce[5]), max(reduce[3], reduce[7]));
    table[local] = m0;
    table[local + 1u] = m1;
  }
  workgroupBarrier();
  let answer = max(table[0], table[1]);`,
  // lane 0 alone, the slot named by the lane - in an array of ONE element, where a bounds clamp makes it constant
  "lane0-lane-indexed-array1": `
  if (local < 32u) { reduce[local] = max(reduce[local], reduce[local + 32u]); } workgroupBarrier();
  if (local < 16u) { reduce[local] = max(reduce[local], reduce[local + 16u]); } workgroupBarrier();
  if (local < 8u) { reduce[local] = max(reduce[local], reduce[local + 8u]); } workgroupBarrier();
  if (local == 0u) {
    let m0 = max(max(reduce[0], reduce[4]), max(reduce[2], reduce[6]));
    let m1 = max(max(reduce[1], reduce[5]), max(reduce[3], reduce[7]));
    one[local] = max(m0, m1);
  }
  workgroupBarrier();
  let answer = one[0];`,
  // ...of two elements
  "lane0-lane-indexed-array2": `
  if (local < 32u) { reduce[local] = max(reduce[local], reduce[local + 32u]); } workgroupBarrier();
  if (local < 16u) { reduce[local] = max(reduce[local], reduce[local + 16u]); } workgroupBarrier();
  if (local < 8u) { reduce[local] = max(reduce[local], reduce[local + 8u]); } workgroupBarrier();
  if (local == 0u) {
    let m0 = max(max(reduce[0], reduce[4]), max(reduce[2], reduce[6]));
    let m1 = max(max(reduce[1], reduce[5]), max(reduce[3], reduce[7]));
    two[local] = m0;
    two[local + 1u] = m1;
  }
  workgroupBarrier();
  let answer = max(two[0], two[1]);`,
  // ...of three
  "lane0-lane-indexed-array3": `
  if (local < 32u) { reduce[local] = max(reduce[local], reduce[local + 32u]); } workgroupBarrier();
  if (local < 16u) { reduce[local] = max(reduce[local], reduce[local + 16u]); } workgroupBarrier();
  if (local < 8u) { reduce[local] = max(reduce[local], reduce[local + 8u]); } workgroupBarrier();
  if (local == 0u) {
    let m0 = max(max(reduce[0], reduce[4]), max(reduce[2], reduce[6]));
    let m1 = max(max(reduce[1], reduce[5]), max(reduce[3], reduce[7]));
    three[local] = m0;
    three[local + 1u] = m1;
    three[local + 2u] = max(m0, m1);
  }
  workgroupBarrier();
  let answer = max(three[2], max(three[0], three[1]));`,
  // no single writer: every lane combines the eight partials itself (what the kernel does now)
  "every-lane-combine8": `
  if (local < 32u) { reduce[local] = max(reduce[local], reduce[local + 32u]); } workgroupBarrier();
  if (local < 16u) { reduce[local] = max(reduce[local], reduce[local + 16u]); } workgroupBarrier();
  if (local < 8u) { reduce[local] = max(reduce[local], reduce[local + 8u]); } workgroupBarrier();
  let m0 = max(max(reduce[0], reduce[4]), max(reduce[2], reduce[6]));
  let m1 = max(max(reduce[1], reduce[5]), max(reduce[3], reduce[7]));
  let answer = max(m0, m1);
  workgroupBarrier();`,
};

const shader = (body) => `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read_write> scratch: array<f32>;
var<workgroup> reduce: array<f32, 64>;
var<workgroup> table: array<f32, 64>;
var<workgroup> shared_value: f32;
var<workgroup> one: array<f32, 1>;
var<workgroup> two: array<f32, 2>;
var<workgroup> three: array<f32, 3>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
  let local = local_id.x;
  // as the kernel: a few elements of real work a lane before the reduction
  var mine = -3.0e38;
  for (var i = 0u; i < 4u; i += 1u) { mine = max(mine, input[(group.x * 64u + local) * 4u + i]); }
  reduce[local] = mine;
  workgroupBarrier();
${body}
  output[group.x * 64u + local] = answer;
}`;

export async function main(device) {
  const groups = 4096;
  const values = new Float32Array(groups * 64 * 4);
  let state = 12345;
  for (let i = 0; i < values.length; i += 1) {
    state = (Math.imul(state, 1103515245) + 12345) >>> 0;
    values[i] = (state / 4294967296) * 2000 - 1000;
  }
  const expected = new Float32Array(groups);
  for (let g = 0; g < groups; g += 1) {
    let largest = -Infinity;
    for (let i = 0; i < 256; i += 1) largest = Math.max(largest, values[g * 256 + i]);
    expected[g] = largest;
  }
  const input = device.createBuffer({ size: values.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(input, 0, values);
  const scratch = device.createBuffer({ size: groups * 4, usage: GPUBufferUsage.STORAGE });
  const rows = [];
  for (const [name, body] of Object.entries(ARMS)) {
    const output = device.createBuffer({ size: groups * 64 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: groups * 64 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    device.pushErrorScope("validation");
    const pipeline = await device.createComputePipelineAsync({ layout: "auto",
      compute: { module: device.createShaderModule({ code: shader(body) }), entryPoint: "main" } });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: input } }, { binding: 1, resource: { buffer: output } },
        ...(body.includes("scratch[") ? [{ binding: 2, resource: { buffer: scratch } }] : [])] }));
    pass.dispatchWorkgroups(groups);
    pass.end();
    encoder.copyBufferToBuffer(output, 0, read, 0, groups * 64 * 4);
    device.queue.submit([encoder.finish()]);
    const error = await device.popErrorScope();
    if (error !== null) { rows.push({ arm: name, error: error.message.slice(0, 200) }); continue; }
    await read.mapAsync(GPUMapMode.READ);
    const got = new Float32Array(read.getMappedRange().slice(0));
    read.unmap();
    let wrongLanes = 0, wrongGroups = 0, laneZeroWrong = 0;
    for (let g = 0; g < groups; g += 1) {
      let any = false;
      for (let l = 0; l < 64; l += 1) {
        if (got[g * 64 + l] !== expected[g]) { wrongLanes += 1; any = true; if (l === 0) laneZeroWrong += 1; }
      }
      if (any) wrongGroups += 1;
    }
    rows.push({ arm: name, wrongLanes, wrongGroups, laneZeroWrong, of: groups * 64 });
    console.log(`${name.padEnd(34)} ${wrongLanes === 0 ? "ok" : `WRONG in ${wrongLanes} of ${groups * 64} lanes, ${wrongGroups} of ${groups} workgroups (lane 0: ${laneZeroWrong})`}`);
    output.destroy(); read.destroy();
  }
  return { rows, wrong: rows.filter((row) => row.wrongLanes !== 0 || row.error !== undefined).map((row) => row.arm) };
}
