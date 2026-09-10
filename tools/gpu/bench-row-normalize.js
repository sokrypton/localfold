/**
 * Row LayerNorm, three arrangements, at the shapes an AF2 block normalises.
 *
 * 🔴 IT IS 7.1% OF A MAIN BLOCK AND HAD NEVER BEEN LOOKED AT. Measured with
 * profile-af2-block.js at 400 residues and 512 sequences: 4.27 ms of 60.44,
 * spread over ten labels, every one of them a row reduced TWICE through a
 * halving tree in workgroup memory - six barriers a reduction, and most of the
 * workgroup idle after the first step.
 *
 * The arms, and what separates them:
 *
 *   tree       one row a workgroup of 64, two halving trees   - what ships
 *   subgroup1  one row a workgroup of 32, one subgroupAdd     - the CONTROL
 *   rows8      one row a SUBGROUP, eight of them a workgroup
 *
 * `subgroup1` is the control that matters: it has the barrier-free reduction
 * and NOT the rows a workgroup covers, so the gap between it and `rows8` is
 * the rows and not the reduction. Upstream reports 1.3x for the reduction
 * alone against 1.6x-3.6x for the eight-row arrangement, which is the claim
 * this file is here to reproduce or refuse.
 *
 * Arms are interleaved and the minimum of three rounds is taken, because this
 * machine drifts by up to 3.2x between runs.
 */
const GRID = 32768;
const REPEATS = 32;

const treeShader = `
const GRID_WIDTH: u32 = ${GRID}u;
struct P { rows: u32, channels: u32, epsilon: f32, pad: u32 };
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
var<workgroup> partial: array<f32, 64>;
var<workgroup> row_mean: array<f32, 1>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let row = group.x + group.y * GRID_WIDTH;
  if (row >= p.rows) { return; }
  let base = row * p.channels;
  var sum = 0.0;
  for (var c = local.x; c < p.channels; c += 64u) { sum += source[base + c]; }
  partial[local.x] = sum;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (local.x < stride) { partial[local.x] += partial[local.x + stride]; }
    workgroupBarrier();
  }
  if (local.x == 0u) { row_mean[0] = partial[0] / f32(p.channels); }
  workgroupBarrier();
  var squared = 0.0;
  for (var c = local.x; c < p.channels; c += 64u) {
    let centered = source[base + c] - row_mean[0];
    squared += centered * centered;
  }
  partial[local.x] = squared;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (local.x < stride) { partial[local.x] += partial[local.x + stride]; }
    workgroupBarrier();
  }
  let inverse_std = inverseSqrt(partial[0] / f32(p.channels) + p.epsilon);
  for (var c = local.x; c < p.channels; c += 64u) {
    output[base + c] = (source[base + c] - row_mean[0]) * inverse_std
      * weights[c] + weights[p.channels + c];
  }
}`;

// One row a subgroup, `rows` of them a workgroup. At rows = 1 this is the
// control: the same barrier-free reduction over one row at a time.
const subgroupShader = (rows) => `
enable subgroups;
const GRID_WIDTH: u32 = ${GRID}u;
const ROWS: u32 = ${rows}u;
struct P { rows: u32, channels: u32, epsilon: f32, pad: u32 };
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(32, ${rows})
fn main(@builtin(local_invocation_id) local: vec3<u32>,
        @builtin(workgroup_id) group: vec3<u32>) {
  // 🔴 NO EARLY RETURN. Every lane of a subgroup shares this row - it depends
  // on local.y and not on local.x - but the compiler cannot prove that, and
  // subgroupAdd must be called from subgroup uniform control flow. So the tail
  // row is CLAMPED and only the store is guarded.
  let row = (group.x + group.y * GRID_WIDTH) * ROWS + local.y;
  let valid = row < p.rows;
  let base = select(0u, row, valid) * p.channels;
  let lane = local.x;
  var sum = 0.0;
  for (var c = lane; c < p.channels; c += 32u) { sum += source[base + c]; }
  let mean = subgroupAdd(sum) / f32(p.channels);
  var squared = 0.0;
  for (var c = lane; c < p.channels; c += 32u) {
    let centered = source[base + c] - mean;
    squared += centered * centered;
  }
  let inverse_std = inverseSqrt(subgroupAdd(squared) / f32(p.channels) + p.epsilon);
  if (valid) {
    for (var c = lane; c < p.channels; c += 32u) {
      output[base + c] = (source[base + c] - mean) * inverse_std
        * weights[c] + weights[p.channels + c];
    }
  }
}`;

const SHAPES = [
  { rows: 204800, channels: 256, note: "msa normalize, 400 res x 512 seq" },
  { rows: 160000, channels: 128, note: "pair track, 400 res" },
  { rows: 29972, channels: 256, note: "upstream's widest" },
  { rows: 60416, channels: 64, note: "upstream's narrowest" },
];

async function build(device, code, entry = "main") {
  return device.createComputePipelineAsync({
    layout: "auto",
    compute: { module: device.createShaderModule({ code }), entryPoint: entry },
  });
}

export async function main(device, args = []) {
  const roundCount = Number((args.find((a) => a.startsWith("--rounds=")) ?? "--rounds=3").slice(9));
  if (!device.features.has("subgroups")) throw new Error("this device has no subgroups");
  const arms = [
    { name: "tree", groupRows: 1, workgroup: 64, pipeline: await build(device, treeShader) },
    { name: "subgroup1", groupRows: 1, workgroup: 32, pipeline: await build(device, subgroupShader(1)) },
    { name: "rows4", groupRows: 4, workgroup: 128, pipeline: await build(device, subgroupShader(4)) },
    { name: "rows8", groupRows: 8, workgroup: 256, pipeline: await build(device, subgroupShader(8)) },
    { name: "rows16", groupRows: 16, workgroup: 512, pipeline: await build(device, subgroupShader(16)) },
  ];

  const results = [];
  for (const shape of SHAPES) {
    const elements = shape.rows * shape.channels;
    const source = device.createBuffer({ size: elements * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const output = device.createBuffer({ size: elements * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const weights = device.createBuffer({ size: shape.channels * 2 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const seed = new Float32Array(Math.min(elements, 1 << 22));
    for (let i = 0; i < seed.length; i += 1) seed[i] = Math.sin(i * 0.37) * 2;
    for (let at = 0; at < elements; at += seed.length) {
      device.queue.writeBuffer(source, at * 4, seed, 0, Math.min(seed.length, elements - at));
    }
    const w = new Float32Array(shape.channels * 2);
    for (let i = 0; i < shape.channels; i += 1) { w[i] = 1 + 0.01 * i; w[shape.channels + i] = 0.001 * i; }
    device.queue.writeBuffer(weights, 0, w);
    const p = new ArrayBuffer(16);
    new Uint32Array(p, 0, 2).set([shape.rows, shape.channels]);
    new Float32Array(p, 8, 1)[0] = 1e-5;
    device.queue.writeBuffer(params, 0, p);

    const timings = new Map(arms.map((a) => [a.name, Infinity]));
    for (let round = 0; round < roundCount; round += 1) {
      for (const arm of arms) {
        const groups = Math.ceil(shape.rows / arm.groupRows);
        const x = Math.min(groups, GRID);
        const y = Math.ceil(groups / GRID);
        const bind = device.createBindGroup({
          layout: arm.pipeline.getBindGroupLayout(0),
          entries: [source, weights, params, output].map((buffer, binding) => ({ binding, resource: { buffer } })),
        });
        // 🔴 REPEATS INSIDE ONE PASS. A single dispatch of any of these arms is
        // shorter than the submit round trip - every arm measured 2.3-2.4 ms,
        // which is `onSubmittedWorkDone` and not the kernel. REPEATS of them in
        // one pass puts the kernel above the floor; the reads are the same
        // every time, so this measures a warm cache, which is what a block sees
        // between its own passes anyway.
        for (const measured of [false, true]) {
          const started = performance.now();
          const encoder = device.createCommandEncoder();
          const pass = encoder.beginComputePass();
          pass.setPipeline(arm.pipeline);
          pass.setBindGroup(0, bind);
          for (let n = 0; n < REPEATS; n += 1) pass.dispatchWorkgroups(x, y);
          pass.end();
          device.queue.submit([encoder.finish()]);
          await device.queue.onSubmittedWorkDone();
          if (measured) {
            const ms = (performance.now() - started) / REPEATS;
            if (ms < timings.get(arm.name)) timings.set(arm.name, ms);
          }
        }
      }
    }
    const tree = timings.get("tree");
    const row = { ...shape };
    for (const arm of arms) {
      row[arm.name] = Number(timings.get(arm.name).toFixed(3));
      if (arm.name !== "tree") row[`${arm.name}x`] = Number((tree / timings.get(arm.name)).toFixed(2));
    }
    results.push(row);
    console.log(`${shape.rows} x ${shape.channels}\t`
      + arms.map((a) => `${a.name} ${timings.get(a.name).toFixed(3)}`).join("  ")
      + `\t${shape.note}`);
    for (const b of [source, output, weights, params]) b.destroy();
  }
  return { results };
}
