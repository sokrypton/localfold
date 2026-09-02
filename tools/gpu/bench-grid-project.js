/**
 * The grid attention's two projection kernels, at several row tiles.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-grid-project.js --tokens=59
 *     node tools/gpu-chrome.mjs tools/gpu/bench-grid-project.js --arms=4,8,16
 *
 * WHY IT EXISTS. `grid.project` and `grid.project-out` are the pair track's
 * other 80 ms, and both are bounded by how many times a workgroup re-reads the
 * same weight matrix - one workgroup a row read the whole output projection to
 * project one row down. The tile that divides that traffic also costs workgroup
 * memory and occupancy, so where the two cross is measured, not derived. An arm
 * costs about a second; bench-trunk.js costs forty and averages 48 blocks.
 *
 * Arms are the row tile. Both kernels are timed at each, and every arm is
 * checked against the first, because a tile the dispatch does not match leaves
 * rows unwritten and reads as a speedup.
 */
import { createGridAttentionShaders } from "../../src/af3/grid-attention-webgpu.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const DIALECT = { swapTransposedBias: false };
const SIZES = (channels, heads, dimension) => [
  ["actNormScale", channels], ["actNormOffset", channels],
  ["pairBiasProjection", channels * heads],
  ["qProjection", channels * heads * dimension], ["kProjection", channels * heads * dimension],
  ["vProjection", channels * heads * dimension], ["gatingQuery", channels * heads * dimension],
  ["outputProjection", heads * dimension * channels],
];

export async function main(device, args) {
  const n = Number(option(args, "tokens", "59"));
  const channels = Number(option(args, "channels", "128"));
  const heads = Number(option(args, "heads", "4"));
  const dimension = Number(option(args, "dimension", "32"));
  const rounds = Number(option(args, "rounds", "9"));
  const iterations = Number(option(args, "iterations", "8"));
  const arms_spec = option(args, "arms", "1,4,8,16,32").split(",").map(Number);
  const pairs = n * n;
  const width = heads * dimension;

  const offsets = {};
  let total = 0;
  for (const [name, size] of SIZES(channels, heads, dimension)) {
    offsets[name] = total;
    total += size;
  }
  let state = 4242;
  const random = (count) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      out[i] = (state / 0x7fffffff - 0.5) * 0.2;
    }
    return out;
  };
  const storage = GPUBufferUsage.STORAGE;
  const upload = (data) => {
    const buffer = device.createBuffer({
      size: data.byteLength, usage: storage, mappedAtCreation: true });
    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();
    return buffer;
  };
  const allocate = (floats) => device.createBuffer({
    size: floats * 4, usage: storage | GPUBufferUsage.COPY_SRC });

  const weights = upload(random(total));
  const normalized = upload(random(pairs * channels));
  const gathered = upload(random(pairs * width));
  const q = allocate(pairs * width);
  const k = allocate(pairs * width);
  const v = allocate(pairs * width);
  const gate = allocate(pairs * width);
  const output = allocate(pairs * channels);
  const readback = device.createBuffer({
    size: pairs * channels * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  const arms = [];
  for (const rows of arms_spec) {
    const sources = createGridAttentionShaders(
      { n, channels, heads, dimension, transpose: false,
        projectRows: rows, projectOutRows: rows },
      offsets, 1e-5, "fast", DIALECT);
    const build = async (source, buffers, tile) => {
      const pipeline = await device.createComputePipelineAsync({
        layout: "auto",
        compute: { module: device.createShaderModule({ code: source }), entryPoint: "main" },
      });
      return {
        pipeline,
        bindGroup: device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
        }),
        groups: Math.ceil(pairs / tile),
      };
    };
    arms.push({
      rows,
      project: await build(sources.project, [normalized, weights, q, k, v, gate],
                           sources.tiles.projectRows),
      projectOut: await build(sources.project_out, [gathered, gate, weights, output],
                              sources.tiles.projectOutRows),
      times: { project: [], projectOut: [] },
    });
  }

  const time = async (kernel) => {
    const encoder = device.createCommandEncoder();
    for (let i = 0; i < iterations; i += 1) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(kernel.pipeline);
      pass.setBindGroup(0, kernel.bindGroup);
      pass.dispatchWorkgroups(kernel.groups);
      pass.end();
    }
    const start = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return (performance.now() - start) / iterations;
  };

  for (const arm of arms) { await time(arm.project); await time(arm.projectOut); }
  for (let round = 0; round < rounds; round += 1) {
    for (const arm of arms) {
      arm.times.project.push(await time(arm.project));
      arm.times.projectOut.push(await time(arm.projectOut));
    }
  }

  const results = [];
  for (const arm of arms) {
    const encoder = device.createCommandEncoder();
    for (const kernel of [arm.project, arm.projectOut]) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(kernel.pipeline);
      pass.setBindGroup(0, kernel.bindGroup);
      pass.dispatchWorkgroups(kernel.groups);
      pass.end();
    }
    encoder.copyBufferToBuffer(output, 0, readback, 0, pairs * channels * 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    results.push(new Float32Array(readback.getMappedRange().slice(0)));
    readback.unmap();
  }
  const reference = results[0];
  const relRms = results.map((out) => {
    let error = 0, scale = 0;
    for (let i = 0; i < reference.length; i += 1) {
      error += (out[i] - reference[i]) ** 2;
      scale += reference[i] ** 2;
    }
    return Math.sqrt(error / scale);
  });

  const median = (values) => [...values].sort((p, r) => p - r)[values.length >> 1];
  return {
    n, pairs, channels, width,
    arms: arms.map((arm, index) => ({
      rows: arm.rows,
      project: Number(median(arm.times.project).toFixed(3)),
      projectOut: Number(median(arm.times.projectOut).toFixed(3)),
      relRmsVsFirst: Number(relRms[index].toExponential(2)),
    })),
  };
}
