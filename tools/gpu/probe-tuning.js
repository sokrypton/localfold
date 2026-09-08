/**
 * Ask THIS device which of the per-device knobs it wants, and print the
 * `Tuning` object it should get.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-tuning.js
 *
 * WHY IT EXISTS. src/runtime/device-profile.js resolves three ways: a
 * capability the API states, a measurement, or an architecture prior. The
 * prior is the weak one - it is a table, it can only name architectures
 * somebody has run, and docs/A100.md records three devices giving three
 * different answers to the queries-per-invocation knob. Upstream reached the
 * same conclusion about that knob and replaced their threshold with a probe.
 * This is that probe, for all of them at once.
 *
 * 🔴 EVERY ARM IS CHECKED AGAINST THE OTHER, because a kernel that skips work
 * is one a stopwatch reads as a speedup. The tile arms must agree exactly - a
 * different tile is a different schedule of the same arithmetic - and the
 * softmax arms must agree to about 1e-6, because grouping the rescale
 * reassociates the online softmax and is not expected to be bitwise equal.
 *
 * 🔴 AND IT IS A HINT, NOT A GATE. A win here says the kernel is faster; it
 * does not say the model is still right. `attentionGroup` changes the answer
 * in the last few digits, so promoting it still means running
 * check-evoformer-attention.js and fold-af2.js. `linearTallTile` does not -
 * it measured relRMS 0 and a bit-identical fold on the one device that wanted
 * it - but "measured on one device" is what this file exists to widen.
 */
import { createLinearShader, LINEAR_TILE_WIDE, LINEAR_TILE_TALL,
  linearTileRows, linearTileColumns } from "../../src/evoformer/transition.js";
import { createAttentionRegisterFlashShader } from "../../src/evoformer/attention.js";
import { deviceProfile } from "../../src/runtime/device-profile.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const median = (v) => [...v].sort((a, b) => a - b)[v.length >> 1];

function noise(count, seed) {
  const out = new Float32Array(count);
  let state = seed >>> 0;
  for (let i = 0; i < count; i += 1) {
    state = (state * 1664525 + 1013904223) >>> 0;
    out[i] = state / 4294967296 - 0.5;
  }
  return out;
}

const relRms = (a, b) => {
  let error = 0; let scale = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    error += d * d; scale += b[i] * b[i];
  }
  return Math.sqrt(error / Math.max(scale, 1e-30));
};

/** One timed, verified comparison of two pipelines over the same buffers. */
async function race(device, arms, dispatchOf, output, sampleFloats, rounds, iterations) {
  const readback = device.createBuffer({
    size: sampleFloats * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const run = async (arm) => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(arm.pipeline);
    pass.setBindGroup(0, arm.bindGroup);
    const [x, y, z] = dispatchOf(arm);
    for (let i = 0; i < iterations; i += 1) pass.dispatchWorkgroups(x, y, z);
    pass.end();
    const start = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return (performance.now() - start) / iterations;
  };
  const sample = async () => {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(output, 0, readback, 0, sampleFloats * 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    return copy;
  };
  for (const arm of arms) { await run(arm); arm.sample = await sample(); arm.times = []; }
  // Interleaved, so a drift that walks through the run lands on both arms.
  for (let round = 0; round < rounds; round += 1) {
    for (const arm of arms) arm.times.push(await run(arm));
  }
  readback.destroy();
  for (const arm of arms) arm.ms = Number(median(arm.times).toFixed(4));
  for (const arm of arms) arm.relRms = Number(relRms(arm.sample, arms[0].sample).toExponential(2));
  return arms;
}

/** The dense projection, wide tile against tall. */
async function linearArms(device, rounds, iterations) {
  const rows = 8192; const inner = 256; const columns = 1024;
  const source = device.createBuffer({ size: rows * inner * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const weights = device.createBuffer({ size: (inner * columns + columns) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const output = device.createBuffer({ size: rows * columns * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  device.queue.writeBuffer(source, 0, noise(rows * inner, 7));
  device.queue.writeBuffer(weights, 0, noise(inner * columns + columns, 9));
  const params = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(params, 0, new Uint32Array([rows, inner, columns, 0, inner * columns, 0, 0, 0]));

  const arms = [];
  for (const [name, tile] of [["wide", LINEAR_TILE_WIDE], ["tall", LINEAR_TILE_TALL]]) {
    const pipeline = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: createLinearShader(tile) }), entryPoint: "main" },
    });
    arms.push({
      name, tile: `${linearTileRows(tile)}x${linearTileColumns(tile)}`,
      pipeline,
      bindGroup: device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries:
        [source, weights, params, output].map((buffer, binding) => ({ binding, resource: { buffer } })) }),
      grid: [Math.ceil(columns / linearTileColumns(tile)), Math.ceil(rows / linearTileRows(tile)), 1],
    });
  }
  const raced = await race(device, arms, (a) => a.grid, output, 1024, rounds, iterations);
  for (const b of [source, weights, output, params]) b.destroy();
  return raced;
}

/** The flash attention, one key per rescale against four. */
async function attentionArms(device, rounds, iterations) {
  const batch = 59; const queries = 512; const heads = 8; const headDim = 32;
  const channels = heads * headDim;
  const elements = batch * queries * channels;
  const make = (seed) => {
    const b = device.createBuffer({ size: elements * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, noise(elements, seed)); return b;
  };
  const q = make(21); const k = make(22); const v = make(23); const gate = make(24);
  const mask = device.createBuffer({ size: batch * queries * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(mask, 0, new Float32Array(batch * queries).fill(1));
  const bias = device.createBuffer({ size: 4 * heads * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(bias, 0, new Float32Array(4 * heads));
  // The `Parameters` struct in src/evoformer/attention.js is sixteen u32.
  const params = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const parameterValues = new Uint32Array(16);
  parameterValues.set([batch, queries, channels, heads, headDim, 0, 0]);
  device.queue.writeBuffer(params, 0, parameterValues);
  const output = device.createBuffer({ size: elements * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

  const precision = device.features.has("shader-f16") ? "chunk16" : "f32";
  const arms = [];
  for (const [name, group, vectorScore] of [["g1", 1, false], ["g4v", 4, true]]) {
    const code = createAttentionRegisterFlashShader(headDim, undefined, { precision, group, vectorScore });
    const pipeline = await device.createComputePipelineAsync({
      layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
    arms.push({ name, group, vectorScore, pipeline,
      bindGroup: device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries:
        [q, k, v, gate, mask, bias, params, output].map((buffer, binding) => ({ binding, resource: { buffer } })) }),
      grid: [Math.ceil(queries / 64), batch, heads] });
  }
  const raced = await race(device, arms, (a) => a.grid, output, 1024, rounds, iterations);
  for (const b of [q, k, v, gate, mask, bias, params, output]) b.destroy();
  return raced;
}

export async function main(device, args) {
  const rounds = Number(option(args, "rounds", "9"));
  const iterations = Number(option(args, "iterations", "16"));
  // A win has to clear the noise, not just be positive.
  const margin = Number(option(args, "margin", "1.05"));

  const profile = deviceProfile(device);
  const notes = [];
  if (profile.software) {
    notes.push("THIS IS A SOFTWARE ADAPTER. Nothing measured here is about a GPU - "
      + "see docs/A100.md for why Linux/NVIDIA answers requestAdapter this way.");
  }
  if (!device.features.has("shader-f16")) {
    notes.push("no shader-f16, so every f16 path in this repository is switched off "
      + "and these arms are its f32 fallbacks");
  }

  const linear = await linearArms(device, rounds, iterations);
  const attention = await attentionArms(device, rounds, iterations);

  const speedup = (arms) => arms[0].ms / arms[1].ms;
  const linearSpeedup = speedup(linear);
  const attentionSpeedup = speedup(attention);
  // The tile arms compute the same arithmetic in a different order of
  // workgroups, so anything but agreement is a dispatch that missed rows.
  const tileExact = linear[1].relRms === 0;
  const softmaxSane = attention[1].relRms < 1e-5;

  const recommended = {
    linearTallTile: tileExact && linearSpeedup >= margin,
    attentionGroup: softmaxSane && attentionSpeedup >= margin ? 4 : 1,
    attentionVectorScore: softmaxSane && attentionSpeedup >= margin,
  };

  return {
    adapter: { vendor: profile.vendor, architecture: profile.architecture, software: profile.software },
    features: [...device.features].sort(),
    margin,
    linear: { arms: linear.map(({ name, tile, ms, relRms }) => ({ name, tile, ms, relRms })),
      speedup: Number(linearSpeedup.toFixed(3)), exact: tileExact },
    attention: { arms: attention.map(({ name, ms, relRms }) => ({ name, ms, relRms })),
      speedup: Number(attentionSpeedup.toFixed(3)) },
    currentTuning: profile.tuning,
    recommended,
    agrees: JSON.stringify({ ...profile.tuning, attentionQueriesPerLane: undefined })
      === JSON.stringify({ ...profile.tuning, ...recommended, attentionQueriesPerLane: undefined }),
    notes,
  };
}
