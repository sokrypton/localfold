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
 *
 * 🔴 A `p` SUFFIX PACKS q/k/v/gate TWO HALVES TO A WORD, which is not only a
 * storage change: `project` gives each lane ONE output channel and consecutive
 * lanes consecutive channels, so neither half of a packed word is owned by the
 * lane that would write it. The packed form gives a lane a PAIR of adjacent
 * channels - half the lanes, twice the accumulators - and twice the
 * accumulators is exactly where src/evoformer/attention.js records an AF2 tile
 * getting SLOWER, because sixteen vec4 spill. That is the question this arm
 * exists to answer:
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-grid-project.js --arms=8,8p
 *
 * A packed arm is EXPECTED to differ from the first in `relRmsVsFirst` - about
 * 1e-3, which is half precision and not a fault. A tile that leaves rows
 * unwritten shows up far larger; that is still what the column is for.
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
  // ...q, k, v and the gate interleaved; see packGridAttentionWeights.
  ["qkvgProjection", 4 * channels * heads * dimension],
  ["outputProjection", heads * dimension * channels],
];

export async function main(device, args) {
  const n = Number(option(args, "tokens", "59"));
  const channels = Number(option(args, "channels", "128"));
  const heads = Number(option(args, "heads", "4"));
  const dimension = Number(option(args, "dimension", "32"));
  const rounds = Number(option(args, "rounds", "9"));
  const iterations = Number(option(args, "iterations", "8"));
  // Arms are the projection row tile; the attention is timed alongside them as
  // an unchanging control.
  const arms_spec = option(args, "arms", "1,4,8,16,32").split(",");
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
  const bias = upload(random(pairs * heads));
  const mask = upload(new Float32Array(pairs).fill(1));
  const attended = allocate(pairs * width);
  const q = allocate(pairs * width);
  const k = allocate(pairs * width);
  const v = allocate(pairs * width);
  const gate = allocate(pairs * width);
  const output = allocate(pairs * channels);
  const readback = device.createBuffer({
    size: pairs * channels * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  const arms = [];
  // 🔴 THE :drop ARMS RETURN WRONG NUMBERS ON PURPOSE, to price one read rather
  // than to compute anything, and they OVERSTATE it - a constant lets the
  // compiler hoist the arithmetic that depended on the read too. Rank with
  // them; never take one as a target. "bias" replaces the attention's pair-bias
  // read, "kv" its key and value reads.
  const surgery = {
    // ...each replacement keeps the binding and is loop-invariant, so the
    // compiler hoists it out of the key loop. Dropping the read entirely would
    // drop the binding, and "auto" layout then rejects the bind group.
    bias: [/bias\[head \* PAIRS \+ i \* N \+ j\]/g, "bias[head * PAIRS]"],
    kv: [/(k|v)\[k_base \+ (\d)u\]/g, "$1[$2u]"],
    konly: [/k\[k_base \+ (\d)u\]/g, "k[$1u]"],
    vonly: [/v\[k_base \+ (\d)u\]/g, "v[$1u]"],
  };
  for (const spec of arms_spec) {
    const [shapeSpec, drop] = spec.split(":");
    const packed = shapeSpec.endsWith("p");
    const [rows, chunk] = (packed ? shapeSpec.slice(0, -1) : shapeSpec).split("/").map(Number);
    const sources = createGridAttentionShaders(
      { n, channels, heads, dimension, transpose: false,
        projectRows: rows, projectOutRows: rows, attendKeyChunk: chunk,
      },
      // ...gathered and normalized stay f32 here: this bench asks about
      // q/k/v/gate alone, which is the one that needs the ownership change.
      offsets, 1e-5, "fast", DIALECT, "f32", "f32", packed ? "f16" : "f32");
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
    let attendSource = sources.attend;
    if (drop) {
      const [pattern, replacement] = surgery[drop];
      const patched = attendSource.replace(pattern, replacement);
      if (patched === attendSource) throw new Error(`arm ${spec} matched nothing`);
      attendSource = patched;
    }
    const attend = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: attendSource }),
                 entryPoint: "main" },
    });
    arms.push({
      arm: spec,
      rows,
      attend: {
        pipeline: attend,
        bindGroup: device.createBindGroup({
          layout: attend.getBindGroupLayout(0),
          entries: [q, k, v, bias, mask, attended].map((buffer, binding) => ({
            binding, resource: { buffer } })),
        }),
        groups: Math.ceil(n / 64), y: n, z: heads,
      },
      project: await build(sources.project, [normalized, weights, q, k, v, gate],
                           sources.tiles.projectRows),
      projectOut: await build(sources.project_out, [gathered, gate, weights, output],
                              sources.tiles.projectOutRows),
      times: { project: [], projectOut: [], attend: [] },
    });
  }

  const time = async (kernel) => {
    const encoder = device.createCommandEncoder();
    for (let i = 0; i < iterations; i += 1) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(kernel.pipeline);
      pass.setBindGroup(0, kernel.bindGroup);
      pass.dispatchWorkgroups(kernel.groups, kernel.y ?? 1, kernel.z ?? 1);
      pass.end();
    }
    const start = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return (performance.now() - start) / iterations;
  };

  for (const arm of arms) {
    await time(arm.project); await time(arm.projectOut); await time(arm.attend);
  }
  for (let round = 0; round < rounds; round += 1) {
    for (const arm of arms) {
      arm.times.project.push(await time(arm.project));
      arm.times.projectOut.push(await time(arm.projectOut));
      arm.times.attend.push(await time(arm.attend));
    }
  }

  const results = [];
  for (const arm of arms) {
    const encoder = device.createCommandEncoder();
    for (const kernel of [arm.project, arm.projectOut]) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(kernel.pipeline);
      pass.setBindGroup(0, kernel.bindGroup);
      pass.dispatchWorkgroups(kernel.groups, kernel.y ?? 1, kernel.z ?? 1);
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
      arm: arm.arm,
      project: Number(median(arm.times.project).toFixed(3)),
      projectOut: Number(median(arm.times.projectOut).toFixed(3)),
      attend: Number(median(arm.times.attend).toFixed(3)),
      relRmsVsFirst: Number(relRms[index].toExponential(2)),
    })),
  };
}
