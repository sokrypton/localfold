/**
 * AF2's q/k/v/gate projection alone, at several register blocks, in one process.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-attention-project.js
 *     node tools/gpu-chrome.mjs tools/gpu/bench-attention-project.js --arms=4x2,8x2@f16
 *
 * WHY IT EXISTS. The two attention projections were 27 ms of a 105 ms evoformer
 * block at 512 MSA rows - the largest thing in it after column attention - and
 * the sweep that chose their tile was run through profile-af2-block.js at a
 * minute an arm, which is the same complaint bench-evoformer-linear.js was
 * written to answer for the transition's kernel. This is that, for this kernel:
 * synthetic weights, one shader, several tiles interleaved, about a second an
 * arm.
 *
 * 🔴 THEY ARE 20 ms OF AN 87 ms BLOCK NOW, AND NO LONGER SECOND. The tile this
 * bench found took them there, and the transitions - 25 ms across their two
 * halves - are the larger group. Column attention is still the largest single
 * kernel at 16.7. The sentence above is kept in the past tense because it is
 * why the file exists, not what the block looks like.
 *
 * Arms are `rowsPerLane x columnsPerLane` (lanes default to 8 by 8) or the full
 * `lanesX x lanesY x rowsPerLane x columnsPerLane`, optionally suffixed `@f16`
 * for accumulators and staged tiles in half precision.
 *
 * 🔴 THE POINT OF THE f16 ARMS IS REGISTERS, NOT ARITHMETIC. This kernel holds
 * `rowsPerLane * columnsPerLane` vec4 accumulators - one per matrix set, where
 * every other kernel here holds a quarter of that - so the register budget runs
 * out four times sooner, and the recorded sweep says 16 vec4 SPILLS and is then
 * slower than the kernel it replaced. Half-precision accumulators are half the
 * registers, which is the one thing that could move that ceiling.
 *
 * Every arm is checked against the first, because a tile the dispatch does not
 * match leaves rows unprojected and reads as a speedup.
 */
import {
  createAttentionProjectShader, attentionProjectTileRows, attentionProjectTileColumns,
} from "../../src/evoformer/attention.js";
import { float32ToFloat16Array } from "../../src/runtime/float16.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const sequences = Number(option(args, "sequences", "512"));
  const length = Number(option(args, "length", "59"));
  const channels = Number(option(args, "channels", "256"));
  const heads = Number(option(args, "heads", "8"));
  const rounds = Number(option(args, "rounds", "9"));
  const iterations = Number(option(args, "iterations", "8"));
  const armsSpec = option(args, "arms", "4x2,4x2@f16,2x2,8x2,8x2@f16,4x4,4x4@f16").split(",");
  const headDim = channels / heads;
  const width = heads * headDim;
  const rows = sequences * length;

  let state = 4242;
  const random = (n) => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 1) {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      out[i] = (state / 0x7fffffff - 0.5) * 0.2;
    }
    return out;
  };
  const storage = GPUBufferUsage.STORAGE;
  const upload = (data, usage = storage) => {
    const buffer = device.createBuffer({ size: data.byteLength, usage, mappedAtCreation: true });
    new data.constructor(buffer.getMappedRange()).set(data);
    buffer.unmap();
    return buffer;
  };

  // The packed weight layout the kernel reads: four channels x width matrices
  // and a gating bias, at the offsets the uniform names.
  const matrix = channels * width;
  const weightData = new Float32Array(4 * matrix + width);
  weightData.set(random(4 * matrix + width), 0);
  const weights = upload(weightData);
  const weightsHalf = device.features.has("shader-f16")
    ? upload(float32ToFloat16Array(weightData)) : weights;
  const source = upload(random(rows * channels));
  const outputs = ["query", "key", "value", "gate"].map(() => device.createBuffer({
    size: rows * width * 4, usage: storage | GPUBufferUsage.COPY_SRC,
  }));
  // batch, queries, channels, heads, head_dim, transpose, has_pair_bias, then
  // the offsets: query, key, value, gating weights, gating bias, and the three
  // this kernel does not read.
  const parameters = upload(new Uint32Array([
    sequences, length, channels, heads, headDim, 0, 0,
    0, matrix, 2 * matrix, 3 * matrix, 4 * matrix, 0, 0, 0, 0,
  ]), GPUBufferUsage.UNIFORM);

  const sampleRows = Math.min(rows, 96);
  const sampleFloats = sampleRows * width;
  const readback = device.createBuffer({
    size: sampleFloats * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const tailOffset = Math.max(0, rows - sampleRows) * width;
  const tailReadback = device.createBuffer({
    size: sampleFloats * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const arms = [];
  const results = [];
  for (const spec of armsSpec) {
    // `4x4@f16` is the accumulator's element; `4x4@f16/f16` narrows the WEIGHT
    // BUFFER too, which is bandwidth rather than registers: this kernel rereads
    // the whole weight set once per row tile, 944 of them at 512 MSA rows.
    const [tileSpec, precisionSpec = "f32"] = spec.split("@");
    const [precision, weightPrecision = "f32"] = precisionSpec.split("/");
    if ((precision !== "f32" || weightPrecision !== "f32")
      && !device.features.has("shader-f16")) {
      results.push({ arm: spec, skipped: "no shader-f16" });
      continue;
    }
    const parts = tileSpec.split("x").map(Number);
    const [lanesX, lanesY, rowsPerLane, columnsPerLane] = parts.length === 4
      ? parts : [8, 8, ...parts];
    if (columnsPerLane === undefined) throw new Error(`arm ${spec} is not a tile`);
    const tile = { lanesX, lanesY, rowsPerLane, columnsPerLane };
    const pipeline = await device.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          code: createAttentionProjectShader(tile, precision, weightPrecision),
        }),
        entryPoint: "main",
      },
    });
    arms.push({
      spec, pipeline,
      bindGroup: device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [source, weightPrecision === "f16" ? weightsHalf : weights, parameters, ...outputs]
          .map((buffer, binding) => ({ binding, resource: { buffer } })),
      }),
      x: Math.ceil(width / attentionProjectTileColumns(tile)),
      y: Math.ceil(rows / attentionProjectTileRows(tile)),
      tile: `${attentionProjectTileRows(tile)}x${attentionProjectTileColumns(tile)}`,
      times: [],
    });
  }

  const run = async (arm) => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(arm.pipeline);
    pass.setBindGroup(0, arm.bindGroup);
    for (let i = 0; i < iterations; i += 1) pass.dispatchWorkgroups(arm.x, arm.y);
    pass.end();
    const start = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return (performance.now() - start) / iterations;
  };
  const readCorner = async (buffer, offsetFloats) => {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(outputs[0], offsetFloats * 4, buffer, 0, sampleFloats * 4);
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const copy = new Float32Array(buffer.getMappedRange().slice(0));
    buffer.unmap();
    return copy;
  };

  for (const arm of arms) await run(arm);
  const reference = { head: null, tail: null };
  for (const arm of arms) {
    await run(arm);
    arm.head = await readCorner(readback, 0);
    arm.tail = await readCorner(tailReadback, tailOffset);
    if (reference.head === null) { reference.head = arm.head; reference.tail = arm.tail; }
  }
  for (let round = 0; round < rounds; round += 1) {
    for (const arm of arms) arm.times.push(await run(arm));
  }

  const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
  const relRms = (a, b) => {
    let error = 0; let scale = 0;
    for (let i = 0; i < a.length; i += 1) { error += (a[i] - b[i]) ** 2; scale += b[i] ** 2; }
    return Math.sqrt(error / Math.max(scale, 1e-30));
  };
  // Four matrices, each rows x channels x width.
  const macs = 4 * rows * channels * width;
  for (const arm of arms) {
    const ms = median(arm.times);
    results.push({
      arm: arm.spec, tile: arm.tile, workgroups: arm.x * arm.y,
      ms: Number(ms.toFixed(3)),
      gflops: Number((2 * macs / (ms / 1000) / 1e9).toFixed(1)),
      relRms: Number(relRms(arm.head, reference.head).toExponential(2)),
      tailRelRms: Number(relRms(arm.tail, reference.tail).toExponential(2)),
    });
  }
  results.sort((a, b) => (a.ms ?? 1e9) - (b.ms ?? 1e9));
  return { sequences, length, channels, heads, headDim, rows, macs, rounds, iterations, results };
}
