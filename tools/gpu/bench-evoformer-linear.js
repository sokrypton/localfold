/**
 * AF2's dense projection alone, at several register blocks, in one process.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-evoformer-linear.js
 *     node tools/gpu-chrome.mjs tools/gpu/bench-evoformer-linear.js --shape=second
 *     node tools/gpu-chrome.mjs tools/gpu/bench-evoformer-linear.js --arms=legacy,4x8,8x8
 *
 * WHY IT EXISTS. `createLinearShader` is the workhorse of the whole AF2 side -
 * both halves of every transition, every attention projection's sibling, the
 * confidence head and all four structure-module kernels - and at 512 MSA rows
 * its two transition dispatches are about 28% of an evoformer block. Moving it
 * through `profile-af2-block.js` costs a minute an arm and cannot separate a
 * change in the kernel from a change in how the block schedules around it.
 * Here an arm is about a second.
 *
 * Arms are `rowsPerLane x columnsPerLane` (lanes default to 8 by 8) or the full
 * `lanesX x lanesY x rowsPerLane x columnsPerLane`, or `legacy` for
 * the two-row scalar-source kernel these replaced - kept verbatim so a claim
 * about the change is a measurement and not two numbers from two processes.
 * `:noload` arms replace one staged operand with a constant to price its read;
 * they compute nothing and their relRMS is expected to be large.
 *
 * Every arm is checked against the first, because a tile the dispatch does not
 * match leaves rows unprojected and reads as a speedup.
 */
import { createLinearShader } from "../../src/evoformer/transition.js";
import { float32ToFloat16Array } from "../../src/runtime/float16.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

// The kernel as it stood before the source tile was transposed: two rows to an
// invocation, read one float at a time. Its tile is 16 rows by 64 columns.
const LEGACY_SHADER = `
struct MatmulParameters {
  rows: u32,
  inner: u32,
  columns: u32,
  weight_offset: u32,
  bias_offset: u32,
  activation: u32,
  padding: vec2<u32>,
};
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> parameters: MatmulParameters;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

var<workgroup> tile_source: array<f32, 128>;
// 🔴 A THREAD'S EIGHT COLUMNS ARE ADJACENT HERE, WHICH IS NOT HOW IT READS THEM
// OUT. The eight output columns an invocation owns are strided by eight in the
// OUTPUT - column + block * 8 - but nothing says the staged copy has to match:
// laid out per thread, its eight weights are two vec4 reads where they were
// eight scalar ones, and the inner loop goes from ten workgroup reads to four
// for the same sixteen multiply-adds. tools/gpu/probe-alu.js puts workgroup
// reads at 394 billion a second against 580 billion vec4 multiply-adds, so for
// this kernel the reads were the larger term.
var<workgroup> tile_weight: array<vec4<f32>, 128>;

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let row = group.y * 16u + local.y;
  let second_row = row + 8u;
  let column = group.x * 64u + local.x;
  let tile_index = local.y * 8u + local.x;
  var value_low = vec4<f32>(0.0);
  var value_high = vec4<f32>(0.0);
  var second_value_low = vec4<f32>(0.0);
  var second_value_high = vec4<f32>(0.0);

  for (var k0 = 0u; k0 < parameters.inner; k0 += 8u) {
    let source_k = k0 + local.x;
    let weight_k = k0 + local.y;
    tile_source[tile_index] = 0.0;
    tile_source[tile_index + 64u] = 0.0;
    if (row < parameters.rows && source_k < parameters.inner) {
      tile_source[tile_index] = source[row * parameters.inner + source_k];
    }
    if (second_row < parameters.rows && source_k < parameters.inner) {
      tile_source[tile_index + 64u] = source[second_row * parameters.inner + source_k];
    }
    var staged_low = vec4<f32>(0.0);
    var staged_high = vec4<f32>(0.0);
    for (var column_block = 0u; column_block < 8u; column_block += 1u) {
      let output_column = column + column_block * 8u;
      var value = 0.0;
      if (output_column < parameters.columns && weight_k < parameters.inner) {
        value = weights[
          parameters.weight_offset + weight_k * parameters.columns + output_column
        ];
      }
      if (column_block < 4u) { staged_low[column_block] = value; }
      else { staged_high[column_block - 4u] = value; }
    }
    tile_weight[local.y * 16u + local.x * 2u] = staged_low;
    tile_weight[local.y * 16u + local.x * 2u + 1u] = staged_high;
    workgroupBarrier();
    for (var k = 0u; k < 8u; k += 1u) {
      let source_value = tile_source[local.y * 8u + k];
      let second_source_value = tile_source[local.y * 8u + k + 64u];
      let weight_base = k * 16u + local.x * 2u;
      let weight_low = tile_weight[weight_base];
      let weight_high = tile_weight[weight_base + 1u];
      value_low += source_value * weight_low;
      value_high += source_value * weight_high;
      second_value_low += second_source_value * weight_low;
      second_value_high += second_source_value * weight_high;
    }
    workgroupBarrier();
  }

  if (row < parameters.rows) {
    for (var column_block = 0u; column_block < 8u; column_block += 1u) {
      let output_column = column + column_block * 8u;
      if (output_column < parameters.columns) {
        let values = select(value_low, value_high, column_block >= 4u);
        var value = values[column_block % 4u];
        value += weights[parameters.bias_offset + output_column];
        if (parameters.activation == 1u) { value = max(value, 0.0); }
        output[row * parameters.columns + output_column] = value;
      }
    }
  }
  if (second_row < parameters.rows) {
    for (var column_block = 0u; column_block < 8u; column_block += 1u) {
      let output_column = column + column_block * 8u;
      if (output_column < parameters.columns) {
        let values = select(second_value_low, second_value_high, column_block >= 4u);
        var value = values[column_block % 4u];
        value += weights[parameters.bias_offset + output_column];
        if (parameters.activation == 1u) { value = max(value, 0.0); }
        output[second_row * parameters.columns + output_column] = value;
      }
    }
  }
}`;
const LEGACY_TILE = { rows: 16, columns: 64 };

const SHAPES = {
  // The MSA transition at 512 rows of a 59-residue alignment, both halves.
  first: { rows: 512 * 59, inner: 256, columns: 1024, activation: 1 },
  second: { rows: 512 * 59, inner: 1024, columns: 256, activation: 0 },
  // What the structure module and confidence head ask for: a few hundred rows,
  // where the row tile is most of the occupancy rather than none of it.
  single: { rows: 59, inner: 384, columns: 384, activation: 1 },
  // A long chain's pair transition, which is the same kernel again.
  pair: { rows: 150 * 150, inner: 128, columns: 512, activation: 1 },
};

export async function main(device, args) {
  const shapeName = option(args, "shape", "first");
  const shape = SHAPES[shapeName];
  if (!shape) throw new Error(`unknown shape ${shapeName}; try ${Object.keys(SHAPES).join(", ")}`);
  const { inner, columns, activation } = shape;
  // The row count is the axis the tile choice turns on, so it overrides.
  const rows = Number(option(args, "rows", String(shape.rows)));
  const rounds = Number(option(args, "rounds", "9"));
  const iterations = Number(option(args, "iterations", "8"));
  const armsSpec = option(args, "arms", "legacy,4x8,8x8,4x4,8x4,4x16,12x8,16x8").split(",");

  let state = 13337;
  const random = (n) => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 1) {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      out[i] = (state / 0x7fffffff - 0.5) * 0.2;
    }
    return out;
  };

  const storage = GPUBufferUsage.STORAGE;
  // 🔴 THE UNIFORM IS Uint32Array AND THE ACTIVATIONS ARE Float32Array, so the
  // staging view has to follow the data. Copying a Uint32Array THROUGH a
  // Float32Array converts the numbers instead of the bytes - `rows` arrives as
  // the bit pattern of 59.0, the k loop runs a billion times and the page hangs
  // with no error to report.
  const upload = (data, usage = storage) => {
    const buffer = device.createBuffer({ size: data.byteLength, usage, mappedAtCreation: true });
    new data.constructor(buffer.getMappedRange()).set(data);
    buffer.unmap();
    return buffer;
  };

  // weights holds [weight | bias] so one binding serves both, as the kernel wants.
  const weightData = new Float32Array(inner * columns + columns);
  weightData.set(random(inner * columns), 0);
  weightData.set(random(columns), inner * columns);
  const weights = upload(weightData);
  // ...and the same values as halves, for the arms that bind them that way.
  const weightsHalf = device.features.has("shader-f16")
    ? upload(float32ToFloat16Array(weightData)) : weights;
  const source = upload(random(rows * inner));
  const output = device.createBuffer({
    size: rows * columns * 4, usage: storage | GPUBufferUsage.COPY_SRC,
  });
  const parameters = upload(new Uint32Array([
    rows, inner, columns, 0, inner * columns, activation, 0, 0,
  ]), GPUBufferUsage.UNIFORM);
  // Only a corner is read back: enough to catch a tile that skips rows without
  // moving 120 MB per arm.
  const sampleRows = Math.min(rows, 96);
  const sampleFloats = sampleRows * columns;
  const readback = device.createBuffer({
    size: sampleFloats * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  // ...and a stripe from the far end, where a dispatch that is short by a tile
  // leaves the output buffer at whatever it was allocated with.
  const tailOffset = Math.max(0, rows - sampleRows) * columns;
  const tailReadback = device.createBuffer({
    size: sampleFloats * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const surgery = {
    // Prices the source tile's read by removing the dependency on it.
    source: [/let (s_\d+_\d+) = tile_source\[[^\]]+\];/g, "let $1 = vec4<f32>(1e-6);"],
    weight: [/let (w_\d+_\d+) = tile_weight\[[^\]]+\];/g, "let $1 = vec4<f32>(1e-6);"],
  };

  const results = [];
  const arms = [];
  for (const spec of armsSpec) {
    // `8x8@f16` names a tile and the element its k loop works in; the suffix is
    // optional and `f32` is what every arm meant before it existed.
    const [tileAndPrecision, drop] = spec.split(":");
    // `8x8@f16` is the k loop's element; `8x8@f16/f16` narrows the WEIGHT
    // BUFFER as well, which is a bandwidth question rather than a register one.
    const [tileSpec, precisionSpec = "f32"] = tileAndPrecision.split("@");
    const [precision, weightPrecision = "f32"] = precisionSpec.split("/");
    if ((precision !== "f32" || precisionSpec.includes("/f16"))
      && !device.features.has("shader-f16")) {
      results.push({ arm: spec, skipped: "no shader-f16" });
      continue;
    }
    let shader; let tile;
    if (tileSpec === "legacy") {
      if (precision !== "f32") throw new Error("the legacy kernel has no precision option");
      shader = LEGACY_SHADER;
      tile = LEGACY_TILE;
    } else {
      const parts = tileSpec.split("x").map(Number);
      if (parts.some((value) => !value)) throw new Error(`arm ${spec} is not a tile`);
      const [lanesX, lanesY, rowsPerLane, columnsPerLane] = parts.length === 4
        ? parts : [8, 8, ...parts];
      if (columnsPerLane === undefined) throw new Error(`arm ${spec} is not a tile`);
      const descriptor = { lanesX, lanesY, rowsPerLane, columnsPerLane };
      shader = createLinearShader(descriptor, false, precision, weightPrecision);
      tile = {
        rows: descriptor.lanesY * descriptor.rowsPerLane,
        columns: descriptor.lanesX * descriptor.columnsPerLane,
      };
    }
    if (drop) {
      const [pattern, replacement] = surgery[drop] ?? [];
      if (!pattern) throw new Error(`arm ${spec} names no known surgery`);
      const patched = shader.replace(pattern, replacement);
      if (patched === shader) throw new Error(`arm ${spec} patched nothing`);
      shader = patched;
    }
    const pipeline = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: shader }), entryPoint: "main" },
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [source, weightPrecision === "f16" ? weightsHalf : weights, parameters, output]
        .map((buffer, binding) => ({
        binding, resource: { buffer },
      })),
    });
    arms.push({
      spec, pipeline, bindGroup, tile,
      x: Math.ceil(columns / tile.columns), y: Math.ceil(rows / tile.rows),
      times: [],
    });
  }

  const readCorner = async (buffer, offsetFloats) => {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(output, offsetFloats * 4, buffer, 0, sampleFloats * 4);
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const copy = new Float32Array(buffer.getMappedRange().slice(0));
    buffer.unmap();
    return copy;
  };

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

  // One untimed pass each, so no arm pays for its own warm-up.
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

  const median = (values) => {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
  };
  const relRms = (a, b) => {
    let error = 0; let scale = 0;
    for (let i = 0; i < a.length; i += 1) { error += (a[i] - b[i]) ** 2; scale += b[i] ** 2; }
    return Math.sqrt(error / Math.max(scale, 1e-30));
  };

  const macs = rows * inner * columns;
  for (const arm of arms) {
    const ms = median(arm.times);
    results.push({
      arm: arm.spec,
      tile: `${arm.tile.rows}x${arm.tile.columns}`,
      workgroups: arm.x * arm.y,
      ms: Number(ms.toFixed(3)),
      gflops: Number((2 * macs / (ms / 1000) / 1e9).toFixed(1)),
      relRms: Number(relRms(arm.head, reference.head).toExponential(2)),
      tailRelRms: Number(relRms(arm.tail, reference.tail).toExponential(2)),
      spread: Number((Math.max(...arm.times) - Math.min(...arm.times)).toFixed(3)),
    });
  }
  results.sort((a, b) => a.ms - b.ms);
  return { shape: shapeName, rows, inner, columns, macs, rounds, iterations, results };
}
