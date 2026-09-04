/**
 * The flash attention kernel alone, at several key-chunk sizes, in one process.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-msa-attention.js
 *     node tools/gpu-chrome.mjs tools/gpu/bench-msa-attention.js --sequences=1024
 *     node tools/gpu-chrome.mjs tools/gpu/bench-msa-attention.js --shape=row
 *
 * WHY IT EXISTS. `msa-column-attention.flash` is the largest single kernel in
 * an AF2 evoformer block at depth - 21 ms of 118 at 512 rows, and quadratic in
 * SEQUENCES where everything else in the block is linear, so it grows fastest
 * with the thing a real alignment adds. Nothing measured it on its own:
 * check-attention-variants.js times a whole AttentionGpu.run (projections
 * included) and profile-af2-block.js costs a minute and cannot be compared
 * across processes, where this machine drifts up to 3.2x.
 *
 * Arms are key-chunk sizes - how many keys a workgroup stages before the inner
 * loop runs over them - plus `auto` for whatever the shader picks itself. The
 * chunk is the only free parameter the kernel has: it trades global loads
 * against workgroup memory, and workgroup memory against how many workgroups a
 * core can hold.
 *
 * Every arm is checked against the first. They are not expected to be bitwise
 * equal - a different chunk reassociates the online softmax - so the report
 * carries relRMS and a rounding envelope rather than an equality.
 */
import { createAttentionRegisterFlashShader } from "../../src/evoformer/attention.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

function noise(count, seed) {
  const values = new Float32Array(count);
  let state = seed >>> 0;
  for (let index = 0; index < count; index += 1) {
    state = (state * 1664525 + 1013904223) >>> 0;
    values[index] = (state / 4294967296) - 0.5;
  }
  return values;
}

export async function main(device, args) {
  const sequences = Number(option(args, "sequences", "512"));
  const length = Number(option(args, "length", "59"));
  const channels = Number(option(args, "channels", "256"));
  const heads = Number(option(args, "heads", "8"));
  const rounds = Number(option(args, "rounds", "9"));
  const iterations = Number(option(args, "iterations", "4"));
  const armsSpec = option(args, "arms", "auto,8,16,32,64").split(",");
  // Column attention attends over SEQUENCES for each residue; row attention
  // attends over residues for each sequence. Same kernel, transposed shapes.
  const shape = option(args, "shape", "column");
  const [batch, queries] = shape === "column" ? [length, sequences] : [sequences, length];
  const headDim = channels / heads;
  if (!Number.isInteger(headDim) || headDim % 4 !== 0) {
    throw new Error(`channels/heads must be a multiple of 4; got ${headDim}`);
  }

  const elements = batch * queries * channels;
  const storage = GPUBufferUsage.STORAGE;
  const upload = (data, usage = storage) => {
    const buffer = device.createBuffer({ size: data.byteLength, usage, mappedAtCreation: true });
    new data.constructor(buffer.getMappedRange()).set(data);
    buffer.unmap();
    return buffer;
  };

  const query = upload(noise(elements, 1));
  const key = upload(noise(elements, 2));
  const value = upload(noise(elements, 3));
  const gate = upload(noise(elements, 4));
  // Ragged on purpose: a mask of all ones never exercises the penalty.
  const maskValues = new Float32Array(batch * queries);
  for (let index = 0; index < maskValues.length; index += 1) {
    maskValues[index] = index % 17 === 0 ? 0 : 1;
  }
  const mask = upload(maskValues);
  const pairBias = upload(new Float32Array(1));
  const parameters = upload(new Uint32Array([
    batch, queries, channels, heads, headDim,
    shape === "column" ? 1 : 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, channels,
  ]), GPUBufferUsage.UNIFORM);
  const output = device.createBuffer({
    size: elements * 4, usage: storage | GPUBufferUsage.COPY_SRC,
  });
  const sampleFloats = Math.min(elements, 64 * channels);
  const readback = device.createBuffer({
    size: sampleFloats * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const arms = [];
  for (const spec of armsSpec) {
    // `32/g4v` is a key chunk, then the keys sharing one rescale (`g<n>`) and
    // whether the q.k reduction runs through a vec4 (`v`).
    const [armSpec, drop] = spec.split(":");
    const [chunkSpec, shapeSpec = ""] = armSpec.split("/");
    const chunk = chunkSpec === "auto" ? undefined : Number(chunkSpec);
    if (chunkSpec !== "auto" && (!Number.isSafeInteger(chunk) || chunk < 1)) {
      throw new Error(`arm ${spec} is not a key chunk`);
    }
    const group = Number(shapeSpec.match(/g(\d+)/)?.[1] ?? 1);
    const vectorScore = shapeSpec.includes("v");
    const lazyRescale = shapeSpec.includes("l");
    const queriesPerLane = Number(shapeSpec.match(/q(\d+)/)?.[1] ?? 1);
    const precision = shapeSpec.includes("h") ? "f16" : shapeSpec.includes("c") ? "chunk16" : "f32";
    if (precision !== "f32" && !device.features.has("shader-f16")) { results.push({ arm: spec, skipped: "no shader-f16" }); continue; }
    let shader = createAttentionRegisterFlashShader(headDim, chunk, { group, vectorScore, lazyRescale, queriesPerLane, precision });
    // `:nokey` and friends remove one term to price it. They compute the wrong
    // answer on purpose - a large relRMS is the expected report, and the number
    // that matters is the millisecond one.
    if (drop) {
      const surgery = {
        // The staged key and value reads, priced by removing the dependency.
        nokey: [/key_chunk\[staged[^\]]*\]/g, "vec4<f32>(1e-6)"],
        noval: [/value_chunk\[staged[^\]]*\]/g, "vec4<f32>(1e-6)"],
        // The staging loop's own global loads.
        nostage: [/(key_chunk\[index\] = )key\[k_base\];/g, "$1vec4<f32>(1e-6);"],
        // Both transcendentals.
        noexp: [/exp\(/g, "noexp("],
        // The per-key global mask load.
        nomask: [/mask\[mask_index\(batch_index, [^;]*?\)\]/g, "1.0"],
      };
      const [pattern, replacement] = surgery[drop] ?? [];
      if (!pattern) throw new Error(`arm ${spec} names no known surgery`);
      const patched = shader.replace(pattern, replacement);
      if (patched === shader) throw new Error(`arm ${spec} patched nothing`);
      shader = drop === "noexp"
        ? patched.replace("@compute", "fn noexp(x: f32) -> f32 { return x * 1.0001; }\n@compute")
        : patched;
      // 🔴 REMOVING A TERM REMOVES ITS BINDING, and `layout: "auto"` then builds
      // a layout the bind group does not match - which reports as a validation
      // error about entry 4 rather than as anything to do with the surgery. A
      // dead branch that names every buffer keeps the layout the shape the
      // reference arm has, and never runs.
      shader = shader.replace("  let local = local_id.x;", `  let local = local_id.x;
  if (p.batch == 4294967295u) {
    output[0] = query[0] + key[0] + value[0] + gate[0]
      + vec4<f32>(mask[0]) + vec4<f32>(pair_bias[0]);
  }`);
    }
    const pipeline = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: shader }), entryPoint: "main" },
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [query, key, value, gate, mask, pairBias, parameters, output]
        .map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    arms.push({ spec, pipeline, bindGroup, queriesPerLane, times: [] });
  }

  const run = async (arm) => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(arm.pipeline);
    pass.setBindGroup(0, arm.bindGroup);
    for (let i = 0; i < iterations; i += 1) {
      pass.dispatchWorkgroups(Math.ceil(queries / (64 * arm.queriesPerLane)), batch, heads);
    }
    pass.end();
    const start = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return (performance.now() - start) / iterations;
  };
  const readCorner = async () => {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(output, 0, readback, 0, sampleFloats * 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    return copy;
  };

  for (const arm of arms) await run(arm);
  let reference = null;
  for (const arm of arms) {
    await run(arm);
    arm.sample = await readCorner();
    if (reference === null) reference = arm.sample;
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
  // Query-key and attention-value, both batch*heads*queries*queries*headDim.
  const macs = 2 * batch * heads * queries * queries * headDim;
  const results = arms.map((arm) => {
    const ms = median(arm.times);
    return {
      arm: arm.spec,
      ms: Number(ms.toFixed(3)),
      gflops: Number((2 * macs / (ms / 1000) / 1e9).toFixed(1)),
      relRms: Number(relRms(arm.sample, reference).toExponential(2)),
      spread: Number((Math.max(...arm.times) - Math.min(...arm.times)).toFixed(3)),
    };
  }).sort((a, b) => a.ms - b.ms);
  return { shape, batch, queries, channels, heads, headDim, macs, rounds, iterations, results };
}
