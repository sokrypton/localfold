/**
 * The two triangle projection kernels alone, at several register blocks.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-triangle-project.js --tokens=59
 *     node tools/gpu-chrome.mjs tools/gpu/bench-triangle-project.js --arms=16x16,32x16
 *
 * WHY IT EXISTS. `tri.project` and `tri.project-out` are 104 ms of a 480 ms
 * pairformer pass, and the grid attention's projections are the same shape
 * again. Both are cooperatively tiled already, so what is left to move is the
 * register block - how many pair rows by hidden channels one invocation owns -
 * and that is a ratio of workgroup reads to multiply-adds rather than anything
 * a profile of the whole trunk can resolve. An arm here costs about a second.
 *
 * Arms are `rows x columns` of the workgroup tile, both multiples of 8. Every
 * arm is checked against the first for a bitwise-comparable result, because a
 * tile the dispatch does not match computes a fraction of the rows and reads as
 * a speedup.
 */
import { createTriangleShaders } from "../../src/triangle/shaders.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const OFFSET_ORDER = [
  ["layerNormInWeight", (c) => c.cZ], ["layerNormInBias", (c) => c.cZ],
  ["linearAPWeight", (c) => c.cZ * c.cH], ["linearAPBias", (c) => c.cH],
  ["linearAGWeight", (c) => c.cZ * c.cH], ["linearAGBias", (c) => c.cH],
  ["linearBPWeight", (c) => c.cZ * c.cH], ["linearBPBias", (c) => c.cH],
  ["linearBGWeight", (c) => c.cZ * c.cH], ["linearBGBias", (c) => c.cH],
  ["layerNormOutWeight", (c) => c.cH], ["layerNormOutBias", (c) => c.cH],
  ["linearZWeight", (c) => c.cH * c.cZ], ["linearZBias", (c) => c.cZ],
  ["linearGWeight", (c) => c.cZ * c.cZ], ["linearGBias", (c) => c.cZ],
];

export async function main(device, args) {
  const length = Number(option(args, "tokens", "59"));
  const cZ = Number(option(args, "channels", "128"));
  const cH = cZ;
  const rounds = Number(option(args, "rounds", "9"));
  const iterations = Number(option(args, "iterations", "8"));
  const arms_spec = option(args, "arms", "16x16,32x16,32x32,64x16,16x32").split(",");
  const pairs = length * length;
  const shape = { length, cZ, cHidden: cH };

  const offsets = {};
  let total = 0;
  for (const [name, size] of OFFSET_ORDER) {
    offsets[name] = total;
    total += size({ cZ, cH });
  }
  let state = 7777;
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
    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();
    return buffer;
  };
  const allocate = (floats, usage = storage | GPUBufferUsage.COPY_SRC) =>
    device.createBuffer({ size: floats * 4, usage });

  const weights = upload(random(total));
  const z = upload(random(pairs * cZ));
  const x = upload(random(pairs * cH));
  const mask = upload(new Float32Array(pairs).fill(1));
  const a = allocate(pairs * cH);
  const contracted = allocate(pairs * cH);
  const b = allocate(pairs * cH);
  const output = allocate(pairs * cZ);
  const readback = device.createBuffer({
    size: pairs * cZ * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  // 🔴 THE :drop ARMS RETURN WRONG NUMBERS ON PURPOSE, to price one class of
  // instruction rather than to compute anything. "barrier" removes the two
  // workgroup barriers from the k loop (a race, and wrong), "x" replaces the
  // source tile's reads with a constant, "w" the weight tile's. Their relRms is
  // expected to be large. Never read a result out of one.
  const surgery = {
    barrier: [/\n\s*workgroupBarrier\(\);/g, ""],
    x: [/(\w+)\[r\] = tile_(?:source|x|z)\[r \* 64u \+ local\.y \* 8u \+ k\];/g,
        "$1[r] = f32(k + r) * 1e-6;"],
    w: [/let packed = tile_weight\[[^\]]+\];/g, "let packed = vec4<f32>(f32(k) * 1e-6);"],
    w2: [/let (\w+)_w = tile_(\w+)_weight\[slot\];/g, "let $1_w = f32(slot) * 1e-6;"],
  };
  const arms = [];
  for (const spec of arms_spec) {
    const [shapeSpec, drop] = spec.split(":");
    // "32x16" sets both tiles; "32x16@32x32" sets the projection's and the
    // contraction's separately, since they do not peak at the same shape.
    const [projectSpec, contractSpec] = shapeSpec.split("@");
    const [rows, columns] = projectSpec.split("x").map(Number);
    const [contractRows, contractColumns] =
      (contractSpec ?? projectSpec).split("x").map(Number);
    // The contraction takes the same tile shape, so one arm prices both.
    const shaders = createTriangleShaders(shape, "f32", offsets, 1e-5, "outgoing",
                                          "fast", { rows, columns }, false,
                                          { rows: contractRows, columns: contractColumns });
    if (drop) {
      const [pattern, replacement] = surgery[drop];
      let matched = 0;
      for (const name of ["projectAB", "projectOutput"]) {
        const patched = shaders[name].replace(pattern, replacement);
        if (patched !== shaders[name]) matched += 1;
        shaders[name] = patched;
      }
      if (matched !== 2) throw new Error(`arm ${spec} patched ${matched} of the two kernels`);
    }
    const build = async (source, buffers) => {
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
        x: Math.ceil(cZ / columns), y: Math.ceil(pairs / rows),
      };
    };
    const contract = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: shaders.contract }),
                 entryPoint: "main" },
    });
    arms.push({
      spec,
      project: await build(shaders.projectAB, [z, mask, weights, a, b]),
      projectOut: await build(shaders.projectOutput, [z, x, weights, output]),
      contract: {
        pipeline: contract,
        bindGroup: device.createBindGroup({
          layout: contract.getBindGroupLayout(0),
          entries: [a, b, contracted].map((buffer, binding) => ({
            binding, resource: { buffer } })),
        }),
        x: Math.ceil(length / contractColumns), y: Math.ceil(length / contractRows), z: cH,
      },
      times: { project: [], projectOut: [], contract: [] },
    });
  }

  const time = async (kernel) => {
    const encoder = device.createCommandEncoder();
    for (let i = 0; i < iterations; i += 1) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(kernel.pipeline);
      pass.setBindGroup(0, kernel.bindGroup);
      pass.dispatchWorkgroups(kernel.x, kernel.y, kernel.z ?? 1);
      pass.end();
    }
    const start = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return (performance.now() - start) / iterations;
  };

  for (const arm of arms) {
    await time(arm.project); await time(arm.projectOut); await time(arm.contract);
  }
  for (let round = 0; round < rounds; round += 1) {
    for (const arm of arms) {
      arm.times.project.push(await time(arm.project));
      arm.times.projectOut.push(await time(arm.projectOut));
      arm.times.contract.push(await time(arm.contract));
    }
  }

  // The output of project-out depends on every cell project-ab wrote, so one
  // readback per arm covers both kernels.
  const results = [];
  for (const arm of arms) {
    const encoder = device.createCommandEncoder();
    for (const kernel of [arm.project, arm.projectOut]) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(kernel.pipeline);
      pass.setBindGroup(0, kernel.bindGroup);
      pass.dispatchWorkgroups(kernel.x, kernel.y, kernel.z ?? 1);
      pass.end();
    }
    encoder.copyBufferToBuffer(output, 0, readback, 0, pairs * cZ * 4);
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

  const median = (values) => [...values].sort((p, q) => p - q)[values.length >> 1];
  return {
    length, channels: cZ, pairs,
    arms: arms.map((arm, index) => ({
      arm: arm.spec,
      workgroups: arm.project.x * arm.project.y,
      project: Number(median(arm.times.project).toFixed(3)),
      projectOut: Number(median(arm.times.projectOut).toFixed(3)),
      contract: Number(median(arm.times.contract).toFixed(3)),
      relRmsVsFirst: Number(relRms[index].toExponential(2)),
    })),
  };
}
