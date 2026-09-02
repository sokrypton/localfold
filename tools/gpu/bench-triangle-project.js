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
  const b = allocate(pairs * cH);
  const output = allocate(pairs * cZ);
  const readback = device.createBuffer({
    size: pairs * cZ * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  const arms = [];
  for (const spec of arms_spec) {
    const [rows, columns] = spec.split("x").map(Number);
    const shaders = createTriangleShaders(shape, "f32", offsets, 1e-5, "outgoing",
                                          "fast", { rows, columns });
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
    arms.push({
      spec,
      project: await build(shaders.projectAB, [z, mask, weights, a, b]),
      projectOut: await build(shaders.projectOutput, [z, x, weights, output]),
      times: { project: [], projectOut: [] },
    });
  }

  const time = async (kernel) => {
    const encoder = device.createCommandEncoder();
    for (let i = 0; i < iterations; i += 1) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(kernel.pipeline);
      pass.setBindGroup(0, kernel.bindGroup);
      pass.dispatchWorkgroups(kernel.x, kernel.y);
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

  // The output of project-out depends on every cell project-ab wrote, so one
  // readback per arm covers both kernels.
  const results = [];
  for (const arm of arms) {
    const encoder = device.createCommandEncoder();
    for (const kernel of [arm.project, arm.projectOut]) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(kernel.pipeline);
      pass.setBindGroup(0, kernel.bindGroup);
      pass.dispatchWorkgroups(kernel.x, kernel.y);
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
      relRmsVsFirst: Number(relRms[index].toExponential(2)),
    })),
  };
}
