/**
 * The outer product mean's contraction alone, at several (i, j) blocks.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-opm.js --tokens=59 --rows=1024
 *     node tools/gpu-chrome.mjs tools/gpu/bench-opm.js --arms=4x4,8x4,4x4@64
 *
 * WHY IT EXISTS. This is the largest kernel in the trunk once an alignment is
 * real - 113 ms of a 302 ms MSA stack at 1024 rows - and it has two costs that
 * pull in different directions: the sweep over sequences, which only an i-by-j
 * BLOCK of pairs divides, and the output projection, which any tile of pairs
 * divides. Where they balance is measured. An arm here costs about a second
 * against bench-trunk.js's minute at that depth.
 *
 * Arms are `i x j` of the pair block, optionally `@cells` for the chunk of
 * products held in workgroup memory. Every arm is checked against the first.
 */
import { createOuterProductMeanShaders } from "../../src/af3/outer-product-mean-webgpu.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const tokens = Number(option(args, "tokens", "59"));
  const sequences = Number(option(args, "rows", "1024"));
  const msaChannels = Number(option(args, "msa-channels", "64"));
  const outerChannels = Number(option(args, "outer", "32"));
  const pairChannels = Number(option(args, "pair", "128"));
  const rounds = Number(option(args, "rounds", "7"));
  const arms_spec = option(args, "arms", "4x4,8x4,4x4@64,8x4@64,2x4,4x2").split(",");
  const pairs = tokens * tokens;
  const rows = sequences * tokens;
  const products = outerChannels * outerChannels;

  const sizes = {
    layerNormInputScale: msaChannels, layerNormInputOffset: msaChannels,
    leftProjection: msaChannels * outerChannels, rightProjection: msaChannels * outerChannels,
    outputW: products * pairChannels, outputB: pairChannels,
  };
  const offsets = {};
  let total = 0;
  for (const [name, size] of Object.entries(sizes)) { offsets[name] = total; total += size; }

  let state = 5150;
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
  const left = upload(random(rows * outerChannels));
  const right = upload(random(rows * outerChannels));
  const mask = upload(new Float32Array(sequences * tokens).fill(1));
  const counts = allocate(pairs);
  const output = allocate(pairs * pairChannels);
  const readback = device.createBuffer({
    size: pairs * pairChannels * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  const build = async (source, buffers, x, y = 1) => {
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
      x, y,
    };
  };

  const arms = [];
  for (const spec of arms_spec) {
    const [block, cells] = spec.split("@");
    const [blockI, blockJ] = block.split("x").map(Number);
    const shape = { sequences, tokens, msaChannels, outerChannels, pairChannels,
                    blockI, blockJ, cellChunk: cells ? Number(cells) : undefined };
    const sources = createOuterProductMeanShaders(shape, offsets, 1e-5, "fast");
    const groups = Math.ceil(tokens / sources.blockI) * sources.blocksPerRow;
    arms.push({
      spec,
      counts: await build(sources.counts, [mask, counts], Math.ceil(pairs / 64)),
      contract: await build(sources.contract, [left, right, counts, weights, output], groups),
      times: [],
    });
  }

  const time = async (arm) => {
    const encoder = device.createCommandEncoder();
    for (const kernel of [arm.counts, arm.contract]) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(kernel.pipeline);
      pass.setBindGroup(0, kernel.bindGroup);
      pass.dispatchWorkgroups(kernel.x, kernel.y);
      pass.end();
    }
    const start = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return performance.now() - start;
  };

  for (const arm of arms) await time(arm);
  for (let round = 0; round < rounds; round += 1) {
    for (const arm of arms) arm.times.push(await time(arm));
  }

  const results = [];
  for (const arm of arms) {
    await time(arm);
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(output, 0, readback, 0, pairs * pairChannels * 4);
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

  const median = (values) => [...values].sort((a, b) => a - b)[values.length >> 1];
  return {
    tokens, sequences, pairs,
    arms: arms.map((arm, index) => ({
      arm: arm.spec,
      workgroups: arm.contract.x,
      ms: Number(median(arm.times).toFixed(2)),
      relRmsVsFirst: Number(relRms[index].toExponential(2)),
    })),
  };
}
