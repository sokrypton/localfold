/**
 * The single track's q/k/v/gate projection, at several splits of its width.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-single-project.js --tokens=59
 *
 * WHY IT EXISTS. This kernel has one row a TOKEN where everything else in the
 * pairformer has one a pair, so a 59-residue chain gives it 59 workgroups and
 * 3,776 invocations - and it measured 163 GFLOP/s against the pair track's
 * ~1000. There are no more tokens, so the only axis left to split is the output
 * width, and how far that pays is a question about occupancy against the
 * LayerNorm each split repeats. Arms are the number of workgroups per token.
 */
import { createSingleAttentionShaders, packSingleAttentionWeights }
  from "../../src/af3/single-attention-webgpu.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const n = Number(option(args, "tokens", "59"));
  const channels = Number(option(args, "channels", "384"));
  const heads = Number(option(args, "heads", "16"));
  const dimension = Number(option(args, "dimension", "24"));
  const rounds = Number(option(args, "rounds", "11"));
  const iterations = Number(option(args, "iterations", "16"));
  const arms_spec = option(args, "arms", "1,2,3,6").split(",").map(Number);
  const width = heads * dimension;

  let state = 31337;
  const random = (count) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      out[i] = (state / 0x7fffffff - 0.5) * 0.2;
    }
    return out;
  };
  const weights = {
    layerNormScale: new Float32Array(channels).fill(1),
    layerNormOffset: new Float32Array(channels),
    qProjection: random(channels * width), qBias: random(width),
    kProjection: random(channels * width), vProjection: random(channels * width),
    gatingQuery: random(channels * width), outputProjection: random(width * channels),
  };
  const packed = packSingleAttentionWeights(weights);

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

  const single = upload(random(n * channels));
  const weightBuffer = upload(packed.data);
  const q = allocate(n * width);
  const k = allocate(n * width);
  const v = allocate(n * width);
  const gate = allocate(n * width);
  const readback = device.createBuffer({
    size: n * width * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  const arms = [];
  for (const splits of arms_spec) {
    const sources = createSingleAttentionShaders(
      { n, channels, heads, dimension, projectSplits: splits },
      packed.offsets, 1e-5, "fast");
    const pipeline = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: sources.project }),
                 entryPoint: "main" },
    });
    arms.push({
      splits, pipeline,
      bindGroup: device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [single, weightBuffer, q, k, v, gate].map((buffer, binding) => ({
          binding, resource: { buffer } })),
      }),
      groups: n * sources.projectSplits,
      times: [],
    });
  }

  const time = async (arm) => {
    const encoder = device.createCommandEncoder();
    for (let i = 0; i < iterations; i += 1) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(arm.pipeline);
      pass.setBindGroup(0, arm.bindGroup);
      pass.dispatchWorkgroups(arm.groups);
      pass.end();
    }
    const start = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return (performance.now() - start) / iterations;
  };

  for (const arm of arms) await time(arm);
  for (let round = 0; round < rounds; round += 1) {
    for (const arm of arms) arm.times.push(await time(arm));
  }

  const results = [];
  for (const arm of arms) {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(arm.pipeline);
    pass.setBindGroup(0, arm.bindGroup);
    pass.dispatchWorkgroups(arm.groups);
    pass.end();
    encoder.copyBufferToBuffer(q, 0, readback, 0, n * width * 4);
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
    n, channels, width,
    arms: arms.map((arm, index) => ({
      splits: arm.splits,
      workgroups: arm.groups,
      ms: Number(median(arm.times).toFixed(4)),
      relRmsVsFirst: Number(relRms[index].toExponential(2)),
    })),
  };
}
