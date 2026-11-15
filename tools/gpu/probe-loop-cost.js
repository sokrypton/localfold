// What a loop, a storage read and a workgroup read each cost in this browser's shader compiler: the denoiser's dense
// inner loop (ffw-out's) cut into arms, each as a loop and unrolled eight to an iteration, timed on the GPU's clock.
//   node tools/gpu-chrome.mjs tools/gpu/probe-loop-cost.js
//   LOCALFOLD_GPU_FIREFOX=1 node tools/gpu-chrome.mjs tools/gpu/probe-loop-cost.js
// Compare a browser's arm with the other browser's SAME arm: the ratio that changes between arms names the construct
// (docs/FIREFOX.md, "Why the same kernel is slower"). `--groups=` workgroups a dispatch, `--rounds=`.
const K = 384, C = 768, REPS = 4;
const BODIES = {
  // ffw-out's inner loop: one workgroup read, one storage read, one vector multiply-add.
  gemm: (i) => `acc += vec4<f32>(wt[${i}]) * f32(weights[c + (${i}) * ${C}u]);`,
  // the same without the storage buffer, and without the workgroup array
  "workgroup-only": (i) => `acc += vec4<f32>(wt[${i}]) * scale;`,
  "storage-only": (i) => `acc += vec4<f32>(f32(weights[c + (${i}) * ${C}u]));`,
  // no memory at all: what the loop itself costs
  arithmetic: (i) => `acc = acc * scale + vec4<f32>(f32(${i}));`,
};
const loopOf = (body, unroll) => {
  const lines = [];
  for (let u = 0; u < unroll; u += 1) lines.push("      " + body(u === 0 ? "i" : `i + ${u}u`));
  return `for (var i = 0u; i < ${K}u; i += ${unroll}u) {\n${lines.join("\n")}\n    }`;
};
const shader = (body, unroll) => `enable f16;
@group(0) @binding(0) var<storage, read> weights: array<f16>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
var<workgroup> wt: array<vec4<f16>, ${K}>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
  let local = local_id.x;
  let c = local;
  let scale = input[0];
  for (var i = local; i < ${K}u; i += 256u) { wt[i] = vec4<f16>(f16(input[i])); }
  workgroupBarrier();
  var acc = vec4<f32>(0.0);
  for (var rep = 0u; rep < ${REPS}u; rep += 1u) {
    ${loopOf(body, unroll)}
  }
  result[group.x * 256u + local] = acc.x + acc.y + acc.z + acc.w;
}`;

export async function main(device, args) {
  const number = (name, fallback) => Number(args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback);
  const groups = number("groups", 8192), rounds = number("rounds", 7);
  const weights = device.createBuffer({ size: (K * C + 256) * 2, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const half = new Float16Array(K * C + 256);
  for (let i = 0; i < half.length; i += 1) half[i] = ((i * 2654435761) % 1024) / 4096 - 0.125;
  device.queue.writeBuffer(weights, 0, half);
  const input = device.createBuffer({ size: K * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const values = new Float32Array(K);
  for (let i = 0; i < K; i += 1) values[i] = 0.5 + ((i * 40503) % 256) / 1024;
  device.queue.writeBuffer(input, 0, values);
  const result = device.createBuffer({ size: groups * 256 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const read = device.createBuffer({ size: 256 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const querySet = device.createQuerySet({ type: "timestamp", count: 2 });
  const resolved = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const stamps = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const arms = [];
  for (const [name, body] of Object.entries(BODIES)) {
    for (const unroll of [1, 8]) {
      const module = device.createShaderModule({ code: shader(body, unroll) });
      const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
      // An arm that names no buffer has no binding for it under an "auto" layout.
      const used = { weights: name !== "workgroup-only" && name !== "arithmetic" };
      const entries = [{ binding: 1, resource: { buffer: input } }, { binding: 2, resource: { buffer: result } }];
      if (used.weights) entries.unshift({ binding: 0, resource: { buffer: weights } });
      const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
      // GPU time by timestamp query: Firefox answers onSubmittedWorkDone on a ~100 ms tick, so a wall clock reads
      // every arm as 100.
      const once = async () => {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass({
          timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } });
        pass.setPipeline(pipeline); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(groups);
        pass.end();
        encoder.resolveQuerySet(querySet, 0, 2, resolved, 0);
        encoder.copyBufferToBuffer(resolved, 0, stamps, 0, 16);
        device.queue.submit([encoder.finish()]);
        await stamps.mapAsync(GPUMapMode.READ);
        const [begin, end] = new BigUint64Array(stamps.getMappedRange().slice(0));
        stamps.unmap();
        return Number(end - begin) / 1e6;
      };
      await once();
      const times = [];
      for (let round = 0; round < rounds; round += 1) times.push(await once());
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(result, 0, read, 0, 256 * 4);
      device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const sample = new Float32Array(read.getMappedRange().slice(0));
      read.unmap();
      arms.push({ arm: name, unroll, ms: Number(Math.min(...times).toFixed(2)), value: sample[7] });
    }
  }
  // An unrolled arm must compute its loop's answer, or the pair is not a comparison.
  for (let i = 0; i < arms.length; i += 2) {
    const [a, b] = [arms[i].value, arms[i + 1].value];
    if (!(Math.abs(a - b) <= 1e-3 * Math.max(1, Math.abs(a)))) throw new Error(`${arms[i].arm}: unrolled ${b} against ${a}`);
  }
  return { groups, iterations: K * REPS, arms };
}
