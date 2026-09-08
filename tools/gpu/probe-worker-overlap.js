/**
 * Can the sampler's host work hide behind the GPU, in a Worker?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-worker-overlap.js
 *
 * WHY. probe-sampler-overhead.js puts the whole non-GPU cost of a denoiser
 * step at 3.2 ms of 48.7 at 240 tokens, and **2.1 ms of it is `inject`** - the
 * host generating `atoms * 3` Gaussian deviates through Box-Muller over a
 * 32-bit LCG. Two transcendentals a draw at about 43 ns each, and it is
 * SERIALISED with the GPU: the sampler generates, submits, waits, generates.
 *
 * A GPU kernel would remove it (probe-gpu-gaussian.js: 7.5x, and it matches the
 * host stream to 6.45e-6 because WGSL has no f64). A WORKER removes it too and
 * keeps the stream EXACTLY, because the worker runs the same sequential
 * generator - it just runs it while the GPU is busy. The draws are a pure
 * function of the seed and do not depend on any denoiser output, so they can be
 * produced arbitrarily far ahead.
 *
 * This measures whether the overlap is real: the same generation, once on the
 * main thread beside a GPU workload and once in a worker beside the same one.
 */
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

// src/af3/fold.js's generator, verbatim.
const GENERATOR = `
function make(seed) {
  let state = seed >>> 0;
  const uniform = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return (state + 1) / 4294967297;
  };
  return () => Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());
}
self.onmessage = (event) => {
  const { seed, count, steps } = event.data;
  const normal = make(seed);
  for (let s = 0; s < steps; s += 1) {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = normal();
    self.postMessage({ step: s, out }, [out.buffer]);
  }
};`;

function hostGenerator(seed) {
  let state = seed >>> 0;
  const uniform = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return (state + 1) / 4294967297;
  };
  return () => Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());
}

export async function main(device, args) {
  const atoms = Number(option(args, "atoms", "5760"));
  const count = atoms * 3;
  const steps = Number(option(args, "steps", "24"));
  const seed = 20260831;

  // A GPU workload that costs roughly what a denoiser step costs, so the
  // overlap is measured against a realistic amount of time to hide inside.
  const size = 1 << 22;
  const buffer = device.createBuffer({ size: size * 4, usage: GPUBufferUsage.STORAGE });
  const shader = `
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  var v = f32(id.x) * 1e-6;
  for (var i = 0u; i < 2048u; i += 1u) { v = v * 1.0000001 + 1e-7; }
  data[id.x] = v;
}`;
  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: shader }), entryPoint: "main" },
  });
  const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer } }] });
  const submitGpu = () => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(size / 256); pass.end();
    device.queue.submit([encoder.finish()]);
    return device.queue.onSubmittedWorkDone();
  };
  await submitGpu();

  // How long the GPU workload alone takes, and the generation alone.
  let start = performance.now();
  for (let s = 0; s < steps; s += 1) await submitGpu();
  const gpuOnly = performance.now() - start;

  const normal = hostGenerator(seed);
  start = performance.now();
  for (let s = 0; s < steps; s += 1) {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = normal();
  }
  const hostOnly = performance.now() - start;

  // 🔴 SERIALISED, WHICH IS WHAT THE SAMPLER DOES TODAY.
  const serial = hostGenerator(seed);
  start = performance.now();
  for (let s = 0; s < steps; s += 1) {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = serial();
    await submitGpu();
  }
  const serialised = performance.now() - start;

  // ...and the same work with the generation in a worker, running ahead.
  const blob = new Blob([GENERATOR], { type: "text/javascript" });
  const worker = new Worker(URL.createObjectURL(blob));
  const queue = [];
  let resolveNext = null;
  worker.onmessage = (event) => {
    queue.push(event.data.out);
    if (resolveNext !== null) { const r = resolveNext; resolveNext = null; r(); }
  };
  const take = async () => {
    while (queue.length === 0) await new Promise((r) => { resolveNext = r; });
    return queue.shift();
  };
  worker.postMessage({ seed, count, steps });
  start = performance.now();
  for (let s = 0; s < steps; s += 1) {
    const out = await take();
    await submitGpu();
    if (out.length !== count) throw new Error("worker returned the wrong length");
  }
  const overlapped = performance.now() - start;
  worker.terminate();

  const per = (total) => Number((total / steps).toFixed(3));
  return {
    atoms, values: count, steps,
    msPerStep: {
      gpuAlone: per(gpuOnly),
      hostGenerationAlone: per(hostOnly),
      serialised: per(serialised),
      workerOverlapped: per(overlapped),
    },
    hiddenMsPerStep: Number(((serialised - overlapped) / steps).toFixed(3)),
    fractionOfHostWorkHidden:
      Number(((serialised - overlapped) / (hostOnly || 1)).toFixed(2)),
  };
}
