/**
 * What this device will actually do, in WGSL, with no memory in the way.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-alu.js
 *
 * WHY IT EXISTS. Every kernel in the trunk lands between 900 GFLOP/s and
 * 1.1 TFLOP/s, and the paper peak for an M2 is 3.58 TFLOP/s - which would say
 * there is 3x left everywhere and that every tiling decision so far has been
 * leaving most of the machine idle. That is a claim about the hardware, and the
 * hardware can be asked: this runs multiply-adds out of registers with no loads
 * at all, and vec4 ones beside them, so the number the kernels are measured
 * against is one that WGSL can actually reach here rather than one from a
 * specification sheet.
 *
 * It also prints the workgroup-memory read rate, because that is what the tiled
 * kernels spend their non-arithmetic instructions on.
 */
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const CHAINS = 8;

function fmaShader(width) {
  const type = width === 1 ? "f32" : `vec${width}<f32>`;
  const zero = width === 1 ? "0.0" : `${type}(0.0)`;
  const chain = (c) => `  var a${c} = ${type}(f32(index) * 1e-6 + ${c}.0);`;
  const step = (c) => `    a${c} = a${c} * m + b;`;
  return `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
${Array.from({ length: CHAINS }, (_, c) => chain(c)).join("\n")}
  let m = ${type}(1.0000001);
  let b = ${type}(1e-7);
  for (var step = 0u; step < ITERATIONS; step += 1u) {
${Array.from({ length: CHAINS }, (_, c) => step(c)).join("\n")}
  }
  var total = ${zero};
${Array.from({ length: CHAINS }, (_, c) => `  total += a${c};`).join("\n")}
  out[index] = ${width === 1 ? "total" : "total.x"};
}`;
}

const SHARED_SHADER = `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

var<workgroup> tile: array<f32, 1024>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let local = local_id.x;
  for (var slot = local; slot < 1024u; slot += 256u) { tile[slot] = f32(slot) * 1e-6; }
  workgroupBarrier();
  var total = 0.0;
  var at = local;
  for (var step = 0u; step < ITERATIONS; step += 1u) {
${Array.from({ length: CHAINS }, (_, c) => `    total += tile[(at + ${c * 37}u) & 1023u];`).join("\n")}
    at += 1u;
  }
  out[id.x] = total;
}`;

export async function main(device, args) {
  const iterations = Number(option(args, "iterations", "4096"));
  const threads = Number(option(args, "threads", "262144"));
  const rounds = Number(option(args, "rounds", "7"));
  const groups = threads / 256;

  const out = device.createBuffer({
    size: threads * 4, usage: GPUBufferUsage.STORAGE });

  const build = async (code) => {
    const source = `const ITERATIONS: u32 = ${iterations}u;\n${code}`;
    const pipeline = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: source }), entryPoint: "main" },
    });
    return {
      pipeline,
      bindGroup: device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: out } }],
      }),
    };
  };

  const arms = {
    "fma f32": { kernel: await build(fmaShader(1)), lanes: 1 },
    "fma vec2": { kernel: await build(fmaShader(2)), lanes: 2 },
    "fma vec4": { kernel: await build(fmaShader(4)), lanes: 4 },
    "workgroup reads": { kernel: await build(SHARED_SHADER), lanes: 0 },
  };

  const time = async (arm) => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(arm.kernel.pipeline);
    pass.setBindGroup(0, arm.kernel.bindGroup);
    pass.dispatchWorkgroups(groups);
    pass.end();
    const start = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return performance.now() - start;
  };

  for (const arm of Object.values(arms)) await time(arm);
  for (let round = 0; round < rounds; round += 1) {
    for (const arm of Object.values(arms)) (arm.times ??= []).push(await time(arm));
  }

  const median = (values) => [...values].sort((a, b) => a - b)[values.length >> 1];
  const operations = threads * iterations * CHAINS;
  const results = {};
  for (const [name, arm] of Object.entries(arms)) {
    const ms = median(arm.times);
    results[name] = arm.lanes === 0
      ? { ms: Number(ms.toFixed(2)),
          billionReadsPerSecond: Number((operations / ms / 1e6).toFixed(1)) }
      : { ms: Number(ms.toFixed(2)),
          gflops: Number((operations * arm.lanes * 2 / ms / 1e6).toFixed(0)) };
  }
  return { threads, iterations, chains: CHAINS, results };
}
