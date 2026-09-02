/**
 * Why the kernels sit at 270 billion instructions a second and the probe says
 * 640: what the machine is waiting for, one variable at a time.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-latency.js
 *     node tools/gpu-chrome.mjs tools/gpu/probe-latency.js --arms=ilp,shared-ilp
 *
 * WHY IT EXISTS. tools/gpu/probe-alu.js measures a ceiling with EIGHT
 * independent accumulator chains, 256 lanes, and no memory at all. Every real
 * kernel here reaches about 40% of it, and "the gap is latency" is a guess
 * until the gap is decomposed. This walks from that microbenchmark toward a
 * real kernel one variable at a time and prices each step:
 *
 *   ilp          multiply-adds with 1, 2, 4, 8, 16 independent chains
 *   shared-ilp   the same for workgroup-memory reads
 *   width        the same eight chains at 32, 64, 128, 256 lanes a workgroup
 *   ballast      the same, with 0 to 32 KB of workgroup memory declared
 *   mixed        R workgroup reads to F multiply-adds, at the ratios the
 *                trunk's kernels actually run
 *   barrier      a workgroupBarrier every N iterations
 *
 * 🔴 EVERY ARM DOES THE SAME NUMBER OF MULTIPLY-ADDS. What changes is only how
 * they are arranged, so the numbers are directly comparable and a drop names
 * the thing that caused it.
 */
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/** Independent multiply-add chains, nothing else in the loop. */
const ilpShader = (chains, lanes = 256, ballastBytes = 0) => {
  const ballast = ballastBytes > 0 ? ballastBytes / 4 : 0;
  return `
${ballast ? `var<workgroup> ballast: array<f32, ${ballast}>;` : ""}
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(${lanes})
fn main(@builtin(global_invocation_id) id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let index = id.x;
${ballast ? `  ballast[local_id.x] = f32(index) * 1e-9;
  workgroupBarrier();` : ""}
${Array.from({ length: chains }, (_, c) =>
  `  var a${c} = f32(index) * 1e-6 + ${c}.0;`).join("\n")}
  let m = 1.0000001;
  let b = 1e-7;
  // ...ITERATIONS is scaled by the chain count so every arm does the same work.
  for (var step = 0u; step < ITERATIONS / ${chains}u; step += 1u) {
${Array.from({ length: chains }, (_, c) => `    a${c} = a${c} * m + b;`).join("\n")}
  }
  var total = 0.0;
${Array.from({ length: chains }, (_, c) => `  total += a${c};`).join("\n")}
${ballast ? "  total += ballast[0];" : ""}
  out[index] = total;
}`;
};

/** The same, for workgroup-memory reads. */
const sharedIlpShader = (chains, lanes = 256) => `
var<workgroup> tile: array<f32, 1024>;
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(${lanes})
fn main(@builtin(global_invocation_id) id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let local = local_id.x;
  for (var slot = local; slot < 1024u; slot += ${lanes}u) { tile[slot] = f32(slot) * 1e-6; }
  workgroupBarrier();
${Array.from({ length: chains }, (_, c) => `  var a${c} = 0.0;`).join("\n")}
  var at = local;
  for (var step = 0u; step < ITERATIONS / ${chains}u; step += 1u) {
${Array.from({ length: chains }, (_, c) =>
  `    a${c} += tile[(at + ${c * 37}u) & 1023u];`).join("\n")}
    at += 1u;
  }
  var total = 0.0;
${Array.from({ length: chains }, (_, c) => `  total += a${c};`).join("\n")}
  out[id.x] = total;
}`;

/**
 * R workgroup reads feeding F multiply-adds, which is what a tiled matmul's
 * inner loop is. The reads are independent of each other and every one is used.
 */
const mixedShader = (reads, fmas, lanes = 256) => `
var<workgroup> tile: array<f32, 1024>;
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(${lanes})
fn main(@builtin(global_invocation_id) id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let local = local_id.x;
  for (var slot = local; slot < 1024u; slot += ${lanes}u) { tile[slot] = f32(slot) * 1e-6; }
  workgroupBarrier();
${Array.from({ length: fmas }, (_, f) => `  var a${f} = f32(local) * 1e-6;`).join("\n")}
  var at = local;
  for (var step = 0u; step < ITERATIONS / ${fmas}u; step += 1u) {
${Array.from({ length: reads }, (_, r) =>
  `    let r${r} = tile[(at + ${r * 61}u) & 1023u];`).join("\n")}
${Array.from({ length: fmas }, (_, f) =>
  `    a${f} = a${f} * r${f % reads} + r${(f + 1) % reads};`).join("\n")}
    at += 1u;
  }
  var total = 0.0;
${Array.from({ length: fmas }, (_, f) => `  total += a${f};`).join("\n")}
  out[id.x] = total;
}`;

/** Eight chains with a barrier every `period` iterations. */
const barrierShader = (period, lanes = 256) => `
var<workgroup> tile: array<f32, 64>;
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(${lanes})
fn main(@builtin(global_invocation_id) id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let index = id.x;
${Array.from({ length: 8 }, (_, c) => `  var a${c} = f32(index) * 1e-6 + ${c}.0;`).join("\n")}
  let m = 1.0000001;
  let b = 1e-7;
  for (var outer = 0u; outer < ITERATIONS / 8u / ${period}u; outer += 1u) {
    for (var step = 0u; step < ${period}u; step += 1u) {
${Array.from({ length: 8 }, (_, c) => `      a${c} = a${c} * m + b;`).join("\n")}
    }
    workgroupBarrier();
  }
  var total = 0.0;
${Array.from({ length: 8 }, (_, c) => `  total += a${c};`).join("\n")}
  out[index] = total;
}`;

export async function main(device, args) {
  const iterations = Number(option(args, "iterations", "8192"));
  const threads = Number(option(args, "threads", "262144"));
  const rounds = Number(option(args, "rounds", "5"));
  const wanted = option(args, "arms", "ilp,shared-ilp,width,ballast,mixed,barrier").split(",");

  const out = device.createBuffer({ size: threads * 4, usage: GPUBufferUsage.STORAGE });
  const build = async (code, lanes) => {
    const source = `const ITERATIONS: u32 = ${iterations}u;\n${code}`;
    const pipeline = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: source }), entryPoint: "main" },
    });
    return {
      pipeline, lanes,
      bindGroup: device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: out } }],
      }),
    };
  };

  const arms = [];
  const add = async (group, label, code, lanes = 256, operations = threads * iterations) => {
    arms.push({ group, label, operations, kernel: await build(code, lanes), times: [] });
  };

  if (wanted.includes("ilp")) {
    for (const chains of [1, 2, 4, 8, 16]) {
      await add("ilp", `${chains} chain${chains > 1 ? "s" : ""}`, ilpShader(chains));
    }
  }
  if (wanted.includes("shared-ilp")) {
    for (const chains of [1, 2, 4, 8, 16]) {
      await add("shared-ilp", `${chains} chain${chains > 1 ? "s" : ""}`, sharedIlpShader(chains));
    }
  }
  if (wanted.includes("width")) {
    for (const lanes of [32, 64, 128, 256]) {
      await add("width", `${lanes} lanes`, ilpShader(8, lanes), lanes);
    }
  }
  if (wanted.includes("ballast")) {
    for (const kb of [0, 4, 8, 16, 32]) {
      await add("ballast", `${kb} KB`, ilpShader(8, 256, kb * 1024));
    }
  }
  if (wanted.includes("mixed")) {
    // 🔴 THE RATIOS ARE THE TRUNK'S OWN. tri.project reads six workgroup values
    // for eight vector multiply-adds; pair-transition's second matmul reads
    // three for two.
    for (const [reads, fmas] of [[1, 8], [2, 8], [4, 8], [6, 8], [8, 8], [3, 2]]) {
      await add("mixed", `${reads} read${reads > 1 ? "s" : ""} : ${fmas} fma`,
                mixedShader(reads, fmas));
    }
  }
  if (wanted.includes("barrier")) {
    for (const period of [1024, 64, 16, 4, 1]) {
      await add("barrier", `every ${period}`, barrierShader(period));
    }
  }

  const time = async (arm) => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(arm.kernel.pipeline);
    pass.setBindGroup(0, arm.kernel.bindGroup);
    pass.dispatchWorkgroups(threads / arm.kernel.lanes);
    pass.end();
    const start = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return performance.now() - start;
  };

  for (const arm of arms) await time(arm);
  for (let round = 0; round < rounds; round += 1) {
    for (const arm of arms) arm.times.push(await time(arm));
  }

  const median = (values) => [...values].sort((a, b) => a - b)[values.length >> 1];
  const groups = {};
  for (const arm of arms) {
    const ms = median(arm.times);
    (groups[arm.group] ??= []).push({
      arm: arm.label,
      ms: Number(ms.toFixed(2)),
      billionPerSecond: Number((arm.operations / ms / 1e6).toFixed(1)),
    });
  }
  return { threads, iterations, groups };
}
