/**
 * What do this device's subgroup matrix units ISSUE at, with the operands
 * already in registers?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-matrix-ceiling.js
 *     node tools/gpu-chrome.mjs tools/gpu/probe-matrix-ceiling.js --accumulators=8,16,32
 *
 * This is probe-alu.js's question for the matrix units: not what a GEMM
 * achieves, which is an arithmetic-intensity question about staging, but what
 * the hardware will retire when nothing is in the way. A kernel that cannot
 * approach this number is losing to its memory path; one that reaches it is
 * done.
 *
 * 🔴 THE OPERANDS ARE LOOP-INVARIANT ON PURPOSE and the accumulators are not.
 * Each multiply-accumulate depends on the previous value of ITS accumulator, so
 * the chain cannot be folded away, and independent accumulators are what give
 * the units something to overlap - one accumulator measures the DEPENDENT
 * LATENCY, not the throughput. `--accumulators` sweeps that.
 *
 * 🔴 AND IT IS TIMED IN THE PASS, NOT AROUND THE SUBMIT. The first version of
 * this probe wrapped queue.onSubmittedWorkDone() and reported 201 TFLOP/s,
 * which was a fixed 2.3 ms round trip divided into a varying flop count: every
 * one of twenty-four configurations spanning 32x different work came back at
 * 2.3-2.4 ms, and the "throughput" was the flop count alone. A timestamp pair
 * inside the pass measures the kernel. `scaling` below is the check that says
 * so - doubling the iterations must double the time, and if it does not, the
 * number is overhead again.
 */
const FEATURE = "chromium-experimental-subgroup-matrix";

const flag = (args, name, fallback) => {
  const hit = (args ?? []).find((a) => a.startsWith(`--${name}=`));
  return hit === undefined ? fallback : hit.slice(name.length + 3);
};

export async function main(_device, args = []) {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (adapter === null) return { error: "no adapter" };
  if (!adapter.features.has(FEATURE)) return { skipped: `no ${FEATURE}` };
  if (!adapter.features.has("timestamp-query")) return { skipped: "no timestamp-query" };

  const required = [FEATURE, "timestamp-query"];
  if (adapter.features.has("shader-f16")) required.push("shader-f16");
  const gpu = await adapter.requestDevice({ requiredFeatures: required });

  const only = flag(args, "config", null);
  const configs = [...(adapter.info?.subgroupMatrixConfigs ?? [])]
    .map((c) => ({
      componentType: c.componentType, resultComponentType: c.resultComponentType,
      M: c.M, N: c.N, K: c.K,
    }))
    .filter((c) => c.componentType === "f16")
    .filter((c) => only === null || `${c.M}x${c.N}x${c.K}` === only);

  const groups = Number(flag(args, "groups", 3456));
  const iterations = Number(flag(args, "iterations", 16384));
  const accumulatorArms = String(flag(args, "accumulators", "1,2,4,8,16,32")).split(",").map(Number);
  const repeats = Number(flag(args, "repeats", 3));

  const querySet = gpu.createQuerySet({ type: "timestamp", count: 2 });
  const resolved = gpu.createBuffer({
    size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const out = gpu.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const operands = gpu.createBuffer({ size: 4096, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
  new Uint16Array(operands.getMappedRange()).fill(0x3800); // 0.5 in half
  operands.unmap();

  const build = async (config, accumulators, loops) => {
    const { M, N, K, resultComponentType: result } = config;
    // Type parameters are <T, COLUMNS, ROWS>: see check-subgroup-matrix-shapes.js.
    const declare = [];
    const mac = [];
    const store = [];
    for (let i = 0; i < accumulators; i += 1) {
      declare.push(`  var acc_${i} = subgroup_matrix_result<${result}, ${N}, ${M}>();`);
      mac.push(`    acc_${i} = subgroupMatrixMultiplyAccumulate(l, r, acc_${i});`);
      store.push(`  subgroupMatrixStore(&staged, 0u, acc_${i}, false, ${N}u);`);
    }
    const code = `enable f16;
enable chromium_experimental_subgroup_matrix;
@group(0) @binding(0) var<storage, read> operands: array<f16>;
@group(0) @binding(1) var<storage, read_write> sink: array<f32>;
var<workgroup> staged: array<${result}, ${M * N}>;
@compute @workgroup_size(32)
fn main(@builtin(local_invocation_index) lane: u32) {
  let l = subgroupMatrixLoad<subgroup_matrix_left<f16, ${K}, ${M}>>(&operands, 0u, false, ${K}u);
  let r = subgroupMatrixLoad<subgroup_matrix_right<f16, ${N}, ${K}>>(&operands, 0u, false, ${N}u);
${declare.join("\n")}
  for (var i = 0u; i < ${loops}u; i = i + 1u) {
${mac.join("\n")}
  }
${store.join("\n")}
  workgroupBarrier();
  // The sink keeps the chain live without adding a dependent read to the loop.
  if (lane == 0u && f32(staged[0]) == 1234.5) { sink[0] = f32(staged[0]); }
}`;
    gpu.pushErrorScope("validation");
    let pipeline = null;
    let message = null;
    try {
      pipeline = await gpu.createComputePipelineAsync({
        layout: "auto",
        compute: { module: gpu.createShaderModule({ code }), entryPoint: "main" },
      });
    } catch (error) { message = String(error.message ?? error).split("\n")[0]; }
    const validation = await gpu.popErrorScope();
    if (pipeline === null) return { error: message ?? validation?.message?.split("\n")[0] ?? "rejected" };
    return { pipeline };
  };

  /** One dispatch, timed by a timestamp pair written by the pass itself. */
  const timeOf = async (pipeline) => {
    const bindGroup = gpu.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [operands, out].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const encoder = gpu.createCommandEncoder();
    const pass = encoder.beginComputePass({
      timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(groups);
    pass.end();
    encoder.resolveQuerySet(querySet, 0, 2, resolved, 0);
    const staging = gpu.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    encoder.copyBufferToBuffer(resolved, 0, staging, 0, 16);
    gpu.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const stamps = new BigUint64Array(staging.getMappedRange().slice(0));
    staging.unmap();
    return Number(stamps[1] - stamps[0]) / 1e6; // nanoseconds -> ms
  };

  const measure = async (config, accumulators, loops) => {
    const built = await build(config, accumulators, loops);
    if (built.error) return { error: built.error };
    await timeOf(built.pipeline);
    const times = [];
    for (let i = 0; i < repeats; i += 1) times.push(await timeOf(built.pipeline));
    times.sort((x, y) => x - y);
    const ms = times[Math.floor(times.length / 2)];
    const flops = groups * loops * accumulators * 2 * config.M * config.N * config.K;
    return { ms: Number(ms.toPrecision(4)), tflops: Number((flops / (ms / 1000) / 1e12).toPrecision(4)) };
  };

  const rows = [];
  for (const config of configs) {
    for (const accumulators of accumulatorArms) {
      const got = await measure(config, accumulators, iterations);
      rows.push({ ...config, accumulators, ...got });
    }
  }
  rows.sort((a, b) => (b.tflops ?? 0) - (a.tflops ?? 0));

  // 🔴 THE CHECK THAT THE NUMBER IS A KERNEL AND NOT AN OVERHEAD. Half the
  // iterations must take half the time; if the ratio is not near 2, whatever
  // is being timed is not the loop.
  const top = rows.find((r) => r.tflops !== undefined);
  let scaling = null;
  if (top) {
    const config = configs.find((c) => c.M === top.M && c.N === top.N && c.K === top.K
      && c.resultComponentType === top.resultComponentType);
    const half = await measure(config, top.accumulators, iterations / 2);
    const full = await measure(config, top.accumulators, iterations);
    scaling = {
      halfMs: half.ms, fullMs: full.ms,
      ratio: Number((full.ms / half.ms).toPrecision(3)),
      linear: Math.abs(full.ms / half.ms - 2) < 0.25,
    };
  }
  return { groups, iterations, best: rows[0], scaling, rows };
}
