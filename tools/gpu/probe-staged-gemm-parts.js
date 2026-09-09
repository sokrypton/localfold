/**
 * Where does the staged matrix GEMM's time actually go?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-staged-gemm-parts.js
 *
 * 🔴 THIS IS BISECTION BY DELETION, WHICH CLAUDE.md WARNS AGAINST, and it is
 * used here because there is no profiler that can see INSIDE one kernel - the
 * timestamp pair measures a pass, and this is one dispatch. So the arms below
 * are indicative only, and the rule stands: whatever they suggest has to be
 * confirmed by making the change and measuring the whole kernel again.
 *
 * The arms strip one thing at a time from the full kernel:
 *  - `full`      everything.
 *  - `noEpilogue` accumulate, then write one value per workgroup. Isolates the
 *                 GEMM from the flush, the bias read and the output traffic.
 *  - `noMac`      stage the panels, never multiply. Isolates the staging loop.
 *  - `noStage`    multiply out of uninitialised workgroup memory. Isolates the
 *                 matrix issue from the memory that feeds it.
 */
import { createStagedMatrixShader } from "./gemm-matrix-staged.js";
import { deviceMatrixConfig } from "../../src/runtime/device-profile.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  // 🔴 IT USED TO REQUEST ITS OWN DEVICE, AND EVERY ARM READ 0.000 ms. A second
  // GPUDevice off a fresh adapter is handed out LOST here: buffers, pipelines
  // and submits all succeed as no-ops, `resolveQuerySet` writes nothing, and
  // the timestamp pair subtracts to exactly zero. Nothing throws, so the probe
  // reported four identical zeroes as a result. The runner's device already
  // asks for every feature and every raised limit this needs - including
  // maxComputeWorkgroupStorageSize, which was the reason the private device
  // existed - so take the one that is passed in.
  const gpu = device;
  if (!gpu.features.has("chromium-experimental-subgroup-matrix")) {
    return { skipped: "no matrix units" };
  }
  if (!gpu.features.has("timestamp-query")) return { skipped: "no timestamp-query" };
  const config = deviceMatrixConfig(gpu, { element: "f16" });
  if (config === null) return { skipped: "no f16 matrix config" };

  const rows = Number(option(args, "rows", "30208"));
  const inner = Number(option(args, "inner", "256"));
  const columns = Number(option(args, "columns", "1024"));
  const BM = Number(option(args, "blockRows", "128"));
  const BN = Number(option(args, "blockColumns", "128"));
  const BK = Number(option(args, "blockInner", "32"));
  const SGY = Number(option(args, "subgroupRows", "1"));
  const SGX = Number(option(args, "subgroupColumns", "8"));

  // Read the right operand straight out of the weight buffer instead of
  // staging it. It needs inner % blockInner == 0, which is asserted here rather
  // than discovered as a wrong answer.
  const direct = option(args, "direct", "0") === "1";
  if (direct && inner % BK !== 0) throw new Error("--direct wants inner divisible by blockInner");

  // The accumulator's width; the device config's own unless named. It is a
  // register knob, so it moves the occupancy and not the arithmetic rate.
  const resultType = option(args, "result", "");
  // Double-buffer the staging: the next panel's global reads are issued before
  // this panel's multiplies. 🔴 THE `noMac` AND `noStage` ARMS DO NOT APPLY to
  // it - they are textual cuts against the single-loop form - so with this on,
  // read `full` and nothing else.
  const prefetch = option(args, "prefetch", "0") === "1";
  const base = createStagedMatrixShader({
    directWeights: direct, prefetch,
    ...(resultType === "" ? {} : { result: resultType }),
    blockRows: BM, blockColumns: BN, blockInner: BK, subgroupRows: SGY, subgroupColumns: SGX,
    element: "f16", result: config.resultComponentType,
    tile: { M: config.M, N: config.N, K: config.K },
  });

  // Each arm is the full shader with one region neutered. The edits are textual
  // and asserted, so a shader change that moves them fails loudly here.
  const cut = (text, needle, replacement) => {
    if (!text.includes(needle)) throw new Error(`probe is stale: no ${JSON.stringify(needle.slice(0, 40))}`);
    return text.replaceAll(needle, replacement);
  };
  const variants = {
    full: base,
    noEpilogue: cut(base,
      "  for (var i = lane; i < 256u; i += 32u) {",
      "  for (var i = lane; i < 1u; i += 32u) {"),
    noMac: base.replace(/      acc_\d+_\d+ = subgroupMatrixMultiplyAccumulate\([^;]+;\n/g, ""),
    noStage: base
      .replace(/    for \(var i = tid; i < \d+u; i \+= \d+u\) \{\n      let k = k0[^}]+\}\n/g, ""),
  };

  const source = gpu.createBuffer({
    size: rows * inner * 2, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true,
  });
  new Uint16Array(source.getMappedRange()).fill(0x3800);
  source.unmap();
  const weights = gpu.createBuffer({
    size: (inner * columns + columns) * 2, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true,
  });
  new Uint16Array(weights.getMappedRange()).fill(0x3400);
  weights.unmap();
  const parameters = gpu.createBuffer({
    size: 32, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true,
  });
  new Uint32Array(parameters.getMappedRange()).set([rows, inner, columns, 0, inner * columns, 0, 0, 0]);
  parameters.unmap();
  const output = gpu.createBuffer({ size: rows * columns * 4, usage: GPUBufferUsage.STORAGE });

  const querySet = gpu.createQuerySet({ type: "timestamp", count: 2 });
  const resolved = gpu.createBuffer({
    size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });

  const results = {};
  for (const [name, code] of Object.entries(variants)) {
    gpu.pushErrorScope("validation");
    let pipeline = null;
    try {
      pipeline = await gpu.createComputePipelineAsync({
        layout: "auto",
        compute: { module: gpu.createShaderModule({ code }), entryPoint: "main" },
      });
    } catch (e) { results[name] = { error: String(e.message ?? e).split("\n")[0] }; }
    const validation = await gpu.popErrorScope();
    if (pipeline === null) {
      if (validation && !results[name]) results[name] = { error: validation.message.split("\n")[0] };
      continue;
    }
    // 🔴 A NEUTERED ARM DROPS A BINDING, AND `layout: "auto"` DROPS IT TOO.
    // `noStage` deletes the source staging loop, so nothing reads binding 0 and
    // the reflected layout has three entries where the bind group has four -
    // which is a validation error, not a slow arm. Bind only what this arm's
    // own layout asks for.
    const layout = pipeline.getBindGroupLayout(0);
    const buffers = [source, weights, parameters, output];
    const entries = [];
    for (let binding = 0; binding < buffers.length; binding += 1) {
      gpu.pushErrorScope("validation");
      const probe = gpu.createBindGroup({
        layout, entries: [{ binding, resource: { buffer: buffers[binding] } }],
      });
      void probe;
      // A binding the layout does not have fails on THAT entry; one it does
      // have fails only because the others are missing. Either way the message
      // names the reason, so read it rather than guessing from the shader text.
      const why = await gpu.popErrorScope();
      if (why === null || !why.message.includes(`binding index ${binding} not present`)) {
        entries.push({ binding, resource: { buffer: buffers[binding] } });
      }
    }
    const bindGroup = gpu.createBindGroup({ layout, entries });
    const once = async () => {
      const encoder = gpu.createCommandEncoder();
      const pass = encoder.beginComputePass({
        timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(columns / BN), Math.ceil(rows / BM));
      pass.end();
      encoder.resolveQuerySet(querySet, 0, 2, resolved, 0);
      const staging = gpu.createBuffer({
        size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      encoder.copyBufferToBuffer(resolved, 0, staging, 0, 16);
      gpu.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const stamps = new BigUint64Array(staging.getMappedRange().slice(0));
      staging.unmap();
      return Number(stamps[1] - stamps[0]) / 1e6;
    };
    await once();
    const times = [];
    for (let i = 0; i < 7; i += 1) times.push(await once());
    times.sort((a, b) => a - b);
    results[name] = { ms: Number(times[3].toPrecision(4)) };
  }
  const flops = 2 * rows * inner * columns;
  for (const [name, r] of Object.entries(results)) {
    if (r.ms) r.tflops = Number((flops / (r.ms / 1000) / 1e12).toPrecision(4));
  }
  return {
    rows, inner, columns, block: `${BM}x${BN}x${BK}x${SGY}x${SGX}`, direct, resultType, prefetch,
    workgroups: Math.ceil(columns / BN) * Math.ceil(rows / BM),
    config, results,
  };
}
