/**
 * Does a vecN multiply-add actually cost what probe-alu.js prices it at?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-alu-lanes.js --iterations=131072
 *
 * WHY IT EXISTS. probe-alu.js builds N independent chains of `a = a * m + b`
 * in a vecN, and then stores `total.x`. Every other component of every chain
 * is dead, and a shader compiler is entitled to delete it. If it does, the
 * vec2 and vec4 arms compute exactly the scalar arm's work, measure exactly
 * the scalar arm's time, and get reported at 2x and 4x the GFLOP/s because
 * the accounting multiplies by `lanes`. The "vector ceiling" would then be an
 * arithmetic identity rather than a property of the device.
 *
 * This runs both readings in ONE process, interleaved: `.x` is what
 * probe-alu.js stores, and `all` sums every component so nothing is dead.
 * If the two agree, the compiler was keeping the lanes and the vector ceiling
 * is real. If `all` is N times slower, it was not.
 */
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const CHAINS = 8;

function fmaShader(width, live, scalar = "f32") {
  const type = width === 1 ? scalar : `vec${width}<${scalar}>`;
  const zero = width === 1 ? `${scalar}(0.0)` : `${type}(${scalar}(0.0))`;
  const chain = (c) => `  var a${c} = ${type}(${scalar}(f32(index) * 1e-6 + ${c}.0));`;
  const step = (c) => `    a${c} = a${c} * m + b;`;
  // The only difference between the arms: which components reach the store.
  const consume = width === 1 ? "total"
    : live ? Array.from({ length: width }, (_, i) => `total[${i}]`).join(" + ")
           : "total.x";
  // 🔴 THE CONSTANTS MUST SURVIVE THE ELEMENT TYPE. probe-alu.js uses
  // 1.0000001 and 1e-7, and in f16 those round to exactly 1.0 and 0.0 - so
  // `a = a * m + b` is `a = a`, the whole loop is hoisted, and the f16 arms
  // report 239 TFLOP/s against this card's 78 non-tensor peak. 1 + 2^-10 and
  // 2^-10 are exactly representable in f16 and are not identities.
  const MULTIPLIER = scalar === "f16" ? "1.0009766" : "1.0000001";
  const ADDEND = scalar === "f16" ? "0.0009766" : "1e-7";
  const enable = scalar === "f16" ? "enable f16;\n" : "";
  return `${enable}
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
${Array.from({ length: CHAINS }, (_, c) => chain(c)).join("\n")}
  let m = ${type}(${scalar}(${MULTIPLIER}));
  let b = ${type}(${scalar}(${ADDEND}));
  for (var step = 0u; step < ITERATIONS; step += 1u) {
${Array.from({ length: CHAINS }, (_, c) => step(c)).join("\n")}
  }
  var total = ${zero};
${Array.from({ length: CHAINS }, (_, c) => `  total += a${c};`).join("\n")}
  out[index] = f32(${consume});
}`;
}

export async function main(device, args) {
  const iterations = Number(option(args, "iterations", "131072"));
  const threads = Number(option(args, "threads", "262144"));
  const rounds = Number(option(args, "rounds", "5"));
  const groups = threads / 256;

  const out = device.createBuffer({ size: threads * 4, usage: GPUBufferUsage.STORAGE });

  const build = async (code) => {
    // A directive must precede every declaration, so lift `enable f16;`.
    const directives = [...code.matchAll(/^\s*enable [^;]+;/gm)].map((m) => m[0].trim());
    const body = code.replace(/^\s*enable [^;]+;/gm, "");
    const source = `${directives.join("\n")}\nconst ITERATIONS: u32 = ${iterations}u;\n${body}`;
    const pipeline = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: source }), entryPoint: "main" },
    });
    return { pipeline, bindGroup: device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: out } }] }) };
  };

  const arms = {
    "scalar": { kernel: await build(fmaShader(1, true)), lanes: 1 },
    "vec2 .x only (probe-alu's arm)": { kernel: await build(fmaShader(2, false)), lanes: 2 },
    "vec2 all components": { kernel: await build(fmaShader(2, true)), lanes: 2 },
    "vec4 .x only (probe-alu's arm)": { kernel: await build(fmaShader(4, false)), lanes: 4 },
    "vec4 all components": { kernel: await build(fmaShader(4, true)), lanes: 4 },
    ...(device.features.has("shader-f16") ? {
      "f16 scalar": { kernel: await build(fmaShader(1, true, "f16")), lanes: 1 },
      "f16 vec2 .x only": { kernel: await build(fmaShader(2, false, "f16")), lanes: 2 },
      "f16 vec2 all": { kernel: await build(fmaShader(2, true, "f16")), lanes: 2 },
      "f16 vec4 .x only": { kernel: await build(fmaShader(4, false, "f16")), lanes: 4 },
      "f16 vec4 all": { kernel: await build(fmaShader(4, true, "f16")), lanes: 4 },
    } : {}),
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

  const median = (v) => [...v].sort((a, b) => a - b)[v.length >> 1];
  const operations = threads * iterations * CHAINS;
  const results = {};
  for (const [name, arm] of Object.entries(arms)) {
    const ms = median(arm.times);
    results[name] = {
      ms: Number(ms.toFixed(2)),
      gflopsAsPriced: Number((operations * arm.lanes * 2 / ms / 1e6).toFixed(0)),
    };
  }
  return { threads, iterations, chains: CHAINS, results };
}
