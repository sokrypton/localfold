/**
 * Is the diffusion transformer's projection occupancy-bound, and does
 * splitting K fix it?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-split-k.js
 *
 * 🔴 THE DENOISER'S PARALLELISM IS `tokens * width` AND NOTHING ELSE. Its
 * projections give one thread per output channel per token, each looping the
 * whole inner extent. At 68 tokens and a width of 768 that is 52,224 threads
 * against an A100's ~221,000 slots - 24% - and the shape cannot supply more.
 * bench-head.js --tile shows the signature: HALVING the workgroups by doubling
 * the token tile makes qkvg 1.47x SLOWER (180.8 -> 265.4 us), which is what
 * being short of parallelism looks like rather than short of bandwidth.
 *
 * Splitting the inner extent is the classic answer and it is the one lever that
 * adds parallelism to a SINGLE sample - batching adds it too, and only for
 * people who want more than one sample. Each of `splitK` groups sums a slice of
 * K into its own partial, and a second pass adds them.
 *
 * This measures the idea on the shape rather than in the kernel, because
 * building it into a fused four-output projection is a day's work and this is
 * twenty minutes.
 */
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const tokens = Number(option(args, "tokens", "68"));
  const inner = Number(option(args, "inner", "768"));
  const width = Number(option(args, "width", "3072"));
  // 🔴 HOW MANY OUTPUTS ONE LANE CARRIES, which is the thing that makes qkvg
  // different from a plain projection: it computes q, k, v AND the gate off one
  // read of x, so a lane holds FOUR accumulators per tile slot. The file that
  // owns that kernel records a spill at tile 8 with two outputs a lane, so the
  // token tile and the output count trade against each other and a probe with
  // one output cannot see it.
  const outputs = Number(option(args, "outputs", "1"));
  const lanes = Number(option(args, "lanes", "256"));
  const repeats = Number(option(args, "repeats", "9"));
  if (!device.features.has("timestamp-query")) return { skipped: "no timestamp-query" };

  const source = device.createBuffer({
    size: tokens * inner * 4, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
  new Float32Array(source.getMappedRange()).fill(0.01);
  source.unmap();
  const weights = device.createBuffer({
    size: inner * width * 4 * outputs, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
  new Float32Array(weights.getMappedRange()).fill(0.01);
  weights.unmap();
  const out = device.createBuffer({
    size: tokens * width * 4 * outputs,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  // 🔴 EVERY ARM IS CHECKED, because a kernel that computes nothing is the
  // fastest kernel there is. Source and weights are 0.01, so every output must
  // be inner * 1e-4 exactly, and a rewrite that dropped a term - these arms are
  // built by string-substitution on each other - shows up as a wrong value
  // rather than as a suspiciously good time.
  const expected = inner * 0.01 * 0.01;
  const verify = async (label) => {
    const staging = device.createBuffer({
      size: tokens * width * 4 * outputs,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(out, 0, staging, 0, staging.size);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const got = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    let worst = 0;
    for (let i = 0; i < got.length; i += 1) worst = Math.max(worst, Math.abs(got[i] - expected));
    return { label, worst: Number(worst.toPrecision(3)), ok: worst < expected * 1e-3 };
  };
  const clear = () => {
    const zero = new Float32Array(tokens * width * outputs);
    device.queue.writeBuffer(out, 0, zero);
  };

  const querySet = device.createQuerySet({ type: "timestamp", count: 2 });
  const resolved = device.createBuffer({
    size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });

  // The shipped shape: one workgroup per token per width-split, one thread per
  // output channel, each walking all of K.
  const plain = `
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(${lanes})
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let token = g.x;
  let column = g.y * ${lanes}u + l.x;
  var acc: array<f32, ${outputs}>;
  for (var o = 0u; o < ${outputs}u; o = o + 1u) { acc[o] = 0.0; }
  for (var k = 0u; k < ${inner}u; k = k + 1u) {
    let x = source[token * ${inner}u + k];
${Array.from({ length: outputs }, (_, o) =>
    `    acc[${o}u] = acc[${o}u] + x * weights[${o}u * ${inner * width}u + k * ${width}u + column];`).join("\n")}
  }
${Array.from({ length: outputs }, (_, o) =>
    `  out[${o}u * ${tokens * width}u + token * ${width}u + column] = acc[${o}u];`).join("\n")}
}`;

  // Split K: `parts` groups each sum a slice, into their own partial. The
  // reduction is a second dispatch, and its cost is counted in the total.
  const split = (parts) => `
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> partials: array<f32>;
@compute @workgroup_size(${lanes})
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let token = g.x;
  let column = g.y * ${lanes}u + l.x;
  let part = g.z;
  let span = ${Math.ceil(inner / parts)}u;
  let start = part * span;
  let stop = min(start + span, ${inner}u);
  var acc: array<f32, ${outputs}>;
  for (var o = 0u; o < ${outputs}u; o = o + 1u) { acc[o] = 0.0; }
  for (var k = start; k < stop; k = k + 1u) {
    let x = source[token * ${inner}u + k];
${Array.from({ length: outputs }, (_, o) =>
    `    acc[${o}u] = acc[${o}u] + x * weights[${o}u * ${inner * width}u + k * ${width}u + column];`).join("\n")}
  }
${Array.from({ length: outputs }, (_, o) =>
    `  partials[(part * ${outputs}u + ${o}u) * ${tokens * width}u + token * ${width}u + column] = acc[${o}u];`).join("\n")}
}`;
  const reduce = (parts) => `
@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(${lanes})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= ${tokens * width}u) { return; }
${Array.from({ length: outputs }, (_, o) => `  {
    var acc = 0.0;
    for (var p = 0u; p < ${parts}u; p = p + 1u) {
      acc = acc + partials[(p * ${outputs}u + ${o}u) * ${tokens * width}u + i];
    }
    out[${o}u * ${tokens * width}u + i] = acc;
  }`).join("\n")}
}`;

  const build = async (code) => device.createComputePipelineAsync({
    layout: "auto",
    compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
  });
  const timeOf = async (steps) => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass({
      timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } });
    for (const s of steps) {
      pass.setPipeline(s.pipeline);
      pass.setBindGroup(0, s.bindGroup);
      pass.dispatchWorkgroups(s.x, s.y, s.z ?? 1);
    }
    pass.end();
    encoder.resolveQuerySet(querySet, 0, 2, resolved, 0);
    const staging = device.createBuffer({
      size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    encoder.copyBufferToBuffer(resolved, 0, staging, 0, 16);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const stamps = new BigUint64Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return Number(stamps[1] - stamps[0]) / 1e6;
  };
  const median = async (steps) => {
    await timeOf(steps);
    const t = [];
    for (let i = 0; i < repeats; i += 1) t.push(await timeOf(steps));
    t.sort((a, b) => a - b);
    return t[Math.floor(t.length / 2)];
  };

  // 🔴 THE TOKEN TILE IS THE ARITHMETIC-INTENSITY ARM, and it is the one that
  // matters. The plain kernel does ONE fused multiply-add per weight it loads -
  // 2 flops for 4 bytes, 0.5 flop/byte - which caps it near 2.5 TFLOP/s off an
  // L2 that gives about 5 TB/s, and it measures 1.3. A workgroup holding T
  // tokens loads each weight ONCE and applies it T times, so the intensity is
  // T/2 flop/byte. What it costs is workgroups: ceil(tokens/T) instead of
  // tokens.
  const tiled = (T) => `
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
var<workgroup> staged: array<f32, ${T * inner}>;
@compute @workgroup_size(${lanes})
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let base = g.x * ${T}u;
  let column = g.y * ${lanes}u + l.x;
  // Stage this workgroup's token rows once.
  for (var i = l.x; i < ${T * inner}u; i = i + ${lanes}u) {
    let t = base + i / ${inner}u;
    staged[i] = select(0.0, source[t * ${inner}u + i % ${inner}u], t < ${tokens}u);
  }
  workgroupBarrier();
  var acc: array<f32, ${T * outputs}>;
  for (var i = 0u; i < ${T * outputs}u; i = i + 1u) { acc[i] = 0.0; }
  for (var k = 0u; k < ${inner}u; k = k + 1u) {
${Array.from({ length: outputs }, (_, o) =>
    `    let w${o} = weights[${o}u * ${inner * width}u + k * ${width}u + column];`).join("\n")}
    for (var t = 0u; t < ${T}u; t = t + 1u) {
      let x = staged[t * ${inner}u + k];
${Array.from({ length: outputs }, (_, o) =>
    `      acc[t * ${outputs}u + ${o}u] = acc[t * ${outputs}u + ${o}u] + x * w${o};`).join("\n")}
    }
  }
  for (var t = 0u; t < ${T}u; t = t + 1u) {
    let token = base + t;
    if (token < ${tokens}u) {
${Array.from({ length: outputs }, (_, o) =>
    `      out[${o}u * ${tokens * width}u + token * ${width}u + column] = acc[t * ${outputs}u + ${o}u];`).join("\n")}
    }
  }
}`;

  const results = [];
  const flops = 2 * tokens * inner * width * outputs;
  {
    const pipeline = await build(plain);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [source, weights, out].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    clear();
    const ms = await median([{ pipeline, bindGroup, x: tokens, y: width / lanes }]);
    const check = await verify("plain");
    results.push({ ...check, splitK: 1, workgroups: tokens * (width / lanes), ms: Number(ms.toPrecision(4)),
      tflops: Number((flops / (ms / 1000) / 1e12).toPrecision(3)) });
  }
  for (const parts of String(option(args, "parts", "2,4,8,16")).split(",").map(Number)) {
    const partials = device.createBuffer({
      size: tokens * width * parts * outputs * 4, usage: GPUBufferUsage.STORAGE });
    const p1 = await build(split(parts));
    const p2 = await build(reduce(parts));
    const b1 = device.createBindGroup({ layout: p1.getBindGroupLayout(0),
      entries: [source, weights, partials].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const b2 = device.createBindGroup({ layout: p2.getBindGroupLayout(0),
      entries: [partials, out].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    clear();
    const ms = await median([
      { pipeline: p1, bindGroup: b1, x: tokens, y: width / lanes, z: parts },
      { pipeline: p2, bindGroup: b2, x: Math.ceil((tokens * width) / lanes), y: 1 },
    ]);
    const check = await verify(`splitK ${parts}`);
    results.push({ ...check, splitK: parts, workgroups: tokens * (width / lanes) * parts,
      ms: Number(ms.toPrecision(4)), tflops: Number((flops / (ms / 1000) / 1e12).toPrecision(3)) });
    partials.destroy();
  }
  for (const T of String(option(args, "tiles", "2,4,8")).split(",").map(Number)) {
    const pipeline = await build(tiled(T));
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [source, weights, out].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const x = Math.ceil(tokens / T);
    clear();
    const ms = await median([{ pipeline, bindGroup, x, y: width / lanes }]);
    const check = await verify(`tile ${T}`);
    results.push({ ...check, tokenTile: T, workgroups: x * (width / lanes), ms: Number(ms.toPrecision(4)),
      tflops: Number((flops / (ms / 1000) / 1e12).toPrecision(3)) });
  }

  // 🔴 THE TWO LEVERS FAIL FOR OPPOSITE REASONS, so the interesting arm is both
  // at once. The token tile raises arithmetic intensity from 0.5 to T/2 flop a
  // byte and pays for it in workgroups (68 -> 68/T); splitting K multiplies the
  // workgroups back and pays in partial traffic. Together, tile 4 with a K
  // split of 4 has tile 4's intensity and the plain kernel's occupancy.
  for (const spec of String(option(args, "combined", "")).split(",").filter(Boolean)) {
    const [T, parts] = spec.split("/").map(Number);
    let code = tiled(T);
    // Rewrite the tiled kernel into a K-split one: the loop takes a slice and
    // the store goes to this part's partial.
    code = code
      .replace("var<storage, read_write> out: array<f32>;", "var<storage, read_write> partials: array<f32>;")
      .replace("let column = ", "let part = g.z;\n  let column = ")
      .replace(`for (var k = 0u; k < ${inner}u; k = k + 1u) {`,
        `let span = ${Math.ceil(inner / parts)}u;
  let kstart = part * span;
  let kstop = min(kstart + span, ${inner}u);
  for (var k = kstart; k < kstop; k = k + 1u) {`);
    for (let o = 0; o < outputs; o += 1) {
      code = code.replace(`out[${o}u * ${tokens * width}u + token * ${width}u + column]`,
        `partials[(part * ${outputs}u + ${o}u) * ${tokens * width}u + token * ${width}u + column]`);
    }
    const partials = device.createBuffer({
      size: tokens * width * parts * outputs * 4, usage: GPUBufferUsage.STORAGE });
    const p1 = await build(code);
    const p2 = await build(reduce(parts));
    const b1 = device.createBindGroup({ layout: p1.getBindGroupLayout(0),
      entries: [source, weights, partials].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const b2 = device.createBindGroup({ layout: p2.getBindGroupLayout(0),
      entries: [partials, out].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const x = Math.ceil(tokens / T);
    clear();
    const ms = await median([
      { pipeline: p1, bindGroup: b1, x, y: width / lanes, z: parts },
      { pipeline: p2, bindGroup: b2, x: Math.ceil((tokens * width) / lanes), y: 1 },
    ]);
    const check = await verify(`tile ${T} + K/${parts}`);
    results.push({ ...check, tokenTile: T, splitK: parts, workgroups: x * (width / lanes) * parts,
      ms: Number(ms.toPrecision(4)), tflops: Number((flops / (ms / 1000) / 1e12).toPrecision(3)) });
    partials.destroy();
  }

  const base = results[0].ms;
  for (const r of results) r.speedup = Number((base / r.ms).toPrecision(3));
  return { tokens, inner, width, lanes, outputs, results };
}
