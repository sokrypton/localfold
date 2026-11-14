/**
 * The f16 triangle contraction on 8 x 8 lanes against the shipped 16 x 16 - whether an M2 wants the smaller workgroup.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-contract-lanes.js --tokens=256 --channels=128
 *
 * webgpu/kernels/triangle/shaders.js's f16 contraction is a 64 x 64 tile on 256 lanes, measured on an M5. Every other
 * 256-lane f16 tile loses on an M2 (docs/M2-TO-M5.md, 3D-2), so this races the same arithmetic - operands scaled by
 * 2^-4 into half, a k step's products summed in f16 and added into f32 - on LN x LN lanes with a QA x QB block of
 * vec4 accumulators a lane (outgoing, f32 channel-major a and b), against the shipped f32 and f16 kernels.
 */
import { createTriangleShaders } from "../../webgpu/kernels/triangle/shaders.js";

const option = (args, name, fallback) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;

export function laneKernel(L, LN, QA, QB) {
  const S = LN + 1;
  const stage = (tile, src, origin, Q) => Array.from({ length: Q }, (_, q) => `    {
      var staged = vec4<f16>(0.0);
      if (k < L) {
        let r0 = ${origin} + local.y + ${q * 4 * LN}u;
        if (r0 + ${3 * LN}u < L) {
          let at = h * PAIRS + r0 * L + k;
          staged = vec4<f16>(f16(${src}[at] * 0.0625), f16(${src}[at + ${LN}u * L] * 0.0625),
                             f16(${src}[at + ${2 * LN}u * L] * 0.0625), f16(${src}[at + ${3 * LN}u * L] * 0.0625));
        } else {
          if (r0 < L) { staged.x = f16(${src}[h * PAIRS + r0 * L + k] * 0.0625); }
          if (r0 + ${LN}u < L) { staged.y = f16(${src}[h * PAIRS + (r0 + ${LN}u) * L + k] * 0.0625); }
          if (r0 + ${2 * LN}u < L) { staged.z = f16(${src}[h * PAIRS + (r0 + ${2 * LN}u) * L + k] * 0.0625); }
          if (r0 + ${3 * LN}u < L) { staged.w = f16(${src}[h * PAIRS + (r0 + ${3 * LN}u) * L + k] * 0.0625); }
        }
      }
      ${tile}[(local.x * ${S}u + local.y) * ${Q}u + ${q}u] = staged;
    }`).join("\n");
  const names = [];
  for (let qa = 0; qa < QA; qa += 1) for (let r = 0; r < 4; r += 1) for (let qb = 0; qb < QB; qb += 1) names.push([qa, r, qb]);
  const inner = Array.from({ length: LN }, (_, k) => `    {
${Array.from({ length: QA }, (_, q) => `      let left${q} = tile_a[(${k * S}u + local.y) * ${QA}u + ${q}u];`).join("\n")}
${Array.from({ length: QB }, (_, q) => `      let right${q} = tile_b[(${k * S}u + local.x) * ${QB}u + ${q}u];`).join("\n")}
${names.map(([qa, r, qb]) => `      part_${qa}_${r}_${qb} = fma(vec4<f16>(left${qa}.${"xyzw"[r]}), right${qb}, part_${qa}_${r}_${qb});`).join("\n")}
    }`).join("\n");
  const stores = names.map(([qa, r, qb]) => `  {
    let i = i0 + local.y + ${(qa * 4 + r) * LN}u;
    let j = j0 + local.x + ${qb * 4 * LN}u;
    if (i < L) {
      let at = h * PAIRS + i * L + j;
      if (j < L) { output[at] = sum_${qa}_${r}_${qb}.x * 256.0; }
      if (j + ${LN}u < L) { output[at + ${LN}u] = sum_${qa}_${r}_${qb}.y * 256.0; }
      if (j + ${2 * LN}u < L) { output[at + ${2 * LN}u] = sum_${qa}_${r}_${qb}.z * 256.0; }
      if (j + ${3 * LN}u < L) { output[at + ${3 * LN}u] = sum_${qa}_${r}_${qb}.w * 256.0; }
    }
  }`).join("\n");
  return `enable f16;
const L: u32 = ${L}u;
const PAIRS: u32 = ${L * L}u;
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
var<workgroup> tile_a: array<vec4<f16>, ${LN * S * QA}>;
var<workgroup> tile_b: array<vec4<f16>, ${LN * S * QB}>;
@compute @workgroup_size(${LN}, ${LN}, 1)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let i0 = group.y * ${LN * 4 * QA}u;
  let j0 = group.x * ${LN * 4 * QB}u;
  let h = group.z;
${names.map(([qa, r, qb]) => `  var sum_${qa}_${r}_${qb} = vec4<f32>(0.0);`).join("\n")}
  for (var k0 = 0u; k0 < L; k0 += ${LN}u) {
    let k = k0 + local.x;
${stage("tile_a", "a", "i0", QA)}
${stage("tile_b", "b", "j0", QB)}
    workgroupBarrier();
${names.map(([qa, r, qb]) => `    var part_${qa}_${r}_${qb} = vec4<f16>(0.0);`).join("\n")}
${inner}
${names.map(([qa, r, qb]) => `    sum_${qa}_${r}_${qb} += vec4<f32>(part_${qa}_${r}_${qb});`).join("\n")}
    workgroupBarrier();
  }
${stores}
}`;
}

export async function main(device, args) {
  const L = Number(option(args, "tokens", "256"));
  const H = Number(option(args, "channels", "128"));
  const pairs = L * L;
  let state = 1;
  const rnd = () => { state = (state * 1103515245 + 12345) & 0x7fffffff; return state / 0x7fffffff - 0.5; };
  const av = new Float32Array(H * pairs).map(() => rnd() * 2);
  const bv = new Float32Array(H * pairs).map(() => rnd() * 2);
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const mk = (data) => { const g = device.createBuffer({ size: data.byteLength, usage }); device.queue.writeBuffer(g, 0, data); return g; };
  const a = mk(av), b = mk(bv);
  const out = device.createBuffer({ size: H * pairs * 4, usage });
  const SAMPLE = Math.min(H * pairs, 1 << 20);
  const read = device.createBuffer({ size: SAMPLE * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const shape = { length: L, cZ: 128, cHidden: H };
  const f32 = createTriangleShaders(shape, "f32", {}, 1e-5, "outgoing", "fast");
  const f16 = createTriangleShaders(shape, "f32", {}, 1e-5, "outgoing", "fast", undefined, true,
    { rows: 32, columns: 32, arithmetic: "f16" });
  const grid = (tile) => ({ x: Math.ceil(L / tile.columns), y: Math.ceil(L / tile.rows) });
  const lanes = (LN, QA, QB) => ({ name: `f16 ${LN}x${LN} lanes, ${4 * QA}x${4 * QB} a lane`, code: laneKernel(L, LN, QA, QB),
    x: Math.ceil(L / (LN * 4 * QB)), y: Math.ceil(L / (LN * 4 * QA)) });
  const arms = [
    { name: "shipped f32", code: f32.contract, ...grid(f32.contractTile) },
    { name: "shipped f16 (16x16 lanes)", code: f16.contract, ...grid(f16.contractTile) },
    lanes(16, 1, 1), lanes(8, 1, 1), lanes(8, 1, 2), lanes(8, 2, 1), lanes(8, 2, 2), lanes(4, 1, 1), lanes(4, 2, 2),
  ];
  for (const arm of arms) {
    arm.pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module: device.createShaderModule({ code: arm.code }), entryPoint: "main" } });
    arm.bind = device.createBindGroup({ layout: arm.pipeline.getBindGroupLayout(0), entries: [a, b, out].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    arm.times = [];
  }
  const run = async (arm, n) => {
    const enc = device.createCommandEncoder();
    for (let i = 0; i < n; i += 1) {
      const p = enc.beginComputePass(); p.setPipeline(arm.pipeline); p.setBindGroup(0, arm.bind); p.dispatchWorkgroups(arm.x, arm.y, H); p.end();
    }
    const t = performance.now(); device.queue.submit([enc.finish()]); await device.queue.onSubmittedWorkDone(); return (performance.now() - t) / n;
  };
  const results = [];
  for (const arm of arms) {
    await run(arm, 1);
    const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(out, (H * pairs - SAMPLE) * 4, read, 0, SAMPLE * 4); device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ); results.push(new Float32Array(read.getMappedRange().slice(0))); read.unmap();
  }
  for (let r = 0; r < 7; r += 1) for (const arm of arms) arm.times.push(await run(arm, 4));
  const rel = (k) => { let e = 0, s = 0;
    for (let i = 0; i < SAMPLE; i += 1) { e += (results[k][i] - results[0][i]) ** 2; s += results[0][i] ** 2; }
    return Math.sqrt(e / s); };
  const med = (v) => [...v].sort((p, q) => p - q)[v.length >> 1];
  return { L, H, arms: arms.map((arm, k) => ({ name: arm.name, ms: Number(med(arm.times).toFixed(3)),
    tflops: Number((2 * L ** 3 * H / med(arm.times) / 1e9).toFixed(2)), relRms: Number(rel(k).toExponential(2)) })) };
}
