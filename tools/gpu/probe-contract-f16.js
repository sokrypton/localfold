/**
 * The triangle contraction in f16 arithmetic at several lane blocks, against the shipped 8 x 8 f32 kernel - the
 * measurement behind webgpu/kernels/triangle/shaders.js's f16 contraction (outgoing, f32 channel-major a and b,
 * uniform inputs; unscaled, so it also shows the precision the scaled shipped kernel has where nothing overflows).
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-contract-f16.js --tokens=522 --channels=32
 *
 * On an M5 under stock flags at 522 tokens: f32 8 x 8 6.38 ms; f16 4 x 4 a lane 4.77 (relRMS 6.6e-4), 8 x 4 5.25,
 * 4 x 8 5.23, 8 x 8 6.05, 4 x 4 with a 32-deep step 4.72 (relRMS 8.8e-4).
 */
import { createTriangleShaders } from "../../webgpu/kernels/triangle/shaders.js";

const option = (args, name, fallback) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;

function f16Kernel(L, RA = 4, CB = 4, BK = 16) {
  // a (16 RA) x (16 CB) tile on 16 x 16 lanes; a, b f32 channel-major [h][i][k] (outgoing); products in f16 over BK k
  // (BK a multiple of 16: the tile is staged 16 k at a time), then added into f32
  const QA = RA / 4, QB = CB / 4;
  const fma = [];
  for (let qa = 0; qa < QA; qa += 1) for (let r = 0; r < 4; r += 1) for (let qb = 0; qb < QB; qb += 1)
    fma.push(`part[${(qa * 4 + r) * QB + qb}] = fma(vec4<f16>(left${qa}.${"xyzw"[r]}), right${qb}, part[${(qa * 4 + r) * QB + qb}]);`);
  return `enable f16;
const L: u32 = ${L}u;
const PAIRS: u32 = ${L * L}u;
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
var<workgroup> tile_a: array<vec4<f16>, ${256 * QA * (BK / 16)}>;
var<workgroup> tile_b: array<vec4<f16>, ${256 * QB * (BK / 16)}>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let i0 = group.y * ${16 * RA}u;
  let j0 = group.x * ${16 * CB}u;
  let h = group.z;
  var sum: array<vec4<f32>, ${RA * QB}>;
  for (var r = 0u; r < ${RA * QB}u; r += 1u) { sum[r] = vec4<f32>(0.0); }
  for (var k0 = 0u; k0 < L; k0 += ${BK}u) {
    for (var kk = 0u; kk < ${BK}u; kk += 16u) {
      let k = k0 + kk + local.x;
      for (var q = 0u; q < ${QA}u; q += 1u) {
        var sa = vec4<f16>(0.0);
        for (var r = 0u; r < 4u; r += 1u) {
          let i = i0 + local.y + (q * 4u + r) * 16u;
          if (i < L && k < L) { sa[r] = f16(a[h * PAIRS + i * L + k]); }
        }
        tile_a[((kk + local.x) * 16u + local.y) * ${QA}u + q] = sa;
      }
      for (var q = 0u; q < ${QB}u; q += 1u) {
        var sb = vec4<f16>(0.0);
        for (var r = 0u; r < 4u; r += 1u) {
          let j = j0 + local.y + (q * 4u + r) * 16u;
          if (j < L && k < L) { sb[r] = f16(b[h * PAIRS + j * L + k]); }
        }
        tile_b[((kk + local.x) * 16u + local.y) * ${QB}u + q] = sb;
      }
    }
    workgroupBarrier();
    var part: array<vec4<f16>, ${RA * QB}>;
    for (var r = 0u; r < ${RA * QB}u; r += 1u) { part[r] = vec4<f16>(0.0); }
    for (var k = 0u; k < ${BK}u; k += 1u) {
      ${Array.from({ length: QA }, (_, q) => `let left${q} = tile_a[(k * 16u + local.y) * ${QA}u + ${q}u];`).join("\n      ")}
      ${Array.from({ length: QB }, (_, q) => `let right${q} = tile_b[(k * 16u + local.x) * ${QB}u + ${q}u];`).join("\n      ")}
      ${fma.join("\n      ")}
    }
    for (var r = 0u; r < ${RA * QB}u; r += 1u) { sum[r] += vec4<f32>(part[r]); }
    workgroupBarrier();
  }
  for (var r = 0u; r < ${RA}u; r += 1u) {
    let i = i0 + local.y + r * 16u;
    if (i >= L) { continue; }
    for (var qb = 0u; qb < ${QB}u; qb += 1u) {
      let cells = sum[r * ${QB}u + qb];
      for (var c = 0u; c < 4u; c += 1u) {
        let j = j0 + local.x + (qb * 4u + c) * 16u;
        if (j < L) { output[h * PAIRS + i * L + j] = cells[c]; }
      }
    }
  }
}`;
}

export async function main(device, args) {
  const L = Number(option(args, "tokens", "522"));
  const H = Number(option(args, "channels", "32"));
  const pairs = L * L;
  let state = 1;
  const rnd = () => { state = (state * 1103515245 + 12345) & 0x7fffffff; return state / 0x7fffffff - 0.5; };
  const av = new Float32Array(H * pairs).map(() => rnd() * 2);
  const bv = new Float32Array(H * pairs).map(() => rnd() * 2);
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const mk = (data) => { const g = device.createBuffer({ size: data.byteLength, usage }); device.queue.writeBuffer(g, 0, data); return g; };
  const a = mk(av), b = mk(bv);
  const out = device.createBuffer({ size: H * pairs * 4, usage });
  const read = device.createBuffer({ size: H * pairs * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const offsets = {};
  const shaders = createTriangleShaders({ length: L, cZ: 128, cHidden: H }, "f32", offsets, 1e-5, "outgoing", "fast");
  const arms = [
    { name: "shipped 32x32 f32", code: shaders.contract, x: Math.ceil(L / 32), y: Math.ceil(L / 32) },
    { name: "f16 4x4 bk16", code: f16Kernel(L, 4, 4, 16), x: Math.ceil(L / 64), y: Math.ceil(L / 64) },
    { name: "f16 8x4 bk16", code: f16Kernel(L, 8, 4, 16), x: Math.ceil(L / 64), y: Math.ceil(L / 128) },
    { name: "f16 4x8 bk16", code: f16Kernel(L, 4, 8, 16), x: Math.ceil(L / 128), y: Math.ceil(L / 64) },
    { name: "f16 8x8 bk16", code: f16Kernel(L, 8, 8, 16), x: Math.ceil(L / 128), y: Math.ceil(L / 128) },
    { name: "f16 4x4 bk32", code: f16Kernel(L, 4, 4, 32), x: Math.ceil(L / 64), y: Math.ceil(L / 64) },
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
    const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(out, 0, read, 0, H * pairs * 4); device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ); results.push(new Float32Array(read.getMappedRange().slice(0))); read.unmap();
  }
  for (let r = 0; r < 7; r += 1) for (const arm of arms) arm.times.push(await run(arm, 4));
  const rel = (k) => { let e = 0, s = 0;
    for (let i = 0; i < results[0].length; i += 1) { e += (results[k][i] - results[0][i]) ** 2; s += results[0][i] ** 2; }
    return Math.sqrt(e / s); };
  const med = (v) => [...v].sort((p, q) => p - q)[v.length >> 1];
  return { L, H, arms: arms.map((arm, k) => ({ name: arm.name, ms: med(arm.times), tflops: 2 * L ** 3 * H / med(arm.times) / 1e9, relRms: rel(k) })) };
}
