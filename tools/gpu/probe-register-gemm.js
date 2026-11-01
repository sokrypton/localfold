/**
 * How far a WGSL GEMM gets on this device's vector units, at the shapes AF3's pair transition runs - the ceiling the
 * trunk's instruction-bound kernels are measured against.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-register-gemm.js [--rows=65025] [--shapes=128x1024,512x128] [--debug]
 *
 * C[rows][n] = A[rows][k] B[k][n], A and B f16 in storage, C f32. Two designs, each with its arithmetic in f32 (operands
 * widened as staged) or f16 (folded into f32 accumulators every k tile, as matrix-linear.js's "mixed" does):
 *   - workgroup-staged: 16 x 16 lanes over a 128 x 128 tile, k 16 at a time, a lane 8 x 8 - 4 workgroup reads for 16
 *     vec4 multiply-adds. A is transposed into its tile as whole vec4s, a lane's 4 x 4 block in registers.
 *   - subgroup: no workgroup memory, operands loaded per lane and exchanged by subgroupShuffle.
 * Timed by timestamp queries, the arms alternated; the output checked against a CPU reference on a sample of rows.
 *
 * 🔴 ON THE M5, STOCK CHROME (2026-10-10), 65025 rows: workgroup-staged f32 2.0 / f16 2.8-2.9 TFLOP/s; subgroup 0.7-1.7.
 * The shipped fused transition runs its 25.6 GFLOP at 2.06 effective (with its LayerNorm and SwiGLU), grid.attend at
 * 1.44 - so a WGSL GEMM's ceiling here is ~1.4x what the trunk's kernels already reach, not probe-alu's vec4 numbers
 * (11.3 f32, 14.3 f16), which no kernel with operands to stage approaches.
 *
 * 🔴 AND ITS FIRST VERSION WAS WRONG IN A WAY WORTH KNOWING: four lanes each wrote ONE COMPONENT of the same vec4 in
 * workgroup memory, and on Metal that is a read-modify-write of the whole vector - rows 1-3 of every group of four came
 * back zero (relRMS 0.585). Write whole vectors to workgroup memory.
 */

function shader({ element, k, n }) {
  const f16 = element === "f16";
  const E = f16 ? "f16" : "f32";
  const acc = [];
  for (let r = 0; r < 8; r += 1) for (let v = 0; v < 2; v += 1) acc.push(`acc${r}_${v}`);
  const blk = (r, v) => (f16 ? `blk${r}_${v}` : `acc${r}_${v}`);
  return `${f16 ? "enable f16;\n" : "enable f16;\n"}
const K: u32 = ${k}u;
const N: u32 = ${n}u;
struct P { rows: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> a: array<vec4<f16>>;    // [rows][K / 4]
@group(0) @binding(1) var<storage, read> b: array<vec4<f16>>;    // [K][N / 4]
@group(0) @binding(2) var<storage, read_write> c: array<vec4<f32>>;  // [rows][N / 4]
@group(0) @binding(3) var<uniform> p: P;
var<workgroup> As: array<vec4<${E}>, ${16 * 128 / 4}>;   // [k][m / 4]
var<workgroup> Bs: array<vec4<${E}>, ${16 * 128 / 4}>;   // [k][n / 4]
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let tx = lid.x; let ty = lid.y; let t = ty * 16u + tx;
  let m0 = wid.y * 128u; let n0 = wid.x * 128u;
${acc.map((x) => `  var ${x} = vec4<f32>(0.0);`).join("\n")}
${f16 ? acc.map((x) => `  var ${x.replace("acc", "blk")} = vec4<f16>(0.0);`).join("\n") : ""}
  for (var k0 = 0u; k0 < K; k0 += 16u) {
    // A: 32 groups of 4 rows x 4 groups of 4 k = 128 tasks, a lane's 4 x 4 block transposed in registers and written
    // as whole vec4s into As[k][m / 4] (component writes from neighbouring lanes race on Metal)
    if (t < 128u) {
      let rg = t / 4u; let kq = t % 4u;
      let r0 = m0 + rg * 4u;
      let x0 = ${E === "f16" ? "" : "vec4<f32>"}(a[min(r0, p.rows - 1u) * (K / 4u) + k0 / 4u + kq]);
      let x1 = ${E === "f16" ? "" : "vec4<f32>"}(a[min(r0 + 1u, p.rows - 1u) * (K / 4u) + k0 / 4u + kq]);
      let x2 = ${E === "f16" ? "" : "vec4<f32>"}(a[min(r0 + 2u, p.rows - 1u) * (K / 4u) + k0 / 4u + kq]);
      let x3 = ${E === "f16" ? "" : "vec4<f32>"}(a[min(r0 + 3u, p.rows - 1u) * (K / 4u) + k0 / 4u + kq]);
      let base = kq * 4u * 32u + rg;
      As[base] = vec4<${E}>(x0.x, x1.x, x2.x, x3.x);
      As[base + 32u] = vec4<${E}>(x0.y, x1.y, x2.y, x3.y);
      As[base + 64u] = vec4<${E}>(x0.z, x1.z, x2.z, x3.z);
      As[base + 96u] = vec4<${E}>(x0.w, x1.w, x2.w, x3.w);
    }
    // B: 16 k x 128 n = 512 vec4 along n, two a lane, straight copies
    for (var q = 0u; q < 2u; q += 1u) {
      let task = t + q * 256u;
      let kk = task / 32u; let nq = task % 32u;
      Bs[kk * 32u + nq] = ${E === "f16" ? "" : "vec4<f32>"}(b[(k0 + kk) * (N / 4u) + n0 / 4u + nq]);
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk += 1u) {
      let a0 = As[kk * 32u + ty * 2u]; let a1 = As[kk * 32u + ty * 2u + 1u];
      let b0 = Bs[kk * 32u + tx]; let b1 = Bs[kk * 32u + 16u + tx];
${[0, 1, 2, 3, 4, 5, 6, 7].map((r) => {
    const s = `${r < 4 ? "a0" : "a1"}[${r % 4}u]`;
    return `      ${blk(r, 0)} += ${s} * b0; ${blk(r, 1)} += ${s} * b1;`;
  }).join("\n")}
    }
${f16 ? acc.map((x) => `    ${x} += vec4<f32>(${x.replace("acc", "blk")}); ${x.replace("acc", "blk")} = vec4<f16>(0.0);`).join("\n") : ""}
    workgroupBarrier();
  }
${[0, 1, 2, 3, 4, 5, 6, 7].map((r) => `  { let row = m0 + ty * 8u + ${r}u;
    if (row < p.rows) { c[row * (N / 4u) + n0 / 4u + tx] = acc${r}_0; c[row * (N / 4u) + n0 / 4u + 16u + tx] = acc${r}_1; } }`).join("\n")}
}`;
}


// The subgroup arm: no workgroup memory. A subgroup (32 lanes, 4 x 8) computes 32 rows x 64 columns, a lane 8 x 8; four
// subgroups a workgroup (64 x 128). Per k step of 8 each lane loads one k of its 8 rows and two k of its 8 columns, and
// the step's operands move between lanes by subgroupShuffle: 4 shuffles (vec4) for 16 vec4 multiply-adds.
function subgroupShader({ element, k, n }) {
  const f16 = element === "f16";
  const E = f16 ? "f16" : "f32";
  const acc = [];
  for (let r = 0; r < 8; r += 1) for (let v = 0; v < 2; v += 1) acc.push(`acc${r}_${v}`);
  const blk = (r, v) => (f16 ? `blk${r}_${v}` : `acc${r}_${v}`);
  const inner = [];
  for (let kk = 0; kk < 8; kk += 1) {
    inner.push(`    {
      let srcA = ry * 8u + ${kk}u; let srcB = ${Math.floor(kk / 2)}u * 8u + cx;
      let aLo = subgroupShuffle(ra0, srcA); let aHi = subgroupShuffle(ra1, srcA);
      let b0 = subgroupShuffle(rb${(kk % 2) * 2}, srcB); let b1 = subgroupShuffle(rb${(kk % 2) * 2 + 1}, srcB);
${[0, 1, 2, 3, 4, 5, 6, 7].map((r) => `      ${blk(r, 0)} += ${r < 4 ? "aLo" : "aHi"}[${r % 4}u] * b0; ${blk(r, 1)} += ${r < 4 ? "aLo" : "aHi"}[${r % 4}u] * b1;`).join("\n")}
    }`);
  }
  const cv = (x) => (f16 ? x : `vec4<f32>(${x})`);
  return `enable f16;
enable subgroups;
const K: u32 = ${k}u;
const N: u32 = ${n}u;
struct P { rows: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> a: array<f16>;            // [rows][K]
@group(0) @binding(1) var<storage, read> b: array<vec4<f16>>;      // [K][N / 4]
@group(0) @binding(2) var<storage, read_write> c: array<vec4<f32>>;  // [rows][N / 4]
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(128, 1, 1)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(subgroup_invocation_id) lane: u32) {
  let sgi = li / 32u;
  let m0 = wid.y * 64u + (sgi / 2u) * 32u; let n0 = wid.x * 128u + (sgi % 2u) * 64u;
  let ry = lane / 8u; let cx = lane % 8u;
${acc.map((x) => `  var ${x} = vec4<f32>(0.0);`).join("\n")}
${f16 ? acc.map((x) => `  var ${x.replace("acc", "blk")} = vec4<f16>(0.0);`).join("\n") : ""}
  var rowOf: array<u32, 8>;
  for (var r = 0u; r < 8u; r += 1u) { rowOf[r] = min(m0 + ry * 8u + r, p.rows - 1u) * K; }
  for (var k0 = 0u; k0 < K; k0 += 8u) {
    let kA = k0 + cx;
    let ra0 = ${cv("vec4<f16>(a[rowOf[0] + kA], a[rowOf[1] + kA], a[rowOf[2] + kA], a[rowOf[3] + kA])")};
    let ra1 = ${cv("vec4<f16>(a[rowOf[4] + kA], a[rowOf[5] + kA], a[rowOf[6] + kA], a[rowOf[7] + kA])")};
    let kB = k0 + ry * 2u;
    let rb0 = ${cv("b[kB * (N / 4u) + n0 / 4u + cx * 2u]")};
    let rb1 = ${cv("b[kB * (N / 4u) + n0 / 4u + cx * 2u + 1u]")};
    let rb2 = ${cv("b[(kB + 1u) * (N / 4u) + n0 / 4u + cx * 2u]")};
    let rb3 = ${cv("b[(kB + 1u) * (N / 4u) + n0 / 4u + cx * 2u + 1u]")};
${inner.join("\n")}
${f16 ? acc.map((x) => `    ${x} += vec4<f32>(${x.replace("acc", "blk")}); ${x.replace("acc", "blk")} = vec4<f16>(0.0);`).join("\n") : ""}
  }
${[0, 1, 2, 3, 4, 5, 6, 7].map((r) => `  { let row = m0 + ry * 8u + ${r}u;
    if (row < p.rows) { c[row * (N / 4u) + n0 / 4u + cx * 2u] = acc${r}_0; c[row * (N / 4u) + n0 / 4u + cx * 2u + 1u] = acc${r}_1; } }`).join("\n")}
}`;
}

function toHalfBits(x) {
  const f = new Float32Array([x]); const u = new Uint32Array(f.buffer)[0];
  const s = (u >>> 16) & 0x8000; let e = ((u >>> 23) & 0xff) - 127 + 15; let m = u & 0x7fffff;
  if (e <= 0) return s; if (e >= 31) return s | 0x7c00;
  return s | (e << 10) | (m >>> 13);
}
function fromHalfBits(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
  return e === 0 ? s * m * 2 ** -24 : s * (1 + m / 1024) * 2 ** (e - 15);
}

export async function main(device, args) {
  const opt = (name, d) => (args.find((x) => x.startsWith(`--${name}=`)) ?? `=${d}`).split("=")[1];
  const rows = Number(opt("rows", "65025"));
  const shapes = opt("shapes", "128x1024,512x128").split(",").map((s) => s.split("x").map(Number));
  const rounds = Number(opt("rounds", "3"));
  const result = { features: [...device.features].sort(), rows, arms: {} };
  if (!device.features.has("timestamp-query")) throw new Error("timestamp-query wanted");
  for (const [k, n] of shapes) {
    const aH = new Uint16Array(rows * k), bH = new Uint16Array(k * n);
    for (let i = 0; i < aH.length; i += 1) aH[i] = toHalfBits(((i * 2654435761) % 1000) / 1000 - 0.5);
    for (let i = 0; i < bH.length; i += 1) bH[i] = toHalfBits((((i * 40503) % 1000) / 1000 - 0.5) * 0.1);
    const buf = (data, usage) => {
      const g = device.createBuffer({ size: Math.ceil(data.byteLength / 16) * 16, usage: usage | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(g, 0, data); return g;
    };
    const A = buf(aH, GPUBufferUsage.STORAGE), B = buf(bH, GPUBufferUsage.STORAGE);
    const C = device.createBuffer({ size: rows * n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const U = buf(new Uint32Array([rows, 0, 0, 0]), GPUBufferUsage.UNIFORM);
    const pipelines = {};
    const ARMS = ["f32", "f16", "sg-f32", "sg-f16"];
    for (const element of ARMS) {
      const sgArm = element.startsWith("sg-");
      const code = sgArm ? subgroupShader({ element: element.slice(3), k, n }) : shader({ element, k, n });
      const module = device.createShaderModule({ code });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((m) => m.type === "error");
      if (errors.length) throw new Error(`${element}: ${errors.map((e) => `${e.lineNum}: ${e.message}`).join("; ")}`);
      pipelines[element] = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
      pipelines[element].sg = sgArm;
    }
    const qs = device.createQuerySet({ type: "timestamp", count: 2 });
    const qb = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const rb = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const time = async (pipeline) => {
      const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: A } }, { binding: 1, resource: { buffer: B } },
        { binding: 2, resource: { buffer: C } }, { binding: 3, resource: { buffer: U } }] });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } });
      pass.setPipeline(pipeline); pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(n / 128, Math.ceil(rows / (pipeline.sg ? 64 : 128)));
      pass.end();
      enc.resolveQuerySet(qs, 0, 2, qb, 0); enc.copyBufferToBuffer(qb, 0, rb, 0, 16);
      device.queue.submit([enc.finish()]);
      await rb.mapAsync(GPUMapMode.READ);
      const t = new BigUint64Array(rb.getMappedRange().slice(0)); rb.unmap();
      return Number(t[1] - t[0]) / 1e6;
    };
    const check = async () => {
      const r = device.createBuffer({ size: rows * n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(C, 0, r, 0, rows * n * 4);
      device.queue.submit([enc.finish()]); await r.mapAsync(GPUMapMode.READ);
      const got = new Float32Array(r.getMappedRange().slice(0)); r.unmap(); r.destroy();
      let num = 0, den = 0;
      for (const row of [0, 1, 127, 128, Math.floor(rows / 2), rows - 1]) {
        for (let j = 0; j < n; j += 7) {
          let s = 0; for (let q = 0; q < k; q += 1) s += fromHalfBits(aH[row * k + q]) * fromHalfBits(bH[q * n + j]);
          num += (got[row * n + j] - s) ** 2; den += s * s;
        }
      }
      if (args.includes("--debug")) {
        const show = [];
        for (const [row, j] of [[0, 0], [0, 1], [1, 0], [0, 4], [0, 64], [4, 0], [8, 0]]) {
          let s = 0; for (let q = 0; q < k; q += 1) s += fromHalfBits(aH[row * k + q]) * fromHalfBits(bH[q * n + j]);
          show.push([row, j, got[row * n + j], s]);
        }
        result.debug = show;
      }
      return Math.sqrt(num / den);
    };
    const flops = 2 * rows * k * n;
    const key = `${rows}x${k}x${n}`;
    result.arms[key] = {};
    for (const element of ARMS) { await time(pipelines[element]); result.arms[key][element] = { ms: [], relRms: 0 }; }
    for (let i = 0; i < rounds; i += 1) {
      for (const element of ARMS) {
        const ms = await time(pipelines[element]);
        result.arms[key][element].ms.push(Number(ms.toFixed(3)));
        if (i === rounds - 1) result.arms[key][element].relRms = Number((await check()).toExponential(2));
      }
    }
    for (const element of ARMS) {
      const best = Math.min(...result.arms[key][element].ms);
      result.arms[key][element].tflops = Number((flops / best / 1e9).toFixed(2));
    }
    for (const g of [A, B, C, U, qb, rb]) g.destroy();
  }
  return result;
}
