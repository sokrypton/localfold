/**
 * Can the sampler's Gaussian draws move to the GPU, what do they cost, and
 * what does it cost to move them?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-gpu-gaussian.js --atoms=5760
 *
 * WHY. `inject` - `noisy[i] = positions[i] + injected * normal()` - is **2.1 ms
 * of a 48.7 ms sampler step at 240 tokens**, the largest single piece of host
 * work in the loop, and it is what stops the positions living on the device
 * across a schedule. src/af3/fold.js's `normalFrom` is Box-Muller over a 32-bit
 * LCG: two uniforms, a log, a sqrt and a cos per draw, in float64.
 *
 * Two GPU forms are timed here against it:
 *
 *   jumped   the SAME LCG, jumped ahead in closed form so lane i can compute
 *            the state at stream position i without walking the stream.
 *            `state_n = A(n)*s0 + C(n)`, and A and C come from the standard
 *            doubling jump in about 32 u32 multiplies. Same values, same
 *            order, same Box-Muller - so it reproduces the host stream up to
 *            PRECISION, and no further, because WGSL has no f64 and JS's log
 *            and cos are f64. That is the whole of the semantic question.
 *   counter  a hash of (seed, index), which is what a GPU RNG normally is.
 *            Cheaper, and unrelated to the host stream.
 *
 * 🔴 AND THE HONEST ANSWER TO "CAN IT" IS YES AND "SHOULD IT" IS A PRODUCT
 * QUESTION, because either way a given seed names a DIFFERENT STRUCTURE than
 * it does today: the jumped form differs in the last bits of every draw, and
 * over 200 sampler steps that is not nothing. It stays deterministic and
 * seeded; it is simply a different mapping.
 */
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

// src/af3/fold.js's generator, verbatim, as the thing to reproduce.
function hostNormals(seed, count) {
  let state = seed >>> 0;
  const uniform = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return (state + 1) / 4294967297;
  };
  const out = new Float32Array(count);
  for (let i = 0; i < count; i += 1) {
    out[i] = Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());
  }
  return out;
}

const SHADER = (mode) => `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: vec4<u32>;   // seed, count, _, _

const A: u32 = 1664525u;
const C: u32 = 1013904223u;

// 🔴 THE LCG JUMPED AHEAD IN CLOSED FORM. Walking the stream is inherently
// serial; a lane needs the state at position n without the n-1 before it.
// state_n = A(n) * s0 + C(n), and the doubling below builds A(n) and C(n) in
// about 32 u32 multiplies - the same trick a parallel LCG always uses.
fn jump(s0: u32, n: u32) -> u32 {
  var a_acc: u32 = 1u;
  var c_acc: u32 = 0u;
  var a_i: u32 = A;
  var c_i: u32 = C;
  var k: u32 = n;
  loop {
    if (k == 0u) { break; }
    if ((k & 1u) != 0u) {
      a_acc = a_acc * a_i;
      c_acc = c_acc * a_i + c_i;
    }
    c_i = c_i * (a_i + 1u);
    a_i = a_i * a_i;
    k = k >> 1u;
  }
  return a_acc * s0 + c_acc;
}

fn uniformAt(seed: u32, index: u32) -> f32 {
  // The host advances THEN reads, so stream position i is jump(seed, i + 1).
  return (f32(jump(seed, index + 1u)) + 1.0) / 4294967297.0;
}

// A counter-based hash, for the arm that does not try to match the host.
fn hash(x: u32) -> u32 {
  var h = x;
  h ^= h >> 16u; h *= 0x7feb352du;
  h ^= h >> 15u; h *= 0x846ca68bu;
  h ^= h >> 16u;
  return h;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= params.y) { return; }
  var u1: f32;
  var u2: f32;
${mode === "jumped" ? `  u1 = uniformAt(params.x, i * 2u);
  u2 = uniformAt(params.x, i * 2u + 1u);`
                    : `  u1 = (f32(hash(params.x ^ (i * 2u))) + 1.0) / 4294967297.0;
  u2 = (f32(hash(params.x ^ (i * 2u + 1u))) + 1.0) / 4294967297.0;`}
  out[i] = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979 * u2);
}`;

export async function main(device, args) {
  const count = Number(option(args, "atoms", "5760")) * 3;
  const seed = Number(option(args, "seed", "20260831"));
  const rounds = Number(option(args, "rounds", "9"));

  // What it costs on the host today.
  const median = (v) => [...v].sort((a, b) => a - b)[v.length >> 1];
  const hostTimes = [];
  let host;
  for (let i = 0; i < rounds; i += 1) {
    const start = performance.now();
    host = hostNormals(seed, count);
    hostTimes.push(performance.now() - start);
  }

  const out = device.createBuffer({ size: count * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const params = device.createBuffer({ size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(params, 0, new Uint32Array([seed, count, 0, 0]));
  const readback = device.createBuffer({ size: count * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  const results = {};
  for (const mode of ["jumped", "counter"]) {
    const pipeline = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: SHADER(mode) }), entryPoint: "main" },
    });
    const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
      entries: [out, params].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const once = async () => {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(count / 256)); pass.end();
      const start = performance.now();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      return performance.now() - start;
    };
    await once();
    const times = [];
    for (let i = 0; i < rounds; i += 1) times.push(await once());

    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(out, 0, readback, 0, count * 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const got = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();

    // How close is it to the host stream, and is it a unit normal at all?
    let worst = 0; let sum = 0; let squares = 0;
    for (let i = 0; i < count; i += 1) {
      worst = Math.max(worst, Math.abs(got[i] - host[i]));
      sum += got[i]; squares += got[i] * got[i];
    }
    const mean = sum / count;
    results[mode] = {
      ms: Number(median(times).toFixed(4)),
      worstAbsoluteDifferenceFromHost: Number(worst.toExponential(2)),
      mean: Number(mean.toFixed(4)),
      standardDeviation: Number(Math.sqrt(squares / count - mean * mean).toFixed(4)),
    };
  }
  return {
    values: count,
    hostMs: Number(median(hostTimes).toFixed(3)),
    speedup: Number((median(hostTimes) / results.jumped.ms).toFixed(1)),
    results,
  };
}
