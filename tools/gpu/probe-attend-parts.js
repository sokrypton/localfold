/**
 * grid.attend's tiled f16 kernel (the M5's shipped form) with parts of its logits removed, timed against itself.
 *     node tools/gpu-chrome.mjs tools/gpu/probe-attend-parts.js --tokens=512
 * Arms: shipped; nomask (the per-key mask read and its branch gone); notail (the key-past-n branch gone); both.
 */
import { createGridAttentionShaders, packGridAttentionWeights } from "../../webgpu/af3/trunk/grid-attention.js";

const option = (args, name, fallback) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
function noise(count, seed, scale) {
  const out = new Float32Array(count); let state = seed >>> 0;
  for (let i = 0; i < count; i += 1) { state = (state * 1664525 + 1013904223) >>> 0; out[i] = (state / 4294967296 - 0.5) * scale; }
  return out;
}

export async function main(device, args) {
  const N = Number(option(args, "tokens", "512")), C = 128, H = 4, Dh = 32, W = H * Dh, pairs = N * N;
  const zeros = (n) => new Float32Array(n);
  const { offsets } = packGridAttentionWeights({ heads: H, dimension: Dh, actNormScale: zeros(C), actNormOffset: zeros(C),
    pairBiasProjection: zeros(C * H), qProjection: zeros(C * W), kProjection: zeros(C * W), vProjection: zeros(C * W),
    gatingQuery: zeros(C * W), outputProjection: zeros(W * C) });
  const shaderFor = (extra) => createGridAttentionShaders({ n: N, channels: C, heads: H, dimension: Dh, transpose: false, residual: true,
    stagedPrecision: "f16", attendTiled: true, attendTiledHalf: true, ...extra,
    maxComputeWorkgroupStorageSize: device.limits.maxComputeWorkgroupStorageSize }, offsets, 1e-5, "fast", { swapTransposedBias: false });
  const { attend, tiles } = shaderFor({});
  const keep = (src) => src.replace("  let local = local_id.x;", "  let local = local_id.x;\n  if (N == 0u) { _ = mask[0]; }");
  const variants = {
    shipped: attend,
    nomask: keep(attend.replace(/let masked(\d+) = mask\[[^\]]*\];/g, "let masked$1 = 1.0;")),
    notail: attend.replace(/\s*if \(key\d+ >= N\) \{ logit\d+_\d+ = -3\.0e38; \}/g, ""),
  };
  variants.both = keep(variants.notail.replace(/let masked(\d+) = mask\[[^\]]*\];/g, "let masked$1 = 1.0;"));
  variants.sgmax = shaderFor({ attendSubgroupMax: true }).attend;
  variants.nokv = attend.replace("  let local = local_id.x;", "  let local = local_id.x;\n  if (N == 0u) { _ = k[0]; _ = v[0]; }").replace(/\n\s*k_tile\[slot \* STRIDE \+ index % HD4\] = [^;]*;\n\s*v_tile\[index\] = [^;]*;/, "");
  variants.nobias = attend.replace("  let local = local_id.x;", "  let local = local_id.x;\n  if (N == 0u) { _ = bias[0]; }").replace(/ \+ bias\[bias_row\d+ \+ jc\d+\]/g, "");
  variants.nop = attend.replace(/\n\s*kp_tile\[\([^;]*\] = vec4<f16>\([^;]*;/g, "");
  // (the parts priced by removing them: wrong answers on purpose)
  variants.noexp = attend.replace(/\bexp\(/g, "fexp(").replace("@compute", "fn fexp(x: f32) -> f32 { return x * 0.001 + 1.0; }\n@compute");
  variants.nopv = attend.replace(/\n\s*o\d+ \+= p\d+\[\d\] \* vv;/g, "");
  variants.noqk = attend.replace(/\n\s*s\d+_\d+ \+= f32\(dot\(qv\d+, kv\d+\)\);/g, "");
  const up = (v) => { const b = device.createBuffer({ size: v.byteLength, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    new Float32Array(b.getMappedRange()).set(v); b.unmap(); return b; };
  const inputs = [up(noise(pairs * W, 1, 2)), up(noise(pairs * W, 2, 2)), up(noise(pairs * W, 3, 1)), up(noise(H * pairs, 4, 2)),
                  up(new Float32Array(pairs).fill(1))];
  const arms = [];
  for (const [name, code] of Object.entries(variants)) {
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
    const out = device.createBuffer({ size: pairs * W * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    arms.push({ name, pipeline, out, times: [], changed: code !== attend,
      bind: device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [...inputs, out].map((buffer, binding) => ({ binding, resource: { buffer } })) }) });
  }
  const groups = [Math.ceil(N / tiles.attendRows), N, H];
  const run = async (arm, it) => {
    const enc = device.createCommandEncoder(); const p = enc.beginComputePass(); p.setPipeline(arm.pipeline); p.setBindGroup(0, arm.bind);
    for (let i = 0; i < it; i += 1) p.dispatchWorkgroups(...groups);
    p.end(); const t = performance.now(); device.queue.submit([enc.finish()]); await device.queue.onSubmittedWorkDone(); return (performance.now() - t) / it;
  };
  for (const arm of arms) await run(arm, 1);
  const read = async (arm) => { const rb = device.createBuffer({ size: 65536 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const e = device.createCommandEncoder(); e.copyBufferToBuffer(arm.out, 0, rb, 0, 65536 * 4); device.queue.submit([e.finish()]);
    await rb.mapAsync(GPUMapMode.READ); const v = new Float32Array(rb.getMappedRange().slice(0)); rb.unmap(); return v; };
  const ref = await read(arms[0]);
  for (const arm of arms) { const v = await read(arm); let e = 0, t = 0; for (let i = 0; i < v.length; i += 1) { e += (v[i] - ref[i]) ** 2; t += ref[i] ** 2; } arm.rel = Math.sqrt(e / t); }
  for (const arm of arms) await run(arm, 2);
  for (let r = 0; r < 7; r += 1) for (const arm of (r % 2 ? [...arms].reverse() : arms)) arm.times.push(await run(arm, 4));
  const med = (v) => [...v].sort((a, b) => a - b)[v.length >> 1];
  return { N, arms: arms.map((a) => ({ name: a.name, changed: a.changed, ms: Number(med(a.times).toFixed(3)), rel: Number(a.rel.toExponential(2)) })) };
}
