/**
 * Is bfloat16 available in WGSL, and would it beat the f16 storage already here?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-bf16.js
 *
 * WHY THE QUESTION. AlphaFold's own inference runs in bfloat16, and this
 * repository's f16 notes keep running into RANGE: "the running max, the running
 * sum, the logit and the accumulators stay f32: the softmax is where the range
 * is, and f16 tops out at 65504". bf16 has f32's exponent, so it does not top
 * out. It pays for that with mantissa - eight bits against f16's eleven.
 *
 * WHAT IS ACTUALLY AVAILABLE. There is no bf16 type in WGSL and no `enable`
 * for one; Dawn has no BF16 anywhere, and its subgroup-matrix component types
 * are F32, F16, U32, I32, U8 and I8. So bf16 cannot be COMPUTED in, on any
 * device, and there is no tensor-core path through it either. What it can be
 * is a STORAGE format, in core WGSL with no device feature, because a bf16 is
 * the top sixteen bits of an f32 - which is the same niche
 * `src/runtime/storage.js` already fills with `pack2x16float`.
 *
 * So the only question worth measuring is which of the two storage formats is
 * better, and where. This round-trips both over the value ranges an activation
 * actually takes and reports the relative error of each.
 *
 * 🔴 AND ROUNDING IS NOT OPTIONAL. Truncating an f32 to its top half is one
 * shift and biases every value toward zero; round-to-nearest-even costs an add
 * and a mask. Both are arms here, because the cheap one is the one somebody
 * reaches for first and it is measurably worse.
 */
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

// 🔴 PACK AND UNPACK MUST BE IN SEPARATE DISPATCHES, THROUGH MEMORY. Written
// as one expression - `unpack2x16float(pack2x16float(v))` - this device's
// compiler folds the pair into the identity and every band reports relRMS 0,
// including values past 65504 that cannot survive an f16. That is a real
// measurement hazard and not a real result: the shipping code packs into a
// buffer that a LATER pass reads, which cannot fold, so the rounding is real
// there and was only absent here.
const PACK_SHADER = `
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> packed: array<u32>;

fn bf16_truncate(v: f32) -> u32 { return bitcast<u32>(v) >> 16u; }
fn bf16_round(v: f32) -> u32 {
  let bits = bitcast<u32>(v);
  if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u) {
    return (bits >> 16u) | 0x0040u;   // keep a NaN a NaN
  }
  let lsb = (bits >> 16u) & 1u;
  return (bits + 0x7fffu + lsb) >> 16u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&source)) { return; }
  let v = source[i];
  switch (MODE) {
    case 0u: { packed[i] = pack2x16float(vec2<f32>(v, 0.0)); }
    case 1u: { packed[i] = bf16_truncate(v); }
    default: { packed[i] = bf16_round(v); }
  }
}`;

const UNPACK_SHADER = `
@group(0) @binding(0) var<storage, read> packed: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&out)) { return; }
  let w = packed[i];
  switch (MODE) {
    case 0u: { out[i] = unpack2x16float(w).x; }
    default: { out[i] = bitcast<f32>(w << 16u); }
  }
}`;

export async function main(device, args) {
  const count = Number(option(args, "count", "65536"));

  // 🔴 FIRST, WHETHER THE TYPE EXISTS AT ALL. A compile that succeeds would
  // make everything below pointless, so it is asked rather than assumed.
  let nativeBf16 = null;
  for (const source of ["enable bf16;\nfn f() -> bf16 { return bf16(1.0); }",
                        "enable chromium_experimental_bf16;\nfn f() -> bf16 { return bf16(1.0); }"]) {
    device.pushErrorScope("validation");
    device.createShaderModule({ code: source });
    const error = await device.popErrorScope();
    if (error === null) { nativeBf16 = source.split("\n")[0]; break; }
  }

  // Values spanning what a fold holds, plus the two ends f16 cannot reach.
  const bands = {
    "activations, |v| ~ 1": (r) => (r * 2 - 1) * 3,
    "logits, |v| ~ 30": (r) => (r * 2 - 1) * 60,
    "weights, |v| ~ 0.05": (r) => (r * 2 - 1) * 0.1,
    "beyond f16's 65504": (r) => (r * 2 - 1) * 1e7,
    "below f16's normal minimum": (r) => (r * 2 - 1) * 1e-6,
  };

  const results = {};
  for (const [band, map] of Object.entries(bands)) {
    const values = new Float32Array(count);
    let state = 12345;
    for (let i = 0; i < count; i += 1) {
      state = (state * 1664525 + 1013904223) >>> 0;
      values[i] = map(state / 4294967296);
    }
    const source = device.createBuffer({ size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const out = device.createBuffer({ size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: count * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(source, 0, values);

    const packedBuffer = device.createBuffer({ size: count * 4, usage: GPUBufferUsage.STORAGE });
    const row = {};
    for (const [name, mode] of [["f16 (pack2x16float)", 0], ["bf16 truncate", 1], ["bf16 round-to-even", 2]]) {
      const build = async (code) => {
        const pipeline = await device.createComputePipelineAsync({
          layout: "auto",
          compute: { module: device.createShaderModule({ code: `const MODE: u32 = ${mode}u;\n${code}` }), entryPoint: "main" },
        });
        return pipeline;
      };
      const packPipeline = await build(PACK_SHADER);
      const unpackPipeline = await build(UNPACK_SHADER);
      const encoder = device.createCommandEncoder();
      const passA = encoder.beginComputePass();
      passA.setPipeline(packPipeline);
      passA.setBindGroup(0, device.createBindGroup({ layout: packPipeline.getBindGroupLayout(0), entries:
        [source, packedBuffer].map((buffer, binding) => ({ binding, resource: { buffer } })) }));
      passA.dispatchWorkgroups(Math.ceil(count / 64)); passA.end();
      const passB = encoder.beginComputePass();
      passB.setPipeline(unpackPipeline);
      passB.setBindGroup(0, device.createBindGroup({ layout: unpackPipeline.getBindGroupLayout(0), entries:
        [packedBuffer, out].map((buffer, binding) => ({ binding, resource: { buffer } })) }));
      passB.dispatchWorkgroups(Math.ceil(count / 64)); passB.end();
      encoder.copyBufferToBuffer(out, 0, read, 0, count * 4);
      device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const got = new Float32Array(read.getMappedRange().slice(0));
      read.unmap();

      let error = 0; let scale = 0; let worst = 0; let broken = 0;
      for (let i = 0; i < count; i += 1) {
        if (!Number.isFinite(got[i])) { broken += 1; continue; }
        const d = got[i] - values[i];
        error += d * d; scale += values[i] * values[i];
        const rel = Math.abs(d) / Math.max(Math.abs(values[i]), 1e-30);
        if (rel > worst) worst = rel;
      }
      row[name] = {
        relRms: Number(Math.sqrt(error / Math.max(scale, 1e-30)).toExponential(2)),
        worstRelative: Number(worst.toExponential(2)),
        nonFinite: broken,
      };
    }
    results[band] = row;
    for (const b of [source, out, read, packedBuffer]) b.destroy();
  }

  return {
    nativeBf16: nativeBf16 ?? "no bf16 type in WGSL - neither `enable bf16` nor a chromium extension compiles",
    subgroupMatrixBf16: device.features.has("chromium-experimental-subgroup-matrix")
      ? "check probe-subgroup-matrix.js for component types" : "no subgroup-matrix feature on this device",
    shaderF16: device.features.has("shader-f16"),
    count,
    results,
  };
}
