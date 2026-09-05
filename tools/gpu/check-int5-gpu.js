/**
 * Does an int5 tensor decode the same on the GPU as on the host?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-int5-gpu.js
 *
 * WHY IT MATTERS. A session's first fold spends ~467 ms decoding the diffusion
 * transformer's weights out of int5 and narrowing them to f16, all on the main
 * thread - `tools/gpu/probe-pack.js` has the split. Every byte of that could be
 * a compute shader instead, reading the packed codes the network already
 * delivered rather than a float32 expansion of them. This checks the only thing
 * that decides whether it can: whether the shader's arithmetic lands on the
 * same f16 bits as `readTensorRange`'s.
 *
 * 🔴 IT IS NOT OBVIOUSLY THE SAME, WHICH IS WHY IT IS MEASURED. JavaScript
 * computes `code * scale + zero` in f64 and narrows once at the store; WGSL has
 * no f64, so it computes in f32 and narrows once at the store. The product is
 * exact in both - a 5-bit code times an f16 scale needs at most 16 mantissa
 * bits - but the SUM can need more than 24, and there f32 rounds where f64 does
 * not. Whether that ever changes the f16 result is an empirical question about
 * the numbers a real model holds.
 */
import { readTensorRange } from "../../src/reference/dtype.js";

const GROUP = 32;
const GROUP_BYTES = 20;

const SHADER = (elements) => `
const ELEMENTS: u32 = ${elements}u;
const GROUP: u32 = ${GROUP}u;
const GROUP_BYTES: u32 = ${GROUP_BYTES}u;

@group(0) @binding(0) var<storage, read> codes: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> zeros: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;

fn byte_at(index: u32) -> u32 {
  return (codes[index >> 2u] >> ((index & 3u) * 8u)) & 255u;
}

fn decode(index: u32, scale: f32, zero: f32) -> f32 {
  let within = index % GROUP;
  let group = index / GROUP;
  let bit = within * 5u;
  let at = group * GROUP_BYTES + (bit >> 3u);
  let pair = byte_at(at) | (byte_at(at + 1u) << 8u);
  return f32((pair >> (bit & 7u)) & 31u) * scale + zero;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  // 🔴 ONE INVOCATION OWNS ONE WORD, WHICH IS TWO ELEMENTS. WGSL cannot write
  // sixteen bits, so the pair that shares a word has to be produced together -
  // see src/runtime/storage.js for the rule and what breaking it costs.
  let word = id.x;
  let first = word * 2u;
  if (first >= ELEMENTS) { return; }

  let groupA = first / GROUP;
  let scaleA = unpack2x16float(scales[groupA >> 1u])[groupA & 1u];
  let zeroA = unpack2x16float(zeros[groupA >> 1u])[groupA & 1u];
  let a = decode(first, scaleA, zeroA);

  var b = 0.0;
  if (first + 1u < ELEMENTS) {
    let groupB = (first + 1u) / GROUP;
    let scaleB = unpack2x16float(scales[groupB >> 1u])[groupB & 1u];
    let zeroB = unpack2x16float(zeros[groupB >> 1u])[groupB & 1u];
    b = decode(first + 1u, scaleB, zeroB);
  }
  output[word] = pack2x16float(vec2<f32>(a, b));
}`;

export async function main(device, args) {
  void args;
  const elements = 32 * 4096;                 // 128 groups of 32, a realistic slab
  const groups = Math.ceil(elements / GROUP);

  // A synthetic tensor in the shard layout readTensorRange expects: codes,
  // then scales, then zeros. Values chosen to exercise the whole 5-bit range
  // and a wide spread of exponents, which is where f32 and f64 could part.
  let state = 12345;
  const random = () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  };
  const codeBytes = groups * GROUP_BYTES + 1;
  const codes = new Uint8Array(codeBytes);
  for (let index = 0; index < codeBytes; index += 1) codes[index] = Math.floor(random() * 256);
  const scales = new Float16Array(groups);
  const zeros = new Float16Array(groups);
  for (let group = 0; group < groups; group += 1) {
    scales[group] = (random() - 0.5) * 10 ** Math.floor(random() * 8 - 4);
    zeros[group] = (random() - 0.5) * 10 ** Math.floor(random() * 8 - 4);
  }

  const scaleAt = Math.ceil(codeBytes / 4) * 4;
  const zeroAt = scaleAt + groups * 2;
  const total = zeroAt + groups * 2;
  const buffer = new ArrayBuffer(Math.ceil(total / 4) * 4);
  new Uint8Array(buffer).set(codes, 0);
  new Uint8Array(buffer).set(new Uint8Array(scales.buffer), scaleAt);
  new Uint8Array(buffer).set(new Uint8Array(zeros.buffer), zeroAt);

  const record = { dtype: "int5", block: GROUP, shape: [elements], byteOffset: 0,
                   scaleOffset: scaleAt, zeroOffset: zeroAt };
  // The host answer, narrowed the way concatenateAs narrows it.
  const wide = readTensorRange(record, buffer, 0, 0, elements);
  const host = new Float16Array(elements);
  for (let index = 0; index < elements; index += 1) host[index] = wide[index];

  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const upload = (bytes, usage = storage) => {
    const padded = Math.ceil(bytes.byteLength / 4) * 4;
    const gpu = device.createBuffer({ size: padded, usage, mappedAtCreation: true });
    new Uint8Array(gpu.getMappedRange()).set(
      new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength));
    gpu.unmap();
    return gpu;
  };
  const codesBuffer = upload(codes);
  const scalesBuffer = upload(new Uint8Array(scales.buffer));
  const zerosBuffer = upload(new Uint8Array(zeros.buffer));
  const words = Math.ceil(elements / 2);
  const output = device.createBuffer({ size: words * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: words * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  const module = device.createShaderModule({ code: SHADER(elements) });
  const pipeline = await device.createComputePipelineAsync({
    layout: "auto", compute: { module, entryPoint: "main" } });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [codesBuffer, scalesBuffer, zerosBuffer, output].map((gpu, binding) => ({
      binding, resource: { buffer: gpu } })),
  }));
  pass.dispatchWorkgroups(Math.ceil(words / 64));
  pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, words * 4);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const gpuBits = new Uint16Array(readback.getMappedRange().slice(0));
  readback.unmap();

  const hostBits = new Uint16Array(host.buffer, host.byteOffset, elements);
  let differing = 0;
  let firstDifference;
  for (let index = 0; index < elements; index += 1) {
    if (hostBits[index] === gpuBits[index]) continue;
    differing += 1;
    firstDifference ??= { index, host: host[index],
                          gpu: new Float16Array(gpuBits.buffer, 0, elements)[index] };
  }
  const result = { elements, groups, differing,
                   fraction: Number((differing / elements).toExponential(2)), firstDifference };
  console.log(`int5 on the GPU\t${elements} elements\t${differing} differ`);
  if (differing > 0) throw new Error(`${differing} of ${elements} f16 results differ from the host decoder`);
  return result;
}
