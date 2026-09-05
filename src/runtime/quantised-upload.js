/**
 * Fill a resident weight buffer by decoding int5 on the GPU.
 *
 * 🔴 A SESSION'S FIRST FOLD USED TO SPEND ~467 ms HERE, ON THE MAIN THREAD.
 * The diffusion transformer's 24 blocks are 200 million elements, 99.9% of them
 * int5, and the path to the device went: decode each block's ~40 tensors out of
 * the shard into float32, narrow all of it into a Float16Array, upload 378 MiB.
 * tools/gpu/probe-pack.js splits that into 264 ms of decode and 203 of narrow
 * and copy. None of it needs to happen on the host: the codes are already in
 * memory, the arithmetic is `code * scale + zero`, and the GPU is idle.
 *
 * So this uploads the CODES - about an eighth of the bytes - and decodes them
 * into the destination with one dispatch per tensor.
 *
 * 🔴 IT IS BIT-IDENTICAL TO THE HOST DECODER, AND THAT WAS MEASURED BEFORE ANY
 * OF THIS WAS WRITTEN. JavaScript computes `code * scale + zero` in f64 and
 * WGSL has no f64; the product is exact in both, but the sum can need more than
 * f32's 24 mantissa bits. `tools/gpu/check-int5-gpu.js` puts 131,072 elements
 * with scales and zeros spanning 10^-4 to 10^4 through both and finds zero
 * differences, and `tools/gpu/check-block-upload.js` does the same for whole
 * real blocks.
 *
 * 🔴 ONE INVOCATION OWNS ONE WORD, WHICH IS TWO ELEMENTS, because WGSL cannot
 * write sixteen bits - see src/runtime/storage.js. So a tensor's destination
 * offset must be EVEN, or its first element would share a word with the
 * previous tensor's last and two dispatches would race for it. `planBlockUpload`
 * refuses a plan that does not satisfy that rather than producing a buffer that
 * is wrong in a way nothing can see.
 */
/**
 * 🔴 NOT THROUGH ComputePipelineCache, BECAUSE THIS ONE NEEDS AN EXPLICIT
 * LAYOUT. `layout: "auto"` cannot know that the uniform is addressed with a
 * dynamic offset - it infers `hasDynamicOffset: false` and the encoder then
 * refuses the offset with "the number of dynamic offsets (1) does not match the
 * number of dynamic buffers (0)". The dynamic offset is what lets forty tensors
 * share one bind group and one buffer, so the layout is written out.
 */
const PIPELINES = new WeakMap();

/** The uniform stride a Params array needs; WebGPU wants 256 for a dynamic offset. */
const PARAM_STRIDE = 256;

const GROUP = 32;
const GROUP_BYTES = 20;
const LANES = 64;

/** Bytes of code for a run of elements, widened to whole quantisation groups. */
function codeSpan(first, count) {
  const firstGroup = Math.floor(first / GROUP);
  const lastGroup = Math.ceil((first + count) / GROUP);
  return { firstGroup, lastGroup,
           byteStart: firstGroup * GROUP_BYTES,
           // ...one byte of slack, because a code may straddle into it; the
           // packer leaves it for exactly this reason.
           byteLength: (lastGroup - firstGroup) * GROUP_BYTES + 1 };
}

const SHADER = `
struct Params {
  codeBase: u32,      // in bytes, into the code buffer
  scaleBase: u32,     // in f16 elements, into the scale table
  firstElement: u32,  // the tensor's first element, absolute
  count: u32,
  destWord: u32,      // first destination word
  groupBase: u32,     // the group scaleBase corresponds to
  pad0: u32,
  pad1: u32,
};

@group(0) @binding(0) var<storage, read> codes: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> zeros: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

fn byte_at(index: u32) -> u32 {
  return (codes[index >> 2u] >> ((index & 3u) * 8u)) & 255u;
}

fn scale_at(index: u32) -> f32 { return unpack2x16float(scales[index >> 1u])[index & 1u]; }
fn zero_at(index: u32) -> f32 { return unpack2x16float(zeros[index >> 1u])[index & 1u]; }

/** One element of this tensor, counted from its first. */
fn value_at(within: u32) -> f32 {
  let absolute = params.firstElement + within;
  let group = absolute / GROUP;
  let table = params.scaleBase + (group - params.groupBase);
  let scale = scale_at(table);
  let zero = zero_at(table);
  let bit = (absolute % GROUP) * 5u;
  let at = params.codeBase + (group - params.groupBase) * GROUP_BYTES + (bit >> 3u);
  let pair = byte_at(at) | (byte_at(at + 1u) << 8u);
  return f32((pair >> (bit & 7u)) & 31u) * scale + zero;
}

const GROUP: u32 = ${GROUP}u;
const GROUP_BYTES: u32 = ${GROUP_BYTES}u;

@compute @workgroup_size(${LANES})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let pair = id.x + id.y * 65535u * ${LANES}u;
  let within = pair * 2u;
  if (within >= params.count) { return; }
  let a = value_at(within);
  var b = 0.0;
  if (within + 1u < params.count) { b = value_at(within + 1u); }
  output[params.destWord + pair] = pack2x16float(vec2<f32>(a, b));
}`;

/**
 * Describe how to fill a packed f16 buffer from a set of int5 tensor ranges.
 *
 * @param {{thunk: object, offset: number, length: number}[]} entries in
 *   destination order; `thunk` is one of `stacked`'s, carrying its store and
 *   range.
 * 🔴 A BLOCK IS NOT ALL int5, AND THE REST IS NOT WORTH A KERNEL. In AF3's
 * transformer 255 tensors are int5 and 151 are float32, but by ELEMENT that is
 * 200.1 million against 0.1 - the float32 ones are biases and layer-norm
 * scales. They come back as `host` entries for the caller to narrow and write
 * the ordinary way, which costs nothing at 0.05% of the bytes and saves a
 * second code path.
 *
 * @returns {{gpu: object, host: object[]} | undefined} undefined when the plan
 *   cannot be run at all - a store with no `tensorSource`, an int5 tensor with
 *   an unexpected group size, or an odd destination offset.
 */
export function planBlockUpload(entries) {
  const codeChunks = [];
  const scaleChunks = [];
  const zeroChunks = [];
  const params = [];
  let codeBytes = 0;
  let halfCount = 0;
  const host = [];
  for (const entry of entries) {
    const { thunk, offset, length } = entry;
    const store = thunk?.store;
    if (typeof store?.tensorSource !== "function") return undefined;
    // 🔴 EVERY OFFSET EVEN, WHICH ALSO MAKES EVERY LENGTH EVEN. The offsets are
    // a running sum, so one odd length would make every later offset odd and
    // this would refuse - which is the point: an odd boundary is two writers
    // sharing a word.
    if (offset % 2 !== 0) return undefined;
    if (thunk.count !== length) return undefined;
    let source;
    try { source = store.tensorSource(thunk.tensorName); } catch { return undefined; }
    const { record, buffer, byteOffset } = source;
    if (record.dtype !== "int5") { host.push(entry); continue; }
    if (record.block !== GROUP || !Number.isInteger(record.zeroOffset)) return undefined;

    const span = codeSpan(thunk.first, thunk.count);
    const base = record.byteOffset ?? 0;
    codeChunks.push({ buffer, byteOffset: byteOffset + span.byteStart,
                      byteLength: span.byteLength, at: codeBytes });
    const groups = span.lastGroup - span.firstGroup;
    scaleChunks.push({ buffer,
                       byteOffset: byteOffset + (record.scaleOffset - base) + span.firstGroup * 2,
                       byteLength: groups * 2, at: halfCount * 2 });
    zeroChunks.push({ buffer,
                      byteOffset: byteOffset + (record.zeroOffset - base) + span.firstGroup * 2,
                      byteLength: groups * 2, at: halfCount * 2 });
    params.push({ codeBase: codeBytes, scaleBase: halfCount, firstElement: thunk.first,
                  count: length, destWord: offset / 2, groupBase: span.firstGroup });
    codeBytes += Math.ceil(span.byteLength / 4) * 4;
    halfCount += Math.ceil(groups / 2) * 2;
  }
  return { gpu: { codeChunks, scaleChunks, zeroChunks, params,
                  codeBytes, halfBytes: halfCount * 2 }, host };
}

/**
 * Run a plan, filling `destination` in place.
 *
 * The caller owns `destination`; it must be at least the packed size and carry
 * STORAGE usage.
 */
async function uploadPipeline(device) {
  const found = PIPELINES.get(device);
  if (found !== undefined) return found;
  const storage = (binding, type) => ({
    binding, visibility: GPUShaderStage.COMPUTE, buffer: { type },
  });
  const layout = device.createBindGroupLayout({
    label: "int5-upload",
    entries: [
      storage(0, "read-only-storage"), storage(1, "read-only-storage"),
      storage(2, "read-only-storage"), storage(3, "storage"),
      { binding: 4, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: PARAM_STRIDE } },
    ],
  });
  const built = device.createComputePipelineAsync({
    label: "int5-upload",
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: { module: device.createShaderModule({ label: "int5-upload.wgsl", code: SHADER }),
               entryPoint: "main" },
  }).then((pipeline) => ({ pipeline, layout }));
  PIPELINES.set(device, built);
  return built;
}

export async function runBlockUpload(device, plan, destination) {
  if (plan.params.length === 0) return () => {};
  const { pipeline, layout } = await uploadPipeline(device);
  const staging = [];
  const make = (size, usage) => {
    const buffer = device.createBuffer({ size: Math.max(4, size), usage });
    staging.push(buffer);
    return buffer;
  };
  const codes = make(plan.codeBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const scales = make(plan.halfBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const zeros = make(plan.halfBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  // 🔴 ASSEMBLED ON THE HOST AND UPLOADED ONCE, NOT WRITTEN CHUNK BY CHUNK.
  // `writeBuffer` takes a multiple of four bytes and these chunks are not: a
  // code run is groups * 20 + 1, and a scale table is groups * 2. Rounding each
  // one up would read past the end of the last tensor in a shard. Copying them
  // into an aligned staging array first is one memcpy of about an eighth of
  // what the old path decoded, and it makes every length the allocator's.
  const assemble = (target, chunks, size) => {
    if (size === 0) return;
    const staged = new Uint8Array(Math.ceil(size / 4) * 4);
    for (const chunk of chunks) {
      staged.set(new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength), chunk.at);
    }
    device.queue.writeBuffer(target, 0, staged, 0, staged.byteLength);
  };
  assemble(codes, plan.codeChunks, plan.codeBytes);
  assemble(scales, plan.scaleChunks, plan.halfBytes);
  assemble(zeros, plan.zeroChunks, plan.halfBytes);

  const uniforms = make(PARAM_STRIDE * plan.params.length,
                        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const table = new Uint32Array(PARAM_STRIDE / 4 * plan.params.length);
  plan.params.forEach((entry, index) => {
    const at = index * (PARAM_STRIDE / 4);
    table[at] = entry.codeBase; table[at + 1] = entry.scaleBase;
    table[at + 2] = entry.firstElement; table[at + 3] = entry.count;
    table[at + 4] = entry.destWord; table[at + 5] = entry.groupBase;
  });
  device.queue.writeBuffer(uniforms, 0, table);

  const bindGroup = device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: { buffer: codes } },
      { binding: 1, resource: { buffer: scales } },
      { binding: 2, resource: { buffer: zeros } },
      { binding: 3, resource: { buffer: destination } },
      { binding: 4, resource: { buffer: uniforms, size: PARAM_STRIDE } },
    ],
  });

  const encoder = device.createCommandEncoder({ label: "int5-upload" });
  const pass = encoder.beginComputePass({ label: "int5-upload" });
  pass.setPipeline(pipeline);
  plan.params.forEach((entry, index) => {
    pass.setBindGroup(0, bindGroup, [index * PARAM_STRIDE]);
    const pairs = Math.ceil(entry.count / 2);
    const groups = Math.ceil(pairs / LANES);
    pass.dispatchWorkgroups(Math.min(groups, 65535), Math.ceil(groups / 65535));
  });
  pass.end();
  device.queue.submit([encoder.finish()]);
  return () => { for (const buffer of staging) buffer.destroy(); };
}
