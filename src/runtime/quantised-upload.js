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

import { packedBits } from "../reference/dtype.js";

/** The uniform stride a Params array needs; WebGPU wants 256 for a dynamic offset. */
const PARAM_STRIDE = 256;

const LANES = 64;

/**
 * 🔴 THE CODEC IS THE TENSOR'S, NOT THIS FILE'S. This decoded int5 at a group
 * of 32 and nothing else, written into the constants and into the shader's
 * `* 5u` and `& 31u`. ESM-C ships **int3 at a group of 128** - the only bundle
 * in the repository that is not int5 - so the whole of its 600M parameters had
 * to be decoded on the HOST, which `src/esmc/tower-webgpu.js` measures at 1121
 * ms of decode and 723 of narrowing across 36 blocks against 660 ms of GPU.
 *
 * Both are the same arithmetic at a different width. What a codec has to
 * satisfy is that a group is a whole number of BYTES - 32 x 5 = 160 bits and
 * 128 x 3 = 384, both exact - and that a code never spans more than two bytes,
 * which holds for any width up to eight.
 */
function codecOf(record) {
  // 🔴 int8 IS SIGNED AND SYMMETRIC, WHICH IS TWO DIFFERENCES AND NOT ONE. Its
  // codes are two's complement rather than unsigned, and it carries a scale per
  // group and NO zero point - `tensorByteLength` already splits on that, and
  // `readTensorRange`'s int8 branch is `code * scale` with no `+ zero`. AF2's
  // 98 MiB bundle is 283 int8 tensors of 337 and 445 ms of host decoding, which
  // is why it is here at all.
  const bits = record.dtype === "int8" ? 8 : packedBits(record.dtype);
  if (bits === null) return null;
  const group = record.block;
  if (!Number.isInteger(group) || group <= 0) return null;
  if ((group * bits) % 8 !== 0) return null;
  return { bits, group, groupBytes: (group * bits) / 8, signed: bits === 8 };
}

/** Bytes of code for a run of elements, widened to whole quantisation groups. */
function codeSpan(first, count, codec) {
  const firstGroup = Math.floor(first / codec.group);
  const lastGroup = Math.ceil((first + count) / codec.group);
  return { firstGroup, lastGroup,
           byteStart: firstGroup * codec.groupBytes,
           // ...one byte of slack, because a code may straddle into it; the
           // packer leaves it for exactly this reason.
           byteLength: (lastGroup - firstGroup) * codec.groupBytes + 1 };
}

const shaderFor = (codec, destination) => `
// 🔴 FOUR SOURCES A DESTINATION, BECAUSE A PACKED WEIGHT IS NOT ALWAYS A COPY.
// The contiguous case is "parts = 1": destination element d reads source
// element "bias0 + d". The reshapes this repository's packers do are all the
// same shape of thing one step out - destination element d reads
//
//     part = d % parts
//     d2   = d / parts
//     src  = bias[part] + (d2 / inner) * outerStride + (d2 % inner) * innerStride
//
// which covers a transpose (parts 1, inner = rows, innerStride = columns,
// outerStride = 1), AF3's interleaved a/b split (inner = C, innerStride = 2C,
// outerStride = 2, bias 0 or 1) and the matrix path's four-role interleave
// (parts 4, one bias a role). The DESTINATION stays contiguous, which is what
// keeps one invocation owning one whole word and lets this be a plain store
// rather than an atomic read-modify-write of sixteen bits.
struct Params {
  codeBase: vec4<u32>,      // in bytes, into the code buffer, one a part
  scaleBase: vec4<u32>,     // in f16 elements, into the scale table
  bias: vec4<u32>,          // the source element each part counts from
  groupBase: vec4<u32>,     // the group scaleBase corresponds to
  count: u32,
  destWord: u32,            // first destination word
  parts: u32,
  inner: u32,
  innerStride: u32,
  outerStride: u32,
  destStride: u32,
  partRun: u32,          // how many destination elements a part owns in a row
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

/** One element of this destination run, counted from its first. */
fn value_at(within: u32) -> f32 {
  // 🔴 SELECTED, NOT SUBSCRIPTED. "params.codeBase[part]" is a dynamic index
  // into a vector, which WGSL puts in addressable memory - see CLAUDE.md, where
  // that costs 4x in a hot loop. Four parts is three selects.
  // 🔴 A PART OWNS A RUN, NOT AN ELEMENT. With partRun 1 the parts alternate
  // element by element, which is the four-role interleave; with partRun equal
  // to a matrix's column count they alternate in whole rows, which is two
  // matrices concatenated along their COLUMNS. ESMFold2's adaptive LayerNorm
  // wants the second: its gate and its shift are one buffer so the kernel reads
  // them in one dispatch, and building that on the host is 54 MiB of copy and
  // narrowing a fold.
  let part = select(0u, (within / params.partRun) % params.parts, params.parts > 1u);
  var codeBase = params.codeBase.x;
  var scaleBase = params.scaleBase.x;
  var bias = params.bias.x;
  var groupBase = params.groupBase.x;
  if (part == 1u) {
    codeBase = params.codeBase.y; scaleBase = params.scaleBase.y;
    bias = params.bias.y; groupBase = params.groupBase.y;
  } else if (part == 2u) {
    codeBase = params.codeBase.z; scaleBase = params.scaleBase.z;
    bias = params.bias.z; groupBase = params.groupBase.z;
  } else if (part == 3u) {
    codeBase = params.codeBase.w; scaleBase = params.scaleBase.w;
    bias = params.bias.w; groupBase = params.groupBase.w;
  }
  let d2 = (within / (params.partRun * params.parts)) * params.partRun
    + (within % params.partRun);
  let absolute = bias + (d2 / params.inner) * params.outerStride
    + (d2 % params.inner) * params.innerStride;
  let group = absolute / GROUP;
  let table = scaleBase + (group - groupBase);
  let scale = scale_at(table);
${codec.signed ? "" : "  let zero = zero_at(table);"}
  let bit = (absolute % GROUP) * ${codec.bits}u;
  let at = codeBase + (group - groupBase) * GROUP_BYTES + (bit >> 3u);
  // 🔴 TWO BYTES ARE ALWAYS ENOUGH: a code starts at bit offset 0-7 and is at
  // most eight bits wide, so it ends by bit 15. Any width up to eight is safe.
  let pair = byte_at(at) | (byte_at(at + 1u) << 8u);
  let code = (pair >> (bit & 7u)) & ${(1 << codec.bits) - 1}u;
${codec.signed
  ? `  // Two's complement in eight bits, and no zero point: see codecOf.
  return f32(i32(code << 24u) >> 24u) * scale;`
  : "  return f32(code) * scale + zero;"}
}

const GROUP: u32 = ${codec.group}u;
const GROUP_BYTES: u32 = ${codec.groupBytes}u;

@compute @workgroup_size(${LANES})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
${destination === "f32" ? `  // 🔴 ONE ELEMENT A WORD, SO THE DESTINATION MAY STRIDE. An f32 element IS a
  // word, so writing every fourth one races nobody - which is what lets grid
  // attention's four interleaved projections be four dispatches instead of one
  // mapping they cannot share. The f16 path below cannot do this: two elements
  // share a word and a strided write would be a read-modify-write of sixteen
  // bits, which WGSL has no way to do.
  let within = id.x + id.y * 65535u * ${LANES}u;
  if (within >= params.count) { return; }
  output[params.destWord + params.destStride * within] = bitcast<u32>(value_at(within));`
: `  let pair = id.x + id.y * 65535u * ${LANES}u;
  let within = pair * 2u;
  if (within >= params.count) { return; }
  let a = value_at(within);
  var b = 0.0;
  if (within + 1u < params.count) { b = value_at(within + 1u); }
  output[params.destWord + pair] = pack2x16float(vec2<f32>(a, b));`}
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
export function planBlockUpload(entries, destination = "f16") {
  const codeChunks = [];
  const scaleChunks = [];
  const zeroChunks = [];
  const params = [];
  let codeBytes = 0;
  let halfCount = 0;
  let codec = null;
  const host = [];
  for (const entry of entries) {
    // 🔴 TWO ENTRY SHAPES, ONE PLAN. The old one is a `thunk` and a contiguous
    // copy; the new one is `sources` plus a stride triple, which is the same
    // thing when there is one source, `inner` is the whole length and the
    // stride is one. Normalising here rather than at the call sites keeps the
    // refusals - even offsets, one codec, a store that can hand over bytes - in
    // one place.
    const { offset, length } = entry;
    const mapped = entry.sources !== undefined;
    const sources = mapped ? entry.sources : [{
      store: entry.thunk?.store, tensorName: entry.thunk?.tensorName,
      first: entry.thunk?.first, count: entry.thunk?.count, bias: entry.thunk?.first,
    }];
    const inner = mapped ? entry.inner : length;
    const innerStride = mapped ? entry.innerStride : 1;
    const outerStride = mapped ? entry.outerStride : 0;
    const partRun = mapped ? (entry.partRun ?? 1) : 1;
    if (!mapped && entry.thunk?.count !== length) return undefined;
    // 🔴 EVERY OFFSET EVEN, WHICH ALSO MAKES EVERY LENGTH EVEN. The offsets are
    // a running sum, so one odd length would make every later offset odd and
    // this would refuse - which is the point: an odd boundary is two writers
    // sharing a word. An f32 destination has no such rule: an element IS a
    // word, which is also why only that one may stride.
    if (destination === "f16" && offset % 2 !== 0) return undefined;
    if (destination === "f16" && (entry.destStride ?? 1) !== 1) return undefined;
    if (sources.length < 1 || sources.length > 4) return undefined;
    if (!(inner > 0)) return undefined;
    if (mapped && length % sources.length !== 0) return undefined;
    // 🔴 A RUN AN f16 DESTINATION CAN OWN IS AN EVEN ONE. One invocation writes
    // one WORD, which is two adjacent destination elements; an odd run would
    // put them in different parts and the second would be written by nobody.
    if (!(partRun >= 1) || (destination === "f16" && partRun > 1 && partRun % 2 !== 0)) {
      return undefined;
    }

    const resolved = [];
    let toHost = false;
    for (const part of sources) {
      if (typeof part.store?.tensorSource !== "function") return undefined;
      let source;
      try { source = part.store.tensorSource(part.tensorName); } catch { return undefined; }
      const entryCodec = codecOf(source.record);
      if (entryCodec === null) { toHost = true; break; }
      // A symmetric codec has no zero table; an asymmetric one without a zero
      // offset is a record this cannot read.
      if (!entryCodec.signed && !Number.isInteger(source.record.zeroOffset)) return undefined;
      // 🔴 ONE CODEC A PLAN, because the shader is generated from it and a plan
      // is one dispatch per tensor sharing one pipeline. No bundle here mixes
      // them - AF3 and OpenDDE are int5 at 32, ESM-C is int3 at 128 - and a
      // bundle that did would send the odd ones out to the host packer rather
      // than getting a wrong answer from the wrong shift.
      if (codec === null) codec = entryCodec;
      else if (codec.bits !== entryCodec.bits || codec.group !== entryCodec.group
        || codec.signed !== entryCodec.signed) {
        toHost = true;
        break;
      }
      resolved.push({ part, source });
    }
    if (toHost) { host.push(entry); continue; }

    const bases = [];
    for (const { part, source } of resolved) {
      const { record, buffer, byteOffset } = source;
      const span = codeSpan(part.first, part.count, codec);
      const base = record.byteOffset ?? 0;
      codeChunks.push({ buffer, byteOffset: byteOffset + span.byteStart,
                        byteLength: span.byteLength, at: codeBytes });
      const groups = span.lastGroup - span.firstGroup;
      scaleChunks.push({ buffer,
                         byteOffset: byteOffset + (record.scaleOffset - base)
                           + span.firstGroup * 2,
                         byteLength: groups * 2, at: halfCount * 2 });
      // 🔴 A SYMMETRIC CODEC STILL BINDS A ZEROS BUFFER, and it is zeros. The
      // shader generated for it never reads them, but the bind group layout is
      // one layout for every codec and a binding it cannot fill is a validation
      // error rather than a fast path.
      if (!codec.signed) {
        zeroChunks.push({ buffer,
                          byteOffset: byteOffset + (record.zeroOffset - base)
                            + span.firstGroup * 2,
                          byteLength: groups * 2, at: halfCount * 2 });
      }
      bases.push({ codeBase: codeBytes, scaleBase: halfCount,
                   bias: part.bias ?? part.first, groupBase: span.firstGroup });
      codeBytes += Math.ceil(span.byteLength / 4) * 4;
      halfCount += Math.ceil(groups / 2) * 2;
    }
    params.push({ sources: bases, count: length,
                  destWord: destination === "f32" ? offset : offset / 2,
                  destStride: entry.destStride ?? 1, partRun,
                  inner, innerStride, outerStride });
  }
  return { gpu: { codeChunks, scaleChunks, zeroChunks, params, codec, destination,
                  codeBytes, halfBytes: halfCount * 2 }, host };
}

/**
 * Run a plan, filling `destination` in place.
 *
 * The caller owns `destination`; it must be at least the packed size and carry
 * STORAGE usage.
 */
async function uploadPipeline(device, codec, destination) {
  let forDevice = PIPELINES.get(device);
  if (forDevice === undefined) {
    forDevice = new Map();
    PIPELINES.set(device, forDevice);
  }
  const key = `${codec.bits}:${codec.group}:${codec.signed ? "s" : "u"}:${destination}`;
  const found = forDevice.get(key);
  if (found !== undefined) return found;
  const storage = (binding, type) => ({
    binding, visibility: GPUShaderStage.COMPUTE, buffer: { type },
  });
  const layout = device.createBindGroupLayout({
    label: `int${codec.bits}-upload`,
    entries: [
      storage(0, "read-only-storage"), storage(1, "read-only-storage"),
      storage(2, "read-only-storage"), storage(3, "storage"),
      { binding: 4, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: PARAM_STRIDE } },
    ],
  });
  const built = device.createComputePipelineAsync({
    label: `int${codec.bits}-upload`,
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: { module: device.createShaderModule({
      label: `int${codec.bits}-upload.wgsl`, code: shaderFor(codec, destination) }),
      entryPoint: "main" },
  }).then((pipeline) => {
    const resolved = { pipeline, layout };
    // 🔴 KEPT AS A VALUE AND NOT ONLY AS A PROMISE. `await` on a settled
    // promise is still a trip through the microtask queue, and this is awaited
    // once per TENSOR - 306 times in an ESMFold2 fold, 474 in an AF2 one. The
    // measured cost of those awaits was 466 ms, which is not CPU but is still
    // time the caller is not decoding in.
    forDevice.set(`${key}!`, resolved);
    return resolved;
  });
  forDevice.set(key, built);
  return built;
}

/** The pipeline if it is already built, without a microtask. */
function builtPipeline(device, codec, destination) {
  const key = `${codec.bits}:${codec.group}:${codec.signed ? "s" : "u"}:${destination}!`;
  return PIPELINES.get(device)?.get(key);
}

/**
 * What every `runBlockUpload` on this page has cost, for the tools.
 *
 * 🔴 BECAUSE THE DECODE IS NOW THE PART NOBODY IS TIMING. AF2's first fold
 * spends 0.2-0.3 s more in its main stack than a warm one does, and that is 47
 * blocks x 9 packs of staging assembly, buffer creation and submit - none of it
 * inside a compute pass, so `tools/gpu/profile.js` cannot see it and neither
 * can the block profiler. Counting it here is one object and no branch.
 */
/**
 * One scratch array for every staging assembly on the page.
 *
 * 🔴 A FRESH `new Uint8Array` PER TENSOR SET IS AN ALLOCATION AND A ZERO-FILL,
 * AND AF2's FIRST FOLD DOES 474 OF THEM. Measured: 129 ms to assemble 92 MiB,
 * which is 713 MB/s for what is a memcpy - the copy is not the cost, the 1422
 * allocations and their zeroing are. The array only has to be big enough and
 * only has to hold what is written before the writeBuffer that reads it, and
 * `writeBuffer` copies synchronously into the queue's own staging, so the next
 * call may overwrite it. It grows and never shrinks; AF2's largest is 4 MiB.
 *
 * 🔴 THE BYTES BEYOND A CHUNK ARE NOT READ, WHICH IS WHY REUSE IS SAFE. Chunk
 * positions are padded up to four bytes and a decode reads at most one byte
 * past its last group - the slack byte the packer leaves, and which
 * `codeSpan` puts INSIDE `byteLength`. Nothing reads the padding, so nothing
 * reads a previous tensor's bytes through it. An empty chunk list is skipped
 * entirely rather than assembled, so a buffer nobody fills stays zeroed.
 */
let sharedStaging = new Uint8Array(0);
function stagingArray(bytes) {
  if (sharedStaging.length < bytes) sharedStaging = new Uint8Array(Math.ceil(bytes * 1.5));
  return sharedStaging.subarray(0, bytes);
}

export const blockUploadStats = {
  calls: 0, tensors: 0, submits: 0, buffers: 0,
  stagingBytes: 0, stagingMs: 0, totalMs: 0,
  pipelineMs: 0, uniformMs: 0, encodeMs: 0, submitMs: 0,
};

/**
 * The GPU staging buffers, pooled per device rather than created per tensor.
 *
 * 🔴 1224 `createBuffer` CALLS AND 1224 `destroy`s IS 400 ms OF AN ESMFold2
 * FOLD. Its 306 uploads take four buffers each - codes, scales, zeros and the
 * uniforms - and the whole decode costs 602 ms of host time of which only 205
 * is the staging assembly. A buffer here lives about a millisecond and is then
 * thrown away, which is the one allocation pattern a pool is unambiguously for.
 *
 * Sizes are rounded up to a power of two so that a fold's hundreds of uploads
 * reuse a handful of buffers, and the pool is bounded: past `POOL_LIMIT` per
 * class the extra is destroyed rather than kept, so a single huge tensor does
 * not pin its buffer for the session. A buffer returns to the pool only when
 * the submit that reads it is done, which is the same signal the destroy used.
 */
const STAGING_POOLS = new WeakMap();
const POOL_LIMIT = 8;
/**
 * 🔴 AND THE POOL IS BOUNDED IN BYTES, NOT ONLY IN COUNT. Eight of a 128 MiB
 * class is a gibibyte held for a fold that will never ask for it again, on a
 * path whose whole purpose is to keep a phone folding - so a release past this
 * destroys instead of keeping, and the pool stays smaller than one block's
 * weights.
 */
const POOL_BYTES = 64 * 1024 * 1024;

function poolFor(device, usage) {
  let state = STAGING_POOLS.get(device);
  if (state === undefined) { state = { bytes: 0, byUsage: new Map() }; STAGING_POOLS.set(device, state); }
  let pool = state.byUsage.get(usage);
  if (pool === undefined) { pool = new Map(); state.byUsage.set(usage, pool); }
  return { pool, state };
}

/** The smallest power of two that holds `size`, and never below 256 bytes. */
function stagingClass(size) {
  let bytes = 256;
  while (bytes < size) bytes *= 2;
  return bytes;
}

function acquireStaging(device, size, usage) {
  const bytes = stagingClass(Math.max(4, size));
  const { pool, state } = poolFor(device, usage);
  const free = pool.get(bytes);
  if (free !== undefined && free.length > 0) {
    state.bytes -= bytes;
    return { buffer: free.pop(), bytes, usage };
  }
  return { buffer: device.createBuffer({ size: bytes, usage }), bytes, usage };
}

function releaseStaging(device, held) {
  for (const { buffer, bytes, usage } of held) {
    const { pool, state } = poolFor(device, usage);
    const free = pool.get(bytes) ?? [];
    if (free.length >= POOL_LIMIT || state.bytes + bytes > POOL_BYTES) {
      buffer.destroy();
      continue;
    }
    free.push(buffer);
    state.bytes += bytes;
    pool.set(bytes, free);
  }
}

export async function runBlockUpload(device, plan, destination) {
  if (plan.params.length === 0) return () => {};
  const startedAt = performance.now();
  const element = plan.destination ?? "f16";
  const pipelineAt = performance.now();
  const { pipeline, layout } = builtPipeline(device, plan.codec, element)
    ?? await uploadPipeline(device, plan.codec, element);
  blockUploadStats.pipelineMs += performance.now() - pipelineAt;
  const staging = [];
  const make = (size, usage) => {
    const held = acquireStaging(device, size, usage);
    staging.push(held);
    return held.buffer;
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
    if (size === 0 || chunks.length === 0) return;
    const bytes = Math.ceil(size / 4) * 4;
    const staged = stagingArray(bytes);
    for (const chunk of chunks) {
      staged.set(new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength), chunk.at);
    }
    device.queue.writeBuffer(target, 0, staged, 0, bytes);
  };
  const stagingAt = performance.now();
  assemble(codes, plan.codeChunks, plan.codeBytes);
  assemble(scales, plan.scaleChunks, plan.halfBytes);
  // 🔴 AN EMPTY CHUNK LIST IS SKIPPED AND NOT ASSEMBLED, which is what makes the
  // shared scratch safe for the ZEROS. A symmetric codec has no zero table, the
  // shader generated for it never reads one, and the binding exists only
  // because the layout is one layout for every codec - so the buffer keeps
  // WebGPU's own zero-initialisation instead of a previous tensor's codes.
  assemble(zeros, plan.zeroChunks, plan.halfBytes);
  blockUploadStats.stagingMs += performance.now() - stagingAt;
  blockUploadStats.stagingBytes += plan.codeBytes + plan.halfBytes * 2;

  const uniformAt = performance.now();
  const uniforms = make(PARAM_STRIDE * plan.params.length,
                        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const table = new Uint32Array(PARAM_STRIDE / 4 * plan.params.length);
  plan.params.forEach((entry, index) => {
    const at = index * (PARAM_STRIDE / 4);
    // Four vec4s, then the six scalars; see the Params struct in shaderFor.
    for (let part = 0; part < 4; part += 1) {
      const source = entry.sources[Math.min(part, entry.sources.length - 1)];
      table[at + part] = source.codeBase;
      table[at + 4 + part] = source.scaleBase;
      table[at + 8 + part] = source.bias;
      table[at + 12 + part] = source.groupBase;
    }
    table[at + 16] = entry.count;
    table[at + 17] = entry.destWord;
    table[at + 18] = entry.sources.length;
    table[at + 19] = entry.inner;
    table[at + 20] = entry.innerStride;
    table[at + 21] = entry.outerStride;
    table[at + 22] = entry.destStride ?? 1;
    table[at + 23] = entry.partRun ?? 1;
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

  blockUploadStats.uniformMs += performance.now() - uniformAt;
  const encodeAt = performance.now();
  const encoder = device.createCommandEncoder({ label: "int5-upload" });
  const pass = encoder.beginComputePass({ label: "int5-upload" });
  pass.setPipeline(pipeline);
  plan.params.forEach((entry, index) => {
    pass.setBindGroup(0, bindGroup, [index * PARAM_STRIDE]);
    const slots = element === "f32" ? entry.count : Math.ceil(entry.count / 2);
    const groups = Math.ceil(slots / LANES);
    pass.dispatchWorkgroups(Math.min(groups, 65535), Math.ceil(groups / 65535));
  });
  pass.end();
  const commands = encoder.finish();
  blockUploadStats.encodeMs += performance.now() - encodeAt;
  const submitAt = performance.now();
  device.queue.submit([commands]);
  blockUploadStats.submitMs += performance.now() - submitAt;
  blockUploadStats.calls += 1;
  blockUploadStats.tensors += plan.params.length;
  blockUploadStats.submits += 1;
  blockUploadStats.buffers += staging.length;
  blockUploadStats.totalMs += performance.now() - startedAt;
  return () => releaseStaging(device, staging);
}
