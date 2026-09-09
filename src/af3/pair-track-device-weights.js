/**
 * AF3's triangle multiplication weights, packed on the DEVICE.
 *
 * 🔴 THE PAIR TRACK'S PACKING IS 3.55 SECONDS OF AN OpenDDE FIRST FOLD.
 * Measured inside a 6MRR fold with the packer timed per tensor: tri-out 1194 ms,
 * tri-in 1204, grid1 641, grid2 654, over 56 blocks. Every one of those is an
 * int5 decode and a narrowing to f16 wrapped around a RESHAPE - which is
 * exactly why `residentPackedOnDevice` could not take them: it wrote one
 * contiguous run per tensor, and a run is what a copy is, not what a transpose
 * is.
 *
 * With the decoder able to reshape (see src/runtime/quantised-upload.js), the
 * triangle's whole packed buffer is one plan:
 *
 * | destination | source | mapping |
 * |---|---|---|
 * | `layerNormInWeight/Bias` | `left_norm_input/{scale,offset}` | a copy |
 * | `linearAPWeight` | `projection` | the a half of the interleaved fused matrix, transposed |
 * | `linearBPWeight` | `projection` | ...the b half, one element further along |
 * | `linearAGWeight`, `linearBGWeight` | `gate` | the same two |
 * | `linearABWeight` (matrix path) | `projection` and `gate` | all four, four roles to a channel |
 * | `layerNormOutWeight/Bias` | `center_norm/{scale,offset}` | a copy |
 * | `linearZWeight` | `output_projection` | transposed, or a COPY on the matrix path |
 * | `linearGWeight` | `gating_linear` | the same |
 * | every `*Bias` here | - | AF3 has none: they are zeros, and a fresh buffer already is |
 *
 * 🔴 THE MATRIX PATH'S TWO OUTPUT MATRICES ARE A COPY, NOT A TRANSPOSE, and
 * that is worth stating because the code says the opposite twice.
 * `af3TriangleWeights` transposes `output_projection` into the AF2 convention
 * and `transposeZG` transposes it back for the matrix kernel - so composed they
 * are the identity, and the packed buffer holds the bundle's own bytes.
 */
import { SOURCES, bindable } from "../runtime/weight-sources.js";
import { residentWeightBufferFilled } from "../runtime/resident.js";
import { planBlockUpload, runBlockUpload } from "../runtime/quantised-upload.js";
import { writeInto } from "../runtime/float16.js";

/** A copy: destination element d is source element d. */
const copy = (from, length) => ({
  length, sources: [{ from, bias: 0 }], inner: length, innerStride: 1, outerStride: 0,
});

/** `output[out * rows + in] = source[in * columns + out]`, over rows x columns. */
const transposed = (from, rows, columns) => ({
  length: rows * columns, sources: [{ from, bias: 0 }],
  inner: rows, innerStride: columns, outerStride: 1,
});

/**
 * One half of AF3's fused `[in][out][2]` projection, transposed into `[out][in]`.
 *
 *     destination[out * C + in] = fused[in * 2C + out * 2 + half]
 */
const splitHalf = (from, channels, half) => ({
  length: channels * channels, sources: [{ from, bias: half }],
  inner: channels, innerStride: 2 * channels, outerStride: 2,
});

/**
 * All four projections as one interleaved `[in][out][4]` block.
 *
 *     destination[in * 4C + out * 4 + role] = split(role)[out * C + in]
 *                                           = fused(role)[in * 2C + out * 2 + half(role)]
 *
 * with role 0 the a projection, 1 the a gate, 2 the b projection, 3 the b gate -
 * the order `interleaveAB` writes them in.
 */
const interleavedAB = (projection, gate, channels) => ({
  length: 4 * channels * channels,
  sources: [{ from: projection, bias: 0 }, { from: gate, bias: 0 },
            { from: projection, bias: 1 }, { from: gate, bias: 1 }],
  inner: channels, innerStride: 2, outerStride: 2 * channels,
});

/** Nothing to write: a fresh GPU buffer is already zero. */
const zeros = (length) => ({ length, zero: true });

/**
 * The destination layout, in `packOrder`'s order, for one of the two layouts.
 *
 * @param {"blocked"|"interleaved"} abLayout
 */
function triangleLayout(channels, abLayout) {
  const C = channels;
  const norms = [
    ["layerNormInWeight", copy("leftNormInputScale", C)],
    ["layerNormInBias", copy("leftNormInputOffset", C)],
  ];
  const middle = abLayout === "interleaved"
    ? [["linearABWeight", interleavedAB("projection", "gate", C)],
       ["linearABBias", zeros(4 * C)]]
    : [["linearAPWeight", splitHalf("projection", C, 0)], ["linearAPBias", zeros(C)],
       ["linearAGWeight", splitHalf("gate", C, 0)], ["linearAGBias", zeros(C)],
       ["linearBPWeight", splitHalf("projection", C, 1)], ["linearBPBias", zeros(C)],
       ["linearBGWeight", splitHalf("gate", C, 1)], ["linearBGBias", zeros(C)]];
  const out = [
    ["layerNormOutWeight", copy("centerNormScale", C)],
    ["layerNormOutBias", copy("centerNormOffset", C)],
    // Transposed for the vector kernel; the matrix path transposes it back, so
    // there the bundle's own order is what the buffer wants.
    ["linearZWeight", abLayout === "interleaved"
      ? copy("outputProjection", C * C) : transposed("outputProjection", C, C)],
    ["linearZBias", zeros(C)],
    ["linearGWeight", abLayout === "interleaved"
      ? copy("gatingLinear", C * C) : transposed("gatingLinear", C, C)],
    ["linearGBias", zeros(C)],
  ];
  // 🔴 THE ORDER IS packOrder's, NOT THIS FILE'S OPINION. `packWeights` filters
  // the four a/b names out and splices the interleaved pair in before
  // `layerNormOutWeight`; the offsets a compiled shader was given come from
  // that same call, so the two have to agree exactly or the buffer is a
  // plausible tensor in the wrong order.
  return abLayout === "interleaved"
    ? [...norms, ...middle, ...out]
    : [...norms, ...middle, ...out];
}

/**
 * Fill a resident buffer with one triangle's packed weights, decoding on the
 * device.
 *
 * @returns {Promise<GPUBuffer | undefined>} undefined when this cannot be done -
 *   an f32 destination, a store that cannot hand over codes, an f32 bundle -
 *   and the caller packs on the host as before.
 */
export async function residentTriangleOnDevice(device, options) {
  const { triangle, channels, abLayout, label, variant = "" } = options;
  // 🔴 THE DESTINATION'S ELEMENT, WHICH IS THE PAIR TRACK'S. AF3 keeps its
  // 128-channel pair weights in f32 and only a wide track goes to f16 - see
  // WIDE_PAIR_TRACK in fold.js - so refusing anything but f16 here left AF3
  // packing its triangles on the host for no reason. 282 ms of a 5.5-second
  // first fold.
  const destination = options.destination ?? "f16";
  if (destination === "f16" && typeof globalThis.Float16Array !== "function") return undefined;
  const sources = triangle?.[SOURCES];
  if (sources === undefined) return undefined;

  const layout = triangleLayout(channels, abLayout);
  const entries = [];
  const hostWrites = [];
  let total = 0;
  for (const [name, plan] of layout) {
    const offset = total;
    total += plan.length;
    if (plan.zero === true) continue;
    // 🔴 EVERY LENGTH HERE IS EVEN AT EVERY WIDTH THIS RUNS AT, and the planner
    // refuses an odd destination offset rather than letting two dispatches
    // share a word. A width that made one odd would fall back, not corrupt.
    // An f32 destination has no such rule: an element is a word.
    if (destination === "f16" && offset % 2 !== 0) return undefined;
    const parts = [];
    let hostOnly = false;
    for (const source of plan.sources) {
      const thunk = sources[source.from];
      if (!bindable(thunk)) {
        hostOnly = true;
        break;
      }
      parts.push({ store: thunk.store, tensorName: thunk.tensorName,
                   first: thunk.first, count: thunk.count,
                   bias: thunk.first + source.bias });
    }
    if (hostOnly) {
      hostWrites.push({ name, offset, length: plan.length, from: plan.sources[0].from });
      continue;
    }
    entries.push({ name, offset, length: plan.length, sources: parts,
                   inner: plan.inner, innerStride: plan.innerStride,
                   outerStride: plan.outerStride });
  }

  const planned = planBlockUpload(entries, destination);
  if (planned === undefined) return undefined;
  // 🔴 A RESHAPED ENTRY THE PLANNER SENT TO THE HOST IS A REFUSAL, NOT A
  // FALLBACK. `host` means "not quantised, narrow it yourself" - which is fine
  // for a copy and wrong for a transpose, because nothing here would do the
  // reshape. Rather than write a second implementation of every mapping, the
  // whole triangle goes back to the host packer.
  for (const entry of planned.host) {
    if (entry.inner !== entry.length || entry.sources.length !== 1
      || entry.innerStride !== 1) return undefined;
    hostWrites.push({ name: entry.name, offset: entry.offset, length: entry.length,
                      from: layout.find(([n]) => n === entry.name)[1].sources[0].from });
  }
  if (planned.gpu.params.length === 0) return undefined;

  const bytes = destination === "f32" ? total * 4 : Math.ceil(total / 2) * 4;
  return residentWeightBufferFilled(device, triangle, label, bytes,
    async (buffer) => {
      for (const write of hostWrites) {
        if (destination === "f32") {
          const values = triangle[write.from];
          device.queue.writeBuffer(buffer, write.offset * 4,
                                   values.buffer, values.byteOffset, values.byteLength);
          continue;
        }
        const half = new globalThis.Float16Array(write.length);
        writeInto(half, triangle[write.from], 0);
        device.queue.writeBuffer(buffer, write.offset * 2,
                                 half.buffer, half.byteOffset, half.byteLength);
      }
      const release = await runBlockUpload(device, planned.gpu, buffer);
      // Not awaited; see src/af3/device-weights.js for why.
      void device.queue.onSubmittedWorkDone().then(release);
    },
    variant);
}

/**
 * Grid attention's packed weights, on the device.
 *
 * 🔴 f32, WHICH IS WHY IT IS A SEPARATE FUNCTION AND A SEPARATE SHADER.
 * `packGridAttentionWeights` takes no precision: this buffer is float32 whatever
 * the pair track holds, so the decoder writes one element a word - and that is
 * also what lets the four interleaved projections be four dispatches with a
 * destination stride of four. They cannot share one source mapping: v is stored
 * the way the kernel reads it and q, k and the gate are TRANSPOSED into v's
 * layout when they are packed, which is a coalescing change the packer makes.
 *
 * Measured inside a 6MRR OpenDDE fold: grid1 641 ms and grid2 654 over 56
 * blocks, the second largest item in a first fold after the triangles.
 *
 * @returns {Promise<GPUBuffer | undefined>}
 */
export async function residentGridOnDevice(device, options) {
  const { grid, label, variant = "" } = options;
  const sources = grid?.[SOURCES];
  if (sources === undefined) return undefined;
  const width = grid.heads * grid.dimension;

  // `packGridAttentionWeights`'s ORDER, and its lengths - which come from the
  // tensors rather than from a formula, because a bundle's widths are its own.
  const lengthOf = (name) => {
    const thunk = sources[name];
    return bindable(thunk) ? thunk.count : undefined;
  };
  const QKVG = ["qProjection", "kProjection", "vProjection", "gatingQuery"];
  const TRANSPOSED = new Set(["qProjection", "kProjection", "gatingQuery"]);
  const plain = ["actNormScale", "actNormOffset", "pairBiasProjection"];
  for (const name of [...plain, ...QKVG, "outputProjection"]) {
    if (lengthOf(name) === undefined) return undefined;
  }
  const qkvgLength = QKVG.reduce((total, name) => total + lengthOf(name), 0);

  const entries = [];
  let total = 0;
  for (const name of plain) {
    const thunk = sources[name];
    entries.push({ name, offset: total, length: thunk.count, destStride: 1,
                   sources: [{ store: thunk.store, tensorName: thunk.tensorName,
                               first: thunk.first, count: thunk.count, bias: thunk.first }],
                   inner: thunk.count, innerStride: 1, outerStride: 0 });
    total += thunk.count;
  }
  const qkvgBase = total;
  QKVG.forEach((name, lane) => {
    const thunk = sources[name];
    const channels = thunk.count / width;
    if (!Number.isInteger(channels)) throw new Error(`${name} is not a multiple of ${width}`);
    // `transposeOutChannels`: out[c * width + o] = values[o * channels + c].
    const mapped = TRANSPOSED.has(name)
      ? { inner: width, innerStride: channels, outerStride: 1 }
      : { inner: thunk.count, innerStride: 1, outerStride: 0 };
    entries.push({ name, offset: qkvgBase + lane, length: thunk.count, destStride: 4,
                   sources: [{ store: thunk.store, tensorName: thunk.tensorName,
                               first: thunk.first, count: thunk.count, bias: thunk.first }],
                   ...mapped });
  });
  total += qkvgLength;
  {
    const thunk = sources.outputProjection;
    entries.push({ name: "outputProjection", offset: total, length: thunk.count, destStride: 1,
                   sources: [{ store: thunk.store, tensorName: thunk.tensorName,
                               first: thunk.first, count: thunk.count, bias: thunk.first }],
                   inner: thunk.count, innerStride: 1, outerStride: 0 });
    total += thunk.count;
  }

  const planned = planBlockUpload(entries, "f32");
  if (planned === undefined) return undefined;
  // 🔴 A RESHAPED HOST ENTRY IS A REFUSAL; A COPY IS NOT. `host` means "not
  // quantised, write it yourself" - and the two LayerNorm vectors here are
  // float32 in every bundle, so refusing on any host entry at all refused the
  // whole grid pack and saved nothing. That is what the first measurement of
  // this said: 1532 ms to 1481, when the grid is 1.3 s of it.
  const hostWrites = [];
  for (const entry of planned.host) {
    if (entry.sources.length !== 1 || entry.inner !== entry.length
      || entry.innerStride !== 1 || (entry.destStride ?? 1) !== 1) return undefined;
    hostWrites.push(entry);
  }
  if (planned.gpu.params.length === 0) return undefined;

  return residentWeightBufferFilled(device, grid, label, total * 4,
    async (buffer) => {
      for (const write of hostWrites) {
        // f32 in, f32 out: no narrowing, so this is the tensor's own bytes.
        const values = grid[write.name];
        device.queue.writeBuffer(buffer, write.offset * 4,
                                 values.buffer, values.byteOffset, values.byteLength);
      }
      const release = await runBlockUpload(device, planned.gpu, buffer);
      void device.queue.onSubmittedWorkDone().then(release);
    },
    variant);
}

/**
 * A whole pair-track block's packed weights, as far as the device can take
 * them.
 *
 * The three stacks that share `packPairTrackWeights` - the pairformer, the MSA
 * stack and the template embedder - all want the same four buffers and all
 * paid the same host decode for them. This is that, once.
 *
 * @param {object} options {channels, pairWeightPrecision, abLayout, resident}
 * @returns {Promise<{buffers: object, want: {triangles: boolean, grids: boolean}}>}
 *   `buffers` holds only what the device could fill; `want` is what the host
 *   packer must still build, in the shape `packPairTrackWeights` reads.
 */
export async function residentPairTrackOnDevice(device, block, options) {
  const { channels, pairWeightPrecision, abLayout = "blocked", resident = true } = options;
  if (!resident) return { buffers: {}, want: { triangles: true, grids: true } };
  // 🔴 THE VARIANT IS THE PRECISION AND NOT THE LAYOUT, which is the same hole
  // the host packer has: two runs in ONE process with `triangleProjectMatrix`
  // set differently would share a buffer packed for the other kernel. Both are
  // safe today because the layout comes from a device-profile knob that is
  // fixed for a process, and every arm this repository measures is its own
  // process. A caller that changed the knob mid-process would get a finite,
  // plausible tensor - so if that ever becomes possible, the layout belongs in
  // the variant on both paths at once.
  const [outgoing, incoming] = await Promise.all([
    residentTriangleOnDevice(device, {
      triangle: block.triangleMultiplicationOutgoing, channels, abLayout,
      label: "w.tri.out", variant: pairWeightPrecision,
      destination: pairWeightPrecision === "f16" ? "f16" : "f32" }),
    residentTriangleOnDevice(device, {
      triangle: block.triangleMultiplicationIncoming, channels, abLayout,
      label: "w.tri.in", variant: pairWeightPrecision,
      destination: pairWeightPrecision === "f16" ? "f16" : "f32" }),
  ]);
  const [grid1, grid2] = await Promise.all([
    residentGridOnDevice(device, { grid: block.pairAttention1, label: "w.grid1" }),
    residentGridOnDevice(device, { grid: block.pairAttention2, label: "w.grid2" }),
  ]);
  // 🔴 BOTH OR NEITHER, of each pair. `packPairTrackWeights` is one call, so
  // skipping half of a pair saves nothing and hands the host packer a shape it
  // has no flag for.
  const triangles = !(outgoing !== undefined && incoming !== undefined);
  const grids = !(grid1 !== undefined && grid2 !== undefined);
  const buffers = {};
  if (!triangles) {
    buffers.outgoing = { buffer: outgoing };
    buffers.incoming = { buffer: incoming };
  }
  if (!grids) {
    buffers.grid1 = { buffer: grid1 };
    buffers.grid2 = { buffer: grid2 };
  }
  return { buffers, want: { triangles, grids } };
}
