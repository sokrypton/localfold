/**
 * The triangle's a/b projection and contraction, packed against unpacked.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-triangle-packed.js --length=68
 *
 * 🔴 THIS EXISTS BECAUSE THE FOLD SAID "WRONG" AND NOT "WHERE". Packing a and b
 * moved AF3's trunk from relRMS 2.7e-2 against its own tensors to 3.5e-1, which
 * is an addressing fault and not half precision - f16 storage is worth about
 * 1e-3 here. A whole fold cannot say which of the two kernels did it, and
 * reading them against each other by eye did not either.
 *
 * It runs projectAB both ways on ONE input and compares `a` and `b` directly,
 * then runs the contraction on each and compares that. Whichever comparison is
 * the first to blow up is the kernel at fault.
 */
import { createTriangleShaders } from "../../src/triangle/shaders.js";
import { packWeights } from "../../src/triangle/weights.js";
import { unpackHalfWords } from "../../src/runtime/storage.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

// 🔴 THE REAL PACKER, NOT A HAND-WRITTEN LIST. The first version of this file
// spelled the weight names out and got two of them wrong, so projectOutput
// failed to compile against a name that does not exist. packWeights owns the
// order and the offsets; asking it is the only way they cannot drift.

export async function main(device, args) {
  const length = Number(option(args, "length", "68"));
  const cZ = Number(option(args, "channels", "128"));
  const cH = cZ;
  const pairs = length * length;

  let state = 99991;
  const random = (count, scale = 1) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      out[i] = (state / 0x7fffffff - 0.5) * scale;
    }
    return out;
  };
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
  const upload = (data) => {
    const buffer = device.createBuffer({
      size: data.byteLength, usage: storage, mappedAtCreation: true });
    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();
    return buffer;
  };
  const blank = (bytes) => device.createBuffer({ size: bytes, usage: storage });

  const matrix = () => random(cH * cZ);
  const vector = () => random(cZ);
  const packedWeights = packWeights({
    layerNormInWeight: vector(), layerNormInBias: vector(),
    linearAPWeight: matrix(), linearAPBias: vector(),
    linearAGWeight: matrix(), linearAGBias: vector(),
    linearBPWeight: matrix(), linearBPBias: vector(),
    linearBGWeight: matrix(), linearBGBias: vector(),
    layerNormOutWeight: vector(), layerNormOutBias: vector(),
    linearZWeight: matrix(), linearZBias: vector(),
    linearGWeight: matrix(), linearGBias: vector(),
  }, "f32");
  const offsets = packedWeights.offsets;
  const weights = upload(packedWeights.data);
  const z = upload(random(pairs * cZ));
  const mask = upload(new Float32Array(pairs).fill(1));
  const readback = device.createBuffer({
    size: cH * pairs * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  // 🔴 SWEEP THE AXES THE FOLD ACTUALLY VARIES. The first version of this check
  // ran one direction at one accumulator precision, said both kernels were
  // right, and the fold went on being wrong - so it was testing a configuration
  // nothing runs. The contraction is the only thing `direction` changes, and
  // `accumulatePrecision` is a separate axis from the storage.
  const direction = option(args, "direction", "outgoing");
  const accumulatePrecision = option(args, "accumulate", "f32");
  const shape = { length, cZ, cHidden: cH, accumulatePrecision };
  const build = (ab) => createTriangleShaders(
    shape, "f32", offsets, 1e-5, direction, "two-pass", undefined, false, undefined, { ab });
  const plain = build("f32");
  const packed = build("f16");

  const pipelineFor = async (source) => device.createComputePipelineAsync({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: source }), entryPoint: "main" },
  });
  const dispatch = async (source, buffers, x, y, zCount = 1, readFrom = undefined, bytes = 0) => {
    const pipeline = await pipelineFor(source);
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    }));
    pass.dispatchWorkgroups(x, y, zCount);
    pass.end();
    if (readFrom !== undefined) encoder.copyBufferToBuffer(readFrom, 0, readback, 0, bytes);
    device.queue.submit([encoder.finish()]);
    if (readFrom === undefined) { await device.queue.onSubmittedWorkDone(); return undefined; }
    await readback.mapAsync(GPUMapMode.READ);
    const out = readback.getMappedRange().slice(0);
    readback.unmap();
    return out;
  };

  const projectGroups = [Math.ceil(cH / plain.projectTile.columns),
                         Math.ceil(pairs / plain.projectTile.rows)];
  const contractGroups = [Math.ceil(length / plain.contractTile.columns),
                          Math.ceil(length / plain.contractTile.rows)];

  // --- a and b, straight out of projectAB ---
  const aPlain = blank(cH * pairs * 4), bPlain = blank(cH * pairs * 4);
  const aPacked = blank(cH * pairs * 2), bPacked = blank(cH * pairs * 2);
  await dispatch(plain.projectAB, [z, mask, weights, aPlain, bPlain], ...projectGroups);
  await dispatch(packed.projectAB, [z, mask, weights, aPacked, bPacked], ...projectGroups);

  const readF32 = async (buffer) => new Float32Array(
    await dispatch(plain.normalizeInput, [z, weights, blank(pairs * cZ * 4)],
      1, 1, 1, buffer, cH * pairs * 4));
  const readWords = async (buffer) => new Uint32Array(
    await dispatch(plain.normalizeInput, [z, weights, blank(pairs * cZ * 4)],
      1, 1, 1, buffer, cH * pairs * 2));

  const compare = (reference, actual) => {
    let error = 0, scale = 0, worst = 0, worstAt = -1;
    for (let i = 0; i < reference.length; i += 1) {
      const d = actual[i] - reference[i];
      error += d * d; scale += reference[i] ** 2;
      if (Math.abs(d) > worst) { worst = Math.abs(d); worstAt = i; }
    }
    return {
      relRms: Number(Math.sqrt(error / scale).toExponential(2)),
      worst: Number(worst.toPrecision(4)),
      worstAt,
      worstChannel: worstAt < 0 ? -1 : Math.floor(worstAt / pairs),
      worstRow: worstAt < 0 ? -1 : worstAt % pairs,
    };
  };

  const aRef = await readF32(aPlain);
  const bRef = await readF32(bPlain);
  // 🔴 UNPACK IN THE LAYOUT THE KERNEL WROTE, NOT THE GENERIC ONE. a and b are
  // paired by CHANNEL, not by linear index: word (h / 2) * PAIRS + row holds
  // channels h and h + 1 of that row. Reading them as if element i lived in
  // word i >> 1 is a permutation, and reports relRMS 1.4 for a kernel that may
  // be perfectly right - which is exactly what it did the first time.
  const inChannelPairs = (words) => {
    const halves = unpackHalfWords(words, words.length * 2);
    const out = new Float32Array(cH * pairs);
    for (let h = 0; h < cH; h += 1) {
      for (let row = 0; row < pairs; row += 1) {
        out[h * pairs + row] = halves[(((h >> 1) * pairs + row) * 2) + (h & 1)];
      }
    }
    return out;
  };
  const aGot = inChannelPairs(await readWords(aPacked));
  const bGot = inChannelPairs(await readWords(bPacked));

  // --- and the contraction that reads them ---
  const contractPlain = blank(cH * pairs * 4);
  const contractPacked = blank(cH * pairs * 4);
  await dispatch(plain.contract, [aPlain, bPlain, contractPlain],
    contractGroups[0], contractGroups[1], cH);
  await dispatch(packed.contract, [aPacked, bPacked, contractPacked],
    contractGroups[0], contractGroups[1], cH);
  const contracted = compare(await readF32(contractPlain), await readF32(contractPacked));

  // --- and now the WHOLE update, as encodePairTrack runs it ---
  // 🔴 THE TWO KERNELS BEING RIGHT DID NOT MAKE THE UPDATE RIGHT, which is the
  // whole reason this arm exists: normalize -> project -> contract ->
  // normalize-hidden -> project-out, on one input, both ways.
  const wholeUpdate = async (set, abPacked) => {
    const s0 = blank(pairs * cZ * 4);
    const s1 = blank(cH * pairs * (abPacked ? 2 : 4));
    const s2 = blank(cH * pairs * (abPacked ? 2 : 4));
    const s3 = blank(cH * pairs * 4);
    const s4 = blank(pairs * cH * 4);
    const out = upload(new Float32Array(pairs * cZ));
    const normGroups = [Math.min(pairs, 32768), Math.ceil(pairs / 32768)];
    await dispatch(set.normalizeInput, [z, weights, s0], ...normGroups);
    await dispatch(set.projectAB, [s0, mask, weights, s1, s2], ...projectGroups);
    await dispatch(set.contract, [s1, s2, s3], contractGroups[0], contractGroups[1], cH);
    await dispatch(set.normalizeHidden, [s3, weights, s4], ...normGroups);
    await dispatch(set.projectOutput, [s0, s4, weights, out], ...projectGroups);
    return new Float32Array(await dispatch(set.normalizeInput, [z, weights, blank(pairs * cZ * 4)],
      1, 1, 1, out, pairs * cZ * 4));
  };
  const updated = compare(await wholeUpdate(plain, false), await wholeUpdate(packed, true));

  return {
    length, pairs, cZ, cH, direction, accumulatePrecision,
    contracted, updated,
    note: "a/b compared element by element in the LOGICAL layout a[h][row]",
    a: compare(aRef, aGot),
    b: compare(bRef, bGot),
    contractTile: plain.contractTile,
    contractGroups,
  };
}
