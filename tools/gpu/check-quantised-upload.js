/**
 * Does the SHIPPED GPU dequantiser decode what the host decoder decodes, at
 * every codec a bundle in this repository uses?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-quantised-upload.js
 *
 * 🔴 IT DRIVES `planBlockUpload` AND `runBlockUpload`, NOT A COPY OF THEM.
 * `check-int5-gpu.js` re-implements the decode in its own WGSL and compares
 * that against the host - which checks the ARITHMETIC and not the kernel. Both
 * are worth having and only this one fails when the shipped planner changes:
 * the offsets, the group spans, the dynamic uniform stride and the two-element
 * word are exactly where an off-by-one lives, and none of them exist in a
 * hand-written comparison shader.
 *
 * 🔴 AND IT SWEEPS THE CODEC, WHICH IS WHY IT EXISTS NOW. The decoder was
 * int5-at-32 written into its constants and its shifts; ESM-C ships **int3 at a
 * group of 128** and so had to be decoded on the host - 1121 ms of decode and
 * 723 of narrowing across 36 blocks, against 660 ms of GPU. A codec is legal
 * here when a group is a whole number of bytes and a code spans at most two of
 * them; 32 x 5 and 128 x 3 both are.
 *
 * 🔴 THE BAR IS ZERO DIFFERING ELEMENTS, NOT A relRMS. JavaScript computes
 * `code * scale + zero` in f64 and WGSL has no f64, so this could differ in the
 * last bit and the point of the check is that it does not - see the note in
 * src/runtime/quantised-upload.js. Anything but zero is a finding.
 */
import { planBlockUpload, runBlockUpload } from "../../src/runtime/quantised-upload.js";
import { readTensorRange } from "../../src/reference/dtype.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/**
 * A shard holding one quantised tensor, laid out the way an exported bundle
 * lays one out: codes, then the f16 scale table, then the f16 zero table.
 */
function shard(elements, bits, group, seed) {
  const groups = Math.ceil(elements / group);
  const groupBytes = (group * bits) / 8;
  const codeBytes = groups * groupBytes;
  const codes = new Uint8Array(codeBytes + 1);
  let state = seed >>> 0;
  const next = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state >>> 8;
  };
  const mask = (1 << bits) - 1;
  const wanted = new Uint8Array(elements);
  for (let index = 0; index < elements; index += 1) {
    const code = next() & mask;
    wanted[index] = code;
    const bit = (index % group) * bits + Math.floor(index / group) * groupBytes * 8;
    const at = bit >> 3;
    const pair = codes[at] | (codes[at + 1] << 8);
    codes[at] = (pair | (code << (bit & 7))) & 255;
    codes[at + 1] = ((pair | (code << (bit & 7))) >> 8) & 255;
  }
  // 🔴 SCALES AND ZEROS SPANNING TEN ORDERS OF MAGNITUDE, because the whole
  // question is whether `code * scale + zero` rounds the same in f32 as in f64
  // and a table of ones would never ask it.
  const scales = new Float16Array(groups);
  const zeros = new Float16Array(groups);
  for (let g = 0; g < groups; g += 1) {
    scales[g] = ((next() % 2000) - 1000) / 1000 * 10 ** (((next() % 9) - 4));
    zeros[g] = ((next() % 2000) - 1000) / 1000 * 10 ** (((next() % 9) - 4));
  }
  const scaleAt = Math.ceil(codes.byteLength / 4) * 4;
  const zeroAt = scaleAt + groups * 2;
  const total = zeroAt + groups * 2;
  const buffer = new ArrayBuffer(Math.ceil(total / 4) * 4);
  new Uint8Array(buffer).set(codes, 0);
  new Uint8Array(buffer).set(new Uint8Array(scales.buffer, 0, groups * 2), scaleAt);
  new Uint8Array(buffer).set(new Uint8Array(zeros.buffer, 0, groups * 2), zeroAt);
  const record = { dtype: `int${bits}`, block: group, shape: [elements], byteOffset: 0,
                   scaleOffset: scaleAt, zeroOffset: zeroAt };
  return { buffer, record, wanted };
}

export async function main(device, args) {
  if (typeof Float16Array !== "function") return { skipped: "no Float16Array in this browser" };
  // 🔴 THE DEFAULT LIST IS WHAT BUNDLES ACTUALLY SHIP PLUS ONE THAT DOES NOT.
  // int5 at 32 is AF3's and OpenDDE's, int3 at 128 is ESM-C's; 3:32 and 5:16
  // are legal codecs nothing exports, and they are here because a decoder that
  // only works at the two shipped points is a decoder waiting for the third
  // bundle. int5 at any group but 32 is deliberately absent: the HOST decoder's
  // int5 fast path reads a 20-byte group and now says so rather than decoding a
  // plausible wrong tensor - which is what this checker found, at 104,170 of
  // 131,072 elements differing. See src/reference/dtype.js.
  const codecs = option(args, "codecs", "5:32,3:128,3:32,2:64,4:32,6:64").split(",");
  const elements = Number(option(args, "elements", "131072"));
  const rows = [];
  let failed = 0;

  for (const spec of codecs) {
    const [bits, group] = spec.split(":").map(Number);
    if ((group * bits) % 8 !== 0) { rows.push({ spec, skipped: "not a whole byte a group" }); continue; }
    const { buffer, record } = shard(elements, bits, group, 12345 + bits * 31 + group);
    const store = { tensorSource: () => ({ record, buffer, byteOffset: 0 }) };
    // 🔴 TWO TENSORS, THE SECOND AT AN OFFSET, because a plan is a run of them
    // sharing one code buffer and one uniform - and `firstElement`, `codeBase`,
    // `groupBase` and `destWord` are four different bases that all have to
    // agree. One tensor at offset zero exercises none of them.
    const halves = [
      { thunk: { store, tensorName: "t", first: 0, count: elements / 2 },
        offset: 0, length: elements / 2 },
      { thunk: { store, tensorName: "t", first: elements / 2, count: elements / 2 },
        offset: elements / 2, length: elements / 2 },
    ];
    const plan = planBlockUpload(halves);
    if (plan === undefined || plan.gpu.params.length !== 2) {
      rows.push({ spec, failed: "planBlockUpload refused it" });
      failed += 1;
      continue;
    }
    const words = Math.ceil(elements / 2);
    const destination = device.createBuffer({
      size: words * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    device.pushErrorScope("validation");
    const release = await runBlockUpload(device, plan.gpu, destination);
    const readback = device.createBuffer({
      size: words * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(destination, 0, readback, 0, words * 4);
    device.queue.submit([encoder.finish()]);
    const error = await device.popErrorScope();
    if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
    await readback.mapAsync(GPUMapMode.READ);
    const got = new Float16Array(readback.getMappedRange().slice(0));
    readback.unmap();
    release();

    // The host answer, narrowed the way concatenateAs narrows it.
    const wide = readTensorRange(record, buffer, 0, 0, elements);
    let differing = 0;
    let worst = 0;
    for (let index = 0; index < elements; index += 1) {
      const want = Math.fround(new Float16Array([wide[index]])[0]);
      if (!Object.is(got[index], want)) {
        differing += 1;
        worst = Math.max(worst, Math.abs(got[index] - want));
      }
    }
    if (differing !== 0) failed += 1;
    rows.push({ spec, bits, group, elements, differing, worst, ok: differing === 0 });
    for (const b of [destination, readback]) b.destroy();
  }
  console.log(rows.map((r) => `${r.spec}\t${r.skipped ?? r.failed ?? `${r.differing} differ`}`).join("\n"));
  if (failed > 0) throw new Error(`${failed} of ${rows.length} codecs differ from the host decoder`);
  return { codecs: rows.length, failed, rows };
}
