/**
 * A REAL bundle's tensors, decoded on the host and on the device, compared bit
 * for bit.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-bundle-device-decode.js
 *     node tools/gpu-chrome.mjs tools/gpu/check-bundle-device-decode.js \
 *       --bundle=/model-af3-int5 --tensors=8
 *
 * 🔴 check-quantised-upload.js BUILDS ITS OWN SHARDS, AND THAT IS THE HALF THIS
 * ONE DOES NOT TEST. It sweeps the codec against a shard it laid out itself, so
 * it checks the kernel and the planner against a layout it also wrote. What it
 * cannot catch is a real bundle whose records say something the planner reads
 * differently - a scaleOffset relative to the wrong base, a shape whose product
 * is not the element count, a tensor that starts mid-group. Those live in the
 * exported bundle, not in the arithmetic.
 *
 * 🔴 AND THE BAR IS ZERO DIFFERING HALVES. Both sides produce f16; the host
 * computes `code * scale + zero` in f64 and narrows, the shader in f32. The
 * claim in src/runtime/quantised-upload.js is that they agree exactly, and
 * anything else is a finding rather than a tolerance.
 */
import { planBlockUpload, runBlockUpload } from "../../src/runtime/quantised-upload.js";
import { readTensor, readTensorAsFloat16 } from "../../src/reference/dtype.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const bundle = option(args, "bundle", "/model-esmc-600m-int3");
  const wanted = Number(option(args, "tensors", "6"));
  const manifest = await (await fetch(`${bundle}/manifest.json`)).json();

  // The biggest quantised tensors, which are the ones a fold's time is in.
  const quantised = Object.entries(manifest.tensors)
    .filter(([, record]) => typeof record.dtype === "string" && record.dtype.startsWith("int")
      && Number.isInteger(record.block))
    .map(([name, record]) => [name, record,
      (record.shape ?? []).reduce((total, extent) => total * extent, 1)])
    .sort((a, b) => b[2] - a[2]);
  if (quantised.length === 0) {
    throw new Error(`${bundle} has no quantised tensors: nothing here to check`);
  }
  // 🔴 THE BIGGEST AND THE SMALLEST, because a tensor whose element count is
  // under one quantisation group is where the span arithmetic has its edge and
  // the big ones will never reach it.
  const picked = [...quantised.slice(0, Math.max(1, wanted - 2)),
                  ...quantised.slice(-2)].slice(0, wanted);

  const shards = new Map();
  const results = [];
  for (const [name, record, elements] of picked) {
    if (!shards.has(record.file)) {
      shards.set(record.file, await (await fetch(`${bundle}/${record.file}`)).arrayBuffer());
    }
    const buffer = shards.get(record.file);
    const byteOffset = record.byteOffset ?? 0;
    // 🔴 THE BITS, NOT THE NUMBERS. `readTensorAsFloat16` hands back a
    // Float16Array whose elements read as numbers; the device hands back the
    // words it wrote. Comparing one against the other says every element
    // differs, which is a checker failing rather than a decoder.
    const half = readTensorAsFloat16(record, buffer, byteOffset);
    const expected = new Uint16Array(half.buffer, half.byteOffset, half.length);

    const thunk = () => { throw new Error("unused"); };
    thunk.store = { tensorSource: () => ({ record, buffer, byteOffset }) };
    thunk.tensorName = name;
    thunk.first = 0;
    thunk.count = elements;
    const planned = planBlockUpload([{ name, thunk, offset: 0, length: elements }]);
    if (planned === undefined || planned.gpu.params.length === 0) {
      throw new Error(`${name} (${record.dtype} at ${record.block}) was refused by the `
        + "planner, so the device decoder cannot take this bundle at all");
    }
    const bytes = Math.ceil(elements / 2) * 4;
    const destination = device.createBuffer({
      label: `decode.${name}`, size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({
      size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    device.pushErrorScope("validation");
    await runBlockUpload(device, planned.gpu, destination);
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(destination, 0, readback, 0, bytes);
    device.queue.submit([encoder.finish()]);
    const error = await device.popErrorScope();
    if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
    await readback.mapAsync(GPUMapMode.READ);
    const actual = new Uint16Array(readback.getMappedRange().slice(0));
    readback.unmap();
    destination.destroy();
    readback.destroy();

    let differing = 0;
    let firstAt = -1;
    for (let index = 0; index < elements; index += 1) {
      if (actual[index] !== expected[index]) {
        differing += 1;
        if (firstAt < 0) firstAt = index;
      }
    }
    results.push({ name, dtype: record.dtype, group: record.block, elements, differing,
                   ...(firstAt < 0 ? {} : { firstAt, host: expected[firstAt],
                                            device: actual[firstAt] }) });
    console.log(`${differing === 0 ? "ok  " : "FAIL"} ${name}\t${record.dtype}@${record.block}`
      + `\t${elements} elements\t${differing} differing`);
  }
  // 🔴 AND THE RESHAPES, WHICH ARE THE HALF THE CONTIGUOUS ARMS CANNOT REACH.
  // The planner takes a stride triple and up to four sources so that a packer's
  // transpose, its interleaved a/b split and the matrix path's four-role
  // interleave can be done in the decode instead of on the host. Each is
  // checked against the host doing the same reshape to the host decode.
  const mapped = [];
  {
    // 🔴 A TENSOR THE RESHAPES CAN ACTUALLY MOVE, chosen rather than inherited.
    // The contiguous arms above deliberately include the SMALLEST quantised
    // tensor, and on AF3 that is a single element - where a transpose is the
    // identity and the control below correctly refused to call it a check.
    // This wants a matrix: both extents above one, and small enough that the
    // host reference loop is not the slowest thing here.
    const matrices = quantised
      .filter(([, record]) => (record.shape ?? []).length >= 2
        && record.shape[0] > 1 && record.shape[1] > 1)
      .filter(([, , elements]) => elements <= 4 * 1024 * 1024);
    if (matrices.length === 0) {
      throw new Error(`${bundle} has no quantised matrix under 4 Mi elements to reshape`);
    }
    const [name, record] = matrices[0];
    const rows = record.shape[0];
    const columns = (record.shape ?? []).reduce((a, b) => a * b, 1) / rows;
    const elements = rows * columns;
    if (!shards.has(record.file)) {
      shards.set(record.file, await (await fetch(`${bundle}/${record.file}`)).arrayBuffer());
    }
    const buffer = shards.get(record.file);
    const byteOffset = record.byteOffset ?? 0;
    const half = readTensorAsFloat16(record, buffer, byteOffset);
    const source = new Uint16Array(half.buffer, half.byteOffset, half.length);
    const store = { tensorSource: () => ({ record, buffer, byteOffset }) };
    const part = { store, tensorName: name, first: 0, count: elements, bias: 0 };

    // A transpose: destination [c][r] reads source [r][c].
    const transposed = new Uint16Array(elements);
    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < columns; c += 1) transposed[c * rows + r] = source[r * columns + c];
    }
    // Two roles interleaved, the shape the matrix triangle projection wants:
    // destination [i][role] reads source element i of role's own run.
    const halves = Math.floor(elements / 2);
    const woven = new Uint16Array(halves * 2);
    for (let i = 0; i < halves; i += 1) {
      woven[i * 2] = source[i];
      woven[i * 2 + 1] = source[halves + i];
    }
    // Two matrices concatenated along their COLUMNS: destination row i is
    // source A's row i followed by source B's row i. `partRun` is what says so.
    const cols = 64;
    const runs = Math.floor(halves / cols);
    const columnwise = new Uint16Array(runs * cols * 2);
    for (let r = 0; r < runs; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        columnwise[r * cols * 2 + c] = source[r * cols + c];
        columnwise[r * cols * 2 + cols + c] = source[halves + r * cols + c];
      }
    }
    for (const [label, expectedBits, entry] of [
      ["transpose", transposed, {
        offset: 0, length: elements, sources: [part],
        inner: rows, innerStride: columns, outerStride: 1 }],
      ["interleave-2", woven, {
        offset: 0, length: halves * 2,
        sources: [part, { ...part, bias: halves }],
        inner: halves, innerStride: 1, outerStride: 0 }],
      ["concatenate-columns", columnwise, {
        offset: 0, length: runs * cols * 2, partRun: cols,
        sources: [part, { ...part, bias: halves }],
        inner: runs * cols, innerStride: 1, outerStride: 0 }],
    ]) {
      const planned = planBlockUpload([entry]);
      if (planned === undefined || planned.gpu.params.length === 0) {
        throw new Error(`the planner refused the ${label} arm`);
      }
      const bytes = Math.ceil(entry.length / 2) * 4;
      const destination = device.createBuffer({
        label: `decode.${label}`, size: bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
      const readback = device.createBuffer({
        size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      device.pushErrorScope("validation");
      await runBlockUpload(device, planned.gpu, destination);
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(destination, 0, readback, 0, bytes);
      device.queue.submit([encoder.finish()]);
      const error = await device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
      await readback.mapAsync(GPUMapMode.READ);
      const actual = new Uint16Array(readback.getMappedRange().slice(0));
      readback.unmap();
      destination.destroy();
      readback.destroy();
      let differing = 0;
      for (let i = 0; i < entry.length; i += 1) {
        if (actual[i] !== expectedBits[i]) differing += 1;
      }
      // 🔴 AND THE CONTROL: a shader that IGNORED the mapping would write the
      // source in source order, so the check only discriminates where the
      // reference and the raw source actually differ. That is not the same as
      // "how much the permutation moves": these are quantised weights with 32
      // distinct codes a group, so a moved element very often lands on a
      // position holding the same bits - on OpenDDE's pair transition, 99.7% of
      // them do. What matters is that ENOUGH positions differ to catch a shader
      // that did nothing, and 7676 of 2,359,296 is enough.
      let sameAsSource = 0;
      for (let i = 0; i < entry.length; i += 1) if (expectedBits[i] === source[i]) sameAsSource += 1;
      const discriminating = entry.length - sameAsSource;
      mapped.push({ label, name, elements: entry.length, differing, discriminating,
                    unmovedFraction: Number((sameAsSource / entry.length).toFixed(4)) });
      console.log(`${differing === 0 ? "ok  " : "FAIL"} ${label}\t${name}`
        + `\t${entry.length} elements\t${differing} differing`
        + `\t${discriminating} positions separate it from doing nothing`);
      if (differing !== 0) {
        throw new Error(`the ${label} mapping decodes ${differing}/${entry.length} `
          + "elements differently from the host doing the same reshape");
      }
      if (discriminating < 1024) {
        throw new Error(`the ${label} arm's reference differs from the raw source at only `
          + `${discriminating} of ${entry.length} positions, so a shader that ignored the `
          + "mapping would very nearly pass it");
      }
    }
  }

  // 🔴 AND THE f32 DESTINATION, WHICH IS A DIFFERENT SHADER AND THE ONLY ONE
  // THAT MAY STRIDE. Grid attention's four projections are interleaved four to
  // a slot and cannot share one source mapping - v is stored the way the kernel
  // wants and q, k and the gate are transposed - so each is its own dispatch
  // writing every fourth word. An f32 element IS a word, so that races nobody.
  const wide = [];
  {
    const [name, record, elements] = (quantised
      .filter(([, r]) => (r.shape ?? []).length >= 2 && r.shape[0] > 1 && r.shape[1] > 1)
      .filter(([, , n]) => n <= 1024 * 1024))[0] ?? quantised[quantised.length - 1];
    if (!shards.has(record.file)) {
      shards.set(record.file, await (await fetch(`${bundle}/${record.file}`)).arrayBuffer());
    }
    const buffer = shards.get(record.file);
    const byteOffset = record.byteOffset ?? 0;
    // 🔴 THE HOST REFERENCE IS THE f32 DECODE, NOT THE f16 ONE NARROWED. The
    // shader writes `code * scale + zero` straight out at f32, and the packers
    // this serves - grid attention's - hold float32. Comparing against a
    // narrowed reference would hold it to the wrong number.
    const expected = readTensor(record, buffer, byteOffset, true);
    const store = { tensorSource: () => ({ record, buffer, byteOffset }) };
    const stride = 4;
    const lane = 3;
    const entry = { name, offset: lane, length: elements, destStride: stride,
                    sources: [{ store, tensorName: name, first: 0, count: elements, bias: 0 }],
                    inner: elements, innerStride: 1, outerStride: 0 };
    const planned = planBlockUpload([entry], "f32");
    if (planned === undefined || planned.gpu.params.length === 0) {
      throw new Error("the planner refused an f32 strided destination");
    }
    const bytes = (elements * stride + lane) * 4;
    const destination = device.createBuffer({
      label: "decode.f32", size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({
      size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    device.pushErrorScope("validation");
    await runBlockUpload(device, planned.gpu, destination);
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(destination, 0, readback, 0, bytes);
    device.queue.submit([encoder.finish()]);
    const error = await device.popErrorScope();
    if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
    await readback.mapAsync(GPUMapMode.READ);
    const actual = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    destination.destroy();
    readback.destroy();
    let differing = 0;
    let spilled = 0;
    for (let i = 0; i < elements; i += 1) {
      if (actual[i * stride + lane] !== expected[i]) differing += 1;
      // 🔴 AND NOTHING OUTSIDE ITS OWN LANE, which is the whole claim a strided
      // write makes. A shader that ignored destStride would fill the front of
      // the buffer and leave the rest zero, and the lane check alone would not
      // say so.
      for (let s = 0; s < stride; s += 1) {
        if (s !== lane && actual[i * stride + s] !== 0) spilled += 1;
      }
    }
    wide.push({ name, elements, stride, lane, differing, spilled });
    console.log(`${differing === 0 && spilled === 0 ? "ok  " : "FAIL"} f32-stride${stride}`
      + `\t${name}\t${elements} elements\t${differing} differing\t${spilled} outside its lane`);
    if (differing !== 0 || spilled !== 0) {
      throw new Error(`the f32 strided destination has ${differing} elements differing from `
        + `the host decode and ${spilled} written outside lane ${lane}`);
    }
  }

  const bad = results.filter((row) => row.differing !== 0);
  if (bad.length > 0) {
    throw new Error(`${bad.length} of ${results.length} tensors decode differently on the `
      + `device: ${bad.map((row) => `${row.name} ${row.differing}/${row.elements}`).join(", ")}`);
  }
  return { bundle, tensors: results.length, results, mapped, wide };
}
