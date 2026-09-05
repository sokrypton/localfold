/**
 * What do subgroupMatrixLoad and subgroupMatrixStore actually mean?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-subgroup-matrix.js
 *
 * 🔴 THE TYPE PARAMETERS ARE <T, COLUMNS, ROWS>, NOT <T, ROWS, COLUMNS>, and at
 * 8x8x8 - the only shape this device offers - that difference is invisible in
 * the declaration and visible only in the answer. This multiplies one 8x8 pair
 * whose true product is known on the host and reports which interpretation the
 * hardware agreed with, so a bench is not measuring a transpose.
 */
const N = 8;

export async function main(device) {
  if (!device.features.has("chromium-experimental-subgroup-matrix")) {
    return { skipped: "no chromium-experimental-subgroup-matrix" };
  }
  // Deliberately asymmetric, so a transpose cannot agree by accident.
  const a = new Float32Array(N * N);
  const b = new Float32Array(N * N);
  for (let i = 0; i < N * N; i += 1) {
    a[i] = ((i * 7) % 13) - 6 + (i % 3) * 0.25;
    b[i] = ((i * 5) % 11) - 5 - (i % 4) * 0.5;
  }
  const at = new Float32Array(N * N);
  const bt = new Float32Array(N * N);
  for (let r = 0; r < N; r += 1) {
    for (let c = 0; c < N; c += 1) { at[c * N + r] = a[r * N + c]; bt[c * N + r] = b[r * N + c]; }
  }
  const matmul = (left, right) => {
    const out = new Float32Array(N * N);
    for (let r = 0; r < N; r += 1) {
      for (let c = 0; c < N; c += 1) {
        let sum = 0;
        for (let k = 0; k < N; k += 1) sum += left[r * N + k] * right[k * N + c];
        out[r * N + c] = sum;
      }
    }
    return out;
  };

  const upload = (data, usage = GPUBufferUsage.STORAGE) => {
    const buffer = device.createBuffer({ size: data.byteLength, usage, mappedAtCreation: true });
    new data.constructor(buffer.getMappedRange()).set(data);
    buffer.unmap();
    return buffer;
  };
  // 🔴 THE SOURCE IS ALLOCATED SHORT ON PURPOSE for the second arm. The tiled
  // kernel is exact when M is a multiple of the region and wrong on a ragged
  // edge, and the question that separates "my indexing is wrong" from "an
  // out-of-bounds matrix load is not clamped the way a scalar one is" is
  // whether the IN-BOUNDS rows of a partially out-of-bounds tile survive.
  const source = upload(a);
  const shortSource = upload(a.slice(0, (N - 3) * N));
  const weights = upload(b);
  const output = device.createBuffer({
    size: N * N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    size: N * N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const shader = `enable chromium_experimental_subgroup_matrix;
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
var<workgroup> staged: array<f32, ${N * N}>;
@compute @workgroup_size(32)
fn main(@builtin(local_invocation_index) lane: u32) {
  let left = subgroupMatrixLoad<subgroup_matrix_left<f32, ${N}, ${N}>>(&source, 0u, false, ${N}u);
  let right = subgroupMatrixLoad<subgroup_matrix_right<f32, ${N}, ${N}>>(&weights, 0u, false, ${N}u);
  var acc = subgroup_matrix_result<f32, ${N}, ${N}>();
  acc = subgroupMatrixMultiplyAccumulate(left, right, acc);
  subgroupMatrixStore(&staged, 0u, acc, false, ${N}u);
  workgroupBarrier();
  for (var i = lane; i < ${N * N}u; i += 32u) { output[i] = staged[i]; }
}`;
  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: shader }), entryPoint: "main" },
  });
  const runWith = async (buffer) => {
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [buffer, weights, output].map((b, binding) => ({ binding, resource: { buffer: b } })),
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    const staging = device.createBuffer({
      size: N * N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    encoder.copyBufferToBuffer(output, 0, staging, 0, N * N * 4);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const copy = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    return copy;
  };
  const got = await runWith(source);
  const shortGot = await runWith(shortSource);

  const transpose = (m) => {
    const out = new Float32Array(N * N);
    for (let r = 0; r < N; r += 1) for (let c = 0; c < N; c += 1) out[c * N + r] = m[r * N + c];
    return out;
  };
  const rel = (x, y) => {
    let num = 0; let den = 0;
    for (let i = 0; i < x.length; i += 1) { num += (x[i] - y[i]) ** 2; den += y[i] ** 2; }
    return Math.sqrt(num / Math.max(den, 1e-30));
  };
  // Every interpretation the two ambiguities can produce.
  const candidates = {
    "A@B": matmul(a, b),
    "(A@B)T": transpose(matmul(a, b)),
    "AT@B": matmul(at, b),
    "A@BT": matmul(a, bt),
    "AT@BT": matmul(at, bt),
    "(AT@BT)T": transpose(matmul(at, bt)),
    "B@A": matmul(b, a),
  };
  const scores = Object.fromEntries(
    Object.entries(candidates).map(([name, value]) => [name, Number(rel(got, value).toPrecision(3))]),
  );
  const best = Object.entries(scores).sort((x, y) => x[1] - y[1])[0];
  // Rows 0 to N-4 of the short arm are backed by real memory; rows N-3 up are
  // not. If only the latter differ, an out-of-bounds matrix load damages only
  // the rows that were out of bounds and a bounds check in the epilogue is
  // enough. If the whole tile moves, it is not, and a ragged edge needs the
  // read itself kept in range.
  const truth = candidates["A@B"];
  const rowRel = [];
  for (let r = 0; r < N; r += 1) {
    rowRel.push(Number(rel(shortGot.slice(r * N, r * N + N), truth.slice(r * N, r * N + N))
      .toPrecision(3)));
  }
  return {
    matched: best[0],
    relRms: best[1],
    scores,
    shortSourceRows: N - 3,
    shortRowRelRms: rowRel,
    inBoundsRowsSurvive: rowRel.slice(0, N - 3).every((v) => v === 0),
  };
}
