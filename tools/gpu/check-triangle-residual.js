/**
 * The residual form of the triangle output projection writes base + out.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-triangle-residual.js
 *
 * 🔴 AF2 HAS NO OFFICIAL-VALUE GATE ON THIS MACHINE and this is the gap it
 * covers. `test/fixtures/evoformer/` is gitignored, so
 * tools/gpu/check-evoformer-stack.js cannot run here, and the residual variant
 * of project-output is reached ONLY from AF2's evoformer and multimer blocks -
 * every AF3 checker exercises the plain form. The variant used to be made by
 * string-replacing the finished WGSL, which returns the subject unchanged when
 * the pattern stops matching: the block would then have OVERWRITTEN the pair
 * representation instead of adding to it, at the right shape, with no error.
 *
 * It runs both forms over identical inputs, one into a zeroed buffer and one
 * into a buffer holding a known base, and asserts the second is the first plus
 * that base. The comparison is to a rounding tolerance rather than bitwise: the
 * plain form rounds the product to f32 before the host adds the base in f64,
 * where the residual form adds in f32 on the device, which differs in the last
 * ulp on about a tenth of the cells. It also asserts that no cell was left
 * standing at the base value, which is what a writeback the dispatch skipped
 * would look like.
 */
import { createTriangleShaders } from "../../src/triangle/shaders.js";

const OFFSET_ORDER = [
  ["layerNormInWeight", (c) => c.cZ], ["layerNormInBias", (c) => c.cZ],
  ["linearAPWeight", (c) => c.cZ * c.cH], ["linearAPBias", (c) => c.cH],
  ["linearAGWeight", (c) => c.cZ * c.cH], ["linearAGBias", (c) => c.cH],
  ["linearBPWeight", (c) => c.cZ * c.cH], ["linearBPBias", (c) => c.cH],
  ["linearBGWeight", (c) => c.cZ * c.cH], ["linearBGBias", (c) => c.cH],
  ["layerNormOutWeight", (c) => c.cH], ["layerNormOutBias", (c) => c.cH],
  ["linearZWeight", (c) => c.cH * c.cZ], ["linearZBias", (c) => c.cZ],
  ["linearGWeight", (c) => c.cZ * c.cZ], ["linearGBias", (c) => c.cZ],
];

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  // 🔴 A LENGTH THE TILE DOES NOT DIVIDE, on purpose: 37 * 37 = 1369 pair rows
  // against a 32-row tile leaves a partial tile, which is where a writeback
  // that runs twice or not at all would show.
  const length = Number(option(args, "tokens", "37"));
  const cZ = Number(option(args, "channels", "128"));
  const cH = cZ;
  const pairs = length * length;

  const offsets = {};
  let total = 0;
  for (const [name, size] of OFFSET_ORDER) {
    offsets[name] = total;
    total += size({ cZ, cH });
  }
  let state = 90210;
  const random = (count, scale = 0.2) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      out[i] = (state / 0x7fffffff - 0.5) * scale;
    }
    return out;
  };

  const storage = GPUBufferUsage.STORAGE;
  const upload = (data, usage = storage) => {
    const buffer = device.createBuffer({
      size: data.byteLength, usage, mappedAtCreation: true });
    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();
    return buffer;
  };

  const base = random(pairs * cZ, 2.0);
  const weights = upload(random(total));
  const z = upload(random(pairs * cZ));
  const x = upload(random(pairs * cH));
  const plainOut = upload(new Float32Array(pairs * cZ), storage | GPUBufferUsage.COPY_SRC);
  const residualOut = upload(base, storage | GPUBufferUsage.COPY_SRC);
  const readback = device.createBuffer({
    size: pairs * cZ * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  const shape = { length, cZ, cHidden: cH };
  const plain = createTriangleShaders(shape, "f32", offsets, 1e-5, "outgoing");
  const residual = createTriangleShaders(
    shape, "f32", offsets, 1e-5, "outgoing", "two-pass", plain.projectTile, true);

  const run = async (source, output) => {
    const pipeline = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: source }), entryPoint: "main" },
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [z, x, weights, output].map((buffer, binding) => ({
        binding, resource: { buffer } })),
    }));
    pass.dispatchWorkgroups(Math.ceil(cZ / plain.projectTile.columns),
                            Math.ceil(pairs / plain.projectTile.rows));
    pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, pairs * cZ * 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    return result;
  };

  const plainResult = await run(plain.projectOutput, plainOut);
  const residualResult = await run(residual.projectOutput, residualOut);

  // One ulp of the sum, which is where the two roundings can differ.
  const TOLERANCE = 1e-6;
  let worst = 0;
  let untouched = 0;
  for (let i = 0; i < plainResult.length; i += 1) {
    const expected = base[i] + plainResult[i];
    const scale = Math.max(1e-6, Math.abs(base[i]) + Math.abs(plainResult[i]));
    worst = Math.max(worst, Math.abs(residualResult[i] - expected) / scale);
    // ...a cell the dispatch never reached would still hold the base exactly.
    if (residualResult[i] === base[i] && plainResult[i] !== 0) untouched += 1;
  }
  const ok = worst <= TOLERANCE && untouched === 0;
  console.log(`${ok ? "PASS" : "FAIL"}\tpairs ${pairs}\ttile ${plain.projectTile.rows}`
    + `x${plain.projectTile.columns}\tworst ${worst.toExponential(2)}\tuntouched ${untouched}`);
  if (!ok) throw new Error(`residual projection is off by ${worst.toExponential(2)} relative `
    + `(bound ${TOLERANCE}), with ${untouched} cells left at the base value`);
  return { length, pairs, tile: plain.projectTile, cells: plainResult.length,
           worstRelative: worst, untouched, ok };
}
