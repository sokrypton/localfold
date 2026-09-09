/**
 * The triangle projection on the matrix units, against the vector kernel.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-triangle-project-matrix.js
 *     node tools/gpu-chrome.mjs tools/gpu/check-triangle-project-matrix.js --channels=384
 *
 * 🔴 IT IS DIFFERENTIAL AGAINST THE SHIPPED KERNEL, which is the reference that
 * matters: `check-triangle-residual.js` and the AF3 block checkers say the
 * vector `projectAB` computes AF3's projection, and this says the matrix one
 * computes the vector one's answer. Chaining those is what makes the second
 * path checked without a second oracle.
 *
 * 🔴 AND IT CHECKS THE PACK AS WELL AS THE KERNEL. The matrix path reads a
 * TRANSPOSED, INTERLEAVED copy of the four projection matrices - see
 * AB_INTERLEAVED - and an interleave that is off by a role produces a finite,
 * plausible tensor. Both arms are built from the same `weights` object here, so
 * the pack is inside the comparison rather than beside it.
 *
 * 🔴 AND THE CONTRACTION, IN BOTH DIRECTIONS. `outgoing` transposes the RIGHT
 * operand and `incoming` the LEFT, and swapping them returns a finite tensor of
 * exactly the same shape - so one arm each, on the same a and b the vector
 * projection produced.
 *
 * 🔴 IT CHECKS BOTH PROJECTIONS. `projectAB` is one GEMM with interleaved
 * columns; `projectOutput` is TWO, over two different sources, with the gate's
 * result read back at the same output cell. They fail differently and they are
 * different weight layouts, so one arm each.
 *
 * 🔴 THE RAGGED SIZES ARE THE POINT. The matrix blocks are 128 rows by 128
 * columns and the epilogue groups four columns; a pair count divisible by 128
 * checks none of the edges. `--tokens=` defaults to a set that is not.
 */
import { createTriangleShaders } from "../../src/triangle/shaders.js";
import { packWeights } from "../../src/triangle/weights.js";
import {
  createTriangleProjectMatrixShader, createTriangleProjectOutMatrixShaders,
  createTriangleContractMatrixShader, triangleContractMatrixDispatch,
  triangleProjectMatrixDispatch, triangleProjectOutMatrixDispatch,
  TRIANGLE_PROJECT_MATRIX_GEOMETRY, triangleProjectMatrixFits,
} from "../../src/triangle/project-matrix.js";
import { deviceMatrixConfig, deviceTuning } from "../../src/runtime/device-profile.js";
import { stagedMatrixBlock } from "../../src/runtime/matrix-linear.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

function deterministic(count, seed) {
  const out = new Float32Array(count);
  let state = seed >>> 0;
  for (let i = 0; i < count; i += 1) {
    state = (state * 1664525 + 1013904223) >>> 0;
    out[i] = ((state / 4294967296) - 0.5) * 0.2;
  }
  return out;
}

function relativeRms(actual, expected) {
  let error = 0;
  let scale = 0;
  for (let i = 0; i < expected.length; i += 1) {
    const d = actual[i] - expected[i];
    error += d * d;
    scale += expected[i] * expected[i];
  }
  return Math.sqrt(error / Math.max(scale, 1e-30));
}

export async function main(device, args) {
  const cZ = Number(option(args, "channels", "256"));
  const cHidden = Number(option(args, "hidden", String(cZ)));
  const tokens = option(args, "tokens", "40,37,16").split(",").map(Number);
  const bound = Number(option(args, "bound", "8e-3"));
  // 🔴 THE MATRIX ARM'S WEIGHT BUFFER, AND ITS SECOND INDEXING PATH.
  // `stagedMatrixDirectWeights` reads the right operand out of the weight
  // buffer instead of staging it, which needs that buffer to already hold
  // halves - so the two are one arm here. The VECTOR reference keeps its f32
  // weights either way, so this arm's residue includes the weights' own
  // rounding; `--bound=` is what admits it.
  const weightPrecision = option(args, "weights", "f32");
  const direct = option(args, "direct", "0") === "1";
  // The staged GEMM's accumulator width. The device config's own answer is
  // f32 here and f16 halves the registers it costs, which is the occupancy
  // this kernel is bound by - so it is a speed knob whose price is exactly
  // this number.
  const resultType = option(args, "result", "");
  const prefetch = option(args, "prefetch", "0") === "1";

  const config = deviceMatrixConfig(device, { element: "f16" });
  if (config === null) return { skipped: "no f16 subgroup matrix configuration" };
  // 🔴 THE BLOCK IS THE SHIPPED ONE UNLESS ASKED OTHERWISE. A checker that runs
  // a geometry nothing ships is a checker of a kernel nobody runs; `--block=`
  // is for checking one before it becomes the default.
  const matrix = { result: config.resultComponentType, matrixElement: config.componentType,
                   tile: { M: config.M, N: config.N, K: config.K },
                   ...stagedMatrixBlock(option(args, "block", null)
                     ?? deviceTuning(device).stagedMatrixBlock) };
  if (direct) matrix.directWeights = true;
  if (prefetch) matrix.prefetch = true;
  // 🔴 THE CONTRACTION KEEPS THE DEVICE'S OWN RESULT TYPE, which is the rule
  // the trunk follows and so is the rule this has to check. Its K is the
  // protein's length where the three projections contract a channel count, and
  // narrowing it takes a fold to 2178 NaN coordinates - so an arm that narrowed
  // all four would be checking a configuration nothing may ship.
  if (resultType !== "") matrix.result = resultType;
  const contractResult = option(args, "contract-result", config.resultComponentType);
  const contractMatrix = { ...matrix, result: contractResult };
  const geometry = { ...TRIANGLE_PROJECT_MATRIX_GEOMETRY, ...matrix };
  if (!triangleProjectMatrixFits(geometry, device.limits.maxComputeWorkgroupStorageSize)) {
    return { skipped: "the geometry does not fit this device's workgroup storage" };
  }

  const mk = (n, seed) => deterministic(n, seed);
  const weights = {
    layerNormInWeight: new Float32Array(cZ).fill(1),
    layerNormInBias: new Float32Array(cZ),
    linearAPWeight: mk(cHidden * cZ, 11), linearAPBias: mk(cHidden, 12),
    linearAGWeight: mk(cHidden * cZ, 13), linearAGBias: mk(cHidden, 14),
    linearBPWeight: mk(cHidden * cZ, 15), linearBPBias: mk(cHidden, 16),
    linearBGWeight: mk(cHidden * cZ, 17), linearBGBias: mk(cHidden, 18),
    layerNormOutWeight: new Float32Array(cHidden).fill(1),
    layerNormOutBias: new Float32Array(cHidden),
    linearZWeight: mk(cZ * cHidden, 19), linearZBias: mk(cZ, 20),
    linearGWeight: mk(cZ * cZ, 21), linearGBias: mk(cZ, 22),
  };
  const blocked = packWeights(weights, "f32");
  const interleaved = packWeights(weights, weightPrecision,
    { abLayout: "interleaved", zgLayout: "transposed", cHidden, cZ });

  const storage = GPUBufferUsage.STORAGE;
  const upload = (data, usage = storage) => {
    const buffer = device.createBuffer({
      size: Math.ceil(data.byteLength / 4) * 4, usage, mappedAtCreation: true });
    new Uint8Array(buffer.getMappedRange()).set(
      new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buffer.unmap();
    return buffer;
  };
  const blockedWeights = upload(blocked.data);
  const interleavedWeights = upload(interleaved.data);

  const rows = [];
  let failed = 0;
  for (const n of tokens) {
    const pairs = n * n;
    const z = upload(deterministic(pairs * cZ, 991 + n));
    // Ragged, so a kernel that drops the mask fails here rather than in a fold.
    const maskValues = new Float32Array(pairs);
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        maskValues[i * n + j] = (i < Math.ceil(n * 0.75) && j < Math.ceil(n * 0.75)) ? 1 : 0;
      }
    }
    const mask = upload(maskValues);
    const bytes = pairs * cHidden * 4;
    const out = () => device.createBuffer({
      size: bytes, usage: storage | GPUBufferUsage.COPY_SRC });
    const [aVector, bVector, aMatrix, bMatrix] = [out(), out(), out(), out()];
    const readback = device.createBuffer({
      size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    // 🔴 THE RESIDUAL FORM, which is what the pair track runs: `projectOutput`
    // ADDS into its target. An arm built without it agrees with one built with
    // it whenever the target starts at zero, so this asks for it and seeds the
    // target with noise.
    const { projectTile, projectGridWidth, ...sources } = createTriangleShaders(
      { length: n, cZ, cHidden }, "f32", blocked.offsets, 1e-5, "outgoing", "two-pass",
      undefined, true);
    const matrixSource = createTriangleProjectMatrixShader(
      { cZ, cHidden }, { weight: weightPrecision }, matrix);
    const [vectorPipe, matrixPipe, outVectorPipe] = await Promise.all(
      [sources.projectAB, matrixSource, sources.projectOutput].map(
        (code) => device.createComputePipelineAsync({
          layout: "auto",
          compute: { module: device.createShaderModule({ code }), entryPoint: "main" } })));
    const readbackOut = device.createBuffer({
      size: pairs * Math.max(cZ, cHidden) * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    const parameters = upload(
      new Uint32Array([pairs, cZ, 4 * cHidden, interleaved.offsets.linearABWeight,
                       interleaved.offsets.linearABBias, 0, 0, 0]),
      GPUBufferUsage.UNIFORM);

    device.pushErrorScope("validation");
    const encoder = device.createCommandEncoder();
    const pass = (pipeline, entries, x, y, z_ = 1) => {
      const p = encoder.beginComputePass();
      p.setPipeline(pipeline);
      p.setBindGroup(0, device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: entries.flatMap(([binding, buffer]) =>
          (buffer === null ? [] : [{ binding, resource: { buffer } }])),
      }));
      p.dispatchWorkgroups(x, y, z_);
      p.end();
    };
    const rowTiles = Math.ceil(pairs / projectTile.rows);
    pass(vectorPipe, [[0, z], [1, mask], [2, blockedWeights], [3, aVector], [4, bVector]],
         Math.ceil(cHidden / projectTile.columns),
         Math.min(rowTiles, projectGridWidth), Math.ceil(rowTiles / projectGridWidth));
    const dispatch = triangleProjectMatrixDispatch({ rows: pairs, cHidden }, matrix);
    pass(matrixPipe,
         [[0, z], [1, interleavedWeights], [2, parameters], [3, aMatrix],
          [4, mask], [5, bMatrix]],
         dispatch.x, dispatch.y);
    device.queue.submit([encoder.finish()]);
    const error = await device.popErrorScope();
    if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);

    const read = async (buffer) => {
      const copy = device.createCommandEncoder();
      copy.copyBufferToBuffer(buffer, 0, readback, 0, bytes);
      device.queue.submit([copy.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const values = new Float32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      return values;
    };
    const aRelRms = relativeRms(await read(aMatrix), await read(aVector));
    const bRelRms = relativeRms(await read(bMatrix), await read(bVector));

    // 🔴 THE OUTPUT PROJECTION, WHICH TAKES TWO SOURCES AND IS RESIDUAL. Both
    // arms start from the same non-zero target, because a residual that is
    // written rather than added agrees with one that is added whenever the
    // target is zero - which is the kind of arm that passes while being wrong.
    const outBytes = pairs * cZ * 4;
    const seedValues = deterministic(pairs * cZ, 4242);
    const outVector = upload(seedValues, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const outMatrix = upload(seedValues, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const hidden = upload(deterministic(pairs * cHidden, 777 + n));
    const gateScratch = device.createBuffer({ size: pairs * cZ * 4, usage: storage });
    const outSources = createTriangleProjectOutMatrixShaders(
      { cZ, cHidden }, { weight: weightPrecision }, matrix);
    const [gatePipe, projectPipe] = await Promise.all(
      [outSources.gate, outSources.project].map((code) => device.createComputePipelineAsync({
        layout: "auto",
        compute: { module: device.createShaderModule({ code }), entryPoint: "main" } })));
    const gateParams = upload(new Uint32Array(
      [pairs, cZ, cZ, interleaved.offsets.linearGWeight, interleaved.offsets.linearGBias, 0, 0, 0]),
      GPUBufferUsage.UNIFORM);
    const projectParams = upload(new Uint32Array(
      [pairs, cHidden, cZ, interleaved.offsets.linearZWeight, interleaved.offsets.linearZBias,
       0, 0, 0]), GPUBufferUsage.UNIFORM);

    device.pushErrorScope("validation");
    const second = device.createCommandEncoder();
    const outPass = (pipeline, entries, x, y, z_ = 1) => {
      const p = second.beginComputePass();
      p.setPipeline(pipeline);
      p.setBindGroup(0, device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: entries.map(([binding, buffer]) => ({ binding, resource: { buffer } })),
      }));
      p.dispatchWorkgroups(x, y, z_);
      p.end();
    };
    // The vector kernel: z, x, weights, output - residual, in place.
    {
      const p = second.beginComputePass();
      p.setPipeline(outVectorPipe);
      p.setBindGroup(0, device.createBindGroup({
        layout: outVectorPipe.getBindGroupLayout(0),
        entries: [z, hidden, blockedWeights, outVector].map(
          (buffer, binding) => ({ binding, resource: { buffer } })),
      }));
      p.dispatchWorkgroups(Math.ceil(cZ / projectTile.columns),
                           Math.min(rowTiles, projectGridWidth),
                           Math.ceil(rowTiles / projectGridWidth));
      p.end();
    }
    const outDispatch = triangleProjectOutMatrixDispatch({ rows: pairs, cZ }, matrix);
    outPass(gatePipe, [[0, z], [1, interleavedWeights], [2, gateParams], [3, gateScratch]],
            outDispatch.x, outDispatch.y);
    outPass(projectPipe,
            [[0, hidden], [1, interleavedWeights], [2, projectParams], [3, outMatrix],
             [4, gateScratch]], outDispatch.x, outDispatch.y);
    device.queue.submit([second.finish()]);
    const outError = await device.popErrorScope();
    if (outError !== null) throw new Error(`WebGPU validation failed: ${outError.message}`);

    const readOut = async (buffer) => {
      const copy = device.createCommandEncoder();
      copy.copyBufferToBuffer(buffer, 0, readbackOut, 0, outBytes);
      device.queue.submit([copy.finish()]);
      await readbackOut.mapAsync(GPUMapMode.READ);
      const values = new Float32Array(readbackOut.getMappedRange().slice(0));
      readbackOut.unmap();
      return values;
    };
    const outRelRms = relativeRms(await readOut(outMatrix), await readOut(outVector));

    // 🔴 THE CONTRACTION, ON THE a AND b THE VECTOR PROJECTION MADE. Feeding
    // the matrix projection's own outputs would fold two kernels' error into
    // one number and hide which moved.
    const contractBytes = pairs * cHidden * 4;
    const contract = {};
    for (const direction of ["outgoing", "incoming"]) {
      const vectorOut = device.createBuffer({
        size: contractBytes, usage: storage | GPUBufferUsage.COPY_SRC });
      const matrixOut = device.createBuffer({
        size: contractBytes, usage: storage | GPUBufferUsage.COPY_SRC });
      const { contractTile, ...directed } = createTriangleShaders(
        { length: n, cZ, cHidden }, "f32", blocked.offsets, 1e-5, direction, "two-pass");
      const source = createTriangleContractMatrixShader(
        { length: n, channels: cHidden }, direction, {}, contractMatrix);
      const [vecPipe, matPipe] = await Promise.all(
        [directed.contract, source].map((code) => device.createComputePipelineAsync({
          layout: "auto",
          compute: { module: device.createShaderModule({ code }), entryPoint: "main" } })));
      const contractParams = upload(
        new Uint32Array([n, n, n, 0, 0, 0, 0, 0]), GPUBufferUsage.UNIFORM);
      device.pushErrorScope("validation");
      const third = device.createCommandEncoder();
      const record = (pipeline, buffers, x, y, z_) => {
        const p = third.beginComputePass();
        p.setPipeline(pipeline);
        p.setBindGroup(0, device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
        }));
        p.dispatchWorkgroups(x, y, z_);
        p.end();
      };
      record(vecPipe, [aVector, bVector, vectorOut],
             Math.ceil(n / contractTile.columns), Math.ceil(n / contractTile.rows), cHidden);
      // ...a is the LEFT operand outgoing and the RIGHT one incoming, which is
      // the whole difference between the two directions at this binding.
      const [left, right] = direction === "outgoing" ? [aVector, bVector] : [bVector, aVector];
      const dispatch = triangleContractMatrixDispatch({ length: n, channels: cHidden }, matrix);
      record(matPipe, [left, right, contractParams, matrixOut],
             dispatch.x, dispatch.y, dispatch.z);
      device.queue.submit([third.finish()]);
      const contractError = await device.popErrorScope();
      if (contractError !== null) {
        throw new Error(`WebGPU validation failed: ${contractError.message}`);
      }
      const readContract = async (buffer) => {
        const copy = device.createCommandEncoder();
        copy.copyBufferToBuffer(buffer, 0, readbackOut, 0, contractBytes);
        device.queue.submit([copy.finish()]);
        await readbackOut.mapAsync(GPUMapMode.READ);
        const values = new Float32Array(readbackOut.getMappedRange().slice(0));
        readbackOut.unmap();
        return values;
      };
      contract[direction] = relativeRms(
        await readContract(matrixOut), await readContract(vectorOut));
      for (const buffer of [vectorOut, matrixOut, contractParams]) buffer.destroy();
    }

    const ok = aRelRms <= bound && bRelRms <= bound && outRelRms <= bound
      && Object.values(contract).every((value) => value <= bound);
    if (!ok) failed += 1;
    rows.push({ tokens: n, pairs, a: aRelRms.toExponential(2), b: bRelRms.toExponential(2),
                out: outRelRms.toExponential(2),
                contractOutgoing: contract.outgoing.toExponential(2),
                contractIncoming: contract.incoming.toExponential(2), bound, ok });
    console.log(`n=${n}\tpairs=${pairs}\ta ${aRelRms.toExponential(2)}\t`
      + `b ${bRelRms.toExponential(2)}\tout ${outRelRms.toExponential(2)}`
      + `\tcontract ${contract.outgoing.toExponential(2)}/${contract.incoming.toExponential(2)}`
      + `\t${ok ? "ok" : "FAIL"}`);
    for (const buffer of [z, mask, aVector, bVector, aMatrix, bMatrix, readback, parameters,
                          outVector, outMatrix, hidden, gateScratch, gateParams, projectParams,
                          readbackOut]) {
      buffer.destroy();
    }
  }

  if (failed > 0) throw new Error(`${failed} triangle projection shape(s) outside tolerance`);
  return { cZ, cHidden, bound, rows };
}
