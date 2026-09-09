/**
 * The pair transition alone, at several row tiles, interleaved in one process.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-transition.js --tokens=59
 *
 * WHY IT EXISTS. `pair-transition` is the largest single kernel in the trunk -
 * 83 ms of a 480 ms pairformer - and the only knob that has ever moved it is
 * how many rows one workgroup carries, because every workgroup reads the whole
 * weight set. Measuring that through bench-trunk.js means a 40 s run per arm
 * and a 48-block average; this synthesises the weights and runs the one shader,
 * so an arm costs about a second and the tiles alternate inside one process.
 * See the note on run-to-run drift in CLAUDE.md.
 *
 * 🔴 WHAT IT SAYS NOW, AT 200 TOKENS (40,000 ROWS), AGAINST A 10.69 ms ARM AT
 * THE TILE AND CHUNK THE TRUNK ACTUALLY RUNS - which is `8:128`, and which the
 * sweep confirms is still the optimum there (4 gives 14.34, 8:512 14.71,
 * 16:128 11.78, 8:256 11.54):
 *
 *     the first matmul's two weight reads   2.08 ms   19%
 *     the normalised tile, from shared      1.24       12%
 *     the second matmul's weight read       0.63        6%
 *     the gated block, from shared          0.41        4%
 *     ---------------------------------------------------
 *     everything left is arithmetic         6.33       59%
 *
 * So it is NOT bandwidth bound: removing every weight read saves 2.7 ms of
 * 10.69. Storing the weights in f16 is worth 1.08 ms - 10% - and costs relRMS
 * 1.35e-3 against the f32 arm, which is why the trunk does not take it by
 * default; see CLAUDE.md's table on where f16 weights buy time and where they
 * do not.
 */
import {
  createTransitionShader, createTransitionSplitShaders, packTransitionWeights,
} from "../../src/af3/transition-webgpu.js";
import { stagedMatrixStorage } from "../../src/runtime/matrix-linear.js";
import { deviceMatrixConfig } from "../../src/runtime/device-profile.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const tokens = Number(option(args, "tokens", "59"));
  const channels = Number(option(args, "channels", "128"));
  const factor = Number(option(args, "factor", "4"));
  const rounds = Number(option(args, "rounds", "9"));
  const iterations = Number(option(args, "iterations", "16"));
  // Arms are "tile" or "tile:chunk"; chunk defaults to the whole intermediate,
  // which is the shape the kernel had before it was chunked.
  const arms_spec = option(args, "arms", "4,8:512,8:128,8:64,16:64,16:128").split(",");
  const rows = Number(option(args, "rows", String(tokens * tokens)));
  const intermediate = channels * factor;

  const random = (n) => {
    const out = new Float32Array(n);
    let state = 22222;
    for (let i = 0; i < n; i += 1) {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      out[i] = (state / 0x7fffffff - 0.5) * 0.1;
    }
    return out;
  };
  const weights = {
    inputLayerNormScale: new Float32Array(channels).fill(1),
    inputLayerNormOffset: new Float32Array(channels),
    transition1: random(channels * intermediate * 2),
    transition2: random(intermediate * channels),
  };
  const packed = packTransitionWeights(weights);
  // 🔴 THE BUFFER AND THE SHADER MUST AGREE ABOUT THE ELEMENT. The offsets are
  // in ELEMENTS and do not depend on it, so a mismatch is a wrong answer
  // rather than an error - see packTransitionWeights.
  const packedHalf = packTransitionWeights(weights, "f16");
  const input = random(rows * channels);

  const storage = GPUBufferUsage.STORAGE;
  // ...by BYTES, not by floats: a packed f16 bundle is a Uint16Array and
  // writing it through a Float32Array view runs off the end of the mapping.
  const upload = (data, usage) => {
    const size = Math.ceil(data.byteLength / 4) * 4;
    const buffer = device.createBuffer({ size, usage, mappedAtCreation: true });
    new Uint8Array(buffer.getMappedRange()).set(
      new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buffer.unmap();
    return buffer;
  };
  const inputBuffer = upload(input, storage);
  const weightBuffer = upload(packed.data, storage);
  const halfWeightBuffer = upload(packedHalf.data, storage);
  const outputBuffer = device.createBuffer({
    size: rows * channels * 4, usage: storage | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({
    size: rows * channels * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  const arms = [];
  // 🔴 --diagnose ARMS RETURN WRONG NUMBERS ON PURPOSE. Each replaces one class
  // of read in the first matmul's inner loop with a constant, to price that
  // class rather than to compute anything: "x" drops the workgroup-memory read
  // of the normalised tile, "w" drops the two weight reads. Their relRms is
  // expected to be large; they exist only to say which read the loop is waiting
  // on. Never read a fold out of one.
  //
  // 🔴 AND EVERY PATTERN HERE ROTS WITH THE KERNEL. Two of the four - `x` and
  // `g` - had stopped matching, and the arm raises rather than reporting a
  // wrong number, which is the right failure but only if someone runs it. They
  // are written against the loop bodies as they stand; when the shader is
  // rearranged again, expect to rewrite them with it.
  const surgery = {
    // ...tolerant of the widen wrapper, which is `vec4<f32>(...)` for a staged
    // f16 tile at lanes 4 and nothing at all for an f32 one. The replacement
    // has to be the same TYPE as what it replaces, so it is written from the
    // capture rather than as a scalar.
    x: [/let x = ((?:vec\d<f32>|f32)?\()?normalized\[g \* CHANNELS \+ c\]\)?;/,
        (whole, open_) => `let x = ${open_ ?? ""}f32(c) * 1e-6${open_ ? ")" : ""};`],
    w: [/let wg = weights\[column \+ i\];\n\s*let wv = weights\[column \+ INTERMEDIATE \+ i\];/,
        "let wg = f32(i) * 1e-6; let wv = f32(c) * 1e-6;"],
    g: [/((?:vec\d<f16>|f16|f32)?\()?gated\[g \* CHUNK \+ slot\]\)?/,
        (whole, open_) => `${open_ ?? ""}f16(slot)${open_ ? ")" : ""}`],
    t2: [/let w = weights\[W_T2 \+ \(chunk0 \+ slot\) \* CHANNELS \+ c\];/,
         "let w = f32(slot) * 1e-6;"],
  };
  // ...the fused arms; `split` is a different shape of arm and is built below.
  for (const spec of arms_spec.filter((a) => a.split("@")[0] !== "split")) {
    // `8:128@f16` names the tile and chunk, then the element the two staged
    // blocks are held in. The suffix is optional and f32 is what every arm
    // meant before it existed.
    // `8:128@f16` narrows the two STAGED blocks; `8:128@f16+f16` narrows the
    // running sum as well, which is a different register story - see
    // accumulatePrecision in src/af3/transition-webgpu.js.
    // `8:128@f16+f16+f16` names the staged blocks, the running sum, and the
    // element the WEIGHT buffer holds - a third axis, and the one that decides
    // how many bytes each workgroup reads. See createTransitionShader.
    const [armSpec, precisionSpec = "f32"] = spec.split("@");
    const [stagePrecision, accumulatePrecision = "f32", weightPrecision = "f32"] =
      precisionSpec.split("+");
    const [tile, chunk, drop, width, lanes] = armSpec.split(":");
    if ((stagePrecision !== "f32" || accumulatePrecision !== "f32"
         || weightPrecision !== "f32") && !device.features.has("shader-f16")) continue;
    let source = createTransitionShader(
      { rows, channels, factor, tile: Number(tile), chunk: chunk ? Number(chunk) : undefined,
        width: width ? Number(width) : undefined,
        lanes: lanes ? Number(lanes) : undefined, stagePrecision, accumulatePrecision,
        weightPrecision },
      packed.offsets, 1e-5, "fast");
    if (drop) {
      const [pattern, replacement] = surgery[drop];
      if (!pattern.test(source)) throw new Error(`--diagnose arm ${spec} matched nothing`);
      source = source.replace(pattern, replacement);
    }
    const module = device.createShaderModule({ code: source });
    const pipeline = await device.createComputePipelineAsync({
      layout: "auto", compute: { module, entryPoint: "main" } });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [inputBuffer, weightPrecision === "f16" ? halfWeightBuffer : weightBuffer,
                outputBuffer].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    arms.push({ spec, passes: [{ pipeline, bindGroup, x: Math.ceil(rows / Number(tile)), y: 1 }],
                times: [] });
  }

  // 🔴 `split` IS THE SAME TRANSITION AS THREE PASSES, AND IT IS THE ARM THAT
  // ANSWERS WHETHER THE FUSION IS STILL PAYING. See the note at the top of
  // createTransitionSplitShaders: the fused kernel holds the WIDENED row in
  // workgroup memory, so its row tile - which is all of its weight-read
  // amortisation - halves every time the channel count doubles. At AF3's 128 it
  // still wins; at ESMFold2's 256 it does not. `split` takes the buffer
  // precisions as `split@f16` or `split@f32`, which narrows the two
  // intermediates and nothing the units do.
  for (const spec of arms_spec.filter((a) => a.split("@")[0] === "split")) {
    const [, precision = "f16"] = spec.split("@");
    const config = deviceMatrixConfig(device, { element: "f16" });
    if (config === null) continue;
    const split = createTransitionSplitShaders(
      { rows, channels, factor }, packed.offsets, 1e-5, "fast",
      { normalizedStorage: precision, wideStorage: precision,
        matrix: { result: config.resultComponentType, matrixElement: config.componentType,
                  tile: { M: config.M, N: config.N, K: config.K } } });
    const bytes = stagedMatrixStorage({
      ...split.geometry, tile: { M: config.M, N: config.N, K: config.K },
      result: config.resultComponentType });
    if (bytes > device.limits.maxComputeWorkgroupStorageSize) continue;
    const width = precision === "f16" ? 2 : 4;
    const normalized = device.createBuffer({ size: rows * channels * width, usage: storage });
    const wideBuffer = device.createBuffer({
      size: rows * intermediate * 2 * width, usage: storage });
    const params = (inner, columns, offset) => upload(
      new Uint32Array([rows, inner, columns, offset, 0, 0, 0, 0]), GPUBufferUsage.UNIFORM);
    const built = await Promise.all([split.normalize, split.wide, split.down].map(
      (code) => device.createComputePipelineAsync({
        layout: "auto",
        compute: { module: device.createShaderModule({ code }), entryPoint: "main" } })));
    const bind = (pipeline, buffers) => device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    // 🔴 THE THIRD PASS IS RESIDUAL AND THIS BENCH'S OUTPUT IS NOT CLEARED
    // BETWEEN ROUNDS, so its VALUES drift upward over the timing loop. That is
    // deliberate and it is why this arm is excluded from the agreement check
    // below rather than compared against the fused one: the timing is what it
    // is here and the correctness is check-transition-split.js's job.
    arms.push({
      spec, residualDrift: true, times: [],
      passes: [
        { pipeline: built[0],
          bindGroup: bind(built[0], [inputBuffer, weightBuffer, normalized,
                                     upload(new Uint32Array([rows, channels, 0, 0]),
                                            GPUBufferUsage.UNIFORM)]),
          x: Math.ceil(rows / split.tiles.normalizeRows), y: 1 },
        { pipeline: built[1],
          bindGroup: bind(built[1], [normalized, weightBuffer,
                                     params(channels, intermediate * 2, packed.offsets.transition1),
                                     wideBuffer]),
          x: Math.ceil(intermediate * 2 / split.tiles.blockColumns),
          y: Math.ceil(rows / split.tiles.blockRows) },
        { pipeline: built[2],
          bindGroup: bind(built[2], [wideBuffer, weightBuffer,
                                     params(intermediate, channels, packed.offsets.transition2),
                                     outputBuffer]),
          x: Math.ceil(channels / split.tiles.blockColumns),
          y: Math.ceil(rows / split.tiles.blockRows) },
      ],
    });
  }

  const once = async (arm) => {
    const encoder = device.createCommandEncoder();
    for (let i = 0; i < iterations; i += 1) {
      for (const step of arm.passes) {
        const pass = encoder.beginComputePass();
        pass.setPipeline(step.pipeline);
        pass.setBindGroup(0, step.bindGroup);
        pass.dispatchWorkgroups(step.x, step.y);
        pass.end();
      }
    }
    const start = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return (performance.now() - start) / iterations;
  };

  for (const arm of arms) await once(arm);           // warm every pipeline first
  for (let round = 0; round < rounds; round += 1) {
    for (const arm of arms) arm.times.push(await once(arm));
  }

  // ...and check the arms agree, because a tile the dispatch does not match
  // computes a fraction of the rows and looks like a speedup.
  const outputs = [];
  for (const arm of arms) {
    if (arm.residualDrift) { outputs.push(null); continue; }
    const encoder = device.createCommandEncoder();
    for (const step of arm.passes) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(step.pipeline);
      pass.setBindGroup(0, step.bindGroup);
      pass.dispatchWorkgroups(step.x, step.y);
      pass.end();
    }
    encoder.copyBufferToBuffer(outputBuffer, 0, readback, 0, rows * channels * 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    outputs.push(new Float32Array(readback.getMappedRange().slice(0)));
    readback.unmap();
  }
  const reference = outputs.find((out) => out !== null);
  const relRms = outputs.map((out) => {
    if (out === null) return null;
    let error = 0, scale = 0;
    for (let i = 0; i < reference.length; i += 1) {
      error += (out[i] - reference[i]) ** 2;
      scale += reference[i] ** 2;
    }
    return Math.sqrt(error / scale);
  });

  const median = (values) => [...values].sort((a, b) => a - b)[values.length >> 1];
  return {
    rows, channels, intermediate,
    arms: arms.map((arm, index) => ({
      arm: arm.spec,
      workgroups: arm.groups,
      ms: Number(median(arm.times).toFixed(3)),
      range: [Math.min(...arm.times), Math.max(...arm.times)].map((v) => Number(v.toFixed(3))),
      // ...null for the split arm, whose target is residual and therefore
      // drifts across the timing loop; check-transition-split.js is where it
      // is compared, on a cleared target.
      relRmsVsFirst: relRms[index] === null ? null : Number(relRms[index].toExponential(2)),
    })),
  };
}
