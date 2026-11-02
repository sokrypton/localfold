/**
 * grid.attend's two vector forms raced on THIS device, once, before a pair track compiles.
 *
 * 🔴 ONE PRIOR SERVES EVERY APPLE GPU AND THE TILED FORM SPLITS THEM. WebGPU reports an M2 and an M5 alike as
 * `apple / metal-3` - adapter.info's device and description empty, the same limits, no matrix configurations - and
 * the register-tiled attend in f16 (`gridAttendTiled` + `gridAttendTiledHalf`) is grid.attend 668 -> 491 ms a
 * 255-token AF3 trunk pass on the M5 and 995 -> 1444 on the M2. So the prior leaves both knobs unset and this asks the
 * device: the trunk's own attend kernel, both forms, at 256 tokens on synthetic q, k and v, alternated. The tiled form
 * is taken only where it wins by RACE_MARGIN and agrees with the untiled one to the f16 bound.
 *
 * 🔴 AND THE GRID PROJECTION'S ARITHMETIC RIDES THE SAME RACE (`gridProjectArithmetic`), because it splits the same
 * two parts the same way: f16 is 1.5x on the M5 and 0.8x on the M2. The trunk's projection, f32 against f16, at 256
 * tokens on the same normalised pair; f16 only where it wins by RACE_MARGIN and agrees to 1e-2 (it reads ~9e-4). On
 * the M5: 4.52 against 2.97 ms; a 255-token trunk pass 3045 / 3201 -> 2996 / 3096 ms. Where the race fails the
 * projection stays f32. 🔴 Profiled runs (`--profile`) wrap the device's passes in timestamps and the race reads them
 * wrong (f32 6.95 against f16 7.45 there) - name the arm with `--tune` when profiling.
 *
 * ...and the triangle contraction's (`triangleContractArithmetic`, webgpu/kernels/triangle/shaders.js's f16 kernel),
 * which was measured on the M5 alone: 0.69 against 0.51 ms at 256 tokens and 32 channels there, relRMS 6.5e-4. The
 * races take ~0.4 s on the M5, once a browser (their arms compiled at once, their noise shared and periodic).
 *
 * 🔴 AWAITED BY THE COMPILE, NOT THE DEVICE. The knob is read when a stack builds its pipelines, so a race still
 * running then would leave it unset for the fold that most wants it; a stack's compile is seconds of driver work, and
 * this is two pipelines and a few dispatches beside it. One race a device - the promise is shared - and only where
 * nothing else has answered: a prior, `--tune` or setDeviceTuning that names `gridAttendTiled` skips it.
 *
 * 🔴 AND ONE RACE A BROWSER, NOT A PAGE LOAD. The answer is the machine's, and on an M2 - where it is always "untiled"
 * - the race cost a first fold 0.2-0.35 s of every visit (4.06 / 4.23 / 4.25 s against 4.05 / 3.93 / 3.90 skipping
 * it, alternated). So it is remembered in localStorage under the browser's user agent (its version: a Chrome update
 * races again) and the adapter's vendor and architecture. A private window, blocked storage or a worker races as
 * before.
 */
import { createGridAttentionShaders, packGridAttentionWeights } from "./grid-attention.js";
import { createTriangleShaders } from "../../kernels/triangle/shaders.js";
import { createAttentionRegisterFlashShader } from "../../kernels/attention.js";
import { deviceProfile, halfPrecisionAvailable, setDeviceTuning } from "../../runtime/device-profile.js";

const RACES = new WeakMap();
const MEMORY = "localfold.gridAttendRace.v4";
const DETAIL = new WeakMap();
const N = 256;
const CHANNELS = 128;
const HEADS = 4;
const DIMENSION = 32;
// the tiled form must be this much faster: a pass is ~2% noise on either part, and a tie keeps the shipped kernel
const RACE_MARGIN = 0.95;

/** What the race saw, for a log or a bench to print; null where none ran. */
export const gridAttendRaceDetail = (device) => DETAIL.get(device) ?? null;

/** Whether this device should race at all: an Apple GPU with f16 that no prior or caller has answered for. */
function wantsRace(device) {
  const profile = deviceProfile(device);
  return profile.vendor === "apple" && !profile.software && profile.tuning.gridAttendTiled === null
    && profile.tuning.gridAttendTiledHalf === null && halfPrecisionAvailable(device);
}

/** The grid projection's arithmetic as the race answered it, unless a prior, `--tune` or a caller already named one. */
function projectAnswer(device, half) {
  return deviceProfile(device).tuning.gridProjectArithmetic === null ? { gridProjectArithmetic: half ? "f16" : "f32" } : {};
}
/** ...and AF2's flash attention's keys a softmax rescale (`attentionGroup`), where the prior leaves it null. */
function groupAnswer(device, group) {
  return deviceProfile(device).tuning.attentionGroup === null ? { attentionGroup: group } : {};
}
/** ...and the triangle contraction's, the same way. */
function contractAnswer(device, half) {
  return deviceProfile(device).tuning.triangleContractArithmetic === null
    ? { triangleContractArithmetic: half ? "f16" : "f32" } : {};
}

/** Which browser and GPU a remembered answer belongs to. */
function machine(device) {
  const profile = deviceProfile(device);
  const agent = typeof navigator === "object" ? navigator.userAgent : "";
  return `${agent}|${profile.vendor ?? ""}/${profile.architecture ?? ""}`;
}

function remembered(device) {
  try {
    const stored = JSON.parse(globalThis.localStorage?.getItem(MEMORY) ?? "null");
    return stored !== null && stored.machine === machine(device) && typeof stored.tiled === "boolean"
      && typeof stored.projectHalf === "boolean" && typeof stored.contractHalf === "boolean"
      && typeof stored.group === "number" ? stored : undefined;
  } catch { return undefined; }
}

function remember(device, tiled, projectHalf, contractHalf, group) {
  try {
    globalThis.localStorage?.setItem(MEMORY,
      JSON.stringify({ machine: machine(device), tiled, projectHalf, contractHalf, group }));
  }
  catch { /* a race next visit, as before */ }
}

/** Forget this browser's remembered answer, so the next device races again (a probe that wants the race itself). */
export function forgetGridAttendRace() {
  try { globalThis.localStorage?.removeItem(MEMORY); } catch { /* nothing remembered */ }
}

/** Settles `gridAttendTiled` / `gridAttendTiledHalf` for this device; resolves once they are final. */
export function settleGridAttendTiling(device) {
  const running = RACES.get(device);
  if (running !== undefined) return running;
  if (!wantsRace(device)) return Promise.resolve();
  const known = remembered(device);
  if (known !== undefined) {
    DETAIL.set(device, { tiled: known.tiled, projectHalf: known.projectHalf, contractHalf: known.contractHalf,
                         group: known.group, remembered: true });
    setDeviceTuning(device, { gridAttendTiled: known.tiled, gridAttendTiledHalf: known.tiled,
                              ...projectAnswer(device, known.projectHalf), ...contractAnswer(device, known.contractHalf),
                              ...groupAnswer(device, known.group) });
    RACES.set(device, Promise.resolve());
    return RACES.get(device);
  }
  const promise = race(device).catch((error) => {
    DETAIL.set(device, { error: String(error?.message ?? error) });
    // ...and a race that failed leaves the projection where every Apple part is safe: f32 (the M2 loses with f16)
    setDeviceTuning(device, { ...projectAnswer(device, false), ...contractAnswer(device, false), ...groupAnswer(device, 1) });
  });
  RACES.set(device, promise);
  return promise;
}

// (a period of 2^20 floats, repeated: the races time kernels and compare two arms on the same input, neither of which
// asks for 8 M independent draws - drawing them all in JavaScript was most of a race's 0.5 s)
function noise(count, seed, scale) {
  const out = new Float32Array(count);
  const period = Math.min(count, 1 << 20);
  let state = seed >>> 0;
  for (let i = 0; i < period; i += 1) {
    state = (state * 1664525 + 1013904223) >>> 0;
    out[i] = (state / 4294967296 - 0.5) * scale;
  }
  for (let at = period; at < count; at += period) out.copyWithin(at, 0, Math.min(period, count - at));
  return out;
}

/** Two arms' GPU time, alternated in one submit on the GPU's clock where the device has timestamps; [arm 0, arm 1]. */
async function timeArms(device, arms, buffers, run) {
  // 🔴 IN ONE SUBMIT, ON THE GPU'S OWN CLOCK, where the device has timestamps: timed submit by
  // submit on an M5 the same dispatch read 4.3 and 44 ms - the GPU's clock falls between submits - so an arm's
  // time said more about when it ran than what it ran. Six passes an arm, alternated, behind a warm-up, the median.
  const ROUNDS = 6;
  let medians;
  if (device.features.has("timestamp-query")) {
    const querySet = device.createQuerySet({ type: "timestamp", count: 4 * ROUNDS });
    const resolved = device.createBuffer({ size: 32 * ROUNDS, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: 32 * ROUNDS, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    buffers.push(resolved, readback);
    const encoder = device.createCommandEncoder({ label: "grid-attend-race" });
    const pass = (arm, slot) => {
      const compute = encoder.beginComputePass(slot === undefined ? {} : {
        timestampWrites: { querySet, beginningOfPassWriteIndex: slot, endOfPassWriteIndex: slot + 1 } });
      compute.setPipeline(arm.pipeline);
      compute.setBindGroup(0, arm.bind);
      compute.dispatchWorkgroups(...arm.groups);
      compute.end();
    };
    for (let i = 0; i < 4; i += 1) pass(arms[i % 2]);
    for (let round = 0; round < ROUNDS; round += 1) {
      const order = round % 2 === 0 ? [0, 1] : [1, 0];
      for (const which of order) pass(arms[which], 4 * round + 2 * which);
    }
    encoder.resolveQuerySet(querySet, 0, 4 * ROUNDS, resolved, 0);
    encoder.copyBufferToBuffer(resolved, 0, readback, 0, 32 * ROUNDS);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const stamps = new BigInt64Array(readback.getMappedRange().slice(0));
    readback.unmap();
    querySet.destroy();
    for (let round = 0; round < ROUNDS; round += 1) {
      for (const which of [0, 1]) {
        const at = 4 * round + 2 * which;
        arms[which].times.push(Number(stamps[at + 1] - stamps[at]) / 1e6);
      }
    }
    const median = (values) => [...values].sort((x, y) => x - y)[values.length >> 1];
    medians = arms.map((arm) => median(arm.times));
  } else {
    // no timestamps: wall clock, three dispatches a submit, the best of each
    for (let round = 0; round < 4; round += 1) {
      for (const arm of (round % 2 === 0 ? arms : [...arms].reverse())) arm.times.push(await run(arm, 3) / 3);
    }
    medians = arms.map((arm) => Math.min(...arm.times));
  }
  return medians;
}

/** grid.project's two arms compiled (f32, then f16 arithmetic), and the packed offsets their weights take. */
function prepareProject(device) {
  const width = HEADS * DIMENSION;
  const zeros = (count) => new Float32Array(count);
  const packed = packGridAttentionWeights({
    heads: HEADS, dimension: DIMENSION,
    actNormScale: zeros(CHANNELS), actNormOffset: zeros(CHANNELS), pairBiasProjection: zeros(CHANNELS * HEADS),
    qProjection: zeros(CHANNELS * width), kProjection: zeros(CHANNELS * width), vProjection: zeros(CHANNELS * width),
    gatingQuery: zeros(CHANNELS * width), outputProjection: zeros(width * CHANNELS),
  });
  return Promise.all([false, true].map(async (half) => {
    const { project, tiles } = createGridAttentionShaders(
      { n: N, channels: CHANNELS, heads: HEADS, dimension: DIMENSION, transpose: false, residual: true,
        stagedPrecision: "f16", ...(half ? { projectArithmetic: "f16" } : {}),
        maxComputeWorkgroupStorageSize: device.limits.maxComputeWorkgroupStorageSize },
      packed.offsets, 1e-5, "fast", { swapTransposedBias: false });
    const pipeline = await device.createComputePipelineAsync({
      label: `grid-project-race.${half ? "f16" : "f32"}`, layout: "auto",
      compute: { module: device.createShaderModule({ code: project }), entryPoint: "main" },
    });
    return { half, pipeline, weightsLength: packed.data.length, groups: [Math.ceil(N * N / tiles.projectRows)], times: [] };
  }));
}

/** One race's two arms, run once each and compared over a sample of `output`, then timed: [arm 0 ms, arm 1 ms, relRMS]. */
async function measure(device, arms, buffers, label, outputOf) {
  const run = async (arm, iterations) => {
    const encoder = device.createCommandEncoder({ label });
    const pass = encoder.beginComputePass({ label });
    pass.setPipeline(arm.pipeline);
    pass.setBindGroup(0, arm.bind);
    for (let i = 0; i < iterations; i += 1) pass.dispatchWorkgroups(...arm.groups);
    pass.end();
    const started = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return performance.now() - started;
  };
  const SAMPLE = 1 << 16;
  const read = async (arm) => {
    await run(arm, 1);
    const readback = device.createBuffer({ size: SAMPLE * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    buffers.push(readback);
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(outputOf(arm), 0, readback, 0, SAMPLE * 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    return values;
  };
  const [reference, candidate] = [await read(arms[0]), await read(arms[1])];
  let error = 0; let scale = 0;
  for (let i = 0; i < SAMPLE; i += 1) {
    const d = candidate[i] - reference[i];
    error += d * d; scale += reference[i] * reference[i];
  }
  const relRms = Math.sqrt(error / Math.max(scale, 1e-30));
  const [first, second] = await timeArms(device, arms, buffers, run);
  return [first, second, relRms];
}

/** A buffer of `count` floats on the device, for the races' outputs. */
function outputBuffer(device, buffers, count) {
  const buffer = device.createBuffer({ size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  buffers.push(buffer);
  return buffer;
}

/** grid.project in f32 against f16 arithmetic at N tokens: { f32Ms, f16Ms, relRms, half } (half: f16 won and agrees). */
async function raceProject(device, buffers, prepared, shared) {
  const width = HEADS * DIMENSION;
  const arms = await prepared;
  // weights of a trained layer's scale over a normalised pair, so the outputs are a few units as a trunk's are
  const weights = shared.upload(noise(arms[0].weightsLength, 5, 0.2));
  for (const arm of arms) {
    arm.outputs = [0, 1, 2, 3].map(() => outputBuffer(device, buffers, N * N * width));
    arm.bind = device.createBindGroup({
      layout: arm.pipeline.getBindGroupLayout(0),
      entries: [shared.noise[0], weights, ...arm.outputs].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
  }
  const [f32Ms, f16Ms, relRms] = await measure(device, arms, buffers, "grid-project-race", (arm) => arm.outputs[0]);
  // (f16 sums 32 products in half: ~1e-3 of the f32 answer, which the f16 attend bound already admits)
  const half = Number.isFinite(relRms) && relRms < 1e-2 && f16Ms < f32Ms * RACE_MARGIN;
  return { f32Ms: Number(f32Ms.toFixed(3)), f16Ms: Number(f16Ms.toFixed(3)), relRms: Number(relRms.toExponential(2)), half };
}

const CHANNELS_RACED = 32;
/** tri.contract's two arms compiled (f32, then f16 arithmetic), 32 channels. */
function prepareContract(device) {
  return Promise.all([false, true].map(async (half) => {
    const { contract, contractTile } = createTriangleShaders(
      { length: N, cZ: 128, cHidden: CHANNELS_RACED }, "f32", {}, 1e-5, "outgoing", "fast", undefined, true,
      half ? { rows: 32, columns: 32, arithmetic: "f16" } : undefined);
    const pipeline = await device.createComputePipelineAsync({
      label: `tri-contract-race.${half ? "f16" : "f32"}`, layout: "auto",
      compute: { module: device.createShaderModule({ code: contract }), entryPoint: "main" },
    });
    return { half, pipeline, times: [],
             groups: [Math.ceil(N / contractTile.columns), Math.ceil(N / contractTile.rows), CHANNELS_RACED] };
  }));
}

/** tri.contract in f32 against f16 arithmetic at N tokens, 32 channels: { f32Ms, f16Ms, relRms, half }. */
async function raceContract(device, buffers, prepared, shared) {
  const arms = await prepared;
  const bytes = CHANNELS_RACED * N * N * 4;
  for (const arm of arms) {
    arm.output = outputBuffer(device, buffers, CHANNELS_RACED * N * N);
    arm.bind = device.createBindGroup({
      layout: arm.pipeline.getBindGroupLayout(0),
      entries: [{ buffer: shared.noise[0], size: bytes }, { buffer: shared.noise[1], size: bytes }, { buffer: arm.output }]
        .map((resource, binding) => ({ binding, resource })),
    });
  }
  const [f32Ms, f16Ms, relRms] = await measure(device, arms, buffers, "tri-contract-race", (arm) => arm.output);
  const half = Number.isFinite(relRms) && relRms < 1e-2 && f16Ms < f32Ms * RACE_MARGIN;
  return { f32Ms: Number(f32Ms.toFixed(3)), f16Ms: Number(f16Ms.toFixed(3)), relRms: Number(relRms.toExponential(2)), half };
}

// the MSA row attention of a 255-residue chain over 128 rows, eight heads of 32 (bench-msa-attention.js --shape=row)
const GROUP_SHAPE = { batch: 128, queries: 255, channels: 256, heads: 8, headDim: 32 };
/** AF2's flash attention's two arms compiled (the shipped f16-staged kernel, one key a rescale, then four). */
function prepareGroup(device) {
  const { batch, queries, heads, headDim } = GROUP_SHAPE;
  return Promise.all([1, 4].map(async (group) => {
    const pipeline = await device.createComputePipelineAsync({
      label: `attention-group-race.g${group}`, layout: "auto",
      compute: { module: device.createShaderModule({
        code: createAttentionRegisterFlashShader(headDim, undefined, { group, precision: "chunk16" }) }), entryPoint: "main" },
    });
    return { group, pipeline, times: [], groups: [Math.ceil(queries / 64), batch, heads] };
  }));
}

/** AF2's flash attention, one key a rescale against four: { g1Ms, g4Ms, relRms, group }. */
async function raceGroup(device, buffers, prepared, shared) {
  const { batch, queries, channels, heads, headDim } = GROUP_SHAPE;
  const arms = await prepared;
  const bytes = batch * queries * channels * 4;
  const maskValues = new Float32Array(batch * queries);
  for (let i = 0; i < maskValues.length; i += 1) maskValues[i] = i % 17 === 0 ? 0 : 1;
  const mask = shared.upload(maskValues);
  const pairBias = shared.upload(new Float32Array(4));
  const parameters = shared.upload(new Uint32Array([batch, queries, channels, heads, headDim, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, channels]), GPUBufferUsage.UNIFORM);
  // q, k, v and the gate over the shared noise (the gate reuses q's: only the inputs' scale matters here)
  const [q, k, v] = shared.noise.map((buffer) => ({ buffer, size: bytes }));
  for (const arm of arms) {
    arm.output = outputBuffer(device, buffers, batch * queries * channels);
    arm.bind = device.createBindGroup({
      layout: arm.pipeline.getBindGroupLayout(0),
      entries: [q, k, v, q, { buffer: mask }, { buffer: pairBias }, { buffer: parameters }, { buffer: arm.output }]
        .map((resource, binding) => ({ binding, resource })),
    });
  }
  const [g1Ms, g4Ms, relRms] = await measure(device, arms, buffers, "attention-group-race", (arm) => arm.output);
  // (a grouped rescale reassociates the online softmax: ~1e-7, the same class as the key chunk's)
  const group = Number.isFinite(relRms) && relRms < 1e-4 && g4Ms < g1Ms * RACE_MARGIN ? 4 : 1;
  return { g1Ms: Number(g1Ms.toFixed(3)), g4Ms: Number(g4Ms.toFixed(3)), relRms: Number(relRms.toExponential(2)), group };
}

async function race(device) {
  const width = HEADS * DIMENSION;
  const pairs = N * N;
  const zeros = (count) => new Float32Array(count);
  const { offsets } = packGridAttentionWeights({
    heads: HEADS, dimension: DIMENSION,
    actNormScale: zeros(CHANNELS), actNormOffset: zeros(CHANNELS), pairBiasProjection: zeros(CHANNELS * HEADS),
    qProjection: zeros(CHANNELS * width), kProjection: zeros(CHANNELS * width), vProjection: zeros(CHANNELS * width),
    gatingQuery: zeros(CHANNELS * width), outputProjection: zeros(width * CHANNELS),
  });
  // the trunk's attend: f32 scratch, f16 staged, residual, unchunked - see pairformer-block.js and pair-track.js
  const shaders = (tiled) => createGridAttentionShaders(
    { n: N, channels: CHANNELS, heads: HEADS, dimension: DIMENSION, transpose: false, residual: true,
      stagedPrecision: "f16", attendTiled: tiled, attendTiledHalf: tiled,
      maxComputeWorkgroupStorageSize: device.limits.maxComputeWorkgroupStorageSize },
    offsets, 1e-5, "fast", { swapTransposedBias: false });
  // 🔴 EVERY ARM'S PIPELINE COMPILED AT ONCE, and the four races share one set of noise: compiled race by race and each
  // drawing its own ~30 M floats in JavaScript, the four took 0.75 s on the M5
  const wantsGroup = deviceProfile(device).tuning.attentionGroup === null;
  const preparing = { project: prepareProject(device), contract: prepareContract(device),
                      group: wantsGroup ? prepareGroup(device) : null };
  const arms = await Promise.all([false, true].map(async (tiled) => {
    const { attend, tiles } = shaders(tiled);
    const pipeline = await device.createComputePipelineAsync({
      label: `grid-attend-race.${tiled ? "tiled" : "untiled"}`, layout: "auto",
      compute: { module: device.createShaderModule({ code: attend }), entryPoint: "main" },
    });
    return { tiled, pipeline, groups: [Math.ceil(N / tiles.attendRows), N, HEADS], times: [] };
  }));

  const buffers = [];
  const upload = (values, extra = 0) => {
    const usage = extra === GPUBufferUsage.UNIFORM ? extra : GPUBufferUsage.STORAGE | extra;
    const buffer = device.createBuffer({ size: values.byteLength, usage, mappedAtCreation: true });
    new values.constructor(buffer.getMappedRange()).set(values);
    buffer.unmap();
    buffers.push(buffer);
    return buffer;
  };
  try {
    // logits of a few units, as a trunk's are; a ragged mask so the masked path runs
    const mask = new Float32Array(pairs);
    const live = Math.ceil(N * 0.8);
    for (let i = 0; i < live; i += 1) for (let j = 0; j < live; j += 1) mask[i * N + j] = 1;
    const inputs = [upload(noise(pairs * width, 1, 2)), upload(noise(pairs * width, 2, 2)),
                    upload(noise(pairs * width, 3, 1)), upload(noise(HEADS * pairs, 4, 2)), upload(mask)];
    const SAMPLE = 1 << 16;
    for (const arm of arms) {
      arm.output = upload(zeros(pairs * width), GPUBufferUsage.COPY_SRC);
      arm.bind = device.createBindGroup({
        layout: arm.pipeline.getBindGroupLayout(0),
        entries: [...inputs, arm.output].map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
    }
    const run = async (arm, iterations) => {
      const encoder = device.createCommandEncoder({ label: "grid-attend-race" });
      const pass = encoder.beginComputePass({ label: "grid-attend-race" });
      pass.setPipeline(arm.pipeline);
      pass.setBindGroup(0, arm.bind);
      for (let i = 0; i < iterations; i += 1) pass.dispatchWorkgroups(...arm.groups);
      pass.end();
      const started = performance.now();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      return performance.now() - started;
    };
    // the outputs, compared after ONE dispatch each (the residual store accumulates)
    for (const arm of arms) await run(arm, 1);
    const read = async (arm) => {
      const readback = device.createBuffer({ size: SAMPLE * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      buffers.push(readback);
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(arm.output, 0, readback, 0, SAMPLE * 4);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const values = new Float32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      return values;
    };
    const [reference, candidate] = [await read(arms[0]), await read(arms[1])];
    let error = 0; let scale = 0;
    for (let i = 0; i < SAMPLE; i += 1) {
      const d = candidate[i] - reference[i];
      error += d * d; scale += reference[i] * reference[i];
    }
    const relRms = Math.sqrt(error / Math.max(scale, 1e-30));
    const [untiled, tiled] = await timeArms(device, arms, buffers, run);
    const agrees = Number.isFinite(relRms) && relRms < 2e-3;
    const take = agrees && tiled < untiled * RACE_MARGIN;
    // 🔴 AND THE GRID PROJECTION'S ARITHMETIC, WHICH SPLITS THE SAME TWO PARTS THE SAME WAY: f16 is grid.project 475 /
    // 500 -> 328 / 344 ms a 255-token pass on the M5 and 707 / 719 -> 870 / 898 on the M2 (the device-profile note).
    // The trunk's projection (f16 staging, f32 q/k/v/gate), both arithmetics, on the same normalised pair and weights.
    const shared = { noise: inputs.slice(0, 3), upload };
    const project = await raceProject(device, buffers, preparing.project, shared);
    // ...and the triangle contraction's (webgpu/kernels/triangle/shaders.js's f16 kernel), the same question again
    const contract = await raceContract(device, buffers, preparing.contract, shared);
    // ...and AF2's flash attention's rescale group, only where the prior left it to the device
    const group = wantsGroup ? await raceGroup(device, buffers, preparing.group, shared) : null;
    DETAIL.set(device, { tokens: N, untiledMs: Number(untiled.toFixed(3)), tiledHalfMs: Number(tiled.toFixed(3)),
                         relRms: Number(relRms.toExponential(2)), tiled: take,
                         samplesMs: arms.map((arm) => arm.times.map((t) => Number(t.toFixed(2)))), project, contract,
                         group });
    // 🔴 BOTH WAYS, so the knob is answered and no second stack races again
    // (each answer only where nothing named it while the race ran - a race now starts with the page's device)
    const unnamed = deviceProfile(device).tuning.gridAttendTiled === null && deviceProfile(device).tuning.gridAttendTiledHalf === null;
    setDeviceTuning(device, { ...(unnamed ? { gridAttendTiled: take, gridAttendTiledHalf: take } : {}),
                              ...projectAnswer(device, project.half),
                              ...contractAnswer(device, contract.half), ...groupAnswer(device, group?.group ?? 1) });
    remember(device, take, project.half, contract.half, group?.group ?? 1);
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
}

