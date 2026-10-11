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
import { deviceProfile, halfPrecisionAvailable, setDeviceTuning } from "../../runtime/device-profile.js";

const RACES = new WeakMap();
const MEMORY = "localfold.gridAttendRace.v1";
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
      ? stored.tiled : undefined;
  } catch { return undefined; }
}

function remember(device, tiled) {
  try { globalThis.localStorage?.setItem(MEMORY, JSON.stringify({ machine: machine(device), tiled })); }
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
    DETAIL.set(device, { tiled: known, remembered: true });
    setDeviceTuning(device, { gridAttendTiled: known, gridAttendTiledHalf: known });
    RACES.set(device, Promise.resolve());
    return RACES.get(device);
  }
  const promise = race(device).catch((error) => {
    DETAIL.set(device, { error: String(error?.message ?? error) });
  });
  RACES.set(device, promise);
  return promise;
}

function noise(count, seed, scale) {
  const out = new Float32Array(count);
  let state = seed >>> 0;
  for (let i = 0; i < count; i += 1) {
    state = (state * 1664525 + 1013904223) >>> 0;
    out[i] = (state / 4294967296 - 0.5) * scale;
  }
  return out;
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
    const buffer = device.createBuffer({ size: values.byteLength, usage: GPUBufferUsage.STORAGE | extra,
                                         mappedAtCreation: true });
    new Float32Array(buffer.getMappedRange()).set(values);
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
    // ...then the race. 🔴 IN ONE SUBMIT, ON THE GPU'S OWN CLOCK, where the device has timestamps: timed submit by
    // submit on an M5 the same dispatch read 4.3 and 44 ms - the GPU's clock falls between submits - so an arm's
    // time said more about when it ran than what it ran. Six passes an arm, alternated, behind a warm-up, the median.
    const ROUNDS = 6;
    let untiled; let tiled;
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
      [untiled, tiled] = arms.map((arm) => median(arm.times));
    } else {
      // no timestamps: wall clock, three dispatches a submit, the best of each
      for (let round = 0; round < 4; round += 1) {
        for (const arm of (round % 2 === 0 ? arms : [...arms].reverse())) arm.times.push(await run(arm, 3) / 3);
      }
      [untiled, tiled] = arms.map((arm) => Math.min(...arm.times));
    }
    const agrees = Number.isFinite(relRms) && relRms < 2e-3;
    const take = agrees && tiled < untiled * RACE_MARGIN;
    DETAIL.set(device, { tokens: N, untiledMs: Number(untiled.toFixed(3)), tiledHalfMs: Number(tiled.toFixed(3)),
                         relRms: Number(relRms.toExponential(2)), tiled: take,
                         samplesMs: arms.map((arm) => arm.times.map((t) => Number(t.toFixed(2)))) });
    // 🔴 BOTH WAYS, so the knob is answered and no second stack races again
    setDeviceTuning(device, { gridAttendTiled: take, gridAttendTiledHalf: take });
    remember(device, take);
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
}

