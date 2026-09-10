/**
 * Where any tool's first run spends its SHADER COMPILATION, by wrapping another
 * tool rather than by being one.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-compiles.js \
 *       --tool=fold-opendde --target=6mrr --repeat=2
 *
 * 🔴 A FIRST FOLD IS ITS OWN COMPILE QUEUE, AND NOTHING WAS TIMING IT. AF2's
 * was 1133 ms of pipeline span inside a 1163 ms fold, with an average of 1.47
 * pipelines ever in flight, because an encode asks for one at the moment it
 * needs it. This browser compiles 32 of them 5.4x faster together than one at a
 * time, so the SPAN against the SUM is the number that says whether a model has
 * that win left. Every flag after `--tool` is passed through untouched.
 *
 * The wrapped tool's own result is returned under `tool`, so a gate does not
 * stop being a gate when it is measured.
 */
import { blockUploadStats } from "../../src/runtime/quantised-upload.js";
import { residentPackStats } from "../../src/runtime/resident.js";
import { tensorDecodeStats } from "../../src/reference/http-tensor-store.js";
import { pipelineCacheStats } from "../../src/runtime/pipeline-cache.js";
import { shaderSourceStats } from "../../src/runtime/shader-source-cache.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export function instrumentCompiles(device) {
  const intervals = [];
  // 🔴 AND WHETHER THE SAME WGSL IS COMPILED TWICE. One module is made per
  // pipeline, and two cache keys that happen to generate identical source pay
  // for it twice - which nothing measured, because the source memo added in
  // this branch keys on the pipeline key and so cannot see across keys.
  const modules = { count: 0, ms: 0, bytes: 0, sizes: [], hashes: new Map() };
  const sync = { count: 0, ms: 0 };
  const makeModule = device.createShaderModule.bind(device);
  device.createShaderModule = (descriptor) => {
    const at = performance.now();
    const built = makeModule(descriptor);
    modules.count += 1;
    modules.ms += performance.now() - at;
    const code = typeof descriptor.code === "string" ? descriptor.code : "";
    // A cheap content hash; a collision here would only merge two rows of a
    // diagnostic, never change a shader.
    let hash = 2166136261;
    for (let at2 = 0; at2 < code.length; at2 += 1) {
      hash = Math.imul(hash ^ code.charCodeAt(at2), 16777619);
    }
    modules.hashes.set(hash, (modules.hashes.get(hash) ?? 0) + 1);
    modules.sizes.push([descriptor.label ?? "?", code.length]);
    // 🔴 HOW MUCH WGSL THIS FOLD WROTE. Generating a shader's SOURCE is host
    // work on the critical path and no profiler here can see it: it happens in
    // a template literal before `createShaderModule` is called. The byte count
    // is the proxy - if a fold writes tens of megabytes of text, that is where
    // a first fold's unexplained host time is.
    modules.bytes += descriptor.code?.length ?? 0;
    return built;
  };
  const makeSync = device.createComputePipeline.bind(device);
  device.createComputePipeline = (descriptor) => {
    const at = performance.now();
    const built = makeSync(descriptor);
    sync.count += 1;
    sync.ms += performance.now() - at;
    return built;
  };
  const buffers = { count: 0, bytes: 0, ms: 0 };
  const makeBuffer = device.createBuffer.bind(device);
  device.createBuffer = (descriptor) => {
    const at = performance.now();
    const built = makeBuffer(descriptor);
    buffers.count += 1;
    buffers.bytes += descriptor.size ?? 0;
    buffers.ms += performance.now() - at;
    return built;
  };
  const writes = { count: 0, bytes: 0, ms: 0 };
  const write = device.queue.writeBuffer.bind(device.queue);
  device.queue.writeBuffer = (...rest) => {
    const at = performance.now();
    const done = write(...rest);
    writes.count += 1;
    writes.bytes += rest[4] ?? rest[2]?.byteLength ?? 0;
    writes.ms += performance.now() - at;
    return done;
  };
  // 🔴 AND THE HOST-DEVICE SYNCS, because a fold that waits 36 times is a fold
  // whose shape is the waiting. `onSubmittedWorkDone` resolving is the GPU
  // catching up, so this is queue drain time and not CPU.
  const syncs = { count: 0, ms: 0 };
  const done = device.queue.onSubmittedWorkDone.bind(device.queue);
  device.queue.onSubmittedWorkDone = () => {
    const at = performance.now();
    syncs.count += 1;
    return done().then((value) => { syncs.ms += performance.now() - at; return value; });
  };
  // 🔴 AND THE BIND GROUPS, because a trunk is thousands of dispatches and each
  // one builds its own. This is pure host time and it is not in a compute pass.
  const groups = { count: 0, ms: 0 };
  const makeGroup = device.createBindGroup.bind(device);
  device.createBindGroup = (descriptor) => {
    const at = performance.now();
    const built = makeGroup(descriptor);
    groups.count += 1;
    groups.ms += performance.now() - at;
    return built;
  };
  const makeAsync = device.createComputePipelineAsync.bind(device);
  device.createComputePipelineAsync = (descriptor) => {
    const at = performance.now();
    return makeAsync(descriptor).then((pipeline) => {
      intervals.push([at, performance.now(), descriptor.label ?? "?"]);
      return pipeline;
    });
  };
  return { intervals, modules, sync, buffers, writes, syncs, groups };
}

/** Span, sum and peak concurrency over a set of [start, end] intervals. */
export function compileSummary(intervals) {
  const round = (value) => Math.round(value * 10) / 10;
  if (intervals.length === 0) return { pipelines: 0 };
  const marks = [];
  let sum = 0;
  let first = Infinity;
  let last = -Infinity;
  for (const [a, b] of intervals) {
    sum += b - a;
    first = Math.min(first, a);
    last = Math.max(last, b);
    marks.push([a, 1]); marks.push([b, -1]);
  }
  marks.sort((x, y) => x[0] - y[0] || x[1] - y[1]);
  let live = 0; let peak = 0;
  // 🔴 THE UNION, NOT THE SUM. A stage that compiles twenty pipelines at once
  // and a stage that compiles them one after another have the same sum and
  // completely different costs; what a fold actually WAITS for is the time
  // with at least one compile in flight, and only that can be compared to the
  // wall. `sumMs` divided by `busyMs` is how many were in flight while any was.
  let busy = 0; let since = 0;
  for (const [at, delta] of marks) {
    if (live === 0 && delta > 0) since = at;
    live += delta;
    if (live === 0) busy += at - since;
    peak = Math.max(peak, live);
  }
  return {
    pipelines: intervals.length,
    spanMs: round(last - first),
    sumMs: round(sum),
    // 🔴 THE ONE NUMBER TO READ. Below about 2 the compiles are serial and a
    // warm is worth most of the span; near the peak they are already packed.
    busyMs: round(busy),
    meanInFlight: Math.round(sum / (last - first) * 100) / 100,
    meanWhileBusy: Math.round(sum / busy * 100) / 100,
    peakInFlight: peak,
    labelCounts: [...intervals.reduce((into, [, , label]) => into.set(label,
      (into.get(label) ?? 0) + 1), new Map()).entries()].sort(),
    slowest: intervals.map(([a, b, label]) => [label, round(b - a)])
      .sort((x, y) => y[1] - x[1]).slice(0, 10),
    // 🔴 GROUPED BY THE LABEL's FIRST WORD, WHICH IS THE STAGE. A model whose
    // mean concurrency is low because its stages compile one after another
    // shows it here: each group's own span is short and they do not overlap.
    // That is a different fix from a stage that compiles serially inside
    // itself, and the two are indistinguishable in the totals.
    groups: Object.entries(intervals.reduce((into, [a, b, label]) => {
      const name = String(label).split(/[:.]/)[0] || "?";
      const group = into[name] ?? (into[name] = { n: 0, first: a, last: b, sum: 0, marks: [] });
      group.n += 1; group.sum += b - a;
      group.marks.push([a, 1], [b, -1]);
      group.first = Math.min(group.first, a); group.last = Math.max(group.last, b);
      return into;
    }, {})).map(([name, g]) => {
      g.marks.sort((x, y) => x[0] - y[0] || x[1] - y[1]);
      let live = 0; let busyMs = 0; let since = 0;
      for (const [at, delta] of g.marks) {
        if (live === 0 && delta > 0) since = at;
        live += delta;
        if (live === 0) busyMs += at - since;
      }
      return { name, n: g.n, atMs: round(g.first - first), spanMs: round(g.last - g.first),
               sumMs: round(g.sum), busyMs: round(busyMs) };
    })
      .sort((x, y) => x.atMs - y.atMs),
  };
}

export async function main(device, args) {
  const tool = option(args, "tool");
  if (tool === undefined) throw new Error("probe-compiles needs --tool=<module under tools/gpu>");
  const rest = args.filter((a) => !a.startsWith("--tool="));
  const instrument = instrumentCompiles(device);
  const uploadsBefore = { ...blockUploadStats };
  const module = await import(`./${tool}.js`);
  const started = performance.now();
  const result = await module.main(device, rest);
  const wallMs = Math.round(performance.now() - started);
  return {
    tool, wallMs, ...compileSummary(instrument.intervals),
    // What the on-device weight decode cost the host over the same run; see
    // blockUploadStats, and note that none of it is inside a compute pass.
    weightDecode: Object.fromEntries(Object.entries(blockUploadStats)
      .map(([key, value]) => [key, Math.round((value - uploadsBefore[key]) * 10) / 10])),
    buffers: instrument.buffers.count,
    bufferMiB: Math.round(instrument.buffers.bytes / 1048576 * 10) / 10,
    bufferMs: Math.round(instrument.buffers.ms * 10) / 10,
    writeBuffers: instrument.writes.count,
    writeMiB: Math.round(instrument.writes.bytes / 1048576 * 10) / 10,
    writeMs: Math.round(instrument.writes.ms * 10) / 10,
    bindGroups: instrument.groups.count,
    bindGroupMs: Math.round(instrument.groups.ms * 10) / 10,
    queueSyncs: instrument.syncs.count,
    queueSyncMs: Math.round(instrument.syncs.ms * 10) / 10,
    hostPack: { calls: residentPackStats.calls,
                ms: Math.round(residentPackStats.ms),
                mib: Math.round(residentPackStats.bytes / 1048576 * 10) / 10,
                byLabel: [...residentPackStats.byLabel.entries()]
                  .map(([label, row]) => ({ label, calls: row.calls,
                    ms: Math.round(row.ms),
                    mib: Math.round(row.bytes / 1048576 * 10) / 10 }))
                  .sort((a, b) => b.ms - a.ms).slice(0, 10) },
    hostDecode: { calls: tensorDecodeStats.calls,
                  ms: Math.round(tensorDecodeStats.ms),
                  megaElements: Math.round(tensorDecodeStats.elements / 1e6 * 10) / 10 },
    // 🔴 THE SOURCE A CACHE HIT THREW AWAY, which is the half `shaderSourceMiB`
    // cannot see: it counts what reached `createShaderModule`, i.e. the sources
    // that were used. `wastedSourceMiB` is what was generated for a pipeline
    // that already existed.
    pipelineCache: { hits: pipelineCacheStats.hits, misses: pipelineCacheStats.misses,
                     shared: pipelineCacheStats.shared,
                     wastedSourceMiB:
                       Math.round(pipelineCacheStats.hitSourceBytes / 1048576 * 10) / 10,
                     byKey: [...pipelineCacheStats.byKey.entries()]
                       .map(([key, row]) => ({ key, hits: row.hits,
                         mib: Math.round(row.bytes / 1048576 * 10) / 10 }))
                       .sort((a, b) => b.mib - a.mib).slice(0, 40) },
    // What the memo actually generated, and what it cost. `misses` is the
    // number of sources built; `hits` is the number of times a fold asked for
    // one it already had.
    shaderGeneration: { built: shaderSourceStats.misses, reused: shaderSourceStats.hits,
                        ms: Math.round(shaderSourceStats.ms),
                        builtMiB: Math.round(shaderSourceStats.bytes / 1048576 * 100) / 100,
                        reusedMiB: Math.round(shaderSourceStats.hitBytes / 1048576 * 10) / 10 },
    shaderModules: instrument.modules.count,
    // Distinct WGSL texts against modules made: the gap is source compiled twice.
    distinctSources: instrument.modules.hashes.size,
    duplicateModules: instrument.modules.count - instrument.modules.hashes.size,
    largestSources: instrument.modules.sizes
      .sort((a, b) => b[1] - a[1]).slice(0, 8)
      .map(([label, bytes]) => [String(label).slice(0, 58), bytes]),
    shaderModuleMs: Math.round(instrument.modules.ms * 10) / 10,
    shaderSourceMiB: Math.round(instrument.modules.bytes / 1048576 * 100) / 100,
    synchronousPipelines: instrument.sync.count,
    synchronousMs: Math.round(instrument.sync.ms * 10) / 10,
    // Only the scalar fields, so a fold's PDB does not come back through here.
    tool_: Object.fromEntries(Object.entries(result ?? {})
      .filter(([, value]) => typeof value === "number" || typeof value === "boolean")),
  };
}
