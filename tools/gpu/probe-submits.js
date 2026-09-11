/**
 * How a fold is SHAPED IN TIME on the host side: submits, the work in each of
 * them, and the places the host stops and waits for the device.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-submits.js \
 *       --tool=fold-opendde --target=6mrr --steps=16 --repeat=2
 *
 * 🔴 IT WRAPS ANOTHER TOOL RATHER THAN BEING ONE, for the reason
 * probe-compiles.js does: a gate does not stop being a gate when it is
 * measured. Every flag after `--tool` is passed through untouched.
 *
 * 🔴 AND IT EXISTS BECAUSE `--profile` CANNOT ANSWER THIS. tools/gpu/profile.js
 * sets `batchComputePasses: false` to get one row per label, which is one pass
 * per dispatch - so every pass count, submit count and idle share ever read off
 * a profiled run describes a fold nobody runs. This probe changes no tuning.
 *
 * What to read:
 *
 *  - `submits` and `dispatchesPerSubmit`. A submit is a round trip to the
 *    driver; a fold that issues thousands of them with three dispatches in each
 *    is paying that trip per kernel.
 *  - `stalls`. Every `mapAsync` and `onSubmittedWorkDone` the fold awaited,
 *    with the wall spent in them. This is the host standing still with an idle
 *    queue behind it, and it is invisible to both profilers here.
 *  - `encodeMs` against `wallMs`. Encoding is host work on the critical path -
 *    bind groups, pass descriptors, the WGSL a cache miss generates - and if it
 *    is a third of the wall the device is being starved by its driver.
 */

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const round = (value) => Math.round(value * 10) / 10;

export function instrumentSubmits(device) {
  const started = performance.now();
  // One row per submit: when it happened, and what had been encoded into it.
  const submits = [];
  let passes = 0;
  let dispatches = 0;
  let encodeMs = 0;
  let sincePasses = 0;
  let sinceDispatches = 0;
  let sinceEncoders = 0;
  // 🔴 AND WHICH CODE PATH ISSUED IT. A submit count is not actionable until it
  // names the encoder label behind it: 602 of them is a number, "one per
  // evoformer block per recycle" is a change.
  let sinceLabels = [];
  const byLabel = new Map();
  // 🔴 THE HOST'S OWN GAPS. Wall between one submit RETURNING and the next
  // being called is host work plus any wait, and the two are told apart by
  // `stalls`, which is the wait alone.
  let lastReturn = started;
  let gapMs = 0;
  let submitMs = 0;

  // 🔴 BIND GROUPS ARE THE HOST WORK NOBODY COUNTS. A trunk is tens of
  // thousands of dispatches and each builds a descriptor object and a native
  // object behind it; none of it is in a compute pass, so neither profiler
  // here can see a millisecond of it.
  const groups = { count: 0, ms: 0 };
  const makeGroup = device.createBindGroup.bind(device);
  device.createBindGroup = (descriptor) => {
    const at = performance.now();
    const built = makeGroup(descriptor);
    groups.count += 1;
    groups.ms += performance.now() - at;
    return built;
  };

  const makeEncoder = device.createCommandEncoder.bind(device);
  device.createCommandEncoder = (descriptor) => {
    const at = performance.now();
    const encoder = makeEncoder(descriptor);
    sinceEncoders += 1;
    sinceLabels.push(String(descriptor?.label ?? "?"));
    const begin = encoder.beginComputePass.bind(encoder);
    encoder.beginComputePass = (passDescriptor) => {
      const beganAt = performance.now();
      const pass = begin(passDescriptor);
      passes += 1;
      sincePasses += 1;
      const dispatch = pass.dispatchWorkgroups.bind(pass);
      pass.dispatchWorkgroups = (...rest) => {
        dispatches += 1;
        sinceDispatches += 1;
        return dispatch(...rest);
      };
      encodeMs += performance.now() - beganAt;
      return pass;
    };
    encodeMs += performance.now() - at;
    return encoder;
  };

  const submit = device.queue.submit.bind(device.queue);
  device.queue.submit = (buffers) => {
    const at = performance.now();
    gapMs += at - lastReturn;
    const done = submit(buffers);
    const back = performance.now();
    submitMs += back - at;
    submits.push({
      atMs: round(at - started),
      buffers: buffers?.length ?? 0,
      passes: sincePasses,
      dispatches: sinceDispatches,
      encoders: sinceEncoders,
      gapMs: round(at - lastReturn),
    });
    const name = [...new Set(sinceLabels)].sort().join("+") || "(no encoder)";
    const row = byLabel.get(name) ?? { submits: 0, dispatches: 0, gapMs: 0 };
    row.submits += 1; row.dispatches += sinceDispatches; row.gapMs += at - lastReturn;
    byLabel.set(name, row);
    sincePasses = 0; sinceDispatches = 0; sinceEncoders = 0; sinceLabels = [];
    lastReturn = back;
    return done;
  };

  // 🔴 THE TWO WAYS A HOST WAITS, and neither profiler here can see either.
  // `mapAsync` is a readback and drains the queue; `onSubmittedWorkDone` is the
  // drain with no data. A fold that does one of these per block has serialised
  // itself against the device whatever its kernels cost.
  const stalls = [];
  const workDone = device.queue.onSubmittedWorkDone.bind(device.queue);
  device.queue.onSubmittedWorkDone = () => {
    const at = performance.now();
    return workDone().then((value) => {
      stalls.push({ kind: "workDone", atMs: round(at - started),
                    ms: round(performance.now() - at) });
      return value;
    });
  };
  const makeBuffer = device.createBuffer.bind(device);
  device.createBuffer = (descriptor) => {
    const buffer = makeBuffer(descriptor);
    const map = buffer.mapAsync?.bind(buffer);
    if (map !== undefined) {
      buffer.mapAsync = (...rest) => {
        const at = performance.now();
        return map(...rest).then((value) => {
          stalls.push({ kind: "mapAsync", atMs: round(at - started),
                        bytes: descriptor.size ?? 0,
                        ms: round(performance.now() - at) });
          return value;
        });
      };
    }
    return buffer;
  };

  const summary = (wallMs) => {
    const counts = submits.map((s) => s.dispatches);
    const sorted = [...counts].sort((a, b) => a - b);
    const at = (q) => sorted.length === 0 ? 0 : sorted[Math.min(sorted.length - 1,
      Math.floor(q * sorted.length))];
    const byKind = new Map();
    for (const stall of stalls) {
      const row = byKind.get(stall.kind) ?? { count: 0, ms: 0 };
      row.count += 1; row.ms += stall.ms;
      byKind.set(stall.kind, row);
    }
    // The submits that carried the LEAST work are the ones a batching change
    // would remove, so name how many of them there are rather than only a mean.
    const singles = counts.filter((c) => c <= 1).length;
    return {
      wallMs: Math.round(wallMs),
      submits: submits.length,
      passes, dispatches,
      dispatchesPerSubmit: submits.length === 0 ? 0
        : Math.round(dispatches / submits.length * 100) / 100,
      passesPerSubmit: submits.length === 0 ? 0
        : Math.round(passes / submits.length * 100) / 100,
      submitsWithAtMostOneDispatch: singles,
      dispatchQuartiles: [at(0), at(0.25), at(0.5), at(0.75), sorted[sorted.length - 1] ?? 0],
      // Host time, split three ways. `encodeMs` is building the commands,
      // `submitMs` is the call itself, `gapMs` is everything between a submit
      // returning and the next one starting - which includes the stalls.
      encodeMs: round(encodeMs),
      submitMs: round(submitMs),
      gapMs: round(gapMs),
      bindGroups: groups.count,
      bindGroupMs: round(groups.ms),
      // 🔴 THE GAPS THEMSELVES, LARGEST FIRST. `gapMs` as a total says the host
      // is not keeping ahead of the device; this says WHERE it stopped, and a
      // handful of long gaps is a different bug from ten thousand short ones.
      gapHistogram: (() => {
        const bands = [0.05, 0.2, 1, 5, 20, Infinity];
        const into = bands.map((upTo) => ({ upToMs: upTo, count: 0, ms: 0 }));
        for (const row of submits) {
          const band = into.find((b) => row.gapMs < b.upToMs) ?? into[into.length - 1];
          band.count += 1; band.ms = round(band.ms + row.gapMs);
        }
        return into.filter((b) => b.count > 0);
      })(),
      slowestGaps: [...submits].sort((a, b) => b.gapMs - a.gapMs).slice(0, 10),
      // 🔴 THE ACTIONABLE TABLE. Sorted by submits, because the question this
      // probe exists to answer is which loop is issuing them.
      byEncoder: [...byLabel.entries()]
        .map(([label, row]) => ({ label, submits: row.submits,
          dispatches: row.dispatches, gapMs: round(row.gapMs) }))
        .sort((a, b) => b.submits - a.submits).slice(0, 20),
      stallMs: round([...byKind.values()].reduce((sum, row) => sum + row.ms, 0)),
      stalls: [...byKind.entries()].map(([kind, row]) =>
        ({ kind, count: row.count, ms: round(row.ms) })),
      slowestStalls: [...stalls].sort((a, b) => b.ms - a.ms).slice(0, 8),
      // Submits per 100 ms of wall, over the run: a fold whose rate is flat is
      // uniformly chatty, one with a spike has a stage that is.
      busiest: submits.reduce((into, row) => {
        const bucket = Math.floor(row.atMs / 100) * 100;
        into.set(bucket, (into.get(bucket) ?? 0) + 1);
        return into;
      }, new Map()),
    };
  };
  return { summary };
}

export async function main(device, args) {
  const tool = option(args, "tool");
  if (tool === undefined) throw new Error("probe-submits needs --tool=<module under tools/gpu>");
  const rest = args.filter((a) => !a.startsWith("--tool="));
  const instrument = instrumentSubmits(device);
  const module = await import(`./${tool}.js`);
  const started = performance.now();
  const result = await module.main(device, rest);
  const summary = instrument.summary(performance.now() - started);
  const busiest = [...summary.busiest.entries()]
    .sort((a, b) => b[1] - a[1]).slice(0, 8)
    .map(([atMs, count]) => [atMs, count]);
  return {
    tool, ...summary, busiest,
    tool_: Object.fromEntries(Object.entries(result ?? {})
      .filter(([, value]) => typeof value === "number" || typeof value === "boolean")),
  };
}
