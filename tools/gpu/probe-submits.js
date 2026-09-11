/**
 * How a fold is SHAPED IN TIME on the host side: submits, the work in each of
 * them, and the places the host stops and waits for the device.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-submits.js \
 *       --tool=fold-opendde --target=6mrr --steps=16 --repeat=2
 *
 * 🔴 AND ONE RUN OF IT IS NOT A MEASUREMENT. This box drifts by up to 3.2x, and
 * a host-side table drifts with it: the same 300-token AF3 fold put 760 ms on
 * `af3-confidence-embed` in one run and 47 in the next, which is most of a
 * morning if the first number is believed. `--repeats=<n>` runs the wrapped
 * tool n times in one process and reports the MEDIAN of each row, which is the
 * only form in which these numbers mean anything.
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

import { GpuBufferAllocator } from "../../src/runtime/allocator.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const round = (value) => Math.round(value * 10) / 10;

/**
 * 🔴 EVERY LABELLED UPLOAD, BY LABEL. `writeBuffer` carries no label, so a
 * device-level wrapper can say a fold moved 776 MiB and never say which tensor
 * - and the expensive ones here are not the weights but the host arrays built
 * to be uploaded, including the all-zero ones a first recycle hands in for the
 * previous pair. This wraps the allocator instead, which has the name.
 */
export function instrumentUploads() {
  const byLabel = new Map();
  const upload = GpuBufferAllocator.prototype.upload;
  GpuBufferAllocator.prototype.upload = function wrapped(label, data, usage) {
    const at = performance.now();
    const result = upload.call(this, label, data, usage);
    const row = byLabel.get(label) ?? { calls: 0, bytes: 0, ms: 0 };
    row.calls += 1;
    row.bytes += data.byteLength ?? 0;
    row.ms += performance.now() - at;
    byLabel.set(label, row);
    return result;
  };
  return {
    restore() { GpuBufferAllocator.prototype.upload = upload; },
    report: () => [...byLabel.entries()]
      .map(([label, row]) => ({ label, calls: row.calls,
        mib: Math.round(row.bytes / 1048576 * 100) / 100,
        ms: Math.round(row.ms * 10) / 10 }))
      .sort((a, b) => b.ms - a.ms).slice(0, 16),
  };
}

export function instrumentSubmits(device) {
  const started = performance.now();
  // 🔴 EVERY WRAPPER IS UNDONE BY `restore`, because `--repeats` instruments
  // the same device again for each run: without it the second round wraps the
  // first round's wrappers and the counts double.
  const original = {
    createCommandEncoder: device.createCommandEncoder,
    createBuffer: device.createBuffer,
    createBindGroup: device.createBindGroup,
    popErrorScope: device.popErrorScope,
    submit: device.queue.submit,
    onSubmittedWorkDone: device.queue.onSubmittedWorkDone,
  };
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
  // 🔴 THE FIRST SUBMIT'S GAP IS NOT A GAP. Everything before it - the weight
  // shards over the network, the page's own start-up - would otherwise be
  // charged to whichever encoder happened to submit first, and on an OpenDDE
  // fold that is 1562 ms landing on `af3-atom-encoder`, whose own stage
  // measures 78. Recorded on its own as `startupMs` instead.
  let lastReturn = null;
  let startupMs = 0;
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
    if (lastReturn === null) { startupMs = at - started; lastReturn = at; }
    gapMs += at - lastReturn;
    const done = submit(buffers);
    const back = performance.now();
    submitMs += back - at;
    const submitLabel = [...new Set(sinceLabels)].sort().join("+") || "(no encoder)";
    submits.push({
      label: submitLabel,
      at,
      atMs: round(at - started),
      buffers: buffers?.length ?? 0,
      passes: sincePasses,
      dispatches: sinceDispatches,
      encoders: sinceEncoders,
      gapMs: round(at - lastReturn),
    });
    const name = submitLabel;
    const row = byLabel.get(name) ?? { submits: 0, dispatches: 0, gapMs: 0 };
    row.submits += 1; row.dispatches += sinceDispatches; row.gapMs += at - lastReturn;
    row.from = lastReturn;
    byLabel.set(name, row);
    sincePasses = 0; sinceDispatches = 0; sinceEncoders = 0; sinceLabels = [];
    lastReturn = back;
    return done;
  };

  // 🔴 THE TWO WAYS A HOST WAITS, and neither profiler here can see either.
  // `mapAsync` is a readback and drains the queue; `onSubmittedWorkDone` is the
  // drain with no data. A fold that does one of these per block has serialised
  // itself against the device whatever its kernels cost.
  // 🔴 A GAP IS NOT HOST WORK UNTIL THE WAITING IS TAKEN OUT OF IT. The time
  // between one submit returning and the next being called contains both the
  // host building commands and the host WAITING for the device, and the two
  // want opposite fixes. Every stall is recorded as an interval so the summary
  // can subtract the overlap and report what the CPU actually did.
  const stalls = [];
  const workDone = device.queue.onSubmittedWorkDone.bind(device.queue);
  device.queue.onSubmittedWorkDone = () => {
    const at = performance.now();
    return workDone().then((value) => {
      stalls.push({ kind: "workDone", from: at, to: performance.now(),
                    atMs: round(at - started), ms: round(performance.now() - at) });
      return value;
    });
  };
  // 🔴 AND THE THIRD WAY A HOST WAITS, which is the one nothing suspected.
  // `popErrorScope` is awaited at the end of nearly every GPU module here - 41
  // sites - and it is a round trip to the GPU process, not a local check.
  const popScope = device.popErrorScope.bind(device);
  const popSites = new Map();
  device.popErrorScope = () => {
    const at = performance.now();
    // Where from: a `popErrorScope` that drains the pipeline is only actionable
    // once it names the module that awaited it.
    const line = (new Error().stack ?? "").split("\n")
      .find((row) => /\/src\//.test(row) && !/probe-submits/.test(row));
    const site = (line ?? "?").trim().replace(/^at\s+/, "")
      .replace(/^.*\/src\//, "src/").replace(/:\d+\)?$/, "");
    return popScope().then((value) => {
      const took = performance.now() - at;
      const row = popSites.get(site) ?? { count: 0, ms: 0 };
      row.count += 1; row.ms += took;
      popSites.set(site, row);
      stalls.push({ kind: "popErrorScope", from: at, to: performance.now(),
                    atMs: round(at - started), ms: round(took) });
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
          stalls.push({ kind: "mapAsync", from: at, to: performance.now(),
                        atMs: round(at - started), bytes: descriptor.size ?? 0,
                        ms: round(performance.now() - at) });
          return value;
        });
      };
    }
    return buffer;
  };

  const restore = () => {
    device.createCommandEncoder = original.createCommandEncoder;
    device.createBuffer = original.createBuffer;
    device.createBindGroup = original.createBindGroup;
    device.popErrorScope = original.popErrorScope;
    device.queue.submit = original.submit;
    device.queue.onSubmittedWorkDone = original.onSubmittedWorkDone;
  };

  const summary = (wallMs) => {
    // 🔴 THE UNION OF THE STALLS, not their sum: a fold holds hundreds of
    // `onSubmittedWorkDone` promises at once - the block-progress ones are
    // deliberately not awaited - so the sum is many times the wall and means
    // nothing. Merged into disjoint intervals, they are the time the host had
    // at least one wait outstanding, and that is what a gap can be charged.
    const merged = [];
    for (const stall of [...stalls].sort((a, b) => a.from - b.from)) {
      const last = merged[merged.length - 1];
      if (last !== undefined && stall.from <= last[1]) {
        last[1] = Math.max(last[1], stall.to);
      } else merged.push([stall.from, stall.to]);
    }
    const waitingBetween = (from, to) => {
      let total = 0;
      for (const [a, b] of merged) {
        if (b <= from) continue;
        if (a >= to) break;
        total += Math.min(b, to) - Math.max(a, from);
      }
      return total;
    };
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
      startupMs: round(startupMs),
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
      // 🔴 THE ACTIONABLE TABLE, AND THE ONLY ONE. An earlier version had a
      // second over the run's second half, on the theory that the warm fold is
      // the interesting one - but it charged the raw gap without subtracting
      // the waits, so its rows said the opposite of this one's for the same
      // encoder. Two tables that disagree are worse than the weaker of them:
      // `--repeats` is how to ask about a repeated fold now.
      byEncoder: (() => {
        const into = new Map();
        for (const row of submits) {
          const found = into.get(row.label)
            ?? { submits: 0, dispatches: 0, gapMs: 0, hostMs: 0 };
          found.submits += 1;
          found.dispatches += row.dispatches;
          found.gapMs += row.gapMs;
          found.hostMs += Math.max(0, row.gapMs - waitingBetween(row.at - row.gapMs, row.at));
          into.set(row.label, found);
        }
        return [...into.entries()]
          .map(([label, row]) => ({ label, submits: row.submits,
            dispatches: row.dispatches, gapMs: round(row.gapMs),
            // What the CPU did, with every outstanding wait subtracted. THIS is
            // the column a host-side change can move.
            hostMs: round(row.hostMs) }))
          .sort((a, b) => b.hostMs - a.hostMs).slice(0, 20);
      })(),
      stallMs: round([...byKind.values()].reduce((sum, row) => sum + row.ms, 0)),
      stalls: [...byKind.entries()].map(([kind, row]) =>
        ({ kind, count: row.count, ms: round(row.ms) })),
      // 🔴 WHICH MODULE AWAITED A PIPELINE DRAIN, and how often.
      popSites: [...popSites.entries()]
        .map(([site, row]) => ({ site, count: row.count, ms: round(row.ms) }))
        .sort((a, b) => b.ms - a.ms).slice(0, 14),
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
  return { summary, restore };
}

const median = (values) => {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted.length === 0 ? 0 : sorted[Math.floor(sorted.length / 2)];
};

export async function main(device, args) {
  const tool = option(args, "tool");
  if (tool === undefined) throw new Error("probe-submits needs --tool=<module under tools/gpu>");
  const repeats = Math.max(1, Number(option(args, "repeats", "1")));
  const rest = args.filter((a) => !a.startsWith("--tool=") && !a.startsWith("--repeats="));
  const module = await import(`./${tool}.js`);
  // 🔴 EVERY REPEAT GETS A FRESH INSTRUMENT, so the rows are one run's and can
  // be medianed. Wrapping once and dividing would average a first fold's
  // compiles into every later one.
  const runs = [];
  let result;
  for (let round = 0; round < repeats; round += 1) {
    const each = instrumentSubmits(device);
    const eachUploads = instrumentUploads();
    const at = performance.now();
    result = await module.main(device, rest);
    runs.push({ summary: each.summary(performance.now() - at),
                uploads: eachUploads.report(), restore: each.restore });
    eachUploads.restore();
    each.restore();
  }
  const instrument = { summary: () => runs[runs.length - 1].summary };
  const uploads = { report: () => runs[runs.length - 1].uploads, restore: () => {} };
  const summary = repeats === 1 ? runs[0].summary : (() => {
    const base = runs[runs.length - 1].summary;
    const numbers = Object.keys(base).filter((k) => typeof base[k] === "number");
    const merged = Object.fromEntries(numbers
      .map((k) => [k, Math.round(median(runs.map((r) => r.summary[k])) * 10) / 10]));
    // Per-encoder rows: median of the runs that HAVE the row, so a label that
    // only appears on a first fold is not silently averaged with zeros.
    const labels = new Set(runs.flatMap((r) => r.summary.byEncoder.map((e) => e.label)));
    return {
      ...base, ...merged, repeats,
      byEncoder: [...labels].map((label) => {
        const rows = runs.map((r) => r.summary.byEncoder.find((e) => e.label === label))
          .filter((row) => row !== undefined);
        return { label, runs: rows.length,
                 submits: median(rows.map((r) => r.submits)),
                 dispatches: median(rows.map((r) => r.dispatches)),
                 gapMs: Math.round(median(rows.map((r) => r.gapMs)) * 10) / 10,
                 hostMs: Math.round(median(rows.map((r) => r.hostMs)) * 10) / 10 };
      }).sort((a, b) => b.hostMs - a.hostMs).slice(0, 20),
    };
  })();
  const busiest = [...summary.busiest.entries()]
    .sort((a, b) => b[1] - a[1]).slice(0, 8)
    .map(([atMs, count]) => [atMs, count]);
  return {
    tool, ...summary, busiest,
    uploads: uploads.report(),
    tool_: Object.fromEntries(Object.entries(result ?? {})
      .filter(([, value]) => typeof value === "number" || typeof value === "boolean")),
  };
}
