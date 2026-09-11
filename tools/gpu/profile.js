/**
 * Per-pass GPU time, by wrapping the device rather than editing the kernels.
 *
 * 🔴 EVERY BISECT IN THIS REPOSITORY SO FAR DISABLED A PASS AND RE-MEASURED,
 * AND IT COST TWO WRONG CONCLUSIONS. That method attributes to a pass whatever
 * changes when it is gone, which folds in scheduling and overlap, and its
 * resolution is the bench's noise - about 10 ms before bench-head.js started
 * taking a median of nine calls. It once reported a REMOVED pass as costing
 * negative time, and it named the atom encoder's attention blocks when the real
 * cost was four times bigger and in a pass nobody had suspected.
 *
 * `timestamp-query` is already among the features src/runtime/device.js
 * requests. This uses it: every compute pass already carries a label, so
 * wrapping createCommandEncoder is enough to time all of them with no change to
 * any kernel.
 *
 * 🔴 CHROME QUANTISES TIMESTAMPS - 100 microseconds at the time of writing, for
 * fingerprinting reasons. A single 0.2 ms pass is therefore unmeasurable, and
 * the report says so by printing the number of passes behind each total: a
 * label with hundreds of passes sums to something meaningful, a label with one
 * does not.
 */

import { setDeviceTuning } from "../../src/runtime/device-profile.js";

/**
 * @param {GPUDevice} device
 * @returns {{report: () => Promise<object[]>, reset: () => void,
 *            restore: () => void} | null} null when unsupported
 */
export function profileDevice(device, options = {}) {
  if (!device.features.has("timestamp-query")) return null;
  // 🔴 PROFILING CHANGES THE ENCODING, AND THIS IS WHERE IT SAYS SO. This
  // attributes time by PASS, so a stack that puts thirty dispatches in one pass
  // is one row with one label - which is no profile at all. `batchComputePasses`
  // off gives every dispatch its own pass, and costs about 2.5% on the
  // diffusion transformer. That is on top of the 2-4% a trunk pass pays for the
  // timestamps themselves and the 45% a denoiser call pays; see docs/A100.md.
  //
  // 🔴 AND `batched: true` IS HOW TO ASK WHETHER A FOLD IS GPU-BOUND AT ALL.
  // Unbatching multiplies the pass count - an OpenDDE fold goes to 2048, which
  // is EXACTLY the query set's capacity, so its profile was a truncated prefix
  // and every "GPU idle share" read off one described part of a fold. Batched,
  // a whole fold fits in a few hundred passes and `summary()` is the truth
  // about the wall; what is lost is per-kernel attribution, which is the other
  // question.
  if (options.batched !== true) setDeviceTuning(device, { batchComputePasses: false });
  // 🔴 4096 IS A DEVICE MAXIMUM, NOT A CHOICE. createQuerySet rejects anything
  // larger, and the rejection is an uncaptured device error rather than a
  // throw - so a bench that asked for more simply died with no stack. A stack
  // with more passes than this loses the tail, which the report's pass counts
  // make visible.
  const capacity = Math.min(options.capacity ?? 4096, 4096);
  const querySet = device.createQuerySet({ type: "timestamp", count: capacity });
  const resolved = device.createBuffer({
    size: capacity * 8,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    size: capacity * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  let next = 0;
  let spans = [];
  // 🔴 WHAT DID NOT FIT, because a profile that silently stops is worse than no
  // profile. The query set is capped at 4096 timestamps by the device, which is
  // 2048 passes - and a whole AF3 fold at 200 tokens uses exactly that in its
  // trunk, so every sampler pass after it was dropped and the report read as
  // "the denoiser costs almost nothing". `attend` showed 38 passes where 16
  // calls of 24 blocks is 384. The report carries this now; a caller with a
  // non-zero `dropped` is reading a prefix, not a fold.
  let dropped = 0;
  let submits = 0;
  const submit = device.queue.submit.bind(device.queue);
  device.queue.submit = (buffers) => { submits += 1; return submit(buffers); };
  const createCommandEncoder = device.createCommandEncoder.bind(device);

  device.createCommandEncoder = (descriptor) => {
    const encoder = createCommandEncoder(descriptor);
    const beginComputePass = encoder.beginComputePass.bind(encoder);
    const finish = encoder.finish.bind(encoder);
    let used = false;
    encoder.beginComputePass = (pass = {}) => {
      // Out of slots, or already timed by the caller: leave it alone.
      if (next + 2 > capacity || pass.timestampWrites !== undefined) {
        if (next + 2 > capacity) dropped += 1;
        return beginComputePass(pass);
      }
      const at = next;
      next += 2;
      used = true;
      const span = { label: pass.label ?? "(unlabelled)", at, groups: 0 };
      spans.push(span);
      const timed = beginComputePass({ ...pass, timestampWrites: {
        querySet, beginningOfPassWriteIndex: at, endOfPassWriteIndex: at + 1,
      } });
      // 🔴 HOW MANY WORKGROUPS A PASS ASKED FOR, which is the question a pass
      // TIME cannot answer and the one that found this model's worst kernel.
      // The atom decoder's `start` was 1.89 ms in 26 workgroups - under 1% of
      // this device - and nothing in a duration says so; a slow pass and an
      // empty one look alike. Recorded per pass and summed per label by
      // report(), so an underfilled kernel is visible without guessing which
      // to suspect.
      const dispatchWorkgroups = timed.dispatchWorkgroups.bind(timed);
      timed.dispatchWorkgroups = (x = 1, y = 1, z = 1) => {
        span.groups += x * y * z;
        return dispatchWorkgroups(x, y, z);
      };
      return timed;
    };
    encoder.finish = (descriptorIn) => {
      // 🔴 RESOLVED BEFORE finish AND OUTSIDE ANY PASS, which is the only place
      // WebGPU allows it. Resolving the whole set each time is wasteful and
      // simple; the alternative is tracking per-encoder ranges for no gain.
      if (used) encoder.resolveQuerySet(querySet, 0, capacity, resolved, 0);
      return finish(descriptorIn);
    };
    return encoder;
  };

  return {
    reset() { next = 0; spans = []; dropped = 0; submits = 0; },
    /** Passes that found no slot. Non-zero means the report is a prefix. */
    dropped: () => dropped,
    capacityPasses: Math.floor(capacity / 2),
    restore() {
      device.createCommandEncoder = createCommandEncoder;
      device.queue.submit = submit;
    },
    /**
     * 🔴 THE SUM OF THE PASSES IS NOT HOW LONG THE GPU WAS BUSY FOR. `report`
     * adds up each pass's own duration, which says nothing about the GAPS
     * between them - and a gap is the GPU idle, waiting for a host that has
     * not submitted the next thing yet. On unified memory those gaps are
     * small; on a discrete card behind an IPC boundary they are where the time
     * goes, and no per-pass total can see them.
     *
     * `span` is the first pass's start to the last pass's end, in GPU time.
     * `idle` is that minus the sum, which is the bubble - and `submits` is how
     * many times the host handed work over, because a bubble divided by the
     * submits is what one hand-over costs.
     */
    async summary() {
      const encoder = createCommandEncoder({ label: "profile.summary" });
      encoder.copyBufferToBuffer(resolved, 0, readback, 0, capacity * 8);
      submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const stamps = new BigInt64Array(readback.getMappedRange().slice(0));
      readback.unmap();
      let first = null; let last = null; let sum = 0; let counted = 0;
      for (const { at } of spans) {
        const start = stamps[at]; const end = stamps[at + 1];
        const ns = Number(end - start);
        if (!Number.isFinite(ns) || ns < 0) continue;
        sum += ns; counted += 1;
        if (first === null || start < first) first = start;
        if (last === null || end > last) last = end;
      }
      const spanNs = first === null ? 0 : Number(last - first);
      return {
        passes: counted,
        // 🔴 NON-ZERO MEANS THIS IS A PREFIX OF A FOLD AND NOT A FOLD. Read it
        // before anything else in this object.
        dropped,
        submits,
        sumMs: Number((sum / 1e6).toFixed(2)),
        spanMs: Number((spanNs / 1e6).toFixed(2)),
        idleMs: Number(((spanNs - sum) / 1e6).toFixed(2)),
        idleShare: spanNs > 0 ? Number(((spanNs - sum) / spanNs).toFixed(3)) : 0,
      };
    },
    async report() {
      const encoder = createCommandEncoder({ label: "profile.readback" });
      encoder.copyBufferToBuffer(resolved, 0, readback, 0, capacity * 8);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const stamps = new BigInt64Array(readback.getMappedRange().slice(0));
      readback.unmap();
      const totals = new Map();
      for (const { label, at, groups } of spans) {
        const nanoseconds = Number(stamps[at + 1] - stamps[at]);
        if (!Number.isFinite(nanoseconds) || nanoseconds < 0) continue;
        const found = totals.get(label) ?? { label, ms: 0, passes: 0, groups: 0 };
        found.ms += nanoseconds / 1e6;
        found.passes += 1;
        found.groups += groups ?? 0;
        totals.set(label, found);
      }
      return [...totals.values()]
        .map((row) => ({
          ...row,
          ms: Number(row.ms.toFixed(2)),
          // The average dispatch, which is what "is this kernel filling the
          // device" is asked of.
          groupsPerPass: Math.round(row.groups / Math.max(row.passes, 1)),
        }))
        .sort((a, b) => b.ms - a.ms);
    },
  };
}
