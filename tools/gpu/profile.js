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

/**
 * @param {GPUDevice} device
 * @returns {{report: () => Promise<object[]>, reset: () => void,
 *            restore: () => void} | null} null when unsupported
 */
export function profileDevice(device, options = {}) {
  if (!device.features.has("timestamp-query")) return null;
  const capacity = options.capacity ?? 4096;
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
  const createCommandEncoder = device.createCommandEncoder.bind(device);

  device.createCommandEncoder = (descriptor) => {
    const encoder = createCommandEncoder(descriptor);
    const beginComputePass = encoder.beginComputePass.bind(encoder);
    const finish = encoder.finish.bind(encoder);
    let used = false;
    encoder.beginComputePass = (pass = {}) => {
      // Out of slots, or already timed by the caller: leave it alone.
      if (next + 2 > capacity || pass.timestampWrites !== undefined) return beginComputePass(pass);
      const at = next;
      next += 2;
      used = true;
      spans.push({ label: pass.label ?? "(unlabelled)", at });
      return beginComputePass({ ...pass, timestampWrites: {
        querySet, beginningOfPassWriteIndex: at, endOfPassWriteIndex: at + 1,
      } });
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
    reset() { next = 0; spans = []; },
    restore() { device.createCommandEncoder = createCommandEncoder; },
    async report() {
      const encoder = createCommandEncoder({ label: "profile.readback" });
      encoder.copyBufferToBuffer(resolved, 0, readback, 0, capacity * 8);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const stamps = new BigInt64Array(readback.getMappedRange().slice(0));
      readback.unmap();
      const totals = new Map();
      for (const { label, at } of spans) {
        const nanoseconds = Number(stamps[at + 1] - stamps[at]);
        if (!Number.isFinite(nanoseconds) || nanoseconds < 0) continue;
        const found = totals.get(label) ?? { label, ms: 0, passes: 0 };
        found.ms += nanoseconds / 1e6;
        found.passes += 1;
        totals.set(label, found);
      }
      return [...totals.values()]
        .map((row) => ({ ...row, ms: Number(row.ms.toFixed(2)) }))
        .sort((a, b) => b.ms - a.ms);
    },
  };
}
