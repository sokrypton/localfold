/**
 * One ESMFold2 weight tensor, decoded from its quantisation codes on the DEVICE.
 *
 * 🔴 THE DENOISER'S WEIGHTS ARE TWO SECONDS OF HOST WORK BEFORE A FOLD MEANS
 * ANYTHING. Loading them out of the int5 bundle is 1.17 s on the main thread -
 * which the fold's own clock does not even cover, because it starts after - and
 * narrowing the token blocks to f16 inside `prepare` is another 880 ms. Neither
 * needs a host: the codes are already in memory and the arithmetic is
 * `code * scale + zero`.
 *
 * This is `src/af3/device-weights.js` for one tensor at a time, because the
 * denoiser's buffers are one tensor each rather than one packed block. It
 * returns undefined for anything it cannot take - an f32 bundle, a reader with
 * no sources, a codec the planner refuses - and the caller narrows on the host
 * as it always did.
 */
import { planBlockUpload, runBlockUpload } from "../runtime/quantised-upload.js";
import { residentWeightBufferFilled } from "../runtime/resident.js";

/** How many elements a manifest record holds. */
export function elementsOf(record) {
  return (record.shape ?? []).reduce((total, extent) => total * extent, 1);
}

/**
 * @param {object} options key, label, variant, source, destination ("f16"|"f32")
 * @returns {Promise<GPUBuffer | undefined>}
 */
export async function residentTensorOnDevice(device, options) {
  const { key, label, variant = "", source, destination } = options;
  if (source === undefined || source === null) return undefined;
  const elements = elementsOf(source.record);
  if (!(elements > 0)) return undefined;
  // A thunk in the shape planBlockUpload reads: it never calls it, it only
  // wants the store, the name and the range. See src/af3/weights.js.
  const thunk = () => { throw new Error("the device decoder does not call the thunk"); };
  thunk.store = { tensorSource: () => source };
  thunk.tensorName = label;
  thunk.first = 0;
  thunk.count = elements;
  const planned = planBlockUpload(
    [{ name: label, thunk, offset: 0, length: elements }], destination);
  if (planned === undefined || planned.gpu.params.length === 0) return undefined;
  const bytes = destination === "f32" ? elements * 4 : Math.ceil(elements / 2) * 4;
  return residentWeightBufferFilled(device, key, label, bytes, async (buffer) => {
    const release = await runBlockUpload(device, planned.gpu, buffer);
    // 🔴 NOT AWAITED. The staging goes when the queue says so; waiting here
    // would put a host-device synchronisation inside `prepare`, which is the
    // one place this is trying to take work OUT of.
    void device.queue.onSubmittedWorkDone().then(release);
  }, variant);
}

/**
 * Two tensors concatenated along their COLUMNS, decoded on the device.
 *
 * 🔴 `partRun` IS WHAT MAKES THIS EXPRESSIBLE. The destination's row i is
 * source A's row i followed by source B's row i, so the parts alternate in runs
 * of `columns` rather than element by element - which is the same mapping the
 * four-role interleave uses with a run of one. See
 * src/runtime/quantised-upload.js.
 *
 * @param {object} options key, label, variant, sources (two), columns, destination
 */
export async function residentPairOnDevice(device, options) {
  const { key, label, variant = "", sources, columns, destination } = options;
  if (sources.length !== 2 || sources.some((s) => s === undefined || s === null)) return undefined;
  const counts = sources.map((s) => elementsOf(s.record));
  if (counts[0] !== counts[1] || !(counts[0] > 0)) return undefined;
  if (counts[0] % columns !== 0) return undefined;
  const parts = sources.map((source, index) => {
    const thunk = () => { throw new Error("the device decoder does not call the thunk"); };
    thunk.store = { tensorSource: () => source };
    thunk.tensorName = `${label}:${index}`;
    thunk.first = 0;
    thunk.count = counts[index];
    return { store: thunk.store, tensorName: thunk.tensorName,
             first: 0, count: counts[index], bias: 0 };
  });
  const length = counts[0] * 2;
  const planned = planBlockUpload([{
    name: label, offset: 0, length, sources: parts, partRun: columns,
    inner: counts[0], innerStride: 1, outerStride: 0,
  }], destination);
  if (planned === undefined || planned.gpu.params.length === 0) return undefined;
  const bytes = destination === "f32" ? length * 4 : Math.ceil(length / 2) * 4;
  return residentWeightBufferFilled(device, key, label, bytes, async (buffer) => {
    const release = await runBlockUpload(device, planned.gpu, buffer);
    void device.queue.onSubmittedWorkDone().then(release);
  }, variant);
}
