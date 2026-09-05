/**
 * Make a packed weight buffer resident by decoding it on the device.
 *
 * 🔴 EVERY PACKER IN THIS MODEL HAS THE SAME SHAPE, so the device path can be
 * written once. A packer takes an ORDER of field names, lays them end to end,
 * and hands the result to `concatenateAs` - which for f16 means decoding int5
 * into float32 and narrowing all of it, on the main thread, before a byte
 * reaches the GPU. Measured cold on the int5 manifest:
 *
 *     the diffusion transformer's 24 blocks   440 ms   378.2 MiB
 *     the trunk's 48 single transitions       196      162.1
 *     ...its 48 pair transitions               15       36.0
 *     ...its 48 outgoing triangles             15       18.2
 *     ...its 48 first grid attentions          19       15.1
 *
 * The two f16 rows are where the time is, because they pay the narrowing as
 * well as the decode; the f32 rows are a decode and a memcpy. So this takes the
 * f16 ones.
 *
 * See src/runtime/quantised-upload.js for the kernel and for what makes it
 * safe: one invocation owns one word, so every offset must be even, and the
 * arithmetic is bit-identical to the host decoder's - which was measured before
 * any of it was written, not assumed.
 */
import { SOURCES } from "./weights.js";
import { planBlockUpload, runBlockUpload } from "../runtime/quantised-upload.js";
import { residentWeightBufferFilled } from "../runtime/resident.js";
import { writeInto } from "../runtime/float16.js";

/**
 * @param {object} options
 * @param {object} options.key      what the residency cache is keyed on - the
 *   same object the host path would pass, so the two cannot both allocate
 * @param {string} options.label    the device-memory label
 * @param {string[]} options.order  the packer's field order
 * @param {object} options.weights  a bound descriptor carrying SOURCES
 * @returns {Promise<GPUBuffer | undefined>} undefined when the weights cannot
 *   be decoded this way, and the caller should pack on the host
 */
export async function residentPackedOnDevice(device, options) {
  const { key, label, order, weights, variant = "" } = options;
  if (typeof globalThis.Float16Array !== "function") return undefined;
  const sources = weights?.[SOURCES];
  if (sources === undefined) return undefined;

  const entries = [];
  let total = 0;
  for (const name of order) {
    const thunk = sources[name];
    // 🔴 LENGTHS FROM THE THUNK. `weights[name].length` materialises the tensor,
    // which is the decode this exists to avoid.
    if (typeof thunk !== "function" || !Number.isInteger(thunk.count)) return undefined;
    entries.push({ name, thunk, offset: total, length: thunk.count });
    total += thunk.count;
  }
  const planned = planBlockUpload(entries);
  if (planned === undefined) return undefined;
  // Nothing for the GPU is not a plan: an f32 manifest sends every tensor to
  // the host list, and there is no time to win there anyway.
  if (planned.gpu.params.length === 0) return undefined;

  return residentWeightBufferFilled(
    device, key, label, Math.ceil(total / 2) * 4,
    async (buffer) => {
      for (const entry of planned.host) {
        const half = new globalThis.Float16Array(entry.length);
        writeInto(half, weights[entry.name], 0);
        device.queue.writeBuffer(buffer, entry.offset * 2,
                                 half.buffer, half.byteOffset, half.byteLength);
      }
      const release = await runBlockUpload(device, planned.gpu, buffer);
      // 🔴 THE STAGING IS RELEASED WHEN THE QUEUE SAYS SO, AND THIS DOES NOT
      // WAIT FOR IT. Awaiting `onSubmittedWorkDone` here put a host-device
      // synchronisation inside the pairformer's block loop - 48 of them, one
      // per block - and that loop is written to run ahead of the device on
      // purpose. Measured: a page fold's Trunk 1/2 phase went 647 ms to 778.
      // The promise still fires, and the buffers still go; nothing between now
      // and then reads them.
      void device.queue.onSubmittedWorkDone().then(release);
    },
    variant);
}
