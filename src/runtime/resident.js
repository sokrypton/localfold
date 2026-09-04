/**
 * Weight buffers uploaded once and left on the device.
 *
 * 🔴 A DENOISER CALLED TWO HUNDRED TIMES SHOULD UPLOAD ITS WEIGHTS ONCE. Every
 * block loop in this repository packed its weights into a fresh Float32Array
 * and wrote them to the device on every call, over weights that never change -
 * about 630 MB a call in the diffusion transformer alone, which at ~3.6 GB/s
 * was most of what its 24 blocks cost.
 *
 * 🔴 KEYED BY THE WEIGHT OBJECT, DEVICE FIRST. The key is whatever the loader
 * built, so the cache lives exactly as long as the model does and two devices
 * cannot hand each other a buffer. Nothing here can tell a MUTATED array from
 * the one it cached - identity is the question "is this the same model", and
 * the loaders never mutate one.
 *
 * 🔴 NOT THROUGH GpuBufferAllocator, whose allocations are pooled and recycled
 * at the end of the run that made them. These have to outlive every run, so
 * they are created directly and never released.
 */
import { noteAllocation, noteDestroy } from "./device-memory.js";

const byDevice = new WeakMap();

/**
 * Every resident buffer on a device, so they can be given back.
 *
 * 🔴 A WeakMap CANNOT BE EMPTIED, and the degraded path needs to empty one. The
 * maps above are keyed on the weight objects so the buffers die with the model,
 * which is right; but when an allocation is refused for want of budget, the
 * memory that has to be reclaimed is exactly the residency built so far, and
 * nothing can enumerate it. This flat list can.
 */
const heldByDevice = new WeakMap();

/**
 * @param {GPUDevice} device
 * @param {object} key      the weight object this data belongs to
 * @param {string} label    names the buffer, and separates two uses of one key
 * @param {() => Float32Array} pack  called only on a miss
 */
/**
 * @param {string} [variant] what distinguishes two buffers that would otherwise
 *   share a label - the element the weights are packed in, say. It is part of
 *   the cache key and NOT of the label, so a device-memory breakdown still
 *   reads as one row per tensor. Without it an f16 pipeline can be handed the
 *   f32 buffer, which is half the values at twice the stride: a wrong answer
 *   rather than an error.
 */
export function residentWeightBuffer(device, key, label, pack, variant = "") {
  let forDevice = byDevice.get(device);
  if (forDevice === undefined) {
    forDevice = new WeakMap();
    byDevice.set(device, forDevice);
  }
  let forKey = forDevice.get(key);
  if (forKey === undefined) {
    forKey = new Map();
    forDevice.set(key, forKey);
  }
  const slot = variant === "" ? label : `${label}\u0000${variant}`;
  const found = forKey.get(slot);
  if (found !== undefined) return found;
  const data = pack();
  const size = Math.ceil(data.byteLength / 4) * 4;
  noteAllocation(device, label, size);
  const buffer = device.createBuffer({
    label,
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
  forKey.set(slot, buffer);
  const held = heldByDevice.get(device) ?? [];
  held.push({ buffer, size, forKey, label: slot });
  heldByDevice.set(device, held);
  return buffer;
}

/**
 * Destroy every weight buffer resident on a device, and forget them.
 *
 * For the caller that has just been refused an allocation and is about to fall
 * back to uploading per pass: without this the residency built up to the
 * refusal is stranded on the device, holding the budget that the fallback then
 * has to fit inside. Anything asking for one of these afterwards packs and
 * uploads it again, which is what the fallback does anyway.
 *
 * @returns {number} bytes reclaimed
 */
export function releaseResidentWeights(device) {
  const held = heldByDevice.get(device);
  if (held === undefined) return 0;
  let bytes = 0;
  for (const entry of held) {
    entry.forKey.delete(entry.label);
    entry.buffer.destroy();
    noteDestroy(device, entry.size);
    bytes += entry.size;
  }
  heldByDevice.delete(device);
  return bytes;
}
