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
import { noteAllocation } from "./device-memory.js";

const byDevice = new WeakMap();

/**
 * @param {GPUDevice} device
 * @param {object} key      the weight object this data belongs to
 * @param {string} label    names the buffer, and separates two uses of one key
 * @param {() => Float32Array} pack  called only on a miss
 */
export function residentWeightBuffer(device, key, label, pack) {
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
  const found = forKey.get(label);
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
  forKey.set(label, buffer);
  return buffer;
}
