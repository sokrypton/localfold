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

/**
 * What `pack()` has cost on this page: the HOST packers behind every resident
 * weight buffer, which no compute-pass profiler can see.
 */
export const residentPackStats = { calls: 0, ms: 0, bytes: 0, byLabel: new Map() };

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
/**
 * A resident buffer of a known size, filled by the caller on the DEVICE.
 *
 * 🔴 THE SIBLING OF residentWeightBuffer, FOR WEIGHTS THAT NEVER TOUCH THE
 * HOST. That one takes a `pack` returning bytes to upload; this one takes a
 * size and a `fill` handed the buffer, so an int5 decode can run as a compute
 * pass instead of a JavaScript loop - see src/runtime/quantised-upload.js. The
 * caching, the labelling and the accounting are the same, and so is the rule
 * about `variant`.
 *
 * `fill` runs only when the buffer is created. A cache hit returns the buffer
 * without calling it, which is what makes a second fold free.
 */
export async function residentWeightBufferFilled(device, key, label, byteLength, fill,
                                                 variant = "") {
  const slotOf = (forKey) => forKey.get(variant === "" ? label : `${label}\u0000${variant}`);
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
  const found = slotOf(forKey);
  if (found !== undefined) return found;
  const size = Math.ceil(byteLength / 4) * 4;
  noteAllocation(device, label, size);
  const buffer = device.createBuffer({
    label,
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  await fill(buffer);
  const slot = variant === "" ? label : `${label}\u0000${variant}`;
  forKey.set(slot, buffer);
  const held = heldByDevice.get(device) ?? [];
  held.push({ buffer, size, forKey, label: slot });
  heldByDevice.set(device, held);
  return buffer;
}

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
  // 🔴 WHAT THE HOST PACKERS STILL COST, WHICH NOTHING COULD SEE. Every device
  // decode falls back to one of these when it cannot take a tensor, silently
  // by design on the AF3 side - and a fallback that costs 400 ms of a first
  // fold looks exactly like a slow machine. `fold-af2.js` has reported
  // `packBy` since the same question was asked there; this is the same
  // accounting one level down, so every model gets it.
  const packedAt = performance.now();
  const data = pack();
  residentPackStats.calls += 1;
  residentPackStats.bytes += data.byteLength;
  residentPackStats.ms += performance.now() - packedAt;
  const byLabel = residentPackStats.byLabel;
  const row = byLabel.get(label) ?? { calls: 0, ms: 0, bytes: 0 };
  row.calls += 1; row.bytes += data.byteLength; row.ms += performance.now() - packedAt;
  byLabel.set(label, row);
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
 * Destroy weight buffers resident on a device, and forget them.
 *
 * For the caller that has just been refused an allocation and is about to fall
 * back to uploading per pass: without this the residency built up to the
 * refusal is stranded on the device, holding the budget that the fallback then
 * has to fit inside. Anything asking for one of these afterwards packs and
 * uploads it again, which is what the fallback does anyway.
 *
 * 🔴 AND A STAGE'S RESIDENCY CAN BE GIVEN BACK WHEN THE STAGE IS OVER, which
 * is what `prefix` is for. An AF3 fold's peak is not in its diffusion and not
 * in its trunk: it is in the CONFIDENCE HEAD, which runs four more pairformer
 * blocks after the sampler has finished, while the diffusion transformer's
 * weights are still resident and can no longer be read by anything. Releasing
 * by prefix at that point is the difference between a fold that fits and one
 * that does not, and it costs only what re-packing costs the NEXT fold.
 *
 * @param {string} [prefix] release only buffers whose label starts with this.
 *   Omit to release everything, which is the refusal path's meaning.
 * @returns {number} bytes reclaimed
 */
export function releaseResidentWeights(device, prefix) {
  const held = heldByDevice.get(device);
  if (held === undefined) return 0;
  let bytes = 0;
  const kept = [];
  for (const entry of held) {
    if (prefix !== undefined && !String(entry.buffer.label ?? "").startsWith(prefix)) {
      kept.push(entry);
      continue;
    }
    entry.forKey.delete(entry.label);
    entry.buffer.destroy();
    noteDestroy(device, entry.size, entry.buffer.label);
    bytes += entry.size;
  }
  if (kept.length === 0) heldByDevice.delete(device);
  else heldByDevice.set(device, kept);
  return bytes;
}

/**
 * A packed weight buffer AND the offsets that came with it, packed once.
 *
 * 🔴 THE OFFSETS ARE WHY A RESIDENT PACK IS NOT JUST A RESIDENT UPLOAD. Every
 * packer in this repository returns `{data, offsets}` and the caller needs the
 * offsets on EVERY pass, for the uniform - so caching only the buffer would
 * still run the pack to get them, which is the host work that costs the time.
 * The offsets are a handful of integers; they are kept, and the array of
 * weights is not.
 *
 * Keyed on the weight object like everything else here, and by `variant`
 * within it, because a pack in a different precision or layout has different
 * offsets as well as different bytes.
 */
const offsetsByKey = new WeakMap();

export function residentPack(device, key, label, pack, variant = "") {
  let forKey = offsetsByKey.get(key);
  if (forKey === undefined) {
    forKey = new Map();
    offsetsByKey.set(key, forKey);
  }
  const slot = `${label}\u0000${variant}`;
  const buffer = residentWeightBuffer(device, key, label, () => {
    const packed = pack();
    forKey.set(slot, packed.offsets);
    return packed.data;
  }, variant);
  // A hit on the buffer with a miss on the offsets cannot happen - they are
  // written together - but a caller that swapped one cache for the other would
  // find out here rather than by reading zeros out of a uniform.
  if (!forKey.has(slot)) forKey.set(slot, pack().offsets);
  return { buffer, offsets: forKey.get(slot) };
}
