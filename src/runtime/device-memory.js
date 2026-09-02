/**
 * What a device is actually holding, and the ceiling it may not go past.
 *
 * 🔴 NOTHING HERE KNEW HOW MUCH GPU MEMORY A FOLD USED. GpuBufferAllocator
 * counts its own allocations, but the weights that matter most do not go
 * through it: `resident.js`, the diffusion transformer's block buffers and the
 * atom encoder's pair buffer each call `device.createBuffer` directly, because
 * they have to outlive the run that made them. So the one number the benches
 * printed was the small half.
 *
 * 🔴 AND AN OVER-ALLOCATION ON THIS HARDWARE IS NOT AN ERROR, IT IS A FROZEN
 * MACHINE. Metal accepts allocations well past the point where macOS starts
 * paging, and a phone's driver will take them and then be killed by the system.
 * WebGPU reports nothing until it is far too late. A budget checked BEFORE
 * createBuffer turns that into a GpuMemoryBudgetError naming the tensor, the
 * resident total and the ceiling - which a page can show and a bench can catch.
 *
 * The budget is opt-in: without one set, this only counts. Every allocator on a
 * device shares the one total, because the device does.
 */

/** Refused before createBuffer, so the caller learns what would not fit. */
export class GpuMemoryBudgetError extends Error {
  constructor(label, bytes, residentBytes, budgetBytes) {
    const mib = (value) => `${(value / (1024 * 1024)).toFixed(1)} MiB`;
    super(`${label} needs ${mib(bytes)}, which would take this device to `
      + `${mib(residentBytes + bytes)} against a budget of ${mib(budgetBytes)}`);
    this.name = "GpuMemoryBudgetError";
    this.label = label;
    this.bytes = bytes;
    this.residentBytes = residentBytes;
    this.budgetBytes = budgetBytes;
  }
}

const byDevice = new WeakMap();

function accountFor(device) {
  let account = byDevice.get(device);
  if (account === undefined) {
    account = { residentBytes: 0, peakBytes: 0, budgetBytes: undefined, count: 0 };
    byDevice.set(device, account);
  }
  return account;
}

/**
 * The ceiling for one device, in bytes. Pass undefined to lift it.
 *
 * 🔴 THIS IS A GUESS ABOUT THE SYSTEM, NOT A DEVICE LIMIT. WebGPU does not
 * report how much memory a device has, and on unified memory the answer is
 * "whatever the OS is not using". budgetForDevice below is where the guess
 * lives; a caller that knows better should say so.
 */
export function setMemoryBudget(device, budgetBytes) {
  accountFor(device).budgetBytes = budgetBytes;
}

/**
 * A budget for a device nobody has measured, from what the browser will admit.
 *
 * `navigator.deviceMemory` is system RAM in GiB, rounded down to a power of two
 * and capped at 8 by Chromium - so it understates a big machine and is roughly
 * right about a small one, which is the direction that matters. A third of it
 * leaves room for the browser, the driver and the system, which on unified
 * memory are all spending the same pool.
 */
export function budgetForDevice(fallbackGiB = 4) {
  const gib = (typeof navigator === "object" && navigator !== null
    && typeof navigator.deviceMemory === "number") ? navigator.deviceMemory : fallbackGiB;
  return Math.round(gib * 1024 * 1024 * 1024 / 3);
}

/**
 * Claim room for a buffer about to be created. Throws rather than let it be.
 *
 * @param {string} label what the buffer is, so a refusal names it
 */
export function noteAllocation(device, label, bytes) {
  const account = accountFor(device);
  if (account.budgetBytes !== undefined && account.residentBytes + bytes > account.budgetBytes) {
    throw new GpuMemoryBudgetError(label, bytes, account.residentBytes, account.budgetBytes);
  }
  account.residentBytes += bytes;
  account.count += 1;
  if (account.residentBytes > account.peakBytes) account.peakBytes = account.residentBytes;
}

/** Give room back, when a buffer is destroyed rather than merely released. */
export function noteDestroy(device, bytes) {
  const account = accountFor(device);
  account.residentBytes -= bytes;
  account.count -= 1;
  if (account.residentBytes < 0) throw new Error("GPU memory accounting underflow");
}

/**
 * Everything on this device right now.
 *
 * 🔴 RESIDENT, NOT LIVE. A buffer a caller has released but the allocator has
 * pooled is still occupying the device, and is counted here - which is the
 * point: the device does not care that we intend to reuse it.
 */
export function memorySnapshot(device) {
  const { residentBytes, peakBytes, budgetBytes, count } = accountFor(device);
  return { residentBytes, peakBytes, budgetBytes, bufferCount: count };
}
