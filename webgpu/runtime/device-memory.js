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
    account = {
      residentBytes: 0, peakBytes: 0, budgetBytes: undefined, count: 0,
      // 🔴 WHAT IS ON THE DEVICE, BY WHAT IT IS. The totals alone say a fold
      // holds 567 MiB and nothing about which tensor to attack, and the
      // allocator's callers already pass a label - it was simply thrown away.
      // Keyed by the label with its trailing index stripped, so the 48 copies
      // of one block's weights read as one row rather than as forty-eight.
      byLabel: new Map(),
      // 🔴 byLabel ABOVE IS CUMULATIVE, WHICH IS THE WRONG QUESTION FOR A PEAK.
      // It sums every allocation a label ever made, so a scratch tensor taken
      // and given back forty-eight times reads as forty-eight times its size -
      // useful for "what churns", useless for "what is the 552 MiB made of".
      // This one is what is LIVE, and the composition is copied out of it at
      // the moment the peak is set, which is the only moment it can be had.
      liveByLabel: new Map(),
      peakByLabel: [],
    };
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
 * `navigator.deviceMemory` is system RAM in GiB, rounded down to a power of two:
 * this 16 GiB Mac reports 16, and a phone reports 4 or less. A third of it
 * leaves room for the browser, the driver and the system, which on unified
 * memory are all spending the same pool - 5461 MiB here, against the 1390 MiB
 * a 31-residue fold actually holds, and 1365 MiB on a 4 GiB phone, which is
 * under it. That is the machine this exists for.
 *
 * 🔴 IT IS A GUESS AND IT IS ALLOWED TO BE WRONG, in either direction: too
 * high and the fallback never fires on a machine that needed it, too low and a
 * fold that would have fitted runs slower. What it must not do is report a
 * number nobody checked as if it were a device limit, which is why the only
 * thing it is used for is choosing between two paths that both work.
 */
export function budgetForDevice(fallbackGiB = 4, { gpuMemoryBytes, learnedBytes } = {}) {
  const gib = (typeof navigator === "object" && navigator !== null
    && typeof navigator.deviceMemory === "number") ? navigator.deviceMemory : fallbackGiB;
  const bytes = [Math.round(gib * 1024 * 1024 * 1024 / 3)];
  // A third of system RAM says nothing about a discrete card's own memory: a
  // 32 GiB desktop with an 8 GiB RTX 3070 was given a 10.9 GiB ceiling, and a
  // fold under it ran the card out and lost the device. The same fifth is kept
  // back for the desktop, the browser and the driver's own overhead.
  if (gpuMemoryBytes !== undefined) bytes.push(Math.round(gpuMemoryBytes * GPU_MEMORY_SHARE));
  if (learnedBytes !== undefined) bytes.push(learnedBytes);
  return Math.min(...bytes);
}

/** How much of a discrete card's own memory one fold may hold. */
export const GPU_MEMORY_SHARE = 0.8;

const GIB = 1024 * 1024 * 1024;

/**
 * The smallest memory each consumer GPU name has shipped with, in GiB.
 *
 * WebGPU reports no memory size outside Chrome's developer-features flag, and
 * the unmasked WebGL renderer string is the one place a page can read the card's
 * name. A model that shipped in several sizes (RTX 3060: 6 on laptops, 8 and 12
 * on desktops) takes the smallest, because this is only ever used as a ceiling:
 * too low costs a warning on a fold that would have fitted, too high costs the
 * device. A name not listed here keeps the ceiling from system RAM alone.
 */
const GPU_MEMORY_GIB = [
  [/\bGTX\s*1050\b/i, 2], [/\bGTX\s*1060\b/i, 3], [/\bGTX\s*1070\b/i, 8], [/\bGTX\s*1080\b/i, 8],
  [/\bGTX\s*1650\b/i, 4], [/\bGTX\s*1660\b/i, 6],
  [/\bRTX\s*2060\b/i, 6], [/\bRTX\s*2070\b/i, 8], [/\bRTX\s*2080\b/i, 8],
  [/\bRTX\s*3050\b/i, 4], [/\bRTX\s*3060\b/i, 6], [/\bRTX\s*3070\b/i, 8], [/\bRTX\s*3080\b/i, 8],
  [/\bRTX\s*3090\b/i, 24],
  [/\bRTX\s*4050\b/i, 6], [/\bRTX\s*4060\b/i, 8], [/\bRTX\s*4070\b/i, 8], [/\bRTX\s*4080\b/i, 12],
  [/\bRTX\s*4090\b/i, 16],
  [/\bRTX\s*5050\b/i, 8], [/\bRTX\s*5060\b/i, 8], [/\bRTX\s*5070\b/i, 8], [/\bRTX\s*5080\b/i, 16],
  [/\bRTX\s*5090\b/i, 24],
  [/\bRX\s*5500\b/i, 4], [/\bRX\s*5600\b/i, 6], [/\bRX\s*5700\b/i, 8],
  [/\bRX\s*6400\b/i, 4], [/\bRX\s*6500\b/i, 4], [/\bRX\s*66[05]0\b/i, 8], [/\bRX\s*6700\b/i, 10],
  [/\bRX\s*6750\b/i, 12], [/\bRX\s*6800\b/i, 12], [/\bRX\s*6900\b/i, 16],
  [/\bRX\s*7600\b/i, 8], [/\bRX\s*7700\b/i, 12], [/\bRX\s*7800\b/i, 16], [/\bRX\s*7900\b/i, 16],
  [/\bRX\s*9060\b/i, 8], [/\bRX\s*9070\b/i, 16],
  [/\bArc\b.*\bA380\b/i, 6], [/\bArc\b.*\bA5[08]0\b/i, 8], [/\bArc\b.*\bA7[57]0\b/i, 8],
  [/\bArc\b.*\bB570\b/i, 10], [/\bArc\b.*\bB580\b/i, 12],
];

/**
 * A card's memory from its name, or undefined when the name says nothing.
 *
 * @param {string | null | undefined} name the unmasked WebGL renderer string
 */
export function gpuMemoryFromName(name) {
  if (typeof name !== "string") return undefined;
  for (const [pattern, gib] of GPU_MEMORY_GIB) {
    if (pattern.test(name)) return gib * GIB;
  }
  return undefined;
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
  const family = labelFamily(label);
  const seen = account.byLabel.get(family);
  if (seen === undefined) account.byLabel.set(family, { bytes, count: 1 });
  else { seen.bytes += bytes; seen.count += 1; }
  const live = account.liveByLabel.get(family);
  if (live === undefined) account.liveByLabel.set(family, { bytes, count: 1 });
  else { live.bytes += bytes; live.count += 1; }
  if (account.residentBytes > account.peakBytes) {
    account.peakBytes = account.residentBytes;
    account.peakByLabel = [...account.liveByLabel.entries()]
      .map(([name, seen]) => ({ label: name, bytes: seen.bytes, count: seen.count }));
  }
}

/**
 * The label without whatever distinguishes one instance of it from the next.
 *
 * A stack allocates "w.tri.out" once a block and the scratch buffers once a
 * pass, so the interesting row is the family and not the instance. Trailing
 * digits and index suffixes are what vary.
 */
function labelFamily(label) {
  return String(label ?? "unlabelled").replace(/[.:#-]?\d+$/, "");
}

/**
 * Give room back, when a buffer is destroyed rather than merely released.
 *
 * @param {string} [label] what is going away, so the live breakdown stays
 *   truthful. A caller that cannot say leaves the totals right and one row of
 *   the breakdown high, which is why every caller here does say.
 */
export function noteDestroy(device, bytes, label) {
  const account = accountFor(device);
  account.residentBytes -= bytes;
  account.count -= 1;
  if (account.residentBytes < 0) throw new Error("GPU memory accounting underflow");
  const live = account.liveByLabel.get(labelFamily(label));
  if (live !== undefined) {
    live.bytes -= bytes;
    live.count -= 1;
    if (live.count <= 0) account.liveByLabel.delete(labelFamily(label));
  }
}

/**
 * Devices that have already refused to hold a model's weights.
 *
 * 🔴 THE DEGRADATION HAS TO OUTLIVE THE OBJECT THAT DISCOVERED IT. The trunk
 * builds a fresh pairformer stack for every pass, so a fallback remembered on
 * the stack is re-discovered on the next one - and each rediscovery costs a
 * whole abandoned stack. Measured at a 400 MiB ceiling that was 654 ms a pass
 * against 531 at 200, because the larger budget got further before it gave up.
 * It belongs to the device, which is the thing that was too small.
 */
const refusedResidency = new WeakSet();

/** Say that this device cannot afford to keep a model's weights on it. */
export function noteResidencyRefused(device) {
  refusedResidency.add(device);
}

/** Whether keeping weights on this device is still worth attempting. */
export function residencyAllowed(device) {
  return !refusedResidency.has(device);
}

/**
 * Whether this device can afford to KEEP its resident weights between folds.
 *
 * 🔴 THE TRADE IS MEASURED AND THE ONLY UNKNOWN IS ROOM. Releasing them costs
 * a later fold 561 MiB of trunk weights in 1341 writeBuffer calls and 325 MiB
 * of sampler weights in 24 int5 upload passes, waited for on the queue rather
 * than inside any compute pass - which is why profile.js saw 144 ms of a 1300
 * ms fold and the rest looked like nothing. `keepTrunkWeights` and
 * `keepSamplerWeights` are per-device priors saying "this card has room", and
 * a card nobody has measured is told nothing at all.
 *
 * A device with no ceiling is a desktop or a server and has room by
 * construction. A device WITH one gets the same question the batched gate
 * gets: is what is already resident small enough against the ceiling that
 * holding it still leaves the fold two thirds of its budget.
 */
export function keepResidentAffordable(device) {
  if (!residencyAllowed(device)) return false;
  const { residentBytes, budgetBytes } = memoryTotals(device);
  if (budgetBytes === undefined || budgetBytes === null) return true;
  return residentBytes * 3 < budgetBytes;
}

/** The ceiling this device was given, or undefined if it has none. */
export function memoryBudgetBytes(device) {
  return accountFor(device).budgetBytes;
}

/**
 * Everything on this device right now.
 *
 * 🔴 RESIDENT, NOT LIVE. A buffer a caller has released but the allocator has
 * pooled is still occupying the device, and is counted here - which is the
 * point: the device does not care that we intend to reuse it.
 */
/**
 * The two totals, and nothing else.
 *
 * 🔴 `memorySnapshot` BUILDS AND SORTS THREE ARRAYS, one per breakdown, which is
 * right for a bench that calls it twice and wrong for anything on a fold's
 * path. The footer's timing panel reads the totals once per phase - a couple of
 * hundred times across a long sampler - and wants none of the breakdowns until
 * a reader opens it. This allocates nothing.
 */
export function memoryTotals(device) {
  const { residentBytes, peakBytes, budgetBytes, count } = accountFor(device);
  return { residentBytes, peakBytes, budgetBytes, bufferCount: count };
}

export function memorySnapshot(device) {
  const { residentBytes, peakBytes, budgetBytes, count, byLabel, peakByLabel } = accountFor(device);
  const largest = [...byLabel.entries()]
    .map(([label, seen]) => ({ label, bytes: seen.bytes, count: seen.count }))
    .sort((a, b) => b.bytes - a.bytes);
  const live = [...accountFor(device).liveByLabel.entries()]
    .map(([label, seen]) => ({ label, bytes: seen.bytes, count: seen.count }))
    .sort((a, b) => b.bytes - a.bytes);
  return {
    residentBytes, peakBytes, budgetBytes, bufferCount: count, byLabel: largest,
    // What is on the device NOW, largest first; sums to residentBytes. Pooled
    // buffers are in here, because the device is still holding them.
    currentByLabel: live,
    // What was actually on the device when it was fullest, largest first. The
    // rows sum to peakBytes; byLabel above does not sum to anything.
    peakByLabel: [...peakByLabel].sort((a, b) => b.bytes - a.bytes),
  };
}
