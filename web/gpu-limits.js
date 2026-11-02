/**
 * What the page can find out about this machine's GPU memory, and what it has
 * learned the hard way.
 *
 * Kept apart from web/model.js, which holds no DOM, and from
 * webgpu/runtime/device-memory.js, which a bench uses in node. Everything here
 * needs a page, and every read is guarded because a failure only costs a tighter
 * or looser guess.
 */
import { GPU_MEMORY_SHARE, gpuMemoryFromName } from "../webgpu/runtime/device-memory.js";

const LEARNED_KEY = "localfold.gpu-limit";
/** A loss under this says nothing about memory: nothing large was being held. */
const LEARN_FROM_BYTES = 512 * 1024 * 1024;

let cachedName;

/** The GPU's name as WebGL reports it, once per page. Null where the browser masks it. */
export function gpuName() {
  if (cachedName !== undefined) return cachedName;
  cachedName = null;
  try {
    const context = document.createElement("canvas").getContext("webgl2");
    const info = context?.getExtension("WEBGL_debug_renderer_info");
    if (info) cachedName = context.getParameter(info.UNMASKED_RENDERER_WEBGL) ?? null;
    context?.getExtension("WEBGL_lose_context")?.loseContext();
  } catch { /* no WebGL, which only means no name */ }
  return cachedName;
}

/** The card's memory in bytes, when its name is one that says. */
export function gpuMemoryBytes(adapter) {
  // Chrome reports this only behind its developer-features flag, and then it is
  // the driver's own answer rather than a table's.
  const heaps = adapter?.info?.memoryHeaps;
  if (Array.isArray(heaps) && heaps.length > 0) {
    const local = heaps.filter((heap) => (heap.properties & 1) !== 0);
    const size = Math.max(...(local.length > 0 ? local : heaps).map((heap) => Number(heap.size)));
    if (Number.isFinite(size) && size > 0) return size;
  }
  return gpuMemoryFromName(gpuName());
}

function readLearned() {
  try {
    const saved = JSON.parse(localStorage.getItem(LEARNED_KEY) ?? "null");
    return saved?.name === (gpuName() ?? "unknown") && Number.isFinite(saved.bytes) ? saved.bytes : undefined;
  } catch { return undefined; }
}

/** The ceiling a lost device has taught this browser for this card, if one has. */
export function learnedLimitBytes() {
  return readLearned();
}

/**
 * Remember how much was on the card when its device was lost.
 *
 * The ceiling above is a guess from a name, and on an unlisted card from system
 * RAM alone. A device lost while holding this much is a measurement, so the next
 * device on this card stops short of it. Reading the same loss twice cannot raise
 * it, and `forgetLearnedLimit` is how a reader takes it back when the cause turned
 * out to be something else.
 *
 * @returns {number | undefined} the new ceiling, when this loss set one
 */
export function learnFromLoss(residentBytes) {
  if (!(residentBytes >= LEARN_FROM_BYTES)) return undefined;
  const bytes = Math.min(Math.round(residentBytes * GPU_MEMORY_SHARE), readLearned() ?? Infinity);
  try {
    localStorage.setItem(LEARNED_KEY, JSON.stringify({ name: gpuName() ?? "unknown", bytes }));
  } catch { /* private window: the lesson lasts this page instead of this browser */ }
  return bytes;
}

export function forgetLearnedLimit() {
  try { localStorage.removeItem(LEARNED_KEY); } catch { /* nothing kept */ }
}
