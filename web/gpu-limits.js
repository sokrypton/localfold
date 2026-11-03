/**
 * What the page can find out about this machine's GPU memory, and what it has
 * learned the hard way.
 *
 * Kept apart from web/model.js, which holds no DOM, and from
 * webgpu/runtime/device-memory.js, which a bench uses in node. Everything here
 * needs a page, and every read is guarded because a failure only costs a tighter
 * or looser guess.
 */
import { GPU_MEMORY_SHARE, gpuMemoryFromName, lossTeachesLimit, UNNAMED_CARD_BYTES } from "../webgpu/runtime/device-memory.js";

const LEARNED_KEY = "localfold.gpu-limit";
const LOSS_KEY = "localfold.gpu-last-loss";
const CHOSEN_KEY = "localfold.gpu-chosen-limit";
const SAME_TROUBLE_MS = 60 * 60 * 1000;

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

/**
 * Whether a name can be the adapter's. WebGL takes the browser's default GPU where WebGPU was asked for the fast one, so on
 * a laptop with two the name can be the other card's, and the table would answer for a card that is not doing the work.
 */
const VENDOR_NAMES = { nvidia: /nvidia|geforce|quadro|tesla/i, intel: /intel/i, amd: /\b(amd|ati|radeon)\b/i };
export const nameIsAdapters = (name, vendor) => VENDOR_NAMES[vendor]?.test(name ?? "") ?? true;

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
  const vendor = adapter?.info?.vendor ?? "";
  const named = nameIsAdapters(gpuName(), vendor) ? gpuMemoryFromName(gpuName()) : undefined;
  if (named !== undefined) return named;
  // Not a table's card. A software adapter and unified memory have no card to run out of.
  if (vendor === "" || vendor === "apple" || vendor === "google" || adapter?.info?.isFallbackAdapter === true) return undefined;
  return UNNAMED_CARD_BYTES;
}

function readLearned() {
  try {
    const saved = JSON.parse(localStorage.getItem(LEARNED_KEY) ?? "null");
    return saved?.name === (gpuName() ?? "unknown") && Number.isFinite(saved.bytes) && saved.bytes > 0 ? saved.bytes : undefined;
  } catch { return undefined; }
}

/** What the reader has said this card can spare, in bytes, if they have. */
export function chosenLimitBytes() {
  try {
    const saved = Number(localStorage.getItem(CHOSEN_KEY));
    return Number.isFinite(saved) && saved > 0 ? saved : undefined;
  } catch { return undefined; }
}

export function setChosenLimitBytes(bytes) {
  try {
    if (bytes === undefined) localStorage.removeItem(CHOSEN_KEY);
    else localStorage.setItem(CHOSEN_KEY, String(Math.round(bytes)));
  } catch { /* a private window: back to the automatic limit */ }
}

/** The ceiling a lost device has taught this browser for this card, if one has. */
export function learnedLimitBytes() {
  return readLearned();
}

/** The last loss this card could put down to memory, if it was recent enough to be the same trouble. */
function lossInTheLastHour(now) {
  try {
    const saved = JSON.parse(localStorage.getItem(LOSS_KEY) ?? "null");
    return saved?.name === (gpuName() ?? "unknown") && now - saved.at >= 0 && now - saved.at < SAME_TROUBLE_MS;
  } catch { return false; }
}

/**
 * Remember how much was on the card when its device was lost, the second time.
 *
 * The ceiling above is a guess from a name, and on an unlisted card from system RAM alone, and a device lost while holding
 * much of what it may hold can correct it. But one loss does not say why. Another program bringing up a device reset the
 * driver under a fold holding 2.8 GiB of an 8 GiB RTX 3070, and read as memory that held the card to 2.3 GiB for as long as
 * the browser remembered. A fold that does not fit fails again as soon as it is tried again, so it is the second loss in
 * an hour that sets a ceiling. Reading a loss cannot raise it, and `forgetLearnedLimit` is how a reader takes it back when
 * the cause turned out to be something else.
 *
 * @returns {number | undefined} the new ceiling, when this loss set one
 */
export function learnFromLoss(residentBytes, budgetBytes, now = Date.now()) {
  if (!lossTeachesLimit(residentBytes, budgetBytes)) return undefined;
  const second = lossInTheLastHour(now);
  const name = gpuName() ?? "unknown";
  const bytes = Math.min(Math.round(residentBytes * GPU_MEMORY_SHARE), readLearned() ?? Infinity);
  try {
    if (second) {
      localStorage.removeItem(LOSS_KEY);
      localStorage.setItem(LEARNED_KEY, JSON.stringify({ name, bytes }));
    } else {
      localStorage.setItem(LOSS_KEY, JSON.stringify({ name, at: now }));
    }
  } catch { /* private window: there is nothing to compare a loss with, so nothing is learned */ }
  return second ? bytes : undefined;
}

export function forgetLearnedLimit() {
  try { localStorage.removeItem(LEARNED_KEY); } catch { /* nothing kept */ }
}
