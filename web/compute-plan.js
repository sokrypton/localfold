/**
 * Where a fold runs and what it will cost, worked out from the sequence before anything is downloaded.
 *
 * A page that learns a fold does not fit when an allocation is refused has by then downloaded a model and spent a minute,
 * and a card that is already full loses its device instead of refusing. So the reader's one question - will this run
 * here, and for how long - is answered beside the button, from the numbers the fold itself works to:
 * webgpu/esmfold2/memory-plan.js for what it holds and webgpu/runtime/device-memory.js for what it may hold.
 *
 * Nothing here touches a device or the DOM. A plan is advice to the page's preflight, and the fold's own account stays
 * the authority once it is running.
 */
import { esmfold2PeakBytes, esmfold2TokenCeiling, FOLD_MARGIN_BYTES, FOLD_WEIGHT_BYTES, FOLD_WEIGHT_BYTES_F32, pairBytes }
  from "../webgpu/esmfold2/memory-plan.js";

const RATES_KEY = "localfold.fold-rates";

/**
 * What a fold holds in host memory on a CPU, at its fullest.
 *
 * Measured with tools/fold-esmfold2-cpu.mjs: see the CPU fold's record. Every pair-sized tensor the GPU's account above
 * keeps apart lives here at once, because the reference kernels hold them as plain arrays, beside the weights as f32.
 */
export const CPU_WEIGHT_BYTES = 2.5 * 1024 ** 3;
export const CPU_PEAK_PAIRS = 6;
/** What a machine's RAM can give one fold: the browser, the desktop and the OS's own caches keep the rest. */
const RAM_SHARE = 0.7;

/**
 * Seconds for a fold of four loops on a card about as fast as the two measured, an RTX 3070 and an A10, which read the
 * same within 5% from 300 to 1200 residues. The cube is the triangle contraction. A card's own first fold replaces this
 * reference with a rate of its own: see `noteFold`.
 */
export function gpuFoldSeconds({ tokens, loops = 4 }) {
  const trunkShare = 0.15 + 0.85 * (loops / 4);
  return 1.5 + (4.94e-5 * tokens ** 2 + 3.29e-8 * tokens ** 3) * trunkShare;
}

/**
 * Seconds on a CPU, fitted to ubiquitin and its repeats at 1, 4 and 8 threads on an eight-CPU cloud machine: 54 s at 8
 * threads for 76 residues, 670 s for 300, and a thread count that stops paying as the memory bus fills.
 */
export function cpuFoldSeconds({ tokens, loops = 4, threads = 1 }) {
  const trunkShare = 0.15 + 0.85 * (loops / 4);
  return 54 * (tokens / 76) ** 1.8 * (0.786 + 1.714 / Math.max(1, threads)) * trunkShare;
}

export const cpuPeakBytes = ({ tokens, channels = 256 }) =>
  CPU_PEAK_PAIRS * pairBytes(tokens, channels) + CPU_WEIGHT_BYTES;

/**
 * @param {object} fold `tokens`, `loops`
 * @param {object} machine
 *   `preference`: "auto" or "cpu"
 *   `gpu`: undefined where there is no adapter, otherwise `{name, budgetBytes, maxBindingBytes, halfWeights, software,
 *   refusing}`; `refusing` is a device request that has failed since the page loaded
 *   `threads`, `ramBytes`, `rates`
 * @returns {{where: "gpu"|"cpu"|"over"|"none", why?: string, needBytes?: number, budgetBytes?: number, spills?: boolean,
 *            seconds?: number, cpuSeconds?: number, cpuFits: boolean, threads: number}}
 *   `over`: the GPU cannot hold it, and the reader decides; `none`: nothing here can.
 */
export function planEsmfold2({ tokens, loops = 4 }, { preference = "auto", gpu, threads = 1, ramBytes, rates = {} }) {
  const cpuFits = ramBytes === undefined || cpuPeakBytes({ tokens }) <= ramBytes * RAM_SHARE;
  const cpuSeconds = cpuFoldSeconds({ tokens, loops, threads }) * (rates.cpu ?? 1);
  const onCpu = (why) => (cpuFits ? { where: "cpu", why, seconds: cpuSeconds, cpuSeconds, cpuFits, threads }
    : { where: "none", why: "ram", needBytes: cpuPeakBytes({ tokens }), cpuFits, threads });
  if (preference === "cpu") return onCpu("chosen");
  if (gpu === undefined) return onCpu("no-gpu");
  if (gpu.refusing === true) return onCpu("gpu-refusing");
  // A software adapter is the CPU with extra steps, and without the threaded kernels the CPU fold is built on.
  if (gpu.software === true) return onCpu("software");

  const weightBytes = gpu.halfWeights === false ? FOLD_WEIGHT_BYTES_F32 : FOLD_WEIGHT_BYTES;
  const whole = esmfold2PeakBytes({ tokens, weightBytes });
  const least = esmfold2PeakBytes({ tokens, weightBytes, spill: true });
  const over = gpu.budgetBytes !== undefined && gpu.budgetBytes !== null
    && least + FOLD_MARGIN_BYTES > gpu.budgetBytes;
  const binding = gpu.maxBindingBytes !== undefined
    && tokens > esmfold2TokenCeiling({ maxBindingBytes: gpu.maxBindingBytes });
  const spills = (gpu.maxBindingBytes !== undefined && pairBytes(tokens) > gpu.maxBindingBytes)
    || (gpu.budgetBytes !== undefined && gpu.budgetBytes !== null && whole + FOLD_MARGIN_BYTES > gpu.budgetBytes);
  const needBytes = spills ? least : whole;
  const seconds = gpuFoldSeconds({ tokens, loops }) * (rates.cards?.[gpu.name] ?? 1);
  if (binding || over) {
    return { where: "over", why: binding ? "binding" : "memory", needBytes, budgetBytes: gpu.budgetBytes, seconds, cpuSeconds,
             cpuFits, threads };
  }
  return { where: "gpu", needBytes, budgetBytes: gpu.budgetBytes, spills, seconds, cpuSeconds, cpuFits, threads };
}

const gib = (bytes) => (bytes / 1024 ** 3).toFixed(1);

/** "about 40 s", "about 3 min", "about 1.5 h": as rough as a figure that has not met this machine should read. */
export function roughDuration(seconds) {
  if (seconds < 20) return `about ${Math.max(1, Math.round(seconds))} s`;
  if (seconds < 55) return `about ${Math.round(seconds / 5) * 5} s`;
  if (seconds < 50 * 60) return `about ${Math.max(1, Math.round(seconds / 60))} min`;
  const hours = seconds / 3600;
  return `about ${hours < 10 ? String(Math.round(hours * 2) / 2) : Math.round(hours)} h`;
}

/**
 * What the line beside the button says.
 *
 * @returns {{text: string, title: string, tone: "plain"|"warn"}}
 */
export function describePlan(plan) {
  const threads = `${plan.threads} CPU thread${plan.threads === 1 ? "" : "s"}`;
  const cpuOption = plan.cpuFits ? ` · CPU ${roughDuration(plan.cpuSeconds)}` : "";
  switch (plan.where) {
    case "gpu": {
      const memory = plan.budgetBytes === undefined || plan.budgetBytes === null ? `${gib(plan.needBytes)} GiB`
        : `${gib(plan.needBytes)} of ${gib(plan.budgetBytes)} GiB`;
      return {
        text: `${memory} · ${roughDuration(plan.seconds)}`,
        title: plan.spills
          ? "Near this card's limit, so the fold keeps two pair-sized tensors in system memory between stages."
            + " Settings says what the limit is and can change it."
          : "What this fold will hold on the GPU, what the page lets one fold hold, and about how long it takes here."
            + " Settings says what the limit is and can change it.",
        tone: "plain",
      };
    }
    case "cpu":
      return {
        text: `${threads} · ${roughDuration(plan.seconds)}`,
        title: {
          chosen: "Settings has this fold on the CPU.",
          "no-gpu": "This browser offers no GPU, so ESMFold2 folds on the CPU.",
          "gpu-refusing": "The GPU is not accepting work, so ESMFold2 folds on the CPU."
            + " Restarting the browser brings the GPU back.",
          software: "This browser's only GPU is a software one, so ESMFold2 folds on the CPU, which is faster.",
        }[plan.why],
        tone: "plain",
      };
    case "over":
      return plan.why === "binding"
        ? { text: `Too long for WebGPU${cpuOption}`,
            title: "Past the longest fold WebGPU's buffer bindings can address, on any card.",
            tone: "warn" }
        : { text: `Needs ${gib(plan.needBytes)} of ${gib(plan.budgetBytes)} GiB${cpuOption}`,
            title: "More than the page lets one fold hold on this card, which keeps the rest for the desktop and other"
              + " programs. Settings can change the limit.",
            tone: "warn" };
    default:
      return { text: `Needs ${gib(plan.needBytes)} GiB of system memory`,
               title: "More than this machine can spare for a CPU fold.", tone: "warn" };
  }
}

/** The same, in a sentence, for a fold that has been refused and offers a way on. */
export function describeRefusal(plan) {
  if (plan.where === "none") return `This fold needs ${gib(plan.needBytes)} GiB of system memory, more than this machine can spare.`;
  if (plan.why === "binding") return "This fold is past the longest WebGPU's buffer bindings can address, on any card.";
  return `This fold needs ${gib(plan.needBytes)} GiB, and the page lets one fold hold ${gib(plan.budgetBytes)} GiB on this card.`;
}

/** What this browser has measured of its own machine: how each card and the CPU run against the references above. */
export function readFoldRates(storage = globalThis.localStorage) {
  try {
    const saved = JSON.parse(storage.getItem(RATES_KEY) ?? "{}");
    return { cpu: Number.isFinite(saved.cpu) ? saved.cpu : undefined, cards: saved.cards ?? {} };
  } catch { return { cards: {} }; }
}

const FOLD_WORTH_TIMING_S = 3;
const RATE_RANGE = [0.02, 50];

/**
 * Learn from a fold that ran to its end.
 *
 * A short fold says little, because its constant is most of it. The average is an even one, so a machine that was busy
 * once does not set its figure for good.
 *
 * @param {{where: "gpu"|"cpu", card?: string, tokens: number, loops: number, threads?: number, seconds: number}} fold
 */
export function noteFold({ where, card, tokens, loops, threads, seconds }, storage = globalThis.localStorage) {
  if (seconds < FOLD_WORTH_TIMING_S) return;
  const reference = where === "cpu" ? cpuFoldSeconds({ tokens, loops, threads }) : gpuFoldSeconds({ tokens, loops });
  const measured = Math.min(RATE_RANGE[1], Math.max(RATE_RANGE[0], seconds / reference));
  const rates = readFoldRates(storage);
  const blend = (old) => (old === undefined ? measured : (old + measured) / 2);
  if (where === "cpu") rates.cpu = blend(rates.cpu);
  else rates.cards[card ?? ""] = blend(rates.cards[card ?? ""]);
  try { storage.setItem(RATES_KEY, JSON.stringify(rates)); } catch { /* a private window: this fold only */ }
}
