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

/** How much of a fold of four passes a fold of this many takes: the sampler and the confidence head run once. */
const passesShare = (loops) => 0.15 + 0.85 * (loops / 4);

/**
 * What a fold holds in host memory on a CPU, at its fullest: the weights as f32, what grows with the language model and
 * the atoms, and the pair-sized arrays. Fitted to the process's high-water mark under tools/fold-esmfold2-cpu.mjs, 2.83 GiB
 * at 76 residues, 4.93 at 300 and 9.97 at 600. The arrays are the plain ones the reference kernels hold, and more of them
 * than a block needs at once, because a fold keeps what it has finished with for the next kernel that asks.
 */
export const CPU_WEIGHT_BYTES = 2.44 * 1024 ** 3;
const CPU_BYTES_PER_TOKEN = 4.15 * 1024 ** 2;
export const CPU_PEAK_PAIRS = 15;
/** What a machine's RAM can give one fold: the browser, the desktop and the OS's own caches keep the rest. */
const RAM_SHARE = 0.7;

/**
 * Seconds for a fold of four loops on a card about as fast as the two measured, an RTX 3070 and an A10, which read the
 * same within 5% from 300 to 1200 residues. The cube is the triangle contraction. A card's own first fold replaces this
 * reference with a rate of its own: see `noteFold`.
 */
export function gpuFoldSeconds({ tokens, loops = 4 }) {
  return 1.5 + (4.94e-5 * tokens ** 2 + 3.29e-8 * tokens ** 3) * passesShare(loops);
}

/**
 * Seconds on a CPU, fitted to ubiquitin and its repeats at 1, 4 and 8 threads on an eight-CPU cloud machine: 54 s at 8
 * threads for 76 residues, 670 s for 300, and a thread count that stops paying as the memory bus fills.
 */
export function cpuFoldSeconds({ tokens, loops = 4, threads = 1 }) {
  return 54 * (tokens / 76) ** 1.8 * (0.786 + 1.714 / Math.max(1, threads)) * passesShare(loops);
}

export const cpuPeakBytes = ({ tokens, channels = 256 }) =>
  CPU_WEIGHT_BYTES + CPU_BYTES_PER_TOKEN * tokens + CPU_PEAK_PAIRS * pairBytes(tokens, channels);

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

/**
 * The other models on an A10 through the page, with no alignment: what the fold's own account reached in MiB, and the
 * seconds of a fold of four passes, at MEASURED_RESIDUES. Between two sizes the estimate is a straight line in n^2 for
 * memory and n^3 for time, and past the last the last line goes on. Each is named by its graph. AlphaFold 3's weights are
 * the reader's to accept, and OpenBind-0 is the same graph at the same widths (shared/bundles/manifests), so it stands in.
 */
const MEASURED_RESIDUES = [76, 300, 600];
const MEASURED_MODELS = {
  bioemu2: { peakMiB: [1524, 1856, 3117], seconds: [13.4, 14.9, 42.1] },
  boltz2: { peakMiB: [677, 1003, 1329], seconds: [4, 11, 38] },
  intellifold2: { peakMiB: [1215, 1515, 3377], seconds: [6, 47, 207] },
  mol1: { peakMiB: [934, 1387, 2443], seconds: [6, 24, 112] },
  monomer: { peakMiB: [388, 747, 1752], seconds: [3.2, 5.7, 19.1] },
  multimer: { peakMiB: [401, 1056, 2764], seconds: [2.8, 5.9, 20.4] },
  openbind0: { peakMiB: [512, 829, 1191], seconds: [4, 9, 30] },
  opendde: { peakMiB: [1114, 1524, 4846], seconds: [6, 36, 142] },
  protenix2: { peakMiB: [659, 935, 1550], seconds: [4, 15, 71] },
  rosettafold3: { peakMiB: [514, 833, 1196], seconds: [4, 9, 34] },
};
MEASURED_MODELS.af3 = MEASURED_MODELS.openbind0;

function along(tokens, values, power) {
  let from = 0;
  while (from < values.length - 2 && tokens > MEASURED_RESIDUES[from + 1]) from += 1;
  const [low, high] = [MEASURED_RESIDUES[from] ** power, MEASURED_RESIDUES[from + 1] ** power];
  return values[from] + (values[from + 1] - values[from]) * (tokens ** power - low) / (high - low);
}

/** `models`: what one press runs, which is five for AlphaFold 2's "all". Undefined for a model nobody measured. */
export function measuredFoldSeconds(family, { tokens, loops = 4, models = 1 }) {
  const measured = MEASURED_MODELS[family];
  return measured === undefined ? undefined : models * along(tokens, measured.seconds, 3) * passesShare(loops);
}

/**
 * A model that folds on a GPU only. This is an estimate for the reader and stops nothing: an alignment adds to it by an
 * amount a sequence cannot predict, and the fold's own account already refuses what does not fit.
 *
 * @param {string} family a graph: see MEASURED_MODELS
 * @param {{tokens: number, loops?: number, models?: number}} fold
 * @param {{preference?: "auto"|"cpu", gpu?: object, rates?: object}} machine as planEsmfold2's
 * @returns {{where: "estimate"|"gpu-only", why?: string, needBytes?: number, budgetBytes?: number, seconds?: number,
 *            over?: boolean}|undefined} undefined for a model nobody measured
 */
export function planMeasuredModel(family, { tokens, loops = 4, models = 1 }, { preference = "auto", gpu, rates = {} }) {
  const measured = MEASURED_MODELS[family];
  if (measured === undefined) return undefined;
  if (preference === "cpu") return { where: "gpu-only", why: "chosen" };
  if (gpu === undefined) return { where: "gpu-only", why: "no-gpu" };
  if (gpu.refusing === true) return { where: "gpu-only", why: "gpu-refusing" };
  if (gpu.software === true) return { where: "gpu-only", why: "software" };
  const needBytes = along(tokens, measured.peakMiB, 2) * 1024 ** 2;
  const seconds = measuredFoldSeconds(family, { tokens, loops, models }) * (rates.cards?.[gpu.name] ?? 1);
  const over = gpu.budgetBytes !== undefined && gpu.budgetBytes !== null && needBytes + FOLD_MARGIN_BYTES > gpu.budgetBytes;
  return { where: "estimate", needBytes, budgetBytes: gpu.budgetBytes, seconds, over };
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
    case "estimate": {
      const memory = plan.budgetBytes === undefined || plan.budgetBytes === null ? `${gib(plan.needBytes)} GiB`
        : `${gib(plan.needBytes)} of ${gib(plan.budgetBytes)} GiB`;
      const measured = "Measured with no alignment, which adds to this.";
      return plan.over
        ? { text: `Needs ${memory}`,
            title: `${measured} More than the page lets one fold hold on this card, which keeps the rest for the desktop`
              + " and other programs. Settings can change the limit.",
            tone: "warn" }
        : { text: `${memory} · ${roughDuration(plan.seconds)}`,
            title: `${measured} Settings says what the limit is and can change it.`,
            tone: "plain" };
    }
    case "gpu-only":
      return {
        text: "Folds on a GPU only",
        title: {
          chosen: "Settings has folds on the CPU, where only ESMFold2 runs. Settings: Fold on, GPU.",
          "no-gpu": "This browser offers no GPU. ESMFold2 folds on the CPU.",
          "gpu-refusing": "The GPU is not accepting work, and restarting the browser brings it back."
            + " ESMFold2 folds on the CPU.",
          software: "This browser's only GPU is a software one. ESMFold2 folds on the CPU.",
        }[plan.why],
        tone: "warn",
      };
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
 * @param {{where: "gpu"|"cpu", card?: string, tokens: number, loops: number, threads?: number, seconds: number,
 *          family?: string, models?: number}} fold `family`: the graph of a model other than ESMFold2. A card's rate is
 *   its own whichever model taught it, so a first fold of any of them improves the figures for all of them.
 */
export function noteFold({ where, card, tokens, loops, threads, seconds, family, models }, storage = globalThis.localStorage) {
  if (seconds < FOLD_WORTH_TIMING_S) return;
  const reference = family !== undefined ? measuredFoldSeconds(family, { tokens, loops, models })
    : where === "cpu" ? cpuFoldSeconds({ tokens, loops, threads }) : gpuFoldSeconds({ tokens, loops });
  if (reference === undefined) return;
  const rate = Math.min(RATE_RANGE[1], Math.max(RATE_RANGE[0], seconds / reference));
  const rates = readFoldRates(storage);
  const blend = (old) => (old === undefined ? rate : (old + rate) / 2);
  if (where === "cpu") rates.cpu = blend(rates.cpu);
  else rates.cards[card ?? ""] = blend(rates.cards[card ?? ""]);
  try { storage.setItem(RATES_KEY, JSON.stringify(rates)); } catch { /* a private window: this fold only */ }
}
