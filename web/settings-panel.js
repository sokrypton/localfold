/**
 * The settings button in the page head: theme, where a fold runs, what the GPU holds
 * and what has been downloaded.
 *
 * Every number shown is read when the panel opens, so nothing polls while it is shut.
 */
import { clearShardCaches, shardCacheContents } from "../shared/bundles/http-tensor-store.js";
import { applyMemoryLimit, askForAdapter, deviceSummary, planningGpu, releaseDevice } from "./model.js";
import { chosenLimitBytes, forgetLearnedLimit, gpuName, learnedLimitBytes, setChosenLimitBytes } from "./gpu-limits.js";

const THEME_KEY = "localfold.theme";
const COMPUTE_KEY = "localfold.compute";
/** The CPU for as long as this tab lasts: a GPU that is missing or refusing today is no choice to remember. */
const COMPUTE_NOW_KEY = "localfold.compute-now";

const read = (key, fallback) => {
  try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; }
};
const write = (key, value) => {
  try { localStorage.setItem(key, value); } catch { /* a private window: this page only */ }
};

const mib = (bytes) => `${Math.round(bytes / 1048576)} MiB`;
const size = (bytes) => (bytes >= 1073741824 ? `${(bytes / 1073741824).toFixed(1)} GiB` : mib(bytes));

/** The same rule the head's inline script applies before first paint. */
function applyTheme(choice) {
  const dark = choice === "dark" || (choice !== "light"
    && window.matchMedia?.("(prefers-color-scheme: dark)").matches === true);
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  document.dispatchEvent(new CustomEvent("localfold-theme"));
}

/** "auto" or "cpu". Auto is the GPU: the CPU is only ever asked for. */
export function computePreference() {
  try {
    if (sessionStorage.getItem(COMPUTE_NOW_KEY) === "cpu") return "cpu";
  } catch { /* a private window */ }
  return read(COMPUTE_KEY, "auto");
}

/**
 * Fold on the CPU without changing what Settings remembers.
 *
 * @param {() => void} [keepPage] saves what must survive the reload that turns isolation on
 * @returns {Promise<boolean>} true if the page is reloading for it
 */
export async function useCpuForNow(keepPage) {
  try { sessionStorage.setItem(COMPUTE_NOW_KEY, "cpu"); } catch { /* a private window */ }
  document.dispatchEvent(new CustomEvent("localfold-compute"));
  if (window.crossOriginIsolated === true) return false;
  keepPage?.();
  try {
    if (await setIsolated(true)) {
      location.reload();
      return true;
    }
  } catch (cause) { console.warn("isolation", cause); }
  return false;
}

/** The physical cores, so the machine stays usable, and no more than memory can feed. */
const MOST_CPU_THREADS = 32;
const coresToUse = () => Math.min(MOST_CPU_THREADS, Math.max(1, Math.ceil((navigator.hardwareConcurrency ?? 4) / 2)));

/** How many threads a CPU fold gets. */
export function cpuThreads() {
  return window.crossOriginIsolated === true ? coresToUse() : 1;
}

/**
 * When the page last reloaded to turn isolation on, so that one it did not isolate does not try for ever. A time and not a
 * flag: a tab's sessionStorage does not follow it across the change of browsing context group that isolation is, so a hard
 * reload back out of an isolated page finds the mark that page cleared in its own copy.
 */
const ISOLATION_TRIED_KEY = "localfold.isolation-tried";
const ISOLATION_RELOAD_MS = 30 * 1000;
/** How long the worker gets to start. One that fails to install leaves `serviceWorker.ready` waiting for ever. */
const ISOLATION_WAIT_MS = 5 * 1000;

/**
 * A browser that will not register the worker (a private window in some), or one where the worker did not give this page
 * its headers, keeps it on one thread.
 */
let isolationRefused = (() => {
  try {
    const tried = Number(sessionStorage.getItem(ISOLATION_TRIED_KEY));
    sessionStorage.removeItem(ISOLATION_TRIED_KEY);
    return Date.now() - tried < ISOLATION_RELOAD_MS && window.crossOriginIsolated !== true;
  } catch { return false; }
})();

/** How many a CPU fold would get, for a plan: a page that is not isolated yet becomes so on the way to one. */
export function cpuThreadsOnceIsolated() {
  return "serviceWorker" in navigator && !isolationRefused ? coresToUse() : cpuThreads();
}

const LIMIT_CHOICES_GIB = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 32, 40, 48, 64, 80];

/** The card's marketing name out of a WebGL renderer string. */
function shortName(name) {
  if (name === null || name === undefined) return "";
  if (/SwiftShader|llvmpipe|Basic Render/i.test(name)) return "Software renderer";
  const angle = /^ANGLE \([^,]*,\s*([^,(]+)/.exec(name);
  return (angle?.[1] ?? name).replace(/\b(NVIDIA|AMD|Intel\(R\)|GeForce|Radeon\(TM\))\s+/g, "").trim();
}

function segments(host, options, current, onChoose) {
  host.replaceChildren();
  for (const [value, label] of options) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.setAttribute("aria-pressed", String(value === current));
    button.addEventListener("click", () => {
      for (const other of host.children) other.setAttribute("aria-pressed", String(other === button));
      onChoose(value);
    });
    host.append(button);
  }
}

/**
 * Isolate the page so a CPU fold can use every core.
 *
 * A SharedArrayBuffer needs COOP and COEP headers, and GitHub Pages sends neither, so
 * a service worker adds them. It is registered only once somebody chooses the CPU, and
 * removed when they choose otherwise: nobody else's page changes behaviour. It takes
 * effect on the next load, so the caller decides when that is.
 */
export async function setIsolated(wanted) {
  if (!("serviceWorker" in navigator)) return false;
  if (wanted) {
    if (isolationRefused) return false;
    try {
      await navigator.serviceWorker.register("./coi-worker.js");
      await Promise.race([navigator.serviceWorker.ready, new Promise((_, reject) => {
        setTimeout(() => reject(new Error("the service worker did not start")), ISOLATION_WAIT_MS);
      })]);
    } catch (cause) {
      isolationRefused = true;
      throw cause;
    }
    try { sessionStorage.setItem(ISOLATION_TRIED_KEY, String(Date.now())); } catch { /* a private window: it may try again */ }
    return true;
  }
  const registrations = await navigator.serviceWorker.getRegistrations();
  await Promise.all(registrations.map((registration) => registration.unregister()));
  return true;
}

/**
 * @param {{onReleaseGpu: () => void, onComputeChange?: (choice: string) => void,
 *          onLimitChange?: () => void, keepPage?: () => void}} hooks
 *   `onReleaseGpu` clears what the page holds outside the device; `keepPage` saves
 *   whatever must survive the reload that turns isolation on.
 */
export function installSettings({ onReleaseGpu, onComputeChange, onLimitChange, keepPage }) {
  const toggle = document.getElementById("settings-toggle");
  const panel = document.getElementById("settings-panel");
  const badge = document.getElementById("device-badge");
  if (toggle === null || panel === null) return { refresh: () => {} };

  segments(panel.querySelector("#settings-theme"), [["auto", "Auto"], ["light", "Light"], ["dark", "Dark"]],
           read(THEME_KEY, "auto"), (choice) => { write(THEME_KEY, choice); applyTheme(choice); });
  window.matchMedia?.("(prefers-color-scheme: dark)").addEventListener?.("change", () => {
    if (read(THEME_KEY, "auto") === "auto") applyTheme("auto");
  });

  const computeNote = panel.querySelector("#settings-compute-note");
  const describeCompute = () => {
    const threads = cpuThreads();
    computeNote.textContent = computePreference() === "cpu"
      ? `ESMFold2 folds on ${threads} CPU thread${threads === 1 ? "" : "s"}, in a worker, and the other`
        + " models are unavailable. Slower than a GPU, and it cannot run out of GPU memory."
      : "Folds on the GPU. CPU runs ESMFold2 only, for when there is no GPU or it is too small.";
  };
  const updateBadge = async () => {
    if (computePreference() === "cpu") {
      const threads = cpuThreads();
      badge.textContent = `CPU · ${threads} thread${threads === 1 ? "" : "s"}`;
      delete toggle.dataset.state;
      return;
    }
    let adapter = null;
    try { adapter = navigator.gpu === undefined ? null : await askForAdapter(); } catch { /* none, or none that answers */ }
    if (adapter === null || adapter === undefined) {
      badge.textContent = "No GPU found";
      toggle.dataset.state = "missing";
      return;
    }
    delete toggle.dataset.state;
    badge.textContent = adapter.info?.isFallbackAdapter === true ? "GPU · software (slow)"
      : `GPU · ${shortName(gpuName()) || adapter.info?.vendor || "ready"}`;
  };
  segments(panel.querySelector("#settings-compute"), [["auto", "GPU"], ["cpu", "CPU"]], computePreference(),
           async (choice) => {
             write(COMPUTE_KEY, choice);
             try { sessionStorage.removeItem(COMPUTE_NOW_KEY); } catch { /* a private window */ }
             describeCompute();
             void updateBadge();
             onComputeChange?.(choice);
             if (choice === "cpu" && window.crossOriginIsolated !== true) {
               // Threads need a load that happens under the worker's headers.
               computeNote.textContent = "Reloading once, so the fold can use every core…";
               keepPage?.();
               try {
                 if (await setIsolated(true)) { location.reload(); return; }
               } catch (cause) { console.warn("isolation", cause); }
               describeCompute();
               onComputeChange?.(choice);
             } else if (choice !== "cpu" && window.crossOriginIsolated === true) {
               await setIsolated(false);
             }
           });
  describeCompute();

  const gpuValue = panel.querySelector("#settings-gpu-value");
  const limitChoice = panel.querySelector("#settings-gpu-limit");
  const limitNote = panel.querySelector("#settings-gpu-limit-note");
  const gpuName_ = panel.querySelector("#settings-gpu-name");
  const release = panel.querySelector("#settings-gpu-release");
  const forget = panel.querySelector("#settings-gpu-forget");
  const modelList = panel.querySelector("#settings-models");
  const modelTotal = panel.querySelector("#settings-models-total");
  const clear = panel.querySelector("#settings-models-clear");

  const refreshGpu = async () => {
    const summary = await deviceSummary();
    gpuName_.textContent = toggle.dataset.state === "missing" && summary === undefined ? "None found"
      : shortName(summary?.name ?? gpuName()) || "Not reported";
    gpuValue.textContent = summary === undefined ? "Nothing held"
      : `${size(summary.residentBytes)} held, ${size(summary.peakBytes)} at peak`;
    const gpu = await planningGpu();
    limitChoice.disabled = gpu === undefined;
    const chosen = chosenLimitBytes();
    // Whole gibibytes up to the card, and whatever was chosen on a card that has since changed.
    const card = gpu?.memoryBytes ?? 32 * 1073741824;
    const sizes = LIMIT_CHOICES_GIB.map((gib) => gib * 1073741824).filter((bytes) => bytes <= card);
    if (chosen !== undefined && !sizes.includes(chosen)) sizes.push(chosen);
    limitChoice.replaceChildren(
      new Option(gpu === undefined ? "Automatic" : `Automatic: ${size(gpu.automaticBytes)}`, ""),
      ...sizes.sort((a, b) => a - b).map((bytes) => new Option(size(bytes), String(bytes))),
    );
    limitChoice.value = chosen === undefined ? "" : String(chosen);
    limitNote.textContent = gpu === undefined ? ""
      : "Leaves the rest of the card to the desktop and other programs. No browser can see how much of it they are"
        + " using, so lower this if they need more, and raise it if nothing else is running.";
    release.disabled = summary === undefined;
    forget.hidden = learnedLimitBytes() === undefined;
  };
  const refreshModels = async () => {
    const contents = await shardCacheContents().catch(() => []);
    modelList.replaceChildren(...contents.map(({ bundle, bytes }) => {
      const row = document.createElement("li");
      row.className = "settings-row";
      const name = document.createElement("span");
      name.textContent = bundle;
      const value = document.createElement("span");
      value.className = "settings-value";
      value.textContent = mib(bytes);
      row.append(name, value);
      return row;
    }));
    const total = contents.reduce((sum, { bytes }) => sum + bytes, 0);
    modelTotal.textContent = contents.length === 0 ? "None yet" : size(total);
    clear.disabled = contents.length === 0;
  };

  limitChoice.addEventListener("change", async () => {
    setChosenLimitBytes(limitChoice.value === "" ? undefined : Number(limitChoice.value));
    await applyMemoryLimit();
    await refreshGpu();
    onLimitChange?.();
  });
  release.addEventListener("click", async () => {
    onReleaseGpu();
    await releaseDevice();
    await refreshGpu();
  });
  forget.addEventListener("click", async () => {
    forgetLearnedLimit();
    await refreshGpu();
  });
  clear.addEventListener("click", async () => {
    await clearShardCaches();
    await refreshModels();
  });

  const setOpen = (open) => {
    panel.hidden = !open;
    toggle.setAttribute("aria-expanded", String(open));
    if (open) { void refreshGpu(); void refreshModels(); }
  };
  toggle.addEventListener("click", () => setOpen(panel.hidden));
  document.addEventListener("click", (event) => {
    if (!panel.hidden && !panel.contains(event.target) && !toggle.contains(event.target)) setOpen(false);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !panel.hidden) { setOpen(false); toggle.focus(); }
  });
  void updateBadge();
  document.addEventListener("localfold-compute", () => { describeCompute(); void updateBadge(); });
  return { refresh: () => { void updateBadge(); if (!panel.hidden) void refreshGpu(); } };
}
