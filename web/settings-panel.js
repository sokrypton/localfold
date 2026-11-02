/**
 * The settings button in the page head: theme, where a fold runs, what the GPU holds
 * and what has been downloaded.
 *
 * Every number shown is read when the panel opens, so nothing polls while it is shut.
 */
import { clearShardCaches, shardCacheContents } from "../shared/bundles/http-tensor-store.js";
import { deviceSummary, releaseDevice } from "./model.js";
import { forgetLearnedLimit, gpuName, learnedLimitBytes } from "./gpu-limits.js";

const THEME_KEY = "localfold.theme";
const COMPUTE_KEY = "localfold.compute";

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
  return read(COMPUTE_KEY, "auto");
}

/** How many threads a CPU fold gets: the physical cores, so the machine stays usable. */
export function cpuThreads() {
  const logical = navigator.hardwareConcurrency ?? 4;
  return window.crossOriginIsolated === true ? Math.max(1, Math.ceil(logical / 2)) : 1;
}

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
    await navigator.serviceWorker.register("./coi-worker.js");
    await navigator.serviceWorker.ready;
    return true;
  }
  const registrations = await navigator.serviceWorker.getRegistrations();
  await Promise.all(registrations.map((registration) => registration.unregister()));
  return true;
}

/**
 * @param {{onReleaseGpu: () => void, onComputeChange?: (choice: string) => void,
 *          keepPage?: () => void}} hooks
 *   `onReleaseGpu` clears what the page holds outside the device; `keepPage` saves
 *   whatever must survive the reload that turns isolation on.
 */
export function installSettings({ onReleaseGpu, onComputeChange, keepPage }) {
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
    try { adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" }); } catch { /* none */ }
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
             describeCompute();
             void updateBadge();
             onComputeChange?.(choice);
             if (choice === "cpu" && window.crossOriginIsolated !== true) {
               // Threads need a load that happens under the worker's headers.
               computeNote.textContent = "Reloading once, so the fold can use every core…";
               keepPage?.();
               if (await setIsolated(true)) location.reload();
             } else if (choice !== "cpu" && window.crossOriginIsolated === true) {
               await setIsolated(false);
             }
           });
  describeCompute();

  const gpuValue = panel.querySelector("#settings-gpu-value");
  const limitValue = panel.querySelector("#settings-gpu-limit");
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
    limitValue.textContent = summary?.budgetBytes === undefined || summary?.budgetBytes === null
      ? "Set when a fold starts" : size(summary.budgetBytes);
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
  return { refresh: () => { void updateBadge(); if (!panel.hidden) void refreshGpu(); } };
}
