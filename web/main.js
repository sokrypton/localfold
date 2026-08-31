import { AlphaFoldQueryOnlyGpu } from "../src/model/query-only.js";
import { getDevice, loadModel } from "./model.js";
import { createStructureViewer } from "./viewer.js";
import { correspondence } from "./align.js";
import { morphFrames, superposeOnto } from "./morph.js";
import { confidenceJson, predictionToPdb, recyclesToPdb } from "./prediction-results.js";
import { createMutationPanel, mutationName, residueAt, substitute, wasClick } from "./mutate.js";
import { cleanSequence, complexSequenceProblem, sequenceChains } from "./sequence.js";
import { isAbortError, throwIfAborted } from "../src/runtime/abort.js";

const element = (id) => {
  const value = document.getElementById(id);
  if (value === null) throw new Error(`missing element #${id}`);
  return value;
};

const sequenceValue = () => cleanSequence(element("sequence").value);
const recycleCount = () => Number(document.getElementById("recycles")?.value) || 0;
const recycleTolerance = () => Number(document.getElementById("tolerance")?.value) || 0;
const randomSeed = () => {
  const input = document.getElementById("random-seed");
  if (input === null || input.value === "") return 0;
  const parsed = Number(input.value);
  return Number.isFinite(parsed) ? Math.max(0, Math.floor(parsed)) : 0;
};
const randomMasking = () => {
  const input = document.getElementById("random-masking");
  return input ? input.checked : false;
};
const blockOpmEnabled = () => {
  const select = document.getElementById("block-opm");
  return select ? select.value !== "off" : true;
};

let currentPdb = "";
let currentScores = "";
let currentSequence = "";
let currentRecycles = [];
let previousPrediction;
let morphedThisRun = false;
const MORPH_STEPS = 14;
const MORPH_MS = 900;
let viewerSequence = "";
let mutationPanel;
let activeFold;

function status(text, isError = false) {
  const node = document.getElementById("status-message");
  if (node === null) return;
  node.textContent = text;
  node.classList.toggle("error", isError);
}

function progress(fraction) {
  const bar = document.getElementById("progress");
  if (bar === null) return;
  if (fraction === null) {
    bar.dataset.state = "idle";
    bar.value = 0;
    return;
  }
  if (fraction === "waiting") {
    bar.dataset.state = "waiting";
    return;
  }
  bar.dataset.state = "running";
  bar.value = fraction;
}

function loadWeights(signal) {
  return loadModel("single", (value) => {
    if (signal?.aborted) return;
    progress(value.totalBytes === 0 ? 0 : value.loadedBytes / value.totalBytes);
    status(`Loading model · ${(value.loadedBytes / 1048576).toFixed(0)}`
      + ` / ${(value.totalBytes / 1048576).toFixed(0)} MiB`);
  }, signal);
}

const structure = createStructureViewer({ container: element("viewer"), canvasHeight: 480 });

function alignedToPrevious(sequence, predicted) {
  const api = window.py2Dmol;
  if (api?.superpose === undefined || previousPrediction === undefined) return predicted;
  try {
    const pairing = correspondence(sequence, previousPrediction.sequence);
    if (pairing.from.length < 3) return predicted;
    return superposeOnto(api, predicted, previousPrediction.structure, sequence.length, pairing);
  } catch (error) {
    console.warn("superposition skipped:", error);
    return predicted;
  }
}

function playMorph(sequence, landed, plddtTo, chainLengths = undefined) {
  const api = window.py2Dmol;
  if (api?.superpose === undefined || previousPrediction === undefined) return false;
  if (previousPrediction.sequence.length !== sequence.length) return false;
  if (!structure.built || structure.frames === 0) return false;
  try {
    const steps = morphFrames(previousPrediction.structure, landed,
      previousPrediction.plddt, plddtTo, sequence.length, MORPH_STEPS);
    const frames = api.framesFromText(recyclesToPdb(sequence, steps, chainLengths));
    if (frames.length !== steps.length) return false;
    const drawn = structure.renderer;
    const object = structure.object;
    const id = structure.generation;
    const lastFrame = structure.frames - 1;
    const started = performance.now();
    const tick = () => {
      if (!structure.built || structure.object !== object || structure.generation !== id) return;
      const through = Math.min(1, (performance.now() - started) / MORPH_MS);
      const at = Math.min(frames.length - 1, Math.round(through * (frames.length - 1)));
      drawn.replaceFrame(frames[at], object);
      structure.show(lastFrame, "morph");
      if (through < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    morphedThisRun = true;
    return true;
  } catch (error) {
    console.warn("morph skipped:", error);
    return false;
  }
}

function resetViewer() {
  structure.reset();
  morphedThisRun = false;
}

function pushFrame(sequence, recycle, chainLengths = undefined) {
  viewerSequence = sequence;
  const landed = alignedToPrevious(sequence, recycle.structure);
  const pdb = predictionToPdb(sequence, landed, recycle.confidence.plddt, chainLengths);
  const { built, frames } = structure.push(pdb, () => {
    element("orient").hidden = false;
  });
  if (frames === 0) return;
  if (built) playMorph(sequence, landed, recycle.confidence.plddt, chainLengths);
  structure.show(frames - 1, "recycle");
}

function stageMutation(indices, residue) {
  const positions = Array.isArray(indices) || indices instanceof Set ? Array.from(indices) : [indices];
  if (positions.length === 0) return;
  const currentSeq = sequenceValue();
  const baseSequence = currentSeq.length > 0
    ? currentSeq
    : (viewerSequence || "AAAAAAAAAA");
  const next = substitute(baseSequence, positions, residue);
  const name = mutationName(baseSequence, positions, residue);
  const box = element("sequence");
  box.value = next;
  box.dispatchEvent(new Event("input", { bubbles: true }));
  if (next === baseSequence) {
    status(positions.length > 1
      ? `Selected positions are already ${residue} — press Fold to re-run`
      : `${baseSequence[positions[0]]}${positions[0] + 1} is already ${residue} — press Fold to re-run`, false);
    return;
  }
  status(`${name} staged — press Fold to see it`, false);
}

mutationPanel = createMutationPanel(element("mutate-host"), stageMutation);

function restoreSelection() {
  const positions = mutationPanel.positions ?? [];
  if (positions.length === 0) return;
  const curSeq = (viewerSequence || sequenceValue()).replace(/:/g, "");
  const valid = positions.filter((at) => at >= 0 && at < curSeq.length);
  if (valid.length === 0) return;
  const drawn = structure.renderer;
  try { drawn?.select?.(valid); } catch { /* ignore */ }
  if (valid.length === 1) {
    const at = valid[0];
    const label = `${drawn?.positionNames?.[at] ?? "UNK"} ${drawn?.residueNumbers?.[at] ?? at + 1}`;
    mutationPanel.show(valid, curSeq[at] ?? "A", label);
  } else {
    const labels = valid.map((pos) => `${curSeq[pos] ?? "A"}${pos + 1}`).join(", ");
    const title = valid.length <= 4 ? labels : `${valid.length} residues`;
    const commonLetter = valid.every((pos) => curSeq[pos] === curSeq[valid[0]])
      ? curSeq[valid[0]]
      : "";
    mutationPanel.show(valid, commonLetter, title);
  }
}

const viewerContainer = element("viewer");
let pressedAt;
viewerContainer.addEventListener("pointerdown", (event) => {
  pressedAt = { x: event.clientX, y: event.clientY };
});
viewerContainer.addEventListener("click", (event) => {
  const click = wasClick(pressedAt, event);
  pressedAt = undefined;
  if (!click) return;
  if (!structure.built || !structure.renderer) return;
  const curSeq = (viewerSequence || sequenceValue()).replace(/:/g, "");
  const residue = residueAt(structure.renderer, curSeq.length, event.clientX, event.clientY);
  const drawn = structure.renderer;
  const isMulti = event.shiftKey || event.metaKey || event.ctrlKey;

  let selected = new Set(isMulti ? mutationPanel.positions : []);

  if (residue < 0) {
    if (!isMulti) {
      mutationPanel.hide();
      try { drawn?.select?.([]); } catch { /* ignore */ }
    }
    return;
  }

  if (isMulti) {
    if (selected.has(residue)) {
      selected.delete(residue);
    } else {
      selected.add(residue);
    }
  } else {
    selected = new Set([residue]);
  }

  if (selected.size === 0) {
    mutationPanel.hide();
    try { drawn?.select?.([]); } catch { /* ignore */ }
    return;
  }

  const positions = Array.from(selected).sort((a, b) => a - b);
  try { drawn?.select?.(positions); } catch { /* ignore */ }

  if (positions.length === 1) {
    const at = positions[0];
    const letter = curSeq[at] ?? "A";
    const label = `${drawn?.positionNames?.[at] ?? "UNK"} ${drawn?.residueNumbers?.[at] ?? at + 1}`;
    mutationPanel.show(positions, letter, label);
  } else {
    const labels = positions.map((pos) => `${curSeq[pos] ?? "A"}${pos + 1}`).join(", ");
    const title = positions.length <= 4 ? labels : `${positions.length} residues`;
    const commonLetter = positions.every((pos) => curSeq[pos] === curSeq[positions[0]])
      ? curSeq[positions[0]]
      : "";
    mutationPanel.show(positions, commonLetter, title);
  }
});

function showRecycle(sequence, recycle, index, total, chainLengths = undefined) {
  const iptmText = recycle.confidence.iptm !== undefined ? ` · ipTM ${recycle.confidence.iptm.toFixed(3)}` : "";
  status(`Pass ${index + 1} of ${total} · pLDDT ${recycle.confidence.meanPlddt.toFixed(1)} · pTM ${recycle.confidence.ptm.toFixed(3)}${iptmText}`);
  pushFrame(sequence, recycle, chainLengths);
}

function setFoldButton(state) {
  const button = element("predict");
  const running = state !== "idle";
  button.classList.toggle("btn-primary", !running);
  button.classList.toggle("btn-danger", running);
  button.disabled = state === "stopping";
  button.setAttribute("aria-label", running ? "Stop prediction" : "Start prediction");
  const icon = button.querySelector("i");
  if (icon !== null) icon.className = running ? "fa-solid fa-stop" : "fa-solid fa-cubes";
  const label = button.querySelector("span");
  if (label !== null) label.textContent = running ? "Stop" : "Fold";
}

async function fold(event) {
  event?.preventDefault();
  if (activeFold !== undefined) {
    activeFold.abort();
    setFoldButton("stopping");
    status("Stopping prediction…");
    return;
  }
  const controller = new AbortController();
  const { signal } = controller;
  activeFold = controller;
  setFoldButton("running");

  const entered = sequenceValue();
  const problem = complexSequenceProblem(entered);
  if (problem !== null) {
    status(problem, true);
    setFoldButton("idle");
    activeFold = undefined;
    return;
  }
  const chains = sequenceChains(entered);
  const sequence = chains.join("");
  const chainLengths = chains.map((c) => c.length);

  if (previousPrediction === undefined || previousPrediction.sequence.length !== sequence.length) {
    structure.forgetCamera();
  }

  const passes = recycleCount() + 1;
  const tolerance = recycleTolerance();
  const seed = randomSeed();
  const started = performance.now();
  currentRecycles = [];
  resetViewer();

  try {
    status("Getting WebGPU device…");
    progress(null);
    const device = await getDevice(signal);
    throwIfAborted(signal);

    const modelData = await loadWeights(signal);
    throwIfAborted(signal);

    const model = new AlphaFoldQueryOnlyGpu(device);
    const runProgress = ({ completed, total, waiting }) => {
      if (signal.aborted) return;
      if (waiting) {
        progress("waiting");
        status("Running the trunk on the GPU…");
        return;
      }
      progress(Math.min(1, completed / total));
      status(`Folding · ${Math.min(100, Math.round(100 * completed / total))}%`);
    };

    const prediction = await model.predictSequence(
      sequence,
      modelData.weights,
      modelData.featureTables,
      {
        recycles: recycleCount(),
        tolerance,
        randomSeed: seed,
        randomMasking: randomMasking(),
        chainLengths,
        blockOpm: blockOpmEnabled(),
        signal,
      },
      modelData.paeBreaks,
      (recycle, index) => {
        showRecycle(sequence, recycle, index, passes, chainLengths);
        currentRecycles.push(recycle);
      },
      runProgress,
    );

    const landed = alignedToPrevious(sequence, prediction.final.structure);
    previousPrediction = {
      sequence,
      structure: landed,
      plddt: prediction.final.confidence.plddt,
    };

    currentSequence = sequence;
    viewerSequence = entered;
    currentRecycles = prediction.recycles;

    if (structure.frames === 0) {
      for (const recycle of prediction.recycles) pushFrame(sequence, recycle, chainLengths);
    }

    currentPdb = recyclesToPdb(sequence, prediction.recycles, chainLengths);
    currentScores = confidenceJson(sequence, prediction.final.confidence);
    restoreSelection();
    const took = `${((performance.now() - started) / 1000).toFixed(1)} s`;
    const converged = prediction.recycles.length < passes
      ? ` · converged at ${prediction.final.recycleDistance.toFixed(2)} Å after ${prediction.recycles.length} passes`
      : "";
    const morphed = morphedThisRun ? " — showing what moved" : "";
    const plddt = ` · pLDDT ${prediction.final.confidence.meanPlddt.toFixed(1)}`;
    const ptm = ` · pTM ${prediction.final.confidence.ptm.toFixed(3)}`;
    const iptm = prediction.final.confidence.iptm !== undefined ? ` · ipTM ${prediction.final.confidence.iptm.toFixed(3)}` : "";
    status(`Done in ${took}${plddt}${ptm}${iptm}${converged}${morphed}`);
  } catch (error) {
    progress(null);
    if (signal.aborted || isAbortError(error)) status("Prediction stopped");
    else status(error instanceof Error ? error.message : String(error), true);
  } finally {
    if (activeFold === controller) activeFold = undefined;
    setFoldButton("idle");
  }
}

element("predict").addEventListener("click", (event) => void fold(event));
const sequenceBox = element("sequence");

const tidySequence = () => {
  const cleaned = cleanSequence(sequenceBox.value);
  if (cleaned !== sequenceBox.value) sequenceBox.value = cleaned;
};
sequenceBox.addEventListener("blur", tidySequence);
sequenceBox.addEventListener("paste", () => setTimeout(tidySequence, 0));

element("orient").addEventListener("click", () => {
  try {
    const renderer = structure.renderer;
    if (!renderer) return;
    if (window.py2dmolOrient) {
      window.py2dmolOrient.orientToBestView(renderer, { positions: [], animate: true });
    } else {
      renderer.orient?.({ positions: [] });
    }
  } catch (error) {
    console.warn("orient skipped:", error);
  }
});

void (async () => {
  const summary = element("gpu-summary");
  try {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null || adapter === undefined) throw new Error("no adapter");
    const name = adapter.info.description || adapter.info.device || adapter.info.vendor || "compatible GPU";
    summary.dataset.state = "ready";
    summary.textContent = `WebGPU · ${name}`;
  } catch {
    summary.dataset.state = "missing";
    summary.textContent = "WebGPU unavailable";
  }
})();
