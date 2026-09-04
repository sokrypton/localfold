/**
 * Protein Hunter, in the page.
 *
 * The method is `src/design/hunter-loop.js`, which knows nothing about a DOM
 * and takes its two models as arguments. This file is the other half: it reads
 * the controls, supplies a real AF3 fold and a real MPNN design, and puts every
 * cycle on screen as it lands.
 *
 * 🔴 THE FOLD IS SINGLE SEQUENCE AND THAT IS NOT A SIMPLIFICATION. Protein
 * Hunter's `--msa_mode mmseqs` builds an alignment for the TARGET, which needs
 * a remote search per run; the designed chain never has one, because it does
 * not exist yet. A hallucination loop that went to the network once per fold
 * would be a different tool. The target's alignment is the one thing this page
 * gives up against the reference, and the note is here rather than nowhere.
 *
 * 🔴 AND THE WEIGHTS LOAD ONCE PER PAGE, NOT ONCE PER FOLD. A run of five
 * cycles is six folds and a hunt of three runs is eighteen; re-reading a
 * quarter-gigabyte checkpoint between them would dwarf the inference. Both
 * loaders memoise - `loadAf3Weights` in web/af3-model.js by module state, the
 * designer by the promise held here.
 */
import { getDevice } from "./model.js";
import { createStructureViewer } from "./viewer.js";
import { AF3_COUNTS, af3SequenceProblem, foldAf3, loadAf3Weights } from "./af3-model.js";
import { runDesign } from "../src/design/hunter-loop.js";
import { designChain, loadDesigner } from "../src/design/mpnn-bridge.js";
import { isAbortError } from "../src/runtime/abort.js";

const element = (id) => {
  const value = document.getElementById(id);
  if (value === null) throw new Error(`missing element #${id}`);
  return value;
};

const number = (id, fallback) => {
  const value = Number(element(id).value);
  return Number.isFinite(value) ? value : fallback;
};

const viewer = createStructureViewer({ container: element("viewer"), canvasHeight: 480 });

/** Every record every run has yielded, in the order they arrived. */
let history = [];
let controller = null;
let designerPromise;

function status(text, isError = false) {
  const node = element("status-message");
  node.textContent = text;
  node.classList.toggle("error", isError);
}

function progress(fraction) {
  const bar = element("progress");
  bar.value = Math.max(0, Math.min(1, fraction));
  bar.dataset.state = fraction >= 1 ? "idle" : "running";
}

/**
 * The chains of the complex, with the designed one first.
 *
 * The reference fixes the design chain as "A" and the targets as B, C, ...,
 * and `toPdb` in src/af3/fold.js names chains in exactly that order - so the
 * designed chain being index 0 is what makes "A" mean the binder in the PDB
 * that MPNN then reads.
 */
function chainsFromControls() {
  const target = element("target").value.toUpperCase().replace(/[^A-Z:]/g, "");
  const targets = target.split(":").filter((chain) => chain.length > 0);
  const start = element("start-sequence").value.toUpperCase().replace(/[^A-Z]/g, "");
  return { chains: [start, ...targets], targets };
}

/** py2Dmol wants a fresh object per cycle: the sequence changes, so the atoms do. */
function show(record) {
  viewer.reset();
  viewer.push(record.pdb);
  viewer.paint();
  element("orient").hidden = false;
}

const percent = (value) => (Number.isFinite(value) ? `${(value * 100).toFixed(0)}%` : "–");
const fixed = (value, digits = 3) => (Number.isFinite(value) ? value.toFixed(digits) : "–");

/**
 * One row per cycle, in the shape of Protein Hunter's summary CSV.
 *
 * 🔴 THE BEST ROW IS RE-MARKED, NOT APPENDED TO. `runDesign` clears the flag on
 * the record it displaces, and the table has to follow - otherwise every cycle
 * that was ever best stays highlighted and the run appears to have several
 * answers.
 */
function renderTable() {
  const body = element("results-body");
  body.replaceChildren();
  for (const record of history) {
    const row = document.createElement("tr");
    if (record.best) row.className = "best";
    const cells = [
      String(record.run + 1),
      String(record.cycle),
      fixed(record.score),
      record.objective,
      fixed(record.meanPlddt, 1),
      fixed(record.iptm),
      percent(record.alanine),
      record.sequence,
    ];
    for (const [index, text] of cells.entries()) {
      const cell = document.createElement("td");
      cell.textContent = text;
      // The sequence is the only cell that can be long, and it is the one
      // worth copying, so it gets the monospace column and a title.
      if (index === cells.length - 1) {
        cell.className = "sequence";
        cell.title = record.complex;
      }
      row.append(cell);
    }
    row.addEventListener("click", () => show(record));
    body.append(row);
  }
  element("results").hidden = history.length === 0;
}

/** The best row of the whole hunt, across runs. */
function bestOverall() {
  let best = null;
  for (const record of history) {
    // 🔴 ACROSS RUNS ONLY WHEN THEY MEASURE THE SAME THING. Every run of one
    // hunt has the same chains, so they all select on the same objective; the
    // guard is here because a mixed comparison would be meaningless rather
    // than merely wrong, and it costs one condition.
    if (!record.best) continue;
    if (best === null || (record.objective === best.objective && record.score > best.score)) {
      best = record;
    }
  }
  return best;
}

function download(name, text, type = "text/plain") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = Object.assign(document.createElement("a"), { href: url, download: name });
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function csv() {
  const header = ["run", "cycle", "score", "objective", "mean_plddt", "iptm",
                  "ptm", "alanine", "sequence", "complex"];
  const rows = history.map((record) => [
    record.run + 1, record.cycle, record.score, record.objective,
    record.meanPlddt, record.iptm, record.ptm, record.alanine,
    record.sequence, record.complex,
  ]);
  return [header, ...rows]
    .map((row) => row.map((value) => (Number.isFinite(value) ? value : `${value ?? ""}`)).join(","))
    .join("\n") + "\n";
}

function setRunning(running) {
  const button = element("hunt");
  button.classList.toggle("btn-primary", !running);
  button.classList.toggle("btn-danger", running);
  button.querySelector("i").className = running ? "fa-solid fa-stop" : "fa-solid fa-crosshairs";
  button.querySelector("span").textContent = running ? "Stop" : "Hunt";
  for (const id of ["target", "start-sequence", "runs", "cycles", "percent-x",
                    "temperature", "seed", "steps", "alanine-bias", "omit",
                    "min-length", "max-length"]) {
    element(id).disabled = running;
  }
}

/** The designer, once per page. */
function designer() {
  designerPromise ??= loadDesigner({
    onProgress: (received, total) => {
      status(total > 0
        ? `Loading the designer · ${Math.round((received / total) * 100)}%`
        : "Loading the designer");
    },
  });
  return designerPromise;
}

async function hunt() {
  if (controller !== null) {
    controller.abort();
    return;
  }

  const { chains, targets } = chainsFromControls();
  // 🔴 THE TARGET IS CHECKED, THE BINDER IS NOT - IT DOES NOT EXIST YET. An
  // empty designed chain is the normal case and `runDesign` draws one; a
  // target with a stray letter in it would otherwise fold as a chain of blanks.
  for (const chain of targets) {
    const problem = af3SequenceProblem(chain);
    if (problem !== null) {
      status(problem, true);
      return;
    }
  }
  if (chains[0].length === 0 && number("min-length", 100) > number("max-length", 150)) {
    status("The binder's shortest length is longer than its longest.", true);
    return;
  }

  controller = new AbortController();
  const { signal } = controller;
  setRunning(true);
  history = [];
  renderTable();
  element("downloads").hidden = true;
  viewer.forgetCamera();

  const runs = Math.max(1, number("runs", 1));
  const cycles = Math.max(0, number("cycles", 5));
  const mode = "flow";
  const calls = number("steps", AF3_COUNTS[mode].preferred);
  const seed = number("seed", 0);
  const total = runs * (cycles + 1);
  const started = performance.now();

  try {
    status("Starting WebGPU");
    const device = await getDevice(signal);
    const weights = await loadAf3Weights((received, expected) => {
      status(`Loading AlphaFold 3 · ${Math.round((received / expected) * 100)}%`);
    });
    const { model } = await designer();

    for (let run = 0; run < runs; run += 1) {
      const iterator = runDesign({
        chains,
        chainIndex: 0,
        cycles,
        run,
        signal,
        // 🔴 ONE SEED PER RUN, NOT ONE PER HUNT. Every run shares the controls,
        // so a single seed would make three runs three copies of one run - and
        // "num_designs" would buy nothing at all.
        seed: seed + run,
        length: chains[0].length > 0 ? chains[0].length : undefined,
        minLength: number("min-length", 100),
        maxLength: number("max-length", 150),
        percentX: number("percent-x", 90),
        temperature: number("temperature", 0.1),
        omit: element("omit").value.toUpperCase().replace(/[^A-Z]/g, ""),
        alanineBias: element("alanine-bias").checked,

        fold: async (sequence, context) => {
          const done = context.run * (cycles + 1) + context.cycle;
          return foldAf3({
            sequence, mode, calls, recycles: 0, weights, device, signal, seed: seed + run,
            onStatus: (text) => {
              if (!signal.aborted) {
                status(`Run ${context.run + 1}/${runs} · cycle ${context.cycle}/${cycles} · ${text}`);
              }
            },
            // The bar spans the whole hunt: a fold's own fraction is the
            // fraction of one cell of it.
            onProgress: (fraction) => {
              if (!signal.aborted) progress((done + fraction) / total);
            },
          });
        },

        design: (pdb, context) => {
          status(`Run ${context.run + 1}/${runs} · cycle ${context.cycle}/${cycles} · designing`);
          return designChain(model, {
            pdb,
            chain: context.chain,
            temperature: context.temperature,
            omit: context.omit,
            alanineBias: context.alanineBias,
            random: context.random,
          });
        },
      });

      for await (const record of iterator) {
        history.push(record);
        show(record);
        renderTable();
      }
    }

    const best = bestOverall();
    element("downloads").hidden = best === null;
    progress(1);
    const took = ((performance.now() - started) / 1000).toFixed(0);
    status(best === null
      ? `Done in ${took} s · nothing under the alanine ceiling`
      : `Done in ${took} s · best ${best.objective} ${fixed(best.score)}`
        + ` (run ${best.run + 1}, cycle ${best.cycle})`);
  } catch (error) {
    progress(0);
    if (isAbortError(error)) {
      element("downloads").hidden = bestOverall() === null;
      status(`Stopped after ${history.length} ${history.length === 1 ? "cycle" : "cycles"}`);
    } else {
      console.error(error);
      status(error.message, true);
    }
  } finally {
    controller = null;
    setRunning(false);
  }
}

element("hunt").addEventListener("click", () => { void hunt(); });

element("download-fasta").addEventListener("click", () => {
  const best = bestOverall();
  if (best === null) return;
  const header = `>design run${best.run + 1}_cycle${best.cycle}`
    + ` ${best.objective}=${fixed(best.score)} plddt=${fixed(best.meanPlddt, 1)}`;
  download("protein-hunter.fasta", `${header}\n${best.sequence}\n`, "text/plain");
});

element("download-pdb").addEventListener("click", () => {
  const best = bestOverall();
  if (best === null) return;
  download("protein-hunter.pdb", best.pdb, "chemical/x-pdb");
});

element("download-csv").addEventListener("click", () => {
  download("protein-hunter.csv", csv(), "text/csv");
});

element("orient").addEventListener("click", () => {
  const renderer = viewer.renderer;
  if (renderer === undefined) return;
  try {
    if (window.py2dmolOrient) {
      window.py2dmolOrient.orientToBestView(renderer, { positions: [], animate: true });
    } else {
      renderer.orient?.({ positions: [] });
    }
  } catch (error) {
    console.warn("orient skipped:", error);
  }
});

// A binder with a starting sequence has a length already; the range is for the
// case where it does not, and showing both at once invites setting one that is
// silently ignored.
const syncLengthControls = () => {
  const given = element("start-sequence").value.replace(/[^A-Za-z]/g, "").length > 0;
  element("length-range").hidden = given;
};
element("start-sequence").addEventListener("input", syncLengthControls);
syncLengthControls();

void (async () => {
  const summary = element("gpu-summary");
  try {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null || adapter === undefined) throw new Error("no adapter");
    const name = adapter.info.description || adapter.info.device
      || adapter.info.vendor || "compatible GPU";
    summary.dataset.state = "ready";
    summary.textContent = `WebGPU · ${name}`;
  } catch {
    summary.dataset.state = "missing";
    summary.textContent = "WebGPU unavailable";
  }
})();
