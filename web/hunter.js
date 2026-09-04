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
 * loaders memoise - `loadAf3Weights` in web/af3-model.js by module state, and
 * `loadDesigner` per MPNN family, which is also why switching the picker back
 * and forth costs nothing after the first read of each.
 */
import { getDevice } from "./model.js";
import { createStructureViewer } from "./viewer.js";
import { AF3_COUNTS, af3SequenceProblem, foldAf3, loadAf3Weights } from "./af3-model.js";
import { runDesign } from "../src/design/hunter-loop.js";
import { designChain, loadDesigner } from "../src/design/mpnn-bridge.js";
import { DESIGNERS, DESIGNER_NAMES, chooseDesigner } from "../src/design/designers.js";
import { createEntityList } from "./entity-ui.js";
import { superposeCycle } from "../src/design/superpose-pdb.js";
import { followActiveFrame, updateScoresCard } from "./scores-card.js";
import { NUCLEIC_TYPES, entitiesProblem, expandEntities } from "./entities.js";
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

const viewer = createStructureViewer({
  container: element("viewer"), canvasHeight: 480, frameLabel: "cycle",
});

/**
 * The target, as index.html's own entity rows.
 *
 * 🔴 IT SHOWS ONE EMPTY ROW, AND `initial: []` DOES NOT MEAN NO ROW.
 * createEntityList falls back to a single protein row when it is handed an
 * empty list, which is right: a control that starts as nothing but a button
 * is a control people do not find. What "empty" means here is the row's
 * VALUE - a blank sequence is a monomer hallucination, which is the method's
 * own first example, so chainsFromControls drops blank rows rather than
 * refusing them.
 */
const targets = createEntityList(element("entity-rows"), element("add-entity"), {
  onChange: () => syncDesignerNote(),
});

/** Every record every run has yielded, in the order they arrived. */
let history = [];
let controller = null;

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
  const entities = targets.read();
  // 🔴 AN EMPTY LIST IS A MONOMER HALLUCINATION, WHICH entitiesProblem CALLS AN
  // ERROR. It is right to for index.html - a fold of nothing is a mistake
  // there - and wrong here, where "design me a protein" is the method's own
  // first example. So the empty case is answered before asking.
  const filled = entities.filter((entity) => entity.value.trim().length > 0);
  const expanded = filled.length === 0
    ? { chains: [], chainKinds: [], ligandCodes: [], modifications: [] }
    : expandEntities(filled);
  const start = element("start-sequence").value.toUpperCase().replace(/[^A-Z]/g, "");
  return {
    // The designed chain is always protein and always first, which is what
    // makes "A" mean the binder in the PDB that MPNN then reads.
    chains: [start, ...expanded.chains],
    chainKinds: ["protein", ...expanded.chainKinds],
    // 🔴 THE MODIFICATIONS' CHAIN INDEX SHIFTS BY ONE. expandEntities numbers
    // them against its own chain list, and the binder is inserted in front of
    // it - so a phosphoserine on the target's chain 0 belongs to chain 1 here.
    // The featuriser reads this index to decide which chain to put the
    // component in, and being one out puts it in the binder.
    modifications: expanded.modifications.map((modification) => ({
      ...modification, chain: modification.chain + 1,
    })),
    ligandCodes: expanded.ligandCodes,
    problem: filled.length === 0 ? null : entitiesProblem(filled),
    // 🔴 COUNTS, NOT LISTS. chooseDesigner takes numbers and compares them with
    // `> 0`; handed the ARRAY of ligand codes, `["HEM"] > 0` is false and Auto
    // reported "the complex is protein only" for a fold that had a haem in it.
    // It folded correctly and designed with the wrong model, silently.
    ligands: expanded.ligandCodes.length,
    nucleic: expanded.chainKinds.filter((kind) => NUCLEIC_TYPES.includes(kind)).length,
  };
}

/**
 * The family this job will design with, and the sentence explaining it.
 *
 * 🔴 THE PICKER IS NEVER WRITTEN TO. A control that changes its own value
 * while you are reading it is a control you cannot trust to still say what you
 * set - so Auto stays Auto and the note beside it says what Auto resolved to.
 */
function resolveDesigner(input) {
  const chosen = element("designer").value;
  if (chosen !== "auto") {
    return { name: chosen, why: "you chose it", automatic: false };
  }
  return {
    ...chooseDesigner({ ligands: input.ligands, nucleic: input.nucleic }),
    automatic: true,
  };
}

function syncDesignerNote() {
  let input;
  try {
    input = chainsFromControls();
  } catch {
    // A half-typed entity is not worth a message here; Hunt reports it.
    return;
  }
  const { name, why, automatic } = resolveDesigner(input);
  element("designer-note").textContent = automatic
    ? `Auto \u2192 ${DESIGNERS[name].label}, because ${why}. ${DESIGNERS[name].note}`
    : DESIGNERS[name].note;
}

/**
 * Every cycle is a FRAME of one object, so the play bar walks the whole hunt.
 *
 * 🔴 THIS USED TO RESET THE VIEWER AND PUSH ONE FRAME, which threw away the
 * thing `viewer.push` exists to do. It appends by design - it builds the
 * object on the first call and adds a frame on every one after - and py2Dmol's
 * transport controls reveal themselves the moment a second frame lands. A
 * reset per cycle meant a viewer that could only ever hold one.
 *
 * 🔴 AND EACH FRAME IS SUPERPOSED ONTO THE FIRST BEFORE IT GOES IN. AF3's
 * sampler randomly re-orients every fold, and `addFrame`'s own alignment needs
 * two frames with equal position counts - which a redesigned chain never has.
 * See src/design/superpose-pdb.js.
 */
let referencePdb;

/**
 * One cycle's frames: its sampler trajectory, then the structure it settled to.
 *
 * 🔴 THE WHOLE CYCLE MOVES BY ONE TRANSFORM, fitted from the settled
 * structure. A diffusion trajectory starts as noise, so fitting each frame on
 * its own target chain fits on a cloud - see superposeCycle().
 *
 * 🔴 AND THE TRAJECTORY IS ADDED AFTER THE CYCLE, NOT DURING IT. `foldAf3`
 * offers an `onFrame` hook and it is deliberately not used: a frame drawn as
 * it arrives cannot be superposed, because the transform comes from a settled
 * structure that does not exist yet. Live frames would therefore be the one
 * thing this module exists to prevent - a structure thrown around a room -
 * and a cycle is a few seconds, which the status line and the bar already
 * account for. What the reader gets instead is every step, kept and
 * scrubbable, which is the part worth watching twice.
 */
function cycleFrames(record) {
  const api = window.py2Dmol;
  const trajectory = keepDiffusionFrames() ? (record.folded?.framePdbs ?? []) : [];
  // The settled structure is the last frame of its own trajectory, so a
  // trajectory that already ends there is not repeated.
  const steps = trajectory.length > 0 ? trajectory.slice(0, -1) : [];
  if (referencePdb === undefined || api?.superpose === undefined) {
    return { steps, settled: record.pdb };
  }
  try {
    const moved = superposeCycle(api.superpose, steps, record.pdb, referencePdb,
                                 { designed: record.chain ?? "A" });
    return { steps: moved.frames, settled: moved.settled };
  } catch (error) {
    // An unalignable cycle is still worth showing; it just sits where it fell.
    console.warn("superposition skipped:", error);
    return { steps, settled: record.pdb };
  }
}

function show(record) {
  const { steps, settled } = cycleFrames(record);
  let frames = 0;
  let built = false;
  for (const [index, step] of steps.entries()) {
    const pushed = viewer.push(step);
    frames = pushed.frames;
    if (pushed.built) {
      built = true;
      // 🔴 THE REFERENCE IS THE FIRST FRAME EVER DRAWN, whatever it is. If
      // that is a sampler step it is a step of cycle 0, which every later
      // cycle is then fitted onto - consistent, which is all the reference has
      // to be.
      referencePdb = step;
    }
    label(frames - 1, `${title(record)} · step ${index + 1}`, undefined);
  }
  const pushed = viewer.push(settled);
  frames = pushed.frames;
  if (pushed.built) { built = true; referencePdb = settled; }
  if (built) {
    viewer.paint();
    element("orient").hidden = false;
  }
  // 🔴 THE SETTLED FRAME CARRIES THE CONFIDENCE AND THE SAMPLER STEPS DO NOT.
  // AF3's confidence head runs once, on the finished sample - so a step has no
  // measured pLDDT and giving it the cycle's would put a confident number on a
  // structure that has not earned it. The card empties on those frames, which
  // is a missing value shown as one.
  label(frames - 1, title(record), record.folded?.confidence);
  jumpTo(frames - 1);
  return frames - 1;
}

const title = (record) => `run ${record.run + 1} · cycle ${record.cycle}`;

/** Name one frame of the object, and hang its confidence on it. */
function label(index, text, confidence) {
  const frame = viewer.renderer?.objectsData?.[viewer.object]?.frames?.[index];
  if (frame === undefined) return;
  frame.name = text;
  frame.label = text;
  frame.title = text;
  frame.confidence = confidence;
}

/** Draw a frame that is already in the object. */
function jumpTo(index) {
  const renderer = viewer.renderer;
  if (renderer === undefined || index < 0) return;
  try {
    renderer.setFrame(index);
    renderer.render("cycle");
  } catch (error) {
    console.warn("frame skipped:", error);
  }
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
    // ...scrub to that cycle's frame rather than rebuilding the viewer, which
    // is the whole reason the frames are in one object.
    const at = record.frame;
    row.addEventListener("click", () => jumpTo(at));
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

const keepDiffusionFrames = () => element("diffusion-frames").checked;

// The card follows whatever the bar is showing, which py2Dmol does not
// announce. See followActiveFrame.
followActiveFrame(() => ({ renderer: viewer.renderer, object: viewer.object }));

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
  for (const id of ["start-sequence", "runs", "cycles", "percent-x", "add-entity",
                    "temperature", "seed", "steps", "alanine-bias", "omit",
                    "length", "designer", "diffusion-frames", "af3-mode",
                    "recycles", "model-family"]) {
    element(id).disabled = running;
  }
  // The entity rows are inputs, buttons and selects built by entity-ui.js, so
  // they are disabled by walking them rather than by id.
  for (const field of element("entity-rows").querySelectorAll("input, select, button")) {
    field.disabled = running;
  }
  for (const field of element("entity-rows").querySelectorAll("[contenteditable]")) {
    field.contentEditable = running ? "false" : "true";
  }
}

/**
 * The designer for one family.
 *
 * `loadDesigner` memoises per family itself, so switching back and forth
 * re-reads nothing; this only adds the progress line.
 */
function designer(name) {
  return loadDesigner({
    name,
    // ...two positional numbers here, unlike loadAf3Weights above. See the
    // note at its call site.
    onProgress: (received, total) => {
      status(total > 0
        ? `Loading ${DESIGNERS[name].label} · ${Math.round((received / total) * 100)}%`
        : `Loading ${DESIGNERS[name].label}`);
    },
  });
}

async function hunt() {
  if (controller !== null) {
    controller.abort();
    return;
  }

  let input;
  try {
    input = chainsFromControls();
  } catch (error) {
    status(error.message, true);
    return;
  }
  const { chains, chainKinds, ligandCodes, modifications } = input;
  // 🔴 THE TARGET IS CHECKED, THE BINDER IS NOT - IT DOES NOT EXIST YET. An
  // empty designed chain is the normal case and `runDesign` draws one. The
  // check itself is entities.js's, which already knows a DNA chain is written
  // in A, C, G and T rather than in the twenty amino acids - the rule this
  // page used to carry a second, shorter copy of.
  if (input.problem !== null) {
    status(input.problem, true);
    return;
  }
  // ...and the binder, when one was typed in. entityProblem does not see it:
  // it is not an entity.
  if (chains[0].length > 0) {
    const problem = af3SequenceProblem(chains[0]);
    if (problem !== null) {
      status(`Binder: ${problem}`, true);
      return;
    }
  }
  if (chains[0].length === 0 && !(number("length", 0) >= 4)) {
    status("A binder is at least four residues.", true);
    return;
  }

  controller = new AbortController();
  const { signal } = controller;
  setRunning(true);
  history = [];
  renderTable();
  element("downloads").hidden = true;
  viewer.forgetCamera();
  viewer.reset();
  referencePdb = undefined;
  updateScoresCard(undefined);

  const runs = Math.max(1, number("runs", 1));
  const cycles = Math.max(0, number("cycles", 5));
  const mode = element("af3-mode").value;
  const calls = number("steps", AF3_COUNTS[mode].preferred);
  const recycles = number("recycles", 0);
  // 🔴 DRAWN ONCE PER HUNT AND REPORTED, NOT DRAWN PER RUN AND LOST. Exploring
  // is the point, so an empty box means a new answer every press - but a run
  // worth keeping has to be repeatable, and a seed nobody can read is not a
  // seed. It goes in the status line at the end, and typing it back in pins
  // the whole hunt.
  const asked = element("seed").value.trim();
  const seed = /^[0-9]+$/.test(asked)
    ? Number(asked) : Math.floor(Math.random() * 0x7fffffff);
  const total = runs * (cycles + 1);
  const started = performance.now();

  try {
    status("Starting WebGPU");
    const device = await getDevice(signal);
    // 🔴 THE TWO LOADERS REPORT PROGRESS DIFFERENTLY, AND GETTING IT WRONG
    // PRINTS "NaN%" RATHER THAN FAILING. loadAf3Weights hands its callback ONE
    // OBJECT, `{loadedBytes, totalBytes}` - it goes through to
    // HttpTensorStore, which reports a whole-store figure - while
    // `Weights.fetch` below hands two positional numbers off one response.
    // Read as `(received, expected)` the object divided by undefined is NaN,
    // and a status line is the one place that does not throw.
    const weights = await loadAf3Weights(({ loadedBytes, totalBytes }) => {
      if (signal.aborted) return;
      progress(totalBytes === 0 ? 0 : loadedBytes / totalBytes);
      status(`Loading AlphaFold 3 · ${(loadedBytes / 1048576).toFixed(0)}`
        + ` / ${(totalBytes / 1048576).toFixed(0)} MiB`);
    });
    const chosen = resolveDesigner(input);
    const { model } = await designer(chosen.name);

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
        length: chains[0].length > 0 ? chains[0].length : number("length", 75),
        percentX: number("percent-x", 90),
        temperature: number("temperature", 0.1),
        omit: element("omit").value.toUpperCase().replace(/[^A-Z]/g, ""),
        alanineBias: element("alanine-bias").checked,

        fold: async (sequence, context) => {
          const done = context.run * (cycles + 1) + context.cycle;
          return foldAf3({
            sequence, mode, calls, recycles, weights, device, signal, seed: seed + run,
            // A ligand and a nucleic chain change the FOLD as well as the
            // designer: featurise gives each ligand a chain of its own and
            // reads a nucleic chain one token per base.
            chainKinds, ligandCodes, modifications,
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
        record.frame = show(record);
        renderTable();
      }
    }

    const best = bestOverall();
    element("downloads").hidden = best === null;
    progress(1);
    const took = ((performance.now() - started) / 1000).toFixed(0);
    status(best === null
      ? `Done in ${took} s · seed ${seed} · nothing under the alanine ceiling`
      : `Done in ${took} s · ${DESIGNERS[chosen.name].label} · best ${best.objective}`
        + ` ${fixed(best.score)} (run ${best.run + 1}, cycle ${best.cycle})`
        + ` · seed ${seed}`);
  } catch (error) {
    progress(0);
    if (isAbortError(error)) {
      element("downloads").hidden = bestOverall() === null;
      status(`Stopped after ${history.length} ${history.length === 1 ? "cycle" : "cycles"}`
        + ` · seed ${seed}`);
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
  element("length-field").hidden = given;
};
element("start-sequence").addEventListener("input", syncLengthControls);
syncLengthControls();

// The picker is built from the registry rather than written into the HTML, so
// a family added to src/design/designers.js and mirrored by tools/sync-mpnn.py
// appears here with nothing else to remember.
for (const name of DESIGNER_NAMES) {
  element("designer").append(Object.assign(document.createElement("option"), {
    value: name, textContent: DESIGNERS[name].label,
  }));
}
// 🔴 EXPOSED FOR tools/protein-hunter-in-page.py, WHICH HAS NO OTHER WAY IN.
// The rows are built by entity-ui.js and their model is a closure; a harness
// that wrote into a row's field would leave that model behind the DOM and the
// fold would run on what the model still held. `set` is the same call
// index.html's paste path makes.
window.__hunterTargets = targets;

/**
 * Rebuild the step dial for the sampler that is selected.
 *
 * 🔴 THE TWO MODES DO NOT SHARE A RANGE. A flow step walks the whole noise
 * schedule and a diffusion step discretises it, so the counts differ by an
 * order of magnitude - 16 against 20 to 320 - and one `<input type=number>`
 * spanning both would offer a flow value that is pointlessly slow and a
 * diffusion value that does not resolve. index.html rebuilds this control on a
 * mode change for the same reason; AF3_COUNTS carries the values and the
 * measurements behind them.
 */
function syncSteps() {
  const mode = element("af3-mode").value;
  const counts = AF3_COUNTS[mode] ?? AF3_COUNTS.flow;
  const select = element("steps");
  const wanted = Number(select.value);
  select.replaceChildren(...counts.values.map((value) => Object.assign(
    document.createElement("option"), { value: String(value), textContent: String(value) })));
  // Keep the reader's choice when the new mode also offers it; otherwise take
  // that mode's own preferred count rather than the first thing in the list.
  select.value = String(counts.values.includes(wanted) ? wanted : counts.preferred);
  element("steps-label").textContent = counts.label;
}
element("af3-mode").addEventListener("change", syncSteps);
syncSteps();

// The entity list notifies through its own onChange; this is the picker.
element("designer").addEventListener("change", syncDesignerNote);
syncDesignerNote();

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
