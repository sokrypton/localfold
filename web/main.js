import { AlphaFoldQueryOnlyGpu } from "../src/model/query-only.js";
import { getDevice, loadModel } from "./model.js";
import { drawLegends, drawPlddt } from "./plddt.js";
import { createStructureViewer } from "./viewer.js";
import { confidenceJson, predictionToPdb, recyclesToPdb } from "./prediction-results.js";
import { createMutationPanel, mutationName, residueAt, substitute, wasClick } from "./mutate.js";
import { correspondence } from "./align.js";
import { cleanSequence, sequenceProblem } from "./sequence.js";
import { morphFrames, superposeOnto } from "./morph.js";

const element = (id) => {
  const value = document.getElementById(id);
  if (value === null) throw new Error(`missing element #${id}`);
  return value;
};

const sequenceValue = () => cleanSequence(element("sequence").value);
// 0 means one pass and no recycling, which is the fast default; the model
// gains about 7 pLDDT on a single sequence by 3, so the choice is the user's.
const recycleCount = () => Number(element("recycles").value);

let currentPdb = "";
let currentScores = "";
let currentSequence = "";
let currentRecycles = [];

function status(text, state = "running") {
  const node = element("status");
  node.dataset.state = state;
  node.textContent = text;
}

function progress(fraction) {
  const bar = element("progress");
  bar.hidden = fraction === null;
  // ...`value` is REMOVED to make it indeterminate, so it has to be set again
  // rather than assumed present.
  if (fraction !== null) bar.value = fraction;
}

// --- the model ------------------------------------------------------------
//
// The loading itself lives in web/model.js, which the MSA page shares. All that
// is left here is turning its byte counts into this page's status line.

function loadWeights() {
  return loadModel("single", (value) => {
    progress(value.totalBytes === 0 ? 0 : value.loadedBytes / value.totalBytes);
    status(`Loading model \u00b7 ${(value.loadedBytes / 1048576).toFixed(0)}`
      + ` / ${(value.totalBytes / 1048576).toFixed(0)} MiB`);
  });
}

// --- the viewer -----------------------------------------------------------
//
// The building, the camera and the frame bookkeeping live in web/viewer.js,
// which the MSA page shares. What stays here is everything that is about THIS
// page: superposing onto the last prediction, morphing into it, and the
// click-to-mutate panel.

const structure = createStructureViewer({ container: element("viewer") });

// ...WHICH SEQUENCE THE DRAWN STRUCTURE IS. Not the box's contents: the box is
// edited by staging a mutation, and until the next fold the picture still shows
// the old residue. Every lookup that turns a click into a letter reads this.
let viewerSequence = "";
let mutationPanel;
let previousPrediction;
let morphedThisRun = false;
const MORPH_STEPS = 14;
const MORPH_MS = 900;

function updateLegend() {
  drawLegends({ plddt: element("legend-plddt"), hydropathy: element("legend-hydro") });
}

/** Start a new animation: the next frame will build the viewer. */
function resetViewer() {
  structure.reset();
  viewerSequence = "";
  morphedThisRun = false;
  // ...AND THE PANEL KEEPS ITS RESIDUE. Hiding it here meant every fold threw
  // away the position you were working on, so mutating the same site twice
  // began with hunting for it again. The structure it points into is rebuilt,
  // not the choice of which residue to point at - see restoreSelection.
}

/**
 * Append one recycle to the viewer, superposed onto the last prediction.
 *
 * EVERY PASS LANDS WHERE THE LAST PREDICTION WAS, so a mutation never moves the
 * picture. The model has no reason to return successive predictions in the same
 * global frame, and without this the structure jumps on the first pass and again
 * at the morph. `addFrame`'s own alignment cannot do it: it refuses two frames
 * of different lengths, and a mutation changes one residue's atom list.
 */
function pushFrame(sequence, recycle) {
  const landed = alignedToPrevious(sequence, recycle.structure);
  const pdb = predictionToPdb(sequence, landed, recycle.confidence.plddt);
  const { built, frames } = structure.push(pdb, (container) => {
    viewerSequence = sequence;
    wireMutationClicks(container);
    element("orient").hidden = false;
  });
  if (frames === 0) return;                       // the viewer could not be built
  // MORPHED INTO, NOT AFTER, and only on the first pass. The morph shows what
  // the mutation did, so the reader watches the old structure travel to the new
  // one and then watches the recycles refine it from there. Running it at the
  // end replayed that journey after they had already seen where it arrived, and
  // against a structure several passes further on.
  if (built) playMorph(sequence, landed, recycle.confidence.plddt);
  structure.show(frames - 1, "recycle");
}

// --- mutating a position --------------------------------------------------
//
// FOR A SINGLE-CHAIN PREDICTION THE POSITION INDEX IS THE SEQUENCE INDEX, which
// is what makes a mutation a substring edit rather than a lookup.
//
// THE CLICK IS HANDLED HERE RATHER THAN THROUGH py2Dmol'S EVENT, for one
// reason: `py2dmol-residue-selection-change` carries no coordinates, and a
// popup has to open somewhere. `pickResidueAt` takes client coordinates
// straight from the pointer event, so the handler that places the menu is also
// the one that decides what was hit. The cost is that a pick answers with the
// ATOM where the selection would have answered with the residue - `residueAt`
// in mutate.js is that walk, and it is there rather than here so it can be
// tested without a GPU.

/**
 * Put the substitution in the box and say so. It does NOT re-fold.
 *
 * A fold is seconds of GPU and the click that starts one should be the same
 * button it always was - a menu that silently spent that on a mis-click would
 * be a menu you stop using. The edit is visible in the sequence box, which is
 * also how you undo it.
 */
function stageMutation(index, residue) {
  const next = substitute(viewerSequence, index, residue);
  const name = mutationName(viewerSequence, index, residue);
  const box = element("sequence");
  box.value = next;
  // ...the length readout is wired to `input`, which assigning value does not
  // fire. Dispatching it keeps one source of truth for that text.
  box.dispatchEvent(new Event("input", { bubbles: true }));
  if (next === viewerSequence) {
    status(`${viewerSequence[index]}${index + 1} is already ${residue} — press Fold to re-run`, "done");
    return;
  }
  status(`${name} staged — press Fold to see it`, "done");
}

/**
 * Point the panel back at the residue it was on, now that the fold is done.
 *
 * The position survives the rebuild but the RESIDUE AT IT MAY NOT: folding the
 * mutation is the whole point, so the letter, the name and which cell is marked
 * all have to be read again from the sequence that was just folded.
 */
function restoreSelection() {
  const at = mutationPanel?.position ?? -1;
  if (at < 0 || at >= viewerSequence.length) return;
  const drawn = structure.renderer;
  const label = `${drawn?.positionNames?.[at] ?? "UNK"} ${drawn?.residueNumbers?.[at] ?? at + 1}`;
  mutationPanel.show(at, viewerSequence[at], label);
  // ...AND IN THE PICTURE, not just in the panel. py2Dmol drew the halo when the
  // residue was clicked, and the rebuild cleared it with everything else - so
  // without this the panel names a residue the structure gives no sign of. The
  // selection set is empty after a rebuild, so adding to it is the whole job.
  try { drawn?.select?.([at]); } catch { /* a position this structure has not got */ }
}

function wireMutationClicks(container) {
  if (mutationPanel !== undefined) return;
  // ...BUILT ONCE, into its own host under the viewer. It is not a child of the
  // container, so the replaceChildren() that starts every fold cannot detach it
  // and there is nothing to re-attach.
  // ONE FOLD BUTTON, and it is the page's. The panel had one too, which was
  // the same press wearing a different hat - and it was the easier of the two
  // to reach, so the duplicate was also the one that hid itself afterwards.
  mutationPanel = createMutationPanel(element("mutate-host"), stageMutation);
  // ON THE CONTAINER, not on the canvas. py2Dmol replaces the canvas whenever a
  // viewer is rebuilt - which every new fold does - and a listener bound to the
  // old one would be attached to a node no longer in the page.
  // 🔴 A ROTATION ENDS IN A CLICK. Dragging to turn the structure fires
  // pointerdown, then pointermove, then a click at the release - so a listener
  // that only watches for clicks opens the panel every time the reader spins
  // the model and happens to let go over a residue.
  //
  // FOUR PIXELS, because that is py2Dmol's own threshold for the same decision
  // (mol.js: `const moved = Math.hypot(...); if (moved < 4)`). Matching it is
  // the point: with a different number the halo and this panel would disagree
  // about whether a press was a click, and one could open without the other.
  let pressedAt;
  container.addEventListener("pointerdown", (event) => {
    pressedAt = { x: event.clientX, y: event.clientY };
  });
  container.addEventListener("click", (event) => {
    const click = wasClick(pressedAt, event);
    pressedAt = undefined;
    if (!click) return;                           // that was a rotation
    // ...NOR A PAN. py2Dmol treats a held Cmd or Ctrl as "grab, not pick", and
    // a pan that happens not to travel far would otherwise read as a click here.
    if (event.metaKey || event.ctrlKey) return;
    const drawn = structure.renderer;
    if (drawn === undefined) return;
    const residue = residueAt(drawn, viewerSequence.length, event.clientX, event.clientY);
    // CLICKING AWAY PUTS IT DOWN. py2Dmol already clears its own halo on a
    // background click - the same rule, in its pointerup handler - so leaving
    // the panel up was the page disagreeing with the picture beside it about
    // whether anything was still selected.
    if (residue < 0) { mutationPanel.hide(); return; }
    const letter = viewerSequence[residue];
    const label = `${drawn.positionNames?.[residue] ?? "UNK"} ${drawn.residueNumbers?.[residue] ?? residue + 1}`;
    mutationPanel.show(residue, letter, label);
  });
}

/**
 * A structure moved onto the previous prediction, or left where it is.
 *
 * Fitted on the alpha carbons - one per residue, glycine included, and never
 * the side chains, which are exactly what the mutation changed and would pull
 * the fit around by the thing it is meant to be measuring.
 */
function alignedToPrevious(sequence, predicted) {
  const api = window.py2Dmol;
  if (api?.superpose === undefined || previousPrediction === undefined) return predicted;
  try {
    // AN INDEL IS STILL WORTH SUPERPOSING, it just cannot be paired by index.
    // This used to give up whenever the lengths differed, so inserting one
    // residue made the structure jump to wherever the model happened to put it.
    // The alignment says which positions are the same residue as before, and
    // superpose fits on exactly those - a fit needs three points, not all of
    // them, which is the whole reason it takes a subset.
    const pairing = correspondence(sequence, previousPrediction.sequence);
    if (pairing.from.length < 3) return predicted;
    return superposeOnto(api, predicted, previousPrediction.structure, sequence.length, pairing);
  } catch (error) {
    console.warn("superposition skipped:", error);
    return predicted;
  }
}

/**
 * Play the change into the structure that is already on screen.
 *
 * 🔴 NO FRAMES ARE ADDED. The obvious way to animate a viewer is to hand it the
 * intermediate structures as frames, and it is the wrong way twice over: the
 * play bar fills with fourteen steps that are not passes of the model, and the
 * viewer has to be REBUILT afterwards to get rid of them, which loses the
 * camera and re-runs every piece of wiring hung off it. py2Dmol's
 * `replaceFrame` swaps a frame's contents and leaves the count alone - it is
 * what the notebook's live path has always used - so this walks the LAST frame
 * from the old conformation to the new one and puts it back exactly as it was.
 * The viewer ends holding what it held: the new prediction's passes.
 *
 * The frames are parsed ONCE, before the loop: `framesFromText` on a
 * fourteen-model PDB costs one parse, where building a frame per tick would
 * put a PDB writer and a parser inside a 60 Hz loop.
 *
 * @returns {boolean} whether a morph started
 */
function playMorph(sequence, landed, plddtTo) {
  const api = window.py2Dmol;
  if (api?.superpose === undefined || previousPrediction === undefined) return false;
  // ...AND STILL ONLY FOR A SUBSTITUTION. Superposing across an indel is a
  // question of which residues correspond, and web/align.js answers it - but a
  // morph is a question of what each atom BECOMES, and an inserted residue was
  // not anything a moment ago. There is no honest path for it to travel, so the
  // structure appears rather than arriving.
  if (previousPrediction.sequence.length !== sequence.length) return false;
  if (!structure.built || structure.frames === 0) return false;
  try {
    const steps = morphFrames(previousPrediction.structure, landed,
      previousPrediction.plddt, plddtTo, sequence.length, MORPH_STEPS);
    const frames = api.framesFromText(recyclesToPdb(sequence, steps));
    if (frames.length !== steps.length) return false;
    const drawn = structure.renderer;
    const object = structure.object;
    const id = structure.generation;
    const lastFrame = structure.frames - 1;
    const started = performance.now();
    const tick = () => {
      // ...A NEW FOLD CANCELS IT. resetViewer clears viewerObject, and a morph
      // still walking frames of a viewer that has been replaced would be
      // writing into the wrong structure.
      if (!structure.built || structure.object !== object || structure.generation !== id) return;
      const through = Math.min(1, (performance.now() - started) / MORPH_MS);
      // THE EASING IS IN THE FRAMES, not here - morphFrames already spaced
      // them by smoothstep, so this walks them at an even rate.
      const at = Math.min(frames.length - 1, Math.round(through * (frames.length - 1)));
      drawn.replaceFrame(frames[at], object);
      structure.show(lastFrame, "morph");
      if (through < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    morphedThisRun = true;
    return true;
  } catch (error) {
    // A MORPH IS A GARNISH. Losing it must not lose the fold that is drawn.
    console.warn("morph skipped:", error);
    return false;
  }
}

function showRecycle(sequence, recycle, index, total) {
  element("results").hidden = false;
  element("mean-plddt").textContent = recycle.confidence.meanPlddt.toFixed(1);
  element("ptm").textContent = recycle.confidence.ptm.toFixed(3);
  element("recycle-count").textContent = `${index + 1} / ${total}`;
  drawPlddt(element("plddt-plot"), recycle.confidence.plddt);
  pushFrame(sequence, recycle);
}

// --- running --------------------------------------------------------------

async function fold(event) {
  event?.preventDefault();
  const button = element("predict");
  button.disabled = true;
  try {
    const sequence = sequenceValue();
    const problem = sequenceProblem(sequence);
    if (problem !== null) throw new Error(problem);
    status("Starting WebGPU");
    const device = await getDevice();
    const model = await loadWeights();
    progress(null);
    const recycles = recycleCount();
    const passes = recycles + 1;
    status(`Folding ${sequence.length} residues · ${passes} pass${passes === 1 ? "" : "es"}`);
    const started = performance.now();
    resetViewer();
    // The weights arrive already rounded through fp16 - see web/model.js, which
    // does it once at load rather than on every press of Fold.
    const weights = model.weights;
    // THE BAR NOW MEANS THE RUN, not the download. Counted in Evoformer blocks
    // because that is the unit the fold actually advances in - 52 of them per
    // pass, against four coarse stage boundaries - so it moves steadily rather
    // than jumping a quarter at a time.
    const runProgress = ({ completed, total, waiting }) => {
      // AN INDETERMINATE BAR WHILE THE DEVICE CATCHES UP. The blocks are queued
      // far ahead of the GPU, so between the last one being encoded and the
      // readback landing there is a long wait with nothing to count. A bar
      // frozen at 78% reads as a hang; one that is visibly animating reads as
      // work, which is what it is.
      if (waiting) {
        const bar = element("progress");
        bar.hidden = false;
        bar.removeAttribute("value");
        status("Running the trunk on the GPU…");
        return;
      }
      progress(Math.min(1, completed / total));
      // ...every step, not every eighth. The steps are Evoformer blocks and
      // structure-module iterations; at one a second on a long sequence, a
      // status line that only moves every eighth of them reads as stuck.
      status(`Folding · ${Math.min(100, Math.round(100 * completed / total))}%`);
    };
    const prediction = await new AlphaFoldQueryOnlyGpu(device).predictSequence(
      sequence, weights, model.featureTables,
      { recycles, randomSeed: 0 }, model.paeBreaks,
      // ...drawn as it lands, not collected and drawn at the end.
      (recycle, index) => {
        showRecycle(sequence, recycle, index, passes);
        status(`Pass ${index + 1} of ${passes} · pLDDT ${recycle.confidence.meanPlddt.toFixed(1)}`);
      },
      runProgress,
    );
    currentSequence = sequence;
    currentRecycles = prediction.recycles;
    // A SAFETY NET, because the failure it catches is invisible. onRecycle is
    // optional the whole way down, so a model path that accepts the callback
    // and never calls it produces a completed fold, a "Done" status, and no
    // viewer at all - which is exactly what shipped once. If nothing drew
    // while the passes ran, draw them now.
    if (structure.frames === 0) {
      for (const recycle of prediction.recycles) pushFrame(sequence, recycle);
      element("results").hidden = false;
      drawPlddt(element("plddt-plot"), prediction.final.confidence.plddt);
      element("mean-plddt").textContent = prediction.final.confidence.meanPlddt.toFixed(1);
      element("ptm").textContent = prediction.final.confidence.ptm.toFixed(3);
    }
    currentPdb = recyclesToPdb(sequence, prediction.recycles);
    currentScores = confidenceJson(sequence, prediction.final.confidence);
    element("recycle-count").textContent = String(prediction.recycles.length);
    restoreSelection();
    // THE SUPERPOSED STRUCTURE IS WHAT TRAVELS FORWARD, as the next
    // prediction's reference - it is what is on screen, and comparing against
    // anything else would be comparing against a picture nobody saw. The morph
    // itself already ran, into the first frame; see pushFrame.
    const landed = alignedToPrevious(sequence, prediction.final.structure);
    previousPrediction = {
      sequence,
      structure: landed,
      plddt: prediction.final.confidence.plddt,
    };
    const took = `${((performance.now() - started) / 1000).toFixed(1)} s`;
    const morphed = morphedThisRun;
    status(morphed ? `Done in ${took} — showing what moved` : `Done in ${took}`, "done");
  } catch (error) {
    progress(null);
    status(error instanceof Error ? error.message : String(error), "failed");
  } finally {
    button.disabled = false;
  }
}

function download(filename, contents, type) {
  const url = URL.createObjectURL(new Blob([contents], { type }));
  const link = document.createElement("a");
  link.href = url; link.download = filename; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

element("prediction-form").addEventListener("submit", (event) => void fold(event));
const sequenceBox = element("sequence");
const showLength = () => {
  element("sequence-length").textContent = `${sequenceValue().length} residues`;
};
sequenceBox.addEventListener("input", showLength);

// ...AND THE BOX IS TIDIED, but not while anybody is typing. Rewriting the value
// on every keystroke moves the caret to the end, so a correction in the middle
// of a sequence throws the reader to the bottom of it. A paste and a blur are
// both moments when the caret is not being used for anything.
const tidySequence = () => {
  const cleaned = cleanSequence(sequenceBox.value);
  if (cleaned !== sequenceBox.value) sequenceBox.value = cleaned;
  showLength();
};
sequenceBox.addEventListener("blur", tidySequence);
sequenceBox.addEventListener("paste", () => setTimeout(tidySequence, 0));
// ...BACK TO A SENSIBLE VIEW. The camera is kept across folds on purpose, so
// a reader who has turned the structure a long way has nothing to return to
// otherwise. py2Dmol animates the flight itself.
element("orient").addEventListener("click", () => {
  try { structure.renderer?.orient(); } catch (error) { console.warn("orient skipped:", error); }
});

element("download-pdb").addEventListener("click",
  () => download("prediction_recycles.pdb", currentPdb, "chemical/x-pdb"));
element("download-scores").addEventListener("click",
  () => download("prediction_scores.json", currentScores, "application/json"));

updateLegend();

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
