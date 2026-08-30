/**
 * LocalFold on py2Dmol's own application.
 *
 * WHAT THIS FILE IS, AND MOSTLY IS NOT. The page is py2Dmol's index.html with
 * one panel swapped: the fetch-and-upload row became a fold row. Everything
 * else - the viewer, the sequence strip, the MSA and PAE panels, selections,
 * sessions, downloads - is py2Dmol's, running its own code, wired by its own
 * app/main.js. None of it is reimplemented here and none of it should be.
 *
 * 🔴 A PREDICTION ENTERS THE WAY A LOADED FILE DOES, WITHOUT BEING ONE.
 *
 * app/main.js already knows how to ingest a fold: its loader dispatches on
 * extension - .pdb as structure or frames, .json paired to it as PAE, .a3m as
 * the alignment - because that is the shape ColabFold writes. And it takes
 * VIRTUAL files, a name and a reader, because a ZIP entry was never a File
 * either. So a prediction computed in this tab is handed straight over through
 * `window.py2dmolLoadFiles`: nothing is written to disk, no File is
 * manufactured, and no change event is replayed on a hidden input.
 *
 * Every panel downstream then lights up for free, and none of it is our code to
 * keep working. What the reader downloads is separate and explicit - see the
 * two buttons at the foot of this file, which write what the model produced.
 */
import { AlphaFoldMonomerGpu } from "../src/model/monomer.js";
import { AlphaFoldQueryOnlyGpu } from "../src/model/query-only.js";
import { parseA3m } from "../src/input/a3m.js";
import { generateMmseqs2Msa } from "../src/input/mmseqs2-api.js";
import { getDevice, loadModel } from "./model.js";
import { confidenceJson, paeMatrix, predictionToPdb, recyclesToPdb } from "./prediction-results.js";
import { cleanSequence, sequenceProblem } from "./sequence.js";

const element = (id) => {
  const value = document.getElementById(id);
  if (value === null) throw new Error(`missing element #${id}`);
  return value;
};

const sequenceValue = () => cleanSequence(element("sequence").value);
const recycleCount = () => Number(element("recycles").value) || 0;
const msaMode = () => element("msa-mode").value;

let uploadedA3m = "";

/** The last prediction, kept so it can be downloaded as it was computed. */
let lastPrediction;

/** The drawn object, once the first pass has landed. See appendPass. */
let viewer;
let viewerObject;

/** py2Dmol's own status line, so folding reports where fetching used to. */
function status(text, isError = false) {
  const node = document.getElementById("status-message");
  if (node === null) return;
  node.textContent = text;
  node.style.color = isError ? "#b91c1c" : "";
}

function progress(fraction) {
  const bar = element("progress");
  bar.hidden = fraction === null;
  if (fraction !== null) bar.value = fraction;
}

// --- the alignment ---------------------------------------------------------

async function alignmentText() {
  switch (msaMode()) {
    case "single": return null;
    case "paste": {
      const text = element("msa-text").value.trim();
      if (text.length === 0) throw new Error("Paste an A3M, or switch the alignment back to none");
      return text;
    }
    case "upload": {
      if (uploadedA3m.length === 0) throw new Error("Choose an A3M file, or switch the alignment back to none");
      return uploadedA3m;
    }
    case "search": {
      // 🔴 THE ONE REQUEST THIS PAGE MAKES OFF THE MACHINE. Everything else runs
      // against weights already on disk. The sequence is sent to the public
      // ColabFold MMseqs2 server, so it is a mode the reader picks rather than
      // a default they discover afterwards.
      const query = sequenceValue();
      const problem = sequenceProblem(query);
      if (problem !== null) throw new Error(problem);
      const searched = await generateMmseqs2Msa(query, {
        onProgress: ({ phase, status: state, elapsedMilliseconds }) => {
          status(`MSA search · ${phase} (${state}) · ${(elapsedMilliseconds / 1000).toFixed(0)}s`
            + " · api.colabfold.com");
        },
      });
      status(`MSA search found ${searched.depth} sequences`);
      return searched.a3m;
    }
    default:
      throw new Error(`unknown alignment mode ${msaMode()}`);
  }
}

// --- handing the prediction to py2Dmol -------------------------------------

/**
 * Hand a finished prediction to py2Dmol.
 *
 * 🔴 NO FILES ARE WRITTEN AND NONE ARE FAKED. py2Dmol's ingestion takes VIRTUAL
 * files - a name and a reader - because a ZIP entry was never a File either. So
 * a prediction computed in this tab is passed straight across: no File objects,
 * no DataTransfer, no synthetic change event on a hidden input. The name is the
 * only thing that has to be right, because extensions are how the app decides
 * what a thing IS.
 *
 * 🔴 AND THE NAMES ARE LOAD-BEARING. A PAE matrix is paired to its structure by
 * a fuzzy basename match that scores the shared prefix and rewards the words
 * "pae", "scores", "full_data" and "aligned_error". A common stem plus a
 * recognised word is what lands the pairing; rename these and the PAE panel
 * stays empty without complaining.
 *
 * The alignment is passed only when there is one - handing the app an A3M for a
 * single-sequence fold would draw a one-row MSA panel that says nothing.
 */
async function loadIntoViewer({ stem, pdb, scores, a3m, pae, length }) {
  const load = window.py2dmolLoadFiles;
  if (typeof load !== "function") {
    throw new Error("this py2Dmol bundle has no py2dmolLoadFiles; it needs the `full` build");
  }
  const file = (name, text) => ({ name, readAsync: () => Promise.resolve(text) });
  const files = [file(`${stem}.pdb`, pdb), file(`${stem}_scores.json`, scores)];
  if (a3m != null) files.push(file(`${stem}.a3m`, a3m));
  const stats = await load(files, true);
  const registry = window.py2dmol_viewers ?? {};
  viewer = registry[Object.keys(registry)[0]]?.renderer;
  viewerObject = viewer?.currentObjectName;
  // 🔴 THE FIRST FRAME NEEDS ITS PAE ON THE FRAME, not only on the renderer.
  // Ingestion sets the panel up, but py2Dmol reads `frame.pae` when the frame
  // CHANGES - so without this, scrubbing the play bar back to the first pass
  // blanks a matrix that was on screen a moment earlier.
  if (pae !== undefined) {
    const frame = viewer?.objectsData?.[viewerObject]?.frames?.[0];
    if (frame !== undefined) { frame.pae = pae; frame.pae_n = length; }
  }
  return stats;
}

/**
 * Append one finished pass to the structure already on screen.
 *
 * 🔴 THE STRUCTURE APPEARS WHILE THE REST IS STILL RUNNING. A four-pass fold of
 * an alignment is half a minute or more, and drawing nothing until the last one
 * lands wastes the first three: the interesting thing about recycling is
 * watching it settle. The first pass goes in through py2Dmol's file ingestion,
 * because that is what builds the object and populates the PAE and MSA panels;
 * every pass after it is a FRAME on that same object, which is what the play
 * bar walks.
 *
 * The per-pass PAE rides on the frame, which is where py2Dmol looks for it
 * (`frame.pae` / `frame.pae_n`), so scrubbing the bar moves the matrix too.
 */
function appendPass(sequence, recycle) {
  const api = window.py2Dmol;
  if (viewer === undefined || viewerObject === undefined || api?.frameFromText === undefined) return;
  const pdb = predictionToPdb(sequence, recycle.structure, recycle.confidence.plddt);
  const frame = api.frameFromText(pdb);
  frame.pae = paeMatrix(recycle.confidence.predictedAlignedError, sequence.length);
  frame.pae_n = sequence.length;
  viewer.addFrame(frame, viewerObject);
  // ...and jump to it, so the newest pass is the one being looked at.
  const object = viewer.objects?.find((entry) => entry.name === viewerObject);
  if (object?.frames?.length) viewer.setFrame(object.frames.length - 1);
  viewer.render("recycle");
}

// --- running ---------------------------------------------------------------

async function fold(event) {
  event?.preventDefault();
  const button = element("predict");
  button.disabled = true;
  try {
    let sequence = sequenceValue();
    const alignment = await alignmentText();
    if (alignment !== null) {
      // THE ALIGNMENT'S QUERY WINS. An A3M carries its own first record, and
      // folding the box's sequence against somebody else's alignment would be
      // folding two different proteins at once.
      sequence = parseA3m(alignment).query;
      element("sequence").value = sequence;
    }
    const problem = sequenceProblem(sequence);
    if (problem !== null) throw new Error(problem);

    status("Starting WebGPU");
    const device = await getDevice();
    const variant = alignment === null ? "single" : "msa";
    const model = await loadModel(variant, (value) => {
      progress(value.totalBytes === 0 ? 0 : value.loadedBytes / value.totalBytes);
      status(`Loading model · ${(value.loadedBytes / 1048576).toFixed(0)}`
        + ` / ${(value.totalBytes / 1048576).toFixed(0)} MiB`);
    });
    progress(null);
    const recycles = recycleCount();
    const passes = recycles + 1;
    const started = performance.now();
    // ...a new run draws afresh: the old object stays until the first pass of
    // this one lands, so the page is never blank between folds.
    viewer = undefined;
    viewerObject = undefined;
    status(`Folding ${sequence.length} residues · ${passes} pass${passes === 1 ? "" : "es"}`);

    // ...DRAWN AS EACH PASS LANDS, not collected and drawn at the end. The
    // first builds the object and the panels; the rest are frames on it.
    const onRecycle = (recycle, index) => {
      status(`Pass ${index + 1} of ${passes} · pLDDT ${recycle.confidence.meanPlddt.toFixed(1)}`);
      if (index === 0) {
        void loadIntoViewer({
          stem: "prediction",
          pdb: predictionToPdb(sequence, recycle.structure, recycle.confidence.plddt),
          scores: confidenceJson(sequence, recycle.confidence),
          a3m: alignment,
          pae: paeMatrix(recycle.confidence.predictedAlignedError, sequence.length),
          length: sequence.length,
        });
      } else {
        appendPass(sequence, recycle);
      }
    };
    const runProgress = ({ completed, total, waiting }) => {
      if (waiting) {
        const bar = element("progress");
        bar.hidden = false;
        bar.removeAttribute("value");
        status("Running the trunk on the GPU…");
        return;
      }
      progress(Math.min(1, completed / total));
      status(`Folding · ${Math.min(100, Math.round(100 * completed / total))}%`);
    };

    const prediction = alignment === null
      ? await new AlphaFoldQueryOnlyGpu(device).predictSequence(
        sequence, model.weights, model.featureTables,
        { recycles, randomSeed: 0 }, model.paeBreaks, onRecycle, runProgress)
      : await new AlphaFoldMonomerGpu(device).predictA3m(
        alignment, model.weights, model.featureTables,
        { recycles, randomSeed: 0 }, model.paeBreaks, onRecycle, runProgress);

    progress(null);
    const final = prediction.final;
    lastPrediction = {
      stem: "prediction",
      pdb: recyclesToPdb(sequence, prediction.recycles),
      scores: confidenceJson(sequence, final.confidence),
      a3m: alignment,
    };
    // 🔴 A SAFETY NET, because the failure it catches is invisible. onRecycle is
    // optional the whole way down, so a model path that accepts the callback and
    // never calls it would produce a finished fold, a "Done" status and an empty
    // page. If nothing drew while the passes ran, draw them all now.
    let stats;
    if (viewer === undefined) stats = await loadIntoViewer(lastPrediction);
    // ...shown beside the PAE panel, which appears at the same moment.
    element("downloads").style.display = "flex";
    const took = ((performance.now() - started) / 1000).toFixed(1);
    const paired = viewer?.paeRenderer?.n > 0 ? " · PAE paired" : "";
    status(`Done in ${took} s · pLDDT ${final.confidence.meanPlddt.toFixed(1)}`
      + ` · pTM ${final.confidence.ptm.toFixed(3)}${paired}`);
  } catch (error) {
    progress(null);
    status(error instanceof Error ? error.message : String(error), true);
  } finally {
    button.disabled = false;
  }
}

// --- wiring ----------------------------------------------------------------

element("predict").addEventListener("click", (event) => void fold(event));

const sequenceBox = element("sequence");
// ...tidied on blur and paste, not on every keystroke: rewriting the value
// moves the caret to the end, which throws a mid-sequence correction away.
const tidySequence = () => {
  const cleaned = cleanSequence(sequenceBox.value);
  if (cleaned !== sequenceBox.value) sequenceBox.value = cleaned;
};
sequenceBox.addEventListener("blur", tidySequence);
sequenceBox.addEventListener("paste", () => setTimeout(tidySequence, 0));

// ...THE SETTINGS FOLD AWAY, the way py2Dmol's own do: same button, same
// caret, same aria contract - a control that reveals something has to say so
// to anything not looking at it.
const optionsButton = element("foldOptionsButton");
const optionsPanel = element("foldOptions");
optionsButton.addEventListener("click", () => {
  const open = optionsPanel.hidden;
  optionsPanel.hidden = !open;
  // ...and the caret follows from aria-expanded in CSS, so there is one
  // source of truth for whether the panel is open.
  optionsButton.setAttribute("aria-expanded", String(open));
});

const modeSelect = element("msa-mode");
const syncMode = () => {
  element("msa-text").hidden = modeSelect.value !== "paste";
  element("msa-file").hidden = modeSelect.value !== "upload";
};
modeSelect.addEventListener("change", syncMode);
syncMode();

element("msa-file").addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (file === undefined) return;
  void file.text().then((text) => {
    try {
      const described = parseA3m(text);
      uploadedA3m = text;
      status(`${described.depth} sequences · ${described.length} columns`);
    } catch (error) {
      uploadedA3m = "";
      status(error instanceof Error ? error.message : String(error), true);
    }
  });
});


// ...THE RAW PREDICTION, downloadable as computed. py2Dmol's own save button
// writes a session; these two write what the model actually produced.
function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url; link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
element("download-pdb").addEventListener("click", () => {
  if (lastPrediction) download(`${lastPrediction.stem}.pdb`, lastPrediction.pdb, "chemical/x-pdb");
});
element("download-scores").addEventListener("click", () => {
  if (lastPrediction) download(`${lastPrediction.stem}_scores.json`, lastPrediction.scores, "application/json");
});
