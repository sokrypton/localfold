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
import { AlphaFoldUnifiedGpu } from "../src/multimer/model.js";
import { parseA3m } from "../src/input/a3m.js";
import { generateMmseqs2ComplexMsa, generateMmseqs2Msa } from "../src/input/mmseqs2-api.js";
import { isAbortError, throwIfAborted } from "../src/runtime/abort.js";
import { AF3_COUNTS, af3SequenceProblem, foldAf3, loadAf3Weights } from "./af3-model.js";
import { getDevice, loadModel } from "./model.js";
import { correspondence } from "./align.js";
import { superposeOnto } from "./morph.js";
import { confidenceJson, paeMatrix, predictionToPdb, recyclesToPdb, safeJobName } from "./prediction-results.js";
import { complexSequenceProblem } from "./sequence.js";
import { entitiesProblem, expandEntities } from "./entities.js";
import { createEntityList } from "./entity-ui.js";

const element = (id) => {
  const value = document.getElementById(id);
  if (value === null) throw new Error(`missing element #${id}`);
  return value;
};

// 🔴 BUILT BEFORE ANYTHING READS IT. app.js is a module, so the DOM is parsed
// by the time this runs; the rows have to exist before the first handler fires
// rather than at the bottom of the file with the other listeners, because
// sequenceValue() and the fold path both go through them.
const entityList = createEntityList(
  element("entity-rows"), element("add-entity"),
  // The default is the sequence the old textarea shipped with, so the page
  // still has something foldable in it on arrival.
  { initial: [{ type: "protein", copies: 1,
    value: "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK" }] });

// 🔴 THE ENTITY LIST IS THE INPUT NOW, and everything below it still reads a
// colon-joined sequence: expandEntities turns copies into repeated chains and
// hands back exactly the string the textarea used to hold, plus the ligand
// codes the textarea could not express. Declared before entityList exists
// because these are called from handlers, never at module scope.
const foldRequest = () => expandEntities(entityList.read());
const sequenceValue = () => {
  const entities = entityList.read();
  return entitiesProblem(entities) === null ? expandEntities(entities).sequence : "";
};
const recycleCount = () => Number(element("recycles").value) || 0;
// 🔴 THE TOLERANCE CONTROL IS GONE AND THE DRIVER'S ARGUMENT IS NOT. Early
// stopping still works; nothing on the page sets it any more, so every fold
// runs the passes it was asked for. element() throws on a missing id, which is
// why this reads the DOM defensively rather than assuming the control is there.
const recycleTolerance = () => Number(document.getElementById("tolerance")?.value) || 0;
const randomSeed = () => {
  const input = document.getElementById("random-seed");
  if (input === null || input.value === "") return 0;
  const parsed = Number(input.value);
  return Number.isFinite(parsed) ? Math.max(0, Math.floor(parsed)) : 0;
};
const maxMsaConfig = () => {
  const select = document.getElementById("max-msa");
  const value = select ? select.value : "512:1024";
  const [msaPart, extraPart] = value.split(":").map((part) => Number(part.trim()));
  const requested = Number.isFinite(msaPart) && msaPart > 0 ? msaPart : 512;
  // 🔴 THE 512 -> 508 IS AlphaFold 2's, AND AF3 MUST NOT INHERIT IT. AF2's
  // monomer config asks for 512 MSA clusters and then spends four of them on
  // templates, so a templated model_1 reads 508. AF3 takes no template rows out
  // of its MSA budget - `num_msa` is the whole of it - so the same dial has to
  // mean 512 there, and reusing the AF2 number would quietly drop four rows.
  const maxMsaSequences = requested === 512 ? 508 : requested;
  const maxExtraSequences = Number.isFinite(extraPart) && extraPart >= 0 ? extraPart : 1024;
  return { maxMsaSequences, maxExtraSequences, requested };
};
/**
 * Which weights to fold with: monomer, multimer, or let the sequence decide.
 *
 * Auto is chain count and nothing else. That is the whole distinction in
 * practice - a single chain has no interface to predict, and a complex is what
 * multimer was trained for - and the explicit settings exist to fold the same
 * input both ways rather than to be reached for routinely.
 */
const modelFamily = (chainCount, ligandCount = 0) => {
  const choice = document.getElementById("model-family")?.value ?? "auto";
  // 🔴 A LIGAND IS AlphaFold 3 ONLY, and choosing otherwise is refused rather
  // than quietly corrected. AF2 has no ligand tokens at all, so folding a
  // complex with one under AF2 would drop it silently and return a confident
  // structure of the protein alone - which is a different answer to the
  // question that was asked, not a worse one.
  if (ligandCount > 0) {
    if (choice === "auto") return "af3";
    if (choice !== "af3") {
      throw new Error(`Ligands need AlphaFold 3; the model is set to ${choice}`);
    }
    return choice;
  }
  // 🔴 EVERY EXPLICIT CHOICE PASSES THROUGH, AND THE LIST USED TO BE WRITTEN
  // OUT. Adding AF3 to the dropdown without adding it here meant the page
  // offered a model, accepted it, and silently folded with AF2 monomer instead
  // - which produces a structure and a pTM and looks entirely successful. Auto
  // is the only value that gets decided from the sequence.
  if (choice !== "auto") return choice;
  return chainCount > 1 ? "multimer" : "monomer";
};

const msaMode = () => {
  const msaToggle = document.getElementById("useMsaToggle");
  if (msaToggle !== null && !msaToggle.checked) return "single";
  return element("msa-mode").value;
};

let uploadedA3m = "";
let predictionCount = 0;
const predictions = new Map();

/** The last prediction, kept so it can be downloaded as it was computed. */
let lastPrediction;

/** The one fold owned by the page; pressing the same button aborts it. */
let activeFold;

/** The drawn object, once the first pass has landed. See appendPass. */
let viewer;
let viewerObject;

/** py2Dmol's own status line, so folding reports where fetching used to. */
function status(text, isError = false) {
  const node = document.getElementById("status-message");
  if (node === null) return;
  node.textContent = text;
  node.classList.toggle("error", isError);
}

/**
 * Drive the bar under the status line.
 *
 * 🔴 NEVER HIDES IT. The bar is a permanent part of the status block, because
 * laying it out only while a fold runs moved everything below it twice a run -
 * see the note in web/localfold.css. `null` means idle, which is a look, not a
 * removal; a fraction fills it; "waiting" sweeps for the stretches that have
 * nothing to count.
 */
function progress(fraction) {
  const bar = element("progress");
  if (fraction === null) {
    bar.dataset.state = "idle";
    bar.value = 0;
    return;
  }
  if (fraction === "waiting") {
    bar.dataset.state = "waiting";
    // 🔴 THE VALUE HAS TO GO, NOT JUST THE COLOUR. A <progress> with a value is
    // determinate however it is painted, so the sweep animated over a bar still
    // showing the last number it was given.
    bar.removeAttribute("value");
    return;
  }
  bar.dataset.state = "running";
  bar.value = fraction;
}

// --- the alignment ---------------------------------------------------------

/**
 * The alignment, as the viewer wants it and as the model wants it.
 *
 * They differ for a complex: the viewer shows one merged A3M, while the model
 * takes the per-chain alignments so clustering, subsampling and masking run
 * separately for each copy. Merging first makes repeated chains identical.
 */
async function alignmentText(chains, signal, family) {
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
      const query = chains.join("");
      const problem = complexSequenceProblem(chains.join(":"));
      if (problem !== null) throw new Error(problem);
      // 🔴 MULTIMER PAIRS, THE MONOMER STAYS BLOCK-DIAGONAL, and neither is a
      // switch. For repeated chains pairing is not an approximation: every copy
      // of one protein is searched once and gets the same homologs, so row s IS
      // one organism across all of them. What makes it correct is the weights -
      // the multimer relative encoding is told which chain is which, so the
      // paired rows mean what they say. The monomer model has no such input, so
      // paired rows would tell it that residues of different copies coevolved,
      // and it stays with the construction it was trained for.
      // ...and the model itself selects the chain merge. See CHAIN_MERGES in
      // mmseqs2-api.js: monomer block-diagonalises everything, multimer is
      // dense within an entity and block-diagonal between, AF3 is dense
      // throughout. They agree on a homomer and differ on a heteromer.
      const searchOptions = {
        signal,
        model: family,
        onProgress: ({ phase, status: state, elapsedMilliseconds }) => {
          if (signal.aborted) return;
          status(`MSA search · ${phase} (${state}) · ${(elapsedMilliseconds / 1000).toFixed(0)}s`
            + " · api.colabfold.com");
        },
      };
      const searched = chains.length === 1
        ? await generateMmseqs2Msa(query, searchOptions)
        : await generateMmseqs2ComplexMsa(chains, searchOptions);
      status(`MSA search found ${searched.depth} sequences`);
      // 🔴 THE BLOCKS COME BACK APART, AND AF3 NEEDS THEM THAT WAY. `text` is
      // the paired rows stacked above the unpaired ones, which is what the
      // viewer draws and what AlphaFold 2 folds; `blocks` keeps them separate,
      // because AF3's `msa` is the paired block followed by the unpaired one
      // and its profile is computed over the second ALONE.
      //
      // A homo-oligomer has no paired block: its unpaired merge is already the
      // paired construction, one search speaking for every copy, so `paired` is
      // null and AF3 does what it does with none. Pairing is a second search
      // and it only happens for distinct sequences.
      return { text: searched.a3m, blocks: searched.blocks ?? { unpaired: searched.a3m } };
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
async function loadIntoViewer({ stem, pdb, scores, a3m, pae, length, confidence }) {
  const load = window.py2dmolLoadFiles;
  if (typeof load !== "function") {
    throw new Error("this py2Dmol bundle has no py2dmolLoadFiles; it needs the `full` build");
  }
  const file = (name, text) => ({ name, readAsync: () => Promise.resolve(text) });
  const files = [
    file(`${stem}.pdb`, pdb),
    file(`${stem}_scores.json`, typeof scores === "string" ? scores : JSON.stringify(scores, null, 2)),
  ];
  if (a3m != null) {
    // 🔴 ONE MERGED A3M, NOT ONE PER CHAIN. This split the complex alignment
    // back into per-chain pieces, and that is exactly what a paired MSA cannot
    // survive: row s means ONE ORGANISM ACROSS THE CHAINS, and the statement
    // lives on the boundary between them - cut there, all that is left is two
    // ordinary single-chain alignments and a viewer with no way to know they
    // were ever related. py2Dmol reads the concatenation itself now (it matches
    // no single chain, so it tries the chains' queries end to end), draws the
    // chain boundaries, keeps the paired rows above the unpaired ones, and
    // scores each row over the blocks it occupies - without which the unpaired
    // half of every complex MSA falls under the coverage filter and vanishes.
    //
    // NOTHING HAS TO BE DECLARED. `blocks.paired` and `pairedDepth` are ours to
    // keep for the model; the viewer infers pairing per row, from which blocks
    // a row has residues in, so an alignment a reader UPLOADS gets the same
    // picture as one this page searched.
    //
    // The query must be the chains concatenated in structure order, which it is:
    // both this and the PDB are written from `sequence`.
    files.push(file(`${stem}.a3m`, a3m));
  }
  const stats = await load(files, true);
  const registry = window.py2dmol_viewers ?? {};
  viewer = registry[Object.keys(registry)[0]]?.renderer;
  viewerObject = viewer?.currentObjectName;
  if (viewer !== undefined) {
    if (typeof viewer.setColorScheme === "function") {
      viewer.setColorScheme("plddt");
    } else if (typeof viewer.colorBy === "function") {
      viewer.colorBy("plddt");
    }
  }
  if (viewer !== undefined && !viewer._scoresHookAttached) {
    viewer._scoresHookAttached = true;
    const origSetFrame = viewer.setFrame.bind(viewer);
    viewer.setFrame = function(frameIndex) {
      const res = origSetFrame(frameIndex);
      syncScoresCardToActiveFrame(frameIndex);
      return res;
    };
    const origRender = viewer.render ? viewer.render.bind(viewer) : null;
    if (origRender) {
      viewer.render = function(...args) {
        const res = origRender(...args);
        syncScoresCardToActiveFrame();
        return res;
      };
    }
  }
  // 🔴 THE FIRST FRAME NEEDS ITS PAE ON THE FRAME, not only on the renderer.
  // Ingestion sets the panel up, but py2Dmol reads `frame.pae` when the frame
  // CHANGES - so without this, scrubbing the play bar back to the first pass
  // blanks a matrix that was on screen a moment earlier.
  if (pae !== undefined || viewerObject !== undefined) {
    const frame = viewer?.objectsData?.[viewerObject]?.frames?.[0];
    if (frame !== undefined) {
      frame.name = "recycle_0";
      frame.label = "recycle_0";
      frame.title = "recycle_0";
      if (confidence !== undefined) frame.confidence = confidence;
      if (pae !== undefined) { frame.pae = pae; frame.pae_n = length; }
    }
  }
  return stats;
}

let lastReportedFrameIdx = -1;

function getActiveFrameConfidence(frameIndex) {
  try {
    if (!viewer) return null;
    const objName = viewerObject ?? viewer.currentObjectName;
    const obj = viewer.objects?.find((entry) => entry.name === objName);
    const objData = viewer.objectsData?.[objName];
    const frames = objData?.frames ?? obj?.frames;
    const idx = frameIndex !== undefined
      ? frameIndex
      : (obj?.currentFrame ?? objData?.currentFrame ?? viewer.currentFrame ?? 0);
    const targetFrame = frames?.[idx];
    const pred = (objName ? predictions.get(objName) : null) ?? lastPrediction;
    const conf = targetFrame?.confidence ?? pred?.recycles?.[idx]?.confidence;
    return { confidence: conf, index: idx };
  } catch (err) {
    return null;
  }
}

function syncScoresCardToActiveFrame(frameIndex) {
  const result = getActiveFrameConfidence(frameIndex);
  if (result && result.confidence) {
    lastReportedFrameIdx = result.index;
    updateScoresCard(result.confidence, `Pass ${result.index + 1}`);
  }
}

// Watch for animation playback changes (py2Dmol play button loop)
setInterval(() => {
  try {
    if (!viewer) return;
    const objName = viewerObject ?? viewer.currentObjectName;
    const obj = viewer.objects?.find((entry) => entry.name === objName);
    const objData = viewer.objectsData?.[objName];
    const idx = obj?.currentFrame ?? objData?.currentFrame ?? viewer.currentFrame;
    if (idx !== undefined && idx !== lastReportedFrameIdx) {
      syncScoresCardToActiveFrame(idx);
    }
  } catch (e) {}
}, 50);

function updateScoresCard(confidence, passBadge = "") {
  const box = document.getElementById("predictionScoresBox");
  if (!box) return;
  if (!confidence) {
    box.style.display = "none";
    return;
  }
  box.style.display = "flex";

  const badge = document.getElementById("scoresPassBadge");
  if (badge) badge.textContent = passBadge;

  const meanPlddt = document.getElementById("metricMeanPlddt");
  if (meanPlddt) {
    meanPlddt.textContent = confidence.meanPlddt !== undefined
      ? Number(confidence.meanPlddt).toFixed(1)
      : "-";
  }

  const ptm = document.getElementById("metricPtm");
  if (ptm) {
    ptm.textContent = confidence.ptm !== undefined
      ? Number(confidence.ptm).toFixed(3)
      : "-";
  }

  const iptmCell = document.getElementById("metricIptmCell");
  const iptm = document.getElementById("metricIptm");
  if (iptmCell && iptm) {
    if (confidence.iptm !== undefined && !Number.isNaN(Number(confidence.iptm))) {
      iptmCell.style.display = "block";
      iptm.textContent = Number(confidence.iptm).toFixed(3);
    } else {
      iptmCell.style.display = "none";
    }
  }

  const multimerCell = document.getElementById("metricMultimerCell");
  const multimer = document.getElementById("metricMultimer");
  if (multimerCell && multimer) {
    const multimerScore = confidence.multimerScore ?? (
      confidence.iptm !== undefined && !Number.isNaN(Number(confidence.iptm)) && confidence.ptm !== undefined
        ? 0.8 * Number(confidence.iptm) + 0.2 * Number(confidence.ptm)
        : undefined
    );
    if (multimerScore !== undefined && !Number.isNaN(Number(multimerScore))) {
      multimerCell.style.display = "block";
      multimer.textContent = Number(multimerScore).toFixed(3);
    } else {
      multimerCell.style.display = "none";
    }
  }
}

let previousFold = undefined;

function alignedToPrevious(sequence, structure) {
  const api = window.py2Dmol;
  if (api?.superpose === undefined || previousFold === undefined) return structure;
  try {
    const pairing = correspondence(sequence, previousFold.sequence);
    if (pairing.from.length < 3) return structure;
    return superposeOnto(api, structure, previousFold.structure, sequence.length, pairing);
  } catch (error) {
    console.warn("superposition skipped:", error);
    return structure;
  }
}

function alignedToFirstPass(sequence, structure, firstPassStructure) {
  const api = window.py2Dmol;
  if (api?.superpose === undefined || firstPassStructure === undefined) return structure;
  try {
    return superposeOnto(api, structure, firstPassStructure, sequence.length);
  } catch (error) {
    console.warn("recycle superposition skipped:", error);
    return structure;
  }
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
function appendPass(sequence, chainLengths, recycle, recycleIndex, firstPassStructure = undefined) {
  const api = window.py2Dmol;
  if (viewer === undefined || viewerObject === undefined || api?.frameFromText === undefined) return;
  const aligned = alignedToFirstPass(sequence, recycle.structure, firstPassStructure);
  const pdb = predictionToPdb(sequence, aligned, recycle.confidence.plddt, chainLengths);
  const frame = api.frameFromText(pdb);
  const index = recycleIndex ?? (viewer?.objectsData?.[viewerObject]?.frames?.length ?? 1);
  frame.name = `recycle_${index}`;
  frame.label = `recycle_${index}`;
  frame.title = `recycle_${index}`;
  frame.confidence = recycle.confidence;
  frame.pae = paeMatrix(recycle.confidence.predictedAlignedError, sequence.length);
  frame.pae_n = sequence.length;
  frame.align = true;
  viewer.addFrame(frame, viewerObject);
  // ...and jump to it, so the newest pass is the one being looked at.
  const object = viewer.objects?.find((entry) => entry.name === viewerObject);
  if (object?.frames?.length) viewer.setFrame(object.frames.length - 1);
  viewer.render("recycle");
}

// --- running ---------------------------------------------------------------

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

/**
 * Show the controls the chosen model actually reads, and hide the rest.
 *
 * 🔴 A CONTROL THAT IS QUIETLY IGNORED IS WORSE THAN A MISSING ONE. AF3 here
 * folds a one-row MSA, so the alignment controls do not reach it; leaving them
 * on screen would invite someone to set one and conclude the model was broken
 * when nothing changed. Model, Recycles and Seed apply to every model and stay
 * put, so the row keeps one order.
 */
function syncModelControls() {
  const af3 = (document.getElementById("model-family")?.value ?? "auto") === "af3";
  for (const id of ["af3ModeGroup", "af3CountGroup"]) {
    const node = document.getElementById(id);
    if (node !== null) node.hidden = !af3;
  }
  // 🔴 A HIDDEN CONTROL HAS TO BE RESTORED. The first version only ever SET
  // hidden, so choosing AF3 and going back to AF2 left the page with no
  // Recycles until it was reloaded.
  //
  // 🔴 RECYCLES AND SEED ARE SHOWN FOR EVERY MODEL, so the row keeps one order
  // whatever is chosen. AF3 recycles too - its embedder has always done
  // `pair += prev_embedding(LayerNorm(recycled pair))`, and the loop driving it
  // is in src/af3/fold.js.
  // The MSA groups belong to syncMode, which hides them whenever the toggle is
  // off. Setting them here as well would give one pair of controls two owners
  // that disagree - so the alignment controls are not touched here at all, and
  // they now mean the same thing for all three models.
  if (af3) syncAf3Count();
}

/** The count dial, rebuilt for the sampler - see AF3_COUNTS for why. */
function syncAf3Count() {
  const mode = document.getElementById("af3-mode")?.value ?? "flow";
  const { label, values, preferred } = AF3_COUNTS[mode] ?? AF3_COUNTS.flow;
  const title = document.getElementById("af3-count-label");
  if (title !== null) title.textContent = label;
  const select = document.getElementById("af3-count");
  if (select === null) return;
  select.replaceChildren(...values.map((value) => Object.assign(
    document.createElement("option"),
    { value: String(value), textContent: String(value), selected: value === preferred })));
  select.value = String(preferred);
}

/**
 * Fly to the best view of what is drawn.
 *
 * 🔴 py2Dmol ORIENTS DURING ITS OWN INGESTION, AND AF3 RUNS OVER THE TOP OF IT.
 * The AF2 path loads one pass and stops, so the ingestion's orient is the last
 * thing to touch the camera. AF3 appends a frame every call and renders each
 * one, which lands on the camera before that orient has settled - so the first
 * structure arrived unframed. Orienting once, after the first frame's load has
 * resolved, is enough for the whole trajectory: every later frame is rigidly
 * fitted to that first one, so the view stays right and never jumps mid-run.
 */
function orientBestView() {
  if (viewer === undefined) return;
  try {
    if (window.py2dmolOrient?.orientToBestView) {
      window.py2dmolOrient.orientToBestView(viewer, { positions: [], animate: false });
    } else {
      viewer.orient?.({ positions: [] });
    }
  } catch { /* a view is not worth losing the structure over */ }
}

/**
 * Hold the viewer on the pLDDT ramp rather than letting `auto` decide.
 *
 * 🔴 auto RESOLVES TO rainbow WHEN THERE IS NO CONFIDENCE, and during an AF3
 * fold there is none - the confidence head does not run until the sample is
 * finished, so the frames drawn on the way carry a zero B-factor. py2Dmol
 * reasonably concludes there is nothing to colour by and paints an N-to-C
 * spectrum, so the animation ran rainbow and snapped to pLDDT at the end.
 * Pinning the mode makes those frames the low end of the confidence ramp
 * instead, which is one palette throughout and does not claim a fold is
 * finished before it is.
 *
 * Driven through the app's own colour <select> because that is the supported
 * path: this build's renderer has no setColor or setColorScheme at all - those
 * belong to the embed build - and reaching past the control into the colour
 * arrays is what made an earlier attempt at this silently do nothing.
 */
function forcePlddtColours() {
  const select = viewer?.colorSelect;
  if (select === undefined || select === null) return;
  try {
    select.value = "plddt";
    select.dispatchEvent(new Event("change", { bubbles: true }));
  } catch { /* a palette is not worth losing the structure over */ }
}

/**
 * One AlphaFold 3 fold, drawn into py2Dmol as it computes.
 *
 * 🔴 THE PANELS ARE BUILT ON THE FIRST FRAME AND THE SCORES ARRIVE ON THE LAST.
 * AF3's confidence head does not run until the sample is finished, so unlike a
 * recycle - which carries its own pLDDT - a trajectory frame has none. py2Dmol
 * takes confidence PER FRAME, which is what makes this work: the early frames
 * are loaded with none and the final structure is appended carrying the real
 * pLDDT and PAE, and that is the frame the page lands on.
 */
async function foldWithAf3(chains, alignment, alignmentBlocks, signal, ligandCodes = []) {
  const sequence = chains.join(":");
  // 🔴 THE COLONS ARE NOT RESIDUES. `sequence` carries them so the featuriser
  // can see the chain split; every length below is the residue count, and a PAE
  // matrix sized from the wrong one is silently the wrong shape.
  const residues = chains.join("").length;
  // 🔴 THE PAE IS DRAWN OVER THE POLYMER, OR OVER EVERYTHING WHEN THERE IS NO
  // POLYMER. AF3 scores tokens and a ligand is one token per heavy atom, so a
  // mixed fold's matrix is wider than the residues the viewer draws and
  // paeMatrix takes the top-left block. With no residues that block is empty,
  // and an empty PAE panel for a fold that produced a real one is a bug the
  // reader has to guess at.
  // ...and the length reported alongside it follows the same rule, so the
  // matrix and the label it is drawn under always agree.
  const paeSize = (values) => (residues > 0 ? residues : Math.round(Math.sqrt(values.length)));
  // 🔴 ONLY WHEN THERE IS A POLYMER TO CHECK. A ligand-only fold has no
  // sequence, and af3SequenceProblem reports an empty one as "Paste a protein
  // sequence first" - which is the right message for an empty box and the wrong
  // one for a job that is complete without it.
  if (chains.length > 0) {
    const problem = af3SequenceProblem(chains.join(""));
    if (problem !== null) throw new Error(problem);
  }

  const mode = document.getElementById("af3-mode")?.value ?? "flow";
  const calls = Number(document.getElementById("af3-count")?.value)
    || AF3_COUNTS[mode].preferred;
  const recycles = recycleCount();

  status("Loading AlphaFold 3 · 0 MiB");
  const weights = await loadAf3Weights(({ loadedBytes, totalBytes }) => {
    if (signal.aborted) return;
    progress(totalBytes === 0 ? 0 : loadedBytes / totalBytes);
    status(`Loading AlphaFold 3 · ${(loadedBytes / 1048576).toFixed(0)}`
      + ` / ${(totalBytes / 1048576).toFixed(0)} MiB`);
  });
  throwIfAborted(signal);

  const device = await getDevice();
  throwIfAborted(signal);

  predictionCount += 1;
  const header = entityList.header();
  const stem = header !== null ? safeJobName(header) : `af3_${predictionCount}`;
  // See the note in the AF2 path: dropping the handle is what stops the
  // score-card poll refilling from the object still on screen.
  viewer = undefined;
  viewerObject = undefined;

  const api = window.py2Dmol;
  let pending = Promise.resolve();
  const { requested: maxMsaSequences } = maxMsaConfig();
  const result = await foldAf3({
    sequence, mode, calls, recycles, weights, device, signal,
    alignment: alignmentBlocks, maxMsaSequences, ligandCodes,
    // Both modes are seeded now: the flow draws its starting positions once at
    // the top of the schedule.
    seed: randomSeed(),
    onStatus: (text) => { if (!signal.aborted) status(text); },
    onProgress: (fraction) => { if (!signal.aborted) progress(fraction); },
    onFrame: (pdb, index) => {
      if (signal.aborted) return;
      if (index === 0) {
        pending = loadIntoViewer({ stem, pdb, scores: { sequence: chains.join("") }, length: residues })
          .then(() => { orientBestView(); forcePlddtColours(); });
        return;
      }
      if (viewer === undefined || viewerObject === undefined) return;
      const frame = api.frameFromText(pdb);
      frame.name = frame.label = frame.title = `${mode}_${index}`;
      viewer.addFrame(frame, viewerObject);
      const object = viewer.objects?.find((entry) => entry.name === viewerObject);
      if (object?.frames?.length) viewer.setFrame(object.frames.length - 1);
      viewer.render("af3");
    },
  });
  await pending;
  throwIfAborted(signal);

  // 🔴 THE TRAJECTORY IS RELOADED ONCE THE pLDDT EXISTS. The frames drawn during
  // the fold have a zero B-factor - the confidence head has not run - so under
  // the pLDDT scheme they are the colour of no confidence at all. Reloading
  // from framePdbs, which all carry the finished structure's pLDDT, colours the
  // whole animation and costs one ingestion of text that is already built.
  if (viewer !== undefined && viewerObject !== undefined && result.framePdbs.length > 0) {
    // 🔴 THE FINISHED STRUCTURE REPLACES THE LAST SAMPLER FRAME, it is not
    // appended after it. The last frame IS that call's output - in flow mode
    // they agree to a fraction of an angstrom - so appending made a redundant
    // extra frame and a play bar that ended on the same picture twice. The
    // returned structure is the authoritative one, so it takes that slot.
    const timeline = [...result.framePdbs.slice(0, -1), result.pdb];
    const camera = { ...viewer.viewerState };
    await loadIntoViewer({
      stem, pdb: timeline[0],
      scores: confidenceJson(chains.join(""), result.confidence),
      a3m: alignment,
      chainLengths: chains.map((chain) => chain.length),
      pae: paeMatrix(result.confidence.predictedAlignedError,
        paeSize(result.confidence.predictedAlignedError)),
      length: paeSize(result.confidence.predictedAlignedError),
      confidence: result.confidence,
    });
    // ...and the reader keeps the view they had. A reload flies to its own,
    // which after watching a fold reads as the structure jumping at the end.
    if (viewer !== undefined) Object.assign(viewer.viewerState, camera);
    // loadIntoViewer names its first frame for a recycle, which is the wrong
    // word for a sampler call.
    const first = viewer?.objectsData?.[viewerObject]?.frames?.[0];
    if (first !== undefined) first.name = first.label = first.title = `${mode}_0`;
    for (const [index, pdb] of timeline.slice(1).entries()) {
      const frame = api.frameFromText(pdb);
      const last = index === timeline.length - 2;
      frame.name = frame.label = frame.title = last ? "final" : `${mode}_${index + 1}`;
      frame.confidence = result.confidence;
      if (last) {
        // The PAE rides on the frame the page lands on, so scrubbing away and
        // back does not blank a matrix that was on screen a moment earlier.
        const size = paeSize(result.confidence.predictedAlignedError);
        frame.pae = paeMatrix(result.confidence.predictedAlignedError, size);
        frame.pae_n = size;
      }
      viewer.addFrame(frame, viewerObject);
    }
    const object = viewer.objects?.find((entry) => entry.name === viewerObject);
    if (object?.frames?.length) viewer.setFrame(object.frames.length - 1);
    forcePlddtColours();
    viewer.render("af3-final");
  }
  updateScoresCard(result.confidence, `${mode} · ${calls}`);
  progress(null);
  // 🔴 BUILT FROM PARTS, because a ligand-only fold has none of the things this
  // line used to state unconditionally: no residues, no chains, and no backbone
  // to measure - `CA-CA NaN Å` was what it printed, which reads as a broken
  // fold rather than as a fold with no protein in it.
  const what = [];
  if (residues > 0) {
    what.push(`${residues} residues`
      + (chains.length === 1 ? "" : ` in ${chains.length} chains`));
  }
  // Named, because a fold that silently ignored the ligand would otherwise
  // report exactly the same line - the residue count is the same either way.
  if (ligandCodes.length > 0) what.push(ligandCodes.join(", "));
  const detail = [`in ${result.seconds.toFixed(0)} s`];
  if (residues > 0) {
    detail.push(result.depth > 1 ? `${result.depth} MSA rows` : "single sequence");
  }
  detail.push(`${recycles + 1} pass${recycles === 0 ? "" : "es"}`);
  detail.push(`pLDDT ${result.meanPlddt.toFixed(1)}`);
  if (Number.isFinite(result.geometry.caca)) {
    detail.push(`CA-CA ${result.geometry.caca.toFixed(2)} Å`);
  }
  status(`AlphaFold 3 · ${what.join(" + ")} · ${detail.join(" · ")}`);
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
  // 🔴 THE LAST FOLD'S NUMBERS GO BEFORE THIS ONE STARTS. The card kept showing
  // a mean pLDDT and a pTM for a structure that was no longer being computed,
  // for as long as the new fold took - which is worse than an empty panel,
  // because a stale number reads as an answer. The structure itself stays: it
  // is still the last thing that WAS predicted, and the page is never blank
  // between folds.
  updateScoresCard(undefined);
  try {
    const entities = entityList.read();
    const enteredProblem = entitiesProblem(entities);
    // A pasted/uploaded A3M remains self-describing: as before, its query may
    // replace an empty or stale entity list. Search and query-only input have
    // no such query row to fall back to, so they require a valid list.
    if (enteredProblem !== null && ["single", "search"].includes(msaMode())) {
      throw new Error(enteredProblem);
    }
    const request = enteredProblem === null
      ? expandEntities(entities) : { chains: [], ligandCodes: [] };
    let chains = request.chains;
    const ligandCodes = request.ligandCodes;
    let sequence = chains.join("");
    const family = modelFamily(chains.length, ligandCodes.length);

    // 🔴 NOTHING TO ALIGN WITHOUT A POLYMER. A ligand-only fold has no sequence
    // to search with, and the search path reports an empty one as a missing
    // sequence - the right message for an empty box, the wrong one for a job
    // that is already complete.
    const alignmentResult = chains.length === 0
      ? null : await alignmentText(chains, signal, family);
    const alignment = typeof alignmentResult === "string"
      ? alignmentResult : (alignmentResult?.text ?? null);
    // A pasted or uploaded A3M is one text and cannot be split into blocks; it
    // becomes the unpaired one, which is what a single alignment means.
    const alignmentBlocks = typeof alignmentResult === "string"
      ? { unpaired: alignmentResult }
      : (alignmentResult?.blocks ?? (alignment === null ? null : { unpaired: alignment }));
    // ...what the model reads. An array means one alignment per chain.
    const alignmentForModel = alignment;
    throwIfAborted(signal);
    if (alignment !== null) {
      // THE ALIGNMENT'S QUERY WINS. An A3M carries its own first record, and
      // folding the box's sequence against somebody else's alignment would be
      // folding two different proteins at once.
      const alignedQuery = parseA3m(alignment).query;
      if (chains.length > 1 && alignedQuery !== sequence) {
        throw new Error("The complex A3M query does not match the colon-separated chain sequences");
      }
      if (chains.length <= 1) {
        sequence = alignedQuery;
        chains = [sequence];
        // The list shows what will be folded, so the row follows the alignment.
        // Ligand rows are kept: an A3M says nothing about them.
        entityList.setChains(chains);
      }
    }
    const chainLengths = chains.map((chain) => chain.length);

    // 🔴 AlphaFold 3 IS A DIFFERENT MODEL BELOW THIS LINE, so it branches here -
    // before AF2's weights are chosen and before anything below assumes a
    // recycle loop over an evoformer. It branches AFTER the alignment, though,
    // and that is the point: search, paste and upload, the query-wins rule and
    // the pairing decision are one implementation for all three models. What
    // differs is only how the A3M is encoded, which is af3MsaFromA3m's job.
    if (family === "af3") {
      await foldWithAf3(chains, alignment, alignmentBlocks, signal, ligandCodes);
      return;
    }

    status("Starting WebGPU");
    const device = await getDevice();
    throwIfAborted(signal);
    // 🔴 MULTIMER ALWAYS TAKES THE A3M DRIVER, even with no alignment. The
    // query-only path is a separate graph that knows nothing about the multimer
    // regime, so selecting multimer there loaded the right weights and ran the
    // WRONG graph - silently, with a plausible number at the end of it. A
    // single sequence becomes a one-row alignment instead.
    // ...and one weight assembly: the A3M driver always wants the full extra
    // stack, so the "single" variant is no longer reachable from the page.
    const variant = "msa";
    const model = await loadModel(variant, (value) => {
      if (signal.aborted) return;
      progress(value.totalBytes === 0 ? 0 : value.loadedBytes / value.totalBytes);
      status(`Loading model · ${(value.loadedBytes / 1048576).toFixed(0)}`
        + ` / ${(value.totalBytes / 1048576).toFixed(0)} MiB`);
    }, signal, family);
    throwIfAborted(signal);
    progress(null);
    const recycles = recycleCount();
    const tolerance = recycleTolerance();
    const seed = randomSeed();
    const passes = recycles + 1;
    const started = performance.now();

    predictionCount += 1;
    const fastaHeader = entityList.header();
    const baseStem = fastaHeader !== null ? safeJobName(fastaHeader) : `prediction_${predictionCount}`;
    let stem = baseStem;
    const existingObjects = new Set((viewer?.objects ?? []).map((o) => o.name));
    let suffix = 1;
    while (existingObjects.has(stem) || predictions.has(stem)) {
      suffix += 1;
      stem = `${baseStem}_${suffix}`;
    }

    // ...a new run draws afresh: the old object stays until the first pass of
    // this one lands, so the page is never blank between folds.
    //
    // 🔴 AND DROPPING THE HANDLE IS WHAT KEEPS THE CARD EMPTY. A setInterval
    // watches the drawn frame and refills the card from it whenever the index
    // moves, so hiding it once is not enough while the previous object is still
    // animating - that poll returns early on a missing viewer.
    viewer = undefined;
    viewerObject = undefined;
    status(`Folding ${sequence.length} residues${chains.length === 1 ? "" : ` in ${chains.length} chains`}`
      + ` · ${passes} pass${passes === 1 ? "" : "es"} · ${family}`);

    // ...DRAWN AS EACH PASS LANDS, not collected and drawn at the end. The
    // first builds the object and the panels; the rest are frames on it.
    let firstPassLanded = undefined;
    let initialLoadPromise = undefined;
    const onRecycle = (recycle, index) => {
      if (signal.aborted) return;
      const distance = index === 0 ? "" : ` · Δ ${recycle.recycleDistance.toFixed(2)} Å`;
      const passText = `Pass ${index + 1} of ${passes}`;
      const iptmText = recycle.confidence.iptm !== undefined ? ` · ipTM ${Number(recycle.confidence.iptm).toFixed(3)}` : "";
      status(`${passText}${distance} · pLDDT ${recycle.confidence.meanPlddt.toFixed(1)}${iptmText}`);
      updateScoresCard(recycle.confidence, `${passText}${distance}`);
      if (index === 0) {
        firstPassLanded = alignedToPrevious(sequence, recycle.structure);
        initialLoadPromise = loadIntoViewer({
          stem,
          pdb: predictionToPdb(sequence, firstPassLanded, recycle.confidence.plddt, chainLengths),
          scores: confidenceJson(sequence, recycle.confidence),
          a3m: alignment,
          chainLengths,
          pae: paeMatrix(recycle.confidence.predictedAlignedError, sequence.length),
          length: sequence.length,
          confidence: recycle.confidence,
        });
      } else {
        appendPass(sequence, chainLengths, recycle, index, firstPassLanded);
      }
    };
    const runProgress = ({ completed, total, waiting }) => {
      if (signal.aborted) return;
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

    const { maxMsaSequences, maxExtraSequences } = maxMsaConfig();
    // 🔴 THE MULTIMER REGIME IS FOUR FACTS, and they travel together. Multimer
    // runs the outer product mean at the top of each block, works in units of
    // 20 angstroms rather than 10, reads chain identity - asym, entity and
    // symmetry - where the monomer reads only a residue index, and RUNS ITS
    // TEMPLATE EMBEDDER WHETHER OR NOT THERE ARE TEMPLATES.
    //
    // That last one is not an option in multimer the way it is in the monomer.
    // `template.enabled` is False for model_1_ptm and True for
    // model_1_multimer_v3, and multimer's embedding wrapper adds the template
    // activation to the pair unconditionally - masking every template off does
    // not zero it, because it reads the pair through a layer norm and adds a
    // learned constant. Skipping it put the pair 30% out from the first block
    // and shattered backbones at high copy counts. Measured against
    // AlphaFold's own forward on the toy oracle, running it takes the trunk
    // from 6.4e-2 to 1.3e-2 and CA RMSD from 1.96 A to 1.02 A - and on float32
    // weights, to 7.9e-7 and 0.000 A.
    const multimer = family === "multimer";
    const regime = multimer
      ? { outerProductMeanFirst: true, positionScale: 20,
        chainAware: true, chainSequences: chains }
      : {};
    // ...?graph=unified runs the MONOMER weights through src/multimer/ instead.
    // With its switches off that graph reproduces the monomer one bit for bit,
    // which is the check that the superset is right; a difference is a graph
    // bug rather than a weights bug.
    const unified = multimer || new URLSearchParams(location.search).get("graph") === "unified";
    // 🔴 ONE PATH, WHETHER OR NOT THERE IS AN ALIGNMENT. A single sequence is an
    // alignment of depth one, and it is folded as such.
    //
    // There used to be a second driver for it, AlphaFoldQueryOnlyGpu, on the
    // grounds that the extra-MSA stack has nothing to attend over with one
    // sequence and can run its pair-only block instead. Measured on this
    // machine, interleaved over five reps at 59 residues, the specialisation is
    // 1.12s against 0.59s - it is 1.9x SLOWER than the general path, not faster
    // - while agreeing with it to 4.9e-5, which is float32 noise.
    //
    // So it bought nothing and cost plenty: being a second driver, it drifted
    // three times. It did not know the multimer regime, it did not receive
    // chainAware, and options added to one were not added to the other. Each
    // drift failed silently with a plausible number.
    const alignmentForDriver = alignment === null ? `>query\n${sequence}\n` : alignmentForModel;
    const prediction = await new (unified ? AlphaFoldUnifiedGpu : AlphaFoldMonomerGpu)(device)
      .predictA3m(
        alignmentForDriver, model.weights, model.featureTables,
        { recycles, randomSeed: seed, maxMsaSequences, maxExtraSequences, chainLengths, tolerance, signal,
          ...regime },
        model.paeBreaks, onRecycle, runProgress);

    progress(null);
    const final = prediction.final;
    const alignedRecycles = prediction.recycles.map((r, i) => ({
      structure: i === 0 ? (firstPassLanded ?? r.structure) : alignedToFirstPass(sequence, r.structure, firstPassLanded),
      confidence: r.confidence,
      recycleDistance: r.recycleDistance,
    }));
    const finalLanded = alignedRecycles[alignedRecycles.length - 1].structure;
    previousFold = {
      sequence,
      structure: finalLanded,
    };
    lastPrediction = {
      stem,
      pdb: recyclesToPdb(sequence, alignedRecycles, chainLengths),
      scores: confidenceJson(sequence, final.confidence),
      a3m: alignment,
      chainLengths,
      recycles: alignedRecycles,
    };
    predictions.set(stem, lastPrediction);
    // 🔴 A SAFETY NET, because the failure it catches is invisible. onRecycle is
    // optional the whole way down, so a model path that accepts the callback and
    // never calls it would produce a finished fold, a "Done" status and an empty
    // page. If nothing drew while the passes ran, draw them all now.
    if (initialLoadPromise !== undefined) {
      await initialLoadPromise;
    } else if (viewer === undefined) {
      await loadIntoViewer({
        stem,
        pdb: lastPrediction.pdb,
        scores: lastPrediction.scores,
        a3m: lastPrediction.a3m,
        chainLengths: lastPrediction.chainLengths,
        pae: paeMatrix(final.confidence.predictedAlignedError, sequence.length),
        length: sequence.length,
        confidence: final.confidence,
      });
    }
    // ...shown beside the PAE panel, which appears at the same moment.
    element("downloads").style.display = "flex";
    updateScoresCard(final.confidence, `Final (Pass ${prediction.recycles.length})`);
    const took = ((performance.now() - started) / 1000).toFixed(1);

    const converged = prediction.recycles.length < passes
      ? ` · converged at ${final.recycleDistance.toFixed(2)} Å after ${prediction.recycles.length} passes`
      : "";
    const finalIptmText = final.confidence.iptm !== undefined ? ` · ipTM ${Number(final.confidence.iptm).toFixed(3)}` : "";
    status(`Done in ${took} s · pLDDT ${final.confidence.meanPlddt.toFixed(1)}`
      + ` · pTM ${final.confidence.ptm.toFixed(3)}${finalIptmText}${converged}`);
  } catch (error) {
    progress(null);
    if (signal.aborted || isAbortError(error)) status("Prediction stopped");
    else {
      // 🔴 THE STACK GOES TO THE CONSOLE, ALWAYS. The status line gets the
      // message because that is what a reader can act on, but a message alone
      // ("Cannot read properties of undefined") names neither the file nor the
      // line, and this catch is wide enough to cover the search, the model and
      // the handoff to the viewer. Swallowing the stack turns a five-minute
      // diagnosis into a bisect.
      console.error("fold failed", error);
      status(error instanceof Error ? error.message : String(error), true);
    }
  } finally {
    if (activeFold === controller) activeFold = undefined;
    setFoldButton("idle");
  }
}

// --- wiring ----------------------------------------------------------------

element("predict").addEventListener("click", (event) => void fold(event));
// ...and only now is it safe to press. See the note on the button in index.html:
// it ships disabled, because until this line runs a click is silently a no-op.
element("predict").disabled = false;

const modeSelect = element("msa-mode");

// 🔴 THE FOOTER'S PRIVACY LINE IS NOT A CONSTANT. It read "All processing is
// performed locally in your browser. No data is uploaded to a server", which
// was true while "None" was the default alignment mode and false the moment
// remote search became it - that mode posts the sequence to the public
// ColabFold MMseqs2 server. The fold itself never leaves the machine either
// way, so the accurate claim depends on the mode, and a page that states the
// stronger one while doing the weaker thing is worse than one that says
// nothing. Written from here so the two cannot drift apart.
const PRIVACY_NOTE = {
  search: ['<i class="fa-solid fa-cloud-arrow-up" style="margin-right: 5px; color: #f59e0b;"></i>',
    "folds locally · MSA via ",
    '<a href="https://colabfold.com" target="_blank" rel="noopener noreferrer"',
    ' style="color: #3b82f6; text-decoration: none;">colabfold.com</a>'].join(""),
  local: ['<i class="fa-solid fa-shield-halved" style="margin-right: 5px; color: #10b981;"></i>',
    "everything runs locally"].join(""),
};

const syncMode = () => {
  const msaToggle = document.getElementById("useMsaToggle");
  const isMsa = msaToggle !== null ? msaToggle.checked : true;
  const msaModeGroup = document.getElementById("msaModeGroup");
  if (msaModeGroup !== null) {
    msaModeGroup.hidden = !isMsa;
  }
  const maxMsaGroup = document.getElementById("maxMsaGroup");
  if (maxMsaGroup !== null) {
    maxMsaGroup.hidden = !isMsa;
  }
  element("msa-text").hidden = !isMsa || modeSelect.value !== "paste";
  element("msa-file").hidden = !isMsa || modeSelect.value !== "upload";
  // ...getElementById rather than element(), which throws on a missing id: the
  // note is index.html's and this file should not require it to exist.
  const note = document.getElementById("privacy-note");
  if (note !== null) {
    note.innerHTML = isMsa && modeSelect.value === "search" ? PRIVACY_NOTE.search : PRIVACY_NOTE.local;
  }
};
modeSelect.addEventListener("change", syncMode);

const msaToggle = document.getElementById("useMsaToggle");
if (msaToggle !== null) {
  msaToggle.addEventListener("change", syncMode);
}
syncMode();

// 🔴 THE MODEL DECIDES WHICH CONTROLS EXIST, and syncMode decides what the MSA
// ones say - so the model listener runs syncModelControls and then syncMode,
// in that order: the second reads the visibility the first just set.
const familySelect = document.getElementById("model-family");
if (familySelect !== null) {
  familySelect.addEventListener("change", () => { syncModelControls(); syncMode(); });
}
document.getElementById("af3-mode")?.addEventListener("change", syncAf3Count);
syncModelControls();

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
const activePrediction = () => {
  const currentName = viewer?.currentObjectName;
  return (currentName ? predictions.get(currentName) : null) ?? lastPrediction;
};

element("download-pdb").addEventListener("click", () => {
  const pred = activePrediction();
  if (pred) download(`${pred.stem}.pdb`, pred.pdb, "chemical/x-pdb");
});
element("download-scores").addEventListener("click", () => {
  const pred = activePrediction();
  if (pred) download(`${pred.stem}_scores.json`, pred.scores, "application/json");
});
