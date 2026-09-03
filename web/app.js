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
// 🔴 mergeSearchedChains IS USED ONLY WHEN A SEARCH IS REUSED, which is why it
// shipped missing from this list. That path needs a cache from an earlier fold
// AND more than one chain, so a first fold never reaches it - and stopping a
// fold partway is one of the few ways to get a filled cache and then fold
// again. test/module-references.test.js now looks for the whole class.
import { generateMmseqs2ComplexMsa, generateMmseqs2Msa, mergeSearchedChains,
  planSearchReuse, searchCacheEntry } from "../src/input/mmseqs2-api.js";
import { isAbortError, throwIfAborted } from "../src/runtime/abort.js";
import { GpuMemoryBudgetError, setMemoryBudget }
  from "../src/runtime/device-memory.js";
import { AF3_COUNTS, af3SequenceProblem, foldAf3, loadAf3Weights } from "./af3-model.js";
import { getDevice, loadModel } from "./model.js";
import { correspondence } from "./align.js";
import { superposeOnto } from "./morph.js";
import { confidenceJson, paeMatrix, predictionToPdb, safeJobName }
  from "./prediction-results.js";
import { complexSequenceProblem } from "./sequence.js";
import { entitiesProblem, expandEntities } from "./entities.js";
import { createEntityList } from "./entity-ui.js";
import { RuntimeEstimator } from "../src/runtime/cost-model.js";
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
const modelFamily = (ligandCount = 0, modificationCount = 0, nucleicCount = 0) => {
  // 🔴 THE CHOICE IS ALWAYS EXPLICIT NOW. "Auto" used to read the chain count
  // and pick between the two AlphaFold 2 models - which made AF2 the silent
  // default for everything and could never choose AF3, so the newest model was
  // the one a reader had to know to ask for. It also meant the page had a
  // state in which what would run was written nowhere on it.
  const choice = document.getElementById("model-family")?.value ?? "af3";
  // 🔴 A LIGAND IS AlphaFold 3 ONLY, and choosing otherwise is refused rather
  // than quietly corrected. AF2 has no ligand tokens at all, so folding a
  // complex with one under AF2 would drop it silently and return a confident
  // structure of the protein alone - which is a different answer to the
  // question that was asked, not a worse one.
  if (ligandCount > 0 && choice !== "af3") {
    throw new Error(`Ligands need AlphaFold 3; the model is set to ${choice}`);
  }
  // 🔴 AND A MODIFIED RESIDUE IS AlphaFold 3 ONLY FOR THE SAME REASON. AF2
  // tokenises one residue per letter and has no way to say that residue 12 is
  // a phosphoserine, so folding under it would drop the modification and return
  // a confident structure of the unmodified chain - which is a different answer
  // to the question, not a worse one. The residue COUNT is unchanged either
  // way, so nothing else on the page would have shown the difference.
  if (modificationCount > 0 && choice !== "af3") {
    throw new Error(`Modified residues need AlphaFold 3; the model is set to ${choice}`);
  }
  // 🔴 AND A NUCLEIC CHAIN IS AlphaFold 3 ONLY, WHICH IS THE LOUDEST OF THE
  // THREE. AF2's alphabet is the twenty amino acids: `ACGT` is not refused
  // there, it is READ - as alanine, cysteine, glycine, threonine - so a DNA
  // chain folded under AF2 comes back as a confident structure of a short
  // peptide that was never asked for, with nothing anywhere saying so.
  if (nucleicCount > 0 && choice !== "af3") {
    throw new Error(`DNA and RNA need AlphaFold 3; the model is set to ${choice}`);
  }
  return choice;
};

// 🔴 "none" IS SPELLED "single" BELOW, and the translation happens here so it
// happens once. Every path downstream already tests for "single" meaning a
// query-only fold; the select says "None" because that is what it does to the
// alignment, not to the sequence.
const msaMode = () => {
  const chosen = element("msa-mode").value;
  return chosen === "none" ? "single" : chosen;
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
 * The status line, with something to press.
 *
 * 🔴 textContent AND A BUTTON, NOT innerHTML. The message can carry a tensor
 * name and a device's own error string, neither of which this page authored -
 * so it goes in as text and the button is built beside it.
 */
function statusWithAction(text, label, title, onClick) {
  const node = document.getElementById("status-message");
  if (node === null) return;
  node.replaceChildren();
  node.classList.add("error");
  node.append(document.createTextNode(text + " "));
  const button = document.createElement("button");
  button.type = "button";
  button.className = "btn btn-grey btn-small status-action";
  button.textContent = label;
  button.title = title;
  button.addEventListener("click", onClick, { once: true });
  node.append(button);
}

/**
 * What a refused allocation looks like to somebody who has to decide about it.
 *
 * 🔴 TWO NUMBERS. What the fold needs, and what it is allowed. Everything else
 * that was here - the tensor's name, the split between what is held and what
 * was asked for, where the ceiling comes from - is true and is not what the
 * reader is deciding on. The tensor name still goes to the console, where the
 * person it helps is already looking.
 *
 * 🔴 AND NOT "FREE RAM", WHICH DOES NOT EXIST TO ASK FOR. No browser API
 * reports free system or GPU memory; navigator.deviceMemory is TOTAL RAM
 * rounded down to a power of two. A "free" figure would be the one number here
 * nobody could check.
 */
function describeBudget(error) {
  const mib = (bytes) => `${Math.round(bytes / 1048576)} MiB`;
  return `Needs ${mib(error.residentBytes + error.bytes)}, over this device's`
    + ` ${mib(error.budgetBytes)} limit.`;
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
  // 🔴 A BAR MUST NOT BE ABLE TO FAIL A FOLD. HTMLProgressElement throws on a
  // non-finite value - "the provided double value is non-finite" - and that
  // exception unwinds through the progress callback into the prediction, which
  // then reports a failure for a fold that was running perfectly. One undefined
  // alignment depth in the cost model did exactly that. The model no longer
  // produces one, and this makes it not matter if it ever does again.
  if (!Number.isFinite(fraction)) {
    bar.dataset.state = "waiting";
    bar.removeAttribute("value");
    return;
  }
  bar.dataset.state = "running";
  bar.value = Math.min(1, Math.max(0, fraction));
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
      // 🔴 THE SEARCH DOES NOT DEPEND ON THE MODEL AND THE MERGE DOES, so
      // changing the model re-merges what is already here rather than asking
      // api.colabfold.com the same question again. It is the one request this
      // page makes off the machine and it is the slow part of a fold, so
      // repeating it to answer a question already answered is the worst thing
      // this path can do.
      //
      // 🔴 EXCEPT WHEN THE NEW MODEL NEEDS PAIRING THE OLD ONE DID NOT SEARCH
      // FOR. Pairing is a second request and only multimer and AF3 make it, so
      // a monomer search has no paired block to re-merge from; the cache is
      // then not usable and the search runs. Reusing it anyway would silently
      // fold a complex with no paired rows.
      const plan = planSearchReuse({ cache: searchCache, chains, family });
      let searched;
      if (plan.reuse === "single") {
        searched = searchCache.raw.single;
        status(`MSA reused · ${searched.depth} sequences`);
      } else if (plan.reuse === "merge") {
        const { chainA3ms, pairedA3ms, depth } = searchCache.raw;
        const merged = mergeSearchedChains({
          sequences: chains, chainA3ms, pairedA3ms, model: family,
        });
        searched = { ...merged, depth };
        status(`MSA reused · ${depth} sequences`);
      } else {
        searched = chains.length === 1
          ? await generateMmseqs2Msa(query, searchOptions)
          : await generateMmseqs2ComplexMsa(chains, searchOptions);
        status(`MSA search found ${searched.depth} sequences`);
        searchCache = { key: plan.key, raw: searchCacheEntry({ chains, searched }) };
      }
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
      iptmCell.style.display = "flex";
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
      multimerCell.style.display = "flex";
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
  const af3 = (document.getElementById("model-family")?.value ?? "af3") === "af3";
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
  // The MSA controls belong to syncMode, which greys Max MSA out when the MSA
  // select reads None. Setting them here as well would give one pair of
  // controls two owners that disagree - so they are not touched here at all,
  // and they mean the same thing for all three models.
  syncMaxMsa();
  if (af3) syncAf3Count();
}

/**
 * The Max MSA dial, rebuilt for the model.
 *
 * 🔴 AlphaFold 3 HAS ONE MSA TRACK AND AlphaFold 2 HAS TWO. AF2 clusters the
 * alignment and runs the leftovers through a SECOND stack, so its dial is a
 * pair - "512:1024" is 512 clusters and 1024 extra sequences. AF3's evoformer
 * truncates one `msa_stack` to `num_msa` and has no extra stack at all: the
 * `extra_msa_target_feat` in its code is a Linear projecting target_feat into
 * the MSA channel, a layer name rather than a track. So the second number does
 * nothing under AF3 - foldWithAf3 already reads only the first - and showing it
 * invites a reader to tune something that is not connected to anything.
 *
 * 🔴 AUTO KEEPS THE PAIR. Which model runs is not known until the fold starts -
 * it depends on the chain count, and on whether there is a ligand - so under
 * Auto the dial shows the form that can express both, and AF3 ignores the half
 * it has no use for. Only an explicit AF3 narrows it.
 */
const MAX_MSA_DEPTHS = [512, 256, 128, 64, 32, 16];

function syncMaxMsa() {
  const select = document.getElementById("max-msa");
  if (select === null) return;
  const af3 = (document.getElementById("model-family")?.value ?? "af3") === "af3";
  const previous = Number.parseInt(select.value, 10);
  const values = MAX_MSA_DEPTHS.map((depth) => (af3 ? String(depth) : `${depth}:${depth * 2}`));
  select.replaceChildren(...values.map((value) => Object.assign(
    document.createElement("option"), { value, textContent: value })));
  // The DEPTH survives the switch, because it is the same quantity either way -
  // only its notation changed, and re-picking it after every model change would
  // be the dial forgetting what it was told.
  const kept = values.find((value) => Number.parseInt(value, 10) === previous);
  select.value = kept ?? values[0];
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
 * The last fold's trunk, so changing only the sampler costs only the sampler.
 *
 * 🔴 THE KEY IS EVERYTHING THE TRUNK READS, and getting it wrong is not a slow
 * fold but a wrong one: a stale trunk produces a structure for the PREVIOUS
 * sequence with a confidence head that agrees with it. Sequence, ligands,
 * alignment, MSA depth, recycles and seed all go in; the sampler and its step
 * count deliberately do not, because they are what this exists to make cheap.
 *
 * 🔴 THE SEED IS IN THERE BECAUSE THE MSA SUBSAMPLE IS SEEDED FROM IT. Before
 * the subsample the seed reached only the sampler's first draw and a new seed
 * would have been free; now it chooses which alignment rows the trunk sees, so
 * a new seed is a new trunk. That is the faithful behaviour - AF3 draws both
 * from one key - and it is the reason "try another seed" is not the cheap path
 * that "try more steps" is.
 */
/**
 * The last MSA search, so changing the model does not repeat it.
 *
 * Keyed on the CHAINS alone: the search is the same whatever will read it, and
 * only the merge is model-specific - see mergeSearchedChains.
 */
let searchCache;

let trunkCache;

/**
 * The last AlphaFold 2 fold's recycle state, so asking for more continues.
 *
 * 🔴 AF2 HAS NO SAMPLER, SO CONTINUATION IS THE ONLY SAVING THERE IS. Every
 * part of an AF2 fold is a recycle - the evoformer stacks run once per pass -
 * so where AF3 reuses a trunk to make re-sampling cheap, here going from three
 * recycles to five simply runs two passes instead of six. `recycles` also
 * carries the earlier passes' results, because the trajectory the page animates
 * is all of them and a continuation returns only the new ones.
 */
let af2Cache;

/** FNV-1a, to key on an alignment without holding a second copy of it. */
const cheapHash = (text) => {
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
};

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
async function foldWithAf3(chains, alignment, alignmentBlocks, signal, ligandCodes = [],
                           modifications = [], chainKinds = []) {
  const sequence = chains.join(":");
  // 🔴 THE COLONS ARE NOT RESIDUES. `sequence` carries them so the featuriser
  // can see the chain split; every length below is the residue count, and a PAE
  // matrix sized from the wrong one is silently the wrong shape.
  const residues = chains.join("").length;
  // 🔴 THE WHOLE MATRIX, LIGAND ROWS AND ALL - AND THE INDEX SPACES ALREADY
  // AGREE, WHICH IS WHY THIS COSTS NOTHING.
  //
  // This used to take the top-left `residues x residues` block, on the
  // reasoning that AF3 scores TOKENS - one per heavy atom for a ligand - so a
  // mixed fold's matrix is wider than what the viewer draws. The second half of
  // that is simply not true: py2Dmol also carries one POSITION per ligand heavy
  // atom, and its parser reads them in file order, which is the order toPdb
  // writes, which is token order. Measured on a ten-residue chain plus a
  // six-atom ligand: AF3 says 16 tokens, py2Dmol says 16 positions, 10 protein
  // then 6 ligand contiguous. The matrix indexes exactly what is on screen.
  //
  // So the crop was throwing away real, correctly-indexed data - reported as
  // the PAE missing the ligand part of a protein+ligand fold. The panel sizes
  // itself from the matrix it is handed (`this.n = paeData.length` for an array
  // of rows), and `pae_n` equal to that width makes its cell-to-residue
  // crossings the identity, so nothing downstream has to be told.
  //
  // A ligand-only fold falls out of the same rule rather than needing the
  // special case it used to have: no residues, and the width is still the
  // width.
  const paeSize = (values) => Math.round(Math.sqrt(values.length));
  // 🔴 ONLY WHEN THERE IS A POLYMER TO CHECK. A ligand-only fold has no
  // sequence, and af3SequenceProblem reports an empty one as "Paste a protein
  // sequence first" - which is the right message for an empty box and the wrong
  // one for a job that is complete without it.
  // ...and only the PROTEIN ones. af3SequenceProblem checks against the twenty
  // amino acids, which every base but A, C and G fails: a DNA chain would be
  // refused here as "T is not one of the twenty", naming a letter that is
  // correct for the row it is in.
  const proteinOnly = chains
    .filter((_, index) => (chainKinds[index] ?? "protein") === "protein").join("");
  if (proteinOnly.length > 0) {
    const problem = af3SequenceProblem(proteinOnly);
    if (problem !== null) throw new Error(problem);
  }

  const mode = document.getElementById("af3-mode")?.value ?? "flow";
  const asked = Number(document.getElementById("af3-count")?.value)
    || AF3_COUNTS[mode].preferred;
  // 🔴 SIXTEEN IS THE FLOOR AND THE DIAL NO LONGER OFFERS LESS, so this is
  // insurance rather than policy - a stale stored value or a hand-edited option
  // is the only way below it now. AF3_COUNTS carries the measurements and the
  // reason; the short version is that a modified residue's atoms are each their
  // own token and eight steps leaves them compressed (0.835 against a control
  // of 1.003) while sixteen does not (0.974).
  const calls = Math.max(asked, modifications.length > 0 ? 16 : 0);
  const recycles = recycleCount();
  const { requested: maxMsaSequences } = maxMsaConfig();
  // 🔴 RECYCLES ARE NOT IN THE KEY, because more of them is a CONTINUATION
  // rather than a different question: the cached trunk is the recycle state, so
  // going from three to five runs two passes. Fewer is not a continuation -
  // nothing can undo a pass - so the cache is offered only when it is at or
  // behind what was asked for.
  const trunkKey = JSON.stringify({
    // ...modifications included, or a fold that only adds one reuses the trunk
    // of the fold without it and silently ignores what was asked for.
    // 🔴 THE KINDS ARE IN THE KEY BECAUSE THE LETTERS DO NOT IMPLY THEM. Folding
    // `ACGT` as a protein and then as DNA is two different questions with the
    // same `chains`, and without this the second reuses the first one's trunk.
    chains, chainKinds, ligandCodes, modifications, maxMsaSequences, seed: randomSeed(),
    alignment: alignmentBlocks === null ? null : cheapHash(JSON.stringify(alignmentBlocks)),
  });
  const cached = trunkCache?.key === trunkKey ? trunkCache.reusable : undefined;
  const reuse = cached !== undefined && cached.recycles <= recycles ? cached : undefined;
  const continued = reuse !== undefined && reuse.recycles < recycles;

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
  const result = await foldAf3({
    sequence, mode, calls, recycles, weights, device, signal,
    alignment: alignmentBlocks, maxMsaSequences, ligandCodes, modifications,
    chainKinds, reuse,
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
  // Kept for the next fold, and kept even when it was itself reused, so a run
  // of re-samples all skip the trunk rather than only the first.
  trunkCache = { key: trunkKey, reusable: result.reusable };

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
    // 🔴 AND THE PREDICTION IS REGISTERED, WHICH AF3 NEVER DID. The download
    // buttons read `predictions`, and only the AlphaFold 2 path ever wrote to
    // it - so an AF3 fold produced a structure on screen with no way to save
    // what the model actually computed, and the panel holding those buttons
    // stayed hidden. The trajectory goes in as one model per sampler call,
    // which is the AF3 analogue of one model per recycle.
    lastPrediction = {
      stem,
      // 🔴 THE FINAL STRUCTURE ONLY, NOT THE TRAJECTORY. Saving every sampler
      // step wrote a file whose MODEL 1 was the FIRST step - measured at a
      // CA-CA of 2.63 A against the final 3.87 - so anything that opens the
      // first model, which is most things, showed a collapsed structure with
      // backbone that does not join up. The trajectory is on screen in the play
      // bar, where it can be watched; what gets saved is the answer.
      pdb: result.pdb,
      scores: confidenceJson(chains.join(""), result.confidence),
      a3m: alignment,
      chainLengths: chains.map((chain) => chain.length),
    };
    predictions.set(stem, lastPrediction);
    element("downloads").style.display = "flex";
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
  // ...and the same argument applies twice over to a modified residue, whose
  // residue COUNT is unchanged by definition: "59 residues" is the line either
  // way, so the only evidence the modification was applied is this.
  if (modifications.length > 0) {
    const named = modifications.map((one) => `${one.code}${one.position}`);
    const shown = named.length > 3 ? `${named.slice(0, 3).join(", ")} +${named.length - 3}`
      : named.join(", ");
    what.push(shown);
  }
  const detail = [`in ${result.seconds.toFixed(0)} s`
    // Said out loud, because a fold that took a third of the time it used to
    // otherwise reads as something having gone wrong.
    + (reuse === undefined ? ""
      : continued ? ` (${recycles - reuse.recycles} more recycle${
        recycles - reuse.recycles === 1 ? "" : "s"})`
        : " (trunk reused)")];
  // ...beside the timing rather than beside the sequence, because it is a
  // statement about how the fold was RUN and not about what was folded.
  if (calls > asked) detail.push(`${calls} steps, raised from ${asked} for the modification`);
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
      ? expandEntities(entities)
      : { chains: [], chainKinds: [], ligandCodes: [], modifications: [] };
    let chains = request.chains;
    let chainKinds = request.chainKinds ?? chains.map(() => "protein");
    const ligandCodes = request.ligandCodes;
    const modifications = request.modifications ?? [];
    let sequence = chains.join("");
    const nucleicCount = chainKinds.filter((kind) => kind !== "protein").length;
    const family = modelFamily(ligandCodes.length, modifications.length, nucleicCount);

    // 🔴 THE ALIGNMENT COVERS THE PROTEIN CHAINS AND NOTHING ELSE, which is
    // what an A3M can mean and what featuriseProtein reads it as: its columns
    // are matched to the protein residues, in chain order, and a nucleic chain
    // has none. Searching with a DNA chain in the query would send `ACGT` to a
    // protein database as a four-residue peptide and align whatever came back
    // over the wrong chain.
    const proteinChains = chains.filter((_, index) => chainKinds[index] === "protein");
    // 🔴 SAID OUT LOUD, BECAUSE THE ALTERNATIVE IS A SILENT DIFFERENCE. With
    // the MSA set to Search, a job that is part DNA gets an alignment for its
    // protein chains and none for the rest - which is what AF3 does for DNA and
    // is NOT what it does for RNA, where the real pipeline searches an RNA
    // database this page has no server for. Either way the reader asked for an
    // alignment and is getting one for some of their chains, so the status line
    // says which.
    if (nucleicCount > 0 && msaMode() !== "single") {
      const kinds = [...new Set(chainKinds.filter((kind) => kind !== "protein"))]
        .map((kind) => kind.toUpperCase()).join(" and ");
      status(proteinChains.length === 0
        ? `${kinds} folds from its own sequence; there is no alignment to search for it`
        : `Aligning the protein chains only - ${kinds} folds from its own sequence`);
    }
    // 🔴 NOTHING TO ALIGN WITHOUT A POLYMER. A ligand-only fold has no sequence
    // to search with, and the search path reports an empty one as a missing
    // sequence - the right message for an empty box, the wrong one for a job
    // that is already complete. A DNA-only fold is the same case: there is no
    // protein to search with, and no RNA database here to search instead.
    const alignmentResult = proteinChains.length === 0
      ? null : await alignmentText(proteinChains, signal, family);
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
      // ...against the PROTEIN chains, since those are the ones it covers.
      const proteinSequence = proteinChains.join("");
      if (proteinChains.length > 1 && alignedQuery !== proteinSequence) {
        throw new Error("The complex A3M query does not match the colon-separated chain sequences");
      }
      // 🔴 AND THE ALIGNMENT ONLY REPLACES THE ENTITY LIST WHEN THE LIST IS ONE
      // PROTEIN. An A3M says nothing about a DNA chain or a ligand, so letting
      // its query become "the chains" on a mixed job would silently delete
      // every other chain in it.
      if (proteinChains.length <= 1 && nucleicCount === 0) {
        sequence = alignedQuery;
        chains = [sequence];
        chainKinds = ["protein"];
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
      await foldWithAf3(chains, alignment, alignmentBlocks, signal, ligandCodes,
                        modifications, chainKinds);
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
    // 🔴 THE UNITS ARE COSTS, NOT COUNTS, and that is what makes a clock
    // possible. src/model/*.js weight every step by what the cost model says it
    // costs, so `completed / total` is a fraction of the WORK - and the ratio
    // of elapsed time to work done is this machine's speed, whatever it is.
    // RuntimeEstimator holds that reasoning; a plan of one stage is enough for
    // it, since the weighting has already happened upstream.
    let runEstimator = null;
    const runProgress = ({ completed, total, waiting }) => {
      if (signal.aborted) return;
      if (waiting) {
        const bar = element("progress");
        bar.hidden = false;
        bar.removeAttribute("value");
        status("Folding…");
        return;
      }
      runEstimator ??= new RuntimeEstimator({ stages: [{ name: "fold", units: total, count: 1 }] });
      runEstimator.completedUnits(completed);
      progress(runEstimator.fraction());
      // See the note on `say` in web/af3-model.js: the percentage, and nothing
      // beside it that moves on its own.
      const percent = Math.min(100, Math.round(100 * runEstimator.fraction()));
      status(`Folding · ${percent}%`);
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
    // 🔴 THE KEY IS EVERYTHING A PASS READS, for the reason the AF3 one gives:
    // a stale state is not a slow fold but a structure for another sequence.
    // Recycles are absent because more of them is a continuation; the tolerance
    // is present because it decides when the passes STOP.
    const af2Key = JSON.stringify({
      sequence, chainLengths, maxMsaSequences, maxExtraSequences, seed, tolerance,
      unified, family, alignment: cheapHash(alignmentForDriver),
    });
    const af2Cached = af2Cache?.key === af2Key ? af2Cache : undefined;
    const resume = af2Cached !== undefined && af2Cached.resumable.recycles < recycles
      ? af2Cached.resumable : undefined;
    // 🔴 AND THE FRAME EVERY PASS IS SUPERPOSED ONTO COMES BACK WITH IT. The
    // reference is the FIRST pass's landed structure, and a continuation does
    // not run pass zero - onRecycle receives the absolute index, so its
    // `index === 0` branch never fires. Without this the fold is right and its
    // COORDINATES are somewhere else: measured at 0.0007 A RMSD from the fresh
    // three-recycle structure after superposition, which is float noise, but a
    // different file for the same prediction.
    if (resume !== undefined) firstPassLanded = af2Cached.firstPassLanded;
    const prediction = await new (unified ? AlphaFoldUnifiedGpu : AlphaFoldMonomerGpu)(device)
      .predictA3m(
        alignmentForDriver, model.weights, model.featureTables,
        { recycles, randomSeed: seed, maxMsaSequences, maxExtraSequences, chainLengths, tolerance, signal,
          resume, ...regime },
        model.paeBreaks, onRecycle, runProgress);

    progress(null);
    const final = prediction.final;
    // 🔴 THE EARLIER PASSES COME BACK FOR THE ANIMATION. A continuation returns
    // only the passes it ran, and the play bar is the whole trajectory - so the
    // cached ones are put back in front of them. `final` is still the last pass
    // actually computed, which is the one the page lands on.
    const allRecycles = resume === undefined
      ? prediction.recycles : [...af2Cached.recycles, ...prediction.recycles];
    af2Cache = { key: af2Key, resumable: prediction.resumable, recycles: allRecycles,
      firstPassLanded };
    const alignedRecycles = allRecycles.map((r, i) => ({
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
      // The final pass, for the reason above: the earlier recycles are the
      // route, not the result.
      pdb: predictionToPdb(sequence, finalLanded, final.confidence.plddt, chainLengths),
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
    updateScoresCard(final.confidence, `Final (Pass ${allRecycles.length})`);
    const took = ((performance.now() - started) / 1000).toFixed(1);

    const converged = allRecycles.length < passes
      ? ` · converged at ${final.recycleDistance.toFixed(2)} Å after ${allRecycles.length} passes`
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
      // 🔴 A CEILING IS A CHOICE, SO IT IS OFFERED BACK. This one is a GUESS -
      // a third of what the browser admits the machine has - and it is
      // deliberately conservative, so it will sometimes refuse a fold that
      // would have finished. The reader is the one who knows what else is
      // running, so they get the numbers and a button rather than a dead end.
      //
      // 🔴 AND THE WARNING IS REAL, NOT A FORMALITY. The ceiling exists because
      // Metal accepts allocations well past the point where macOS starts
      // paging and reports nothing: without it the failure is not an error
      // message, it is a machine that stops responding. That is what the title
      // on the button says, in those words.
      if (error instanceof GpuMemoryBudgetError && !ceilingLifted) {
        statusWithAction(
          describeBudget(error),
          "Fold anyway",
          "Lifts the limit for this session. The limit is what turns running out"
          + " of memory into a message; without it the tab may stop responding.",
          () => { void foldWithoutCeiling(); });
      } else status(error instanceof Error ? error.message : String(error), true);
    }
  } finally {
    if (activeFold === controller) activeFold = undefined;
    setFoldButton("idle");
  }
}

// --- wiring ----------------------------------------------------------------

/**
 * Whether the reader has taken the ceiling off, which lasts the session.
 *
 * 🔴 ASKED ONCE, NOT EVERY TIME. A second refusal after the button has been
 * pressed is not the same question - the ceiling is already gone, so whatever
 * failed the second time failed for another reason and offering the same
 * button again would be a loop.
 */
let ceilingLifted = false;

/** Take the ceiling off this device and fold again. */
async function foldWithoutCeiling() {
  ceilingLifted = true;
  status("Folding without the memory ceiling…");
  const device = await getDevice();
  setMemoryBudget(device, undefined);
  await fold();
}

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
  const isMsa = modeSelect.value !== "none";
  // 🔴 GREYED OUT, NOT HIDDEN. Max MSA means nothing without an alignment, but
  // removing it moves every control beside it - the row reflows on a change
  // that did not concern them - and a reader who set a depth once cannot see
  // what it still says. Disabled keeps it legible and inert.
  const maxMsaGroup = document.getElementById("maxMsaGroup");
  if (maxMsaGroup !== null) {
    maxMsaGroup.classList.toggle("fetch-option-disabled", !isMsa);
    const select = document.getElementById("max-msa");
    if (select !== null) select.disabled = !isMsa;
  }
  element("msa-text").hidden = modeSelect.value !== "paste";
  element("msa-file").hidden = modeSelect.value !== "upload";
  // ...getElementById rather than element(), which throws on a missing id: the
  // note is index.html's and this file should not require it to exist.
  const note = document.getElementById("privacy-note");
  if (note !== null) {
    note.innerHTML = modeSelect.value === "search" ? PRIVACY_NOTE.search : PRIVACY_NOTE.local;
  }
};
modeSelect.addEventListener("change", syncMode);

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
