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
import { distogramContactProbabilities } from "../src/heads/distogram.js";
import { GpuMemoryBudgetError, setMemoryBudget }
  from "../src/runtime/device-memory.js";
import { AF3_COUNTS, af3SequenceProblem, foldAf3, loadAf3Weights } from "./af3-model.js";
import { getDevice, loadModel } from "./model.js";
import { AF3_FAMILIES } from "../src/reference/manifests/index.js";
import { devBeginRun, devEndRun, devNote, devStatus, devUseDevice } from "./dev-log.js";
import { installDevPanel } from "./dev-panel.js";
import { correspondence } from "./align.js";
import { superposeOnto } from "./morph.js";
import { CHAIN_IDS, confidenceJson, paeMatrix, predictionToPdb, safeJobName }
  from "./prediction-results.js";
import { complexSequenceProblem } from "./sequence.js";
// 🔴 SHARED WITH proteinhunter.html, which shows the same card against its
// own play bar. See web/scores-card.js.
import { updateScoresCard } from "./scores-card.js";
import { entitiesProblem, expandEntities, templateKind } from "./entities.js";
import { buildFoldArchive, msasFromArchive } from "./fold-archive.js";
import { looksLikeZip, readZip, writeZip } from "./zip.js";
import { createEntityList } from "./entity-ui.js";
import { describeCoverage, fetchStructure } from "./template-source.js";
import { fetchMmseqs2Templates } from "../src/input/mmseqs2-api.js";
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
    value: "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK" }],
    // See createEntityList: the entity list owns the template kind and this
    // page owns the MSA control, so it answers rather than reaching for it.
    msaIsSearch: () => msaMode() === "search" });

// 🔴 EXPOSED FOR tools/fold-in-page.py, WHICH HAS NO OTHER WAY IN. The rows are
// built by entity-ui.js and their model is a closure; a harness that wrote into
// a row's field would leave that model behind the DOM, and the fold would run
// on what the model still held. `set` is the same call the paste path makes.
window.__entityList = entityList;

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
/**
 * What the model row is set to, and whether that is an AlphaFold 3 graph.
 *
 * 🔴 TWO BUNDLES BUILD THAT GRAPH NOW, so `=== "af3"` no longer means what its
 * five call sites meant by it. Every one of them was asking "is this the AF3
 * pipeline", not "is this DeepMind's checkpoint", and a second AF3-graph family
 * would have taken the AlphaFold 2 branch at each - which is a page that runs
 * the wrong driver rather than one that says so.
 */
/**
 * Ask about AlphaFold 3's model parameters, once per browser.
 *
 * 🔴 THE PARAMETERS ARE NOT THIS PROJECT'S TO LICENCE. LocalFold's own code is
 * one thing and DeepMind's weights are another: they permit non-commercial use
 * only and carry a prohibited-use policy, and nothing here grants anybody
 * anything. `tools/build_site.py` already refuses to PUBLISH them without
 * LOCALFOLD_ACCEPT_MODEL_TERMS - but that asks the deployer, and the terms are
 * addressed to whoever folds.
 *
 * 🔴 AND IT OFFERS SOMEWHERE ELSE TO GO. A dialog whose only button is "I
 * agree" is a toll gate and teaches people to click through it. OpenBind runs
 * the same graph under Apache 2.0, so this is a choice between two models
 * rather than an obstacle in front of one - which is also why the switch is the
 * button styled as the primary action.
 *
 * @param {string} family what the model row is set to
 * @returns {Promise<string|null>} the family to fold with, or null if the
 *   dialog was dismissed - which cancels the fold rather than picking for them.
 */
async function agreeModelTerms(family) {
  if (family !== "af3" || termsAccepted()) return family;
  const dialog = document.getElementById("model-terms");
  // 🔴 NO DIALOG MEANS NO FOLD IS BLOCKED. single.html and the bundled offline
  // build do not carry this markup, and a missing element must not make the
  // page unfoldable - the deploy-side gate still stands either way.
  if (dialog === null || typeof dialog.showModal !== "function") return family;

  dialog.returnValue = "";
  dialog.showModal();
  await new Promise((resolve) => dialog.addEventListener("close", resolve, { once: true }));

  if (dialog.returnValue === "accept") {
    rememberTermsAccepted();
    return "af3";
  }
  if (dialog.returnValue === "openbind") {
    // 🔴 THE ROW IS UPDATED, NOT JUST THE FOLD. Folding with a model the
    // control does not name is a page whose state is written nowhere on it -
    // the same fault the "Auto" model setting had before it was removed.
    const select = document.getElementById("model-family");
    if (select !== null) {
      select.value = "openbind";
      syncModelControls();
      syncMode();
    }
    return "openbind";
  }
  // Escape, or a click on the backdrop. Not an answer, so not a fold.
  return null;
}

/**
 * Whether this browser has already accepted AlphaFold 3's parameter terms.
 *
 * 🔴 EVERY READ AND WRITE IS GUARDED. localStorage throws outright in a few
 * contexts - a browser set to block site data, some private windows - and an
 * exception here would stop a fold that has nothing to do with storage. A
 * failure to remember means being asked again, which is the safe direction.
 */
const TERMS_KEY = "localfold.modelTerms.alphafold3";

function termsAccepted() {
  try {
    return globalThis.localStorage?.getItem(TERMS_KEY) === "accepted";
  } catch {
    return false;
  }
}

function rememberTermsAccepted() {
  try {
    globalThis.localStorage?.setItem(TERMS_KEY, "accepted");
  } catch {
    // Asked again next time, which is better than a fold that cannot start.
  }
}

/**
 * `?model=` in the URL, so a link can name which model it means.
 *
 * 🔴 THE SELECT'S OWN OPTIONS ARE THE AUTHORITY, not a list written here. A
 * build that ships without a bundle drops its option, and a URL pointing at a
 * model this page does not have must not leave the row set to something it
 * cannot load.
 *
 * 🔴 AND A URL CANNOT ACCEPT ANYBODY'S TERMS. `?model=af3` selects AlphaFold 3
 * and nothing more - the licence dialog still opens on the first fold. A link
 * that could dismiss it would let one person agree on another's behalf, which
 * is the one thing this whole mechanism exists to prevent.
 *
 * 🔴 AND AN UNKNOWN NAME IS SAID OUT LOUD. A query parameter that is silently
 * ignored looks exactly like one that worked, and the reader finds out from the
 * fold they get. `of3` is deliberately NOT an alias for `openbind`: OpenFold3's
 * preview-2 and its v0.5.0 release are different models with different forward
 * conventions (see src/af3/dialect.js), so quietly resolving one to the other
 * would hand somebody a model they did not ask for.
 */
const MODEL_ALIASES = { ob: "openbind", af2: "monomer", mono: "monomer",
                        multi: "multimer" };

function applyModelFromUrl() {
  let asked;
  try {
    asked = new URL(globalThis.location?.href ?? "").searchParams.get("model");
  } catch {
    return;
  }
  if (asked === null || asked.trim() === "") return;
  const select = document.getElementById("model-family");
  if (select === null) return;
  const wanted = MODEL_ALIASES[asked.trim().toLowerCase()] ?? asked.trim().toLowerCase();
  const offered = [...select.options].map((option) => option.value);
  if (!offered.includes(wanted)) {
    // 🔴 RECORDED, NOT WRITTEN HERE. The viewer's own "Ready." message lands
    // asynchronously AFTER this runs and overwrites the status line, so a
    // complaint written now is a complaint nobody sees - which is precisely
    // the silently-ignored parameter this exists to prevent. Reported once the
    // page is ready instead; see reportModelFromUrl.
    modelFromUrlProblem = `This page has no model called "${asked}" - it offers `
      + `${offered.join(", ")}. Folding with ${select.value}.`;
    return;
  }
  select.value = wanted;
}

let modelFromUrlProblem;

/**
 * Say so, once the page has finished writing its own opening line over ours.
 *
 * 🔴 A POLL, BECAUSE THE MESSAGE COMES FROM THE VENDORED BUNDLE. `Ready.` is
 * set from `window.py2dmolReadyMessage` inside py2Dmol's initialisation, not
 * from anything here, so there is no callback to hang this on and no ordering
 * to rely on. It waits for the line to say something and then replaces it,
 * giving up rather than looping forever if it never does.
 */
function reportModelFromUrl(attempt = 0) {
  if (modelFromUrlProblem === undefined) return;
  const node = document.getElementById("status-message");
  if (node === null) return;
  // 🔴 WAIT FOR THE OPENING LINE, NOT FOR ANY LINE. The first attempt at this
  // waited for the status to be non-empty, which it already was - so the
  // complaint was written and then overwritten a moment later by exactly the
  // message it was waiting for. What it has to wait for is that specific
  // string, which the page hands the viewer in index.html.
  const ready = globalThis.py2dmolReadyMessage;
  if (typeof ready === "string" && node.textContent !== ready && attempt < 60) {
    setTimeout(() => reportModelFromUrl(attempt + 1), 50);
    return;
  }
  status(modelFromUrlProblem, true);
  modelFromUrlProblem = undefined;
}

const chosenFamily = () => document.getElementById("model-family")?.value ?? "af3";
const isAf3Family = (family) => AF3_FAMILIES.includes(family);

/** What to call each model while its weights download. */
const MODEL_LABELS = {
  af3: "AlphaFold 3",
  openbind: "OpenBind",
  monomer: "AlphaFold 2",
  multimer: "AlphaFold 2",
};

const modelFamily = (ligandCount = 0, modificationCount = 0, nucleicCount = 0) => {
  // 🔴 THE CHOICE IS ALWAYS EXPLICIT NOW. "Auto" used to read the chain count
  // and pick between the two AlphaFold 2 models - which made AF2 the silent
  // default for everything and could never choose AF3, so the newest model was
  // the one a reader had to know to ask for. It also meant the page had a
  // state in which what would run was written nowhere on it.
  const choice = document.getElementById("model-family")?.value ?? "af3";
  // 🔴 A LIGAND IS AlphaFold 3 ONLY, and choosing otherwise is refused rather
  // than quietly corrected.
  //
  // 🔴 "AlphaFold 3" HERE MEANS THE GRAPH, NOT DEEPMIND'S CHECKPOINT. OpenBind
  // runs the same featuriser and the same token layout, so it has ligand
  // tokens, modified residues and nucleic chains exactly as AF3 does - what
  // differs is whose parameters are in it. Written as `choice !== "af3"` this
  // refused every one of those inputs the moment somebody switched models,
  // with a message naming a capability the model actually has. AF2 has no ligand tokens at all, so folding a
  // complex with one under AF2 would drop it silently and return a confident
  // structure of the protein alone - which is a different answer to the
  // question that was asked, not a worse one.
  if (ligandCount > 0 && !isAf3Family(choice)) {
    throw new Error("Ligands need an AlphaFold 3 model - AF3 or OpenBind;"
      + ` the model is set to ${choice}`);
  }
  // 🔴 AND A MODIFIED RESIDUE IS AlphaFold 3 ONLY FOR THE SAME REASON. AF2
  // tokenises one residue per letter and has no way to say that residue 12 is
  // a phosphoserine, so folding under it would drop the modification and return
  // a confident structure of the unmodified chain - which is a different answer
  // to the question, not a worse one. The residue COUNT is unchanged either
  // way, so nothing else on the page would have shown the difference.
  if (modificationCount > 0 && !isAf3Family(choice)) {
    throw new Error("Modified residues need an AlphaFold 3 model - AF3 or OpenBind;"
      + ` the model is set to ${choice}`);
  }
  // 🔴 AND A NUCLEIC CHAIN IS AlphaFold 3 ONLY, WHICH IS THE LOUDEST OF THE
  // THREE. AF2's alphabet is the twenty amino acids: `ACGT` is not refused
  // there, it is READ - as alanine, cysteine, glycine, threonine - so a DNA
  // chain folded under AF2 comes back as a confident structure of a short
  // peptide that was never asked for, with nothing anywhere saying so.
  if (nucleicCount > 0 && !isAf3Family(choice)) {
    throw new Error("DNA and RNA need an AlphaFold 3 model - AF3 or OpenBind;"
      + ` the model is set to ${choice}`);
  }
  return choice;
};

// 🔴 "none" IS SPELLED "single" BELOW, and the translation happens here so it
// happens once. Every path downstream already tests for "single" meaning a
// query-only fold. The select says "Single Sequence", which names what is
// FOLDED rather than what is missing - it used to say "None", which reads as
// an absent setting rather than a choice about the input.
const msaMode = () => {
  const chosen = element("msa-mode").value;
  return chosen === "none" ? "single" : chosen;
};

let uploadedA3m = "";
/**
 * The per-chain alignments out of an uploaded archive, when one was uploaded.
 *
 * 🔴 KEPT SEPARATELY FROM `uploadedA3m` BECAUSE THEY ARE NOT THE SAME THING. A
 * bare a3m is one text with no record of which rows were paired; an archive
 * carries the blocks apart, which is the whole reason the archive exists. See
 * msasFromArchive in web/fold-archive.js.
 */
let uploadedMsas;
let predictionCount = 0;
const predictions = new Map();

/**
 * The per-chain alignments a downloaded archive should carry.
 *
 * 🔴 THE SPLIT SURVIVES ONLY WHERE IT EXISTED. A search produces one alignment
 * per chain and, for distinct sequences, a paired block beside it - that is
 * what `searchCache.raw` already holds, and writing it out is a copy rather
 * than a computation. A pasted a3m never had the split, and an uploaded one had
 * it only if it arrived as an archive. Each case is written as what it is; the
 * one thing this must not do is present a merged alignment as chain A's
 * unpaired block, which reads back as a fold nobody ran.
 */
function archiveMsas(chains, alignment) {
  if (msaMode() === "upload" && uploadedMsas !== undefined) return uploadedMsas;
  if (msaMode() === "search" && searchCache?.raw !== undefined) {
    const { chainA3ms, pairedA3ms, single } = searchCache.raw;
    if (chainA3ms !== undefined) {
      return {
        unpaired: chainA3ms,
        paired: chains.map((chain) => pairedA3ms?.get(chain)),
      };
    }
    if (single !== undefined) return { unpaired: [single.a3m] };
  }
  return alignment ? { merged: alignment } : {};
}

/**
 * What the running fold was GIVEN, as opposed to what it produced.
 *
 * 🔴 THE ARCHIVE NEEDS BOTH HALVES AND THEY ARE KNOWN IN DIFFERENT PLACES. The
 * entities, the settings and the alignment are settled in `runFold`, before it
 * branches on the model; the structure and the scores exist only inside
 * whichever branch ran. Threading the first set through two long signatures to
 * meet the second was the alternative, and it means every future field is two
 * more parameters on functions that already take eight.
 */
let foldContext = {};

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
  // 🔴 THE TIMELINE IS FED BEFORE THE EARLY RETURN, so a page whose status line
  // is missing still records. It costs one string compare a write, and only a
  // CHANGE of leading segment records a row - the sampler rewriting a
  // percentage several times a second is one phase, not four hundred.
  devStatus(text);
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
/**
 * The model-loading dial, to the right of the status line.
 *
 * 🔴 A SECOND INDICATOR BECAUSE THERE ARE NOW TWO JOBS AT ONCE. The weights and
 * the MSA search used to run one after the other, so a single status line could
 * narrate both. Started together they would fight over it - each overwriting
 * the other's message several times a second, which reads as a page that cannot
 * make up its mind. The dial says how the download is doing without taking the
 * line away from the search.
 *
 * It appears only once a load actually reports itself partway through: a model
 * already in the shard cache resolves without a single progress callback, and a
 * dial that flashed on and off for it would be noise about nothing happening.
 *
 * @param {number|null} fraction 0..1, NaN for a load with no total, null to hide
 * @param {string} [detail] the tooltip, e.g. "AlphaFold 3 · 92 / 150 MiB"
 */
function modelProgress(fraction, detail = "") {
  const node = document.getElementById("model-load");
  if (node === null) return;
  if (fraction === null) {
    node.hidden = true;
    return;
  }
  node.hidden = false;
  node.title = detail;
  node.setAttribute("aria-label", detail);
  const label = node.querySelector("#model-load-text");
  if (label !== null) label.textContent = detail;
  const fill = node.querySelector(".model-load-fill");
  if (fill === null) return;
  // 🔴 THE SAME GUARD THE BAR NEEDED. A non-finite fraction must not be able to
  // fail a fold; here it means "loading, total unknown", which the stylesheet
  // paints as a spin rather than an arc.
  if (!Number.isFinite(fraction)) {
    node.dataset.state = "unknown";
    return;
  }
  const value = Math.min(Math.max(fraction, 0), 1);
  // 2πr for the r=9 circle in the markup; the arc is drawn by holding back the
  // dash rather than by redrawing the path.
  const circumference = 2 * Math.PI * 9;
  fill.style.strokeDasharray = `${circumference}`;
  fill.style.strokeDashoffset = `${circumference * (1 - value)}`;
  node.dataset.state = "loading";
}

/**
 * Begin fetching the model's weights, without waiting for them.
 *
 * 🔴 THE DOWNLOAD AND THE SEARCH DO NOT NEED EACH OTHER, AND USED TO WAIT
 * ANYWAY. The weights were loaded inside the fold, which runs after the
 * alignment - so a cold page with the MSA set to search spent the whole MMseqs2
 * round trip with the network otherwise idle, and then spent the whole download
 * with the search already answered. They are independent: one is a static file
 * from a CDN and the other is a query against a server that queues. Started
 * together the slower one sets the pace, which is the best either can do.
 *
 * Both loaders memoise their promise, so the fold awaiting the same call later
 * gets this one rather than a second download.
 *
 * @returns {Promise<object>} awaited by whichever fold path runs
 */
function startModelPreload(family, signal) {
  const name = MODEL_LABELS[family] ?? "AlphaFold 2";
  // 🔴 THE LABEL MUST NOT CHANGE WIDTH WHILE IT COUNTS. `tabular-nums` holds
  // every DIGIT to one width, which is not the problem: the problem is that the
  // number of digits grows, so "1 / 265" became "10 / 265" became "100 / 265"
  // and the box stepped wider twice per load - moving the dial right and
  // squeezing the status line beside it, twice, during every download.
  //
  // The loaded figure is padded to the width of the total, which is known from
  // the first callback and does not change. U+2007 FIGURE SPACE is the pad: it
  // is defined as the width of a digit, so with tabular digits the string is
  // the same width at 1 MiB as at 265.
  const mib = (bytes) => (bytes / 1048576).toFixed(0);
  const report = ({ loadedBytes = 0, totalBytes = 0 }) => {
    if (signal.aborted) return;
    const total = mib(totalBytes);
    modelProgress(totalBytes === 0 ? NaN : loadedBytes / totalBytes,
      totalBytes === 0
        ? `${name} · ${mib(loadedBytes)} MiB`
        : `${name} · ${mib(loadedBytes).padStart(total.length, "\u2007")}`
          + ` / ${total} MiB`);
  };
  const load = AF3_FAMILIES.includes(family)
    ? loadAf3Weights(report, family)
    : loadModel("msa", report, signal, family);
  // 🔴 A REJECTION HANDLER NOW, OR AN UNHANDLED ONE LATER. Nothing awaits this
  // promise until the fold reaches it, and a download that fails before then is
  // an unhandled rejection - which in a page means a console error and, with
  // some hosts, a reported crash for a fold that goes on to report the failure
  // properly itself. Attaching a handler marks it handled; the original still
  // throws where it is awaited.
  load.then(() => modelProgress(null), () => modelProgress(null));
  return load;
}

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
      // 🔴 AN ARCHIVE RESTORES THE BLOCKS; A BARE a3m NEVER HAD THEM. This is
      // the half of the round trip that makes downloading an alignment worth
      // anything: the per-chain files are merged back through the SAME function
      // the search path uses, so an uploaded archive reaches the model as
      // exactly what its fold reached it as. A single a3m keeps the old
      // meaning - one text, recorded as the unpaired block - because that is
      // genuinely all it says.
      if (uploadedMsas?.chains > 0) {
        const merged = mergeSearchedChains({
          sequences: chains,
          chainA3ms: uploadedMsas.chainA3ms,
          pairedA3ms: new Map(chains.map((chain, index) =>
            [chain, uploadedMsas.pairedA3ms.get(index)])),
          model: family,
        });
        if (uploadedMsas.chains !== chains.length) {
          throw new Error(`that archive holds ${uploadedMsas.chains} alignments`
            + ` and this fold has ${chains.length} chain`
            + `${chains.length === 1 ? "" : "s"}`);
        }
        status(`Alignment from the archive · ${uploadedMsas.chains} chains`);
        return { text: merged.a3m, blocks: merged.blocks };
      }
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
        const { chainA3ms, pairedA3ms, depth, templateHits } = searchCache.raw;
        const merged = mergeSearchedChains({
          sequences: chains, chainA3ms, pairedA3ms, model: family,
        });
        // ...and the hits with them. `mergeSearchedChains` re-merges the
        // ALIGNMENTS and knows nothing about templates, so without this a
        // reused search reports no hits and an automatic template says the MSA
        // is not a search.
        searched = { ...merged, depth, templateHits };
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
      // ...and the template hits, which came out of the same tar and cost
      // nothing. See extractMmseqs2TemplateHits.
      return { text: searched.a3m, blocks: searched.blocks ?? { unpaired: searched.a3m },
               templateHits: searched.templateHits };
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
    // and an estimated pLDDT has to say that it is one.
updateScoresCard(result.confidence);
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

/**
 * The trunk's contact map, as the heatmap panel's byte format.
 *
 * 🔴 IT IS A RESHAPE, NOT A COMPUTATION. The distogram head already sums its
 * bins up to 8 A into P(d <= 8 A) for every pair and the result is already read
 * back to the host, so this costs one pass over tokens^2 bytes and no GPU work
 * at all.
 *
 * 🔴 AND IT NEEDS NO COLOURS OR BOUNDS FROM HERE. `contact` is a scale the
 * panel knows - 0 to 1, white to a dark blue - and a map that states its own
 * would override exactly the thing that makes it read correctly: white is zero
 * and the ink is the signal, which is the opposite of PAE's reading. `vmin`
 * and `vmax` are given because the BYTES are encoded against them and a map
 * that does not say so is trusting two tables to agree.
 *
 * 🔴 IT GOES ON FRAME 0, NOT THE LAST ONE. The panel resolves each map by
 * searching BACKWARD from the frame being drawn, and the contact map is a
 * property of the trunk rather than of any sampler step - fixed for the whole
 * fold - so one copy at the start is on screen for every frame. The PAE stays
 * where it is, on the final frame, because it only exists there.
 */
/**
 * One chain id per residue, in the ids `predictionToPdb` will use.
 *
 * The heatmap only cares where the id CHANGES, but matching the writer means
 * the lines do not move when the real structure replaces this.
 */
function trunkChainIds(chains) {
  const ids = [];
  for (let chain = 0; chain < chains.length; chain += 1) {
    const id = CHAIN_IDS[chain] ?? CHAIN_IDS[CHAIN_IDS.length - 1];
    for (let within = 0; within < chains[chain].length; within += 1) ids.push(id);
  }
  return ids;
}

function contactMapFor(contactProbs) {
  const n = Math.round(Math.sqrt(contactProbs.length));
  if (n * n !== contactProbs.length) return undefined;
  const data = new Uint8Array(n * n);
  for (let index = 0; index < data.length; index += 1) {
    data[index] = Math.max(0, Math.min(255, Math.round(contactProbs[index] * 255)));
  }
  return { data, n, vmin: 0, vmax: 1 };
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
/**
 * Show the viewer, for a fold that has a structure before it has a file.
 *
 * 🔴 THE CONTAINER STARTS `display: none` AND ONLY THE FILE-LOAD PATH OPENS
 * IT. py2Dmol reveals it inside applyPendingObjects, which runs when a file is
 * ingested - so the trunk previews were added to the object, drawn, and
 * displayed inside a hidden container. Frames existed, the panel updated, and
 * the page looked as though nothing had happened until the sampler's first
 * frame arrived through the normal path.
 *
 * 🔴 AND THE CANVAS HAS TO BE RE-MEASURED, because it was sized while hidden
 * and a hidden element measures zero. That is the same trap the heatmap panel
 * documents about its own layout; here it would leave a 0-pixel canvas that
 * never draws.
 */
function revealViewer(renderer) {
  const container = document.getElementById("viewer-container");
  if (container === null || getComputedStyle(container).display !== "none") return;
  container.style.display = "flex";
  const top = document.getElementById("sequence-viewer-container");
  if (top !== null) top.style.display = "block";
  try {
    renderer?._updateCanvasDimensions?.();
  } catch { /* a resize is not worth losing the fold over */ }
}

/**
 * A name no object and no earlier prediction is already using.
 *
 * 🔴 IT READS objectsData, NOT `viewer.objects`, WHICH DOES NOT EXIST. The AF2
 * path checked `viewer?.objects` - an optional chain that always yields
 * undefined on this build - so its uniquifying loop only ever consulted
 * `predictions` and would have collided with any object loaded another way.
 */
function uniqueStem(base) {
  const registry = window.py2dmol_viewers ?? {};
  const renderer = registry[Object.keys(registry)[0]]?.renderer;
  const taken = new Set(Object.keys(renderer?.objectsData ?? {}));
  if (!taken.has(base) && !predictions.has(base)) return base;
  let suffix = 2;
  while (taken.has(`${base}_${suffix}`) || predictions.has(`${base}_${suffix}`)) suffix += 1;
  return `${base}_${suffix}`;
}

/**
 * Open an empty object for the fold that is about to start.
 *
 * 🔴 THE PREVIOUS PREDICTION USED TO STAY ON SCREEN UNTIL THE FIRST FRAME OF
 * THE NEW ONE LANDED, so the page would never be blank. That is the wrong
 * trade: the trunk is the long part - tens of seconds at AF3's sizes, and four
 * passes now that recycles default to three - and for all of it the reader is
 * looking at the LAST fold's structure with this fold's progress bar over it.
 * Nothing marks it stale, and the scores card and the heatmap panel are
 * showing the old numbers too, so all three agree and all three are wrong.
 *
 * 🔴 IT ADDS AN OBJECT RATHER THAN CLEARING THEM ALL. clearAllObjects() would
 * also drop a previous prediction someone is comparing against - py2Dmol holds
 * several and the page's own `predictions` map expects them to survive. With
 * `shownObjects` at its resting state only the CURRENT object draws, so
 * switching to an empty one blanks the view and keeps the rest.
 *
 * 🔴 AND IT RUNS BEFORE THE HANDLES ARE DROPPED, because `viewer` is about to
 * become undefined - that is what stops the score-card poll refilling from
 * whatever is still animating - so this reaches the renderer through the
 * registry instead.
 */
function openBlankFold(stem, keep = []) {
  const registry = window.py2dmol_viewers ?? {};
  const renderer = registry[Object.keys(registry)[0]]?.renderer;
  if (renderer === undefined) return;
  try {
    renderer.addObject(stem);
    // 🔴 addObject KEEPS THE FRAMES OF AN OBJECT THAT ALREADY HAS THEM - "only
    // clear if it has no frames", which is right for a data refresh and wrong
    // for this. A fold whose name repeats therefore APPENDED to the previous
    // run: its frames, its colours and its maps stayed in front of the new
    // ones. This function is called openBlankFold, so it blanks.
    // 🔴 REWOUND, NOT ALWAYS EMPTIED. A fold that continues a cached trunk
    // keeps the frames those passes already produced - only the sampler's are
    // stale - so the object is truncated to them and the new frames append.
    // With nothing to keep this is the blank it says it is.
    const existing = renderer.objectsData?.[stem];
    if (existing?.frames !== undefined) existing.frames.length = 0;
    // 🔴 A KEPT FRAME IS A WHOLE FRAME, NOT JUST COORDINATES. AF3's previews
    // are bare PDB strings, but AF2's recycles each carry their own pLDDT, PAE
    // and contact map - and a rewind that dropped those would put the frames
    // back with no panels behind them, which is worse than losing them.
    for (const [index, entry] of keep.entries()) {
      const api = window.py2Dmol;
      if (api?.frameFromText === undefined) break;
      const spec = typeof entry === "string" ? { pdb: entry } : entry;
      try {
        const frame = api.frameFromText(spec.pdb);
        frame.name = frame.label = frame.title = spec.name ?? `trunk_${index + 1}`;
        if (spec.confidence !== undefined) frame.confidence = spec.confidence;
        if (spec.pae !== undefined) { frame.pae = spec.pae; frame.pae_n = spec.pae_n; }
        if (spec.maps !== undefined) frame.maps = spec.maps;
        if (spec.align) frame.align = true;
        renderer.addFrame(frame, stem);
      } catch { break; }
    }
    // 🔴 AND THIS OBJECT ALONE IS SHOWN. `shownObjects` is a SET of names once
    // anything has toggled object visibility, and addObject ADDS to it - so
    // every previous fold stayed in the set and kept drawing alongside the new
    // one. Two structures in one viewer, coloured by two different folds'
    // confidence, is what "the old run bleeding into this one" looks like.
    // Null is py2Dmol's resting state, where only the current object draws;
    // narrowing the set to this name is the same thing said explicitly.
    if (renderer.shownObjects instanceof Set) {
      renderer.shownObjects = new Set([stem]);
    }
    // 🔴 SWITCHING TO THE OBJECT ALREADY ON SCREEN RESETS THE CAMERA, so a
    // rewind must not ask for it. _switchToObject restores the target's saved
    // viewerState, and that is only ever SAVED when switching away from an
    // object - so for the object already current there is nothing saved and
    // the restore falls back to its default, which is the identity rotation.
    // Measured across a rewind: centre, zoom and extent all held, and the
    // rotation went from [0.732, -0.597, -0.329] to [1, 0, 0]. That is the
    // view jumping.
    if (renderer.currentObjectName === stem) {
      // ...already here; the frames changed under it and nothing else has to.
    } else if (typeof renderer._switchToObject === "function") {
      renderer._switchToObject(stem);
    } else {
      renderer.currentObjectName = stem;
    }
    // ...and the THREE panels that describe a fold must stop describing the old
    // one. The heatmap is told about an object with no frames, which is what
    // makes it hide rather than keep the last matrix up.
    //
    // 🔴 THE MSA WAS THE ONE THAT WAS MISSED, and it is the most visible of
    // them: the panel is populated by py2Dmol's own file ingestion at the END
    // of a fold, so between pressing Fold and the structure landing the
    // previous job's alignment sat there for the whole search and the whole
    // trunk - which on a complex is a minute of a picture of something else.
    //
    // 🔴 BY HAND, BECAUSE THE VENDOR'S TWO WAYS OUT ARE BOTH WRONG HERE.
    // `updateMSAContainerVisibility` reads `#msa-viewer-container`, which is
    // py2Dmol's own site markup and is not in this page - so calling it is a
    // no-op that looks like a fix. `clearAllObjects` does hide the right box,
    // and hides the VIEWER and the sequence strip with it, which is exactly
    // the blank page between folds the note above this refuses. What is left
    // is the box and the viewer's own MSA state. The vendor sets this back to
    // `block` wherever it loads MSA data, so the next fold with an alignment
    // fills it again through loadIntoViewer, as the first one did.
    updateScoresCard(undefined);
    const msaPanel = document.getElementById("msa-buttons");
    if (msaPanel !== null) msaPanel.style.display = "none";
    try { window.MSA?.clear?.(); } catch { /* the viewer's own state; best effort */ }
    window.Heatmap?.updateFrame(renderer, renderer.objectsData?.[stem], 0);
    renderer.render("blank-fold");
  } catch (cause) {
    console.warn("could not open a blank object for this fold", cause);
  }
}

/**
 * Tell the heatmap panel a frame's maps changed.
 *
 * 🔴 `render()` DOES NOT DO THIS, WHICH IS WHY NO CONTACT MAP EVER APPEARED.
 * py2Dmol drives the panel from `setFrame` and from its loader - both call
 * Heatmap.updateFrame - and a plain render redraws the 3D scene without
 * re-resolving which maps the current frame has. So a map attached AFTER the
 * frame was added, which is what computing it off the critical path means,
 * reached the frame object and was never looked at again.
 *
 * 🔴 AND IT MUST NOT MOVE THE VIEW. Re-calling setFrame would work and would
 * yank a reader who has scrubbed elsewhere, so the panel is told about the
 * frame it is ALREADY showing.
 */
function refreshHeatmap() {
  // 🔴 objectsData, NOT `viewer.objects`, WHICH DOES NOT EXIST. Measured on
  // the page: `typeof viewer.objects` is "undefined", so every
  // `viewer.objects?.find(...)` in this file is an optional-chain that
  // silently yields undefined. The frames live in objectsData[name], and the
  // entry there has no `name` of its own - which Heatmap.updateFrame tolerates,
  // because an object with no name skips its owner guards and goes straight to
  // the map resolution this wants.
  const object = viewer?.objectsData?.[viewerObject];
  if (viewer === undefined || object?.frames === undefined) return;
  window.Heatmap?.updateFrame(viewer, object, viewer.currentFrame ?? 0);
  viewer.render("contact");
}

/**
 * The contact map for one AF2 recycle, attached once the frame is on screen.
 *
 * 🔴 EVERY RECYCLE GETS ITS OWN, WHICH IS THE WHOLE POINT. AF2's distogram is
 * recomputed from the pair representation on every pass, so the contact map is
 * a picture of the model changing its mind - and each recycle is already its
 * own frame here, so the panel's backward search lands on the right one. AF3
 * is the opposite: its recycles all finish before the sampler emits a frame,
 * so one map at frame 0 is correct for its whole trajectory.
 *
 * 🔴 AND IT IS COMPUTED OFF THE CRITICAL PATH. The head is L*L*128*64
 * multiply-adds on the CPU - measured 131 ms at 128 residues and 712 ms at 300
 * - which is a few percent of an AF2 fold but enough to stall a paint if it
 * ran before the frame was added. The frame goes up first; this fills the map
 * in after and asks for one more render.
 *
 * 🔴 UNDEFINED WEIGHTS ARE NOT AN ERROR. A bundle from before the head was
 * appended has no distogram section, and losing the contact map is the right
 * price for that - losing the fold is not.
 */
function attachContactMap(frame, recycle, weights, length) {
  if (weights?.distogram === undefined || recycle.pair === undefined) return;
  setTimeout(() => {
    try {
      const head = weights.distogram;
      const contacts = distogramContactProbabilities(
        recycle.pair, head.halfLogitsWeights, head.halfLogitsBias, length,
        { bins: head.bins, first: head.firstBreak, last: head.lastBreak });
      const contact = contactMapFor(contacts);
      if (contact === undefined) return;
      frame.maps = { ...frame.maps, contact };
      // ...and kept, so a rewind can put this frame back without recomputing a
      // head that costs 131 ms at 128 residues and 712 at 300.
      recycle.contactMap = contact;
      // 🔴 AND THE PROBABILITIES THEMSELVES, NOT ONLY THE BYTES. `contactMapFor`
      // quantises to 0-255 for the heatmap, which is all the panel needs and is
      // a lossy thing to put in a results file - the archive writes the same
      // numbers AlphaFold 3 does, so it wants what the head produced.
      recycle.contactProbs = contacts;
      refreshHeatmap();
    } catch (cause) {
      console.warn("contact map unavailable for this pass", cause);
    }
  }, 0);
}

function appendPass(sequence, chainLengths, recycle, recycleIndex, firstPassStructure = undefined,
                    weights = undefined) {
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
  attachContactMap(frame, recycle, weights, sequence.length);
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
  const af3 = isAf3Family(chosenFamily());
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
  const af3 = isAf3Family(chosenFamily());
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
function orientBestView(renderer = viewer) {
  // 🔴 IT TAKES A RENDERER, because the first thing drawn in an AF3 fold is
  // now a trunk preview and `viewer` is deliberately undefined until the
  // sampler's first frame lands. Oriented only at that point, the whole trunk
  // phase was drawn at whatever camera the blank object happened to have.
  if (renderer === undefined) return;
  try {
    if (window.py2dmolOrient?.orientToBestView) {
      window.py2dmolOrient.orientToBestView(renderer, { positions: [], animate: false });
    } else {
      renderer.orient?.({ positions: [] });
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
function forcePlddtColours(renderer = viewer) {
  // ...for the same reason orientBestView takes one: the previews are drawn
  // before `viewer` exists, and without this they are painted by `auto`, which
  // resolves to rainbow rather than to the pLDDT ramp they are coloured for.
  const select = renderer?.colorSelect;
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
                           modifications = [], chainKinds = [], templates = [],
                           modelLoad = undefined, family = "af3") {
  // 🔴 THE FOLD SAYS WHICH MODEL MADE IT, and this is not decoration. Two
  // bundles run this same function; a status line and an archive that both
  // read "AlphaFold 3" for an OpenBind fold are a record of the wrong
  // provenance - and provenance is the whole reason the licence dialog exists.
  const modelName = MODEL_LABELS[family] ?? family;
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
    // 🔴 THE MODEL IS IN THE KEY, AND LEAVING IT OUT BROKE FOLDS SILENTLY. Two
    // bundles build this graph now, and the cached trunk is a PAIR AND SINGLE
    // REPRESENTATION - the same shapes whichever parameters produced them. Fold
    // with OpenBind, switch to AlphaFold 3, fold the same sequence: every other
    // field here matched, so AF3's diffusion head was handed OpenBind's trunk
    // and denoised coordinates out of a representation it had never seen. It
    // does not error, it does not warn, and what comes back is a chain whose
    // atoms are no longer attached to each other.
    //
    // The weight loader has the same hazard and the same answer - one memo per
    // family, not one memo. A cache is the easiest place for a second model to
    // be mistaken for the first.
    family,
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
  // 🔴 A CONTINUATION REWINDS RATHER THAN RESTARTS. Asking for more recycles
  // reuses the trunk and runs only the passes that are missing.

  // 🔴 AWAITED HERE, STARTED LONG AGO. startModelPreload kicked this off before
  // the templates and the alignment, so on a cold page the 150 MB came down
  // beside the MMseqs2 round trip rather than after it.
  //
  // 🔴 AND IT WRITES NOTHING TO THE STATUS LINE. It used to say "Loading
  // AlphaFold 3 · N MiB" there, which was fine while the download was the only
  // thing happening and is not fine now that it runs beside the search: the two
  // overwrite each other, and the message that loses is the one about the
  // server that might be queuing for a minute. The download reports itself on
  // the right instead, dial and label both.
  const weights = await (modelLoad ?? loadAf3Weights());
  throwIfAborted(signal);

  const device = await getDevice();
  throwIfAborted(signal);

  predictionCount += 1;
  const header = entityList.header();
  // 🔴 A HEADER MAKES THE NAME THE SAME EVERY FOLD, which is how the previous
  // run's frames came to be in front of this one's: safeJobName(header) does
  // not change between folds, so every fold reopened the SAME object. The AF2
  // path has always uniquified; this one never did.
  //
  // 🔴 A CONTINUATION REWINDS THE OBJECT IT ALREADY HAS, as the AF2 path does,
  // and for the same reason: opening a new one resets the camera. There is
  // nothing to carry forward now that the trunk draws no structures - the
  // whole trajectory is the sampler's and it is re-run either way - so the
  // rewind is simply an empty object under the name already on screen.
  const stem = reuse === undefined
    ? uniqueStem(header !== null ? safeJobName(header) : `af3_${predictionCount}`)
    : trunkCache.stem;
  // ...and the view goes blank first, so the trunk is not spent showing the
  // previous fold. See openBlankFold.
  openBlankFold(stem);
  // See the note in the AF2 path: dropping the handle is what stops the
  // score-card poll refilling from the object still on screen.
  viewer = undefined;
  viewerObject = undefined;

  const api = window.py2Dmol;
  let pending = Promise.resolve();
  let liveContacts;
  let oriented = false;
  /**
   * Add one frame drawn while the fold is running - a trunk preview or a
   * sampler step - and give it whatever contact map the trunk has produced.
   *
   * 🔴 THE SAME CODE FOR BOTH, which is the point. The sampler's first frame
   * used to go through loadIntoViewer, the virtual-FILE path, which rebuilds
   * the object: the previews were discarded at the handover, the structure
   * blinked, and the heatmap panel lost the frame its map was on. Nothing here
   * is specific to which half of the fold produced the frame.
   */
  let liveSampler = 0;
  const drawLiveFrame = (pdb, kind) => {
    if (signal.aborted || api?.frameFromText === undefined) return;
    const registry = window.py2dmol_viewers ?? {};
    const renderer = registry[Object.keys(registry)[0]]?.renderer;
    const object = renderer?.objectsData?.[renderer?.currentObjectName];
    if (renderer === undefined || object === undefined) return;
    try {
      const index = object.frames.length;
      // ...opened before the frame is added, so the canvas is measured against
      // a container that is actually on screen.
      if (index === 0) revealViewer(renderer);
      const frame = api.frameFromText(pdb);
      // ...numbered by the sampler's own count. Every frame in the object is
      // the sampler's now; the trunk draws none.
      frame.name = frame.label = frame.title = `${kind}_${liveSampler++}`;
      // ...and the map of the pass that produced it, so the panel has
      // something to resolve on every frame rather than only the first.
      if (liveContacts !== undefined) frame.maps = { contact: liveContacts };
      renderer.addFrame(frame, renderer.currentObjectName);
      renderer.setFrame(object.frames.length - 1);
      if (index === 0) {
        // The camera and the palette are set on the FIRST thing drawn, or the
        // fold is watched from the default view in rainbow.
        //
        // 🔴 EXCEPT ON A CONTINUATION, WHICH KEEPS THE VIEW THE READER HAS.
        // A rewind empties the object it is continuing, so the first frame of
        // the new sampler run is index 0 and this fired - flying the camera to
        // the best view of a structure the reader was already looking at.
        // Measured across a continuation, the rotation moved from
        // [0.837, 0.153, 0.526] to [0.797, 0.195, 0.571]: a small tilt, and a
        // tilt nobody asked for. The centre and the focal length still follow
        // the molecule, because a re-sample really does land somewhere else.
        if (reuse === undefined) orientBestView(renderer);
        forcePlddtColours(renderer);
        oriented = true;
      }
      renderer.render("live-frame");
    } catch (cause) {
      console.warn("could not draw a frame", cause);
    }
  };
  const result = await foldAf3({
    sequence, mode, calls, recycles, weights, device, signal,
    alignment: alignmentBlocks, maxMsaSequences, ligandCodes, modifications,
    chainKinds, reuse,
    // 🔴 TEXT AND A CHAIN, NOT A SLOT. foldAf3 places them, because a slot is
    // indexed by TOKEN and a modified residue is several tokens - so a chain's
    // first token is not the sum of the preceding chains' residue counts, and
    // only the featuriser knows the difference.
    templates,
    // 🔴 CACHED WHEN THE TRUNK EXISTS, NOT WHEN THE FOLD FINISHES. This used to
    // be written after foldAf3 resolved, so a fold that hit the memory ceiling
    // in the SAMPLER threw the trunk away with the exception and "Fold anyway"
    // started from featurisation - re-running minutes of work that had already
    // succeeded. An aborted fold now leaves its trunk behind too.
    onTrunk: (reusable) => {
      // ...the carried previews stay with it: they belong to passes this trunk
      // has already run, and a continuation must not lose them.
      trunkCache = { key: trunkKey, reusable, stem };
    },
    // Both modes are seeded now: the flow draws its starting positions once at
    // the top of the schedule.
    seed: randomSeed(),
    onStatus: (text) => { if (!signal.aborted) status(text); },
    onProgress: (fraction) => { if (!signal.aborted) progress(fraction); },
    // 🔴 THE CONTACT MAP ARRIVES BEFORE THE FIRST FRAME DOES, so it is held
    // until there is something to hang it on. The trunk knows it before the
    // sampler runs; the viewer has no object until the first denoiser call
    // lands, and the heatmap panel is driven by an object's frames.
    onContacts: (contactProbs) => {
      liveContacts = contactMapFor(contactProbs);
      // 🔴 AND STRAIGHT TO THE PANEL WHILE THERE IS NO FRAME TO HANG IT ON.
      // The trunk finishes every recycle before the sampler emits anything, so
      // for the longest part of an AF3 fold the viewer holds the blank object
      // openBlankFold made and the panel has nothing to resolve. Pushing the
      // map at the renderer shows it evolving through the recycles; the
      // frame-driven path takes over by itself once frame 0 lands, because
      // that goes through updateFrame.
      if (liveContacts === undefined) return;
      const registry = window.py2dmol_viewers ?? {};
      const renderer = registry[Object.keys(registry)[0]]?.renderer;
      const frames = renderer?.objectsData?.[renderer?.currentObjectName]?.frames;
      if (renderer?.heatmapRenderer === undefined || (frames?.length ?? 0) > 0) return;
      try {
        // 🔴 AND THE CHAIN LAYOUT WITH IT, OR A COMPLEX GETS NO DIVIDER LINES.
        // The panel rules a line wherever the chain changes and reads the
        // chains off the RENDERER, which fills them in when a structure is
        // parsed - so on this path, which exists precisely because there is no
        // structure yet, `renderer.chains` is empty and _drawChainBoundaries
        // returns before drawing anything. A complex's contact map came up
        // unruled for the whole trunk and grew its lines when the sampler's
        // first frame landed, which reads as the panel changing its mind.
        //
        // 🔴 AND WRITTEN EVERY TIME, NOT ONLY WHILE IT IS EMPTY. The guard used
        // to be `if ((renderer.chains?.length ?? 0) === 0)`, to avoid fighting
        // the parser - but the parser fills `chains` from the last structure it
        // PARSED, which on a second fold is the PREVIOUS fold's. So a complex
        // folded after a monomer, or after a complex with different chain
        // lengths, drew the old fold's divider lines across the new fold's
        // contact map for the whole trunk, and they snapped into place when the
        // sampler's first frame landed.
        //
        // There is nothing to fight: this path is only reached when the current
        // object has NO frames - the line above returns otherwise - so nothing
        // has parsed a structure for this fold and these ids are the only
        // authority there is. The parser overwrites them the moment it has one.
        renderer.chains = trunkChainIds(chains);
        renderer.heatmapRenderer.setMaps({ contact: liveContacts });
        window.Heatmap?.updateVisibility?.(renderer);
        renderer.render("trunk-contacts");
      } catch (cause) {
        console.warn("could not show the trunk's contact map", cause);
      }
    },
    // 🔴 A STRUCTURE DURING THE TRUNK, REPLACED EACH RECYCLE. There is nothing
    // else to look at for the longest part of an AF3 fold - the sampler has
    // not started - and one flow cycle against the current trunk is a real
    // backbone. Each preview REPLACES the last: they are the same structure
    // getting better, not a trajectory, and leaving them stacked would put
    // four of them in front of the real one on the play bar.
    // 🔴 THE SAMPLER'S FRAMES GO THE SAME WAY THE PREVIEWS DO. This called
    // loadIntoViewer for index 0 - the virtual-FILE path - which rebuilds the
    // object: the trunk's previews were discarded at that moment, the
    // structure blinked, and the heatmap lost the frame carrying its map.
    // loadIntoViewer still runs, once, at the end of the fold with the
    // finished trajectory, its alignment, its scores and its PAE.
    //
    // 🔴 AND `viewer` IS NOT AVAILABLE HERE ANY MORE, which is why this went
    // through the registry: the handle is deliberately undefined until the
    // final load, so that the score-card poll cannot refill from a fold in
    // progress. drawLiveFrame reaches the renderer the same way openBlankFold
    // does.
    onFrame: (pdb) => { drawLiveFrame(pdb, mode); },
  });
  await pending;
  throwIfAborted(signal);
  // ...`onTrunk` above has already cached this, and it is the same object.
  // Kept for the next fold, and kept even when it was itself reused, so a run
  // of re-samples all skip the trunk rather than only the first.
  trunkCache = { key: trunkKey, reusable: result.reusable, stem };

  // 🔴 THE HANDLES ARE ACQUIRED HERE, because nothing during the fold sets
  // them any more. drawLiveFrame reaches the renderer through the registry so
  // that the score-card poll cannot refill from a fold in progress - which
  // left `viewer` undefined at the replay, and the replay is guarded on it.
  // The whole rebuild was silently skipped: the finished animation kept the
  // LIVE frames, so it carried the raw distogram colours instead of the
  // calibrated ones (83.7 against a real pLDDT of 54.0), no PAE, and no final
  // frame. A guard that turns a rebuild into a no-op is the worst shape of
  // bug; this is the point at which the fold IS finished, so it is where they
  // belong.
  if (viewer === undefined) {
    const registry = window.py2dmol_viewers ?? {};
    viewer = registry[Object.keys(registry)[0]]?.renderer;
    viewerObject = viewer?.currentObjectName;
  }

  // 🔴 THE TRAJECTORY IS RELOADED ONCE THE CONFIDENCE EXISTS. The frames drawn
  // during the fold have a zero B-factor - the confidence head has not run - so
  // under the pLDDT scheme they are the colour of no confidence at all.
  // Reloading from framePdbs colours the whole animation and costs one
  // ingestion of text that is already built.
  //
  // 🔴 AND EACH FRAME NOW CARRIES ITS OWN. They used to all take the finished
  // structure's pLDDT, which is a constant colour on a moving structure; each
  // is now scored on how well it agrees with the trunk's distogram, calibrated
  // to this fold's own pLDDT. See the note in web/af3-model.js: it is a picture
  // of a structure resolving and not a per-residue claim, and the FINISHED
  // structure below still carries the real pLDDT.
  if (viewer !== undefined && viewerObject !== undefined && result.framePdbs.length > 0) {
    // 🔴 THE FINISHED STRUCTURE REPLACES THE LAST SAMPLER FRAME, it is not
    // appended after it. The last frame IS that call's output - in flow mode
    // they agree to a fraction of an angstrom - so appending made a redundant
    // extra frame and a play bar that ended on the same picture twice. The
    // returned structure is the authoritative one, so it takes that slot.
    // 🔴 THE TRAJECTORY IS THE SAMPLER'S, AND ONLY THE SAMPLER'S. The trunk
    // used to contribute one frame per recycle; it draws no structures now, so
    // the play bar starts where the sampler does. The recycles are watched
    // through the contact map instead, which moves per pass and costs nothing.
    const timeline = [...result.framePdbs.slice(0, -1), result.pdb];
    /** What a frame is called: the sampler's calls, then the answer. */
    const frameName = (index, last) => (last ? "final" : `${mode}_${index}`);
    const camera = { ...(viewer?.viewerState ?? {}) };
    // ...and the live frames are dropped first. They are the same structures,
    // drawn with the uncalibrated colour and named by their position in a list
    // that was still growing; loadIntoViewer appends to an object that has
    // frames rather than clearing it, so without this the trajectory is drawn
    // twice, once wrong.
    const live = viewer?.objectsData?.[viewerObject];
    if (live?.frames !== undefined) live.frames.length = 0;
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
      // ...the contacts travel WITH the confidence, because everything that
      // reads one reads the other: the scores file, the archive's full_data,
      // and the heatmap all want the same token-by-token matrices.
      confidence: { ...result.confidence, contactProbs: result.contactProbs },
      scores: confidenceJson(chains.join(""),
        { ...result.confidence, contactProbs: result.contactProbs }),
      a3m: alignment,
      chains,
      chainLengths: chains.map((chain) => chain.length),
      model: modelName,
      ...foldContext,
    };
    predictions.set(stem, lastPrediction);
    element("downloads").style.display = "flex";
    // ...and the reader keeps the view they had. A reload flies to its own,
    // which after watching a fold reads as the structure jumping at the end.
    if (viewer !== undefined) Object.assign(viewer.viewerState, camera);
    // loadIntoViewer names its first frame for a recycle, which is the wrong
    // word for a sampler call.
    const first = viewer?.objectsData?.[viewerObject]?.frames?.[0];
    if (first !== undefined) {
      first.name = first.label = first.title = frameName(0, false);
      // ...and frame zero's own estimate too. It is built by loadIntoViewer
      // rather than by the loop below, so it is easy to leave carrying whatever
      // that put there.
      first.confidence = {
        predictedAlignedError: result.confidence.predictedAlignedError,
        plddt: result.confidence.plddt,
      };
      // ...and frame zero's contact map, which is the finished trunk's: every
      // recycle is over before the sampler emits anything.
      const contact = result.contactProbs === undefined
        ? undefined : contactMapFor(result.contactProbs);
      if (contact !== undefined) first.maps = { ...first.maps, contact };
    }
    for (const [index, pdb] of timeline.slice(1).entries()) {
      const frame = api.frameFromText(pdb);
      const last = index === timeline.length - 2;
      frame.name = frame.label = frame.title = frameName(index + 1, last);
      // 🔴 NO MAP PAST FRAME ZERO, DELIBERATELY. The trunk finishes before the
      // sampler emits anything, so every frame of this trajectory has the same
      // contact map - and the panel resolves a map by searching BACKWARD from
      // the frame drawn, so carrying it once at frame 0 is exactly right and
      // repeating it would be the same picture stored sixteen times.
      // 🔴 THE FRAME'S OWN NUMBER, NOT THE FINISHED ONE. Every frame used to
      // carry `result.confidence`, so scrubbing the trajectory showed the final
      // pLDDT on a structure that had not reached it. An intermediate frame now
      // reports the distogram estimate its own colour is drawn from, and says
      // so; the last frame is the finished structure and keeps the real head's
      // answer.
      //
      // 🔴 AN INTERMEDIATE FRAME HAS NO pLDDT, pTM OR ipTM AND NOW SAYS SO.
      // All three come from the confidence head, which runs once on the
      // finished structure. The card used to show the head's finished numbers
      // on every frame, and then a distogram estimate labelled as a pLDDT;
      // both told the reader something the frame does not support. It shows a
      // dash for all three instead.
      frame.confidence = last ? result.confidence : {
        predictedAlignedError: result.confidence.predictedAlignedError,
        plddt: result.confidence.plddt,
      };
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
  updateScoresCard(result.confidence);
  progress(null);
  // 🔴 BUILT FROM PARTS, because a ligand-only fold has none of the things this
  // line used to state unconditionally: no residues and no chains. It once
  // printed `CA-CA NaN Å` for one, which reads as a broken fold rather than as
  // a fold with no protein in it - that field has since moved to the tools, but
  // the residue and chain counts have the same problem and this is the fix.
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
  // 🔴 THE COVERAGE GOES BACK ON THE ROW THAT ASKED FOR IT, and is the only
  // thing that says a template arrived: a fold that lost one folds and scores,
  // and the number is merely different. `origin` is the entity's own object -
  // see expandEntities - so this reaches the popup the reader opened.
  (result.templateCoverage ?? []).forEach((coverage, index) => {
    const described = describeCoverage(coverage);
    if (templates[index]?.origin !== undefined) {
      templates[index].origin.status = described;
    }
    detail.push(`template ${templates[index]?.source ?? index + 1}:`
      + ` ${coverage.residues}/${coverage.of}`);
  });
  // 🔴 THE BACKBONE CA-CA IS STILL MEASURED AND IS NO LONGER SHOWN. It is the
  // number a wrong sampler cannot fake - docs/AF3.md records a batch with one broken
  // gather folding 17 A of spaghetti at pLDDT 55 - so `foldBatch` keeps
  // computing it and every probe that judges a fold still prints it. But "3.81"
  // means nothing to somebody who wanted a structure, and a status line that
  // ends in a diagnostic reads as a diagnostic. It belongs to the tools.
  status(`${modelName} · ${what.join(" + ")} · ${detail.join(" · ")}`);
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
  devBeginRun(`fold · ${element("model-family").value}`
    + ` · alignment ${element("msa-mode").value}`
    + ` · ${element("recycles").value} recycles`);
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
      : { chains: [], chainKinds: [], ligandCodes: [], modifications: [], templates: [] };
    let chains = request.chains;
    let chainKinds = request.chainKinds ?? chains.map(() => "protein");
    const ligandCodes = request.ligandCodes;
    const modifications = request.modifications ?? [];
    // 🔴 DECIDED BEFORE ANY NETWORK WORK, because the download starts here.
    // Nothing below changes it: the only reassignment of `chainKinds` is the
    // pasted-A3M branch, which runs only where `nucleicCount` is already zero
    // and sets it to the protein it already was.
    const nucleicCount = chainKinds.filter((kind) => kind !== "protein").length;
    let family = modelFamily(ligandCodes.length, modifications.length, nucleicCount);
    // 🔴 THE TERMS ARE ASKED BEFORE THE DOWNLOAD, NOT BEFORE THE PAGE. AF3's
    // parameters carry DeepMind's own terms, and the moment they apply is the
    // moment the bytes are fetched - which is the next line. Asking on page
    // load would put a dialog in front of somebody who came to fold with AF2,
    // and asking afterwards would ask about something already done.
    //
    // It can answer `openbind`, which is why `family` is no longer const: the
    // dialog offers a way past the terms rather than only a way through them,
    // and taking it has to change what this fold loads.
    family = await agreeModelTerms(family);
    if (family === null) {
      status("Fold cancelled - no model chosen.");
      return;
    }
    // ...and started, not awaited. The templates and the alignment below are
    // network work of their own; this runs beside them.
    const modelLoad = startModelPreload(family, signal);
    // 🔴 FETCHED HERE AND NOT INSIDE THE FOLD, so a structure that cannot be
    // reached stops the run with its own message rather than surfacing as a
    // fold that scored badly. AF3 only: AF2's drivers take a template through
    // a different path and nothing on this page builds one for them yet.
    const templateSources = [];
    for (const template of request.templates ?? []) {
      const kind = templateKind(template);
      const source = (template.source ?? "").trim();
      const common = { chain: template.chain, spanChains: template.spanChains === true,
                       origin: template.origin };
      if (kind === "search") {
        // Resolved after the search, which is when the hits exist.
        templateSources.push({ ...common, auto: true });
        continue;
      }
      if (kind === "upload") {
        // 🔴 NOTHING IS FETCHED, AND THE CHAIN BOX MAY BE EMPTY. An uploaded
        // file is already text; `chainResidues` takes the first polymer chain
        // when it is not told which, which is right for the single-chain files
        // most people upload and wrong silently for the rest - hence the box.
        templateSources.push({ ...common, text: template.text,
          chainId: source === "" ? undefined : source,
          source: template.filename ?? "the uploaded structure" });
        continue;
      }
      if (kind === "none" || source === "") continue;
      status(`Fetching template ${source}`);
      const structure = await fetchStructure(source, { signal, kind });
      templateSources.push({ ...common, chainId: structure.chain,
        text: structure.text, source });
    }
    let sequence = chains.join("");

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
    // 🔴 THE AUTOMATIC TEMPLATES ARE RESOLVED AFTER THE SEARCH, because that is
    // when the hits exist. A chain asking for them without a search gets
    // nothing and is told so - single sequence has no hits, and folding
    // silently without the template someone asked for is the failure this
    // whole path is trying to avoid.
    const hits = typeof alignmentResult === "string"
      ? undefined : alignmentResult?.templateHits;
    for (const template of templateSources) {
      if (template.auto !== true) continue;
      const best = (hits?.get(template.chain) ?? [])[0];
      if (best === undefined) {
        throw new Error(hits === undefined
          ? "Automatic templates need an MSA search: set the MSA to search, or"
            + " name a structure instead."
          : `The search found no template for chain ${template.chain + 1}.`);
      }
      status(`Fetching template ${best.target}`);
      const structures = await fetchMmseqs2Templates([best.target], { signal });
      const text = structures.get(best.id);
      if (text === undefined) throw new Error(`No structure came back for ${best.target}`);
      template.text = text;
      template.chainId = best.chain;
      template.source = best.target;
    }
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

    // 🔴 RECORDED BEFORE THE BRANCH, because this is where it is all known. See
    // foldContext: the entities, the settings and the alignment are settled
    // here and the structure exists only inside whichever branch runs.
    foldContext = {
      entities,
      templates: templateSources,
      msas: archiveMsas(chains, alignment),
      msaOrigin: {
        single: "none (single sequence)",
        search: "MMseqs2 search at api.colabfold.com",
        paste: "pasted by hand",
        upload: uploadedMsas === undefined ? "uploaded a3m" : "uploaded archive",
      }[msaMode()] ?? msaMode(),
      settings: {
        seed: randomSeed(),
        recycles: recycleCount(),
        "max msa": maxMsaConfig().requested,
      },
    };

    // 🔴 AlphaFold 3 IS A DIFFERENT MODEL BELOW THIS LINE, so it branches here -
    // before AF2's weights are chosen and before anything below assumes a
    // recycle loop over an evoformer. It branches AFTER the alignment, though,
    // and that is the point: search, paste and upload, the query-wins rule and
    // the pairing decision are one implementation for all three models. What
    // differs is only how the A3M is encoded, which is af3MsaFromA3m's job.
    if (isAf3Family(family)) {
      await foldWithAf3(chains, alignment, alignmentBlocks, signal, ligandCodes,
                        modifications, chainKinds, templateSources, modelLoad, family);
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
    // 🔴 AWAITED, NOT STARTED, and silent on the status line. startModelPreload
    // began this before the alignment and reports itself on the right; see the
    // note in foldWithAf3 for why it no longer writes to the line.
    const model = await modelLoad;
    throwIfAborted(signal);
    progress(null);
    const recycles = recycleCount();
    const tolerance = recycleTolerance();
    const seed = randomSeed();
    const passes = recycles + 1;
    const started = performance.now();

    // 🔴 THE RESUME DECISION COMES BEFORE THE OBJECT, because what the object
    // is rewound TO depends on it. These four were declared further down, next
    // to the model call that reads them; the key needs them here.
    const { maxMsaSequences, maxExtraSequences } = maxMsaConfig();
    const multimer = family === "multimer";
    const unified = multimer || new URLSearchParams(location.search).get("graph") === "unified";
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

    predictionCount += 1;
    const fastaHeader = entityList.header();
    const baseStem = fastaHeader !== null
      ? safeJobName(fastaHeader) : `prediction_${predictionCount}`;
    // 🔴 uniqueStem READS objectsData; the loop that used to be here read
    // `viewer.objects`, which does not exist on this build.
    const stem = resume === undefined ? uniqueStem(baseStem) : af2Cache.stem;

    // 🔴 A CONTINUATION REWINDS THE OBJECT IT ALREADY HAS; IT DOES NOT OPEN A
    // NEW ONE. Asking for more recycles resumes the cached passes and computes
    // only the missing ones - but the page opened a fresh object for it and
    // named it uniquely, so the frames of the passes being resumed were
    // stranded on the previous object and the new one started empty. Measured:
    // a one-recycle fold followed by a three-recycle one left prediction_1 with
    // recycle_0 and recycle_1 and gave prediction_2 a single frame.
    //
    // Keeping the NAME is what makes it a rewind rather than a copy: the
    // alignment, the MSA panel and everything else py2Dmol hangs off an object
    // stay attached, and only the frames are replayed.
    const kept = resume === undefined ? [] : af2Cached.recycles.map((pass, index) => ({
      pdb: predictionToPdb(sequence, index === 0
        ? (af2Cached.firstPassLanded ?? pass.structure)
        : alignedToFirstPass(sequence, pass.structure, af2Cached.firstPassLanded),
        pass.confidence.plddt, chainLengths),
      name: `recycle_${index}`,
      confidence: pass.confidence,
      pae: paeMatrix(pass.confidence.predictedAlignedError, sequence.length),
      pae_n: sequence.length,
      maps: pass.contactMap === undefined ? undefined : { contact: pass.contactMap },
      align: true,
    }));

    // ...a new run draws afresh: the old object stays until the first pass of
    // this one lands, so the page is never blank between folds.
    //
    // 🔴 AND DROPPING THE HANDLE IS WHAT KEEPS THE CARD EMPTY. A setInterval
    // watches the drawn frame and refills the card from it whenever the index
    // moves, so hiding it once is not enough while the previous object is still
    // animating - that poll returns early on a missing viewer.
    openBlankFold(stem, kept);
    viewer = undefined;
    viewerObject = undefined;
    // 🔴 AND A REWIND KEEPS ITS HANDLES, because the passes it is about to run
    // are NOT pass zero. `onRecycle` gets the absolute index, so the branch
    // that calls loadIntoViewer - the only place `viewer` is ever set - never
    // fires on a continuation, and every appendPass returned at its first line.
    // That is why the resumed passes never appeared.
    if (kept.length > 0) {
      const registry = window.py2dmol_viewers ?? {};
      viewer = registry[Object.keys(registry)[0]]?.renderer;
      viewerObject = viewer === undefined ? undefined : stem;
    }
    status(`Folding ${sequence.length} residues${chains.length === 1 ? "" : ` in ${chains.length} chains`}`
      + ` · ${passes} pass${passes === 1 ? "" : "es"} · ${family}`);

    // ...DRAWN AS EACH PASS LANDS, not collected and drawn at the end. The
    // first builds the object and the panels; the rest are frames on it.
    let firstPassLanded = undefined;
    let initialLoadPromise = undefined;
    const onRecycle = (recycle, index) => {
      if (signal.aborted) return;
      // 🔴 A PASS DOES NOT WRITE THE STATUS LINE. It used to put its own
      // number there - "Pass 2 of 4 · Δ 0.41 Å · pLDDT 63.4" - while the
      // progress callback writes "Folding · 62%" many times a second between
      // passes. The two alternate, and a line that swaps between two different
      // sentences is unreadable: it reads as flicker rather than as progress.
      // The percentage is the only thing there that moves smoothly, so it is
      // the only thing there. See the same note in web/af3-model.js.
      //
      // 🔴 THE NUMBERS ARE NOT LOST, they are in the place that is meant to
      // hold them: the scores card, which is a panel rather than a line and
      // can be read at leisure while it updates once a pass.
      updateScoresCard(recycle.confidence);
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
        // ...recycle 0's frame is built by loadIntoViewer rather than by
        // appendPass, so its contact map has to be attached here or the first
        // pass is the one frame without one - and it is the frame on screen
        // while every later pass is still running.
        void initialLoadPromise.then(() => {
          const frame = viewer?.objectsData?.[viewerObject]?.frames?.[0];
          if (frame !== undefined) {
            attachContactMap(frame, recycle, model.weights, sequence.length);
          }
        });
      } else {
        appendPass(sequence, chainLengths, recycle, index, firstPassLanded, model.weights);
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
    const regime = multimer
      ? { outerProductMeanFirst: true, positionScale: 20,
        chainAware: true, chainSequences: chains }
      : {};
    // ...?graph=unified runs the MONOMER weights through src/multimer/ instead.
    // With its switches off that graph reproduces the monomer one bit for bit,
    // which is the check that the superset is right; a difference is a graph
    // bug rather than a weights bug.
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
      firstPassLanded, stem };
    const alignedRecycles = allRecycles.map((r, i) => ({
      structure: i === 0 ? (firstPassLanded ?? r.structure) : alignedToFirstPass(sequence, r.structure, firstPassLanded),
      confidence: r.confidence,
      recycleDistance: r.recycleDistance,
      // 🔴 THE DRIVER'S OWN PASS, KEPT BY REFERENCE. Its contact map is attached
      // in a setTimeout long after this map runs, so copying the field here
      // copies `undefined`; holding the object means whatever lands on it later
      // is visible to anything that reads this afterwards.
      pass: r,
    }));
    const finalLanded = alignedRecycles[alignedRecycles.length - 1].structure;

    // 🔴 THE BEST PASS, NOT THE LAST. Recycling is not monotonic - a pass can
    // score worse than the one before it, and AlphaFold's own pipeline ranks
    // its outputs rather than taking whichever finished last. The criterion is
    // ColabFold's `rank_by: auto`: the multimer score for a complex, mean pLDDT
    // for a monomer.
    //
    // 🔴 AND THE SEARCH STARTS FROM THE LAST ONE, so a tie keeps it. Passes
    // often converge to the same score to several decimals, and preferring an
    // earlier one on an exact tie would hand back a less converged structure
    // for no gain.
    const rankOf = (confidence) => (family === "multimer"
      ? (confidence?.multimerScore ?? confidence?.iptm ?? Number.NEGATIVE_INFINITY)
      : (confidence?.meanPlddt ?? Number.NEGATIVE_INFINITY));
    let bestIndex = alignedRecycles.length - 1;
    for (let index = alignedRecycles.length - 1; index >= 0; index -= 1) {
      if (rankOf(alignedRecycles[index].confidence)
        > rankOf(alignedRecycles[bestIndex].confidence)) bestIndex = index;
    }
    const best = alignedRecycles[bestIndex];
    previousFold = {
      sequence,
      structure: finalLanded,
    };
    lastPrediction = {
      stem,
      // The BEST pass, and its own scores with it - a structure from one pass
      // beside another pass's pLDDT would be a file that describes nothing that
      // was ever computed.
      pdb: predictionToPdb(sequence, best.structure, best.confidence.plddt, chainLengths),
      confidence: best.confidence,
      scores: confidenceJson(sequence, best.confidence),
      a3m: alignment,
      chains,
      chainLengths,
      recycles: alignedRecycles,
      bestPass: bestIndex,
      contactSource: best.pass,
      model: `AlphaFold 2 (${family})`,
      ...foldContext,
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
    // 🔴 THE CARD SCORES WHAT WILL BE SAVED, which is the best pass and not
    // always the last. Showing the last pass's numbers beside a download of the
    // best one is the kind of disagreement nobody reads a status line closely
    // enough to catch.
    updateScoresCard(best.confidence);
    const took = ((performance.now() - started) / 1000).toFixed(1);

    const converged = allRecycles.length < passes
      ? ` · converged at ${final.recycleDistance.toFixed(2)} Å after ${allRecycles.length} passes`
      : "";
    const bestIptmText = best.confidence.iptm !== undefined
      ? ` · ipTM ${Number(best.confidence.iptm).toFixed(3)}` : "";
    // ...and said out loud when the two differ, because the play bar is still
    // sitting on the last pass while the download is a different one.
    const ranked = bestIndex !== alignedRecycles.length - 1
      ? ` · saved pass ${bestIndex + 1} of ${alignedRecycles.length}` : "";
    status(`Done in ${took} s · pLDDT ${best.confidence.meanPlddt.toFixed(1)}`
      + ` · pTM ${best.confidence.ptm.toFixed(3)}${bestIptmText}${ranked}${converged}`);
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
    // ...whatever happened, including a stop. The timeline is most useful about
    // the fold that did NOT finish, so it is closed here and not on the way out
    // of the success path.
    devEndRun();
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
// was true while single-sequence was the default alignment mode and false the
// moment the MMseqs2 search became it - that mode posts the sequence to the
// public ColabFold server. The fold itself never leaves the machine either
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

installDevPanel();
syncMode();

// 🔴 THE MODEL DECIDES WHICH CONTROLS EXIST, and syncMode decides what the MSA
// ones say - so the model listener runs syncModelControls and then syncMode,
// in that order: the second reads the visibility the first just set.
const familySelect = document.getElementById("model-family");
if (familySelect !== null) {
  familySelect.addEventListener("change", () => { syncModelControls(); syncMode(); });
}
document.getElementById("af3-mode")?.addEventListener("change", syncAf3Count);
// 🔴 URL FIRST, THEN BOTH SYNCS, IN THE LISTENER'S ORDER. `?model=` moves the
// row after syncMode() has already read it above, so the controls have to be
// brought back into agreement exactly as a change event would - and
// syncModelControls before syncMode, because the second reads the visibility
// the first sets.
applyModelFromUrl();
syncModelControls();
syncMode();
// ...and only after the parameter has been read can a complaint about it be
// made. The first version called this beside the Fold button's enabling,
// which runs EARLIER than this block - so it always found nothing to report
// and said nothing, which is the same silence it was written to fix.
reportModelFromUrl();

element("msa-file").addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (file === undefined) return;
  // 🔴 THE BYTES DECIDE, NOT THE EXTENSION. A fold archive renamed to .a3m is
  // still an archive and an a3m called .zip is still an alignment, and the
  // failure of guessing by name is a confusing parse error rather than a
  // refusal. See looksLikeZip.
  void file.arrayBuffer().then(async (buffer) => {
    const bytes = new Uint8Array(buffer);
    try {
      if (looksLikeZip(bytes)) {
        const restored = msasFromArchive(await readZip(bytes));
        if (restored.chains === 0 && restored.merged === undefined) {
          throw new Error("that archive holds no alignments");
        }
        if (restored.chains === 0) {
          // An archive whose fold was given one merged alignment carries it
          // back as exactly that, with no split to restore.
          uploadedMsas = { merged: restored.merged };
          uploadedA3m = restored.merged;
          const described = parseA3m(restored.merged);
          status(`archive · ${described.depth} sequences · ${described.length} columns`);
          return;
        }
        uploadedMsas = restored;
        uploadedA3m = "";
        const paired = restored.pairedA3ms.size;
        status(`archive · ${restored.chains} chain${restored.chains === 1 ? "" : "s"}`
          + `${paired > 0 ? `, ${paired} with paired rows` : ", no paired rows"}`);
        return;
      }
      const text = new TextDecoder().decode(bytes);
      const described = parseA3m(text);
      uploadedA3m = text;
      uploadedMsas = undefined;
      status(`${described.depth} sequences · ${described.length} columns`);
    } catch (error) {
      uploadedA3m = "";
      uploadedMsas = undefined;
      status(error instanceof Error ? error.message : String(error), true);
    }
  });
});


// ...THE RAW PREDICTION, downloadable as computed. py2Dmol's own save button
// writes a session; these two write what the model actually produced.
function download(name, text, type) {
  downloadBlob(name, text, type);
}

/** The same, for bytes as readily as text. */
function downloadBlob(name, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
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
// 🔴 EVERYTHING THE FOLD USED AND PRODUCED, NOT JUST THE SCORES. What this
// replaces wrote pLDDT, PAE and pTM into one JSON and dropped the rest on the
// floor - the alignment, the templates, the request - so a fold could not be
// reproduced or handed on once the tab was closed. See web/fold-archive.js for
// the layout and why it is the AlphaFold 3 server's.
element("download-all").addEventListener("click", async () => {
  const pred = activePrediction();
  if (!pred) return;
  const button = element("download-all");
  button.disabled = true;
  try {
    const files = buildFoldArchive({
      stem: pred.stem,
      model: pred.model ?? "AlphaFold",
      settings: pred.settings,
      entities: pred.entities,
      msas: pred.msas ?? {},
      msaOrigin: pred.msaOrigin ?? "none (single sequence)",
      templates: pred.templates ?? [],
      prediction: {
        pdb: pred.pdb,
        chainLengths: pred.chainLengths,
        tokens: pred.tokens,
        confidence: {
          ...pred.confidence,
          // 🔴 RESOLVED HERE, NOT WHEN THE FOLD FINISHED. AlphaFold 2 computes
          // its contact map in a setTimeout - the distogram head costs 131 ms
          // at 128 residues and is deliberately off the fold's critical path -
          // so at the moment the prediction was stored it does not exist yet.
          // By the time anyone presses this it does. `contactSource` is the
          // pass the saved structure came from, which is not always the last.
          contactProbs: pred.confidence?.contactProbs
            ?? pred.contactSource?.contactProbs,
        },
      },
    });
    downloadBlob(`${pred.stem}.zip`, await writeZip(files), "application/zip");
  } catch (error) {
    status(error instanceof Error ? error.message : String(error), true);
  } finally {
    button.disabled = false;
  }
});
