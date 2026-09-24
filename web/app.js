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
import { AlphaFoldMonomerGpu } from "../src/af2/model/monomer.js";
import { AlphaFoldUnifiedGpu } from "../src/af2/multimer/model.js";
import { blankChainColumns, foldsAsSingleSequence, parseA3m }
  from "../src/input/a3m.js";
// 🔴 mergeSearchedChains IS USED ONLY WHEN A SEARCH IS REUSED, which is why it
// shipped missing from this list. That path needs a cache from an earlier fold
// AND more than one chain, so a first fold never reaches it - and stopping a
// fold partway is one of the few ways to get a filled cache and then fold
// again. test/module-references.test.js now looks for the whole class.
import { generateMmseqs2ComplexMsa, generateMmseqs2Msa, mergeSearchedChains,
  expandSearchedChains, planSearchReuse, searchCacheEntry }
  from "../src/input/mmseqs2-api.js";
import { isAbortError, throwIfAborted } from "../src/runtime/abort.js";
import { distogramContactProbabilities } from "../src/heads/distogram.js";
import { GpuMemoryBudgetError, setMemoryBudget }
  from "../src/runtime/device-memory.js";
import { AF3_COUNTS, OPENDDE_COUNTS, OPENDDE_SAMPLER_MODE, NO_FLOW_SAMPLER_FAMILIES,
  samplerModeFor, af3SequenceProblem, alphaCarbons, fittedPdb, foldAf3,
  loadAf3Weights, toPoints, warmAf3Pipelines } from "./af3-model.js";
import { actualSteps, ESMFOLD2_COUNTS, ESMFOLD2_SAMPLER_MODE, languageModelRunner,
  loadEsmfold2Weights } from "./esmfold2-model.js";
import { SAMPLER_PRESETS, foldEsmfold2 } from "../src/esmfold2/fold.js";
import { spreadOverAtoms, toDensePositions } from "../src/esmfold2/featurise.js";
import { toPdb } from "../src/af3/fold.js";
import { chainGeometryOf, chainGeometryVerdict } from "../src/af3/chain-geometry.js";
import { ccdUrl, parseCcdComponent } from "../src/af3/featurise/ccd-component.js";
import { smilesComponent } from "../src/chem/component.js";
import { GpuBufferAllocator } from "../src/runtime/allocator.js";
import { getDevice, loadModel, releaseModel } from "./model.js";
import { AF3_FAMILIES, ALL_ATOM_FAMILIES, MODEL_BUNDLES, MODELS_WITHOUT_CONFIDENCE,
  SINGLE_SEQUENCE_FAMILIES, graphFamily }
  from "../src/bundles/manifests/index.js";
import { devAdopt, devBeginRun, devEndRun, devNote, devOnEntry, devSourceIs, devStatus,
  devUseDevice } from "./dev-log.js";
import { installDevPanel } from "./dev-panel.js";
import { correspondence } from "./align.js";
import { superposeOnto } from "./morph.js";
import { CHAIN_IDS, confidenceJson, contactMapFor, matrixForViewer, modifiedPositions,
  paeMatrix, predictionToPdb, safeJobName, viewerTokens } from "./prediction-results.js";
import { complexSequenceProblem } from "./sequence.js";
// 🔴 SHARED WITH proteinhunter.html, which shows the same card against its
// own play bar. See web/scores-card.js.
import { updateScoresCard } from "./scores-card.js";
import { entitiesFromText, entitiesProblem, expandEntities, POLYMER_TYPES,
         templateKind } from "./entities.js";
import { buildFoldArchive, tokenLayoutFrom, msasFromArchive,
         SINGLE_SEQUENCE_ORIGIN } from "./fold-archive.js";
import { jobFromJson } from "./job-json.js";
import {
  clearSession, jobMeta, readSession, readSessionMeta, saveSession,
} from "./fold-session.js";
import { looksLikeZip, readZip, writeZip } from "./zip.js";
import { createEntityList } from "./entity-ui.js";
import { buildTemplate, describeCoverage, fetchStructure } from "./template-source.js";
import { fetchMmseqs2Templates } from "../src/input/mmseqs2-api.js";
import { RuntimeEstimator } from "../src/runtime/cost-model.js";
import { colabRole, installColabBridge, remoteCommand, remoteEvents, remoteHead,
  revivePrediction, tapOut } from "./colab-bridge.js";
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

/**
 * ONE NAME for a fold: the viewer object, every file in the archive, and the
 * `name` inside its job request.
 *
 * 🔴 THEY USED TO BE TWO. The archive's request carried the fold's stem while
 * a loaded job's own `name` was thrown away, so `calmodulin_4calcium.json`
 * came back out of the page called `af3_1`. A name BOX was built for that and
 * then removed as not earning its place; what the box was worth keeping is
 * this - one resolver, used by all three fold paths, so the object in the
 * picker, the `.pdb` button and every member of the archive cannot drift
 * apart. A pasted FASTA `>header` names the fold, and the model prefix is the
 * fallback.
 */
function foldStem(fallback) {
  const header = entityList.header();
  return uniqueStem(header === null ? fallback : safeJobName(header));
}

// 🔴 EXPOSED FOR tools/fold-in-page.py, WHICH HAS NO OTHER WAY IN. The rows are
// built by entity-ui.js and their model is a closure; a harness that wrote into
// a row's field would leave that model behind the DOM, and the fold would run
// on what the model still held. `set` is the same call the paste path makes.
window.__entityList = entityList;

// 🔴 AND THE FINISHED PREDICTION, FOR THE SAME REASON. A backend driving this
// page headlessly (tools/colab_backend.py) could read the STRUCTURE out of the
// download button and nothing else - so a remote fold arrived with no
// alignment, no confidence and no scores card, which is most of what the page
// shows about a fold. `lastPrediction` is a module binding; a function rather
// than the value because it is REASSIGNED on every fold and a captured
// reference would hand back the one before.
window.__lastPrediction = () => lastPrediction;

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
/**
 * How far consecutive passes may move, in angstroms, before recycling stops.
 *
 * 🔴 IT READ AN ELEMENT THAT DID NOT EXIST. The driver has taken a tolerance
 * since it was written - `shouldStopAfterRecycle` compares consecutive passes'
 * alpha carbons - and this returned 0 for every fold ever run here because
 * index.html carried no `#tolerance`. There is one now, and **0.1 is
 * selected**: a converged fold runs 3 passes of 4 rather than 4, for a
 * thousandth of a pLDDT, and a fold that has not settled runs all of them.
 *
 * 🔴 AND 0.0 IS THE REFERENCE'S ANSWER FOR ONE OF OUR TWO MODELS, NOT BOTH.
 * AlphaFold's own config carries `recycle_early_stop_tolerance` 0.0 in `CONFIG`
 * and **0.5 in `CONFIG_MULTIMER`**, and ColabFold's CLI default of None leaves
 * whichever the checkpoint names. Ours is one control for both, defaulting to
 * the monomer's: a deliberate deviation for the multimer, because the saving is
 * measured (a converged fold stops at 2 passes of 4) and the cost is not - no
 * target here has both a deep alignment and a crystal. See docs/AF2.md.
 */
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

  // 🔴 THE ALTERNATIVES ARE THE MODEL ROW'S OWN OPTIONS, READ WHEN THE DIALOG
  // OPENS. The dialog used to name OpenBind-0 alone, so declining AlphaFold 3
  // meant accepting one particular substitute or going and working the control
  // yourself. Cloning the row is what keeps the two lists from drifting: a
  // model added to the page appears here, and one build_site.py drops from
  // dist/index.html because its bundle is unservable never does.
  //
  // 🔴 AND THE GATED FAMILY IS NOT AMONG THEM. Offering af3 as its own escape
  // would hand back the model whose terms this is asking about, without the
  // acceptance it exists to record.
  const row = document.getElementById("model-family");
  const chooser = document.getElementById("model-terms-alternative");
  if (chooser !== null && row !== null) {
    const held = chooser.value;
    chooser.replaceChildren(...[...row.options]
      .filter((option) => option.value !== family)
      .map((option) => new Option(option.textContent.trim(), option.value)));
    const offered = [...chooser.options].map((option) => option.value);
    // ...OpenBind-0 unless somebody has already moved it: it runs this same
    // graph under Apache 2.0, which makes it the nearest model to the one being
    // declined rather than merely the first in the list.
    const start = offered.includes(held) ? held
      : (offered.includes("openbind0") ? "openbind0" : offered[0]);
    if (start !== undefined) chooser.value = start;
  }

  dialog.returnValue = "";
  dialog.showModal();
  await new Promise((resolve) => dialog.addEventListener("close", resolve, { once: true }));

  if (dialog.returnValue === "accept") {
    rememberTermsAccepted();
    return "af3";
  }
  if (dialog.returnValue === "switch") {
    const picked = chooser?.value;
    // 🔴 NOTHING TO SWITCH TO IS NOT A FOLD. An empty chooser - no dialog
    // markup, or a page whose model row holds only the gated family - must
    // cancel rather than fall through to folding with the model just declined.
    if (picked === undefined || picked === "" || picked === family) return null;
    // 🔴 THE ROW IS UPDATED, NOT JUST THE FOLD. Folding with a model the
    // control does not name is a page whose state is written nowhere on it -
    // the same fault the "Auto" model setting had before it was removed.
    if (row !== null) {
      row.value = picked;
      syncModelControls();
      syncMode();
    }
    return picked;
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
 * would hand somebody a model they did not ask for. `openbind` IS accepted,
 * because that is the name upstream publishes the blob under and the name this
 * page used before the release number was added - but it resolves to
 * `openbind0` rather than standing for whatever OpenBind means next.
 */
const MODEL_ALIASES = { openbind: "openbind0", ob: "openbind0", ob0: "openbind0",
                        af2: "monomer", mono: "monomer", multi: "multimer",
                        // ...the name this model shipped under first.
                        esmfold2: "ef2-fast-600m", ef2: "ef2-fast-600m",
                        // 🔴 THE VERSION IS PART OF THE NAME, for the same
                        // reason OpenFold3's preview is not an alias for
                        // OpenBind above: Boltz-1 and Boltz-2 are different
                        // models, as are Protenix and Protenix-v2, so the bare
                        // family name resolves to the release this page
                        // actually carries rather than standing for whatever
                        // the name means next.
                        boltz: "boltz2", protenix: "protenix2" };

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
  // 🔴 A FAMILY THE MODEL ROW DOES NOT OFFER MAY STILL BE REACHABLE. EF2-fast's
  // two checkpoints share one entry and the PLM row picks between them, so
  // `?model=ef2-fast-300m` names a real family that is not an <option> - and
  // without this it took the "no model called that" branch, which is the
  // silently-ignored parameter this function exists to prevent, wearing a
  // complaint that names a family the page really does have.
  const viaPlm = Object.entries(PLM_FAMILIES)
    .find(([, family]) => family === wanted && wanted !== PLM_FAMILIES.none);
  const plm = document.getElementById("plm-mode");
  if (viaPlm !== undefined && plm !== null) {
    const host = [...select.options].map((option) => option.value)
      .find((value) => SINGLE_SEQUENCE_FAMILIES.includes(value));
    if (host !== undefined) {
      select.value = host;
      plm.value = viaPlm[0];
      return;
    }
  }
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

/**
 * The family this page will fold with.
 *
 * 🔴 THE MODEL ROW IS NOT ALWAYS THE WHOLE ANSWER. EF2-fast ships as two
 * checkpoints that differ only in the language model they were trained against,
 * and the model row shows ONE entry for both - so the PLM row decides which,
 * and everything downstream that keys on a family (the weight cache, the trunk
 * cache, the download stem, the labels) has to see the resolved one or two
 * bundles get mistaken for each other. See PLM_FAMILIES.
 */
/**
 * 🔴 AND MOVING THE MODEL ROW NOW CHANGES NOTHING ON SCREEN.
 *
 * There was a COVER over the previous model's result, and then there was
 * `startNewSession`, which emptied the page instead: one fold at a time, each
 * its own session, so moving the row was starting a new one. Both existed to
 * answer the same question - is what I am looking at this model's? - and both
 * are gone, because the page no longer has one answer to give.
 *
 * Every fold keeps its own object and every panel reads `activePrediction()`,
 * which is keyed by the object being edited: the scores card, the heatmap and
 * the two download buttons describe the fold you are LOOKING at, and the model
 * row describes the fold you are about to MAKE. Those are different questions
 * and they now have different controls, so a switch has nothing to clear.
 *
 * What went with the cover: `shownFamily`, `foldingFamily`, `resultIsFrom`,
 * `resultIsStale`, `syncPendingResult`, `RESULT_REGION`, `firstFrameIsOurs`
 * and `.result-pending` in web/localfold.css. What went with the session:
 * `startNewSession` itself.
 */

/**
 * THE DOWNLOAD ROW OFFERS WHAT THERE IS TO DOWNLOAD, AND NOTHING ELSE.
 *
 * 🔴 A BUTTON THAT WRITES NOTHING IS WORSE THAN NO BUTTON. `download-pdb` and
 * `download-all` read `activePrediction()` and return silently when there is
 * none - so before the first fold, and under a model that did not make what is
 * on screen, the row offered files that either do not exist or belong to the
 * model the veil is covering. The same claim as the veil, one control along.
 *
 * 🔴 AND THE SESSION BUTTON ASKS A DIFFERENT QUESTION, so it gets a different
 * answer: py2Dmol's session is everything the VIEWER is showing, which a
 * dropped file and a restored session are as much as a fold - it follows what
 * is drawn, not what was predicted, and a fold's own downloads follow the fold.
 */
function syncDownloads() {
  const pred = activePrediction();
  const foldable = !!(pred && pred.pdb);
  const registry = window.py2dmol_viewers ?? {};
  const renderer = registry[Object.keys(registry)[0]]?.renderer;
  const drawn = Object.keys(renderer?.objectsData ?? {}).length > 0;
  const offer = (id, on, why) => {
    const button = document.getElementById(id);
    if (button === null) return;
    // The button's own title is kept the first time it is seen, because the
    // reason it is off has to give way to what it does when it comes back.
    if (button.dataset.title === undefined) {
      button.dataset.title = button.getAttribute("title") ?? "";
    }
    button.disabled = !on;
    button.setAttribute("title", on ? button.dataset.title : why);
  };
  // ...and the only reason left is the only one there can be: a model switch
  // empties the page, so there is never a fold on screen that these would
  // offer the wrong files for.
  offer("download-pdb", foldable, "nothing has been folded yet");
  offer("download-all", foldable, "nothing has been folded yet");
  offer("saveStateButton", drawn, "the viewer is empty");
  const row = document.getElementById("downloads");
  if (row !== null) row.style.display = (foldable || drawn) ? "flex" : "none";
}

/**
 * A FINISHED PREDICTION, IN ONE PLACE.
 *
 * 🔴 THREE PATHS BUILT ONE AND ONLY TWO REGISTERED IT. Recording a result is
 * three things - it is the last one, it is filed under its stem, and the
 * downloads follow - and `foldWithEsmfold2` did the first only, so an
 * ESMFold2 fold left the download row describing the fold before it. A
 * funnel rather than a third copy of the lines: a path can no longer do half
 * of it.
 *
 * 🔴 AND THE FAMILY TRAVELS ON THE PREDICTION, as an id rather than as
 * prose. The archive, the session's `localfold` block and the Colab reader
 * all want to know which model made a result, and the only thing written in
 * one used to be the LABEL - `AlphaFold 2 (monomer-3)` from this page's AF2
 * path, where MODEL_LABELS says `AlphaFold 2 (model 3)`, with both AF2
 * families sharing the bare string `AlphaFold 2`. Reading a name back to
 * find an identity is a lookup that is neither total nor one-to-one.
 */
function recordPrediction(prediction, family) {
  // The family still travels ON the prediction: the archive, the session's
  // `localfold` block and the Colab reader all read it, and a label is a
  // thing to read rather than an identifier - see the note above.
  prediction.family = family;
  // 🔴 AND THIS IS WHERE THE PAGE CLAIMS THE OBJECT AS ITS OWN, which is
  // what lets a fold and a new session recycle it and leave the reader's
  // restored folds alone. `openBlankFold` sets it too, at the START of a
  // local fold - but the Colab reader never opens one: a finished result
  // arrives with no frames, `loadIntoViewer` makes the object, and nothing
  // said it was ours. So a model switch there cleared nothing and the
  // result sat on under the new model's name. A restore deliberately does
  // NOT come through here.
  lastFoldStem = prediction.stem ?? lastFoldStem;
  lastPrediction = prediction;
  if (prediction.stem !== undefined) predictions.set(prediction.stem, prediction);
  syncDownloads();
  return prediction;
}

const chosenFamily = () => {
  const chosen = document.getElementById("model-family")?.value ?? "af3";
  if (SINGLE_SEQUENCE_FAMILIES.includes(chosen)) return PLM_FAMILIES[plmChoice()] ?? chosen;
  // 🔴 AND AlphaFold 2's MODEL NUMBER RESOLVES HERE FOR THE SAME REASON THE PLM
  // ROW DOES. AF2 is FIVE models, one training run continued five ways, and
  // people run all five and compare - so the row shows one AF2-mono and a
  // number beside it, and the number picks the bundle. Resolving it here is
  // what keeps the weight cache, the download stem and the labels all naming
  // the model that actually folded; models 2 to 5 are a 43 MiB delta on
  // model_1's 97 (see tools/pack_delta_model.py).
  // 🔴 BOTH AF2 FAMILIES, because the multimer is five models too - and its
  // four deltas are 44 MiB against a bundle's 74. Neither family drops the
  // control: what differs is that the monomer's 3, 4 and 5 have no template
  // embedder and the multimer's five all do.
  if (chosen === "monomer" || chosen === "multimer") {
    const number = document.getElementById("af2Model")?.value ?? "1";
    // 🔴 "all" RESOLVES TO THE FIRST MODEL OF THE SWEEP, NOT TO A SIXTH FAMILY.
    // One press of Fold then runs all five, but everything that reads this -
    // the weight cache, the merge rule, the licence, the download stem - is
    // asked before any of them has run and has to name a real bundle. The
    // sweep itself is af2Sweep, and the run renames what it has to (the stem,
    // the provenance) once it knows which model won.
    if (number === "all") return chosen;
    return number === "1" ? chosen : `${chosen}-${number}`;
  }
  return chosen;
};
/** The graph a family runs, which for a delta is the graph of its base. */
const graphOf = graphFamily;
/**
 * Every AlphaFold 2 family one press of Fold will run, in order.
 *
 * One family normally, and all five when the Model # row says "all" - which is
 * what AlphaFold's own pipeline does, and the reason ColabFold ranks its
 * outputs rather than returning the last one.
 *
 * 🔴 IT READS THE OPTIONS, NOT THE REGISTRY, so that trimming the control
 * trims the sweep. `build_site.py` DELETES an `<option>` whose delta bundle has
 * no `remote` - the page would 404 on shard zero otherwise - and a sweep built
 * from `MODEL_BUNDLES` would have gone on folding the number it had just
 * removed. The registry still has the last word on whether the name exists.
 */
const af2Sweep = (family) => {
  const row = document.getElementById("model-family")?.value ?? "";
  if (row !== "monomer" && row !== "multimer") return [family];
  if ((document.getElementById("af2Model")?.value ?? "1") !== "all") return [family];
  return [...document.querySelectorAll("#af2Model option")]
    .map((option) => option.value)
    .filter((value) => /^[0-9]+$/.test(value))
    .map((number) => (number === "1" ? row : `${row}-${number}`))
    .filter((name) => MODEL_BUNDLES[name] !== undefined);
};
const isAf3Family = (family) => AF3_FAMILIES.includes(family);
/**
 * 🔴 "CAN THIS MODEL SEE AN ATOM" IS NOT "IS THIS AN AlphaFold 3 GRAPH", AND
 * THE SECOND MODEL THAT NEEDED THEM SPLIT THEM. ESMFold2 runs a different graph
 * and the same all-atom representation, so a guard written as `!isAf3Family`
 * refuses a ligand under a model that has ligand tokens - with a message naming
 * a capability it has. That is the mistake recorded in CLAUDE.md for the
 * AF3/OpenBind split, one model later, and this is the name that prevents it.
 */
const supportsAllAtom = (family) => ALL_ATOM_FAMILIES.includes(family);

/**
 * What a fold's viewer object and downloaded files are called, per model.
 *
 * 🔴 THE OBJECT NAME IS THE ONLY PLACE TWO FOLDS ARE TOLD APART ON SCREEN, and
 * it did not name the model. Both AF3-graph bundles produced `af3_N` and both
 * AlphaFold 2 models produced `prediction_N`, so a page holding an AlphaFold 3
 * fold and an OpenBind-0 fold showed two objects with the same prefix and no
 * way to say which was which - and the stem also becomes the archive's file
 * names, so a downloaded `af3_1_model_0.pdb` could be either.
 *
 * A FASTA header still wins: a name somebody supplied beats a generated one.
 */
const MODEL_STEMS = {
  af3: "af3",
  openbind0: "openbind0",
  opendde: "opendde",
  boltz2: "boltz2",
  protenix2: "protenix2",
  intellifold2: "intellifold2",
  rosettafold3: "rosettafold3",
  monomer: "af2",
  // 🔴 AND THE MODEL NUMBER IS IN THE STEM, because it is the only place on
  // screen that says which of AlphaFold 2's five folded this. Two objects
  // called `af2_1` from model_1 and model_4 is exactly the collision this
  // table exists to prevent, one axis later.
  "monomer-2": "af2_model2",
  "monomer-3": "af2_model3",
  "monomer-4": "af2_model4",
  "monomer-5": "af2_model5",
  multimer: "af2_multimer",
  "multimer-2": "af2_multimer_model2",
  "multimer-3": "af2_multimer_model3",
  "multimer-4": "af2_multimer_model4",
  "multimer-5": "af2_multimer_model5",
  "ef2-fast-600m": "ef2_fast_600m",
  "ef2-fast-300m": "ef2_fast_300m",
};

/** What to call each model while its weights download. */
/**
 * WHICH MODEL'S ANSWER IS ON SCREEN, AND WHETHER IT IS STILL THE CHOSEN ONE.
 *
 * 🔴 A RESULT OUTLIVES THE ROW THAT MADE IT, AND NOTHING SAID SO. Switching the
 * model row left the previous model's structure, plots and confidence numbers
 * exactly where they were, under the new model's name - and every one of those
 * looks the same whichever row is selected, so there is no way to read the
 * picture and see that it is the other model's. Reported as wanting the viewers
 * to say PENDING instead of the previous result.
 *
 * `undefined` is "nothing here came from a fold on this page" - a dropped file,
 * a restored session - and nothing is claimed about it, because the claim would
 * be the fault by another route. A family is what one of the three fold paths
 * ingested; `null` is a result whose model could not be named, which is stale
 * against every row.
 */


/** The family a prediction's own label names, where the label is one we write. */
const familyFromLabel = (label) => Object.keys(MODEL_LABELS)
  .find((family) => MODEL_LABELS[family] === label);

const MODEL_LABELS = {
  af3: "AlphaFold 3",
  // Upstream's own name for this release. See src/af3/dialect.js for why the
  // number is not decoration.
  openbind0: "OpenBind-0",
  opendde: "OpenDDE",
  // Upstream's own names, for the reason OpenBind-0 keeps its number: the
  // release is what the weights are, and a shortened label would name a family
  // that has more than one member.
  boltz2: "Boltz-2",
  protenix2: "Protenix-v2",
  intellifold2: "IntelliFold-2",
  rosettafold3: "RoseTTAFold3",
  monomer: "AlphaFold 2",
  multimer: "AlphaFold 2",
  // 🔴 THE NAME IS THE CHECKPOINT'S, NOT THE FAMILY'S. `ESMFold2` alone read as
  // ESM's released ESMFold2-Fast, which folds from ESM-C 6B and is a different
  // and better model; this is an experimental one sized to fit a browser.
  //
  // 🔴 AND THE SIZE IS NOT IN THE LABEL, BECAUSE THE PLM ROW SAYS IT. "600M"
  // names the TOWER, not the folding model - which is 171 M in every published
  // variant - so carrying it in the model name said the wrong thing twice over
  // once a row appeared naming the language model outright. The family id keeps
  // it (`ef2-fast-600m` is the checkpoint `base600M-step1500k`), because a
  // 300M sibling would be a DIFFERENT fold bundle rather than a tower swap: its
  // shim is trained for 30 layers x 960 against this one's 36 x 1152.
  "ef2-fast-600m": "EF2-fast",
  // ...the same folding model against the smaller tower, and its own
  // checkpoint. The label names the tower because that is what differs.
  "ef2-fast-300m": "EF2-fast (300M)",
  // 🔴 AND THE FIVE AF2 MODELS SAY WHICH ONE IS LOADING, DERIVED RATHER THAN
  // TYPED. A delta downloads its BASE as well as itself, so the dial reading
  // "AlphaFold 2 · 73 / 116 MiB" under model 2 is the only place a reader is
  // told the two halves are one model - and eight typed rows is eight chances
  // to label model_4's weights model_3. Both AF2 families are "AlphaFold 2";
  // the number is what the row does not already say.
  ...Object.fromEntries(Object.entries(MODEL_BUNDLES)
    .filter(([, bundle]) => ["monomer", "multimer"].includes(bundle.delta?.base))
    .map(([family]) => [family, `AlphaFold 2 (model ${family.split("-")[1]})`])),
};

const modelFamily = (ligandCount = 0, modificationCount = 0, nucleicCount = 0,
                     templateCount = 0) => {
  // 🔴 THE CHOICE IS ALWAYS EXPLICIT NOW. "Auto" used to read the chain count
  // and pick between the two AlphaFold 2 models - which made AF2 the silent
  // default for everything and could never choose AF3, so the newest model was
  // the one a reader had to know to ask for. It also meant the page had a
  // state in which what would run was written nowhere on it.
  // 🔴 THE RESOLVED FAMILY, NOT THE ROW'S RAW VALUE. This is what `runFold`
  // preloads and folds with, and EF2-fast's two checkpoints share one <option>
  // - so reading the select directly folded the 600M pair while every label,
  // stem and archive field came from `chosenFamily()` and said 300M. The page
  // reported a model it had not run, and the structures were byte-identical
  // across the two settings, which is how it was caught. A cache is the easiest
  // place for a second model to be mistaken for the first; so is a second
  // reader of the same control.
  const choice = chosenFamily();
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
  if (ligandCount > 0 && !supportsAllAtom(choice)) {
    throw new Error("Ligands need AF3, OpenBind-0 or ESMFold2;"
      + ` the model is set to ${choice}`);
  }
  // 🔴 AND A MODIFIED RESIDUE IS AlphaFold 3 ONLY FOR THE SAME REASON. AF2
  // tokenises one residue per letter and has no way to say that residue 12 is
  // a phosphoserine, so folding under it would drop the modification and return
  // a confident structure of the unmodified chain - which is a different answer
  // to the question, not a worse one. The residue COUNT is unchanged either
  // way, so nothing else on the page would have shown the difference.
  if (modificationCount > 0 && !supportsAllAtom(choice)) {
    throw new Error("Modified residues need AF3, OpenBind-0 or ESMFold2;"
      + ` the model is set to ${choice}`);
  }
  // 🔴 AND A NUCLEIC CHAIN IS AlphaFold 3 ONLY, WHICH IS THE LOUDEST OF THE
  // THREE. AF2's alphabet is the twenty amino acids: `ACGT` is not refused
  // there, it is READ - as alanine, cysteine, glycine, threonine - so a DNA
  // chain folded under AF2 comes back as a confident structure of a short
  // peptide that was never asked for, with nothing anywhere saying so.
  if (nucleicCount > 0 && !supportsAllAtom(choice)) {
    throw new Error("DNA and RNA need AF3, OpenBind-0 or ESMFold2;"
      + ` the model is set to ${choice}`);
  }
  // 🔴 AND A TEMPLATE IS THE ONE THING ESMFold2 REALLY CANNOT DO. `grep -rn
  // template` over its whole upstream package returns nothing and z_init has
  // five terms with no template among them - so a template set on an entity row
  // would be fetched, aligned, and silently dropped. It is the same refusal as
  // the three above and for the opposite reason: those were a capability the
  // model has and the guard denied, this is one it does not.
  //
  // 🔴 AND AlphaFold 2's MONOMER IS NOT IN THAT CLASS ANY MORE. Its term was
  // always there and oracle-checked; the driver simply never forwarded the
  // slot, so this refusal was right for the wrong reason. The MULTIMER stays
  // refused here: it forwards a template in its own driver, but its embedder
  // is a different dialect with a different feature set and nothing on this
  // page has ever built one for it - which is exactly the gap that made the
  // monomer's term look supported for a year.
  if (templateCount > 0 && !isAf3Family(choice) && graphOf(choice) !== "monomer") {
    throw new Error("Templates need AF3, OpenBind-0 or AlphaFold 2 monomer;"
      + ` the model is set to ${choice}`);
  }
  // 🔴 AND THREE OF ALPHAFOLD 2's FIVE MODELS HAVE NO TEMPLATE EMBEDDER AT ALL.
  // model_3, model_4 and model_5 are the template-free ones - `template.enabled`
  // is false in their config and the 67 tensors are simply not in the
  // checkpoint - so a template set under one of them would be fetched, aligned
  // and dropped. Refused by name rather than ignored, which is this file's rule
  // everywhere else; hiding the row would not be enough, because hiding a
  // control does not change its value.
  // 🔴 AND THE TEST IS OVER THE WHOLE SWEEP, because "All 5" RUNS models 3, 4
  // and 5 while resolving to model 1's name. Asked of `choice` alone this
  // passed, the run started, and the third model of five threw in the middle of
  // a fold that had already drawn two - which is the worst moment to find out.
  const templateless = af2Sweep(choice)
    .filter((name) => MODEL_BUNDLES[name]?.noTemplateEmbedder === true);
  if (templateCount > 0 && templateless.length > 0) {
    throw new Error(`${MODEL_BUNDLES[templateless[0]].model} has no template embedder -`
      + " AlphaFold 2's models 3, 4 and 5 are the template-free ones."
      + (templateless.length > 1
        ? " Choose model 1 or 2 rather than All 5, or remove the template."
        : " Choose model 1 or 2, or remove the template."));
  }
  // 🔴 AND THE MONOMER'S TERM TAKES EXACTLY ONE. `QueryOnlyTemplateGpu` reads
  // `input.template`, singular - AF3 runs a forward per slot and averages, and
  // this one does not - so a second row would be silently dropped.
  if (templateCount > 1 && graphOf(choice) === "monomer") {
    throw new Error("AlphaFold 2's monomer takes one template;"
      + ` ${templateCount} are set`);
  }
  return choice;
};

// 🔴 "none" IS SPELLED "single" BELOW, and the translation happens here so it
// happens once. Every path downstream already tests for "single" meaning a
// query-only fold. The select says "Single Sequence", which names what is
// FOLDED rather than what is missing - it used to say "None", which reads as
// an absent setting rather than a choice about the input.
const msaMode = () => {
  // 🔴 A SINGLE-SEQUENCE MODEL DECIDES THIS, NOT THE SELECT, AND HIDING THE ROW
  // IS NOT ENOUGH. `syncModelControls` hides the MSA controls for ESMFold2
  // because it has no alignment to take - `disable_msa_features` is true in its
  // checkpoint - but hiding a control does not change its VALUE, so a page that
  // had been set to Search kept returning "search" from behind the hidden row.
  // A monomer survived that (the search runs, the result is discarded); an
  // OLIGOMER did not, because a multi-chain search reaches `mergeSearchedChains`
  // before the fold branches by model, and that looks its merge rule up by
  // family - "unknown model ef2-fast-600m: expected monomer, multimer, af3,
  // openbind0". The control was ignored everywhere except the one place it
  // could still throw.
  if (SINGLE_SEQUENCE_FAMILIES.includes(chosenFamily())) return "single";
  const chosen = element("msa-mode").value;
  // 🔴 `job` IS AN ACTION AND NOT A MODE, SO IT NEVER REACHES A FOLD. Selecting
  // it opens a file picker; it answers here with the mode it replaced, because
  // a value this function cannot map is a value the fold would run on - the
  // same trap as a hidden control keeping its old value, one entry above.
  // `syncMode` puts the real mode back as soon as the file is read, so this
  // only covers the window in between, and the case where somebody picks it
  // and folds without choosing a file.
  if (chosen === "job") return modeBeforeJob === "none" ? "single" : modeBeforeJob;
  return chosen === "none" ? "single" : chosen;
};

/**
 * The MSA mode that "Load job JSON…" interrupted, to be put back after.
 *
 * 🔴 A JOB CAN SET THIS ITSELF AND THAT WINS. A file carrying `unpairedMsa: ""`
 * is asking to fold with no alignment - AlphaFold 3 reads an empty string and
 * an absent field as opposite instructions - so `applyJob` moves the dial to
 * Single Sequence and says so in the status line. Restoring the stashed mode
 * over that would silently run the search the file asked us not to run.
 */
let modeBeforeJob = "search";

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

/**
 * EVERYTHING THE FORM NEEDS TO MAKE THIS FOLD AGAIN, AS THE CONTROLS HOLD IT.
 *
 * 🔴 A SESSION PUT THE PICTURE BACK AND LEFT THE PAGE SET TO SOMETHING ELSE.
 * Restoring loaded py2Dmol's viewer state and the prediction behind it - so
 * the structure, the maps and the downloads all came back - and touched no
 * control: the sequence box still held whatever was in it, the model row
 * still named whatever was chosen, the dials were the dials. Pressing Fold
 * after a restore therefore folded the CURRENT form, which on a fresh page
 * is the default sequence. A restored fold could be looked at and not
 * continued from. Reported as: after a restore, Fold should behave as it
 * would if the reader had redone every step up to that point.
 *
 * 🔴 ONE LIST, READ AND WRITTEN BY THE TWO FUNCTIONS UNDER IT. What a fold
 * IS lives in these controls plus the entity rows, and the saved `settings`
 * block cannot stand in for them: it is written for the ARCHIVE, in prose a
 * reader wants ("trunk passes", "diffusion steps"), it differs per model
 * path, and half of it is what the fold RESOLVED rather than what was asked
 * for. Reading a form back out of a report is how the two drift.
 */
const FOLD_CONTROLS = ["model-family", "af2Model", "plm-mode", "msa-mode",
                       "msa-text", "max-msa", "recycles", "tolerance",
                       "af3-mode", "af3-count", "random-seed"];

/** What the form says now, in one object the session can carry. */
function formInputs() {
  const controls = {};
  for (const id of FOLD_CONTROLS) {
    const element_ = document.getElementById(id);
    if (element_ !== null) {
      controls[id] = element_.type === "checkbox" ? element_.checked : element_.value;
    }
  }
  return { entities: entityList.read(), controls };
}

/**
 * ...and put it back. Returns whether it had anything to put.
 *
 * 🔴 WITHOUT `change` EVENTS. Those listeners exist to bring the other
 * controls into agreement with the row, and they used to empty the page as
 * well - so dispatching one here would have thrown away the fold this is
 * being called to restore. The syncs are called directly instead, in the
 * listeners' own order, which is also one fewer thing to keep in step.
 *
 * 🔴 AND A SELECT ONLY TAKES A VALUE IT HAS. Assigning an unknown one leaves
 * a select EMPTY rather than throwing, which is a control that says nothing
 * and folds as whatever the code reads out of "" - so an option that is no
 * longer offered (a model this build dropped) leaves that row alone.
 */
function applyInputs(inputs) {
  if (inputs === undefined || inputs === null) return false;
  if (Array.isArray(inputs.entities) && inputs.entities.length > 0) {
    try {
      entityList.set(inputs.entities);
    } catch (cause) {
      console.warn("could not put the sequence rows back", cause);
    }
  }
  for (const [id, value] of Object.entries(inputs.controls ?? {})) {
    const element_ = document.getElementById(id);
    if (element_ === null || value === undefined) continue;
    if (element_.type === "checkbox") { element_.checked = !!value; continue; }
    if (element_.tagName === "SELECT"
        && ![...element_.options].some((option) => option.value === value)) continue;
    element_.value = value;
  }
  syncModelControls();
  syncMode();
  syncAf3Count();
  return true;
}

/** The last prediction, kept so it can be downloaded as it was computed. */
let lastPrediction;

/**
 * The object THIS PAGE'S FOLDING made, so the next fold can recycle it and
 * leave everything else alone. Undefined until the first fold: before that
 * there is nothing of ours on screen, and whatever is there was restored or
 * dropped by the reader.
 */
let lastFoldStem;

/** The one fold owned by the page; pressing the same button aborts it. */
let activeFold;

/** The drawn object, once the first pass has landed. See appendPass. */
let viewer;
let viewerObject;

/** py2Dmol's own status line, so folding reports where fetching used to. */
// 🔴 A TAP, NOT A SECOND FOLD PATH. When this page is the one a Colab runtime
// is driving headlessly, the page a reader is looking at is somewhere else -
// so the status line, the bar and every sampler frame have to travel. They
// travel as the SAME calls the local fold already makes, handed to
// web/colab-bridge.js here; there is no remote-only code path to keep in step
// with the real one, which is the whole reason the runtime runs this page
// rather than a port of it.
//
// `tapOut` PUSHES, in the task that made the event - it used to park it in an
// array for tools/colab_backend.py to collect over CDP every 250 ms, and a
// busy page answers a debugger when it feels like it: reported as the reader's
// bar sitting at "embedder · 1%" for a whole fold. Off the runtime it is one
// property read per status write and nothing else.
function remoteTap(kind, payload) {
  tapOut(kind, payload);
}

function status(text, isError = false) {
  remoteTap("status", text);
  const node = document.getElementById("status-message");
  // 🔴 THE TIMELINE IS FED BEFORE THE EARLY RETURN, so a page whose status line
  // is missing still records. It costs one string compare a write, and only a
  // CHANGE of leading segment records a row - the sampler rewriting a
  // percentage several times a second is one phase, not four hundred.
  // 🔴 NOT THIS PAGE'S TIMELINE WHEN THE FOLD IS SOMEWHERE ELSE. A reader's
  // status line is a REPLAY of the runtime's, so recording it here would time
  // the runtime's phases against this browser's clock and file them under this
  // browser's (empty) device - the rows that arrive as `dev` events are the
  // real ones, taken where the work happened.
  if (colabRole() !== "reader") devStatus(text);
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
/**
 * Whether this fold runs the protein language model.
 *
 * 🔴 THE ANSWER IS READ, NOT ASSUMED, EVEN WHERE THE ROW IS HIDDEN - which is
 * the lesson `msaMode` records one function above: hiding a control does not
 * change its value, and a select left on "none" behind a hidden row goes on
 * returning "none". For a family that has no such row the answer is simply no,
 * because none of them has a language model to run.
 */
function usesLanguageModel(family = chosenFamily()) {
  if (!SINGLE_SEQUENCE_FAMILIES.includes(family)) return false;
  return plmChoice() !== "none";
}

const plmChoice = () => document.getElementById("plm-mode")?.value ?? "esmc-600m";

/**
 * What to call the language model in a record of the fold.
 *
 * 🔴 READ OFF THE CONTROL, NOT WRITTEN DOWN TWICE. This was the literal
 * "ESM-C 600M", which the archive then reported for a fold that had run against
 * the 300M tower - a settings block naming an input the fold did not use, which
 * is the thing that whole section exists to prevent. The option's own text is
 * what the reader chose and needs no second table to drift from.
 */
function plmLabel() {
  if (!usesLanguageModel()) return "none";
  const select = document.getElementById("plm-mode");
  return select?.selectedOptions?.[0]?.textContent?.trim() ?? plmChoice();
}

/**
 * Which EF2-fast checkpoint the PLM row is asking for.
 *
 * 🔴 THE TOWER AND THE FOLDING MODEL ARE ONE CHOICE, NOT TWO. Biohub publish
 * `base600M-step1500k` and `base300M-step1500k` as separate checkpoints whose
 * shims are trained for 36 layers x 1152 and 30 x 960 - so picking a tower
 * picks a folding bundle with it, and offering them as independent controls
 * would let a page ask for a pairing that has never existed. The model row
 * therefore shows ONE "EF2-fast" and this row says which.
 *
 * 🔴 AND "None" KEEPS WHICHEVER FOLDING MODEL IS ALREADY THERE, because it is
 * the folding model that folds: with no tower every token takes
 * `shimSingleForZeroState`, which each bundle has its own version of. 600M's is
 * the default and the one the measurements above were made against.
 */
const PLM_FAMILIES = {
  "esmc-600m": "ef2-fast-600m",
  "esmc-300m": "ef2-fast-300m",
  none: "ef2-fast-600m",
};

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
  // 🔴 THREE LOADERS NOW, AND THE THIRD IS TWO BUNDLES. ESMFold2 reads a
  // folding bundle and a language model with separate manifests and separate
  // licences, so it has its own entry point rather than a family argument to
  // one of the others - and it reports ONE progress stream over both, or the
  // dial resets to zero halfway through a 347 MiB download.
  // 🔴 A FOLD WITH NO PROTEIN NEEDS NO LANGUAGE MODEL, AND THAT IS 224 MiB.
  // ESM-C is handed protein tokens only, so a ligand, DNA or RNA input has no
  // row for it - the fold already skips the call, and this skips the download
  // too. Read from the entities as they stand: an empty list is a page nobody
  // has typed into yet, where a protein is much the likeliest thing next.
  const typed = entityList.read();
  const needsLanguageModel = usesLanguageModel(family)
    && (typed.length === 0 || typed.some((entity) => entity.type === "protein"));
  // 🔴 THE TEST IS THE CAPABILITY, NOT THE CHECKPOINT. EF2-fast ships as two
  // families and a third is published; `family === "ef2-fast-600m"` sends the
  // 300M one down AlphaFold 2's branch, which is the `family === "af3"` mistake
  // this file already records once.
  const load = SINGLE_SEQUENCE_FAMILIES.includes(family)
    ? loadEsmfold2Weights(report,
                          { languageModel: needsLanguageModel, family })
    : (AF3_FAMILIES.includes(family)
      ? loadAf3Weights(report, family)
      : loadModel("msa", report, signal, family));
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
  remoteTap("progress", fraction);
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
/**
 * An alignment that carries nothing but its own query, as a single-sequence
 * fold - or null when it carries more and should be folded as an alignment.
 *
 * 🔴 `depth` COUNTS ROWS. A search that matched nothing returns TWO of them,
 * because extractMmseqs2A3m joins the uniref and environmental blocks and each
 * begins with its own `>101`; a pasted or uploaded A3M can say the same thing
 * in one row or in ten. Folding that is not an alignment: it is the query
 * repeated, and the repetition moves the answer - pTM 0.3965 -> 0.4148 on the
 * 59-mer. `text: null` is the state "Single Sequence" mode already produces, so
 * every consumer below already handles it.
 *
 * 🔴 AND FOR A PASTED OR UPLOADED ONE, ONLY WHEN ITS QUERY IS WHAT WE ARE
 * FOLDING. An A3M's own first record WINS over the sequence box further down -
 * that is deliberate, so a reader can paste an alignment and fold what it
 * describes - and returning `text: null` throws that away. If the two differ,
 * the alignment is left alone and folds the protein it names, exactly as
 * before. The search path cannot reach this: generateMmseqs2Msa already refuses
 * an A3M whose query is not the sequence it asked about.
 */
function singleSequenceIfOnlyQuery(text, chains, where, extra = {}) {
  if (!foldsAsSingleSequence(text, chains.join(""))) return null;
  status(`${where} · folding the single sequence`);
  return { text: null, blocks: null, ...extra };
}

async function alignmentText(chains, signal, family, wantsMsa = []) {
  // 🔴 "THIS CHAIN FOLDS FROM ITS SEQUENCE ALONE" HAS NO CONDITIONS ON IT.
  // That is what the entity popup says, and it was honoured on the SEARCH
  // path only - where every chain has its own a3m and blanking one is a
  // substitution. A pasted or uploaded alignment is ONE text over the
  // concatenated chains, so the setting did nothing at all there: an existing
  // job with an alignment in the box folded with it however the entity rows
  // were set. Reported exactly that way.
  //
  // Two answers, because they are two different statements. With EVERY chain
  // turned off there is no alignment left to give the model and this is the
  // single-sequence path by another name. With SOME off, the text keeps its
  // shape and those chains lose their rows' residues - see blankChainColumns.
  const chainOff = chains.map((_, index) => wantsMsa[index] === false);
  const allOff = chains.length > 0 && chainOff.every(Boolean);
  // ONE EXIT FOR THE TEXT THE READER SUPPLIED, so a third way of supplying
  // one cannot forget the setting the way these two did.
  const fromText = (text, label) => {
    if (allOff) return null;
    // 🔴 AND THE SPANS ARE ONLY KNOWABLE WHEN THE QUERY IS THE ONE IN THE
    // BOX. A pasted alignment's own first record WINS over the sequence box -
    // deliberately, so a reader can paste an alignment and fold what it
    // describes (see singleSequenceIfOnlyQuery) - and the chain spans here
    // are measured off the BOX. Where the two differ there is nothing to
    // measure against, so blanking would gap arbitrary columns of somebody
    // else's alignment. It says so and folds what was pasted.
    const mismatch = chainOff.some(Boolean)
      && parseA3m(text).query !== chains.join("");
    if (mismatch) {
      status(`${label} is for a different sequence - folding it as it is,`
        + " with every chain's alignment");
      return singleSequenceIfOnlyQuery(text, chains, label) ?? text;
    }
    const kept = blankChainColumns(text, chains, chainOff);
    return singleSequenceIfOnlyQuery(kept, chains, label) ?? kept;
  };
  switch (msaMode()) {
    case "single": return null;
    case "paste": {
      const text = element("msa-text").value.trim();
      if (text.length === 0) throw new Error("Paste an A3M, or switch the alignment back to none");
      return fromText(text, "The pasted alignment");
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
        if (allOff) return null;
        // ...per chain here rather than on the merged text, because an
        // archive HAS the per-chain alignments: the same substitution the
        // search path makes, one step earlier. A chain that is off is folded
        // against its own query row and pairs with nothing.
        const queryOnly = (sequence) => `>101\n${sequence}\n`;
        const merged = mergeSearchedChains({
          sequences: chains,
          chainA3ms: uploadedMsas.chainA3ms.map((a3m, index) =>
            (chainOff[index] ? queryOnly(chains[index]) : a3m)),
          pairedA3ms: new Map(chains.map((chain, index) =>
            [chain, chainOff[index] ? queryOnly(chain)
              : uploadedMsas.pairedA3ms.get(index)])),
          model: family,
        });
        if (uploadedMsas.chains !== chains.length) {
          throw new Error(`that archive holds ${uploadedMsas.chains} alignments`
            + ` and this fold has ${chains.length} chain`
            + `${chains.length === 1 ? "" : "s"}`);
        }
        status(`Alignment from the archive · ${uploadedMsas.chains} chains`);
        return singleSequenceIfOnlyQuery(merged.a3m, chains, "The uploaded archive")
          ?? { text: merged.a3m, blocks: merged.blocks };
      }
      if (uploadedA3m.length === 0) throw new Error("Choose an A3M file, or switch the alignment back to none");
      return fromText(uploadedA3m, "The uploaded alignment");
    }
    case "search": {
      // 🔴 THE ONE REQUEST THIS PAGE MAKES OFF THE MACHINE. Everything else runs
      // against weights already on disk. The sequence is sent to the public
      // ColabFold MMseqs2 server, so it is a mode the reader picks rather than
      // a default they discover afterwards.
      // ...validated over EVERY chain, because every one of them is being
      // folded. What is SEARCHED for is a subset; see `searchChains`.
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
      // 🔴 A CHAIN WITH ITS MSA OFF IS SEARCHED FOR AND THEN GIVEN ITS QUERY
      // ROW ALONE, which is what AlphaFold reads as "no alignment for this
      // chain". The merge already pads a short chain with gaps
      // (mergeRowAlignedChainA3ms), so a query-only block lands as the real
      // sequence on row 0 and gaps under it - the model's own representation,
      // not an approximation of one.
      //
      // 🔴 AND THE SEARCH DOES NOT RUN FOR IT. It used to: the chain was
      // searched for and its answer thrown away afterwards, on the reasoning
      // that dropping it from the request changes what the PAIRING is
      // computed over and so changes the OTHER chains' alignments. That is
      // true and it is the wrong trade, for two reasons the note it replaces
      // did not weigh.
      //
      // The sequence still went to the public ColabFold server. "No
      // alignment for this chain" is a reasonable way to say "do not send
      // this one anywhere", and it did not mean that.
      //
      // And pairing picks species present in two or more chains, so keeping
      // a chain whose alignment is about to be discarded biases which
      // organisms are paired for the chains that kept theirs. The alignments
      // it preserved were identical to a fold nobody asked for.
      //
      // What it costs: flipping the row back needs a new search, because the
      // cache now holds what was asked for rather than everything. Reported
      // as mmseqs2 searching for both chains when only one wanted it.
      const blankMsa = (index) => (wantsMsa[index] === false);
      const someOff = chains.some((_, index) => blankMsa(index));
      const queryOnlyA3m = (sequence) => `>101\n${sequence}\n`;

      // WHAT IS ACTUALLY SEARCHED FOR, and therefore what is cached and what
      // the template hits are numbered in: the chains that asked for an
      // alignment, in their own order. Everything below maps back.
      const searchChains = someOff
        ? chains.filter((_, index) => !blankMsa(index)) : chains;
      // ...and with none of them asking, nothing is sent at all. This is the
      // single-sequence path, which every consumer already handles.
      if (searchChains.length === 0) return null;
      const plan = planSearchReuse({ cache: searchCache, chains: searchChains, family });
      let searched;
      if (plan.reuse === "single") {
        searched = searchCache.raw.single;
        status(`MSA reused · ${searched.depth} sequences`);
      } else if (plan.reuse === "merge") {
        const { chainA3ms, pairedA3ms, depth, templateHits } = searchCache.raw;
        const merged = mergeSearchedChains({
          sequences: searchChains, chainA3ms, pairedA3ms, model: family,
        });
        // ...and the hits with them. `mergeSearchedChains` re-merges the
        // ALIGNMENTS and knows nothing about templates, so without this a
        // reused search reports no hits and an automatic template says the MSA
        // is not a search.
        searched = { ...merged, depth, templateHits };
        status(`MSA reused · ${depth} sequences`);
      } else {
        searched = searchChains.length === 1
          ? await generateMmseqs2Msa(searchChains[0], searchOptions)
          : await generateMmseqs2ComplexMsa(searchChains, searchOptions);
        status(`MSA search found ${searched.depth} sequences`);
        searchCache = { key: plan.key,
                        raw: searchCacheEntry({ chains: searchChains, searched }) };
      }
      // ...and the chains that asked for none are put back, as query-only
      // blocks, in the FOLD's order. What came back covers `searchChains`
      // alone; everything downstream counts chains the way the entity rows
      // do, so this is the one place the two numberings meet.
      if (someOff) {
        // The parts, from whichever shape holds them: a one-chain search
        // keeps its result whole (`single`), a complex keeps the per-chain
        // blocks, and a reused merge has already been turned back into text.
        const raw = searchCache?.raw ?? {};
        const parts = searchChains.length === 1
          ? [raw.single?.a3m ?? searched.a3m]
          : (raw.chainA3ms ?? searched.chainA3ms);
        // ...and the two numberings meet in ONE tested function, not here.
        const { chainA3ms, pairedA3ms, templateHits } = expandSearchedChains({
          chains,
          off: chains.map((_, index) => blankMsa(index)),
          parts,
          pairedA3ms: searchChains.length === 1
            ? undefined : (raw.pairedA3ms ?? searched.pairedA3ms),
          templateHits: raw.templateHits ?? searched.templateHits,
        });
        searched = { ...searched,
                     ...mergeSearchedChains({ sequences: chains, chainA3ms,
                                              pairedA3ms, model: family }),
                     depth: searched.depth,
                     templateHits };
        const off = chains.length - searchChains.length;
        status(`MSA search · ${searchChains.length} chain`
               + `${searchChains.length === 1 ? "" : "s"} searched, ${off}`
               + ` folded from ${off === 1 ? "its" : "their"} query alone`);
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
      // 🔴 A SEARCH THAT FINDS ONLY THE QUERY IS A SINGLE-SEQUENCE FOLD, AND
      // SAYING SO IS THE HONEST ANSWER AS WELL AS THE FAST ONE. `depth` counts
      // ROWS, and `extractMmseqs2A3m` returns the uniref block and the
      // environmental block WHOLE - each starting with its own `>101` - so a
      // search that matched nothing comes back as depth 2 carrying one
      // sequence. Folding that is not "an alignment of two": it is the query,
      // twice, and the second copy measurably moves the answer (pTM 0.3965 ->
      // 0.4148 on the 59-mer). Counting DISTINCT rows is the only honest depth.
      //
      // Routing it to the query-only path is what the "Single Sequence" mode
      // already does, so this is not a new code path - `alignment === null` is
      // a state every consumer below already handles. The template hits are
      // kept: a protein with no homologs may still have a structure to lean on,
      // and they came out of the same tar.
      const onlyQuery = singleSequenceIfOnlyQuery(
        searched.a3m, chains,
        `MSA search found only the query (${searched.depth} rows, 1 distinct)`,
        { templateHits: searched.templateHits });
      if (onlyQuery !== null) return onlyQuery;
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
/**
 * Set py2Dmol's colour mode, through the API that exists.
 *
 * 🔴 `setColorScheme` AND `colorBy` DO NOT EXIST ON THIS RENDERER, AND BOTH
 * CALL SITES WERE GUARDED BY `typeof === "function"` - so they were silent
 * no-ops that had never coloured anything. The viewer stayed on `colorMode:
 * "auto"`, which resolves to `rainbow` for a single chain with no confidence
 * data. Reported as "still not seeing colors, though certainty is showing up in
 * the status", with the B-factor probe passing: the values were in every frame
 * and nothing was reading them.
 *
 * 🔴 THE REAL ONE IS `py2Dmol.setColor`, and what makes it work is not the
 * assignment but the three lines after it: `colorsNeedUpdate`,
 * `plddtColorsNeedUpdate` and a `render()`. Setting `colorMode` alone leaves
 * the cached colours in place, which is a different silent no-op.
 *
 * Valid modes are auto, chain, rainbow, plddt, deepmind, entropy, object and
 * hydrophobicity, plus anything in `window.py2dmol_customColors`.
 */
function setColourMode(mode) {
  const registry = window.py2dmol_viewers ?? {};
  const renderer = registry[Object.keys(registry)[0]]?.renderer;
  // 🔴 THROUGH py2Dmol's OWN SELECT FIRST, because that keeps the dropdown the
  // reader can see in step with what is drawn. Its change handler validates the
  // mode, sets `colorMode` and both dirty flags, and renders - so this is the
  // whole job, and it is what the AF3 path has always done.
  const select = renderer?.colorSelect;
  if (select !== undefined && select !== null) {
    try {
      select.value = mode;
      select.dispatchEvent(new Event("change", { bubbles: true }));
      if (renderer.colorMode === mode) return true;
    } catch { /* fall through to the API */ }
  }
  const api = window.py2Dmol;
  if (typeof api?.setColor === "function") {
    try { api.setColor(mode); return true; } catch (error) {
      console.warn(`colour mode ${mode} refused:`, error);
    }
  }
  // ...and the manual path, which is what setColor does anyway. Setting
  // `colorMode` alone leaves the cached colours in place, which is a silent
  // no-op of its own - the flags and the render are the operative part.
  if (renderer === undefined) return false;
  renderer.colorMode = mode;
  renderer.colorsNeedUpdate = true;
  renderer.plddtColorsNeedUpdate = true;
  renderer.render?.("localfold.setColourMode");
  return true;
}

async function loadIntoViewer({ stem, pdb, scores, a3m, pae, length, confidence,
                                frameName = "recycle_0" }) {
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
  // 🔴 pLDDT ONLY WHERE THERE IS A pLDDT. `loadIntoViewer` is handed `scores`
  // exactly when the model has a confidence head, and painting the pLDDT ramp
  // over an absent B-factor makes every residue the colour of NO CONFIDENCE -
  // which reads as a terrible fold rather than an absent measurement. Chain
  // colours are what EF2-fast uses for the same reason.
  if (viewer !== undefined) setColourMode(scores === undefined ? "chain" : "plddt");
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
      // ...and the name is the caller's under a sweep, where frame zero belongs
      // to model 1 of five rather than to the only fold there is. See appendPass.
      frame.name = frameName;
      frame.label = frameName;
      frame.title = frameName;
      if (confidence !== undefined) frame.confidence = confidence;
      if (pae !== undefined) { frame.pae = pae; frame.pae_n = length; }
    }
  }
  return stats;
}

let lastReportedObject;
let lastReportedFrameIdx = -1;

function getActiveFrameConfidence(frameIndex) {
  try {
    if (!viewer) return null;
    // 🔴 THE OBJECT BEING EDITED, NOT THE ONE THIS PAGE LAST FOLDED.
    // `viewerObject` is pinned to the running fold's object - which is right
    // for the frames being APPENDED to it - and the card describes what is
    // on SCREEN. With folds accumulating those are different objects the
    // moment a reader picks another one, and the card went on reporting the
    // last fold's numbers over somebody else's structure.
    const objName = viewer.currentObjectName ?? viewerObject;
    const obj = viewer.objects?.find((entry) => entry.name === objName);
    const objData = viewer.objectsData?.[objName];
    const frames = objData?.frames ?? obj?.frames;
    const idx = frameIndex !== undefined
      ? frameIndex
      : (obj?.currentFrame ?? objData?.currentFrame ?? viewer.currentFrame ?? 0);
    const targetFrame = frames?.[idx];
    const pred = objName ? predictions.get(objName) : null;
    // ...and the FOLD'S own numbers where the frame's are not the card's.
    // 🔴 PRESENT IS NOT THE SAME AS SCORED. A sampler frame carries a
    // `confidence` of its own - the per-atom pLDDT the ribbon is coloured
    // from - with no meanPlddt and no pTM in it, and the card renders a
    // missing number as "-". So taking the frame's whenever it exists blanked
    // every cell the moment a reader picked another fold, on a page that had
    // just shown pLDDT 74.7. What the card wants is a SUMMARY, and the fold's
    // own is the one to fall back to.
    const scored = (c) => !!c && (c.meanPlddt !== undefined || c.ptm !== undefined);
    const frameConf = targetFrame?.confidence;
    const predConf = pred?.recycles?.[idx]?.confidence ?? pred?.confidence;
    const conf = scored(frameConf) ? frameConf
      : (scored(predConf) ? predConf : frameConf);
    return { confidence: conf, index: idx, object: objName };
  } catch (err) {
    return null;
  }
}

function syncScoresCardToActiveFrame(frameIndex) {
  const result = getActiveFrameConfidence(frameIndex);
  if (result && result.confidence) {
    lastReportedFrameIdx = result.index;
    lastReportedObject = result.object;
    // and an estimated pLDDT has to say that it is one.
    updateScoresCard(result.confidence);
  }
}

// Watch for animation playback changes (py2Dmol play button loop)
setInterval(() => {
  try {
    if (!viewer) return;
    const objName = viewer.currentObjectName ?? viewerObject;
    const obj = viewer.objects?.find((entry) => entry.name === objName);
    const objData = viewer.objectsData?.[objName];
    const idx = obj?.currentFrame ?? objData?.currentFrame ?? viewer.currentFrame;
    // ...and the OBJECT is half of what the card is describing. Two folds can
    // be on the same frame index, so watching the index alone left the card
    // on the fold you had just switched away from.
    if (idx !== undefined
        && (idx !== lastReportedFrameIdx || objName !== lastReportedObject)) {
      syncScoresCardToActiveFrame(idx);
    }
  } catch (e) {}
}, 50);

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


/**
 * THE CONTACT MAP IS THE MAIN VIEW WHILE A FOLD RECYCLES, AND THE STRUCTURE
 * TAKES IT BACK WHEN RECYCLING ENDS.
 *
 * The recycles are the model changing its mind about which residues touch, and
 * the contact map is the picture of that; the structure is a consequence of it,
 * and for AF3 and ESMFold2 there is no structure at all until the sampler
 * starts. py2Dmol's two slots take a STANDING choice (`setSlots`), so asking
 * once at the start is enough - the map takes the big slot the moment it exists
 * and the structure sits in the small one beside it once IT exists.
 *
 * 🔴 HANDED BACK, NOT SET TO `structure`. `null` returns both slots to the
 * automatic choice, which is structure big whenever there is one - so the end
 * of a fold looks exactly as a loaded file does, and nothing stays pinned into
 * the next thing the reader opens.
 *
 * 🔴 AND A READER WHO CLICKS A SLOT TAB DURING THE FOLD HAS TAKEN THE LAYOUT
 * OVER. Their choice is the standing one from then on and is not handed back
 * for them. Asked of the click rather than of `getSlots()`, which reports what
 * is SHOWN: when a fold is stopped before any map exists, the structure is shown
 * big by fallback while the contact map is still what was asked for, and
 * reading "shown" there would leave the contact map pinned big for good.
 */
let contactsHeldBig = false;
document.addEventListener("click", (event) => {
  if (event.target?.closest?.(".py2dmol-slot-tab")) contactsHeldBig = false;
}, true);
function contactsBig(on) {
  const registry = window.py2dmol_viewers ?? {};
  const renderer = registry[Object.keys(registry)[0]]?.renderer;
  if (typeof renderer?.setSlots !== "function") return;
  try {
    if (on) {
      renderer.setSlots({ big: "contact", small: "structure" });
      contactsHeldBig = true;
    } else if (contactsHeldBig) {
      contactsHeldBig = false;
      renderer.setSlots({ big: null, small: null });
    }
  } catch (cause) {
    console.warn("could not arrange the slots", cause);
  }
}

/**
 * THE MODIFICATION IS THE POINT OF THE JOB, AND IT IS INVISIBLE BY DEFAULT.
 *
 * A cartoon draws a phosphoserine exactly as it draws the serine it was made
 * from - the ribbon runs through the alpha carbon and the phosphate is a
 * side-chain atom, and py2Dmol keeps side chains off unless something asks.
 * So a reader who put SEP at position 3 folds it and sees no evidence that it
 * arrived. These residues and no others get theirs drawn.
 *
 * 🔴 AND IT IS `showSidechains`, WHICH IS RELATIVE. It ADDS to whatever is
 * out, so a reader who turned some on by hand keeps them, and a second fold
 * does not have to undo the first - the viewer is given a new object either
 * way.
 */
function showModifiedSidechains(renderer, object, positions) {
  if (!(positions?.length > 0) || typeof renderer?.showSidechains !== "function") return;
  try {
    // 🔴 NAMED WITH ITS OBJECT, because `positions` are indices into what is
    // DRAWN. With two folds merged (the Multi button, a restored session)
    // that array is both of them and residue 3 of this fold is residue 3 of
    // the first one - py2Dmol offsets them through `localRangeOf` when the
    // object is named, and answers the identity when nothing is merged.
    renderer.showSidechains({ object, positions });
  } catch (cause) {
    console.warn("could not draw the modified residues' side chains", cause);
  }
}

/**
 * A trunk's contact map, shown before there is a structure to hang it on. AF3
 * and ESMFold2 both finish their recycles before the sampler emits a frame.
 */
function showTrunkContacts(liveContacts, chains) {
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
    // 🔴 AND THE VIEWER IS OPENED FOR IT. It is display:none until the first
    // fold draws something, and on the first fold of a visit that used to be
    // the sampler's first frame - so the whole trunk's contact maps were
    // pushed into a panel nobody could see.
    revealViewer(renderer);
    renderer.chains = trunkChainIds(chains);
    renderer.heatmapRenderer.setMaps({ contact: liveContacts });
    window.Heatmap?.updateVisibility?.(renderer);
    renderer.render("trunk-contacts");
  } catch (cause) {
    console.warn("could not show the trunk's contact map", cause);
  }
}

/**
 * The pAE panel's bytes.
 *
 * 🔴 IT IS QUANTISED AGAINST A FIXED 0-32 A, NOT AGAINST ITS OWN RANGE. A PAE
 * plot is read by the SHAPE of its blocks against a scale everybody knows, and
 * rescaling each fold to its own extremes would make a confident structure and
 * a hopeless one look identical. 32 A is the range AlphaFold reports over.
 */
function paeMapFor(alignedError, maximum = 32) {
  if (alignedError === undefined) return undefined;
  const n = Math.round(Math.sqrt(alignedError.length));
  if (n * n !== alignedError.length) return undefined;
  const data = new Uint8Array(n * n);
  for (let index = 0; index < data.length; index += 1) {
    data[index] = Math.max(0, Math.min(255,
      Math.round((alignedError[index] / maximum) * 255)));
  }
  return { data, n, vmin: 0, vmax: maximum };
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
 * EVERYTHING BELOW THE STATUS LINE, WHILE A FOLD IS WAITED FOR.
 *
 * 🔴 A RESULT ON SCREEN WHILE ANOTHER FOLD RUNS IS THE LAST ONE'S, and the
 * page says nothing to that effect: the structure, the strip, the alignment,
 * the scores and the two download buttons all look exactly as they did when
 * they were the answer. Asked for in one line - "while waiting for fold, the
 * previous results should disappear (everything below status menu)" - and it
 * is the other half of the rule above it: setting the next run up changes
 * nothing, and PRESSING FOLD clears the page.
 *
 * The objects are NOT touched. Folds accumulate and the picker still lists
 * every one of them; what goes is the SIGHT of the last result while there
 * is nothing to say about this one. `revealViewer` brings it all back, and
 * it is already called the moment this fold has something to draw - the
 * trunk's contact map, which is the earliest thing there is.
 */
function hideResults() {
  for (const id of ["viewer-container", "sequence-viewer-container",
                    "msa-buttons"]) {
    const box = document.getElementById(id);
    if (box !== null) box.style.display = "none";
  }
  msaCanvases("none");
}

/**
 * 🔴 THE ALIGNMENT IS NOT INSIDE `#msa-buttons`. That box holds the HEADER -
 * the mode menu, the filters, the chain picker - and `panels/msa.js` appends
 * every `.msa-canvas` to `viewEl.parentElement`, so the drawn alignment is its
 * SIBLING. Hiding the box therefore took the controls away and left the
 * picture: reported as the previous object's MSA still displaying after
 * pressing Fold. It went unnoticed because `openBlankFold` calls
 * `MSA.clear()`, which REMOVES the canvases - so the leftover only shows in
 * the window before the trunk runs, which on a complex is the search, the
 * weights and a minute of looking at another job's alignment.
 *
 * Hidden rather than cleared, because a fold that is stopped before it draws
 * has to put this back (see restoreResults) and a cleared one comes back as a
 * header row over nothing. The class is py2Dmol's own interface here, the way
 * the DOM ids above are.
 */
function msaCanvases(display) {
  for (const box of document.querySelectorAll(".msa-canvas")) {
    box.style.display = display;
  }
}

/**
 * ...and put them back where a fold ended with nothing to show.
 *
 * 🔴 A STOPPED FOLD MUST NOT LEAVE A BLANK PAGE. The reveal above happens
 * when the new fold DRAWS, so a fold that fails, is stopped, or is refused
 * before it draws anything would leave the reader looking at nothing with
 * every previous fold still in the viewer. There is nothing to put back on a
 * page that never had a result, which is what the object count answers.
 */
function restoreResults() {
  const registry = window.py2dmol_viewers ?? {};
  const renderer = registry[Object.keys(registry)[0]]?.renderer;
  if (!renderer || !Object.keys(renderer.objectsData ?? {}).length) return;
  revealViewer(renderer);
  try { window.updateMSAContainerVisibility?.(); } catch { /* optional panel */ }
  // ...and the alignment itself, which is not in that box - see msaCanvases.
  // Only where the panel came back: an object with no MSA leaves the header
  // hidden, and its canvases have been removed rather than hidden anyway.
  if (document.getElementById("msa-buttons")?.style.display !== "none") {
    msaCanvases("block");
  }
  syncDownloads();
}

/**
 * A name no object and no earlier prediction is already using.
 *
 * 🔴 IT READS objectsData, NOT `viewer.objects`, WHICH DOES NOT EXIST. The AF2
 * path checked `viewer?.objects` - an optional chain that always yields
 * undefined on this build - so its uniquifying loop only ever consulted
 * `predictions` and would have collided with any object loaded another way.
 */
/**
 * THE JOB'S OWN NAME, UNLESS SOMETHING ELSE ON SCREEN HAS IT.
 *
 * 🔴 RETURNING THE BARE NAME DESTROYED A RESTORED FOLD. The suffix used to
 * avoid every name in the viewer, and with one fold at a time that looked
 * like dead weight. But `openBlankFold` REWINDS the object it opens - that
 * is how it recycles our own - and a restored session is very often the same
 * job under the same name, so folding after restoring it opened the
 * RESTORED object and emptied it. Nothing removed it; it was overwritten,
 * which is worse: the name and the picker entry stay and the frames are
 * gone. Reported as: past results are lost, past objects cleared during a
 * new fold.
 *
 * 🔴 AND OUR OWN PREVIOUS FOLD IS A CLASH TOO, NOW THAT FOLDS ACCUMULATE.
 * It was taken out of the set - reusing it was the whole of one fold at a
 * time - and with every fold keeping its own object, a name already on the
 * page is taken whoever put it there. So a second fold of the same job is
 * `design_a_2` beside `design_a` rather than on top of it.
 */
function uniqueStem(base) {
  const registry = window.py2dmol_viewers ?? {};
  const renderer = registry[Object.keys(registry)[0]]?.renderer;
  const taken = new Set(Object.keys(renderer?.objectsData ?? {}));
  if (!taken.has(base)) return base;
  let suffix = 2;
  while (taken.has(`${base}_${suffix}`)) suffix += 1;
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
 * 🔴 AND IT DROPS THE FOLD BEFORE IT. This used to ADD an object and keep
 * every earlier one, so py2Dmol's picker could switch between runs and the
 * page's `predictions` map held them all - comparison by accumulation. That
 * is the thing this page is no longer trying to be: ONE FOLD AT A TIME, each
 * its own saved session, and two folds side by side are two browser tabs.
 * Asked for: "dont allow for multiobjects, each fold would be a seperate
 * session".
 *
 * What it buys is that every "which one is this about?" on the page stops
 * being a question. The scores card, the heatmap panel, both download
 * buttons and the cover all describe `lastPrediction`, and with one object
 * there is nothing else they COULD describe - where before each was a
 * separate promise to keep in step, and the cover was the third of them to
 * be found broken. The picker goes with it, without being touched: py2Dmol
 * hides that row by itself once there is one object to pick.
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
  // Every path that folds opens here, so this is where the contact map is
  // asked for as the main view. See contactsBig.
  contactsBig(true);
  try {
    // 🔴 AND THE FOLD BEFORE IT STAYS. This removed it - one fold at a time,
    // each its own session - and the reader asked for the opposite: every
    // fold is its own object, listed in the picker, and you switch between
    // them. Reported as "starting new prediction deletes the previous
    // prediction/object". What made one-at-a-time defensible was that the
    // page could only describe ONE result; every panel reads
    // `activePrediction()` now, which is keyed by the object being edited,
    // so the scores card, the heatmap and the two download buttons follow
    // whichever fold you are looking at and nothing has to be thrown away to
    // keep them honest.
    lastFoldStem = stem;
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
    // 🔴 AND NOTHING HERE TOUCHES THE SHOWN SET. It used to be narrowed to
    // `new Set([stem])`, because every previous fold stayed in it and kept
    // drawing alongside the new one - which was a symptom of py2Dmol's
    // `clearAllObjects` leaving an EMPTY SET where the resting state is null,
    // and an empty set is Multi with everything switched off, so `addObject`
    // joined each later fold to it. Fixed upstream; see the shown-set entry in
    // ../py2Dmol/CLAUDE.md.
    //
    // Narrowing it was also wrong where it did reach: a reader who presses
    // Multi has asked to see several, and a new fold joining them is the rule
    // rather than the bug. It wrote the field directly, past
    // `setShownObjects`, which is what keeps `_framedObjects` in step.
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
/**
 * One mean per chain, skipping the tokens a reading does not apply to.
 *
 * 🔴 -1 IS "NOT APPLICABLE", NOT A LOW SCORE. The interface reading is -1 for
 * every token with no cross-chain partner, which on a monomer is all of them -
 * so a mean that included them would report a confident fold as a bad one.
 */
function meanByChain(asymId, values) {
  if (values === undefined) return undefined;
  const chains = [...new Set(asymId)].sort((a, b) => a - b);
  return chains.map((chain) => {
    let sum = 0, seen = 0;
    for (let token = 0; token < asymId.length; token += 1) {
      if (asymId[token] !== chain || !(values[token] >= 0)) continue;
      sum += values[token];
      seen += 1;
    }
    return seen === 0 ? null : Math.round((sum / seen) * 100) / 100;
  });
}

function attachContactMap(frame, recycle, weights, length) {
  if (weights?.distogram === undefined || recycle.pair === undefined) return;
  setTimeout(() => {
    try {
      const head = weights.distogram;
      const contacts = distogramContactProbabilities(
        recycle.pair, head.halfLogitsWeights, head.halfLogitsBias, length,
        { bins: head.bins, first: head.firstBreak, last: head.lastBreak });
      // AF2 tokenises one residue per letter - nothing to collapse, and it
      // refuses a modified residue outright (see modelFamily).
      const contact = contactMapFor(contacts, undefined);
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
      // 🔴 AND THE SAVED COPY IS REWRITTEN, because it was written before this
      // arrived. AF2's contact map is the panel its archive is worth keeping
      // for, and the fold was already saved without it by the time this runs.
      // One record, so this replaces rather than adds.
      if (lastPrediction?.contactSource === recycle) void rememberSession(lastPrediction);
    } catch (cause) {
      console.warn("contact map unavailable for this pass", cause);
    }
  }, 0);
}

function appendPass(sequence, chainLengths, recycle, recycleIndex, firstPassStructure = undefined,
                    weights = undefined, label = undefined) {
  const api = window.py2Dmol;
  if (viewer === undefined || viewerObject === undefined || api?.frameFromText === undefined) return;
  const aligned = alignedToFirstPass(sequence, recycle.structure, firstPassStructure);
  const pdb = predictionToPdb(sequence, aligned, recycle.confidence.plddt, chainLengths);
  const frame = api.frameFromText(pdb);
  const index = recycleIndex ?? (viewer?.objectsData?.[viewerObject]?.frames?.length ?? 1);
  // 🔴 THE NAME IS THE CALLER'S WHERE THE CALLER KNOWS BETTER. A single fold's
  // frames are its recycles and `recycle_3` says everything; under "All 5" the
  // play bar carries five models' passes in one strip, and twenty frames
  // numbered straight through cannot say which model a reader is looking at.
  const name = label ?? `recycle_${index}`;
  frame.name = name;
  frame.label = name;
  frame.title = name;
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

/**
 * NOTHING CAN BE FOLDED FROM THIS PAGE ANY MORE, AND IT SAYS SO.
 *
 * 🔴 A PAGE SERVED BY A RUNTIME THAT HAS BEEN STOPPED IS A VIEWER. Its fold
 * button would reach a service that is gone, and the reader pressed Disconnect
 * to end it - so what is left is the thing they still want: the structure, the
 * plots, the downloads of what was already folded. Every control whose only
 * job is to shape the NEXT fold goes with the button; the viewer, the session
 * and the files do not.
 */
let foldingRetired = false;

const RETIRED_CONTROLS = ["predict", "add-entity", "model-family", "af2Model",
  "plm-mode", "msa-mode", "af3-mode", "af3-count", "recycles", "max-msa",
  "random-seed"];

function retireFolding(why) {
  foldingRetired = true;
  for (const id of RETIRED_CONTROLS) {
    const control = document.getElementById(id);
    if (control === null) continue;
    control.disabled = true;
    control.title = why;
  }
  // The files stay: "viewing what is already here" is the whole of what this
  // page is for now, and the prediction it holds is untouched.
  syncDownloads();
}

function setFoldButton(state) {
  const button = element("predict");
  const running = state !== "idle";
  button.classList.toggle("btn-primary", !running);
  button.classList.toggle("btn-danger", running);
  // 🔴 AND IT STAYS DISABLED ONCE THE SERVICE IS GONE, whatever else asks for
  // it. Every path that ends a fold comes back through here, and a button
  // re-enabled by one of them is a page that folds into a stopped runtime.
  button.disabled = state === "stopping" || foldingRetired;
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
  const family = chosenFamily();
  const af3 = isAf3Family(family);
  // 🔴 THE MODEL NUMBER IS AF2's ALONE, and the test is the ROW's value rather
  // than the resolved family - `chosenFamily` has already folded the number
  // into it, so asking the resolved one whether to show the control that
  // produced it is circular.
  const af2Node = document.getElementById("af2ModelGroup");
  const af2Row = document.getElementById("model-family")?.value ?? "";
  if (af2Node !== null) {
    af2Node.hidden = af2Row !== "monomer" && af2Row !== "multimer";
  }
  // ...and the convergence stop, which only AlphaFold 2's drivers read. AF3's
  // recycles run to the count they are given; `recycleTolerance` has one caller
  // and it is in the AF2 branch.
  const toleranceNode = document.getElementById("toleranceGroup");
  if (toleranceNode !== null) {
    toleranceNode.hidden = af2Row !== "monomer" && af2Row !== "multimer";
  }
  // 🔴 THE SAMPLER ROW IS SHARED, BECAUSE IT IS THE SAME QUESTION. ESMFold2's
  // structure head is an EDM sampler with a churn factor, exactly as AF3's is,
  // so "flow or diffusion, and how many steps" means the same thing under both
  // - what differs is the numbers, which is why the count dial is rebuilt from
  // a per-model table rather than shared.
  const sampled = af3 || SINGLE_SEQUENCE_FAMILIES.includes(family);
  const countNode = document.getElementById("af3CountGroup");
  if (countNode !== null) countNode.hidden = !sampled;
  // 🔴 THE STEP COUNT IS SHARED AND THE MODE IS NOT. ESMFold2's own sampler is
  // eleven steps, so a flow arm has nothing to escape from and runs MORE of
  // them than the shipped one - see ESMFOLD2_COUNTS. Offering a choice whose
  // every option is equivalent-or-worse is the same fault as offering one that
  // is ignored, so the mode row is hidden for it.
  // 🔴 AND OpenDDE HIDES IT FOR THE SAME REASON, MEASURED ON ITS OWN SAMPLER.
  // Flow and diffusion cost it the SAME 16.1 s at sixteen steps and flow is
  // plainly worse - TM 0.8307 and 0.8601 against 0.9044 and 0.9169 on 6MRR,
  // two seeds each and no overlap. Its sampler re-noises every step, which is
  // what a flow arm exists to escape, and escaping it here loses the structure
  // rather than buying time. See OPENDDE_SAMPLER_MODE.
  // 🔴 AND rosettafold3 HIDES IT BECAUSE FLOW BREAKS THE STRUCTURE, which is a
  // different reason from OpenDDE's "measurably worse": N-CA 6.94 A against
  // 1.46 and a collapsed backbone, with pLDDT reading 81.47 as if fine. See
  // NO_FLOW_SAMPLER_FAMILIES.
  const modeNode = document.getElementById("af3ModeGroup");
  if (modeNode !== null) {
    modeNode.hidden = !af3 || family === "opendde"
      || NO_FLOW_SAMPLER_FAMILIES.includes(family);
  }
  // 🔴 AND A MODEL WITH NO ALIGNMENT HIDES THE MSA ROW RATHER THAN IGNORING IT.
  // `disable_msa_features` is true in ESMFold2's checkpoint; a search left on
  // screen would run, take a minute of somebody else's server, and be
  // discarded - which is the "quietly ignored control" this function exists to
  // prevent, one step worse.
  const singleSequence = SINGLE_SEQUENCE_FAMILIES.includes(family);
  for (const id of ["msaModeGroup", "maxMsaGroup"]) {
    const node = document.getElementById(id);
    if (node !== null) node.hidden = singleSequence;
  }
  // ...and the row that replaces them, which is the same question for a model
  // whose evolutionary information comes from a language model rather than an
  // alignment. Shown exactly where the MSA row is not.
  const plmNode = document.getElementById("plmGroup");
  if (plmNode !== null) plmNode.hidden = !singleSequence;
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
  if (sampled) syncAf3Count();
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
  const mode = samplerModeFor(chosenFamily(),
    document.getElementById("af3-mode")?.value ?? "diffusion");
  // ...and ESMFold2's table has one mode, so the shared select cannot pick a
  // row that is not there.
  const ef2 = SINGLE_SEQUENCE_FAMILIES.includes(chosenFamily());
  // ...and OpenDDE's own, because more steps make its fold worse; see
  // OPENDDE_COUNTS for the two targets and nine folds that say so.
  const table = ef2 ? ESMFOLD2_COUNTS
    : chosenFamily() === "opendde" ? OPENDDE_COUNTS : AF3_COUNTS;
  const { label, values, preferred } = table[ef2 ? ESMFOLD2_SAMPLER_MODE : mode]
    ?? table.flow ?? table.diffusion;
  const title = document.getElementById("af3-count-label");
  if (title !== null) title.textContent = label;
  const select = document.getElementById("af3-count");
  if (select === null) return;
  // 🔴 THE OPTION SHOWS WHAT RUNS AND CARRIES WHAT THE MODEL CALLS IT. For
  // ESMFold2 the two differ - `max_inference_sigma` turns a 15-step schedule
  // into 11 - so a dial reading 15 beside a status line reading 11 is the page
  // contradicting itself. The VALUE stays the preset's own number, because that
  // is what names a preset; only the text changes.
  const shown = ef2
    ? (value) => String(actualSteps(`${ESMFOLD2_SAMPLER_MODE}-${value}`))
    : String;
  select.replaceChildren(...values.map((value) => Object.assign(
    document.createElement("option"),
    { value: String(value), textContent: shown(value), selected: value === preferred })));
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
function forcePlddtColours(scored = true) {
  // ...one implementation. This was the only one that worked - it goes through
  // py2Dmol's own select - and `setColourMode` now starts there for every
  // caller, so this is the same call under the name the AF3 path uses.
  //
  // 🔴 AND `scored` IS NOT DECORATION. OpenDDE has no confidence head, so its
  // B-factor column is zero everywhere - and the pLDDT ramp paints zero RED,
  // which reads as a uniformly terrible fold rather than an absent
  // measurement. EF2-fast takes chain colours for the same reason.
  setColourMode(scored ? "plddt" : "chain");
}

/** Whether the model now selected produces a pLDDT at all. */
function hasConfidenceHead() {
  return !MODELS_WITHOUT_CONFIDENCE.includes(chosenFamily());
}

/** EF2-fast's own, keyed the same way and for the same reason. */
let esmfold2Trunk;

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
  // 🔴 ...EXCEPT WHERE A RESIDUE IS SEVERAL TOKENS, WHICH IS EVERY MODIFIED
  // AMINO ACID. The note above is right about ligands and wrong about these:
  // AF3 atomises a modified residue into one token PER ATOM (boltz2 is the
  // exception) while py2Dmol draws it as ONE position, because toPdb writes
  // those atoms under one residue number with a backbone among them. Measured
  // on a twelve-residue chain with SEP at position 3: 21 tokens against 12
  // positions, so the matrix was nine rows too wide and everything after the
  // modification addressed the wrong residue. viewerTokens is the map and
  // these two are the only places that need to know.
  const collapses = (values, keep) => keep !== undefined && keep.length < paeSize(values);
  const paeForViewer = (values, keep) => (collapses(values, keep)
    ? matrixForViewer(values, keep) : paeMatrix(values, paeSize(values)));
  const viewerWidth = (values, keep) => (collapses(values, keep)
    ? keep.length : paeSize(values));
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

  // 🔴 FORCED IN CODE, NOT ONLY HIDDEN. Hiding a control does not change its
  // value: the shared `#af3-mode` select still reads "flow" behind a hidden
  // row, which is the trap docs/EF2FAST.md records for that model an hour
  // after hiding its own.
  const opendde = chosenFamily() === "opendde";
  // 🔴 FORCED FOR rosettafold3 TOO, and for a worse reason than OpenDDE's - see
  // `samplerModeFor`. Hiding the row does not change the select's value, which
  // is the trap the comment above records.
  const mode = samplerModeFor(chosenFamily(),
    document.getElementById("af3-mode")?.value ?? "diffusion");
  const counts = opendde ? OPENDDE_COUNTS : AF3_COUNTS;
  // 🔴 THE SAME FALLBACK AS `syncAf3Count`, AND IT WAS MISSING HERE. That
  // function reads `table[mode] ?? table.flow ?? table.diffusion`; this one
  // subscripted the table and took `.preferred` off whatever came back, so a
  // mode with no row - a stale select value, a removed option, a family whose
  // table is narrower than AF3's - threw "cannot read properties of undefined"
  // in the middle of starting a fold rather than falling back. Two readings of
  // one table, one of them guarded, is this file's own stale-allow-list trap.
  // `test/sampler-options.test.js` gates the two lists against each other.
  const row = counts[mode] ?? counts.diffusion ?? counts.flow;
  const asked = Number(document.getElementById("af3-count")?.value)
    || row.preferred;
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
    // A declared bond changes what is folded, so a trunk cached without one is
    // not this fold's - the `chainKinds` rule beside it. Off `foldContext`,
    // because this function is handed chains and ligands and never the rows.
    bonds: foldContext.bonds ?? [],
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
    ? foldStem(`${MODEL_STEMS[family] ?? family}_${predictionCount}`)
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
    remoteTap("frame", pdb);
    if (signal.aborted || api?.frameFromText === undefined) return;
    const registry = window.py2dmol_viewers ?? {};
    const renderer = registry[Object.keys(registry)[0]]?.renderer;
    const object = renderer?.objectsData?.[renderer?.currentObjectName];
    if (renderer === undefined || object === undefined) return;
    try {
      const index = object.frames.length;
      // ...opened before the frame is added, so the canvas is measured against
      // a container that is actually on screen.
      // ...and the recycles are over, so the structure takes the big slot
      // back. See contactsBig.
      if (index === 0) { revealViewer(renderer); contactsBig(false); }
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
        // ...pLDDT only where there is one; see setColourMode's call in
        // loadIntoViewer. A model with no confidence head paints a zero
        // B-factor as the colour of no confidence.
        forcePlddtColours(hasConfidenceHead());
        oriented = true;
      }
      renderer.render("live-frame");
    } catch (cause) {
      console.warn("could not draw a frame", cause);
    }
  };
  let viewerKeep;
  let viewerModified = [];
  const result = await foldAf3({
    sequence, mode, calls, recycles, weights, device, signal,
    alignment: alignmentBlocks, maxMsaSequences, ligandCodes, modifications,
    chainKinds, reuse, bonds: foldContext.bonds,
    // 🔴 WHICH TOKENS THE VIEWER DRAWS, and the reason every matrix below goes
    // through it: a modified residue is one POSITION and ten TOKENS, so its
    // PAE and its contact map are wider than the structure they belong to and
    // every residue after it reads somebody else's row. Handed over before the
    // trunk starts, so the live contact map is collapsed the same way.
    onBatch: (batch) => {
      viewerKeep = viewerTokens(batch);
      viewerModified = modifiedPositions(batch, viewerKeep);
    },
    // 🔴 WHAT PRODUCED THE FILE, BECAUSE THE FILE DID NOT SAY. A saved PDB from
    // this path carried no REMARK at all - no model, nothing about the B-factor
    // column - while the AlphaFold 2 path beside it has always written one and
    // EF2-fast writes two. Seven families share the AF3 writer, so "AlphaFold 3"
    // here is `modelName` and not a literal: an OpenBind-0 fold that claimed to
    // be AlphaFold 3 would be worse than an anonymous file.
    //
    // Only the provenance is ours to state. Whether the B-factor column holds a
    // pLDDT depends on whether this family has a confidence head, which is not
    // known until the fold returns - so af3-model.js adds that line, where
    // `result.scores` can be read.
    remark: [`${modelName} PREDICTION BY LOCALFOLD (https://localfold.org)`],
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
      liveContacts = contactMapFor(contactProbs, viewerKeep);
      showTrunkContacts(liveContacts, chains);
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
    // 🔴 A MODEL WITHOUT A CONFIDENCE HEAD HANDS THE VIEWER NO SCORES AND NO
    // PAE. OpenDDE's head is its own design on its own distance grid, so a
    // fold returns a structure and a contact map and nothing else - and the
    // pLDDT palette over an absent B-factor paints a uniform "no confidence",
    // which is a claim rather than a blank. EF2-fast established the shape:
    // colour by chain, and say so on the status line.
    const scored = result.confidence !== undefined;
    const pae = scored ? result.confidence.predictedAlignedError : undefined;
    await loadIntoViewer({
      stem, pdb: timeline[0],
      scores: scored ? confidenceJson(chains.join(""), result.confidence) : undefined,
      a3m: alignment,
      chainLengths: chains.map((chain) => chain.length),
      ...(pae === undefined ? {}
        : { pae: paeForViewer(pae, viewerKeep), length: viewerWidth(pae, viewerKeep) }),
      confidence: result.confidence,
    });
    // 🔴 AND THE PREDICTION IS REGISTERED, WHICH AF3 NEVER DID. The download
    // buttons read `predictions`, and only the AlphaFold 2 path ever wrote to
    // it - so an AF3 fold produced a structure on screen with no way to save
    // what the model actually computed, and the panel holding those buttons
    // stayed hidden. The trajectory goes in as one model per sampler call,
    // which is the AF3 analogue of one model per recycle.
    recordPrediction({
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
      // ...and with no confidence head, the contacts travel ALONE - which is
      // what EF2-fast's archive does, and why `contactSource` is one field on
      // every path rather than a copy inside the confidence object.
      confidence: scored
        ? { ...result.confidence, contactProbs: result.contactProbs } : undefined,
      scores: scored ? confidenceJson(chains.join(""),
        { ...result.confidence, contactProbs: result.contactProbs }) : undefined,
      a3m: alignment,
      chains,
      chainLengths: chains.map((chain) => chain.length),
      // ...the same one field, so the archive has one thing to read. The
      // confidence object keeps its own copy because the scores card and the
      // heatmap take the whole object; this is the archive's single door.
      contactSource: { contactProbs: result.contactProbs },
      model: modelName,
      // ...AF3 needs this for exactly the same reason, and nothing had ever set
      // it: a fold with a ligand or a modified residue has more tokens than
      // residues, so the archive's own fallback refused it.
      tokens: result.batch === undefined ? undefined
        : tokenLayoutFrom(result.batch.asymId, result.batch.residueIndex),
      ...foldContext,
    }, family);
    void rememberSessionWhenSettled(lastPrediction);
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
      // ...and nothing at all where the model has no confidence head; see the
      // note at `scored` above.
      first.confidence = scored ? {
        predictedAlignedError: result.confidence.predictedAlignedError,
        plddt: result.confidence.plddt,
      } : undefined;
      // ...and frame zero's contact map, which is the finished trunk's: every
      // recycle is over before the sampler emits anything.
      // 🔴 AND IN THE VIEWER'S INDEX SPACE, like the live one above and the
      // PAE below it. This was the ONE call of the four that a modified
      // residue reaches and that did not collapse - so the fold that landed
      // carried a 13-wide PAE beside a 22-wide contact map, measured on
      // GWSTELEKHRSVQ + SEP@3. The live map is thrown away the moment frame
      // zero exists, so this is the one a reader ever looks at.
      const contact = result.contactProbs === undefined
        ? undefined : contactMapFor(result.contactProbs, viewerKeep);
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
      frame.confidence = !scored ? undefined : last ? result.confidence : {
        predictedAlignedError: result.confidence.predictedAlignedError,
        plddt: result.confidence.plddt,
      };
      if (last && scored) {
        // The PAE rides on the frame the page lands on, so scrubbing away and
        // back does not blank a matrix that was on screen a moment earlier.
        const errors = result.confidence.predictedAlignedError;
        frame.pae = paeForViewer(errors, viewerKeep);
        frame.pae_n = viewerWidth(errors, viewerKeep);
      }
      viewer.addFrame(frame, viewerObject);
    }
    const object = viewer.objects?.find((entry) => entry.name === viewerObject);
    if (object?.frames?.length) viewer.setFrame(object.frames.length - 1);
    forcePlddtColours(scored);
    showModifiedSidechains(viewer, viewerObject, viewerModified);
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
  // ...and a SMILES ligand names itself by its string, truncated, because the
  // whole of a drug-like SMILES does not fit on a status line and the first
  // twenty characters are enough to recognise the one you typed.
  if (ligandCodes.length > 0) {
    what.push(ligandCodes.map((entry) => (typeof entry === "string" ? entry
      : entry.smiles.length > 24 ? `${entry.smiles.slice(0, 21)}...` : entry.smiles))
      .join(", "));
  }
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
  // 🔴 ONLY WHERE THE MODEL HAS A CONFIDENCE HEAD. OpenDDE returns no pLDDT,
  // and `undefined.toFixed` is what the status line said instead of a result.
  if (result.meanPlddt !== undefined) {
    detail.push(`pLDDT ${result.meanPlddt.toFixed(1)}`);
  } else {
    detail.push("no confidence head");
  }
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
  //
  // 🔴 BUT A BROKEN CHAIN IS SAID OUT LOUD, which the page claimed to do and did
  // not: the check meant for this fold was written against `result.geometry`
  // and placed in the AlphaFold 2 path, where no `result` exists, so no AF3-
  // lineage fold was ever checked. The number stays with the tools; the WARNING
  // appears only when `chainGeometryVerdict` refuses the fold - IntelliFold-2 in
  // Flow returns one such fold in six at pLDDT 83.
  const chain = chainGeometryVerdict(result.geometry ?? {}, { plddt: result.meanPlddt });
  if (!chain.ok) detail.push("🔴 NOT A CHAIN - the backbone is broken, and pLDDT does not measure that");
  status(`${modelName} · ${what.join(" + ")} · ${detail.join(" · ")}`);
}

/**
 * A whole ESMFold2 fold, from the page.
 *
 * 🔴 IT IS THE SHORTEST FOLD PATH HERE, BECAUSE THE MODEL HAS THE FEWEST
 * INPUTS. No alignment, no template, no recycle setting a reader can turn -
 * `num_loops` is the checkpoint's - and no confidence head, so there is no PAE
 * panel, no pLDDT colouring and no scores card. What it does have is ligands,
 * DNA, RNA and complexes, because its featuriser is AF3's.
 *
 * 🔴 AND THE COLOUR SCHEME IS NOT pLDDT. Every other fold on this page is
 * coloured by the confidence head's per-atom answer; this model has no such
 * head, and a structure drawn under the pLDDT scheme with a zero B-factor is
 * uniformly the colour of NO confidence - which reads as a terrible fold rather
 * than as an absent measurement. It is coloured by chain instead, and the
 * status line says the model reports no confidence rather than leaving a reader
 * to notice the card is missing.
 */
/**
 * The sampler row, as one of SAMPLER_PRESETS.
 *
 * 🔴 IT REFUSES A COMBINATION IT CANNOT NAME rather than falling back to the
 * default. A dial that silently ignores what it was set to is the failure
 * syncModelControls exists to prevent, and a preset table is exactly the kind
 * of thing that gains a value on the page before it gains one in the code.
 */
function samplerPreset() {
  // 🔴 THE MODEL DECIDES THE MODE, NOT THE HIDDEN SELECT. Hiding a control does
  // not change its value - that is what put "unknown model esmfold2" in front
  // of somebody folding an oligomer - so this reads the model's own answer and
  // never the shared row.
  const mode = ESMFOLD2_SAMPLER_MODE;
  // 🔴 AN EMPTY DIAL IS "NOTHING CHOSEN", NOT AN UNKNOWN CHOICE. A `<select>`
  // assigned a value none of its options carry reports "" - which is what
  // happens whenever the count dial has not been rebuilt for this model yet, or
  // a caller sets a step count from another model's table. The first version
  // threw on it and the message named a sampler called `flow-`, which describes
  // the symptom and not the cause.
  const chosen = document.getElementById("af3-count")?.value;
  const steps = chosen === undefined || chosen === ""
    ? String(ESMFOLD2_COUNTS[mode]?.preferred ?? ESMFOLD2_COUNTS.diffusion.preferred)
    : chosen;
  const name = `${mode}-${steps}`;
  if (SAMPLER_PRESETS[name] === undefined) {
    throw new Error(`no ESMFold2 sampler called ${name}; `
      + `known: ${Object.keys(SAMPLER_PRESETS).join(", ")}`);
  }
  return name;
}

async function foldWithEsmfold2(chains, chainKinds, ligandCodes, signal, modelLoad,
                                modifications = []) {
  // ...read ONCE, at the top, because the row can move while a fold runs and
  // the name and the family have to describe the same model.
  const family = chosenFamily();
  const modelName = MODEL_LABELS[family] ?? "EF2-fast";
  const sequence = chains.join(":");
  status(`${modelName} · loading`);
  // ...the long name is for the download dial, where provenance matters; the
  // status line uses the short one, because it is written many times a fold.
  // 🔴 THE LIGAND DICTIONARY IS FETCHED, NOT BUNDLED, exactly as on the AF3
  // path - and from the same place, because these are the same components. A
  // fold touches only the codes its ligands name and the PDB serves each as one
  // small mmCIF; the 21 polymer components stay baked.
  const ligands = [];
  for (const entry of ligandCodes) {
    // A structure rather than a code; see the note in web/af3-model.js.
    if (typeof entry !== "string") {
      status(`${modelName} · building ${entry.code ?? "ligand"}`);
      ligands.push(await smilesComponent(entry.smiles, { code: entry.code ?? "LIG" }));
      continue;
    }
    status(`${modelName} · fetching ligand ${entry}`);
    const response = await fetch(ccdUrl(entry), { signal });
    if (!response.ok) {
      throw new Error(`No chemical component ${entry} at the PDB (${response.status})`);
    }
    ligands.push(parseCcdComponent(await response.text()));
  }
  // 🔴 AND A MODIFIED RESIDUE'S COMPONENT FROM THE SAME PLACE, for the reason
  // web/af3-model.js gives: the featuriser is synchronous, and this is the one
  // piece of a batch that cannot be computed from the sequence.
  const modifyWith = [];
  for (const modification of modifications) {
    status(`${modelName} · fetching modified residue ${modification.code}`);
    const response = await fetch(ccdUrl(modification.code), { signal });
    if (!response.ok) {
      throw new Error(`No chemical component ${modification.code}`
        + ` at the PDB (${response.status})`);
    }
    modifyWith.push({ chain: modification.chain, position: modification.position,
                      ...parseCcdComponent(await response.text()) });
  }
  throwIfAborted(signal);
  const loaded = await (modelLoad
    ?? loadEsmfold2Weights(undefined,
                           { languageModel: usesLanguageModel(), family: chosenFamily() }));
  // 🔴 ASKED AGAIN HERE, BECAUSE THE PRELOAD DECIDED IT EARLIER AND THE MEMO
  // OUTLIVES BOTH. `startModelPreload` skips the tower's 223.6 MiB when the
  // entities hold no protein or the PLM row says none - and either can have
  // changed since, or the promise can have been built for a previous fold that
  // did not want it. Idempotent, and it is a head start rather than a
  // correctness fix: the store serves the blocks on demand either way.
  if (usesLanguageModel()) loaded.language.prefetch?.();
  throwIfAborted(signal);
  const device = await getDevice();
  throwIfAborted(signal);

  predictionCount += 1;
  const stem = foldStem(
    `${MODEL_STEMS[chosenFamily()] ?? "ef2_fast"}_${predictionCount}`);
  openBlankFold(stem);
  viewer = undefined;
  viewerObject = undefined;

  const api = window.py2Dmol;
  let liveContacts;
  let drawn = 0;
  let colouring = false;
  const framePdbs = [];
  const drawLiveFrame = (pdb) => {
    remoteTap("frame", pdb);
    if (signal.aborted || api?.frameFromText === undefined) return;
    const registry = window.py2dmol_viewers ?? {};
    const renderer = registry[Object.keys(registry)[0]]?.renderer;
    const object = renderer?.objectsData?.[renderer?.currentObjectName];
    if (renderer === undefined || object === undefined) return;
    try {
      if (object.frames.length === 0) { revealViewer(renderer); contactsBig(false); }
      const frame = api.frameFromText(pdb);
      frame.name = frame.label = frame.title = `sampler_${drawn++}`;
      if (liveContacts !== undefined) frame.maps = { contact: liveContacts };
      renderer.addFrame(frame, renderer.currentObjectName);
      renderer.setFrame(object.frames.length - 1);
      // 🔴 THE COLOUR MODE IS SET ON THE FIRST LIVE FRAME, NOT AT THE END. It
      // was only set after the fold finished, so every frame drawn WHILE the
      // sampler ran came up in `auto`, which resolves to rainbow - and the
      // whole point of a per-frame certainty is watching it during the fold.
      // Reported as "frames added during diffusion still showing rainbow".
      //
      // 🔴 AND ONCE, NOT PER FRAME. The select's change handler renders, so
      // calling it eleven times is eleven extra renders for one state change.
      //
      // 🔴 AND THE VIEW IS FOUND ON THE FIRST FRAME, WHICH THIS PATH NEVER DID.
      // py2Dmol orients when it INGESTS A FILE, and this path draws frames
      // instead - so the whole trajectory ran at whatever camera the blank
      // object happened to have, and then `loadIntoViewer` orientated at the
      // very end. That is both halves of what was reported: no best view on the
      // first frame, and a different angle on the last.
      if (!colouring) {
        colouring = setColourMode(certainty === undefined ? "chain" : "plddt");
        orientBestView(renderer);
      }
    } catch (error) {
      console.warn("live frame skipped:", error);
    }
  };

  // 🔴 THE FRAMES MUST BE FITTED, BECAUSE THIS SAMPLER RE-POSES EVERY STEP.
  // `centreRandomAugmentation` draws a fresh rotation and translation of the
  // whole system at the top of each step - it is how the sampler is equivariant
  // and the model was trained with it in the loop - so consecutive frames differ
  // by a rigid motion far larger than anything the denoiser did, and unfitted
  // playback is a protein tumbling. AF3's path has fitted its trajectory since
  // it had one; this is the same function, not a second one.
  //
  // 🔴 AND TO THE FIRST FRAME, NOT THE LAST, because the frames are drawn as
  // they are computed and there is no last one yet.
  let reference = null;
  let slots;
  // 🔴 THE COLOUR IS THE DISTOGRAM'S CERTAINTY, NOT A pLDDT, AND THE PDB SAYS
  // SO IN A REMARK. This checkpoint has no confidence head - 820 tensors and
  // not one named confidence, plddt, pae or pde - so what goes in the B-factor
  // is an ORDERING with nothing to calibrate a number against. It is written
  // there because that is the only column a viewer can colour from, and a
  // downloaded file that carried an uncommented pLDDT-shaped column would be
  // read as one. See CERTAINTY in src/esmfold2/distogram-webgpu.js for the
  // sweep that chose its three constants.
  let certainty;
  let lastFrameCertainty;
  const REMARK = "REMARK   1 B-FACTOR IS DISTOGRAM CERTAINTY (0-100), NOT pLDDT."
    + "\nREMARK   1 THIS ESMFOLD2 CHECKPOINT HAS NO CONFIDENCE HEAD.";
  const withRemark = (pdb) => `${REMARK}\n${pdb}`;

  const started = performance.now();
  // 🔴 THE SAME CACHE AlphaFold 3's PATH HAS, AND FOR THE SAME REASON: the trunk
  // is the fold, so changing only the sampler should cost only the sampler. The
  // key is what the TRUNK depends on and nothing else - the checkpoint, the
  // chains and their kinds, the ligands, the pass count, and which language
  // model, since "none" and ESM-C 600M share a family and produce different
  // pairs. The seed is in it only when masking is on, because that is the only
  // way the seed reaches the trunk: `lm_mask_pct` is zero in this checkpoint, so
  // asking for a different SAMPLE reuses the trunk here where AF3 re-runs it.
  // 🔴 AND THE MASK IS IN IT, WITH THE SEED BEHIND IT. `lm_mask_pct` replaces a
  // fraction of the residues with the mask token BEFORE the tower runs, drawn
  // from the seed - so with masking on, two seeds are two different trunk
  // inputs and a key without them hands the second fold the first one's pair.
  // This checkpoint sets the fraction to 0, so the seed never reaches the trunk
  // and changing it reuses; but the config class documents single-sequence
  // checkpoints as setting 0.1, so a future bundle turns this on by existing
  // and the key has to be right before that rather than after.
  const lmMask = (loaded.shape.lmMaskPct ?? 0);
  const trunkKey = JSON.stringify({
    family: chosenFamily(), chains, chainKinds, ligandCodes,
    // A modification changes what is folded, so a trunk cached for the plain
    // chain is not this fold's.
    modifications: modifications.map((one) => `${one.code}@${one.position}`),
    loops: recycleCount() + 1,
    plm: plmChoice(), languageModel: usesLanguageModel(),
    lmMask, maskSeed: lmMask > 0 ? randomSeed() : null,
  });
  const reuse = esmfold2Trunk?.key === trunkKey ? esmfold2Trunk.reusable : undefined;
  let viewerKeep;
  let viewerModified = [];
  const result = await foldEsmfold2(device, {
    reuse,
    wantReusable: true,
    sequence,
    entities: { sequence, chainKinds, ligands, modifications: modifyWith },
    // ...and which tokens the viewer will draw, so this path's contact map is
    // collapsed the way the AF3 one is. See viewerTokens.
    onBatch: (batch) => {
      viewerKeep = viewerTokens(batch);
      viewerModified = modifiedPositions(batch, viewerKeep);
    },
    // 🔴 THE RECYCLE DIAL DRIVES THIS TRUNK TOO, AND USED NOT TO. Its loop
    // count came from the checkpoint and the control beside it did nothing -
    // the "quietly ignored control" syncModelControls exists to prevent, which
    // is why the MSA row is hidden here rather than left on screen. The mapping
    // is exact: upstream runs `range(num_loops + 1)` and this checkpoint's
    // `num_loops` is 3, which is the dial's own default, so the default fold is
    // the same four passes it always was.
    shape: { ...loaded.shape, loops: recycleCount() + 1 },
    weights: loaded.weights,
    tower: languageModelRunner(device, new GpuBufferAllocator(device), loaded,
                               loaded.shape.pairChannels),
    sampler: samplerPreset(),
    seed: randomSeed(),
    // 🔴 THIS MODEL'S "SINGLE SEQUENCE". Without ESM-C it has no evolutionary
    // information at all - measured on a 76-mer, the fold moves 10.96 A, the
    // distogram predicts NO long-range contact, and the certainty falls from
    // 0.95 to 0.42, which is the confidence estimate correctly reporting that
    // the answer is worthless.
    languageModel: usesLanguageModel(),
    // ...and how big it is, so the bar's language band is this tower's and not
    // the one the constants were fitted against.
    languageModelMiB: loaded.language.megabytes,
    // 🔴 EACH FRAME GETS ITS OWN COLOUR, WHICH NEEDS THE DISTOGRAM RESIDENT.
    // The trunk's own certainty is fixed for a fold, so every frame would wear
    // the same one - and the interesting thing about a trajectory is watching
    // it become confident. Scoring each frame against the distogram costs the
    // logits staying on the device, 46 MiB at 300 tokens, released with the
    // last frame.
    frameCertainty: true,
    // 🔴 THE LINE AND THE BAR ARE TWO CALLBACKS NOW, AS AF3's ARE. One phase
    // word plus a percentage on the line; the fraction drives the bar. The
    // first version wrote a stage name per stage, and a two-millisecond recycle
    // between two multi-second trunk passes made it flicker.
    onStatus: (text) => { if (!signal.aborted) status(`${modelName} · ${text}`); },
    onProgress: (fraction) => { if (!signal.aborted) progress(fraction); },
    // 🔴 THE CONTACT MAP EXISTS BEFORE ANY STRUCTURE DOES, because the
    // distogram head runs off the trunk and the sampler has not started. It is
    // held until there is a frame to hang it on, exactly as the AF3 path holds
    // its own.
    onContacts: (contacts, trunkCertainty) => {
      liveContacts = contactMapFor(contacts, viewerKeep);
      certainty = trunkCertainty;
      showTrunkContacts(liveContacts, chains);
    },
    // 🔴 `denoised` AND NOT `coordinates`, AND THE REASON IS THE CAMERA. The
    // sampler's own walk starts as Gaussian noise at sigma 411 and ends at a
    // protein, so no fixed camera holds both and the early frames are not a
    // picture of anything. `denoised` is the model's predicted structure at each
    // call - EDM preconditioning included, so at a large noise level it is
    // almost all network - and is protein-sized in every frame. AF3's path
    // records the same finding, measured: a radius of gyration of 1896 A at
    // step 4 against 11.1 at the end.
    onStep: ({ denoised, features, certainty: frameCertainty }) => {
      if (signal.aborted) return;
      const dense = toDensePositions(features, denoised);
      if (slots === undefined) slots = alphaCarbons(features.batch);
      if (reference === null) {
        reference = toPoints(dense, features.batch.tokens * features.batch.dense);
      }
      // ...this frame's OWN agreement with the distogram, so an early frame
      // that has not converged is coloured as one rather than wearing the
      // finished structure's confidence. The trunk's mode-based certainty is
      // the fallback, and it is the same quantity measured a different way -
      // the two scored a tie on the sweep.
      const shown = frameCertainty ?? certainty;
      const pdb = withRemark(fittedPdb(features.batch, dense, reference, slots,
        shown === undefined ? null : spreadOverAtoms(features, shown, 100)));
      framePdbs.push(pdb);
      drawLiveFrame(pdb);
      lastFrameCertainty = shown;
    },
  });
  throwIfAborted(signal);

  // 🔴 THE ANSWER IS FITTED ONTO THE SAME REFERENCE AS THE TRAJECTORY, or the
  // last frame of the play bar jumps by a rigid motion the fold did not make.
  // It is still `result.coordinates` - the sampler's own answer, not the last
  // denoiser call - and at the bottom of the schedule the two agree to a
  // fraction of an angstrom anyway.
  // 🔴 THE FINISHED STRUCTURE KEEPS THE LAST FRAME'S SCORE, not the trunk's.
  // The two are the same quantity read two ways, but the play bar would step
  // from a per-frame colour to a different one on its last frame, which reads
  // as the fold changing its mind at the end.
  certainty = lastFrameCertainty ?? result.certainty ?? certainty;
  const bFactors = certainty === undefined
    ? null : spreadOverAtoms(result.features, certainty, 100);
  const finalDense = toDensePositions(result.features, result.coordinates);
  const pdb = withRemark(reference === null
    ? toPdb(result.features.batch, finalDense, bFactors)
    : fittedPdb(result.features.batch, finalDense, reference,
                slots ?? alphaCarbons(result.features.batch), bFactors));
  // 🔴 AND IN THE VIEWER'S SPACE, like the live one above: this is the copy
  // that lands on frame zero, and it is the one the AF3 path was caught by.
  const contactMap = contactMapFor(result.contacts, viewerKeep);
  // 🔴 THE ESTIMATED pAE IS NOT DRAWN, AND THE REASON IS MEASURED. It orders
  // pairs WITHIN a fold at 0.746 against AlphaFold 3's real PAE - genuinely
  // useful - and ACROSS folds it is INVERTED, at -0.867. Three random sequences
  // that fold to nothing (0-3 contacts, certainty 0.53-0.62) score 6.6-7.1 A
  // where two real proteins that fold well score 8.7-8.8; ubiquitin with its
  // language model removed collapses to 0 contacts and certainty 0.42 and its
  // pAE IMPROVES, 8.84 -> 7.97. A panel that looks BETTER on a failed fold is
  // the worst possible panel, because the fold somebody checks the PAE on is
  // the one they doubt - the same reason the per-residue certainty colour was
  // measured and not shipped. `src/esmfold2/aligned-error.js` and
  // `tools/pae-transfer.py` keep the estimator and the numbers; nothing draws
  // it until the inversion is fixed.
  const paeMap = undefined;
  // 🔴 THE CAMERA IS SAVED ACROSS THE RELOAD, OR THE VIEW JUMPS AT THE END.
  // `loadIntoViewer` ingests a FILE, and py2Dmol orients the camera when it
  // parses one - so the trajectory the reader has been watching, and possibly
  // rotating, snaps to a new angle the moment the last frame lands. The AF3
  // path has saved and restored it since it had a trajectory; this one had
  // not. Reported as "the frames change angle when last frame is added".
  // ...taken off the RENDERER, not off `viewer`, which is undefined until
  // loadIntoViewer runs and therefore held no camera to save.
  const registry = window.py2dmol_viewers ?? {};
  const liveRenderer = registry[Object.keys(registry)[0]]?.renderer;
  const camera = { ...(liveRenderer?.viewerState ?? {}) };
  const live = liveRenderer?.objectsData?.[liveRenderer?.currentObjectName];
  if (live?.frames !== undefined) live.frames.length = 0;
  await loadIntoViewer({ stem, pdb: framePdbs[0] ?? pdb, scores: {} });
  if (viewer !== undefined && Object.keys(camera).length > 0) {
    Object.assign(viewer.viewerState, camera);
    viewer.render?.("localfold.restore-camera");
  }
  if (api?.frameFromText !== undefined && viewer !== undefined) {
    const object = viewer.objectsData?.[viewerObject];
    const first = object?.frames?.[0];
    if (first !== undefined) {
      // 🔴 py2Dmol NAMES THE FRAME IT INGESTED `recycle_0`, WHICH THIS MODEL
      // HAS NONE OF. Its file path assumes an AlphaFold 2 trajectory; every
      // frame here is a sampler step, and a play bar that starts at "recycle_0"
      // and continues "sampler_1" describes two things that are one thing.
      first.name = first.label = first.title = "sampler_0";
      if (contactMap !== undefined) first.maps = { ...first.maps, contact: contactMap };
      if (paeMap !== undefined) first.maps = { ...first.maps, pae: paeMap };
    }
    // 🔴 THE FINISHED STRUCTURE REPLACES THE LAST SAMPLER FRAME, as on the AF3
    // path: the last frame IS that step's output, so appending it makes a play
    // bar that ends on the same picture twice.
    for (const [index, text] of [...framePdbs.slice(1, -1), pdb].entries()) {
      const frame = api.frameFromText(text);
      const last = index === framePdbs.length - 2;
      frame.name = frame.label = frame.title = last ? "final" : `sampler_${index + 1}`;
      // ...the same two maps on every frame. They come off the TRUNK, so they
      // are the same for the whole trajectory; the panel reads whichever frame
      // the play bar is on, and a frame with no maps blanks it.
      const maps = { ...(contactMap === undefined ? {} : { contact: contactMap }),
                     ...(paeMap === undefined ? {} : { pae: paeMap }) };
      if (Object.keys(maps).length > 0) frame.maps = maps;
      viewer.addFrame(frame, viewerObject);
    }
    // 🔴 THE pLDDT PALETTE ON A NUMBER THAT IS NOT A pLDDT, DELIBERATELY. It is
    // the right palette for a 0-100 confidence-like scale and every reader of
    // this page already knows how to read it; what must not happen is the WORD
    // appearing anywhere, which is why the status line names the quantity and
    // the file carries a REMARK. With no certainty at all it falls back to
    // chain colours rather than colouring a zero B-factor as no confidence.
    // ...again at the end, because loadIntoViewer's own ingestion resets the
    // renderer's data and recomputes its colours.
    setColourMode(certainty === undefined ? "chain" : "plddt");
    viewer.setFrame((viewer.objectsData?.[viewerObject]?.frames?.length ?? 1) - 1);
    // ...and the modification drawn, as on the AF3 path: the ribbon runs
    // through its alpha carbon exactly as through the residue it replaced.
    showModifiedSidechains(viewer, viewerObject, viewerModified);
    // ...and DRAWN: the set is stored on the object and the atoms are
    // materialised by the next frame, which on the AF3 path is the render
    // that follows it there. Without this the modification's side chain is
    // asked for and not shown - measured, the object's set held residue 2
    // while the drawn array stayed at 13 positions.
    viewer.render("ef2-final");
  }

  recordPrediction({
    stem, pdb, chains,
    chainLengths: chains.map((chain) => chain.length),
    // 🔴 NO `confidence`, AND THAT IS THE HONEST SHAPE. Everything that reads a
    // prediction's confidence - the scores card, the archive's summary, the PAE
    // panel - asks for fields this checkpoint has no head to compute. An object
    // carrying zeros would be read as the model's opinion.
    // 🔴 ONE FIELD FOR THE CONTACT MAP, WHATEVER PRODUCED IT. See the note on
    // `contactSource` at the download button: this used to be `contacts` here,
    // `confidence.contactProbs` on the AF3 path and `contactSource` on AF2's,
    // and the archive knew about two of the three - so the model whose contact
    // map is its ONLY score wrote an archive without one while the panel on
    // screen showed it.
    contactSource: { contactProbs: result.contacts },
    // 🔴 AND THE pAE, WHICH IS NOT A `confidence` FIELD AND MUST NOT BECOME
    // ONE. It is estimated from the distogram rather than predicted by a head -
    // see src/esmfold2/aligned-error.js - so putting it under `confidence`
    // would let every reader that tests for that object conclude this
    // checkpoint has one, and start looking for the pLDDT and pTM beside it.
    // It is its own field, named for what it is.
    alignedError: result.alignedError,
    // 🔴 WITHIN EACH CHAIN AND ACROSS IT, KEPT APART. The certainty a residue
    // wears is about its own chain; the interface is a different question and
    // averaging them gives a number that answers neither. Measured on a
    // two-chain fold: 0.712 within, 0.370 across, 0.630 mixed.
    chainCertainty: meanByChain(result.features.asymId, certainty),
    chainInterfaceCertainty: meanByChain(result.features.asymId,
                                         result.interfaceCertainty),
    model: modelName,
    // 🔴 THE TOKEN LAYOUT, because a ligand is one token per heavy atom and the
    // archive cannot infer that from the chain lengths - it refuses to guess
    // and throws. See tokenIdentifiers.
    tokens: tokenLayoutFrom(result.features.asymId, result.features.residueIndex),
    // ...and the entities and the templates, which the archive's job request is
    // made of. Without these "Download all" wrote a request naming no
    // sequences.
    ...foldContext,
    // 🔴 THIS MODEL'S OWN SETTINGS, OVER THE SHARED DIALS'. `foldContext` is
    // built before the branch and carries the recycle count and the MSA depth
    // that AF2 and AF3 read - and this model reads NEITHER: it folds from the
    // sequence alone, which is why the page hides its MSA row, and its trunk
    // loops a number of times the checkpoint fixes rather than the dial. The
    // archive said `recycles: 1` and `max msa: 128` for a fold that used one
    // value of neither.
    settings: {
      seed: foldContext.settings?.seed,
      "trunk passes": recycleCount() + 1,
      "language model": plmLabel(),
      sampler: samplerPreset(),
      "diffusion steps": result.steps,
    },
    // ...and no alignment or template line, rather than "none", which reads as
    // a choice. This model takes neither: `grep -rn template` over ESMFold2's
    // whole upstream package returns nothing, and `z_init` has five terms with
    // none of them one.
    msaOrigin: undefined,
    msas: {},
    templates: undefined,
  }, family);
  void rememberSessionWhenSettled(lastPrediction);

  esmfold2Trunk = result.reusable === undefined ? esmfold2Trunk
    : { key: trunkKey, reusable: result.reusable };
  const seconds = ((performance.now() - started) / 1000).toFixed(1);
  const mean = certainty === undefined ? undefined
    : [...certainty].reduce((total, value) => total + value, 0) / certainty.length;
  // 🔴 THE WORD "certainty" IS THE DISCLAIMER NOW. "(not pLDDT)" was here to
  // stop a number under a pLDDT palette being read as one - but the line never
  // says pLDDT, the model row's tooltip says the model reports no confidence,
  // the PDB carries a REMARK naming the quantity, and the archive's README
  // spells it out. A parenthesis denying something nothing claimed reads as a
  // disclaimer rather than a result.
  // 🔴 RESIDUES, NOT TOKENS. They are the same number until a modification
  // atomises one - and then this line read "22 res" for a thirteen-residue
  // chain, which is the model's own bookkeeping leaking onto the status bar.
  // The AF3 path names the modification here too, so this one does.
  const residueCount = chains.reduce((total, chain) => total + chain.length, 0);
  const named = modifications.map((one) => `${one.code}${one.position}`);
  status(`${modelName} · ${residueCount} res`
    + (named.length === 0 ? "" : ` + ${named.join(", ")}`)
    + ` · ${result.steps} steps · ${seconds}s`
    + (result.trunkReused ? " (trunk reused)" : "")
    + (mean === undefined ? "" : ` · certainty ${mean.toFixed(2)}`));
  progress(null);
}

/**
 * WHERE THIS FOLD HAPPENS, AND HOW TO ASK.
 *
 * 🔴 THE PAGE IS SERVED BY THE THING THAT FOLDS, so this is SAME-ORIGIN and
 * there is no CORS question at all: `notebooks/localfold.ipynb` runs
 * tools/colab_backend.py on the runtime, that server serves this checkout,
 * and the link that cell prints opens index.html on it. `?backend=colab` is
 * the cell saying which of the two machines should do the work, and `t` is
 * the token that server requires of every request - it is in the URL because
 * a page cannot be handed a header by whoever framed it.
 *
 * THE SAME SERVER RUNS A SECOND COPY OF THIS PAGE HEADLESSLY, at
 * `?role=runtime`, and that is the one that folds. The two talk through
 * web/colab-bridge.js: this page posts a command, that page pushes what it
 * says and draws. Neither knows anything about the other's machine.
 *
 * Absent the parameter this returns null and nothing anywhere changes: the
 * website folds where it always did, in the reader's own browser.
 */
function remoteBackend() {
  return colabRole() === "reader" ? {} : null;
}

/**
 * The same job, folded on the runtime, ingested by the same door.
 *
 * 🔴 IT COMES BACK AS A FILE AND GOES IN THROUGH `loadIntoViewer`, which is
 * the path a dropped PDB already takes - so the viewer, the sequence strip,
 * the heatmap panel and the downloads all behave as they do for a local fold
 * without knowing one machine from another. What is NOT here yet is the
 * trajectory: the backend returns the finished structure, so the play bar has
 * one frame rather than the sampler's walk, and the scores card stays empty
 * until the service returns its confidence JSON too.
 */
async function foldOnBackend({ chains, chainKinds, ligandCodes, modifications,
                               templates, family, signal }) {
  const entities = entityList.read();
  const request = {
    entities, model: family,
    steps: Number(element("af3-count")?.value ?? 25),
    recycles: Number(element("recycles")?.value ?? 3),
    // 🔴 THE CONTROL'S OWN VALUE, NOT THE RESOLVED MODE. `msaMode()` maps
    // "none" to "single" for the code below it; the runtime sets its page's
    // `msa-mode` SELECT from this, and a select silently refuses a value it
    // has no option for - so "single" left the control empty there and the
    // fold died with "unknown alignment mode". Sending the raw value lets the
    // runtime's page resolve it with the same function this one uses.
    msa: element("msa-mode")?.value ?? "none",
  };
  const label = MODEL_LABELS[family] ?? family;
  status(`${label} · folding on the runtime…`);
  progress("waiting");
  // 🔴 THE WATERMARK IS TAKEN BEFORE THE COMMAND IS SENT. The broker keeps
  // every event of the session, so a reader that started at zero would replay
  // the last fold's status writes and frames as this one's - and one that
  // asked after sending could miss the first of this fold's. `n` is where the
  // stream stands at the instant before the runtime is told anything.
  let since = (await remoteHead(signal)).n ?? 0;
  const { error: refused } = await remoteCommand("fold", request);
  if (refused) throw new Error(refused);

  // 🔴 AND STOPPING HAS TO REACH THE OTHER MACHINE. The abort signal ends this
  // loop, which on a local fold is the whole of stopping - here it would leave
  // the runtime folding, its GPU held, and the next fold refused.
  const stopThere = () => { remoteCommand("stop", null).catch(() => {}); };
  signal.addEventListener("abort", stopThere, { once: true });

  await followRemoteFold({ since, label, signal });
}

/**
 * WATCH A FOLD ON THE RUNTIME AND INGEST WHAT IT PRODUCES.
 *
 * 🔴 SEPARATE FROM ASKING FOR ONE, BECAUSE A READER CAN ARRIVE MID-FOLD. A
 * Colab fold is minutes long and the page in front of it is an ordinary tab:
 * reloaded, reopened from the notebook's link, opened in a second window. Only
 * the page that pressed Fold used to be following, so any of those left a
 * reader looking at an idle page while their own fold ran on - and the result,
 * when it came, landed in a page nobody was watching. `attachToRunningFold`
 * below is the other caller, and there is one loop between them.
 */
async function followRemoteFold({ since, label, signal }) {
  const stem = uniqueStem(safeJobName(entityList.header() ?? "fold"));
  const draw = remoteFrameDrawer(stem);
  const framePdbs = [];
  let result;
  for (;;) {
    throwIfAborted(signal);
    const state = await remoteEvents(since, signal);
    // 🔴 THE BROKER DROPS ITS OLDEST EVENTS, AND SAYING NOTHING ABOUT IT IS
    // THE ONE THING IT MUST NOT DO. A session is not one fold, so the mailbox
    // is capped; `from` is where the stream now starts, and a reader that has
    // fallen behind that point has lost what it never applied - frames, most
    // likely, since they are the bulk of it. The fold is not lost with them
    // (the result carries the finished structure), so this does not take the
    // status line away from the runtime's own words: it is recorded, and
    // `window.__remoteGap` is what a report can be built on.
    if ((state.from ?? 0) > since) {
      const lost = state.from - since;
      window.__remoteGap = (window.__remoteGap ?? 0) + lost;
      console.warn(`colab bridge: ${lost} event(s) were dropped before this`
        + " page could apply them - the trajectory will have a gap");
    }
    since = state.n ?? since;
    // 🔴 IN THE RUNTIME PAGE'S ORDER, NOT THE NETWORK'S. Events are pushed as
    // they happen and several sends can be in flight at once - which is what
    // keeps the feed live while that page is busy - so `seq` is the page's own
    // count and this is where it is put back in order.
    const batch = [...(state.events ?? [])]
      .sort((a, b) => (a.seq ?? 0) - (b.seq ?? 0));
    for (const said of batch) {
      // The page's own calls, replayed here: the same status writes, the same
      // bar fractions, the same sampler frames, in the order they happened.
      if (said.kind === "status") status(said.payload);
      else if (said.kind === "progress") progress(said.payload);
      else if (said.kind === "frame") { framePdbs.push(said.payload); draw(said.payload); }
      // ...and the runtime's own timing rows, recorded on its card against its
      // clock, rather than a reconstruction of them from over here.
      else if (said.kind === "dev") devAdopt(said.payload);
      // 🔴 AND THE LAG IS RECORDED RATHER THAN ARGUED ABOUT. Each event
      // carries the runtime page's own clock and the broker's arrival stamp,
      // so "the fold was slow" and "the feed was slow" are two numbers. It is
      // what the pulled version could not tell apart.
      else if (said.kind === "result") { result = said.payload ?? {}; }
      if (said.at !== undefined && said.got !== undefined) {
        (window.__remoteLag = window.__remoteLag ?? []).push(said.got - said.at);
      }
    }
    if (result !== undefined) break;
    // 🔴 AND A RUNTIME THAT HAS GONE MUST NOT BE POLLED FOR EVER. Colab
    // recycles a runtime when the notebook is closed or left idle, and the
    // broker's busy flag is raised HERE and lowered by the runtime page - so a
    // page that has gone takes the flag with it and this loop would wait out
    // the rest of the session on a fold nobody is doing. `runtimeSeen` is how
    // long it has been since that page asked for its commands, which it does
    // three times a second.
    // 🔴 A BUSY PAGE IS NOT A DEAD RUNTIME, AND THE FIRST VERSION OF THIS
    // COULD NOT TELL THEM APART. The heartbeat is the runtime page's own
    // command poll, which STOPS while that page holds its main thread -
    // measured at 6.2 s from six seconds of deliberate long tasks, and a real
    // fold (shader compilation, a big upload) can hold it longer. Giving up
    // on silence alone would abort a fold that was working. `browserAlive` is
    // the DevTools endpoint answering, which is the browser PROCESS rather
    // than the page: gone means gone.
    const quiet = state.runtimeSeen ?? 0;
    if (quiet > 20000 && state.browserAlive === false) {
      throw new Error("the runtime stopped answering - its notebook may have"
        + " been closed or its runtime recycled; run the Colab cell again");
    }
    // ...and a page that is alive but silent for five minutes is a fold that
    // has hung rather than one that is thinking. Long, because the cost of
    // being wrong here is abandoning a fold somebody waited for.
    if (quiet > 300000) {
      throw new Error("the runtime's page has not spoken for five minutes -"
        + " its fold may have hung; run the Colab cell again");
    }
    await new Promise((done) => setTimeout(done, 300));
  }
  if (result.error) throw new Error(`${result.error}${result.status ? ` · ${result.status}` : ""}`);

  // 🔴 THE FILE STILL GOES IN THROUGH `loadIntoViewer`, because that is what
  // fills the sequence strip, the download buttons and the scores card - the
  // streamed frames are a picture and not an ingestion. It CLEARS the object's
  // frames, so the trajectory is put back afterwards, which is exactly the
  // dance the local sampler path does a few hundred lines above.
  const registry = window.py2dmol_viewers ?? {};
  const liveRenderer = registry[Object.keys(registry)[0]]?.renderer;
  const camera = { ...(liveRenderer?.viewerState ?? {}) };
  const live = liveRenderer?.objectsData?.[liveRenderer?.currentObjectName];
  if (live?.frames !== undefined) live.frames.length = 0;
  // 🔴 EVERYTHING THE LOCAL PATH INGESTS, NOT JUST THE STRUCTURE. This used to
  // pass `{pdb, scores: {}}`, so a remote fold came back with no MSA panel, no
  // PAE plot and an empty scores card - the page looked like it had folded
  // nothing but coordinates, because it had been given nothing else.
  await loadIntoViewer({
    stem, pdb: framePdbs[0] ?? result.pdb,
    scores: result.scores ?? {},
    a3m: result.a3m,
    confidence: result.confidence,
    length: result.length,
  });
  if (viewer !== undefined && Object.keys(camera).length > 0) {
    Object.assign(viewer.viewerState, camera);
    viewer.render?.("localfold.restore-camera");
  }
  const api = window.py2Dmol;
  if (api?.frameFromText !== undefined && viewer !== undefined && framePdbs.length > 0) {
    const first = viewer.objectsData?.[viewerObject]?.frames?.[0];
    if (first !== undefined) first.name = first.label = first.title = "sampler_0";
    for (const [index, text] of [...framePdbs.slice(1, -1), result.pdb].entries()) {
      try {
        const frame = api.frameFromText(text);
        const last = index === framePdbs.length - 2;
        frame.name = frame.label = frame.title = last ? "final" : `sampler_${index + 1}`;
        viewer.addFrame(frame, viewerObject);
      } catch (cause) { console.warn("frame skipped:", cause); }
    }
  }
  // 🔴 AND THE DOWNLOAD BUTTONS NEED A PREDICTION, WHICH THIS PAGE NEVER MADE.
  // `activePrediction()` reads `predictions.get(name)` and falls back to
  // `lastPrediction` - both written by the LOCAL fold paths - so on a remote
  // fold the buttons were either a silent no-op or, worse, handed back
  // whatever this tab had folded BEFORE: the wrong structure, downloaded
  // without a word. The runtime sends its whole prediction object and it is
  // registered here under THIS page's stem, which is the name the viewer knows
  // the object by and therefore the one `activePrediction` looks up.
  if (result.predJson) {
    try {
      const remote = revivePrediction(result.predJson);
      remote.stem = stem;
      // ...through the one funnel, so this path records what every other
      // one does: the last prediction, the map entry, the downloads, and
      // the page's claim on the object (see recordPrediction).
      recordPrediction(remote, remote.family ?? familyFromLabel(remote.model));
    } catch (cause) {
      console.warn("the runtime's prediction did not parse:", cause);
    }
  }
  // ...and the runtime's own summary, which already reads the way this page's
  // status line does - it is the same code, on the other machine.
  status(result.status || `${label} · folded on the runtime`);
  progress(null);
}

/**
 * A FOLD THAT WAS ALREADY RUNNING WHEN THIS PAGE OPENED.
 *
 * 🔴 THE BROKER KNOWS, AND IT IS ONE QUESTION AT LOAD. `head=1` carries
 * `folding` - raised when a fold command is accepted, lowered by the runtime
 * page's own result - so a page that arrives in the middle of one attaches to
 * it rather than sitting idle while it finishes somewhere else. The watermark
 * is the CURRENT head, not zero: what is wanted is the rest of this fold, not
 * a replay of everything the session has said.
 *
 * It is deliberately not an abortable job: the reader who opened this page did
 * not start this fold, so Stop is not theirs to press, and pressing Fold while
 * one runs is refused by the broker with its own words.
 */
async function attachToRunningFold() {
  if (colabRole() !== "reader") return;
  try {
    const head = await remoteHead();
    if (!head.folding) return;
    status("a fold is already running on the runtime - following it");
    progress("waiting");
    await followRemoteFold({
      since: head.n ?? 0,
      label: "the runtime",
      signal: new AbortController().signal,
    });
  } catch (cause) {
    console.warn("could not attach to the running fold:", cause.message);
  }
}

/**
 * The live frames of a fold happening somewhere else.
 *
 * 🔴 IT OPENS THE OBJECT ON THE FIRST FRAME, NOT BEFORE. A blank fold opened
 * when the request is sent would sit empty for however long the runtime spends
 * on the trunk - and a fold the runtime REFUSES (429, a second reader) would
 * leave an empty object on the page with nothing ever arriving in it.
 */
function remoteFrameDrawer(stem) {
  let drawn = 0;
  let opened = false;
  let colouring = false;
  return (pdb) => {
    const api = window.py2Dmol;
    if (api?.frameFromText === undefined) return;
    const registry = window.py2dmol_viewers ?? {};
    const renderer = registry[Object.keys(registry)[0]]?.renderer;
    if (renderer === undefined) return;
    if (!opened) { openBlankFold(stem); opened = true; }
    const object = renderer.objectsData?.[renderer.currentObjectName];
    if (object === undefined) return;
    try {
      if (object.frames.length === 0) { revealViewer(renderer); contactsBig(false); }
      const frame = api.frameFromText(pdb);
      frame.name = frame.label = frame.title = `sampler_${drawn++}`;
      renderer.addFrame(frame, renderer.currentObjectName);
      renderer.setFrame(object.frames.length - 1);
      // The B-factor column of every model this backend drives is a pLDDT, so
      // the trajectory is watchable in confidence from its first frame - the
      // same choice the local sampler path makes, and for the same reason.
      if (!colouring) { colouring = setColourMode("plddt"); orientBestView(renderer); }
    } catch (cause) {
      console.warn("live frame skipped:", cause);
    }
  };
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
  // 🔴 THE PAGE SAYS WHETHER IT IS FOLDING, because everything else is a
  // proxy. A backend driving this page headlessly used to watch for a NEW
  // object with frames, which is true of a first fold and FALSE of a second:
  // a repeat fold reuses the stem, so the watcher waited out its whole timeout
  // on a fold that had finished in a second (measured - status line
  // "AlphaFold 3 · 13 residues · in 1 s (trunk reused)", watcher timed out).
  // The button is no good either: it stays enabled throughout, being how you
  // stop one. See tools/colab_backend.py.
  window.__foldState = { running: true, since: Date.now() };
  setFoldButton("running");
  // ...`msaMode()` and not the select, so the dev log records what the fold
  // will actually do rather than what a hidden control still says.
  devBeginRun(`fold · ${element("model-family").value}`
    + ` · alignment ${msaMode()}`
    + ` · ${element("recycles").value} recycles`);
  // 🔴 THE LAST FOLD'S NUMBERS GO BEFORE THIS ONE STARTS. The card kept showing
  // a mean pLDDT and a pTM for a structure that was no longer being computed,
  // for as long as the new fold took - which is worse than an empty panel,
  // because a stale number reads as an answer. The structure itself stays: it
  // is still the last thing that WAS predicted, and the page is never blank
  // between folds.
  updateScoresCard(undefined);
  // ...and the rest of the last result with it - see hideResults.
  hideResults();
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
      : { chains: [], chainKinds: [], chainMsa: [], ligandCodes: [],
          modifications: [], templates: [] };
    let chains = request.chains;
    let chainKinds = request.chainKinds ?? chains.map(() => "protein");
    // ...and which chains asked for an alignment, in chain order. Absent
    // means all of them, which is every job written before the row existed.
    const chainMsa = request.chainMsa ?? chains.map(() => true);
    const ligandCodes = request.ligandCodes;
    const modifications = request.modifications ?? [];
    // 🔴 THE BONDS COME OFF THE ROWS, NOT OUT OF A VARIABLE. They were held in
    // `jobBonds`, set when a job loaded and invisible thereafter - the same
    // shape as the job NAME that was built and then removed for being state
    // nobody could see. A `contact` row sits in the entity list with the chains
    // it names, so editing the rows edits the bonds, and `expandEntities`
    // resolves the letters against the chain order the fold will actually see.
    const bonds = request.bonds ?? [];
    // 🔴 DECIDED BEFORE ANY NETWORK WORK, because the download starts here.
    // Nothing below changes it: the only reassignment of `chainKinds` is the
    // pasted-A3M branch, which runs only where `nucleicCount` is already zero
    // and sets it to the protein it already was.
    const nucleicCount = chainKinds.filter((kind) => kind !== "protein").length;
    let family = modelFamily(ligandCodes.length, modifications.length, nucleicCount,
                             (request.templates ?? []).length);
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
    // 🔴 A FOLD THAT HAPPENS SOMEWHERE ELSE LEAVES HERE, BEFORE THE WEIGHTS.
    // In a Colab cell the page is served BY the runtime that folds, so this
    // machine has no reason to download half a gigabyte of parameters to
    // watch. Everything above still runs - the entities are validated, the
    // model is resolved, the terms are asked - because those are questions
    // about the JOB, and the job is the same wherever it runs.
    if (remoteBackend() !== null) {
      await foldOnBackend({ chains, chainKinds, ligandCodes, modifications,
                            templates: request.templates ?? [], family, signal });
      return;
    }
    // ...and started, not awaited. The templates and the alignment below are
    // network work of their own; this runs beside them.
    const modelLoad = startModelPreload(family, signal);
    // 🔴 AND THE SHADERS, WHILE THE SHARDS ARE STILL ARRIVING. The tools have
    // done this since the warm existed - fold-opendde.js measures 2533 ms
    // against 2839 with `--no-warm` - and the page never did, because
    // `loadAf3Weights` resolves only once every tensor is decoded and nothing
    // could reach the store before that. It can now; see warmAf3Pipelines.
    //
    // 🔴 THE RESIDUE COUNT, NOT THE TOKEN COUNT, and that is a deliberate
    // approximation: the tokens are known only once the batch is featurised,
    // which happens inside foldAf3 after the weights are awaited - by which
    // time there is nothing left to hide behind. A ligand or a modified residue
    // makes the real count larger, so the warm compiles a subset and the fold
    // compiles the rest, which is slower than a perfect warm and faster than
    // none. A speculative warm cannot give a wrong ANSWER, only waste.
    if (isAf3Family(family)) {
      void getDevice().then((device) =>
        warmAf3Pipelines(family, chains.join("").length, device)).catch(() => {});
    }
    // 🔴 FETCHED HERE AND NOT INSIDE THE FOLD, so a structure that cannot be
    // reached stops the run with its own message rather than surfacing as a
    // fold that scored badly. Fetched for AF3 AND for AlphaFold 2's monomer -
    // this used to say "AF3 only: AF2's drivers take a template through a
    // different path and nothing on this page builds one for them yet", and
    // the reason was that monomer.js never forwarded the slot. It does now.
    // The MULTIMER is still refused upstream, in chosenFamily's guard.
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
    // 🔴 AND WHICH OF THEM ASKED FOR AN ALIGNMENT, in the same order, because
    // that is the only thing that says which block belongs to which chain.
    // Off `chainMsa`, which expandEntities builds in chain order beside
    // `chainKinds` - a protein with its MSA set to none is folded with its
    // query row alone while the rest of the complex keeps theirs.
    const proteinWantsMsa = chainMsa
      .filter((_, index) => chainKinds[index] === "protein");
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
      ? null : await alignmentText(proteinChains, signal, family, proteinWantsMsa);
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
    const searchHits = typeof alignmentResult === "string"
      ? undefined : alignmentResult?.templateHits;
    // 🔴 THE HITS COUNT PROTEIN CHAINS; A TEMPLATE NAMES A FOLD CHAIN. The
    // search is handed `proteinChains` and its hits come back numbered in
    // THAT list, while `template.chain` is the index `expandEntities` gave
    // it - which counts every polymer, DNA and RNA among them. The two agree
    // only while every polymer is a protein, so a job with a nucleic chain
    // BEFORE a protein one read its automatic template off the wrong chain,
    // or off none. Both numberings exist here and nowhere else, so this is
    // where they meet.
    const proteinAt = [];
    chainKinds.forEach((kind, index) => {
      if (kind === "protein") proteinAt.push(index);
    });
    const hits = searchHits === undefined ? undefined : new Map(
      [...searchHits].map(([at, found]) => [proteinAt[at] ?? at, found]));
    for (const template of templateSources) {
      if (template.auto !== true) continue;
      const best = (hits?.get(template.chain) ?? [])[0];
      if (best === undefined) {
        // 🔴 AND TWO SETTINGS FROM THE SAME PANEL CAN CONTRADICT EACH
        // OTHER. "A template from the MSA search" and "no alignment for
        // this chain" are both set on the ⋮ menu, and since the search now
        // covers only the chains that asked for one, the first cannot be
        // answered for a chain that took the second. Saying "the search
        // found no template" there would be false - it was never searched
        // for - and the reader would go looking for a homolog that exists.
        throw new Error(hits === undefined
          ? "Automatic templates need an MSA search: set the MSA to search, or"
            + " name a structure instead."
          : chainMsa[template.chain] === false
            ? `Chain ${template.chain + 1} asks for a template from the MSA`
              + " search and for no alignment, and it is not searched for at"
              + " all - so there are no hits to take one from. Give it an"
              + " alignment, or name a structure."
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
      bonds,
      // 🔴 THE FORM AS IT IS RIGHT NOW, which is what makes a restored fold
      // continuable: it rides into every prediction through the spread of
      // this object and into the session through jobMeta. Read HERE, at the
      // start, rather than when the session is written - a fold takes a
      // minute and the boxes can be edited while it runs, and what the
      // session should put back is what produced this answer.
      inputs: formInputs(),
      // 🔴 NOT RECORDED HERE AT ALL ANY MORE. The request's `name` is the
      // fold's `stem`, which `foldStem` has already resolved from this same
      // box - so `archiveFor` reads `pred.stem` and the two cannot disagree.
      // Carrying the box separately is what let them.
      templates: templateSources,
      msas: archiveMsas(chains, alignment),
      msaOrigin: {
        single: SINGLE_SEQUENCE_ORIGIN,
        search: "MMseqs2 search at api.colabfold.com",
        paste: "pasted by hand",
        upload: uploadedMsas === undefined ? "uploaded a3m" : "uploaded archive",
      }[msaMode()] ?? msaMode(),
      settings: {
        seed: randomSeed(),
        recycles: recycleCount(),
        // ...the resolved number, not the word: an archive saying "reference"
        // would not say what ran, and the reference's own answer depends on
        // which model it was.
        "early stop": recycleTolerance(),
        "max msa": maxMsaConfig().requested,
      },
    };

    // 🔴 AlphaFold 3 IS A DIFFERENT MODEL BELOW THIS LINE, so it branches here -
    // before AF2's weights are chosen and before anything below assumes a
    // recycle loop over an evoformer. It branches AFTER the alignment, though,
    // and that is the point: search, paste and upload, the query-wins rule and
    // the pairing decision are one implementation for all three models. What
    // differs is only how the A3M is encoded, which is af3MsaFromA3m's job.
    if (SINGLE_SEQUENCE_FAMILIES.includes(family)) {
      // 🔴 MODIFICATIONS TRAVEL HERE TOO, AND USED NOT TO. `modelFamily`
      // refuses one for AlphaFold 2 by name - it would "return a confident
      // structure of the unmodified chain" - and ACCEPTS one for EF2-fast,
      // which then folded exactly that: reported as the modified residue not
      // being displayed, and measured, a SEP@3 job came back GLY,TRP,SER with
      // no SEP anywhere in it.
      await foldWithEsmfold2(chains, chainKinds, ligandCodes, signal, modelLoad,
                             modifications);
      return;
    }
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
    // 🔴 THE SWEEP IS ONE MODEL UNLESS THE ROW SAYS "All 5". AlphaFold 2 is
    // five models, one training run continued five ways, and AlphaFold's own
    // pipeline runs all of them and RANKS the results - which is why every pass
    // of every model here lands on ONE object and one ranking decides what is
    // saved. `model` is reassigned per model, so it is no longer const, and the
    // first one is the download that started before the alignment.
    const sweep = af2Sweep(family);
    let model = await modelLoad;
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
    // 🔴 THE GRAPH, NOT THE FAMILY. `multimer-2` is a delta on model_1 and runs
    // model_1's graph; asking whether its NAME is "multimer" sent it through
    // the monomer driver, which has no multimer template embedder and died in
    // QueryOnlyTemplateGpu with no weights. See graphFamily.
    const multimer = graphOf(family) === "multimer";
    const unified = multimer || new URLSearchParams(location.search).get("graph") === "unified";
    const alignmentForDriver = alignment === null ? `>query\n${sequence}\n` : alignmentForModel;

    // 🔴 THE MONOMER'S TEMPLATE, BUILT BEFORE af2Key, because a trunk cached
    // from a fold WITHOUT one is not this fold's trunk - which is what that
    // comment below means by "everything a pass reads".
    //
    // 🔴 AND IT IS atom37, NOT AF3's DENSE 24. `buildTemplate` already does
    // everything else this needs and the AF3 path uses it unchanged: it sniffs
    // PDB against mmCIF, ALIGNS a homolog to the query (a search hit is not the
    // query's own sequence, so the CLI tool's identity map would be wrong here)
    // and drops low-confidence residues. The layouts differ only in which slot
    // builder it ends on, they are the same rank, and NEITHER THROWS ON THE
    // OTHER - so `layout` is the whole of the difference and all of the risk.
    // See test/template-atom37-layout.test.js.
    let af2Template;
    if (templateSources.length > 0) {
      // 🔴 REFUSED RATHER THAN DROPPED. `?graph=unified` runs the MULTIMER's
      // graph over a monomer, and that embedder is a different dialect nothing
      // here builds a slot for - so folding on would quietly ignore it, which
      // is the failure this whole path exists to avoid.
      if (unified) {
        // ...and it names which of the two it is, because a session restored
        // from a job that set both reaches here without passing chosenFamily's
        // guard, and "drop ?graph=unified" is not advice a multimer can take.
        throw new Error(multimer
          ? "AlphaFold 2 multimer's template embedder is a different dialect"
            + " and this page does not build a slot for it"
          : "the unified graph has no monomer template embedder;"
            + " drop ?graph=unified to fold with a template");
      }
      const source = templateSources[0];
      status(`Aligning template ${source.source ?? ""}`);
      af2Template = buildTemplate({
        text: source.text, chain: source.chainId, query: sequence,
        tokens: sequence.length, minConfidence: source.minConfidence ?? 0,
        layout: "atom37",
      });
      throwIfAborted(signal);
    }

    // 🔴 THE KEY IS EVERYTHING A PASS READS, for the reason the AF3 one gives:
    // a stale state is not a slow fold but a structure for another sequence.
    // Recycles are absent because more of them is a continuation; the tolerance
    // is present because it decides when the passes STOP.
    const af2Key = JSON.stringify({
      sequence, chainLengths, maxMsaSequences, maxExtraSequences, seed, tolerance,
      unified, family, alignment: cheapHash(alignmentForDriver),
      // ...the SOURCE rather than the slot: the slot is megabytes of float and
      // the text plus the chain is what decides every one of them.
      template: af2Template === undefined ? null
        : cheapHash(`${templateSources[0].text}\u0000${templateSources[0].chainId ?? ""}`),
    });
    // 🔴 AND A SWEEP IS NEVER A CONTINUATION. The cache holds ONE model's trunk
    // under a key naming that model, so resuming a five-model run would replay
    // model_1's passes and then fold four models on top of them, in an object
    // rewound to a different fold's frames. A sweep starts from pass zero and
    // leaves no resumable state behind; see the clear after the loop.
    const af2Cached = sweep.length === 1 && af2Cache?.key === af2Key ? af2Cache : undefined;
    const resume = af2Cached !== undefined && af2Cached.resumable.recycles < recycles
      ? af2Cached.resumable : undefined;

    predictionCount += 1;
    // ...and a sweep says so in the file name, because the five models are one
    // prediction here and the archive is the only place that can say which.
    // 🔴 uniqueStem READS objectsData; the loop that used to be here read
    // `viewer.objects`, which does not exist on this build. It is inside
    // foldStem now, which every fold path shares.
    const stem = resume === undefined
      ? foldStem(`${MODEL_STEMS[family] ?? family}`
        + `${sweep.length > 1 ? "_all5" : ""}_${predictionCount}`)
      : af2Cache.stem;

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
    let firstPassLanded = undefined;
    let initialLoadPromise = undefined;
    if (resume !== undefined) firstPassLanded = af2Cached.firstPassLanded;
    // 🔴 EVERY MODEL'S PASSES IN ONE LIST, RANKED TOGETHER. AlphaFold's own
    // pipeline folds all five and ranks the outputs, and that is what "All 5"
    // is: the play bar is the whole sweep, and the structure this page saves is
    // the best PASS of the best MODEL rather than five separate answers the
    // reader has to compare by eye. With one model selected the list is one
    // model's passes and everything below is what it always was.
    const alignedRecycles = [];
    let final;
    // ...the next model's bytes, fetched while this one folds. `openStore`
    // caches by family and a delta shares its base with model_1, so the four
    // deltas are 44 MiB each on top of a base that is already here - and the
    // dial on the right names the model it is fetching, which is how a reader
    // can tell that the wait is the next download rather than this fold.
    let ahead = undefined;
    for (const [modelIndex, foldFamily] of sweep.entries()) {
      if (modelIndex > 0) {
        model = await (ahead ?? startModelPreload(foldFamily, signal));
        throwIfAborted(signal);
      }
      ahead = sweep[modelIndex + 1] === undefined
        ? undefined : startModelPreload(sweep[modelIndex + 1], signal);

      status(`Folding ${sequence.length} residues${chains.length === 1 ? "" : ` in ${chains.length} chains`}`
        + ` · ${passes} pass${passes === 1 ? "" : "es"} · ${foldFamily}`
        + (sweep.length === 1 ? "" : ` · model ${modelIndex + 1} of ${sweep.length}`));

      // ...DRAWN AS EACH PASS LANDS, not collected and drawn at the end. The
      // first builds the object and the panels; the rest are frames on it.
      //
      // 🔴 AND `base` IS WHAT MAKES FIVE MODELS ONE SET OF FRAMES. The driver
      // counts its own passes from zero, so every model after the first would
      // rebuild the object and overwrite frame 0; offsetting by what is already
      // on it turns 5 x (recycles + 1) passes into one play bar. It is also what
      // decides the superposition: only the very first pass of the whole run
      // becomes `firstPassLanded`, and all the rest are aligned onto it, so the
      // animation does not jump between models.
      const base = alignedRecycles.length;
      // ...captured, because `model` advances to the next one while this model's
      // contact map is still being attached in a `.then`.
      const weights = model.weights;
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
        if (base + index === 0) {
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
            frameName: sweep.length === 1 ? undefined : "model1_recycle_0",
          });
          // ...recycle 0's frame is built by loadIntoViewer rather than by
          // appendPass, so its contact map has to be attached here or the first
          // pass is the one frame without one - and it is the frame on screen
          // while every later pass is still running.
          void initialLoadPromise.then(() => {
            const frame = viewer?.objectsData?.[viewerObject]?.frames?.[0];
            if (frame !== undefined) {
              attachContactMap(frame, recycle, weights, sequence.length);
            }
          });
        } else {
          appendPass(sequence, chainLengths, recycle, base + index, firstPassLanded, weights,
                     sweep.length === 1 ? undefined
                     : `model${foldFamily.split("-")[1] ?? "1"}_recycle_${index}`);
        }
      };
      // 🔴 THE UNITS ARE COSTS, NOT COUNTS, and that is what makes a clock
      // possible. src/af2/model/*.js weight every step by what the cost model says it
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
        // 🔴 THE BAR IS THE RUN'S, NOT THE MODEL'S. Five models each driving one
        // bar from 0 to 100 is five bars, and a reader cannot tell the fourth
        // from the first; the model's own fraction is a fifth of the sweep,
        // offset by the models already done.
        const done = (modelIndex + runEstimator.fraction()) / sweep.length;
        progress(done);
        // See the note on `say` in web/af3-model.js: the percentage, and nothing
        // beside it that moves on its own.
        const percent = Math.min(100, Math.round(100 * done));
        status(`Folding · ${percent}%`
          + (sweep.length === 1 ? "" : ` · model ${modelIndex + 1} of ${sweep.length}`));
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
      // ...?graph=unified runs the MONOMER weights through src/af2/multimer/ instead.
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
      const prediction = await new (unified ? AlphaFoldUnifiedGpu : AlphaFoldMonomerGpu)(device)
        .predictA3m(
          alignmentForDriver, weights, model.featureTables,
          // 🔴 `resumable: true` IS WHAT ASKS FOR THE CONTINUATION STATE, and this
          // is the only caller that wants it. It is the trunk's MSA and pair
          // representation copied to the host - 781 MB at 825 residues, 1.26 s of
          // a 25.7 s fold - and it exists for `af2Cache`, so that raising the
          // recycle count continues rather than restarts. Every other caller
          // (the CLI tools, the differential gates, an embedder) folds once and
          // used to pay for it anyway.
          { recycles, randomSeed: seed, maxMsaSequences, maxExtraSequences, chainLengths, tolerance, signal,
          // ...and `pairHost: true` for the distogram contact overlay, which is
          // the only reader of the host copy of the pair representation.
          // ...and the template slot, which monomer.js forwards into
          // QueryOnlyTemplateGpu. `undefined` is a fully masked template, which
          // is what every fold on this page was before it.
            resume, resumable: true, pairHost: true, template: af2Template?.slot, ...regime },
          model.paeBreaks, onRecycle, runProgress);

      // 🔴 THE EARLIER PASSES COME BACK FOR THE ANIMATION. A continuation returns
      // only the passes it ran, and the play bar is the whole trajectory - so the
      // cached ones are put back in front of them. `final` is still the last pass
      // actually computed, which is the one the page lands on.
      const allRecycles = resume === undefined
        ? prediction.recycles : [...af2Cached.recycles, ...prediction.recycles];
      // ...and only a single-model run leaves a continuation behind; see the
      // note on `af2Cached` above.
      if (sweep.length === 1) {
        af2Cache = { key: af2Key, resumable: prediction.resumable, recycles: allRecycles,
          firstPassLanded, stem };
      }
      for (const [i, r] of allRecycles.entries()) {
        const at = base + i;
        alignedRecycles.push({
          structure: at === 0 ? (firstPassLanded ?? r.structure)
            : alignedToFirstPass(sequence, r.structure, firstPassLanded),
          confidence: r.confidence,
          recycleDistance: r.recycleDistance,
          // 🔴 THE DRIVER'S OWN PASS, KEPT BY REFERENCE. Its contact map is attached
          // in a setTimeout long after this map runs, so copying the field here
          // copies `undefined`; holding the object means whatever lands on it later
          // is visible to anything that reads this afterwards.
          pass: r,
          // ...and WHICH MODEL made it, because the best pass of a sweep is one
          // model's and the file that gets saved has to say whose.
          family: foldFamily,
        });
      }
      final = prediction.final;
      // 🔴 AND THIS MODEL'S WEIGHTS GO, OR FIVE OF THEM ARE HELD AT ONCE.
      // Measured in the page before this line existed: a 68-residue sweep at
      // one recycle took the JS heap from 9 MiB to 3412, against Chrome's own
      // ~4 GB ceiling - so the sweep that works on a 68-mer is the one that
      // kills the TAB on a real protein. The base is kept, because the four
      // deltas are differences on it; each delta is dropped once its passes are
      // in hand. A single-model fold keeps its weights exactly as before, which
      // is what makes switching model and back free.
      if (sweep.length > 1 && foldFamily !== graphOf(foldFamily)) releaseModel(foldFamily);
    }
    // 🔴 A SWEEP CANNOT BE CONTINUED, so it does not leave a key that looks like
    // one. `af2Cache` still holds whichever single-model fold ran before this,
    // whose stem names an object this run has replaced.
    if (sweep.length > 1) af2Cache = undefined;
    progress(null);

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
    //
    // 🔴 AND UNDER "All 5" THE SAME LINE RANKS ACROSS MODELS, which is the
    // whole point of running them: the criterion is the one AlphaFold's own
    // pipeline uses to order five models, applied here to every pass of every
    // model at once. A tie still keeps the later one, so it prefers a later
    // model only where nothing separates them.
    // ...and the graph again, because models 2 to 5 of the multimer rank the
    // way model_1 does. See graphFamily.
    const rankOf = (confidence) => (graphOf(family) === "multimer"
      ? (confidence?.multimerScore ?? confidence?.iptm ?? Number.NEGATIVE_INFINITY)
      : (confidence?.meanPlddt ?? Number.NEGATIVE_INFINITY));
    let bestIndex = alignedRecycles.length - 1;
    for (let index = alignedRecycles.length - 1; index >= 0; index -= 1) {
      if (rankOf(alignedRecycles[index].confidence)
        > rankOf(alignedRecycles[bestIndex].confidence)) bestIndex = index;
    }
    const best = alignedRecycles[bestIndex];
    // 🔴 AND UNDER A SWEEP, THE BEST PASS OF EVERY MODEL - not just of the run.
    // Five models folded and one structure saved is four thrown away: the
    // frames are all on the play bar, so the data exists, and an archive that
    // carries one of them cannot be compared against anything. ColabFold writes
    // one structure per model, rank-ordered, and that is the convention people
    // running all five expect. Ranked by the SAME criterion the overall best
    // uses, so `rank_001` is the file the page itself chose.
    const perModel = sweep.length === 1 ? undefined : sweep
      .map((foldFamily) => {
        let pick;
        for (const pass of alignedRecycles) {
          if (pass.family !== foldFamily) continue;
          if (pick === undefined || rankOf(pass.confidence) >= rankOf(pick.confidence)) pick = pass;
        }
        return pick === undefined ? undefined : { family: foldFamily, pass: pick };
      })
      .filter((entry) => entry !== undefined)
      .sort((a, b) => rankOf(b.pass.confidence) - rankOf(a.pass.confidence))
      .map((entry, rank) => ({
        rank: rank + 1,
        family: entry.family,
        // ...the number a reader picked in the row, which is what names the file.
        number: entry.family.split("-")[1] ?? "1",
        confidence: entry.pass.confidence,
        pdb: predictionToPdb(sequence, entry.pass.structure,
                             entry.pass.confidence.plddt, chainLengths),
        scores: confidenceJson(sequence, entry.pass.confidence),
      }));
    previousFold = {
      sequence,
      structure: finalLanded,
    };
    // ...and the family recorded is the WINNER's, not the row's: a sweep
    // folds five and the one on screen is whichever won, which is the same
    // reason the `model` field below says so.
    recordPrediction({
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
      // ...every model's own best, ranked, for the archive. Absent for a
      // single-model fold, which has nothing to rank.
      perModel,
      // ...the model that MADE the saved pass, which under a sweep is whichever
      // of the five won rather than the one the row resolved to.
      model: `AlphaFold 2 (${best.family ?? family})`,
      ...foldContext,
    }, best.family ?? family);
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
    syncDownloads();
    void rememberSessionWhenSettled(lastPrediction);
    // 🔴 THE CARD SCORES WHAT WILL BE SAVED, which is the best pass and not
    // always the last. Showing the last pass's numbers beside a download of the
    // best one is the kind of disagreement nobody reads a status line closely
    // enough to catch.
    updateScoresCard(best.confidence);
    const took = ((performance.now() - started) / 1000).toFixed(1);

    // 🔴 COUNTED OVER THE WHOLE RUN, because `allRecycles` is one MODEL's and a
    // sweep has five of them. Reading it here was a ReferenceError waiting for
    // the first fold that finished.
    const converged = alignedRecycles.length < passes * sweep.length
      ? ` · converged at ${final.recycleDistance.toFixed(2)} Å after`
        + ` ${alignedRecycles.length} pass${alignedRecycles.length === 1 ? "" : "es"}`
      : "";
    const bestIptmText = best.confidence.iptm !== undefined
      ? ` · ipTM ${Number(best.confidence.iptm).toFixed(3)}` : "";
    // ...and said out loud when the two differ, because the play bar is still
    // sitting on the last pass while the download is a different one.
    //
    // 🔴 AND A SWEEP ALWAYS SAYS IT, because "which of the five" is the answer
    // the reader asked for by pressing it. The pass is numbered within its own
    // model rather than across the run - pass 14 of 20 is not a number anyone
    // can act on, and a model that converged early makes the arithmetic wrong
    // as well as unreadable.
    const modelNumber = (name) => name?.split("-")[1] ?? "1";
    const firstOfWinner = alignedRecycles.findIndex((r) => r.family === best.family);
    const ranked = sweep.length > 1
      ? ` · best of ${sweep.length} models: model ${modelNumber(best.family)},`
        + ` pass ${bestIndex - firstOfWinner + 1}`
      : (bestIndex !== alignedRecycles.length - 1
        ? ` · saved pass ${bestIndex + 1} of ${alignedRecycles.length}` : "");
    // 🔴 AND SAY SO WHEN THE FOLD IS NOT A CHAIN, WHICH THIS PAGE NEVER DID.
    // Every command-line fold in this repository gates on `chainGeometryVerdict`
    // and the one path a visitor takes did not - the same shape as
    // LOCALFOLD_STOCK_FLAGS, where the configuration every gate checks was not
    // the one that ships. Measured: intellifold2 in Flow returns a fold this
    // rule REFUSES on 1 seed in 6 (CA median 4.255 A against 3.80) with pLDDT
    // 83.30, and the page drew it with a confident number beside it.
    //
    // It WARNS rather than refusing: the structure is still shown, because a
    // visitor who asked for a fast sampler is entitled to see what it made, and
    // hiding it would be worse than labelling it. What is not acceptable is
    // showing it as though the number were the whole story.
    //
    // 🔴 AND IT IS MEASURED HERE, FROM THIS FOLD'S OWN atom37. The first version
    // of this line read `result.geometry` - the AlphaFold 3 path's result, which
    // does not exist in this function - and never imported the verdict, so every
    // AlphaFold 2 fold on the page finished and then threw `ReferenceError`
    // where its "Done" line belonged. AF2 has no `backboneGeometry`; the spacing
    // below is tools/gpu/fold-af2.js's, which has gated every AF2 fold for
    // months: consecutive alpha carbons (atom 1 of 37), skipping the join
    // between two chains, where a complex's chains sit wherever they fold.
    const atom37 = best.structure?.atom37;
    const spacings = [];
    if (atom37 !== undefined) {
      const joins = new Set();
      let boundary = 0;
      for (const length of chainLengths.slice(0, -1)) { boundary += length; joins.add(boundary - 1); }
      for (let residue = 0; residue + 1 < sequence.length; residue += 1) {
        if (joins.has(residue)) continue;
        const a = (residue * 37 + 1) * 3;
        const b = ((residue + 1) * 37 + 1) * 3;
        spacings.push(Math.hypot(
          atom37[a] - atom37[b], atom37[a + 1] - atom37[b + 1], atom37[a + 2] - atom37[b + 2]));
      }
    }
    const chain = chainGeometryVerdict(chainGeometryOf(spacings),
      { plddt: best.confidence.meanPlddt });
    const broken = chain.ok ? "" : " · 🔴 NOT A CHAIN - the backbone is broken,"
      + " and pLDDT does not measure that";
    // 🔴 THE COVERAGE GOES BACK ON THE ROW THAT ASKED FOR IT, exactly as the
    // AF3 path does and for the same reason: a fold that LOST its template
    // folds and scores, and the number is merely different, so this line is
    // the only thing on screen that says one arrived.
    let templateText = "";
    if (af2Template !== undefined) {
      const source = templateSources[0];
      if (source.origin !== undefined) {
        source.origin.status = describeCoverage(af2Template.coverage);
      }
      templateText = ` · template ${source.source ?? ""}`
        + ` ${af2Template.coverage.residues}/${af2Template.coverage.of}`;
    }
    status(`Done in ${took} s · pLDDT ${best.confidence.meanPlddt.toFixed(1)}`
      + ` · pTM ${best.confidence.ptm.toFixed(3)}${bestIptmText}${ranked}${converged}`
      + templateText + broken);
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
    window.__foldState = { running: false, since: Date.now() };
    // ...whatever happened, including a stop. The timeline is most useful about
    // the fold that did NOT finish, so it is closed here and not on the way out
    // of the success path.
    devEndRun();
    // ...and the slots handed back, for AF2 - whose recycles ARE its frames,
    // so the end of the fold is the end of recycling - and for any fold that
    // stopped or failed before its sampler drew.
    contactsBig(false);
    // ...and what was on screen before, where this fold never drew: a stop or
    // a failure must not leave the page blank with every fold still in it.
    restoreResults();
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
// 🔴 AND IT NAMES THE SERVICE, NOT A SITE. It linked to colabfold.com, which
// is a link to click in a line whose whole job is to say where the sequence
// goes - and the answer is an MMseqs2 server, which is what the reader is
// being told about. The citation belongs in the README, where it is.
const PRIVACY_NOTE = {
  search: ['<i class="fa-solid fa-cloud-arrow-up" style="margin-right: 5px; color: #f59e0b;"></i>',
    "folds locally · MSA via MMseqs2 server"].join(""),
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
  // 🔴 ONE FILE INPUT FOR BOTH, AND THE `accept` FOLLOWS THE ASK. The upload
  // mode takes an alignment, an archive or a job; "Load job JSON…" takes a
  // job, and narrowing the picker is the whole difference a reader sees
  // between them. It is the same element because `readHandedFile` is the same
  // router - two inputs would be two places for the routing to drift.
  const wantsJob = modeSelect.value === "job";
  const file = element("msa-file");
  file.hidden = modeSelect.value !== "upload" && !wantsJob;
  file.accept = wantsJob ? ".json" : ".a3m,.fasta,.fa,.txt,.zip,.json";
  // ...getElementById rather than element(), which throws on a missing id: the
  // note is index.html's and this file should not require it to exist.
  const note = document.getElementById("privacy-note");
  if (note !== null) {
    note.innerHTML = modeSelect.value === "search" ? PRIVACY_NOTE.search : PRIVACY_NOTE.local;
  }
};
// 🔴 THE MODE IT REPLACED IS RECORDED BEFORE THE CHANGE, NOT AFTER. By the time
// a `change` fires the select already holds the new value, so the only place
// the old one still exists is here - `modeBeforeJob` is what `msaMode()` answers
// with while the picker is open and what goes back when the file is read.
modeSelect.addEventListener("change", () => {
  if (modeSelect.value === "job") {
    // ...and opening the picker is the whole point of choosing it. A reader who
    // picks "Load job JSON…" and then has to find a second control has been
    // given a label rather than a door.
    element("msa-file").hidden = false;
    element("msa-file").accept = ".json";
    element("msa-file").click();
  } else {
    modeBeforeJob = modeSelect.value;
  }
  syncMode();
});

installDevPanel();
syncMode();

// 🔴 THE MODEL DECIDES WHICH CONTROLS EXIST, and syncMode decides what the MSA
// ones say - so the model listener runs syncModelControls and then syncMode,
// in that order: the second reads the visibility the first just set.
const familySelect = document.getElementById("model-family");
if (familySelect !== null) {
  // 🔴 AND ALL THREE ROWS THAT MOVE THE FAMILY HAVE TO SAY SO. `chosenFamily`
  // reads this select, the AF2 number and the PLM row, so a veil hung on only
  // the first would leave the other two switching models in silence - the same
  // "everything keyed on a family has to be refreshed here" the comments below
  // already make about the controls.
  familySelect.addEventListener("change", () => {
    syncModelControls(); syncMode();
  });
}
document.getElementById("af3-mode")?.addEventListener("change", syncAf3Count);
// 🔴 AND THE MODEL NUMBER CHANGES THE FAMILY TOO, exactly as the PLM row does -
// `chosenFamily` folds it into the name, so everything keyed on a family (the
// weight cache, the download stem, the status line) has to be refreshed here or
// the page goes on describing the model it was showing before.
document.getElementById("af2Model")?.addEventListener("change", () => {
  syncModelControls();
  syncMode();
});
// 🔴 THE PLM ROW CHANGES THE FAMILY, so everything the model row's own listener
// refreshes has to refresh here too - `chosenFamily()` reads this select, and a
// control left showing the other checkpoint's options is the same
// quietly-wrong state syncModelControls exists to prevent.
document.getElementById("plm-mode")?.addEventListener("change", () => {
  syncModelControls();
  syncAf3Count();
});
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

/**
 * A job JSON, applied to the page: the entities, the seed, and the alignment
 * dial when the file asked for none.
 *
 * 🔴 IT SETS CONTROLS, AND SAYS WHICH ONES. Loading a file that silently moved
 * the seed and the MSA dial is how somebody folds a job they did not ask for
 * and cannot see - so everything it touched goes into the status line, along
 * with anything the file said that this page answered differently.
 *
 * 🔴 AND IT DOES NOT PICK THE MODEL. A ligand, a nucleic chain or a modified
 * residue needs AF3, and the guard that says so already exists at fold time
 * with a message naming the model that is set. A second decision here would be
 * a second reader of the same control - the mistake `chosenFamily` was written
 * to end.
 */

function applyJob(job) {
  entityList.set(job.entities);
  const said = [];
  if (job.seed !== undefined) {
    const input = document.getElementById("random-seed");
    if (input !== null && String(job.seed) !== input.value) {
      input.value = String(job.seed);
      said.push(`seed ${job.seed}`);
    }
  }
  if (job.singleSequence && modeSelect.value !== "none") {
    modeSelect.value = "none";
    syncMode();
    said.push("MSA off");
  } else if (modeSelect.value === "job") {
    // 🔴 THE DIAL GOES BACK, because "Load job JSON…" is a door and not a
    // setting - left on it, the row would be describing an alignment source
    // that does not exist for every fold after this one. The branch above wins
    // where it fires: a file that asks for no alignment has SAID what the dial
    // should read, and restoring the previous mode over that would run the
    // search it asked us to skip.
    modeSelect.value = modeBeforeJob;
    syncMode();
  }
  // 🔴 CHAINS, NOT ROWS. A contact row is a bond between two chains and a
  // ligand is not a chain either, so counting rows reported "2 chains + 1
  // ligand" for a protein, a ligand and the bond between them.
  const chains = job.entities.reduce((total, entity) =>
    total + (POLYMER_TYPES.includes(entity.type) ? entity.copies : 0), 0);
  const ligands = job.entities.filter((entity) => entity.type === "ligand").length;
  said.unshift(`${chains} chain${chains === 1 ? "" : "s"}`
    + (ligands === 0 ? "" : ` + ${ligands} ligand${ligands === 1 ? "" : "s"}`));
  return [...said, ...job.notes].join(" · ");
}

/**
 * A file somebody handed the page: an alignment, a fold archive, or a JOB.
 *
 * 🔴 ONE ROUTER, BECAUSE THE FORMAT IS DECIDED BY THE BYTES AND A SECOND
 * PLACE TO DECIDE IT IS A SECOND PLACE TO DECIDE IT DIFFERENTLY. This was the
 * body of the `msa-file` change listener, and the drop target below now reads
 * the same four kinds through the same branches - a fold archive renamed to
 * .a3m is still an archive and an a3m called .zip is still an alignment, so
 * guessing by extension gives a confusing parse error rather than a refusal.
 * See looksLikeZip. The whole-object rule one section up, applied to a reader.
 *
 * Throws; the callers below turn that into the status line.
 */
async function readHandedFile(bytes, { name = "", textIs = "alignment" } = {}) {
  if (looksLikeZip(bytes)) {
    const files = await readZip(bytes);
    // 🔴 AND THE JOB, NOT ONLY THE ALIGNMENT. The README in this very
    // archive tells the reader to drop it back "to fold again with exactly
    // these alignments" - and until now that restored the a3m and nothing
    // else: the sequence, the ligands, the modifications and the seed all
    // had to be retyped from the request file by hand. The two belong
    // together in any case, since an archive's alignment is FOR its own
    // sequence and attaching it to a different one is the query-wins rule
    // papering over a mismatch.
    let loadedJob;
    const requestName = [...files.keys()].find(
      (path) => path.endsWith("job_request.json"));
    if (requestName !== undefined) {
      // ...a refusal here is reported and does not cost the alignment: an
      // archive from a newer format still carries usable a3m files.
      try { loadedJob = applyJob(jobFromJson(files.get(requestName))); }
      catch (error) { loadedJob = `job not loaded: ${error.message}`; }
    }
    const restored = msasFromArchive(files);
    if (restored.chains === 0 && restored.merged === undefined) {
      if (loadedJob !== undefined) { status(`archive · ${loadedJob}`); return; }
      throw new Error("that archive holds no alignments");
    }
    const alsoJob = loadedJob === undefined ? "" : ` · ${loadedJob}`;
    if (restored.chains === 0) {
      // An archive whose fold was given one merged alignment carries it
      // back as exactly that, with no split to restore.
      uploadedMsas = { merged: restored.merged };
      uploadedA3m = restored.merged;
      const described = parseA3m(restored.merged);
      status(`archive · ${described.depth} sequences`
        + ` · ${described.length} columns${alsoJob}`);
      return;
    }
    uploadedMsas = restored;
    uploadedA3m = "";
    const paired = restored.pairedA3ms.size;
    status(`archive · ${restored.chains} chain${restored.chains === 1 ? "" : "s"}`
      + `${paired > 0 ? `, ${paired} with paired rows` : ", no paired rows"}`
      + alsoJob);
    return;
  }
  const text = new TextDecoder().decode(bytes);
  // 🔴 THE BYTES DECIDE HERE TOO. A job JSON and an a3m are both text, and
  // parseA3m reads `[{"name": ...` as a record whose sequence is the file -
  // no error, a "1 sequence" status, and a fold against an alignment made
  // of punctuation. The first non-space character is what separates them.
  if (/^\s*[[{]/.test(text)) {
    status(`job · ${applyJob(jobFromJson(text))}`);
    return;
  }
  // 🔴 A STRUCTURE IS REFUSED BY NAME, BECAUSE parseA3m WOULD TAKE IT. An a3m
  // is "any text that is not JSON" by the time control reaches here, and a PDB
  // is text - so a dropped structure became an alignment of ATOM records
  // rather than an error, which is the silent-wrong-answer shape this file
  // objects to everywhere else. It matters now that this page owns the drop:
  // dropping a .pdb used to show it in py2Dmol's viewer, and a reader who
  // tries that is owed the reason it no longer does.
  if (/^(ATOM|HETATM|HEADER|MODEL|CRYST1|REMARK|data_|loop_)/m.test(
        text.slice(0, 4096))) {
    throw new Error("that looks like a structure, and this page folds sequences"
      + " - drop an AlphaFold 3 job JSON, a fold archive or an alignment,"
      + " or set a template on the chain's \u22ee menu");
  }
  // 🔴 PLAIN TEXT MEANS DIFFERENT THINGS AT THE TWO DOORS, AND THAT IS THE ONE
  // PLACE THEY DIVERGE - so it is a parameter with a name rather than a second
  // copy of this function. The MSA box is a control that says "this is my
  // ALIGNMENT" and always has. A drop on the page says "this is what I want to
  // fold", which is what makes a dropped FASTA fill the chain rows. Nothing
  // regresses on the split: the page-wide drop did not exist until today - it
  // went to py2Dmol - so no reader has ever dropped an a3m here and had it
  // taken as one.
  if (textIs === "input" && !looksLikeAlignment(text, name)) {
    const rows = entitiesFromText(text);
    if (rows.length === 0) throw new Error("no sequence in that file");
    entityList.set(rows);
    const chains = rows.reduce((total, row) => total + row.copies, 0);
    status(`${chains} chain${chains === 1 ? "" : "s"} from ${name || "that file"}`
      + " · MSA ▸ Upload file if it was meant as an alignment");
    return;
  }
  const described = parseA3m(text);
  uploadedA3m = text;
  uploadedMsas = undefined;
  status(`${described.depth} sequences · ${described.length} columns`);
}

/**
 * Is this text an ALIGNMENT rather than a list of chains to fold?
 *
 * 🔴 IT HAS TO BE ASKED, BECAUSE `entitiesFromText` WOULD TAKE AN a3m AND MAKE
 * A ROW PER SEQUENCE. A 7907-row alignment dropped on the page would become
 * 7907 entity rows - not an error, not a fold, just a page that stops
 * responding while it renders them. The three tests below are each a thing an
 * alignment HAS and a handful of chains does not, and the status line always
 * says which way it went, so a wrong guess is visible and one click from
 * fixed.
 */
function looksLikeAlignment(text, name) {
  // An .a3m says what it is.
  if (/\.a3m$/i.test(name)) return true;
  const records = text.split(/^>/m).slice(1);
  // 🔴 LOWERCASE IS AN a3m INSERTION, which is the format's own marker for a
  // column that is not in the query - a plain FASTA of chains has none.
  if (records.some((record) => /[a-z]/.test(
    record.split(/\r?\n/).slice(1).join("")))) return true;
  // ...and nobody hand-drops a nine-chain complex, where a search returns
  // hundreds of rows. A complex that big goes in through the rows or a job.
  return records.length > 8;
}

/**
 * ...from a file input or a drop, with the refusal going to the status line.
 *
 * 🔴 AND A FAILED READ CLEARS THE ALIGNMENT RATHER THAN LEAVING THE LAST ONE
 * IN PLACE. A file that did not load must not fold with whatever the previous
 * one left behind, which would be the wrong alignment reported as the right
 * one.
 */
function loadHandedFile(file, textIs = "alignment") {
  void file.arrayBuffer().then(async (buffer) => {
    try {
      await readHandedFile(new Uint8Array(buffer), { name: file.name, textIs });
    } catch (error) {
      uploadedA3m = "";
      uploadedMsas = undefined;
      status(error instanceof Error ? error.message : String(error), true);
    }
  });
}

element("msa-file").addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (file !== undefined) loadHandedFile(file);
});

/**
 * ...AND FROM A DROP ANYWHERE ON THE PAGE, WHICH THIS PAGE NOW OWNS OUTRIGHT.
 *
 * 🔴 py2Dmol OWNED THE DROP, AND A JOB DROPPED ON IT WAS READ AS A BROKEN
 * STRUCTURE FILE. Its `initDragAndDrop` binds the four drag events on
 * document.body and hands every file to `handleFileUpload`, which routes .zip
 * to its own session reader and everything else to `processFiles`. Measured
 * with this listener disabled, dropping one of AlphaFold 3's own example jobs
 * on the page gives **"Error processing loose files: No structural files
 * (*.cif, *.pdb, *.ent) found."** and leaves every row as it was.
 *
 * 🔴 AND THE WRONG MESSAGE WAS THE PROBLEM, NOT THE LOSS. The page has had a
 * reader for that exact file since web/job-json.js existed - nine of
 * DeepMind's fourteen examples load - and the one error a reader saw sent them
 * looking for a structure problem in a file that has no structures in it and
 * never should have. The door that did work was the alignment upload box,
 * hidden until the MSA dropdown is set to "Upload file". A capability nobody
 * can find is a capability nobody has.
 *
 * 🔴 TWO READERS FOR ONE GESTURE IS THE WHOLE BUG, SO THERE IS NOW ONE. An
 * earlier version of this listener claimed `.json`, parsed it, and handed it
 * back to py2Dmol when the top level had no `sequences` - and every refusal
 * then had to be guessed at twice, because which reader answered depended on
 * how far the other one got. This page reads what it folds WITH: a job JSON in
 * either dialect, a fold archive, an alignment. Anything else is refused by
 * name. `readHandedFile` is the one router and the alignment upload box shares
 * it, so the two entry points cannot drift.
 *
 * 🔴 WHAT THAT COSTS, SAID OUT LOUD: dropping a .pdb or .cif no longer shows
 * it in the viewer, and py2Dmol's `paeFromJSON` pairing - a structure and its
 * PAE .json dropped together - is gone with it. Both were reachable here and
 * both worked. They are the price of one reader, and the refusal names them
 * rather than leaving a reader wondering whether the drop registered.
 *
 * 🔴 AND THE FOUR EVENTS GO TOGETHER, NOT JUST `drop`. py2Dmol shows its
 * overlay on dragenter and counts enters against leaves in a closure this
 * module cannot reach; taking only the drop left the count stuck and the
 * overlay up for ever, which an earlier version had to undo with a synthetic
 * empty drop. Taking all four means its counter never moves at all, and this
 * page drives `#global-drop-overlay` itself - the same element, so the visual
 * is unchanged.
 *
 * 🔴 `#file-upload` AND `#upload-button` STAY IN index.html, HIDDEN. Deleting
 * them throws inside py2Dmol's `setupEventListeners`, which silently aborts
 * the rest of `initializeApp` and takes the MSA panel's wiring with it - the
 * warning is in index.html beside them. They are in a panel at
 * `display: none`, so nothing reaches them.
 *
 * Gated by `tools/fold-in-page.py --drop-job`, whose arms were each watched
 * failing.
 */
const DROP_OVERLAY = () => document.getElementById("global-drop-overlay");

for (const kind of ["dragenter", "dragover", "dragleave"]) {
  window.addEventListener(kind, (event) => {
    // 🔴 preventDefault IS WHAT MAKES A DROP HAPPEN AT ALL. Without it on
    // dragover the browser navigates to the file instead, which is the
    // default this page used to get from py2Dmol's own handler.
    event.preventDefault();
    event.stopPropagation();
    const overlay = DROP_OVERLAY();
    if (overlay === null) return;
    // 🔴 COUNTED OFF `relatedTarget`, NOT OFF A DEPTH TALLY. A tally has to be
    // right on every enter and leave or it sticks - which is exactly how
    // py2Dmol's overlay got stuck when this page took its drop away. Leaving
    // the window gives a null relatedTarget, and nothing else has to balance.
    overlay.style.display =
      kind === "dragleave" && event.relatedTarget === null ? "none" : "flex";
  }, true);
}

window.addEventListener("drop", (event) => {
  event.preventDefault();
  event.stopPropagation();
  const overlay = DROP_OVERLAY();
  if (overlay !== null) overlay.style.display = "none";
  const file = [...(event.dataTransfer?.files ?? [])][0];
  if (file === undefined) return;
  // 🔴 THE SAME ROUTER THE UPLOAD BOX USES, so a job, an archive and a
  // structure mean the same thing whichever way they arrive, and a file this
  // page does not read is refused by name rather than dropped on the floor.
  // `input` is the one difference and it is the drop's whole character: a file
  // let go on the page is WHAT TO FOLD, so a FASTA fills the chain rows, where
  // the same file chosen in the MSA box is an alignment because that is what
  // that control is for.
  loadHandedFile(file, "input");
}, true);


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
/**
 * The archive for a prediction, with or without the alignment it used.
 *
 * 🔴 ONE BUILDER FOR THE BUTTON AND THE SAVED SESSION. The download path reads
 * more of `lastPrediction` than anything else does, and it is where every
 * field a fold forgot to store has surfaced - a second copy of this call would
 * be a second place for a model's missing field to go unnoticed. The only
 * difference between the two is the alignment, and it is a parameter.
 *
 * 🔴 AND `alignmentOmitted` IS NOT THE SAME AS AN ABSENT `msaOrigin`. The
 * first says "this fold used one and it is not in here", the second says "this
 * model does not take one", and the README says something different for each.
 * Empty `msas` without the flag writes a file telling the reader to drop it on
 * the upload box to reproduce the fold, describing an `msas/` that is absent.
 */
/** Whether an alignment set has anything in it, in either of its two shapes. */
function holdsAlignment(msas) {
  if (msas === undefined || msas === null) return false;
  if (typeof msas.merged === "string" && msas.merged !== "") return true;
  return (msas.unpaired ?? []).some((text) => typeof text === "string" && text !== "");
}

function archiveFor(pred, { includeAlignment = true } = {}) {
  // 🔴 WHAT IS ACTUALLY HELD DECIDES, NOT WHERE THE PREDICTION CAME FROM. This
  // read `pred.restored !== true`, from when a restored session never carried
  // an alignment: now one usually does, and a flag about its provenance would
  // have thrown away the alignment it had and written the caveat anyway. The
  // question the archive asks is "is there an a3m to put in `msas/`", and that
  // is answerable by looking. A restored session whose alignment was dropped
  // for space still lands in the third state, correctly - see
  // web/fold-archive.js.
  const holds = includeAlignment && holdsAlignment(pred.msas);
  return buildFoldArchive({
    stem: pred.stem,
    model: pred.model ?? "AlphaFold",
    settings: pred.settings,
    entities: pred.entities,
    msas: holds ? pred.msas : {},
    msaOrigin: pred.msaOrigin,
    // 🔴 OMITTED MEANS THERE WAS ONE AND IT IS NOT HERE. A fold that ran on the
    // single sequence has nothing to omit, and saying it was left out invites
    // the reader to go looking for an archive that carries it.
    alignmentOmitted: !holds && pred.msaOrigin !== undefined
      && pred.msaOrigin !== SINGLE_SEQUENCE_ORIGIN,
      // 🔴 NOT `?? []`, WHICH IS THE DIFFERENCE BETWEEN "none were used" AND
      // "this model has no such control". Defaulting it here silently undid
      // the distinction the archive was taught to make.
      templates: pred.templates,
      // ...and every model's own best where a sweep produced them. See the
      // note in buildFoldArchive: one file per model, ranked.
      perModel: pred.perModel,
      prediction: {
        pdb: pred.pdb,
        chainLengths: pred.chainLengths,
        tokens: pred.tokens,
        // ...no `alignedError`: it inverts across folds. See the note at
        // `paeMap` in the EF2-fast path, and aligned-error.js.
        confidence: {
          ...pred.confidence,
          // ...a model with no confidence head still has these two.
          chainCertainty: pred.chainCertainty,
          chainInterfaceCertainty: pred.chainInterfaceCertainty,
          // 🔴 RESOLVED HERE, NOT WHEN THE FOLD FINISHED. AlphaFold 2 computes
          // its contact map in a setTimeout - the distogram head costs 131 ms
          // at 128 residues and is deliberately off the fold's critical path -
          // so at the moment the prediction was stored it does not exist yet.
          // By the time anyone presses this it does. `contactSource` is the
          // pass the saved structure came from, which is not always the last.
          // 🔴 ONE FIELD, AND IT IS A REFERENCE RATHER THAN A COPY. The three
          // models produce this at three different MOMENTS - AF3 with the
          // trunk, EF2-fast with the trunk and no confidence object to put it
          // in, AF2 in a setTimeout off the saved pass, because its distogram
          // head costs 131 ms at 128 residues and is deliberately off the
          // fold's critical path. So `contactSource` holds the OBJECT that
          // carries them, which for AF2 is still filling in when the
          // prediction is stored and is filled by the time anyone presses
          // this. It was three fields and the archive knew two of them.
          contactProbs: pred.contactSource?.contactProbs,
        },
      },
  });
}

element("download-all").addEventListener("click", async () => {
  const pred = activePrediction();
  if (!pred) return;
  const button = element("download-all");
  button.disabled = true;
  try {
    const files = archiveFor(pred, { includeAlignment: true });
    downloadBlob(`${pred.stem}.zip`, await writeZip(files), "application/zip");
  } catch (error) {
    status(error instanceof Error ? error.message : String(error), true);
  } finally {
    button.disabled = false;
  }
});

/**
 * Keep this fold, so closing the tab does not lose it.
 *
 * 🔴 CALLED TWICE ON THE AF2 PATH, ON PURPOSE. AlphaFold 2 computes its
 * contact map in a `setTimeout` off the finished pass - the distogram head
 * costs 131 ms at 128 residues and is deliberately off the critical path - so
 * a fold saved the instant it finishes has no contacts in it, and for a model
 * whose contact map is the panel a reader looks at, that is most of what they
 * wanted kept. `contactSource` holds the OBJECT rather than a copy, so calling
 * this again once the map lands rewrites the one record with it. Delaying the
 * save by a guessed interval would be the bisect-by-guessing this repository
 * has been wrong with before.
 */
/**
 * Save once the viewer has stopped changing.
 *
 * 🔴 A FOLD IS NOT FINISHED WHEN ITS PREDICTION IS STORED. Measured: at the
 * moment `predictions.set` runs, the viewer object holds ONE frame - the
 * sampler's trajectory is added to it afterwards - so a save fired there wrote
 * a session with 1 frame of 16 and no contact map, and the reader who reloaded
 * got exactly that back. `visibilitychange` catches the settled state, but only
 * for someone who hides the tab first: press reload straight after a fold and
 * that signal never comes, which is the case this got wrong.
 *
 * So this waits for the COUNT TO STOP GROWING rather than for a guessed
 * interval - the bisect-by-guessing this repository has been wrong with twice -
 * and saves when it has been still for six checks. AF2's contact map lands
 * later still, in a `setTimeout` off the finished pass, and has its own re-save
 * where it arrives.
 */
async function rememberSessionWhenSettled(pred) {
  if (!pred?.pdb) return;
  const registry = window.py2dmol_viewers ?? {};
  const renderer = registry[Object.keys(registry)[0]]?.renderer;
  const count = () => renderer?.objectsData?.[pred.stem]?.frames?.length ?? 0;
  let last = -1;
  let still = 0;
  for (let tries = 0; tries < 200; tries += 1) {
    const now = count();
    if (now > 0 && now === last) {
      still += 1;
      if (still >= 6) break;
    } else {
      still = 0;
      last = now;
    }
    await new Promise((done) => setTimeout(done, 50));
  }
  await rememberSession(pred);
}

async function rememberSession(pred) {
  if (!pred?.pdb) return;
  try {
    // 🔴 py2Dmol BUILDS THIS, NOT US. `buildViewerState` is what its own Save
    // button writes, so the session carries every frame - the whole sampler
    // trajectory - with the camera, colour mode, style, side chains, PAE and
    // every heatmap already on them. Our own serialiser would have restored
    // one frame and been a second description of the viewer to keep in step.
    const state = globalThis.buildViewerState?.();
    if (!state) return;
    // ...what the viewer actually held when this ran, so a session that comes
    // back with fewer frames than the fold had names the moment it was taken.
    const held = viewer?.objectsData?.[pred.stem]?.frames?.length;
    // ...and the half py2Dmol has no idea about: which model ran, against what,
    // and what the confidence head said. The loader ignores unknown keys.
    state.localfold = jobMeta({
      stem: pred.stem,
      model: pred.model ?? "AlphaFold",
      prediction: pred,
      sequence: (pred.chains ?? []).join(":"),
      settings: pred.settings,
      entities: pred.entities,
      // ...and the form that made it, read at the moment the fold STARTED
      // (foldContext) so that editing the boxes while it runs does not
      // rewrite what this fold was.
      inputs: pred.inputs,
      msaOrigin: pred.msaOrigin,
      // 🔴 THE ALIGNMENT, so a restored fold can be REPRODUCED rather than only
      // looked at. See the note in web/fold-session.js for why it is affordable
      // now and was not before, and why it is the one field allowed to go.
      msas: pred.msas,
    });
    state.localfold.framesAtSave = held;
    // ...and HOW MANY FOLDS this session will put back, which is what the
    // offer row describes. Counted from the state rather than from our own
    // bookkeeping: what a restore brings back is exactly what py2Dmol saved.
    state.localfold.folds = (state.objects ?? []).length || 1;
    let saved = await saveSession(state);
    // 🔴 THE ALIGNMENT IS DROPPED AND THE SESSION IS SAVED AGAIN, rather than
    // the whole session being lost to one deep MSA. Everything else in the
    // record is bounded by the fold; an alignment is bounded by whatever a
    // public server returned, so it is the field that can make a save fail -
    // and a session without its alignment is exactly what this used to store,
    // which the archive already knows how to describe. Said out loud, because
    // the difference is whether the fold can be reproduced.
    if (saved === "quota" && state.localfold.msas !== undefined) {
      delete state.localfold.msas;
      saved = await saveSession(state);
      if (saved !== "quota") {
        status("saved without its alignment - there was no room for it", true);
      }
    }
    if (saved === "quota") {
      status("no room to save this session - clear site data to save again", true);
      return;
    }
    // 🔴 AND THE OFFER IS RE-ASKED, because the record it describes has just
    // been replaced. A page that restored a session and then folded something
    // else left the row on screen still advertising the OLD fold - by then the
    // save had already overwritten it, so pressing Restore would have brought
    // back the new fold under the old fold's description. Re-asking hides the
    // row, which is the right answer: what is saved is what is on screen.
    void offerSession();
  } catch (error) {
    // ...a fold that cannot be saved is still a fold on screen.
    console.warn("could not save this session", error);
  }
}

/** How long ago, in the units a person would say it in. */
function agoLabel(then, now = Date.now()) {
  const seconds = Math.max(0, Math.round((now - then) / 1000));
  if (seconds < 60) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

/**
 * Offer the saved fold, if there is one and it is not already on screen.
 *
 * 🔴 IT HIDES ONCE THE FOLD IT DESCRIBES IS THE FOLD IN THE VIEWER. Otherwise
 * the page finishes a fold and immediately offers to restore the thing the
 * reader is looking at, which reads as the page not knowing what it is doing.
 */
async function offerSession() {
  const row = element("session");
  if (row === null) return;
  // 🔴 THE SUMMARY, NOT THE SESSION. This read the whole record to draw one
  // line - fine at 52 KB, and 2.8 MB of gzip over 9.2 MB of JSON once the
  // alignment travelled, on every page load, to decide whether to show a row.
  const meta = await readSessionMeta();
  if (meta === undefined || predictions.has(meta.stem)) {
    row.hidden = true;
    return;
  }
  const residues = `${meta.residues} residue${meta.residues === 1 ? "" : "s"}`;
  const plddt = meta.confidence?.meanPlddt;
  const score = plddt === undefined ? "" : ` · pLDDT ${plddt.toFixed(1)}`;
  // 🔴 THE ROW DESCRIBES THE SESSION, AND A SESSION IS EVERY FOLD ON SCREEN.
  // py2Dmol's `buildViewerState` saves every object, and folds accumulate now
  // - so the line named the LAST fold's model, size, ligands and pLDDT while
  // the button would put three folds back, two of them nothing to do with any
  // of those numbers. Asked for: less verbose, because the restore can include
  // runs unrelated to the last one.
  const folds = Number(meta.folds) || 1;
  // 🔴 THE LIGAND AND THE MODIFICATION ARE NAMED, for the reason the status
  // line names them: neither changes the residue COUNT, so a phosphorylated
  // fold and its parent produce the same row and the offer describes the wrong
  // one of the two. The row is the only thing the reader has to recognise it
  // by; the sequence is in the tooltip and says nothing about either.
  const extras = [];
  for (const entity of meta.entities ?? []) {
    if (entity.type === "ligand" && (entity.value ?? "").trim() !== "") {
      extras.push(entity.value.trim().toUpperCase());
    }
    for (const one of entity.modifications ?? []) {
      if ((one.code ?? "").trim() !== "") extras.push(`${one.code}${one.position}`);
    }
  }
  const named = extras.length === 0 ? ""
    : ` + ${extras.length > 3 ? `${extras.slice(0, 3).join(", ")} +${extras.length - 3}`
      : extras.join(", ")}`;
  // 🔴 textContent, NEVER innerHTML: the sequence is user input.
  // ...and the detail goes to the TOOLTIP rather than off the page: with one
  // fold it is still the thing a reader recognises the session by, and the
  // ligand and the modification are in it for the reason the status line
  // names them - neither changes the residue count, so a phosphorylated fold
  // and its parent read the same without them.
  const summary = folds > 1
    ? `${folds} folds`
    : `${meta.model} · ${residues}${named}`;
  element("session-text").textContent =
    `Last session · ${summary} · ${agoLabel(meta.savedAt)}`;
  element("session-text").title = [
    folds > 1 ? `${folds} folds, last: ${meta.model}` : meta.model,
    `${residues}${named}${score}`,
    meta.sequence ?? "",
  ].filter((part) => part !== "").join(" · ");
  row.hidden = false;
}

/**
 * Put the saved fold back on screen.
 *
 * 🔴 THIS IS NOT A RE-FOLD. What comes back is the structure the model
 * produced with its PAE and contacts, read out of the archive rather than
 * recomputed. What does not come back is the alignment, so folding again
 * searches afresh - the status line says so, and so does the saved README.
 *
 * 🔴 AND THE UPLOAD BOX'S ZIP PATH IS NOT THIS PATH. `msasFromArchive` reads an
 * archive for its ALIGNMENTS and drops the structure on the floor - it feeds
 * the next fold rather than restoring the last one - which is why this needed a
 * reader of its own rather than the one already there.
 */
/**
 * The PAE, the contact map and the per-residue pLDDT, out of the frames.
 *
 * 🔴 THE FRAMES ALREADY HOLD ALL THREE, so the saved session does not carry a
 * second copy. This page writes `frame.pae` as float ROWS (see `paeMatrix`)
 * and py2Dmol's session rounds them to one decimal; a map in `frame.maps` is
 * bytes with the bounds it was encoded against, which inverts exactly -
 * `contactMapFor` writes `round(p * 255)` with vmin 0 and vmax 1, and
 * `mapsOfFrame` normalises every producer to that same `{data, n, vmin, vmax}`
 * shape. Decoding through the map's OWN bounds rather than a constant here is
 * what keeps this correct for a map some other path encoded differently.
 *
 * 🔴 AND THE FIRST FRAME THAT HAS ONE WINS, because a trajectory carries a
 * contact map on one frame and coordinates on all of them - AF3 attaches it to
 * `flow_0` - so a search that looked only at the frame on screen would find
 * nothing on the fifteenth.
 */
function matricesFromFrames(renderer) {
  const frames = renderer?.objectsData?.[renderer?.currentObjectName]?.frames ?? [];
  const out = {};
  const decode = (entry) => {
    const raw = entry?.data ?? entry;
    if (raw === undefined || raw === null || typeof raw === "string") return undefined;
    const low = Number(entry?.vmin ?? 0);
    const high = Number(entry?.vmax ?? 1);
    const span = (high - low) / 255;
    const values = new Float32Array(raw.length);
    for (let index = 0; index < raw.length; index += 1) values[index] = low + raw[index] * span;
    return values;
  };
  for (const frame of frames) {
    if (out.predictedAlignedError === undefined && Array.isArray(frame.pae) && frame.pae.length > 0) {
      // ...rows from this page, a flat vector from anything that wrote one.
      out.predictedAlignedError = Array.isArray(frame.pae[0])
        ? Float32Array.from(frame.pae.flat())
        : Float32Array.from(frame.pae);
    }
    if (out.contactProbs === undefined && frame.maps?.contact !== undefined) {
      out.contactProbs = decode(frame.maps.contact);
    }
  }
  // 🔴 THE pLDDT IS THE LAST FRAME'S, AND WALKING FORWARDS TOOK THE FIRST
  // FRAME'S ZEROS. The confidence head runs ONCE, on the finished structure,
  // so every frame before it is unmeasured and is written with a ZERO
  // B-factor on purpose (see the note in af3-model.js: "a frame whose
  // confidence is not known is coloured as zero"). Those zeros are a
  // full-length array, so `plddts?.length > 0` was satisfied by frame 0 and
  // the loop above never reached the answer.
  //
  // Measured on a real AF3 fold: frame `flow_0` is 58 values all 0.0 and
  // `final` spans 56.9 to 80.6, so a restored prediction's pLDDT was 58 zeros
  // - which makes `fraction_disordered` 1.0 in the archive it writes back
  // (the threshold is 50) and gives `fullDataJson` an all-zero `atom_plddts`
  // to choose on.
  //
  // The maps above are the opposite case and stay as they are: the contact map
  // is carried ONCE, on frame 0, and the PAE is on the first and the last and
  // is the same matrix either way - so for those the first match IS the
  // answer. This one has a direction because the quantity does.
  for (let index = frames.length - 1; index >= 0; index -= 1) {
    const plddts = frames[index]?.plddts;
    if (plddts?.length > 0) {
      out.plddt = Float32Array.from(plddts);
      break;
    }
  }
  return out;
}

async function restoreSession() {
  const state = await readSession();
  if (state === undefined) {
    status("there is no saved session to restore", true);
    return;
  }
  if (typeof globalThis.loadViewerState !== "function") {
    status("this build cannot restore a session", true);
    return;
  }
  try {
    // 🔴 py2Dmol'S OWN LOADER, WHICH IS NOT THE UPLOAD BOX'S PATH.
    // `msasFromArchive` reads a fold ARCHIVE for its alignments and drops the
    // structure on the floor - it feeds the next fold rather than restoring
    // the last one. This is the reader that has always handled a dropped
    // `.py2dmol.json`, and it puts back every frame, the camera, the colour
    // mode, the style, the side chains, the PAE and every heatmap.
    // 🔴 BESIDE THE FOLD ON SCREEN, NOT OVER IT. `loadViewerState` clears
    // every object by default, so restoring a past session threw away the
    // fold the reader was looking at - they had one thing, asked to see
    // another, and ended up with one thing again. Asked for: the previous
    // session should appear as a previous OBJECT.
    //
    // 🔴 AND THIS IS THE ONE PLACE THAT ACCUMULATES, WHICH IS THE WHOLE RULE.
    // Folding replaces (openBlankFold clears first): a new fold is a new
    // session and the page does not silently fill up. Restoring ADDS,
    // because it is the only act where a reader has said, in as many words,
    // that they want an old fold back while keeping what they have. py2Dmol
    // shows its object picker the moment there are two, so the way between
    // them arrives with the second one.
    const beside = Object.keys(
      (window.py2dmol_viewers?.[Object.keys(window.py2dmol_viewers ?? {})[0]]
        ?.renderer?.objectsData) ?? {}).length > 0;
    await globalThis.loadViewerState(state, { append: beside });

    // 🔴 AND WAITED FOR, NOT SLEPT ON. `loadViewerState` resolves BEFORE it is
    // finished: its last act is a `setTimeout(..., 100)` that picks the current
    // object, syncs the heatmap and renders. Calling `refreshHeatmap` straight
    // after the await therefore runs while `currentObjectName` is still unset,
    // and it returns at its first guard - the structure appears and the panel
    // does not. This waits for the condition rather than guessing an interval,
    // which is the failure mode this repository keeps rediscovering.
    const registry = window.py2dmol_viewers ?? {};
    const renderer = registry[Object.keys(registry)[0]]?.renderer;
    for (let tries = 0; tries < 60; tries += 1) {
      const name = renderer?.currentObjectName;
      if (name !== undefined && renderer?.objectsData?.[name]?.frames?.length > 0) break;
      await new Promise((done) => setTimeout(done, 50));
    }

    const meta = state.localfold;
    const stem = meta?.stem ?? state.current_object ?? "session";

    // 🔴 AND THE JOB COMES BACK FROM OUR OWN KEY, because py2Dmol's frames do
    // not carry it. Without this the structure returns with a blank score card:
    // `updateScoresCard` hides its box outright when handed undefined, and the
    // frames know coordinates and maps but not what the confidence head said.
    // 🔴 THE MATRICES ARE READ BACK OUT OF THE FRAMES, NOT STORED TWICE. See
    // `jobMeta`: the PAE and the contact map are already in py2Dmol's session,
    // so keeping float copies beside them wrote every pair a second time - and
    // n^2 is the term that grows fastest with chain length. What comes back is
    // quantised (1 decimal for the PAE, 1/255 for the contacts), which the
    // archive's own two-decimal rounding absorbs entirely for the contacts and
    // costs the PAE one digit.
    const recovered = matricesFromFrames(renderer);

    // 🔴 A MODEL WITH NO CONFIDENCE HEAD STILL HAS A CONTACT MAP, AND IT IS ITS
    // ONLY SCORE. This collapsed the whole object to undefined whenever the
    // summary was absent, which threw away the matrices just recovered from the
    // frames - so EF2-fast's restored archive lost `contact_probs` and its
    // `_summary_confidences_0.json` entirely, the one file
    // `chain_pair_max_contact` lives in. The summary being absent says nothing
    // about the maps.
    const scored = meta?.confidence?.meanPlddt !== undefined;
    const savedConfidence = {
      ...(meta?.confidence ?? {}),
      ...(recovered.predictedAlignedError === undefined ? {}
        : { predictedAlignedError: recovered.predictedAlignedError }),
      ...(recovered.contactProbs === undefined ? {}
        : { contactProbs: recovered.contactProbs }),
      // 🔴 AND pLDDT IS ATTACHED ONLY WHERE THERE IS ONE. `fullDataJson` picks
      // `atom_plddts` over `atom_certainty` on exactly this field's presence,
      // so handing it EF2-fast's B-factors - which are a distogram certainty,
      // under a REMARK saying so - would label them as the model's pLDDT in
      // the one file a reader is most likely to parse.
      ...(scored && recovered.plddt !== undefined ? { plddt: recovered.plddt } : {}),
    };

    // 🔴 EVERYTHING BOTH DOWNLOAD BUTTONS READ, or they are on screen and
    // broken. "PDB" writes `prediction.pdb` and "All" runs the whole archive
    // builder over it - the structure, the token layout, the confidences and
    // the contact map - so a restored prediction missing any of them is a
    // button that fails when pressed rather than one that is not offered.
    // 🔴 `msas` COMES BACK WHEN IT WAS SAVED AND IS ABSENT WHEN IT WAS NOT, and
    // the archive reads that difference itself rather than being told by a
    // flag. A session written before the alignment travelled, or one whose
    // alignment was dropped for space, restores without it and its README
    // carries the "may find different hits" caveat; one that has it writes a
    // real `msas/` and reproduces.
    const restored = {
      stem,
      pdb: meta?.pdb,
      model: meta?.model ?? "saved session",
      settings: meta?.settings,
      entities: meta?.entities,
      msaOrigin: meta?.msaOrigin,
      msas: meta?.msas,
      chains: (meta?.sequence ?? "").split(":").filter(Boolean),
      chainLengths: meta?.chainLengths ?? [],
      tokens: meta?.tokens,
      // ...and undefined rather than [] when the record predates this, so an
      // older session still says "this model has no such control" instead of
      // claiming a template-taking model used none.
      templates: meta?.templates,
      confidence: savedConfidence,
      contactSource: { contactProbs: savedConfidence?.contactProbs },
      restored: true,
    };
    // 🔴 UNDER THE NAME THE RENDERER GAVE IT, which an append can change: a
    // session whose object is called `fold` landing beside a `fold` already
    // here comes in as `fold_2`. `activePrediction` looks this map up by the
    // object being edited, so filing it under the session's own stem would
    // leave the downloads reading the fallback - the right files today and
    // the wrong ones the moment the picker moves.
    const landedAs = renderer?.currentObjectName ?? stem;
    predictions.set(landedAs, restored);
    restored.stem = landedAs;
    lastPrediction = restored;

    // 🔴 AND THE FORM COMES BACK WITH IT, so the page is where it was rather
    // than showing one fold while set up for another. A restore used to put
    // the picture back and touch no control: pressing Fold then folded
    // whatever was in the boxes, which on a fresh page is the default
    // sequence - a fold you could look at and not continue from.
    const putBack = applyInputs(meta?.inputs);

    // ...and for a session written before the form travelled, the model row
    // alone, which is the half that would otherwise be visibly wrong: a row
    // reading AlphaFold 3 over a restored Boltz-2 structure. Nothing else in
    // an old record says what the controls held.
    const wasFrom = restored.family ?? meta?.family;
    const row = document.getElementById("model-family");
    if (!putBack && row !== null && typeof wasFrom === "string"
        && [...row.options].some((option) => option.value === wasFrom)) {
      row.value = wasFrom;
      syncModelControls();
      syncMode();
    }
    // ...and the buttons are shown only when there is something behind them.
    syncDownloads();

    // 🔴 THE MODULE'S OWN HANDLES ARE RE-POINTED. `refreshHeatmap` reads
    // `viewer` and `viewerObject`, which are set when a FOLD loads a structure
    // and are undefined on a fresh page - so the panel had no object to draw
    // and returned at its first guard.
    viewer = renderer;
    viewerObject = renderer?.currentObjectName ?? stem;

    // ...and the card is shown only where there are scores to put in it: an
    // object carrying just a contact map would otherwise draw the box with
    // dashes, claiming a fold was scored and the numbers lost.
    updateScoresCard(scored ? savedConfidence : undefined);
    // 🔴 AND THE PANEL IS TOLD. loadViewerState calls Heatmap.syncToDrawn, but
    // this page's panel is driven by `refreshHeatmap` off the module's own
    // handles - which is what the fold path calls and what the restore has to
    // call too, or the maps are on the frames and nothing draws them.
    refreshHeatmap();
    element("session").hidden = true;
    const plddt = meta?.confidence?.meanPlddt;
    status(`restored ${stem}`
      + (plddt === undefined ? "" : ` · pLDDT ${plddt.toFixed(1)}`)
      + (meta?.msaOrigin === undefined ? ""
        : " · alignment not saved, so folding again will search afresh"));
  } catch (error) {
    status(error instanceof Error ? error.message : String(error), true);
  }
}

element("session-restore")?.addEventListener("click", () => void restoreSession());
element("session-forget")?.addEventListener("click", async () => {
  await clearSession();
  element("session").hidden = true;
});

/**
 * 🔴 THE SESSION IS SAVED WHEN THE READER LEAVES, NOT WHEN THE FOLD ENDS.
 * Measured: at the moment a fold completes the viewer object holds ONE frame -
 * `framesAtSave: 1` against the sixteen the object ends up with - because the
 * trajectory lands in it after the prediction is stored. AlphaFold 2's contact
 * map arrives later still, in a `setTimeout` off the finished pass. Saving at
 * completion therefore captures a session that is not yet the one on screen,
 * and every fix for that is a guessed delay - which is the bisect-by-guessing
 * this repository has been wrong with twice.
 *
 * `visibilitychange` needs no guess: whatever is on screen when the tab is
 * hidden IS the session, trajectory settled, contact map arrived, and with the
 * camera and colour mode the reader chose rather than the ones the fold ended
 * on. The save at completion stays as a floor, for a tab that is killed
 * without ever being hidden.
 */
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState !== "hidden") return;
  const pred = activePrediction();
  if (pred !== undefined && pred !== null) void rememberSession(pred);
});

void offerSession();

/**
 * 🔴 AND ON A COLAB RUNTIME THIS PAGE IS THE ONE FOLDING, WITH NOBODY LOOKING,
 * while the page a reader OPENED gets the badge that says so - both halves are
 * `installColabBridge`, because the one question "which half of this am I" has
 * one answer and should be asked once.
 * `?role=runtime` is the backend saying so: the bridge then announces itself,
 * pushes every status write, bar fraction and sampler frame as it happens, and
 * takes its instructions from the reader's page over the broker. Off that
 * runtime `installColabBridge` returns immediately and nothing here runs.
 * See web/colab-bridge.js.
 */
// 🔴 AND ON THE RUNTIME, THE DEV LOG IS THE THING WORTH SENDING: it is the one
// record made where the work happened, with that card's memory in it. One hook,
// beside the status tap, for the same reason - these are the page's own calls
// and there is no second reporting path to keep in step.
if (colabRole() === "runtime") devOnEntry((entry) => remoteTap("dev", entry));
installColabBridge();
// ...and if one is already under way on the runtime, follow it from here.
void attachToRunningFold();

// 🔴 THE DOWNLOAD ROW IS ASKED AT LOAD AND WHENEVER THE VIEWER'S CONTENT MOVES,
// not only when a fold ends. A structure can arrive without a fold - a dropped
// file, a restored session - and the Session button belongs to what is DRAWN,
// so the one event py2Dmol dispatches when frames land is what tells us. That
// bus is document-scoped, which is what this page has one viewer for.
syncDownloads();
document.addEventListener("py2dmol-frame-change", syncDownloads);

// 🔴 THE BRIDGE SAYS WHEN THE RUNTIME HAS BEEN STOPPED, on an event rather
// than by calling in: web/colab-bridge.js is imported BY this file, so a call
// the other way would be a cycle. One listener, and the page is a viewer.
document.addEventListener("localfold-runtime-stopped", (event) => {
  retireFolding(event.detail?.why ?? "the fold service has been stopped");
});
