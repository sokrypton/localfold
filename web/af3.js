/**
 * The AF3 page: a sequence in, a structure in the viewer.
 *
 * Everything that decides the answer lives in src/af3/ - featurise.js builds
 * the batch, fold.js runs the model, weights.js reads the checkpoint. This file
 * is the part that knows about the DOM: a text box, a progress bar, a licence
 * gate and a viewer.
 *
 * 🔴 THE WEIGHTS ARE A QUARTER OF A GIGABYTE AND THE FOLD IS MINUTES. Those two
 * facts drive every decision here. The download is gated behind an explicit
 * acceptance and reported byte by byte; the fold reports the diffusion step it
 * is on, because a progress bar that sits at "folding..." for two and a half
 * minutes is indistinguishable from one that has hung.
 *
 * 🔴 THE STEP COUNT IS THE QUALITY DIAL, AND IT IS THE ONLY HONEST ONE. The
 * trunk is 3.7 s of a 146 s fold at 200 steps; diffusion is the other 97%, and
 * it is linear in the step count. Nothing else on this page changes the time
 * materially - and since every step is now a frame, the animation is linear in
 * it too.
 */
import { featuriseProtein } from "../src/af3/featurise.js";
import { foldBatch, toPdb, atomName } from "../src/af3/fold.js";
import { confidenceWeights, trunkWeights } from "../src/af3/weights.js";
import { diffusionWeights, atomReference, targetFeatureWeights }
  from "../src/af3/diffusion-weights.js";
import { createStructureViewer } from "./viewer.js";
import { requestAlphaFoldDevice } from "../src/runtime/device.js";
import { isAbortError, throwIfAborted } from "../src/runtime/abort.js";
import { HttpTensorStore } from "../src/reference/http-tensor-store.js";
import { MODEL_BUNDLES, loadManifest } from "../src/reference/manifests/index.js";
/** Where the acceptance is remembered, per browser. */
const ACCEPTED = "localfold.af3.termsAccepted";

const element = (id) => document.getElementById(id);
const text = (id, value) => { const node = element(id); if (node) node.textContent = value; };

function status(message, isError = false) {
  const node = element("status-message");
  if (node === null) return;
  node.textContent = message;
  node.classList.toggle("error", isError);
}

function progress(fraction) {
  const bar = element("progress");
  if (bar === null) return;
  bar.dataset.state = fraction === null ? "idle" : "busy";
  bar.value = fraction ?? 0;
}

/**
 * 🔴 ONLY THE 20 AMINO ACIDS AND X. featurise.js maps anything else to UNK,
 * which is a silent four-atom residue - so a sequence pasted with its FASTA
 * header, or one carrying a nucleotide alphabet, would fold as a chain of
 * blanks and look merely disappointing. Rejected here instead, by name.
 */
export function cleanSequence(raw) {
  const stripped = raw.split("\n").filter((line) => !line.startsWith(">")).join("");
  const sequence = stripped.replace(/\s/g, "").toUpperCase();
  const unknown = [...new Set(sequence)].filter((c) => !"ACDEFGHIKLMNPQRSTVWYX".includes(c));
  if (unknown.length > 0) {
    throw new Error(`${unknown.join(", ")} ${unknown.length === 1 ? "is not an" : "are not"}`
      + ` amino acid ${unknown.length === 1 ? "code" : "codes"}.`
      + " This page folds protein chains written in the 20 one-letter codes, or X.");
  }
  if (sequence.length === 0) throw new Error("Paste a protein sequence first.");
  return sequence;
}

/**
 * The Fold button doubles as the Stop button, as it does on the other pages.
 *
 * 🔴 A FOLD IS MINUTES AND THERE IS NO OTHER WAY OUT OF IT. Without this the
 * only way to abandon a 320-step run is to close the tab, which also throws
 * away the 265 MiB the page just downloaded.
 */
function setFoldButton(state) {
  const button = element("predict");
  if (button === null) return;
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

/** The seed box, where an empty field means zero and zero means zero. */
function readSeed() {
  const raw = element("seed")?.value ?? "";
  const parsed = Number(raw);
  return raw.trim() === "" || !Number.isFinite(parsed) ? 0 : Math.floor(parsed);
}

/** Nothing but the input panel, until there is something to show. */
function showResults(visible) {
  const panel = element("results");
  if (panel) panel.hidden = !visible;
}

/** Forget the last fold: its structure, its numbers, and its frames. */
function clearResults(viewer) {
  viewer.cancelAnimations();
  viewer.reset();
  viewer.forgetCamera();
  text("plddt-value", "—");
  text("geometry-value", "—");
  text("time-value", "—");
}

let devicePromise;
function getDevice() {
  devicePromise ??= (async () => {
    if (navigator.gpu === undefined) {
      throw new Error("This browser has no WebGPU."
        + " It ships in current Chrome, Edge, Safari and Firefox.");
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) throw new Error("No compatible WebGPU adapter was found.");
    return requestAlphaFoldDevice(adapter);
  })();
  return devicePromise;
}

let weightsPromise;
/**
 * The whole checkpoint, once per page.
 *
 * 🔴 ONCE PER PAGE, NOT ONCE PER FOLD. This page exists to be poked at - try a
 * sequence, change a residue, try again - and re-reading a quarter of a
 * gigabyte between two attempts would make that unusable.
 */
function loadWeights() {
  weightsPromise ??= (async () => {
    const onProgress = ({ loadedBytes, totalBytes }) => {
      progress(totalBytes > 0 ? loadedBytes / totalBytes : 0);
      status(`Downloading the model… ${(loadedBytes / 2 ** 20).toFixed(0)}`
        + ` of ${(totalBytes / 2 ** 20).toFixed(0)} MiB`);
    };
    // 🔴 THE TENSOR TABLE IS COMPILED IN, NOT FETCHED. A deploy once 404'd on a
    // manifest and died before asking for a single shard, which is a failure
    // about metadata dressed as a failure about weights. src/reference/
    // manifests/ holds it as a module, imported lazily so its 116 KiB is not in
    // front of a visitor who never folds anything.
    const bundle = MODEL_BUNDLES.af3;
    const store = await HttpTensorStore.fromManifest(
      bundle.directory, await loadManifest("af3"), onProgress);
    return {
      trunk: await trunkWeights(store, 48, 4),
      diffusion: await diffusionWeights(store),
      confidence: await confidenceWeights(store),
      atomReference: await atomReference(store),
      targetFeat: await targetFeatureWeights(store),
    };
  })();
  return weightsPromise;
}

/** True once the reader has accepted the model terms in this browser. */
export function hasAccepted() {
  try {
    return localStorage.getItem(ACCEPTED) !== null;
  } catch {
    // Private windows and blocked site data throw rather than return null. An
    // unreadable store means "not accepted", never "accepted".
    return false;
  }
}

function remember() {
  try {
    localStorage.setItem(ACCEPTED, new Date().toISOString());
  } catch {
    // Nothing to do: the gate simply appears again next time.
  }
}

/**
 * Where the bar should be at each handover, so it runs roughly linear in TIME
 * rather than in stages. Measured on a 68-residue chain:
 *
 *     input features   4.9 s   CPU, and it cannot report sub-progress
 *     trunk            3.7 s   48 pairformer blocks, which can
 *     diffusion        0.85 s a step
 *
 * 🔴 THE SHARES CANNOT BE CONSTANTS. Features and trunk are fixed costs while
 * diffusion is not, so features are a THIRD of a 20-step fold and a fortieth of
 * a 320-step one. A fixed split makes the bar stall on one setting and jump on
 * the other.
 *
 * 🔴 AND THE FEATURE BAND IS A DEAD ZONE. buildTargetFeat is one synchronous
 * call, so nothing can move while it runs - the bar holds at the band's start
 * and the status line carries the explanation instead. It is the largest single
 * thing on this page still waiting for a GPU kernel.
 */
function timeShares(steps) {
  const features = 4.9;
  const trunk = 3.7;
  const total = features + trunk + 0.85 * steps;
  return { features: features / total, trunk: (features + trunk) / total };
}

/** The dense slot of every alpha carbon, which is what the frames are fitted on. */
function alphaCarbons(batch) {
  const slots = [];
  for (let token = 0; token < batch.tokens; token += 1) {
    for (let atom = 0; atom < batch.dense; atom += 1) {
      const slot = token * batch.dense + atom;
      if (!batch.predDenseAtomMask[slot]) continue;
      if (atomName(batch.refAtomNameChars, slot) === "CA") slots.push(slot);
    }
  }
  return slots;
}

/** Flat xyz as py2Dmol's list of points, one entry per dense slot. */
function toPoints(positions, count) {
  const points = new Array(count);
  for (let index = 0; index < count; index += 1) {
    points[index] = [positions[index * 3], positions[index * 3 + 1], positions[index * 3 + 2]];
  }
  return points;
}

/**
 * One frame of the trajectory, rigidly fitted onto a reference and rendered as
 * a PDB.
 *
 * 🔴 THE FRAMES ARE IN DIFFERENT REFERENCE FRAMES AND MUST BE FITTED. AF3's
 * sampler calls randomAugmentation at the top of EVERY step - a fresh random
 * rotation and translation of the whole system - so consecutive frames differ
 * by a rigid motion far larger than anything the denoiser did. Played back
 * unfitted the protein tumbles wildly and the folding is invisible. This is
 * also the trap that once made the sampler look divergent: a convergence
 * metric on raw coordinates measures the tumbling, not the model.
 *
 * 🔴 FITTED TO THE FIRST FRAME, NOT THE LAST, BECAUSE THE FRAMES ARE SHOWN AS
 * THEY ARE COMPUTED. There is no finished structure to fit to while the fold is
 * still running, and waiting for one would mean watching a blank panel for the
 * whole fold. Any fixed reference removes the tumbling equally well; the first
 * frame is simply the one that exists.
 *
 * 🔴 FITTED, NOT INTERPOLATED. These are the sampler's own frames; nothing
 * between them is invented.
 */
function fittedPdb(batch, positions, reference, slots, plddt) {
  const api = window.py2Dmol;
  const count = batch.tokens * batch.dense;
  if (api?.superpose === undefined || reference === null) {
    return toPdb(batch, positions, plddt);
  }
  const moved = api.superpose(toPoints(positions, count), reference,
                              { from: slots, to: slots });
  const fitted = new Float32Array(count * 3);
  for (let index = 0; index < count; index += 1) {
    fitted[index * 3] = moved[index][0];
    fitted[index * 3 + 1] = moved[index][1];
    fitted[index * 3 + 2] = moved[index][2];
  }
  return toPdb(batch, fitted, plddt);
}

export async function fold({ sequence, steps, seed, viewer, signal }) {
  const batch = featuriseProtein(sequence);
  status(`Loading the model…`);
  const weights = await loadWeights();
  // 🔴 LET THE DOWNLOAD'S LAST FRAME PAINT. The final progress callback and the
  // reset for the fold phase were in the same task, so the bar went from
  // whatever it had last drawn straight back to zero and never showed 100% -
  // which reads as a download that stopped early.
  await new Promise((resolve) => setTimeout(resolve, 0));

  progress(0);
  const started = performance.now();
  const share = timeShares(steps);

  const slots = alphaCarbons(batch);
  let reference = null;
  let shown = 0;

  const result = await foldBatch(await getDevice(), batch, weights, {
    steps, seed,
    onStage: async (name, detail) => {
      if (name === "target-feat-start") {
        status(`Building input features for ${batch.atomCount} atoms…`);
        // The yield is the point: what follows blocks the main thread for
        // seconds, so the line above has to be painted before it starts.
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
      if (name === "target-feat") {
        progress(share.features);
        status(`Running the trunk over ${batch.tokens} tokens…`);
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
      // 🔴 THE PAIRFORMER IS THE TRUNK, as far as a progress bar is concerned:
      // 48 blocks, and the other four stages together are a fraction of one of
      // them. Without this the bar sat at zero for the whole trunk and then
      // jumped, which reads as a hang.
      if (name === "pairformer-block") {
        // 🔴 THE TWO AWAITED CALLBACKS ARE THE ONLY PLACES A FOLD CAN BE
        // INTERRUPTED. Everything between them is a GPU submission that has to
        // finish, so Stop lands within one pairformer block or one diffusion
        // step - a fraction of a second either way.
        throwIfAborted(signal);
        progress(share.features
          + (share.trunk - share.features) * ((detail.index + 1) / detail.total));
        status(`Trunk: pairformer block ${detail.index + 1} of ${detail.total}`);
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
      if (name === "trunk-done") {
        progress(share.trunk);
        status(`Diffusing ${batch.atomCount} atoms over ${steps} steps…`);
      }
    },
    onStep: async ({ step, denoised }) => {
      throwIfAborted(signal);
      // 🔴 `denoised` AND NOT `positions`, AND THE REASON IS THE CAMERA.
      // `positions` is the sampler's actual walk and the more literal picture
      // of diffusion - but measured over a 200-step fold its radius of gyration
      // runs from 1896 A at step 4 down to 11.1 A at the end, a range of 170x.
      // No fixed camera holds both: framed on the finished protein the early
      // frames are entirely off-screen, and framed on the opening cloud the
      // protein is a speck. Those early frames are Gaussian noise at sigma =
      // 2273 in any case, which is not a picture of anything.
      //
      // `denoised` is the model's PREDICTED structure at each step - 11.5 A of
      // gyration at step 4, 11.1 A at the end - so it is protein-sized in every
      // frame and stays in view. What the animation shows is the prediction
      // being refined, which is the part worth watching.
      //
      // 🔴 EVERY STEP IS A FRAME. This used to keep a bounded twenty-four of
      // them, on the grounds that each appended frame costs py2Dmol a
      // side-chain rebuild - about 0.6 s here - so a long fold paid for frames
      // nobody would resolve at playback speed. Keeping all of them makes the
      // animation the trajectory rather than a sample of it, and makes the step
      // count mean one thing instead of two: at 320 steps it is 320 frames and
      // the drawing costs about as much as the diffusion.
      if (reference === null) reference = toPoints(denoised, batch.tokens * batch.dense);
      viewer.push(fittedPdb(batch, denoised, reference, slots, null));
      shown += 1;
      // Follow the newest frame while the fold is running, so the panel shows
      // the structure as it is now rather than parking on frame zero.
      viewer.show(shown - 1, "diffusion");
      if (shown === 1) viewer.orient();
      progress(share.trunk + (1 - share.trunk) * (step / steps));
      const elapsed = (performance.now() - started) / 1000;
      const remaining = elapsed * (steps / step - 1);
      status(`Diffusion step ${step} of ${steps}`
        + `  ·  about ${Math.ceil(remaining)} s left`);
      // 🔴 YIELD, OR THE PAGE NEVER PAINTS. Every await in the sampler resolves
      // from a GPU callback, which is a microtask - so without a real task
      // boundary the status text above would be written and never drawn, and
      // the page would look frozen for the whole fold.
      await new Promise((resolve) => setTimeout(resolve, 0));
    },
  });

  const seconds = (performance.now() - started) / 1000;

  // 🔴 THE LAST FRAME IS THE SAMPLER'S OUTPUT, NOT ITS LAST GUESS. Every frame
  // above is `denoised`; what the fold actually returns is `positions` after
  // the final update, and that is the structure the pLDDT and the geometry
  // below describe. At the end of the schedule the two are within a fraction
  // of an angstrom, which is exactly why appending the real one costs nothing
  // and showing a guess instead would be wrong.
  //
  // It is also the first frame carrying pLDDT: the confidence head does not run
  // until the sample is finished, so the frames drawn during the fold have no
  // per-atom confidence to colour by.
  // 🔴 THE LAST FRAME IS THE SAMPLER'S OUTPUT, NOT ITS LAST GUESS. Every frame
  // above is `denoised`; what the fold returns is `positions` after the final
  // update, and that is the structure the pLDDT and geometry below describe. At
  // the end of the schedule the two agree to a fraction of an angstrom, which
  // is why appending the real one costs nothing.
  //
  // 🔴 AND IT IS THE ONLY FRAME WITH A REAL pLDDT. The confidence head does not
  // run until the sample is finished, so every frame before this one carries a
  // zero B-factor and the pLDDT palette paints it the colour of no confidence.
  // That reads as "uncertain early, confident at the end", which is a fair
  // picture of what diffusion does - but the reds are a MISSING value, not a
  // measured one, and nothing here should be read as a per-step confidence.
  viewer.push(fittedPdb(batch, result.positions, reference, slots, result.scores.plddt));
  viewer.paint();
  viewer.show(shown, "final");

  text("plddt-value", result.meanPlddt.toFixed(1));
  text("geometry-value", `${result.geometry.caca.toFixed(2)} Å`);
  text("time-value", `${seconds.toFixed(0)} s`);
  progress(null);
  status(`Done: ${batch.tokens} residues in ${seconds.toFixed(0)} s,`
    + ` mean pLDDT ${result.meanPlddt.toFixed(1)}.`);
  return result;
}

export function start() {
  const viewer = createStructureViewer({ container: element("viewer"), frameLabel: "step" });

  const gate = element("terms-gate");
  const showGate = (show) => { if (gate) gate.hidden = !show; };
  showGate(!hasAccepted());

  element("accept-terms")?.addEventListener("click", () => {
    remember();
    showGate(false);
    status("Ready. Paste a sequence and press Fold.");
  });

  let activeFold;

  element("predict")?.addEventListener("click", async () => {
    // A second click while a fold is running is Stop, not another fold.
    if (activeFold !== undefined) {
      activeFold.abort();
      setFoldButton("stopping");
      status("Stopping…");
      return;
    }
    if (!hasAccepted()) { showGate(true); return; }
    let sequence;
    try {
      sequence = cleanSequence(element("sequence").value);
    } catch (error) {
      status(error.message, true);
      return;
    }

    const controller = new AbortController();
    activeFold = controller;
    setFoldButton("running");
    // 🔴 THE PREVIOUS FOLD GOES BEFORE THE NEXT ONE STARTS. Leaving it up meant
    // the old structure and its pLDDT sat there describing nothing for the
    // minute the new fold took, which is worse than an empty panel because it
    // looks like an answer.
    clearResults(viewer);
    showResults(true);
    try {
      await fold({
        sequence,
        steps: Number(element("steps")?.value) || 20,
        // 🔴 ZERO IS A SEED, NOT A MISSING VALUE. `Number(value) || default`
        // silently replaces it with the default, so the one seed a reader is
        // most likely to type first would quietly fold something else.
        seed: readSeed(),
        viewer,
        signal: controller.signal,
      });
    } catch (error) {
      progress(null);
      if (controller.signal.aborted || isAbortError(error)) {
        // A stopped fold leaves its partial trajectory on screen: those frames
        // are real predictions, and throwing them away would be a worse answer
        // than keeping them. Only the numbers, which describe a structure that
        // was never finished, are cleared.
        text("plddt-value", "—");
        text("geometry-value", "—");
        text("time-value", "—");
        status("Stopped.");
      } else {
        status(error.message, true);
        throw error;
      }
    } finally {
      activeFold = undefined;
      setFoldButton("idle");
    }
  });

  // What the page says about the machine it is on, before anything is asked of
  // it - a fold that cannot run should say so on load, not two minutes in.
  const summary = element("gpu-summary");
  if (summary) {
    getDevice().then(
      () => { summary.dataset.state = "ready"; summary.textContent = "WebGPU ready"; },
      (error) => { summary.dataset.state = "error"; summary.textContent = error.message; });
  }
}
