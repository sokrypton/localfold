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
 * materially.
 */
import { featuriseProtein } from "../src/af3/featurise.js";
import { foldBatch, toPdb, atomName } from "../src/af3/fold.js";
import { confidenceWeights, openAf3Store, trunkWeights } from "../src/af3/weights.js";
import { diffusionWeights, atomReference, targetFeatureWeights }
  from "../src/af3/diffusion-weights.js";
import { createStructureViewer } from "./viewer.js";
import { requestAlphaFoldDevice } from "../src/runtime/device.js";

const MANIFEST = "./model-af3-int5/manifest.json";
/**
 * How many trajectory frames to keep, whatever the step count.
 *
 * 🔴 EACH ONE COSTS ABOUT 0.6 s OF WALL CLOCK - py2Dmol rebuilds the side-chain
 * atoms of every appended frame - so this is a real trade against the fold
 * itself. Fifty frames added 32 s to a 47 s fold; twenty-four is enough to see
 * the structure settle and costs about half that.
 */
const TRAJECTORY_FRAMES = 24;
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
    const store = await openAf3Store(MANIFEST, null, ({ loadedBytes, totalBytes }) => {
      progress(totalBytes > 0 ? loadedBytes / totalBytes : 0);
      status(`Downloading the model… ${(loadedBytes / 2 ** 20).toFixed(0)}`
        + ` of ${(totalBytes / 2 ** 20).toFixed(0)} MiB`);
    });
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
 * Steps 1..N of the sampler, reported as one fraction. The trunk is given the
 * first few percent because it really does take about that share of the fold.
 */
const TRUNK_SHARE = 0.03;

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

export async function fold({ sequence, steps, seed, viewer }) {
  const batch = featuriseProtein(sequence);
  status(`Loading the model…`);
  const weights = await loadWeights();

  status(`Running the trunk over ${batch.tokens} tokens…`);
  progress(0);
  const started = performance.now();

  // Keep a bounded number of frames whatever the step count - the trajectory at
  // 200 steps is 200 x 1632 x 3 floats otherwise, and forty frames is already
  // more than the eye resolves at playback speed.
  const keepEvery = Math.max(1, Math.round(steps / TRAJECTORY_FRAMES));
  const slots = alphaCarbons(batch);
  let reference = null;
  let shown = 0;

  viewer.reset();
  viewer.forgetCamera();

  const result = await foldBatch(await getDevice(), batch, weights, {
    steps, seed,
    onStage: (name) => {
      if (name === "trunk-done") {
        progress(TRUNK_SHARE);
        status(`Diffusing ${batch.atomCount} atoms over ${steps} steps…`);
      }
    },
    onStep: async ({ step, denoised }) => {
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
      if (step % keepEvery === 0 || step === steps) {
        if (reference === null) reference = toPoints(denoised, batch.tokens * batch.dense);
        viewer.push(fittedPdb(batch, denoised, reference, slots, null));
        shown += 1;
        // Follow the newest frame while the fold is running, so the panel shows
        // the structure as it is now rather than parking on frame zero.
        viewer.show(shown - 1, "diffusion");
        if (shown === 1) viewer.orient();
      }
      progress(TRUNK_SHARE + (1 - TRUNK_SHARE) * (step / steps));
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
  let running = false;

  const gate = element("terms-gate");
  const showGate = (show) => { if (gate) gate.hidden = !show; };
  showGate(!hasAccepted());

  element("accept-terms")?.addEventListener("click", () => {
    remember();
    showGate(false);
    status("Ready. Paste a sequence and press Fold.");
  });

  element("predict")?.addEventListener("click", async () => {
    if (running) return;
    if (!hasAccepted()) { showGate(true); return; }
    let sequence;
    try {
      sequence = cleanSequence(element("sequence").value);
    } catch (error) {
      status(error.message, true);
      return;
    }
    running = true;
    element("predict").disabled = true;
    try {
      await fold({
        sequence,
        steps: Number(element("steps")?.value) || 200,
        seed: Number(element("seed")?.value) || 20260831,
        viewer,
      });
    } catch (error) {
      progress(null);
      status(error.message, true);
      throw error;
    } finally {
      running = false;
      element("predict").disabled = false;
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
