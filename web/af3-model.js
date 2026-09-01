/**
 * AlphaFold 3 as one more model the page can fold with.
 *
 * NO DOM IN HERE. web/app.js owns index.html's controls, its viewer and its
 * status line; this is the part that knows about AF3 - which weights, which
 * featuriser, which sampler - and reports progress through callbacks. It exists
 * because AF3 shares nothing below the sequence with AlphaFold 2: a different
 * featuriser, a different checkpoint, a different head, and coordinates that
 * arrive from a sampler rather than a structure module.
 *
 * 🔴 SINGLE SEQUENCE ONLY, SO THE MSA CONTROLS DO NOT APPLY. featurise.js builds
 * a one-row MSA from the query, which is what AF3 is given here - a real
 * alignment would change three arrays in that file and nothing in this one, but
 * it is not wired, and offering the page's alignment controls for an AF3 fold
 * would be offering something that is quietly ignored.
 */
import { featuriseProtein } from "../src/af3/featurise.js";
import { foldBatch, toPdb, atomName } from "../src/af3/fold.js";
import { confidenceWeights, trunkWeights } from "../src/af3/weights.js";
import { diffusionWeights, atomReference, targetFeatureWeights }
  from "../src/af3/diffusion-weights.js";
import { HttpTensorStore } from "../src/reference/http-tensor-store.js";
import { MODEL_BUNDLES, loadManifest } from "../src/reference/manifests/index.js";
import { throwIfAborted } from "../src/runtime/abort.js";

const ALPHABET = "ACDEFGHIKLMNPQRSTVWYX";

/**
 * What the count dial offers per mode, and what it is called.
 *
 * 🔴 THE TWO NUMBERS ARE NOT INTERCHANGEABLE. A ramp CYCLE walks the whole
 * schedule, so eight is a finished structure. A diffusion STEP discretises it,
 * and below twenty the sampler does not land - ten gives 5.91 A on 6MRR with a
 * CA-CA of 8.40 A, a chain that is not connected. So the dial is rebuilt on a
 * mode change rather than carrying a number across.
 */
export const AF3_COUNTS = {
  ramp: { label: "Cycles", values: [2, 4, 8, 16, 32], preferred: 8 },
  diffusion: { label: "Steps", values: [20, 40, 80, 160, 320], preferred: 20 },
};

/**
 * 🔴 ONLY THE 20 AMINO ACIDS AND X. featurise.js maps anything else to UNK,
 * which is a silent four-atom residue - so a sequence carrying a nucleotide
 * alphabet would fold as a chain of blanks and look merely disappointing.
 * Rejected here by name instead.
 */
export function af3SequenceProblem(sequence) {
  const unknown = [...new Set(sequence)].filter((code) => !ALPHABET.includes(code));
  if (unknown.length > 0) {
    return `${unknown.join(", ")} ${unknown.length === 1 ? "is not an" : "are not"}`
      + ` amino acid ${unknown.length === 1 ? "code" : "codes"}.`
      + " AlphaFold 3 folds protein chains written in the 20 one-letter codes, or X.";
  }
  if (sequence.length === 0) return "Paste a protein sequence first.";
  return null;
}

let weightsPromise;

/**
 * The whole AF3 checkpoint, once per page.
 *
 * 🔴 ONCE PER PAGE, NOT ONCE PER FOLD - it is a quarter of a gigabyte, and
 * re-reading it between two attempts would make the page unusable to poke at.
 *
 * 🔴 AND THE TENSOR TABLE IS COMPILED IN, NOT FETCHED. A deploy once 404'd on a
 * manifest and died before asking for a single shard, which is a failure about
 * metadata wearing the costume of a failure about weights.
 */
export function loadAf3Weights(onProgress) {
  weightsPromise ??= (async () => {
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

/** The dense slot of every alpha carbon, which frames are fitted on. */
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

const toPoints = (positions, count) => Array.from(
  { length: count }, (_, i) => [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]]);

/**
 * One trajectory frame, rigidly fitted onto a reference, as a PDB.
 *
 * 🔴 THE FRAMES ARE IN DIFFERENT REFERENCE FRAMES IN DIFFUSION MODE AND MUST BE
 * FITTED. AF3's sampler calls randomAugmentation at the top of every step - a
 * fresh rotation and translation of the whole system - so consecutive frames
 * differ by a rigid motion far larger than anything the denoiser did, and
 * unfitted playback is a protein tumbling. Ramp mode rotates nothing, so there
 * the fit is a no-op that costs one superposition a frame.
 *
 * 🔴 FITTED TO THE FIRST FRAME, NOT THE LAST, because the frames are shown as
 * they are computed and there is no last one yet.
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

/**
 * Where the bar should be at each handover, so it runs roughly linear in TIME
 * rather than in stages. Measured on a 68-residue chain: input features 4.9 s,
 * trunk 3.7 s, and 0.85 s a call after that.
 *
 * 🔴 THE SHARES CANNOT BE CONSTANTS. Features and trunk are fixed costs while
 * the sampler is not, so features are a third of an 8-cycle fold and a
 * fortieth of a 320-step one.
 *
 * 🔴 AND THE FEATURE BAND IS A DEAD ZONE. buildTargetFeat is one synchronous
 * call - the per-atom conditioning and the conditioning atom encoder, both on
 * the CPU - so nothing can move while it runs. It is the largest single thing
 * here still waiting for a GPU kernel.
 */
function timeShares(calls, passes) {
  const features = 4.9;
  const trunk = 3.7 * passes;
  const total = features + trunk + 0.85 * calls;
  return { features: features / total, trunk: (features + trunk) / total };
}

/**
 * Fold one chain with AlphaFold 3.
 *
 * @param {{sequence: string, mode: "ramp"|"diffusion", calls: number, seed: number,
 *          signal: AbortSignal, device: GPUDevice,
 *          onStatus: (text: string) => void, onProgress: (fraction: number) => void,
 *          onFrame?: (pdb: string, index: number) => void}} options
 */
export async function foldAf3(options) {
  const { sequence, mode, calls, recycles, seed, signal, device, onStatus, onProgress } = options;
  const batch = featuriseProtein(sequence);
  const share = timeShares(calls, (recycles ?? 0) + 1);
  const started = performance.now();

  const slots = alphaCarbons(batch);
  let reference = null;
  let shown = 0;

  const result = await foldBatch(device, batch, options.weights, {
    mode, steps: calls, recycles, seed,
    onStage: async (name, detail) => {
      throwIfAborted(signal);
      if (name === "target-feat-start") {
        // 🔴 A SWEEP, NOT A NUMBER. buildTargetFeat is ONE synchronous call - 98%
        // of it is the atom cross-attention encoder, 4.9 s of the 5 on a
        // 68-residue chain - so there is nothing to count and nothing can move
        // while it runs. A bar frozen at a value reads as a hang; the
        // indeterminate state says "working, no idea how far", which is true.
        // The real fix is a GPU kernel: Af3AtomEncoderGpu already has this
        // shape, and porting it would remove the wait rather than dress it.
        onProgress("waiting");
        onStatus(`Building input features for ${batch.atomCount} atoms…`);
        // The yield is the point: what follows blocks the main thread for
        // seconds, so the line above has to be painted before it starts.
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
      if (name === "target-feat") {
        onProgress(share.features);
        onStatus(`Running the trunk over ${batch.tokens} tokens…`);
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
      if (name === "pairformer-block") {
        // Each recycle is another whole trunk, so the trunk band is divided
        // between them rather than replayed.
        const done = (detail.pass + (detail.index + 1) / detail.total) / detail.passes;
        onProgress(share.features + (share.trunk - share.features) * done);
        onStatus(`Trunk · pass ${detail.pass + 1} of ${detail.passes}`
          + ` · pairformer block ${detail.index + 1} of ${detail.total}`);
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
      if (name === "trunk-done") {
        onProgress(share.trunk);
        onStatus(mode === "ramp"
          ? `Refining ${batch.atomCount} atoms over ${calls} cycles…`
          : `Diffusing ${batch.atomCount} atoms over ${calls} steps…`);
      }
    },
    onStep: async ({ step, denoised }) => {
      throwIfAborted(signal);
      // 🔴 `denoised` AND NOT `positions`, AND THE REASON IS THE CAMERA. The
      // sampler's actual walk runs from a radius of gyration of 1896 A at step
      // 4 down to 11.1 A at the end - 170x - so no fixed camera holds both, and
      // its early frames are Gaussian noise at sigma 2273, which is not a
      // picture of anything. `denoised` is the model's predicted structure at
      // each call and is protein-sized in every frame.
      if (reference === null) reference = toPoints(denoised, batch.tokens * batch.dense);
      options.onFrame?.(fittedPdb(batch, denoised, reference, slots, null), shown);
      shown += 1;
      onProgress(share.trunk + (1 - share.trunk) * (step / calls));
      const elapsed = (performance.now() - started) / 1000;
      onStatus(`${mode === "ramp" ? "Cycle" : "Diffusion step"} ${step} of ${calls}`
        + `  ·  about ${Math.ceil(elapsed * (calls / step - 1))} s left`);
      // 🔴 YIELD, OR THE PAGE NEVER PAINTS. Every await in the sampler resolves
      // from a GPU callback, which is a microtask - so without a real task
      // boundary the status above is written and never drawn.
      await new Promise((resolve) => setTimeout(resolve, 0));
    },
  });

  // 🔴 THE VIEWER WANTS ONE pLDDT A RESIDUE AND THE HEAD GIVES ONE AN ATOM.
  // Taking the alpha carbon's is what AlphaFold 2's own per-residue pLDDT means
  // here, and it is the value the cartoon is coloured by.
  const plddt = slots.map((slot) => result.scores.plddt[slot]);

  return {
    batch,
    pdb: result.pdb,
    meanPlddt: result.meanPlddt,
    geometry: result.geometry,
    seconds: (performance.now() - started) / 1000,
    confidence: {
      plddt,
      meanPlddt: result.meanPlddt,
      // 🔴 NO pTM. AF3's confidence head emits PAE and PDE here; pTM and ipTM
      // are not implemented, so the field is absent rather than invented - a
      // plausible number nobody computed is worse than a missing one.
      predictedAlignedError: result.scores.pae,
    },
  };
}
