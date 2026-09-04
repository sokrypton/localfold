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
 * THE ALIGNMENT ARRIVES AS A3M TEXT, exactly as it does for the two AlphaFold 2
 * models: the page's search, paste and upload controls are shared, and the only
 * AF3-specific step is af3MsaFromA3m() turning that text into AF3's codes. With
 * no alignment the MSA is the query alone, which is what AF3 itself produces
 * for a single-sequence input rather than a stub.
 */
import { featuriseProtein } from "../src/af3/featurise.js";
import { ccdUrl, parseCcdComponent } from "../src/af3/ccd-component.js";
import { af3MsaFromA3m } from "../src/af3/msa-features.js";
import { foldBatch, toPdb, atomName, uniformFrom } from "../src/af3/fold.js";
import { confidenceWeights, trunkWeights } from "../src/af3/weights.js";
import { diffusionWeights, atomReference, targetFeatureWeights }
  from "../src/af3/diffusion-weights.js";
import { HttpTensorStore } from "../src/reference/http-tensor-store.js";
import { bundleBaseUrl, loadManifest } from "../src/reference/manifests/index.js";
import { throwIfAborted } from "../src/runtime/abort.js";
import { yieldToBrowser } from "../src/runtime/yield.js";
import { af3Plan, RuntimeEstimator }
  from "../src/runtime/cost-model.js";

const ALPHABET = "ACDEFGHIKLMNPQRSTVWYX";

/**
 * What the count dial offers per mode, and what it is called.
 *
 * 🔴 THE TWO NUMBERS ARE NOT INTERCHANGEABLE. A flow CYCLE walks the whole
 * schedule; a diffusion STEP discretises it, and below twenty the sampler does
 * not land - ten gives 5.91 A on 6MRR with a CA-CA of 8.40 A, a chain that is
 * not connected. So the dial is rebuilt on a mode change rather than carrying a
 * number across.
 *
 * 🔴 AND SIXTEEN IS THE LOWEST EITHER OFFERS, WHICH IT WAS NOT. Flow used to
 * start at two and prefer eight, and eight IS a finished structure for a chain
 * of standard residues - but not for a modified one, whose atoms are each their
 * own token rather than coming from a shared residue conformer and so have to
 * be placed individually. Measured with tools/gpu/probe-modified.js on a
 * phosphoserine, as the median predicted-to-ideal bond ratio against the
 * unmodified residues of the same chain:
 *
 *     flow-8    0.835   (control 1.003)
 *     flow-16   0.974   (control 1.007)
 *     flow-32   0.996   (control 1.010)
 *
 * A dial whose lowest setting is wrong for a supported input is a trap, and one
 * number that is always good enough beats two and a rule about when to use
 * which. The cost is a few seconds. AF3 itself scores 0.956 against this port's
 * 0.953 at matched settings, so what is left at sixteen is the architecture's
 * price for an atom-tokenised residue and not something more steps will fix.
 */
export const AF3_COUNTS = {
  // 🔴 BOTH ARE CALLED "Steps" ON THE PAGE, because the dial sits beside
  // Recycles and "Cycles" beside "Recycles" reads as the same word twice. The
  // note above still applies to what the numbers MEAN - a flow step walks the
  // whole schedule, a diffusion step discretises it - which is why the values
  // differ by an order of magnitude and the dial is rebuilt on a mode change.
  flow: { label: "Steps", values: [16, 32, 64], preferred: 16 },
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
    const store = await HttpTensorStore.fromManifest(
      bundleBaseUrl("af3"), await loadManifest("af3"), onProgress);
    // 🔴 EVERY SHARD AT ONCE, because the loaders below walk tensors in order
    // and await each one - so without this the network runs one shard at a time
    // and idles through every dequantisation. See HttpTensorStore.prefetch.
    // This path reads the whole model, so there is nothing to be careful about.
    store.prefetch();
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
  const predicted = [];
  for (let token = 0; token < batch.tokens; token += 1) {
    for (let atom = 0; atom < batch.dense; atom += 1) {
      const slot = token * batch.dense + atom;
      if (!batch.predDenseAtomMask[slot]) continue;
      predicted.push(slot);
      if (atomName(batch.refAtomNameChars, slot) === "CA") slots.push(slot);
    }
  }
  // 🔴 A LIGAND HAS NO CA, and a superposition with no correspondences returns
  // NaN coordinates rather than failing. These slots exist to take a rigid
  // motion out of the trajectory, and any consistent set of atoms does that -
  // so a fold with no polymer in it fits on the atoms it does have.
  return slots.length > 0 ? slots : predicted;
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
 * unfitted playback is a protein tumbling. Flow mode rotates nothing, so there
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
 * The plan a fold's bar and clock run off, and where each band ends.
 *
 * 🔴 THIS USED TO BE THREE CONSTANTS THAT DID NOT MENTION LENGTH. A trunk pass
 * was "3.7 s" and a denoiser call "0.85", so the trunk was always 4.35 calls'
 * worth of work whatever was being folded. Measured, that ratio is 3.1 at 59
 * tokens and 15.3 at 256 - the trunk grows as L squared and the sampler barely
 * faster than linearly - so the bar raced through the trunk band and stalled on
 * anything long. src/runtime/cost-model.js has the fits and the measurements.
 */
function foldPlan({ tokens, rows, passes, calls, atoms }) {
  const plan = af3Plan({ tokens, rows, passes, calls, atoms });
  const at = (name) => plan.stages.find((stage) => stage.name === name);
  const features = at("features").units;
  const trunk = at("trunk").units * passes;
  const warmup = at("sampler-warmup").units;
  return {
    plan,
    estimator: new RuntimeEstimator(plan),
    // Where each phase's work ends, in units, so a partial phase can be placed.
    featuresEnd: features,
    trunkEnd: features + trunk,
    samplerStart: features + trunk + warmup,
    warmupUnits: warmup,
    callUnits: at("sampler").units,
  };
}

/**
 * Fold one chain with AlphaFold 3.
 *
 * @param {{sequence: string, mode: "flow"|"diffusion", calls: number, seed: number,
 *          signal: AbortSignal, device: GPUDevice,
 *          alignment?: string|{paired?: string|null, unpaired?: string|null}|null,
 *          maxMsaSequences?: number, ligandCodes?: string[],
 *          chainKinds?: ("protein"|"dna"|"rna")[],
 *          reuse?: {trunk: object, targetFeat: Float32Array},
 *          onTrunk?: (reusable: object) => void,
 *          onContacts?: (contactProbs: Float32Array, pass?: number,
 *                        passes?: number) => void,
 *          onPreview?: (preview: {pdb: string, pass: number,
 *                                 passes: number}) => void|Promise<void>,
 *          schedule?: {sigmaMax?: number, sigmaMin?: number, rho?: number},
 *          onStatus: (text: string) => void, onProgress: (fraction: number) => void,
 *          onFrame?: (pdb: string, index: number) => void}} options
 */
export async function foldAf3(options) {
  const { sequence, mode, calls, recycles, seed, signal, device, onStatus, onProgress } = options;
  // 🔴 THE MSA IS BUILT BEFORE THE WEIGHTS ARE TOUCHED, because its depth is a
  // shape: the MSA stack's pipelines are keyed on the row count, so a fold that
  // discovers its depth later would compile a second set of them.
  const alignment = options.alignment ?? null;
  const rows = alignment === null
    ? { msa: [], deletionMatrix: [], depth: 1, unpairedFrom: 0 }
    : af3MsaFromA3m(alignment, {
      maxSequences: options.maxMsaSequences,
      // 🔴 SEEDED FROM THE FOLD'S OWN SEED, so a subsample is part of what a
      // seed names. AF3 draws its shuffle from the same key that drives the
      // rest of the model; here two seeds are two alignments as well as two
      // starting draws, and one seed is reproducible.
      random: uniformFrom(options.seed ?? 0),
    });
  // 🔴 THE LIGAND DICTIONARY IS FETCHED, NOT BUNDLED. AF3's own featuriser
  // reads a 515 MB CCD pickle; a fold touches only the components its ligands
  // name, and the PDB serves each as one small mmCIF. The 21 polymer components
  // stay baked in reference-conformers.js, because every fold needs those.
  const componentCache = new Map();
  const component = async (code, what) => {
    if (!componentCache.has(code)) {
      onStatus(`Fetching ${what} ${code}`);
      let response;
      try {
        response = await fetch(ccdUrl(code), { signal });
      } catch (cause) {
        throw new Error(`Could not reach the PDB for ${what} ${code}`, { cause });
      }
      if (!response.ok) {
        throw new Error(`No chemical component ${code} at the PDB (${response.status})`);
      }
      componentCache.set(code, parseCcdComponent(await response.text()));
    }
    return componentCache.get(code);
  };
  // 🔴 A MODIFIED RESIDUE'S COMPONENT COMES FROM THE SAME PLACE AS A LIGAND'S,
  // and it is fetched here rather than in the featuriser because the featuriser
  // is synchronous - it is the one piece of a batch that cannot be computed
  // from the sequence alone.
  const modifications = [];
  for (const modification of options.modifications ?? []) {
    modifications.push({
      chain: modification.chain,
      position: modification.position,
      ...(await component(modification.code, "modified residue")),
    });
  }
  const ligands = [];
  for (const code of options.ligandCodes ?? []) {
    await component(code, "ligand");
    // Each instance is its own chain, so repeated codes are repeated entries -
    // featuriseProtein gives them one entity_id and successive sym_ids, which
    // is the same rule it applies to repeated sequences.
    ligands.push(componentCache.get(code));
  }

  const batch = featuriseProtein(sequence, {
    ligands,
    modifications,
    // What each chain's letters mean. Absent, every chain is protein, which is
    // what every caller before nucleic acids meant.
    chainKinds: options.chainKinds,
    msa: rows.msa,
    deletionMatrix: rows.deletionMatrix,
    unpairedFrom: rows.unpairedFrom,
    // The profile's rows, which are not the MSA's: AF3 computes the profile
    // before deduplicating the unpaired block against the paired one and before
    // cropping either. See af3MsaFromA3m.
    profileMsa: rows.profileMsa,
    profileDeletionMatrix: rows.profileDeletionMatrix,
  });
  // 🔴 ONLY THE PASSES THAT WILL ACTUALLY RUN, or the bar spends a share of
  // itself waiting for work that never happens and then jumps. A reused trunk
  // runs none; a continued one runs the difference.
  const passes = options.reuse === undefined
    ? (recycles ?? 0) + 1
    : Math.max(0, (recycles ?? 0) - options.reuse.recycles);
  // 🔴 THE CLOCK STARTS WHEN THE GPU DOES, NOT WHEN THE FOLD IS ASKED FOR.
  // Featurisation is a synchronous CPU call of a few hundred milliseconds that
  // the bar already shows as indeterminate, and it is not work this plan
  // models - so counting it as elapsed time makes the first ETA several times
  // too long, which is the estimate people actually read.
  // 🔴 BUILT ON FIRST USE, BECAUSE A REUSED TRUNK SKIPS THE FIRST STAGE.
  // Folding a second time in the same page runs no trunk passes at all, so
  // `target-feat` never fires - and a budget created only there is null when
  // the sampler asks it where to put the bar. It threw "Cannot read properties
  // of null" onto the status line, which is a fold that does not run.
  //
  // 🔴 AND THE CLOCK STARTS WITH IT, WHICH MEANS BEFORE FEATURISATION. The
  // estimator learns this machine's speed from time elapsed over units done, so
  // crediting featurisation's units without its three seconds makes every
  // estimate short - measured at 10-15 s remaining when 18-25 s remained. Time
  // and units have to cover the same work. That is why the first stage calls
  // this before doing anything, not after.
  let budget = null;
  const plan = () => (budget ??= foldPlan({
    tokens: batch.tokens, rows: rows.depth, passes, calls, atoms: batch.atomCount,
  }));
  /**
   * The status line: what is running, and how far in.
   *
   * 🔴 THREE WORDS FOR THE WHOLE FOLD: Preparing, Trunk, Folding. What went was
   * the SAMPLER's name - the line used to read "Refining" or "Diffusing"
   * depending on the mode, which is a distinction the mode dial already makes
   * and a reader watching a bar has no reason to care about.
   *
   * 🔴 TWO FIELDS, NOT SIX. It used to read "Trunk · pass 1 of 4 · pairformer
   * block 23 of 48", which is a number that changes forty-eight times a pass
   * next to two that barely move - so the eye tracks the one part that does not
   * matter. The percentage says the same thing about the whole fold, which is
   * the question being asked.
   *
   * 🔴 AND NO ESTIMATE. A time remaining has to be RIGHT to be worth reading,
   * and this one is a cost model against a machine that drifts up to 3.2x
   * between runs - so it moved around, and a number that moves around next to a
   * percentage that does not is the thing that made the line feel unsteady. The
   * percentage is the honest half; the model still drives the BAR, where being
   * approximately right is all a bar needs.
   */
  const say = (phase) => {
    if (budget === null) { onStatus(phase); return; }
    onStatus(`${phase} · ${Math.round(100 * budget.estimator.fraction())}%`);
  };
  /** Move the bar to a point in the plan, in units. */
  const reached = (units) => {
    const active = plan();
    active.estimator.completedUnits(units);
    onProgress(active.estimator.fraction());
  };
  const started = performance.now();

  const slots = alphaCarbons(batch);
  let reference = null;
  let shown = 0;
  // The trajectory, kept as coordinates rather than as text: the frames have to
  // be rendered twice - once uncoloured while they are computed, and again with
  // the pLDDT that does not exist until the confidence head has run.
  const trajectory = [];
  // 🔴 THE LIVE COLOUR, BUILT THE MOMENT THE TRUNK EXISTS. `trunk-done` fires
  // before the sampler runs and carries the distogram, so the frames drawn
  // DURING the fold can be coloured too - they used to be handed a null
  // B-factor, which the pLDDT scheme paints as no confidence at all.
  //
  // 🔴 IT IS THE RAW SCORE, NOT THE CALIBRATED ONE, because there is no
  // finished structure to calibrate against yet. That is only usable because
  // the raw score sits near the pLDDT scale on its own - 8.1 points from the
  // real mean across the panel, within 7 on six of eight. It did not, until
  // MINIMUM_SEPARATION started being read; before that it ran ~30 points low
  // and a live trajectory would have been red until the fold landed.
  //
  // 🔴 SO THE LIVE FRAMES AND THE REPLAYED ONES DIFFER, AND THE REPLAY IS THE
  // ONE TO GET RIGHT. Measured on a 58-mer, live 13.9 53.1 68.6 70.7 71.7 ...
  // against a calibrated replay of 46.2 63.9 70.9 71.9 72.3 ... - they agree
  // from the third frame on and part company at the start, where calibration
  // lifts the low end. The alternative is to drop the calibration so the two
  // agree exactly, and it is worse: the timeline's LAST frame is the finished
  // structure's real pLDDT either way, so an uncalibrated trajectory would end
  // on a step - 76.3 to 96.6 on trp-cage - and that is the frame everyone
  // looks at. A difference between a transient live frame and its replay is
  // cheaper than a jump at the end of the animation.
  const previewPdbs = [];
  // 🔴 THE POSITIONS TOO, because the PDB written during the fold carries the
  // RAW estimate - there is no calibration until the fold has finished - and
  // the finished trajectory must be one scale. See below.
  const previewPositions = [];
  // ...and the distogram each preview belongs to, so a trunk frame can carry
  // the contact map of ITS pass rather than the last one's.
  const previewContacts = [];
  let latestContacts;

  const result = await foldBatch(device, batch, options.weights, {
    mode, steps: calls, recycles, seed, reuse: options.reuse,
    // 🔴 ONE STRUCTURE PER RECYCLE, WHILE THE TRUNK IS STILL RUNNING. See
    // fold.js: a single flow cycle against that pass's trunk is a real
    // backbone, and it is the only structure that exists before the sampler
    // starts. Written as a PDB here, like every other frame, so the page draws
    // it the same way - and coloured by the trunk's own agreement, which is
    // all the confidence there is at this point.
    onPreview: options.onPreview === undefined ? undefined : async (preview) => {
      if (reference === null) reference = toPoints(preview.positions, batch.tokens * batch.dense);
      const colour = null;
      const pdb = fittedPdb(batch, preview.positions, reference, slots, colour);
      // 🔴 KEPT, NOT JUST SHOWN. The previews are frames of the same
      // trajectory - the structure the model believed at each recycle - so
      // they are handed back with the rest and the replay rebuilds them in
      // order rather than throwing away the only record of the trunk.
      previewPdbs.push(pdb);
      previewPositions.push(Float32Array.from(preview.positions));
      previewContacts.push(latestContacts);
      await options.onPreview({ pdb, pass: preview.pass, passes: preview.passes });
      await yieldToBrowser();
    },
    // ...forwarded for probes that move the noise schedule. Unset for a page
    // fold, which is AF3's own for the sampler and 160 A for the flow.
    schedule: options.schedule,
    onStage: async (name, detail) => {
      throwIfAborted(signal);
      if (name === "recycle-done" && detail.trunk?.contactProbs !== undefined) {
        // 🔴 ONE PER PASS, so the map sharpens as the trunk recycles rather
        // than appearing once at the end. AF3 runs every recycle before the
        // sampler emits a single frame, so for the longest part of a fold this
        // is the only thing the model has to show.
        latestContacts = detail.trunk.contactProbs;
        options.onContacts?.(detail.trunk.contactProbs, detail.pass, detail.passes);
      }
      if (name === "trunk-done") {
        // ...handed up the moment it exists, so a caller can cache it before
        // anything downstream has had a chance to fail. See fold.js.
        options.onTrunk?.(detail.reusable);
        // 🔴 AND THE CONTACT MAP, WHICH IS READY BEFORE THE FIRST DENOISER
        // CALL. It used to reach the page only in the replay after the whole
        // fold, which is minutes late for something the trunk already knows -
        // the sampler has not run yet at this point.
        if (detail.trunk?.contactProbs !== undefined) {
          options.onContacts?.(detail.trunk.contactProbs);
        }
      }
      if (name === "target-feat-start") {
        plan();               // ...starts the clock; see the note on plan().
        // A sweep rather than a number: what is left here is one synchronous
        // CPU call, so there is still nothing to count - but it is now a few
        // hundred milliseconds rather than five seconds.
        // 🔴 BACK TO ZERO EXPLICITLY, because the bar it inherits is FULL. The
        // same element showed the weight download, which ends at 100%, and the
        // sweep that used to sit here hid the handover. Without either it would
        // stay full through featurisation and then appear to run backwards on
        // the trunk's first report.
        onProgress(0);
        onStatus("Preparing · 0%");
        // The yield is the point: what follows blocks the main thread for
        // seconds, so the line above has to be painted before it starts.
        await yieldToBrowser();
      }
      if (name === "target-feat") {
        reached(plan().featuresEnd);
        say("Trunk");
        await yieldToBrowser();
      }
      if (name === "pairformer-block") {
        // 🔴 THIS ONE IS THE YIELD, NOT THE REPORT. It fires when a block is
        // ENCODED, and sixteen are encoded in the time the device takes over
        // one - so it is awaited (a real macrotask, so the page can paint) and
        // says nothing. `pairformer-block-done` below is what it paints.
        await yieldToBrowser();
      }
      if (name === "pairformer-block-done") {
        // Each recycle is another whole trunk, so the trunk band is divided
        // between them rather than replayed.
        const done = (detail.pass + detail.completed / detail.total) / detail.passes;
        reached(plan().featuresEnd
          + (plan().trunkEnd - plan().featuresEnd) * done);
        say(detail.passes > 1 ? `Trunk ${detail.pass + 1}/${detail.passes}` : "Trunk");
      }
      if (name === "trunk-done") {
        reached(plan().trunkEnd);
        // 🔴 THE COMPILE HAS NO MILESTONES AND BLOCKS THE MAIN THREAD, so the
        // bar can neither be advanced nor animated through it: at 150 tokens
        // that is 4.9 s of a 27 s fold in which no timer fires and nothing
        // paints. A crawl on setInterval was tried and never ticked once.
        //
        // What CAN be done is to size the band correctly - which the plan now
        // does, so the jump lands where it should - and to say what is
        // happening before the thread goes away, with a sweeping bar rather
        // than a still one. Both need this paint to land first, which is what
        // the yield below the status line is for.
        // 🔴 THE BAR HOLDS ITS VALUE HERE, IT NO LONGER SWEEPS. This band is a
        // pipeline compile that blocks the main thread with no milestones, and
        // the sweep was meant to show it was alive - but a determinate bar
        // going indeterminate and back is a visible flip in the middle of a
        // fold, twice, and a bar that stops for four seconds already reads as
        // busy next to a status line that says so. Holding is steadier than
        // sweeping, and it keeps the bar monotonic from end to end.
        say("Folding");
        await yieldToBrowser();
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
      trajectory.push(Float32Array.from(denoised));
      options.onFrame?.(
        fittedPdb(batch, denoised, reference, slots, null), shown);
      shown += 1;
      reached(plan().samplerStart + plan().callUnits * step);
      // ...the sampler used to run its OWN clock here, from its own elapsed
      // time over its own steps. It was the only honest one on the page, and
      // only because a denoiser call is the one unit that repeats identically.
      // The estimator generalises exactly that idea to the whole fold.
      say(`Folding ${step}/${calls}`);
      // 🔴 YIELD, OR THE PAGE NEVER PAINTS. Every await in the sampler resolves
      // from a GPU callback, which is a microtask - so without a real task
      // boundary the status above is written and never drawn.
      await yieldToBrowser();
    },
  });

  // 🔴 THE VIEWER WANTS ONE pLDDT A RESIDUE AND THE HEAD GIVES ONE AN ATOM.
  // Taking the alpha carbon's is what AlphaFold 2's own per-residue pLDDT means
  // here, and it is the value the cartoon is coloured by.
  const plddt = slots.map((slot) => result.scores.plddt[slot]);

  // 🔴 EVERY FRAME TAKES THE FINISHED STRUCTURE'S REAL pLDDT, WHICH IS A
  // CONSTANT COLOUR ON A MOVING STRUCTURE. It is the honest one: the head runs
  // once, on the final sample, and there is no per-frame confidence here that
  // is a measurement rather than a guess. A distogram-derived stand-in used to
  // fill the gap and has been removed - see the note in HANDOFF.md for what it
  // scored and why it was not worth what it claimed.
  const frameScores = { coloured: () => result.scores.plddt };

  // 🔴 EVERY FRAME IS RE-EMITTED, because the frames drawn during the fold
  // carry a zero B-factor - the pLDDT scheme paints that the colour of no
  // confidence at all, which is a claim rather than a missing value. It used to
  // re-emit them all with the FINISHED structure's pLDDT, which is a constant
  // colour on a moving structure; each frame now carries its own, from the
  // distogram. See below.
  //
  // 🔴 AND THE FINAL STRUCTURE IS FITTED LIKE THE REST OF THEM. It used to be
  // appended straight from the sampler, which leaves it in whatever frame
  // randomAugmentation last rotated into - so the animation ran smoothly and
  // then jumped on its last frame.
  const framePdbs = trajectory.map(
    (positions) => fittedPdb(batch, positions, reference, slots, frameScores.coloured(positions)));
  // 🔴 AND THE PREVIEWS ARE RECOLOURED ON THE SAME SCALE. They were written
  // during the trunk with the RAW distogram estimate, which is not on pLDDT's
  // scale and reads HIGH: measured, a two-step preview showed 54.8 to 58.1
  // while the finished structure's real pLDDT was 54.1, so the roughest frames
  // in the animation looked as confident as the answer. The calibration only
  // exists once the confidence head has run, so the fix is to rebuild them
  // here rather than to colour them differently there.
  const previewFrames = previewPositions.map((positions) =>
    fittedPdb(batch, positions, reference, slots, frameScores.coloured(positions)));
  // ...and the finished structure keeps the REAL pLDDT, which is the one number
  // here that is a claim about the prediction rather than about the animation.
  const finalPdb = fittedPdb(batch, result.positions, reference, slots, result.scores.plddt);

  return {
    batch,
    // Handed back so the caller can re-sample without the trunk. See foldBatch.
    reusable: result.reusable,
    depth: rows.depth,
    // 🔴 THE TRUNK'S OWN CONTACT MAP, WHICH COSTS NOTHING TO HAND OVER. The
    // distogram head already computes P(d <= 8 A) for every pair and it is
    // already read back to the host - so the heatmap panel's `contact` map is
    // a reshape, not a computation, and unlike the PAE it exists the moment
    // the trunk does rather than after the confidence head.
    contactProbs: result.trunk.contactProbs,
    framePdbs,
    // One per recycle, in order, before the sampler's first frame - recoloured
    // on the finished fold's scale, so the whole animation is one ramp.
    previewPdbs: previewFrames,
    previewContacts,
    pdb: finalPdb,
    meanPlddt: result.meanPlddt,
    geometry: result.geometry,
    seconds: (performance.now() - started) / 1000,
    confidence: {
      plddt,
      meanPlddt: result.meanPlddt,
      ptm: result.ptm,
      // NaN for a single chain - there is no interface to score - and the card
      // shows a dash for it rather than a number.
      iptm: result.iptm,
      // ...and one ipTM per interface, which the pooled one averages away on
      // more than two chains. Empty for a monomer; the same number as `iptm`
      // for exactly two chains. See src/heads/tm-score.js.
      chainPairIptm: result.chainPairIptm,
      predictedAlignedError: result.scores.pae,
    },
  };
}
