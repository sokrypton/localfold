/**
 * generate.js
 * -----------
 * AI Context: RFD4-PROTEINA ON THIS PAGE — the second method behind the picker
 *
 * The `hunt.js` of the other method. `webgpu/rfd4p/` is the engine; this is the
 * half that feeds it: the config, the weights, a batch built from a contig, and
 * the progress reporting a long generation needs.
 *
 * 🔴 A DIFFERENT MACHINE, NOT A DIFFERENT SETTING. Protein Hunter folds a
 * mostly-X sequence and redesigns it in a loop; RFD4-Proteina generates all-atom
 * structure by flow matching in one pass. They share this page, the object list
 * and the viewer - not a code path. So this file is a sibling of `hunt.js` and
 * borrows nothing from it but its shape.
 *
 * 🔴 THE WEIGHTS ARE ~420 MB AND COME FROM ANOTHER REPOSITORY. The chain is a
 * local export → `sokrypton/rfd4-proteina-webgpu`'s Pages URL → Hugging Face
 * (`shared/rfd4p/safetensors.js`), so the first generation on a fresh page is a
 * long download and every later one is served from the Cache API. The page says
 * which of those is happening rather than looking stalled.
 *
 * 🔴 AND ONE DEVICE PER PAGE. RFD4 ships its own `requestWebGpuDevice`, which
 * asks an adapter for its own limits raised to the maximum; this page uses
 * LocalFold's `getDevice` instead, because two WebGPU devices on one page means
 * two copies of everything resident and the fold path already owns one.
 */
import { getDevice } from "./model.js";
import { buildBatch } from "../shared/rfd4p/pipeline/index.js";
import { fetchRfd4pConfig, fetchShardedWeights } from "../shared/rfd4p/safetensors.js";
import { generateRfd4p } from "../webgpu/rfd4p/generate.js";
import { processSampleToAtoms, toPdbString } from "../shared/rfd4p/output.js";
import { decompress } from "../shared/rfd4p/vendor/fzstd.js";

/** How many flow-matching steps a generation takes when the caller does not say. */
export const DEFAULT_STEPS = 60;

let cached = null;

/**
 * The config and weights, once per page.
 *
 * 🔴 MEMOISED HERE RATHER THAN RELOADED, because the download is the expensive
 * part by orders of magnitude and a page may generate many times. The fold
 * path's `loadAf3Weights` memoises in its own module for the same reason.
 */
async function loadRfd4p({ status, progress, signal }) {
  if (cached !== null) return cached;
  status("reading RFD4-Proteina's configuration");
  const cfg = await fetchRfd4pConfig();
  status("downloading RFD4-Proteina's weights (~420 MB the first time, cached after)");
  const { tensors, config } = await fetchShardedWeights(null, {
    includeEncoder: true,
    signal,
    onProgress: (value) => {
      const total = value?.totalBytes ?? 0;
      const loaded = value?.loadedBytes ?? 0;
      if (total > 0) progress(loaded / total);
    },
  });
  cached = { weights: tensors, cfg: config ?? cfg };
  return cached;
}

/** A preset's template structure, decompressed if it is `.zst`. */
async function templateCif(url, signal) {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  // 🔴 SNIFFED, NOT TAKEN FROM THE EXTENSION. The examples ship `.cif.zst` but a
  // user's own upload may be either, and a plain CIF run through a zstd decoder
  // fails with a message about frame magic that says nothing about CIF.
  const zstd = bytes[0] === 0x28 && bytes[1] === 0xb5 && bytes[2] === 0x2f && bytes[3] === 0xfd;
  return new TextDecoder().decode(zstd ? decompress(bytes) : bytes);
}

/**
 * Generate one structure.
 *
 * @param {object} options
 * @param {string} options.contig the RFdiffusion-style contig, e.g.
 *   `"A18-132,/0,76-76"` - a given segment, a chain break, a generated span
 * @param {object} [options.conditions] the condition spec (`C_CRD`, `C_SEQ`,
 *   `C_HOT`, ...) as the presets carry it
 * @param {string} [options.cifUrl] a template structure the contig refers to;
 *   required whenever the contig names a given segment like `A18-132`
 * @param {boolean} [options.fromAnnotations] drive the batch from the
 *   structure's own condition annotations rather than from `contig`. This is
 *   what a preset without `specMode` means, and it is not interchangeable - see
 *   the note at `buildFrom` below.
 * @param {string} [options.inputCif] the same thing as text, for a structure the
 *   user loaded rather than one a preset points at. Wins over `cifUrl`.
 * @param {number} [options.steps]
 * @param {string} [options.recycleSchedule]
 * @param {number} [options.partialTrajectoryFractionSteps] partial diffusion:
 *   start the flow partway along, from the input structure rather than from
 *   noise, and re-denoise the rest. 0.5 is the preset's.
 * @param {boolean} [options.inverseFolding] hold the backbone and resample only
 *   the sequence
 * @param {number} [options.seed=0]
 * @param {(text: string) => void} [options.status]
 * @param {(fraction: number) => void} [options.progress]
 * @param {(step: object) => void} [options.onStep] called per flow step, for a
 *   trajectory the page can show
 * @param {AbortSignal} [options.signal]
 * @returns {Promise<{pdb: string, cif: string, tokens: number, steps: number}>}
 */
export async function generate(options) {
  const {
    contig, conditions, cifUrl, inputCif: givenCif, fromAnnotations = false,
    steps = DEFAULT_STEPS, recycleSchedule, seed = 0,
    partialTrajectoryFractionSteps, inverseFolding,
    status = () => {}, progress = () => {}, onStep, frames = 0, signal,
  } = options;

  if (typeof contig !== "string" || contig.trim().length === 0) {
    throw new Error("a generation needs a contig, e.g. \"100-100\" for an unconditional monomer");
  }
  // 🔴 TWO MODALITIES NEED THE INPUT STRUCTURE EVEN WITH NO GIVEN SEGMENT, and
  // saying so here is the difference between a clear refusal and a generation
  // that quietly ignores what was asked. Partial diffusion starts the flow
  // partway along FROM that structure; inverse folding holds its backbone and
  // resamples the sequence. Both are meaningless without one.
  // 🔴 `null` IS NOT `undefined`, AND ONE PRESET SHIPS IT. custom_cath_monomer's
  // cifUrl is `null` - it is pure spec, no structure - and a `=== undefined`
  // test let that through to fetch("null"), which 404s with a message naming no
  // file at all.
  const templateUrl = cifUrl === null ? undefined : cifUrl;
  const haveStructure = givenCif !== undefined || templateUrl !== undefined;
  if (partialTrajectoryFractionSteps !== undefined && !haveStructure) {
    throw new Error("partial diffusion starts from an existing structure and re-denoises it,"
      + " so it needs one to start from");
  }
  if (inverseFolding === true && !haveStructure) {
    throw new Error("inverse folding holds a backbone and resamples its sequence,"
      + " so it needs the backbone");
  }
  // 🔴 REFUSED HERE RATHER THAN DEEP IN THE PIPELINE. A contig naming a chain
  // segment with no structure to take it from fails somewhere inside the
  // validation pipeline with a message about an absent chain; said here it names
  // the thing the user has to supply.
  // 🔴 A LABELLED SPAN IS NOT A CHAIN SEGMENT, and a regex looking for
  // "letter then digit" cannot tell them apart: `h1:16-16P` starts exactly like
  // `A18-132`. The difference is the COLON - a label names the span, a segment
  // names a chain - so each piece is tested whole.
  const needsTemplate = contig.split(",").some((piece) => {
    const token = piece.trim();
    if (token.includes(":")) return false;              // a labelled span
    return /^[A-Za-z][A-Za-z0-9]*-?\d+-\d+$/.test(token) && /^[A-Za-z]/.test(token);
  });
  if (needsTemplate && !haveStructure) {
    throw new Error(`the contig "${contig.trim()}" names a segment of an existing structure,`
      + " so it needs one to take it from - load an example or add a structure");
  }

  const device = await getDevice(signal);
  // Weight loading owns the first 70% of the bar: it is minutes on a cold
  // page and seconds on a warm one, and the steps after it are quick.
  const { weights, cfg } = await loadRfd4p({
    status, signal, progress: (fraction) => progress(0.7 * fraction),
  });

  status("preparing the input");
  const inputCif = givenCif ?? (templateUrl === undefined
    ? undefined : await templateCif(templateUrl, signal));
  // 🔴 TWO WAYS TO BUILD A BATCH, AND THEY ARE NOT INTERCHANGEABLE.
  //
  //   - FROM A SPEC: a contig plus conditions, with a structure as a template.
  //     This is what a user-built job is, and what a preset marked `specMode`
  //     is.
  //   - FROM AN ANNOTATED STRUCTURE: the mmCIF carries `condition_coordinate`,
  //     `condition_sequence`, `condition_index`, `expsegmin`/`expsegmax` per
  //     residue, and those ARE the job. A preset without `specMode` means this.
  //
  // 🔴 AND THE SHIPPED EXAMPLES ONLY WORK THE SECOND WAY, which is how this was
  // found: `motif_idx_1prw`'s contig asks for `A52-71`, its own example CIF has
  // chain A at residues 0-42, and building from the spec fails with "Preserved
  // segment 'A52-71' matched no atoms". The contig in those presets is written
  // in the ORIGINAL 1PRW numbering and is a label, not an input. Running them
  // from the spec was wrong for every conditioned preset.
  const { batch, n } = fromAnnotations && inputCif !== undefined
    ? buildBatch(inputCif, { seed })
    : buildBatch({ contig, ...(conditions === undefined ? {} : { conditions }) },
      { seed, ...(inputCif === undefined ? {} : { inputCif }) });

  status(`generating ${n} tokens over ${steps} steps`);
  const result = await generateRfd4p(device, weights, cfg, batch, {
    nsteps: steps,
    seed,
    signal,
    ...(recycleSchedule === undefined ? {} : { recycleSchedule }),
    ...(partialTrajectoryFractionSteps === undefined
      ? {} : { partialTrajectoryFractionSteps }),
    // 🔴 INVERSE FOLDING AND PARTIAL DIFFUSION BOTH NEED THE ENCODER, which is
    // a separate ~28 MB of shards the loader can skip. They read the input
    // structure into the model's latent rather than starting from noise, and
    // that is what the encoder is for.
    ...(inverseFolding === true ? { inverseFolding: true } : {}),
    withEncoder: inverseFolding === true || partialTrajectoryFractionSteps !== undefined,
    inputAtoms: inputCif === undefined ? undefined : batch.__atoms,
    // 🔴 A LIVE FRAME IS A DECODER PASS, SO IT IS ASKED FOR BY COUNT, NOT BY
    // STEP. `decodeEvery` makes the engine run the all-atom decoder on that
    // cadence - real GPU work on top of the flow - so a 60-step generation
    // decoded every step would pay 60 decodes to draw 60 nearly identical
    // frames. `frames` is how many the caller wants to WATCH, and the cadence
    // falls out of it; 0 turns the whole thing off and nothing is decoded.
    ...(frames > 0 ? { decodeEvery: Math.max(1, Math.round(steps / frames)) } : {}),
    onStep: (step) => {
      // The remaining 30% of the bar is the flow itself.
      const index = step?.step ?? step?.i ?? 0;
      progress(0.7 + 0.3 * Math.min(1, (index + 1) / steps));
      status(`generating · step ${index + 1}/${steps}`);
      // 🔴 AND THE CALLER IS HANDED A PDB, NOT THE RAW DECODE. `step.decoded`
      // is `formatSampleLocalLatents`'s {coors, atomMask, residueType}, which
      // is two transforms short of something a viewer can draw - and those are
      // exactly the two the FINAL structure goes through, so doing them here
      // keeps one path rather than letting the page reimplement the tail of
      // the pipeline.
      let pdb;
      if (step?.decoded != null) {
        try {
          pdb = toPdbString(processSampleToAtoms({
            coors: step.decoded.coors,
            atomMask: step.decoded.atomMask,
            residueType: step.decoded.residueType,
            paddingMask: step.paddingMask,
            insertionLogits: step.insertionLogits ?? null,
            inputAtoms: inputCif === undefined ? undefined : batch.__atoms,
            batch,
          }));
        } catch {
          // A frame that cannot be drawn is not a reason to lose the run; the
          // final structure comes from its own call below.
          pdb = undefined;
        }
      }
      if (onStep !== undefined) onStep({ ...step, pdb });
    },
  });

  progress(1);
  return { pdb: result.pdb, cif: result.cif, tokens: n, steps: result.stepsRun ?? steps };
}
