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
import { isAnnotatedCif } from "../shared/rfd4p/pipeline/spec.js";
import { ccdBlocksForRawStructure } from "../shared/rfd4p/pipeline/ccd.js";
import { fetchRfd4pConfig, fetchShardedWeights } from "../shared/rfd4p/safetensors.js";
import { generateRfd4p } from "../webgpu/rfd4p/generate.js";
import { processSampleToAtoms, toPdbString } from "../shared/rfd4p/output.js";
import { conditioningMetricsTags, designMetrics, jobMetricsTags } from "../shared/rfd4p/metrics.js";
import { decompress } from "../shared/rfd4p/vendor/fzstd.js";
import { resolveSampling } from "../shared/rfd4p/flow.js";

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
 * @param {object} [options.sampling] the samplers over the released ones, in upstream's config
 *   shape (`bb_ca` / `local_latents`, each `simulation_step_params`, `schedule`, `gt`) - what
 *   a job's `sampling` is (shared/rfd4p/flow.js `resolveSampling`)
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
 * @param {string[]} [options.metricsTags] upstream's `metrics_tags`: which of its design metrics
 *   and pre-oracle filters apply (shared/rfd4p/metrics.js). Default: `design` and the tags
 *   upstream's validation sets give this job's conditioning (`conditioningMetricsTags`), so a
 *   motif's RMSD is reported where there is a motif
 * @returns {Promise<{pdb: string, cif: string, tokens: number, steps: number, seed: number,
 *   job: object, inputCif: (string|Uint8Array|undefined), metrics: object, preOracle: object}>}
 *   `job` is the job that ran as a job file (`input` naming the structure, `inputCif`, beside it)
 */
export async function generate(options) {
  const {
    contig, conditions, cifUrl, inputCif: givenCif, fromAnnotations = false,
    steps = DEFAULT_STEPS, recycleSchedule, sampling, seed = 0,
    partialTrajectoryFractionSteps, inverseFolding, metricsTags: givenTags,
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

  // Resolved before any work, so a bad setting is refused in upstream's words, not minutes later.
  const resolvedSampling = sampling === undefined ? undefined : resolveSampling(sampling);

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
  // 🔴 A STRUCTURE UNDER A USER-BUILT JOB IS READ AS UPSTREAM READS A SPEC'S `input:`. An
  // annotated CIF (a condition spec's output, as the 1PRW examples are) keeps its conditions -
  // upstream's `input_as_annotated_cif: true`, which a page decides from the file - and anything
  // else is a raw structure, PREPARED (pipeline/fresh.js): missing atoms added, waters dropped,
  // its non-standard components completed from dictionary entries fetched first. Upstream refuses
  // the 1PRW crops as raw structures, for the expandable-segment sentinels inside their chains.
  const annotated = inputCif !== undefined && isAnnotatedCif(inputCif);
  const ccdBlocks = !fromAnnotations && inputCif !== undefined && !annotated
    ? await ccdBlocksForRawStructure(inputCif) : undefined;
  const { batch, n } = fromAnnotations && inputCif !== undefined
    ? buildBatch(inputCif, { seed })
    : buildBatch({
      contig,
      ...(conditions === undefined ? {} : { conditions }),
      ...(annotated ? { input_as_annotated_cif: true } : {}),
    }, { seed, ...(inputCif === undefined ? {} : { inputCif, ccdBlocks }) });

  status(`generating ${n} tokens over ${steps} steps`);
  const result = await generateRfd4p(device, weights, cfg, batch, {
    nsteps: steps,
    seed,
    signal,
    // 🔴 THE PAGE HAD NEVER ASKED FOR f16, SO THE ONE CONFIGURATION NOBODY RAN WAS
    // THE ONE THAT SHIPPED. Every tool in `tools/gpu/` defaults to `half`, so each
    // number recorded for this port describes an arm a visitor never took, and the
    // f32 arm is 1.56x slower at n = 200 and about 2.6x at n = 400 - the pair track
    // narrows, so `chunkPlan` also keeps one chunk to n = 512 instead of 362 and a
    // 400-residue design stops falling onto the chunked kernels entirely.
    //
    // 🔴 AND IT IS A DIFFERENT STRUCTURE, KNOWINGLY: `tools/gpu/check-rfd4p-length.js`
    // measures the two arithmetics at n = 200 over 60 steps at one seed at **mean CA
    // 2.687 A, worst 7.935**, against the 0.5 A `tools/gpu/check-generate.js` holds
    // its own half arm to at the oracle's length. Both arms produce a well-formed
    // chain (medians 3.87 and 3.86, nothing torn), so this is the SAME DISTRIBUTION
    // SAMPLED DIFFERENTLY - a per-pass difference of order 1e-3 compounded over 120
    // passes of a chaotic trajectory, which is what `cuda/rfd4p`'s own notes say from
    // the other side: "one ulp in one draw diverges a sixty-step trajectory". The
    // design is not the f32 arm's design, and that is the trade taken here: a
    // generator draws from a distribution, and the speed is what makes a reader try
    // another seed. 🔴 WHAT WOULD MAKE IT MORE THAN A TRADE is an oracle dump at
    // 200-400 tokens (there is none) or an independent re-fold of both designs.
    //
    // 🔴 AND THE COMMENT ABOVE CAME TRUE AGAIN ONE LEVEL DOWN, WHICH IS WHY IT IS
    // STILL HERE. Asking for `half` is what put a visitor on the CHUNKED f16 pair
    // track past n = 512, and that arm was WRONG on a stock Chrome - relRMS 0.43 on
    // the alpha carbons against the single-chunk arm - while passing on the
    // subgroup-matrix flags every tool in `tools/gpu/` is run with. Two stock-only
    // kernel gaps, fixed in a821bd1f, found by an M2 running
    // `tools/gpu/check-rfd4p-chunked.js` under `LOCALFOLD_STOCK_FLAGS=1`. So the
    // numbers quoted above are developer-flag numbers, and the arm this line selects
    // is now held by that gate under BOTH flag sets - stock 1.3e-4 to 2.9e-4,
    // developer 6.6e-4, over even and ragged plans. A design through the ragged
    // 4-chunk f16 path on stock flags, n = 200 at 60 steps: median CA-CA 3.864 A,
    // 199 of 199 bonds in band.
    half: true,
    ...(resolvedSampling === undefined ? {} : { sampling: resolvedSampling }),
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
  // 🔴 THE JOB THAT RAN, AS A JOB FILE THE NATIVE PORT TAKES (`localfold-rfd4p --job=`), so a
  // design downloaded from this page can be made again without it. A preset driven by its
  // structure is that structure as the job (upstream's `{input, input_as_annotated_cif: true}`);
  // anything else is its contig and conditions over the structure it named.
  const metricsTags = givenTags ?? ["design", ...conditioningMetricsTags(batch.__atoms)];
  const job = {
    ...(inputCif === undefined ? {} : { input: "input" }),
    ...(fromAnnotations && inputCif !== undefined
      ? { input_as_annotated_cif: true }
      : {
        contig,
        ...(conditions === undefined ? {} : { conditions }),
        ...(annotated ? { input_as_annotated_cif: true } : {}),
      }),
    nsteps: steps,
    ...(recycleSchedule === undefined ? {} : { recycleSchedule }),
    ...(sampling === undefined ? {} : { sampling }),
    ...(partialTrajectoryFractionSteps === undefined ? {} : { partialTrajectoryFractionSteps }),
    ...(inverseFolding === true ? { inverseFolding: true } : {}),
    ...(metricsTags.length === 0 ? {} : { metrics_tags: metricsTags }),
  };
  // 🔴 THE DESIGN'S METRICS, AS UPSTREAM'S EVALUATION COMPUTES THEM, under the tags the job
  // records - so `localfold-rfd4p --job=job.json --metrics=` reports the same numbers. Upstream's
  // generator computes none unless asked; the page asks for `design`, its own inline set, and the
  // conditioning tags its validation sets would give this job.
  const { metrics, preOracle } = designMetrics(result.structure, batch.__atoms,
    { metricsTags: jobMetricsTags(job) });
  return {
    pdb: result.pdb, cif: result.cif, tokens: n, steps: result.stepsRun ?? steps, seed, job, inputCif,
    metrics, preOracle,
  };
}
