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
 * @param {number} [options.steps]
 * @param {string} [options.recycleSchedule]
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
    contig, conditions, cifUrl, steps = DEFAULT_STEPS, recycleSchedule, seed = 0,
    status = () => {}, progress = () => {}, onStep, signal,
  } = options;

  if (typeof contig !== "string" || contig.trim().length === 0) {
    throw new Error("a generation needs a contig, e.g. \"100-100\" for an unconditional monomer");
  }
  // 🔴 REFUSED HERE RATHER THAN DEEP IN THE PIPELINE. A contig naming a chain
  // segment with no structure to take it from fails somewhere inside the
  // validation pipeline with a message about an absent chain; said here it names
  // the thing the user has to supply.
  const needsTemplate = /(^|,)\s*[A-Za-z]\d/.test(contig);
  if (needsTemplate && cifUrl === undefined) {
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
  const inputCif = cifUrl === undefined ? undefined : await templateCif(cifUrl, signal);
  const spec = { contig, ...(conditions === undefined ? {} : { conditions }) };
  const { batch, n } = buildBatch(spec, { seed, ...(inputCif === undefined ? {} : { inputCif }) });

  status(`generating ${n} tokens over ${steps} steps`);
  const result = await generateRfd4p(device, weights, cfg, batch, {
    nsteps: steps,
    seed,
    signal,
    ...(recycleSchedule === undefined ? {} : { recycleSchedule }),
    onStep: (step) => {
      // The remaining 30% of the bar is the flow itself.
      const index = step?.step ?? step?.i ?? 0;
      progress(0.7 + 0.3 * Math.min(1, (index + 1) / steps));
      status(`generating · step ${index + 1}/${steps}`);
      if (onStep !== undefined) onStep(step);
    },
  });

  progress(1);
  return { pdb: result.pdb, cif: result.cif, tokens: n, steps: result.stepsRun ?? steps };
}
