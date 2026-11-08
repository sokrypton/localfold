/**
 * `foldBatch`'s contract (webgpu/af3/fold.js) on the WebGL2 fold, for a page without WebGPU.
 *
 * The same batch and the same host weights in, the same result out - positions, `scores` (pLDDT
 * per atom slot, PAE, the tm-adjusted PAE), mean pLDDT, pTM, ipTM and the per-chain and
 * per-interface scores, the trunk's contact map, the backbone geometry - so web/af3-model.js
 * `foldAf3` builds its PDB and confidence from either backend the same way. The fold is
 * `foldAf3Async`: it waits on a fence between trunk blocks and denoiser calls, so the page keeps
 * painting, and stops there when `signal` aborts.
 *
 * The alignment is drawn as webgpu/af3/fold.js draws it: each trunk pass reads its own rows of the pool
 * (`msaDrawRows`/`msaNativeRows`, shared/af3/msa-draw.js), so a deep search folds here as it does there.
 * 🔴 IT REFUSED ONE DEEPER THAN num_msa UNTIL 2026-11-08, and a real MMseqs2 search returns thousands of rows,
 * so Safari 16.3 - the browser this path exists for - refused most proteins: "the WebGL2 fold does not draw
 * from an alignment (8077 rows)". Every gate had folded single sequences or a 4-row search.
 *
 * 🔴 WHAT IT DOES NOT CARRY: a structural-token model (OpenDDE, not ported), and the WebGPU path's extras
 * (trunk reuse, steering, motifs, warm starts) - each refused by name.
 */
import { createWebGL2Context } from "../../runtime/context.js";
import { Execution } from "../../runtime/execution.js";
import { foldAf3Async, releaseAf3Weights, uploadAf3WeightsSteps } from "./fold.js";
import { drainAsync, gpuIdle } from "../../runtime/steps.js";
import { gatherRows, msaDrawRows, msaNativeRows } from "../../../shared/af3/msa-draw.js";
import { af3TmScores, chainTmInfo, interfaceTmTerm, tmPerBinFor, tmScoreD0, tmTermFromLogits, tokenFrameMask }
  from "../../../shared/heads/tm-score.js";
import { errorBinCentres, MAX_ERROR_BIN, NUM_BINS } from "../confidence/confidence.js";
import { backboneGeometry } from "../../../webgpu/af3/fold.js";

/** Whether this browser can run the WebGL2 fold: WebGL2 that renders to float textures. */
export function webgl2FoldAvailable() {
  try {
    const gl = document.createElement("canvas").getContext("webgl2");
    return gl !== null && gl.getExtension("EXT_color_buffer_float") !== null;
  } catch {
    return false;
  }
}

/**
 * The AF3 fold's texture pool: on, and bounded, with the least recently used shapes evicted first
 * (`Storage.poolLimitBytes`).
 *
 * 🔴 THIS FOLD IS NOT WAITING ON ARITHMETIC. Drawing every 59-row diffusion projection TWICE (`--double`, 8,714 extra
 * draws) moved a 59-residue fold by nothing, and creating a fresh texture for every tensor did: pooled, 9.9 s -> 7.7 s
 * at 59 residues and 79.8 -> 65.4 s at 200, bit-identical. Unbounded that cost 1225 -> 1807 MiB of peak at 200, the
 * growth that made pooling a loss on bioemu2; bounded at 128 MiB it is 68.2 s at 1244 MiB (Chrome, M2,
 * tools/gpu/fold-af3-webgl.js). `--no-pool` on that tool is the control arm.
 */
export const AF3_POOL_LIMIT_BYTES = 128 * 2 ** 20;

/**
 * The smallest weight the AF3 fold holds in float16 (`Storage.halfWeightFloor`), where the backend's default is 65,536.
 *
 * 🔴 EVERY PAIR PROJECTION IN AN AF3 TRUNK WAS UNDER THE OLD FLOOR - 128 x 128 and 128 x 256 are 16 K and 32 K
 * floats - so they streamed float32 weights, and `probe-webgl-packed-x.js` reads a projection ~1.5x faster with an f16
 * weight even when it sits in cache. 59 residues, 8076-row a3m at 128 rows, Chrome on an M2: 5.55 s -> 5.27 at 4096
 * (trunk 1.95 -> 1.75), peak 971 -> 926 MiB, against WebGPU's fold relative RMS 7.7e-4 -> 6.5e-4; 256 is no faster.
 * WebGPU holds the same weights in f16.
 */
export const AF3_HALF_WEIGHT_FLOOR = 4096;

// One context for the page, and the last family's weights on it: a second family's upload
// replaces the first's (each is ~800 MB as halves), the same family's is kept.
let execution = null;
let resident = null; // { weights, ready: Promise<uploaded> }

/** The weights on the GPU, uploaded a module at a time with the thread let go in between. */
function uploadedFor(weights) {
  // The configuration fold-af3-webgl.js gates against the WGSL fold: half weights and activations, a bounded pool.
  execution ??= new Execution(createWebGL2Context(),
    { halfWeights: true, halfActivations: true, int5Weights: true,
      pooling: true, poolLimitBytes: AF3_POOL_LIMIT_BYTES, halfWeightFloor: AF3_HALF_WEIGHT_FLOOR });
  if (resident?.weights !== weights) {
    const previous = resident;
    resident = {
      weights,
      ready: (async () => {
        if (previous !== null) releaseAf3Weights(execution, await previous.ready);
        return drainAsync(uploadAf3WeightsSteps(execution, weights), () => gpuIdle(execution.gl));
      })(),
    };
  }
  return resident.ready;
}

/**
 * The fold's alignment as `foldAf3` takes it: each pass's rows, as webgpu/af3/fold.js `msaRowsOf` draws them - up
 * to num_msa (1024) of the pool, a fresh draw every pass; the pool's first rows for a batch read from a dump (no
 * `numMsa`) and for chai-1; protenix's own count, padded with masked rows up to a power of two. An alignment that
 * fits whole is the same every pass, so it is one set of rows uploaded once rather than a function of the pass.
 *
 * @param {object} batch the featurised batch
 * @param {object} dialect `weights.trunk.dialect`
 * @param {{seed?: number, numMsa?: number, msaDraw?: string}} options
 */
export function foldMsa(batch, dialect, options = {}) {
  const { tokens } = batch;
  const pool = batch.msa.length / tokens;
  const cap = Math.min(options.numMsa ?? 1024, batch.numMsa ?? Infinity);
  const drawMode = batch.numMsa === undefined || dialect.chaiMsaFeatures === true
    ? "prefix" : (options.msaDraw ?? "pass");
  const native = dialect.msaNativeSample === true && batch.numMsa !== undefined
    && options.msaDraw !== "prefix";
  const drawSeed = options.seed ?? 20260831;
  const msaOfPass = (pass) => {
    const chosen = native ? msaNativeRows(drawSeed, options.msaDraw === "once" ? 0 : pass, pool)
      : msaDrawRows(drawMode, drawSeed, pass, pool, cap);
    const rows = native ? Math.min(pool, 2 ** Math.ceil(Math.log2(chosen.length))) : chosen.length;
    const widen = (array, pad) => {
      const out = new Float32Array(rows * tokens).fill(pad);
      out.set(gatherRows(array, tokens, chosen));
      return out;
    };
    return { codes: widen(batch.msa, 21), deletions: widen(batch.deletionMatrix, 0),
             mask: widen(batch.msaMask, 0), sequences: rows };
  };
  return !native && pool <= cap ? msaOfPass(0) : msaOfPass;
}

/**
 * Upload a family's weights to the WebGL2 context now, ahead of the fold that will want them (the same memoised upload
 * `foldBatchWebGl2` awaits, so that fold finds it done).
 *
 * 🔴 THE PAGE DOWNLOADS THE WEIGHTS BESIDE THE MMseqs2 SEARCH AND USED TO UPLOAD THEM AFTER IT. The search is seconds of
 * waiting on a server with the GPU idle, and on Safari 16.3 the upload - decoding the int5 bundle, narrowing to float16,
 * texSubImage2D - is ~1.6 s of a first fold that otherwise waits on nothing. web/app.js starts this when the download
 * lands and the backend is known to be WebGL2.
 */
export function preloadWebGl2Weights(weights) {
  return uploadedFor(weights);
}

/**
 * @param {object} batch the featurised batch
 * @param {object} weights web/af3-model.js `loadAf3Weights`'s tree
 * @param {{steps: number, recycles?: number, seed?: number, templateSlots?: object[],
 *   signal?: AbortSignal, onStep?: (frame: {step: number, denoised: Float32Array}) => void,
 *   onStage?: (name: string, detail: object) => void}} options
 */
export async function foldBatchWebGl2(batch, weights, options) {
  if (weights.trunk.dialect.structuralTokens === true) {
    throw new Error("this model re-tokenises after the trunk (OpenDDE), which the WebGL2 fold has not ported");
  }
  for (const key of ["reuse", "steer", "motif", "start"]) {
    if (options[key] !== undefined) throw new Error(`the WebGL2 fold does not take \`${key}\``);
  }
  const { tokens, dense } = batch;
  const uploaded = await uploadedFor(weights);
  const fold = await foldAf3Async(execution, {
    batch, weights: uploaded, hostWeights: weights,
    msa: foldMsa(batch, weights.trunk.dialect, options),
    recycles: options.recycles ?? 0, steps: options.steps, seed: options.seed ?? 42,
    templateSlots: options.templateSlots,
    signal: options.signal,
    onStep: options.onStep,
    onStage: options.onStage,
  });

  const { positions } = fold;
  const { seqMask, asymId } = batch;
  let scores;
  let ptm;
  let iptm;
  let chainPairIptm;
  let chainPtm;
  let chainIptm;
  let meanPlddt;
  if (fold.paeLogits !== undefined) {
    // As WebGPU's head computes it: one d0 over every token, the expectation over the PAE bins.
    const tmPerBin = tmPerBinFor(errorBinCentres(NUM_BINS, MAX_ERROR_BIN), tmScoreD0(tokens));
    const tmAdjusted = tmTermFromLogits(fold.paeLogits, tokens, tmPerBin);
    // ...and AF3's interface term for the chain-level scores (d0 from the pair's own chains), anchored, as pTM is,
    // on the tokens with frames - webgpu/af3/fold.js's reduction (reported by Yeqing Lin)
    const tmInterface = interfaceTmTerm(fold.paeLogits, tokens,
      chainTmInfo(asymId, seqMask, errorBinCentres(NUM_BINS, MAX_ERROR_BIN)));
    scores = { plddt: fold.plddt, pae: fold.pae, tmAdjusted, tmInterface };
    const frame = tokenFrameMask({ tokens, dense, isLigand: batch.isLigand, asymId,
      residueIndex: batch.residueIndex, refPos: batch.refPos, refMask: batch.refMask });
    // ...reduced by the one function webgpu/af3/fold.js reduces through, so the
    // two backends cannot drift apart on what the five numbers MEAN - only on
    // the arithmetic that produced the terms.
    ({ ptm, iptm, chainPairIptm, chainPtm, chainIptm } = af3TmScores({
      tmAdjusted, tmInterface, tokens, asymId, seqMask, frame,
    }));
    let total = 0;
    let atoms = 0;
    for (let index = 0; index < tokens * dense; index += 1) {
      if (!batch.predDenseAtomMask[index]) continue;
      total += fold.plddt[index];
      atoms += 1;
    }
    meanPlddt = total / atoms;
  }
  return {
    positions, scores, ptm, iptm, chainPairIptm, chainPtm, chainIptm, meanPlddt,
    trunk: { contactProbs: fold.contactProbs },
    reusable: undefined,
    geometry: backboneGeometry(batch, positions),
  };
}
