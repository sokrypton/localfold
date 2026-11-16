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
import { drainAsync, laggedPause } from "../../runtime/steps.js";
import { gatherRows, msaDrawRows, msaNativeRows } from "../../../shared/af3/msa-draw.js";
import { af3TmScores, chainTmInfo, tmPerBinFor, tmScoreD0, tmTermsFromLogits, tokenFrameMask }
  from "../../../shared/heads/tm-score.js";
import { errorBinCentres, MAX_ERROR_BIN, NUM_BINS } from "../confidence/confidence.js";
import { backboneGeometry } from "../../../webgpu/af3/fold.js";

/** Whether this browser can run the WebGL2 fold: WebGL2 that renders to float textures. */
/**
 * The renderer's name where this browser's WebGL2 is drawn on the CPU, else null.
 *
 * 🔴 A SOFTWARE RENDERER FOLDS HUNDREDS OF TIMES SLOWER, AND TO A READER IT LOOKS LIKE A HANG. Firefox on Linux
 * draws WebGL on Mesa's llvmpipe when it has no usable GPU driver: a 4-block, 8-step AlphaFold 3 prefix that is
 * 1.3 s on a GPU ran past ten minutes there at one core (docs/FIREFOX.md, "Linux"), and the page's own fold is
 * twelve times that work. So the page refuses with the renderer's name rather than start (`foldBackend`).
 * Firefox sanitises the name ("llvmpipe, or similar") and keeps the word that matters.
 */
export const SOFTWARE_RENDERER = /llvmpipe|lavapipe|softpipe|swiftshader|software|basic render/i;

export function webgl2SoftwareRenderer(gl) {
  const debug = gl.getExtension("WEBGL_debug_renderer_info");
  const name = String(gl.getParameter(debug === null ? gl.RENDERER : debug.UNMASKED_RENDERER_WEBGL));
  return SOFTWARE_RENDERER.test(name) ? name : null;
}

/** The renderer's name if WebGL2 here can fold but only on the CPU (see `webgl2SoftwareRenderer`), else null. */
export function webgl2FoldIsSoftware() {
  try {
    const gl = document.createElement("canvas").getContext("webgl2");
    return gl === null ? null : webgl2SoftwareRenderer(gl);
  } catch {
    return null;
  }
}

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
 *
 * 🔴 AND 128 MiB WAS TOO SMALL TO HOLD ONE PROJECTION'S PLANES PAST ~250 TOKENS, SO EVERY ONE WAS A FRESH TEXTURE AGAIN.
 * A blocked projection writes eight row planes, each `rows / 8 x outChannels` - ~80 MB each for the pair transition's
 * up projection at 400 tokens - and a pool that cannot hold them creates and zero-fills them on every call: the trunk's
 * 128-input projections fell from ~600 GFLOP/s at 59 tokens to ~100 at 400 in the fold while the same kernel benched at
 * 230-270. At 512 MiB (8 trunk blocks, 1 step): 400 tokens 40.1 -> 18.9 s at +41 MiB of peak (1856), 250 tokens
 * 5.73 -> 4.50 s at +343 (1347), 59 tokens level at +52 (988). 4 GiB took 400 tokens to 16.3 s at a 4738 MiB peak.
 */
export const AF3_POOL_LIMIT_BYTES = 512 * 2 ** 20;

/**
 * The pool's bound for a fold of `tokens`: 512 MiB, or thirty-two pair tensors where that is more.
 *
 * 🔴 WHAT THE POOL MUST HOLD IS WHAT A PAIRFORMER BLOCK FREES AND ASKS FOR AGAIN, AND THAT GROWS WITH THE FOLD. A
 * constant cannot be right at every length: at 262 tokens 512 MiB deleted and re-created ~1.5 GiB of textures a
 * block. That is the whole cost of a long fold where a fresh texture is expensive - 0.86 ms a MiB of first write
 * through NVIDIA's desktop OpenGL (a Linux reader's WebGL2) against 0.08 through ANGLE's Vulkan on the same A100,
 * the driver's allocation and not the browser's zeroing: **79 s for the fold where Vulkan was 10.6**.
 *
 * Two things, neither of them a measurement of the device. `Storage` serves a tensor from a TALLER pooled texture
 * when none is its own height (`anyFit`), so one shape's spare is every shorter shape's; and the bound follows the
 * fold. 262 tokens, 8 trunk blocks, NVIDIA's OpenGL (exact-match pool / any-fit, seconds at GiB of peak):
 *
 *     512 MiB   12.8 at 1.43   9.4 at 1.65
 *     768 MiB   10.2 at 1.70   3.9 at 1.84
 *     1 GiB      5.5 at 1.97   2.9 at 2.09      <- thirty-two pair tensors at 262 tokens
 *     4 GiB      3.1 at 2.49   2.8 at 2.13
 *
 * `fold-af3-webgl.js --pool-mib=512` and `--no-any-fit` are the arms without (docs/FIREFOX.md, "Linux").
 */
export function af3PoolLimitBytes(tokens, pairChannels, gl = null) {
  if (gl !== null && !af3PoolFollowsFold(gl)) return AF3_POOL_LIMIT_BYTES;
  return Math.max(AF3_POOL_LIMIT_BYTES, 32 * tokens * tokens * pairChannels * 4);
}

/**
 * Whether this context takes the pool above - any-fit reuse and a bound that follows the fold - or the pool as
 * it was (exact match, 512 MiB). False on an Apple renderer.
 *
 * 🔴 ON METAL A FRESH TEXTURE IS CHEAP, SO THE BIGGER POOL BUYS NOTHING AND COSTS MEMORY. The same 262-token fold
 * (48 + 4 blocks, 25 steps) on an M2 through ANGLE's Metal, seconds at GiB of peak:
 *
 *     exact match, 512 MiB          28.3 at 1.64      <- kept
 *     any-fit, 512 MiB              30.9 at 1.96      (a taller texture under a shorter tensor is live bytes)
 *     any-fit, bound follows fold   27.0 at 2.29
 *
 * and the answer is the same in all three. A reader folding in WebGL2 on Apple hardware has no WebGPU, which
 * includes phones, where 0.65 GiB is the risk. `fold-af3-webgl.js --any-fit --pool-mib=<n>` is the arm with.
 */
export function af3PoolFollowsFold(gl) {
  const debug = gl.getExtension("WEBGL_debug_renderer_info");
  const renderer = String(debug === null ? gl.getParameter(gl.RENDERER) : gl.getParameter(debug.UNMASKED_RENDERER_WEBGL));
  return !/Apple|Metal/i.test(renderer);
}

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
  if (execution === null) {
    const gl = createWebGL2Context();
    execution = new Execution(gl,
      { halfWeights: true, halfActivations: true, int5Weights: true, anyFit: af3PoolFollowsFold(gl),
        pooling: true, poolLimitBytes: AF3_POOL_LIMIT_BYTES, halfWeightFloor: AF3_HALF_WEIGHT_FLOOR });
  }
  if (resident?.weights !== weights) {
    const previous = resident;
    resident = {
      weights,
      ready: (async () => {
        if (previous !== null) releaseAf3Weights(execution, await previous.ready);
        // ...one module behind the GPU rather than at its drained end (`laggedPause`): the upload is host work, and
        // waiting for an EMPTY queue at every module boundary cost a 4 ms poll each, ~60 of them, on a first fold
        const behind = laggedPause(execution.gl);
        try {
          return await drainAsync(uploadAf3WeightsSteps(execution, weights), behind);
        } finally {
          behind.finish();
        }
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
             mask: widen(batch.msaMask, 0), sequences: rows, queryDrawn: chosen[0] === 0 };
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
  // ...the pool's bound follows the fold (`af3PoolLimitBytes`; not on an Apple renderer); a shorter fold after a long
  // one evicts down to its own
  execution.storage.poolLimitBytes = af3PoolLimitBytes(tokens, weights.trunk.embedder.pairChannels, execution.gl);
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
    // ...and AF3's interface term for the chain-level scores (d0 from the pair's own chains), anchored, as pTM is,
    // on the tokens with frames - webgpu/af3/fold.js's reduction (reported by Yeqing Lin). `foldAf3` has already
    // taken both from these logits with these chains (`tmTermsFromLogits`), so they are reused, not taken again.
    const { tmAdjusted, tmInterface } = fold.tmAdjusted !== undefined && fold.tmInterface !== undefined
      ? fold
      : (({ adjusted, interface: chained }) => ({ tmAdjusted: adjusted, tmInterface: chained }))(
        tmTermsFromLogits(fold.paeLogits, tokens,
          tmPerBinFor(errorBinCentres(NUM_BINS, MAX_ERROR_BIN), tmScoreD0(tokens)),
          chainTmInfo(asymId, seqMask, errorBinCentres(NUM_BINS, MAX_ERROR_BIN))));
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
