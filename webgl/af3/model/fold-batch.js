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
 * 🔴 WHAT IT DOES NOT CARRY: a draw from an alignment (all rows when WebGPU would take them all,
 * refused rather than folded differently when it would draw), a structural-token model (OpenDDE, not ported), and
 * the WebGPU path's extras (trunk reuse, steering, motifs, warm starts) - each refused by name.
 */
import { createWebGL2Context } from "../../runtime/context.js";
import { Execution } from "../../runtime/execution.js";
import { foldAf3Async, releaseAf3Weights, uploadAf3WeightsSteps } from "./fold.js";
import { drainAsync, gpuIdle } from "../../runtime/steps.js";
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

// One context for the page, and the last family's weights on it: a second family's upload
// replaces the first's (each is ~800 MB as halves), the same family's is kept.
let execution = null;
let resident = null; // { weights, ready: Promise<uploaded> }

/** The weights on the GPU, uploaded a module at a time with the thread let go in between. */
function uploadedFor(weights) {
  // The configuration fold-af3-webgl.js gates against the WGSL fold: half weights and activations.
  execution ??= new Execution(createWebGL2Context(),
    { halfWeights: true, halfActivations: true, int5Weights: true });
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
  // Every row, in order, as WebGPU's fold takes them when they fit under its cap (msaDrawRows'
  // prefix): a multi-chain refold's per-chain query rows. A draw from a deeper alignment is not ported.
  const rows = batch.msa.length / tokens;
  const cap = Math.min(1024, batch.numMsa ?? Infinity);
  const native = weights.trunk.dialect.msaNativeSample === true && batch.numMsa !== undefined;
  if (rows > cap || (native && rows > 1)) {
    throw new Error(`the WebGL2 fold does not draw from an alignment (${rows} rows)`);
  }
  const uploaded = await uploadedFor(weights);
  const fold = await foldAf3Async(execution, {
    batch, weights: uploaded, hostWeights: weights,
    msa: {
      codes: Float32Array.from(batch.msa),
      deletions: Float32Array.from(batch.deletionMatrix),
      mask: Float32Array.from(batch.msaMask),
      sequences: rows,
    },
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
