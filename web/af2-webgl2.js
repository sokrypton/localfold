/**
 * AlphaFold 2 on the page without WebGPU: the WebGL2 fold (webgl/af2/model/fold.js) behind the interface the page's
 * WebGPU drivers have - `predictA3m(a3m, weights, featureTables, options, paeBreaks, onRecycle, onProgress)` returning
 * `{recycles, final, resumable}` - so web/app.js's AlphaFold 2 path chooses a driver and changes nothing else.
 *
 * 🔴 THE FOLD ITSELF IS NOT NEW. `tools/gpu/fold-af2-webgl.js` holds it to the WGSL fold on both graphs (the monomer
 * 0.0565 A at the worst atom against a 0.0355 A floor, pLDDT 57.937 against 57.938; the multimer 0.0005 A), in the same
 * configuration as here - float16 weights, float16 x, the int5 codes decoded in the shader. What this file adds is the
 * translation between the two drivers' results, which is where a page path goes wrong without an error:
 *
 *   - a WebGPU pass carries `structure: {atom37, atom37Mask, ...}`, a WebGL2 one carries those fields at the top level;
 *   - the WebGPU confidence head returns pTM, ipTM and the chain-level scores, the WebGL2 one returns the per-pair
 *     `tmTerm` and leaves the reductions to its caller - so they are reduced here, through the SAME
 *     `tmScoresFromTerm` the WebGPU head calls, which is what keeps the two backends' numbers meaning the same thing.
 *     A complex's chain-level scores need AlphaFold 3's interface term, which needs the PAE logits
 *     (`keepPaeLogits`, read back only for a complex);
 *   - progress is reported in Evoformer blocks, as the WebGPU drivers report theirs.
 *
 * 🔴 NO CONTINUATION. The WebGPU drivers can hand back their recycle state (`resumable`) so raising the recycle count
 * continues rather than restarts; this returns none, so `planRecycleReuse` plans such a press as a fresh fold. A replay
 * of passes already in hand still works, because that reads the cached passes and not the state.
 */
import { foldAf2, releaseAf2Weights, uploadAf2Weights } from "../webgl/af2/model/fold.js";
import { paeCentres } from "../webgl/heads/confidence.js";
import { createWebGL2Context } from "../webgl/runtime/context.js";
import { Execution } from "../webgl/runtime/execution.js";
import { tmScoresFromTerm } from "../webgpu/heads/confidence.js";
import { makeA3mFeatures } from "../shared/input/a3m-features.js";

// One context for AlphaFold 2 on the page, and one model's weights on it: a sweep's next model (or a change of model)
// releases the last one's textures before uploading its own, as the AF3 path keeps one family resident.
let execution = null;
let resident = null; // { weights, uploaded }

function executionFor() {
  // The configuration tools/gpu/fold-af2-webgl.js gates against the WGSL fold (`webglExecutionOptions` with no flags).
  execution ??= new Execution(createWebGL2Context(),
    { halfWeights: true, halfActivations: true, int5Weights: true, anyFit: true });
  return execution;
}

function uploadedFor(weights, graph) {
  if (resident?.weights === weights) return resident.uploaded;
  if (resident !== null) releaseAf2Weights(execution, resident.uploaded);
  resident = null;
  // web/model.js's tree names the distogram head `distogram`; the WebGL2 upload names it `distogramHead`
  const uploaded = uploadAf2Weights(execution, { ...weights, distogramHead: weights.distogram ?? null }, graph);
  resident = { weights, uploaded };
  return uploaded;
}

/** Let the WebGL2 AlphaFold 2 weights go (the page switched to a model this file does not fold). */
export function releaseAf2WebGl2() {
  if (resident !== null && execution !== null) releaseAf2Weights(execution, resident.uploaded);
  resident = null;
}

/** A WebGL2 pass in the WebGPU drivers' shape. */
function asPass(raw, { length, chainLengths, paeBreaks, elapsedMilliseconds }) {
  const { confidence } = raw;
  const multiChain = Array.isArray(chainLengths) && chainLengths.length > 1;
  const scores = tmScoresFromTerm(confidence.tmTerm, length, multiChain ? chainLengths : undefined,
    multiChain ? { paeLogits: confidence.paeLogits, centers: paeCentres(paeBreaks) } : {});
  return {
    msaFirstRow: raw.msaFirstRow,
    pair: raw.pair,
    contactProbs: raw.contactProbs,
    structure: {
      atom37: raw.atom37, atom37Mask: raw.atom37Mask, atom14: raw.atom14,
      affine: raw.affine, angles: raw.angles, unnormalizedAngles: raw.unnormalizedAngles,
    },
    confidence: {
      plddt: confidence.plddt,
      meanPlddt: confidence.meanPlddt,
      predictedAlignedError: confidence.predictedAlignedError,
      maxPredictedAlignedError: confidence.maxPredictedAlignedError,
      ptm: scores.ptm,
      iptm: scores.iptm,
      multimerScore: scores.multimerScore,
      chainPairIptm: scores.chainPairIptm,
      chainPtm: scores.chainPtm,
      chainIptm: scores.chainIptm,
    },
    recycleDistance: raw.recycleDistance,
    elapsedMilliseconds,
  };
}

const abortError = () => new DOMException("The fold was stopped", "AbortError");

/** AlphaFold 2 (either graph) folded in WebGL2, called as `AlphaFoldMonomerGpu`/`AlphaFoldUnifiedGpu` are. */
export class AlphaFoldWebGl2 {
  /** @param {{graph: "monomer" | "multimer"}} options */
  constructor({ graph }) {
    if (graph !== "monomer" && graph !== "multimer") {
      throw new RangeError(`AlphaFoldWebGl2: the graph is monomer or multimer, not \`${graph}\``);
    }
    this.graph = graph;
  }

  async predictA3m(a3mText, weights, featureTables, options = {}, paeBreaks, onRecycle, onProgress) {
    if (options.resume !== undefined) {
      throw new Error("the WebGL2 AlphaFold 2 fold cannot continue a previous fold's recycles");
    }
    const { signal, chainLengths } = options;
    if (signal?.aborted) throw abortError();
    const start = performance.now();
    // The same host featurisation the WebGPU drivers run with `hostFeaturisation` (their nearest-centre search on the
    // device is the only difference, and it is held to this one at zero differing assignments).
    const featuresByRecycle = makeA3mFeatures(a3mText, featureTables, options);
    const length = featuresByRecycle[0].aatype.length;
    const run = executionFor();
    const uploaded = uploadedFor(weights, this.graph);

    // Progress in Evoformer blocks, every pass's extra and main stacks, as the WebGPU drivers count it.
    const blocksPerPass = weights.extraStack.length + weights.mainStack.length;
    const total = blocksPerPass * featuresByRecycle.length;
    let completed = 0;
    onProgress?.({ completed: 0, total, waiting: true });

    const multiChain = Array.isArray(chainLengths) && chainLengths.length > 1;
    const passes = [];
    let passStart = performance.now();
    const call = {
      featuresByRecycle, weights: uploaded, paeBreaks, graph: this.graph,
      template: options.template,
      tolerance: options.tolerance,
      contactMap: weights.distogram !== undefined && weights.distogram !== null,
      keepPaeLogits: multiChain,
    };
    // The programs this fold draws with, linked together and off the thread before the first draw (the tool's own
    // order: `prepare` packs the light ones into combined programs). A second fold of the same shape finds them linked.
    await run.prepare((await run.collectPrograms(() => foldAf2(run, call))).keys());
    if (signal?.aborted) throw abortError();

    await foldAf2(run, {
      ...call,
      onBlock: () => {
        completed += 1;
        onProgress?.({ completed, total, waiting: false });
      },
      onRecycle: (raw, recycle) => {
        const now = performance.now();
        const pass = asPass(raw, { length, chainLengths, paeBreaks, elapsedMilliseconds: now - passStart });
        passStart = now;
        passes.push(pass);
        onRecycle?.(pass, recycle);
      },
      // Every draw is issued synchronously, so the page gets its frames back between stages here - and a Stop.
      yieldTo: async () => {
        await new Promise((resolve) => requestAnimationFrame(() => resolve()));
        if (signal?.aborted) throw abortError();
      },
    });
    return {
      recycles: passes, final: passes[passes.length - 1], resumable: undefined,
      elapsedMilliseconds: performance.now() - start,
    };
  }
}
