/**
 * BioEmu2 on the page without WebGPU: the WebGL2 fold (webgl/bioemu2/model/fold.js) behind the call webgpu/bioemu2/fold.js's
 * `foldBioemu2` has - `(read, features, {samples, steps, seed, mode, passes, onSamples, onContacts, onProgress, signal})`
 * returning `{samples, contact, stopped, ...}` - so web/app.js's BioEmu2 path chooses a driver and changes nothing else.
 *
 * 🔴 THE FOLD ITSELF IS NOT NEW. `tools/gpu/fold-bioemu2-webgl.js` folds it end to end (6MRR at 68 tokens, 20 steps:
 * rmsd 1.595, TM 0.9083, CA-CA 3.853) and the stage gates hold each stage to the WGSL one, in the configuration here -
 * float16 weights, float16 x, the int5 codes decoded in the shader, x narrowed on aligned shapes. What this file adds:
 *
 *   - progress in trunk blocks and sampler steps, reported as `foldBioemu2` reports them (`onProgress("trunk", ...)`,
 *     then `("diffusion", done, total, denoised)` with the step's denoised estimate, which the sampler already has on
 *     the host - so the page's live frames cost it nothing here);
 *   - the thread let go at every pair block and every denoiser layer, one slice behind the GPU (`laggedPause`), so
 *     the page repaints and Stop is seen within a layer rather than at the end of a fold;
 *   - Stop keeping the batches that finished (`stopped: true`), as the WGSL fold does.
 *
 * 🔴 PROPAGATION IS REFUSED BY NAME. The WebGL2 sampler carries no companion frame (`setCompanion` is a no-op there and
 * the denoiser has no dynamics conditioning), so a propagation would sample an unconditioned ensemble and call it a
 * trajectory.
 *
 * 🔴 NO WEIGHTS ARE KEPT BETWEEN FOLDS, DELIBERATELY. The fold uploads each stage's weights and releases them before
 * the next stage's (the trunk's before the denoiser's - see the fold's header), and the uploads are ~0.5 s of a fold
 * whose every sampler step is ~0.8 s at 68 tokens on an M2; keeping both stacks resident would hold the memory the
 * stage order exists to avoid. And the texture pool is emptied after every fold, so a context that is not folding
 * holds nothing but its linked programs - which is what makes a change of model safe here without a release call.
 */
import { foldBioemu2OnWebGl2 } from "../webgl/bioemu2/model/fold.js";
import { createWebGL2Context } from "../webgl/runtime/context.js";
import { laggedPause } from "../webgl/runtime/steps.js";
import { AF3_POOL_LIMIT_BYTES } from "../webgl/af3/model/fold-batch.js";
import { Execution } from "../webgl/runtime/execution.js";
import { bioemu2Batches } from "../cpu/bioemu2/sampler.js";

let execution = null;
/** The programs prepared so far, by the feature shape that reached them (a kernel's source carries its sizes). */
const prepared = new Set();

function executionFor() {
  // tools/gpu/fold-bioemu2-webgl.js's configuration, with the AF3 path's bounded texture pool: without it every draw
  // creates and deletes its output texture through the browser's GPU process. Measured with that tool (6MRR, 68
  // tokens, 20 steps, Chrome on an M2): 25.1 s -> 20.2 s with `--pool --pool-mib=512`, the peak 411 -> 739 MiB, and the
  // same fold to the last digit (CA-CA 3.8529611222441167 both ways).
  // 🔴 AND `shortBlock`, WHICH IS WHAT MADE A CHAIN UNDER 64 RESIDUES TWICE AS SLOW A STEP IN SAFARI AS ONE OVER IT.
  // Below 64 rows `blockedApplies` sends an [out][in] projection to the one-texel kernel, and every token-stream
  // projection of the denoiser is [out][in] at the chain's length: in Safari 16.3 on an M2 that kernel is 2-2.5x the
  // blocked one at these widths (58 x 768 -> 6144: 15.4 ms a call against ~6.6 at 68 rows blocked), so 58 residues
  // stepped at 1144 ms where 68 stepped at 586. Blocked: 525 ms (2.18x), bit-identical in f32 (the same sums in the
  // same order - check-webgl-bioemu2-denoiser.js at 10 and 13 tokens), and Chrome unmoved (357 / 358 ms).
  execution ??= new Execution(createWebGL2Context(),
    { halfWeights: true, halfActivations: true, int5Weights: true, narrowAlignedX: true, narrowUnblockedX: true,
      anyFit: true, pooling: true, poolLimitBytes: AF3_POOL_LIMIT_BYTES, shortBlock: true });
  return execution;
}

const abortError = () => new DOMException("The fold was stopped", "AbortError");

/**
 * One BioEmu2 fold in WebGL2, called as the page calls `foldBioemu2` (the device and the network are not arguments:
 * there is no device, and the weights are read stage by stage from `read`).
 */
export async function foldBioemu2WebGl2(read, features, options = {}) {
  if (options.propagation !== undefined && options.propagation !== null) {
    throw new Error("BioEmu2 propagation needs WebGPU, which this browser does not have - the WebGL2 sampler carries"
      + " no starting frame. Structure and Equilibrium sample here in WebGL2.");
  }
  const { signal } = options;
  if (signal?.aborted) throw abortError();
  const run = executionFor();
  const progress = options.onProgress ?? (() => {});
  const steps = options.steps;
  const samples = options.samples ?? 5;
  const call = { steps, samples, seed: options.seed, passes: options.passes, mode: options.mode };

  // The programs this fold draws with, linked off the thread before the first draw - a dry run of two steps and one
  // sample reaches every one (tools/gpu/fold-bioemu2-webgl.js's order); a second fold of the same shape finds them.
  // (and by the batch: the denoiser's sources carry a batch's rows, so the dry run walks one batch of that size)
  const { size } = bioemu2Batches(samples);
  const shape = `${features.res_type.length}:${features.atom_pad_mask.length}:${features.msa.length}:${size}`;
  if (!prepared.has(shape)) {
    await run.prepare((await run.collectPrograms(
      () => foldBioemu2OnWebGl2(run, read, features, { ...call, steps: 2, samples: size }))).keys());
    prepared.add(shape);
  }
  if (signal?.aborted) throw abortError();

  const behind = laggedPause(run.gl);
  try {
    return await foldBioemu2OnWebGl2(run, read, features, {
      ...call,
      signal,
      pause: () => behind(),
      onTrunk: (block, blocks) => progress("trunk", block, blocks),
      onContacts: options.onContacts,
      // (steps over every batch, and the batch's denoised estimate - webgpu/bioemu2/fold.js's progress exactly)
      onStep: (done, total, denoised) => progress("diffusion", done, total, denoised),
      onSamples: options.onSamples,
    });
  } finally {
    behind.finish();
    // ...and the pool goes back to the browser: a context between folds holds its programs and nothing else
    run.storage.drainPool();
  }
}
